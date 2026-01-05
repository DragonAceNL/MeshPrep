# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Subprocess wrapper for crash-prone actions like PyMeshLab.

Some native libraries (PyMeshLab, Open3D) can crash with access violations
that cannot be caught by Python exception handling. This wrapper runs such
actions in isolated subprocesses so crashes don't kill the main process.

The system learns which action+mesh combinations cause crashes and can
skip them in the future.
"""

import json
import logging
import multiprocessing as mp
import os
import sqlite3
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

# Database for tracking crashes
CRASH_DB_PATH = Path(__file__).parent.parent / "learning_data" / "action_crashes.db"

# Actions known to potentially crash (run in subprocess)
CRASH_PRONE_ACTIONS = {
    "meshlab_reconstruct_poisson",
    "meshlab_reconstruct_ball_pivoting", 
    "meshlab_alpha_wrap",
    "meshlab_boolean_union",
    "meshlab_repair",
    "poisson_reconstruction",  # Open3D version
    "ball_pivoting",
}

# PyMeshLab version for tracking
_PYMESHLAB_VERSION: Optional[str] = None


def get_pymeshlab_version() -> str:
    """Get the installed PyMeshLab version."""
    global _PYMESHLAB_VERSION
    if _PYMESHLAB_VERSION is None:
        try:
            import pymeshlab
            _PYMESHLAB_VERSION = pymeshlab.__version__
        except:
            _PYMESHLAB_VERSION = "unknown"
    return _PYMESHLAB_VERSION


CRASH_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_info (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- Track actions that crash
CREATE TABLE IF NOT EXISTS action_crashes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action_name TEXT NOT NULL,
    
    -- Mesh characteristics when crash occurred
    face_count INTEGER,
    vertex_count INTEGER,
    body_count INTEGER,
    size_bin TEXT,
    
    -- Library versions
    pymeshlab_version TEXT,
    python_version TEXT,
    
    -- Crash details
    crash_type TEXT,  -- 'access_violation', 'segfault', 'timeout', 'unknown'
    error_message TEXT,
    
    -- Model info
    model_id TEXT,
    model_fingerprint TEXT,
    
    -- Timestamp
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(action_name, model_id)
);

-- Aggregated crash patterns
CREATE TABLE IF NOT EXISTS crash_patterns (
    action_name TEXT NOT NULL,
    size_bin TEXT NOT NULL,
    pymeshlab_version TEXT NOT NULL,
    
    crash_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    
    -- Recommendation
    should_skip INTEGER DEFAULT 0,
    
    last_crash TEXT,
    last_success TEXT,
    
    PRIMARY KEY(action_name, size_bin, pymeshlab_version)
);

CREATE INDEX IF NOT EXISTS idx_crashes_action ON action_crashes(action_name);
CREATE INDEX IF NOT EXISTS idx_crashes_model ON action_crashes(model_id);
CREATE INDEX IF NOT EXISTS idx_patterns_action ON crash_patterns(action_name);

INSERT OR REPLACE INTO schema_info (key, value) VALUES ('version', '1');
"""


@dataclass
class MeshInfo:
    """Mesh information for crash tracking."""
    face_count: int
    vertex_count: int
    body_count: int = 1
    model_id: str = ""
    model_fingerprint: str = ""
    
    @property
    def size_bin(self) -> str:
        """Get size category."""
        if self.face_count < 10_000:
            return "tiny"
        elif self.face_count < 50_000:
            return "small"
        elif self.face_count < 100_000:
            return "medium"
        elif self.face_count < 500_000:
            return "large"
        return "huge"


class CrashTracker:
    """Tracks and learns from action crashes."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or CRASH_DB_PATH
        self._ensure_db()
    
    def _ensure_db(self):
        """Ensure database exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(CRASH_DB_SCHEMA)
    
    @contextmanager
    def _get_connection(self):
        """Get database connection."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def record_crash(
        self,
        action_name: str,
        mesh_info: MeshInfo,
        crash_type: str = "unknown",
        error_message: str = "",
    ) -> None:
        """Record an action crash."""
        pymeshlab_version = get_pymeshlab_version()
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        
        with self._get_connection() as conn:
            # Record individual crash
            conn.execute("""
                INSERT OR REPLACE INTO action_crashes
                (action_name, face_count, vertex_count, body_count, size_bin,
                 pymeshlab_version, python_version, crash_type, error_message,
                 model_id, model_fingerprint, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                action_name, mesh_info.face_count, mesh_info.vertex_count,
                mesh_info.body_count, mesh_info.size_bin,
                pymeshlab_version, python_version, crash_type, error_message,
                mesh_info.model_id, mesh_info.model_fingerprint,
                datetime.now().isoformat()
            ))
            
            # Update crash pattern
            conn.execute("""
                INSERT INTO crash_patterns 
                (action_name, size_bin, pymeshlab_version, crash_count, last_crash)
                VALUES (?, ?, ?, 1, ?)
                ON CONFLICT(action_name, size_bin, pymeshlab_version) DO UPDATE SET
                    crash_count = crash_count + 1,
                    last_crash = excluded.last_crash
            """, (action_name, mesh_info.size_bin, pymeshlab_version, datetime.now().isoformat()))
            
            # Check if we should mark for skipping (>= 2 crashes)
            row = conn.execute("""
                SELECT crash_count, success_count FROM crash_patterns
                WHERE action_name = ? AND size_bin = ? AND pymeshlab_version = ?
            """, (action_name, mesh_info.size_bin, pymeshlab_version)).fetchone()
            
            if row and row["crash_count"] >= 2:
                success_rate = row["success_count"] / (row["crash_count"] + row["success_count"])
                if success_rate < 0.5:  # More crashes than successes
                    conn.execute("""
                        UPDATE crash_patterns SET should_skip = 1
                        WHERE action_name = ? AND size_bin = ? AND pymeshlab_version = ?
                    """, (action_name, mesh_info.size_bin, pymeshlab_version))
                    logger.warning(
                        f"Marked {action_name} as should_skip for {mesh_info.size_bin} meshes "
                        f"(PyMeshLab {pymeshlab_version})"
                    )
    
    def record_success(self, action_name: str, mesh_info: MeshInfo) -> None:
        """Record a successful action execution."""
        pymeshlab_version = get_pymeshlab_version()
        
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO crash_patterns 
                (action_name, size_bin, pymeshlab_version, success_count, last_success)
                VALUES (?, ?, ?, 1, ?)
                ON CONFLICT(action_name, size_bin, pymeshlab_version) DO UPDATE SET
                    success_count = success_count + 1,
                    last_success = excluded.last_success
            """, (action_name, mesh_info.size_bin, pymeshlab_version, datetime.now().isoformat()))
    
    def should_skip(self, action_name: str, mesh_info: MeshInfo) -> Tuple[bool, str]:
        """Check if action should be skipped due to crash history.
        
        Returns:
            Tuple of (should_skip, reason)
        """
        pymeshlab_version = get_pymeshlab_version()
        
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT should_skip, crash_count, success_count FROM crash_patterns
                WHERE action_name = ? AND size_bin = ? AND pymeshlab_version = ?
            """, (action_name, mesh_info.size_bin, pymeshlab_version)).fetchone()
            
            if row and row["should_skip"]:
                return True, f"Crashes on {mesh_info.size_bin} meshes (PyMeshLab {pymeshlab_version})"
        
        return False, ""
    
    def get_crash_stats(self) -> Dict[str, Any]:
        """Get crash statistics."""
        with self._get_connection() as conn:
            total_crashes = conn.execute("SELECT COUNT(*) FROM action_crashes").fetchone()[0]
            
            patterns = conn.execute("""
                SELECT action_name, size_bin, pymeshlab_version, 
                       crash_count, success_count, should_skip
                FROM crash_patterns
                ORDER BY crash_count DESC
            """).fetchall()
            
            return {
                "total_crashes": total_crashes,
                "patterns": [
                    {
                        "action": row["action_name"],
                        "size_bin": row["size_bin"],
                        "version": row["pymeshlab_version"],
                        "crashes": row["crash_count"],
                        "successes": row["success_count"],
                        "skip": bool(row["should_skip"]),
                    }
                    for row in patterns
                ],
            }


# Global tracker instance
_crash_tracker: Optional[CrashTracker] = None


def get_crash_tracker() -> CrashTracker:
    """Get or create the global crash tracker."""
    global _crash_tracker
    if _crash_tracker is None:
        _crash_tracker = CrashTracker()
    return _crash_tracker


def _run_action_in_subprocess(
    action_name: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    params: dict,
    result_queue: mp.Queue,
) -> None:
    """Run an action in a subprocess.
    
    This function runs in a child process. It imports the action,
    executes it, and puts the result (or error) in the queue.
    """
    try:
        import trimesh
        
        # Reconstruct mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Import the action registry
        # We need to import here because we're in a subprocess
        parent_dir = Path(__file__).parent.parent / "v2"
        sys.path.insert(0, str(parent_dir))
        
        from meshprep_poc.actions import ActionRegistry
        
        # Execute action
        start = time.perf_counter()
        result_mesh = ActionRegistry.execute(action_name, mesh, params)
        duration_ms = (time.perf_counter() - start) * 1000
        
        # Return result
        result_queue.put({
            "success": True,
            "vertices": result_mesh.vertices.copy(),
            "faces": result_mesh.faces.copy(),
            "duration_ms": duration_ms,
        })
        
    except Exception as e:
        result_queue.put({
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}",
            "traceback": traceback.format_exc(),
        })


def execute_action_safe(
    action_name: str,
    mesh,  # trimesh.Trimesh
    params: dict,
    timeout_s: float = 120,
    mesh_info: Optional[MeshInfo] = None,
) -> Tuple[bool, Optional[Any], str]:
    """Execute an action safely in a subprocess.
    
    If the action crashes, the subprocess dies but the main process continues.
    Crashes are recorded for learning.
    
    Args:
        action_name: Name of the action to execute
        mesh: Input trimesh mesh
        params: Action parameters
        timeout_s: Timeout in seconds
        mesh_info: Optional mesh info for crash tracking
        
    Returns:
        Tuple of (success, result_mesh or None, error_message)
    """
    import trimesh
    
    # Create mesh info if not provided
    if mesh_info is None:
        try:
            body_count = len(mesh.split(only_watertight=False))
        except:
            body_count = 1
        mesh_info = MeshInfo(
            face_count=len(mesh.faces),
            vertex_count=len(mesh.vertices),
            body_count=body_count,
        )
    
    # Check if we should skip based on crash history
    tracker = get_crash_tracker()
    should_skip, reason = tracker.should_skip(action_name, mesh_info)
    if should_skip:
        logger.warning(f"Skipping {action_name}: {reason}")
        return False, None, f"Skipped due to crash history: {reason}"
    
    # Create queue for results
    result_queue = mp.Queue()
    
    # Start subprocess
    process = mp.Process(
        target=_run_action_in_subprocess,
        args=(action_name, mesh.vertices.copy(), mesh.faces.copy(), params, result_queue),
    )
    process.start()
    
    # Wait for result
    try:
        result = result_queue.get(timeout=timeout_s)
        process.join(timeout=5)
        
        if result["success"]:
            # Record success
            tracker.record_success(action_name, mesh_info)
            
            # Reconstruct mesh
            result_mesh = trimesh.Trimesh(
                vertices=result["vertices"],
                faces=result["faces"],
            )
            return True, result_mesh, ""
        else:
            # Action failed but didn't crash
            return False, None, result.get("error", "Unknown error")
            
    except Exception as e:
        # Timeout or queue error - process likely crashed
        logger.error(f"Action {action_name} crashed or timed out: {e}")
        
        # Kill the process
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        
        # Determine crash type
        exit_code = process.exitcode
        if exit_code is None:
            crash_type = "timeout"
            error_msg = f"Timed out after {timeout_s}s"
        elif exit_code < 0:
            # Negative exit code = killed by signal (Unix) or access violation (Windows)
            crash_type = "access_violation"
            error_msg = f"Crashed with exit code {exit_code}"
        else:
            crash_type = "unknown"
            error_msg = f"Failed with exit code {exit_code}"
        
        # Record crash
        tracker.record_crash(
            action_name, mesh_info,
            crash_type=crash_type,
            error_message=error_msg,
        )
        
        return False, None, error_msg


def is_crash_prone_action(action_name: str) -> bool:
    """Check if an action should be run in a subprocess."""
    # Check if action name contains any crash-prone keywords
    action_lower = action_name.lower()
    for prone in CRASH_PRONE_ACTIONS:
        if prone.lower() in action_lower or action_lower in prone.lower():
            return True
    return False
