# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Subprocess wrapper and failure tracking for crash-prone actions.

Some native libraries (PyMeshLab, Open3D) can crash with access violations
that cannot be caught by Python exception handling. This wrapper runs such
actions in isolated subprocesses so crashes don't kill the main process.

Additionally, the system tracks action failures (caught exceptions) and
learns which actions fail on which mesh characteristics. This helps avoid
wasting time on actions that consistently fail for certain types of meshes.

Failure categories tracked:
- crash: Process crash (access violation, segfault)
- timeout: Action exceeded time limit
- error: Python exception (e.g., PyMeshLab filter requirements not met)
- geometry_loss: Output failed geometry validation
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

# Import error logger
try:
    from .error_logger import log_action_error, log_crash, log_pymeshlab_error
    ERROR_LOGGER_AVAILABLE = True
except ImportError:
    ERROR_LOGGER_AVAILABLE = False
    log_action_error = None
    log_crash = None
    log_pymeshlab_error = None

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
_MESHPREP_VERSION: Optional[str] = None


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


def get_meshprep_version() -> str:
    """Get the installed MeshPrep version."""
    global _MESHPREP_VERSION
    if _MESHPREP_VERSION is None:
        try:
            from . import __version__
            _MESHPREP_VERSION = __version__
        except:
            _MESHPREP_VERSION = "unknown"
    return _MESHPREP_VERSION


CRASH_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_info (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- Track software versions to detect upgrades
CREATE TABLE IF NOT EXISTS version_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pymeshlab_version TEXT,
    meshprep_version TEXT,
    python_version TEXT,
    first_seen TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(pymeshlab_version, meshprep_version, python_version)
);

-- Track action failures (crashes, errors, timeouts)
CREATE TABLE IF NOT EXISTS action_failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action_name TEXT NOT NULL,
    
    -- Mesh characteristics when failure occurred
    face_count INTEGER,
    vertex_count INTEGER,
    body_count INTEGER,
    size_bin TEXT,
    
    -- Library versions
    pymeshlab_version TEXT,
    meshprep_version TEXT,
    python_version TEXT,
    
    -- Failure details
    failure_type TEXT,  -- 'crash', 'error', 'timeout', 'geometry_loss'
    error_message TEXT,
    error_category TEXT,  -- Extracted category like 'normals_required', 'memory', etc.
    
    -- Model info
    model_id TEXT,
    model_fingerprint TEXT,
    
    -- Timestamp
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Legacy table name for backward compatibility
CREATE TABLE IF NOT EXISTS action_crashes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action_name TEXT NOT NULL,
    face_count INTEGER,
    vertex_count INTEGER,
    body_count INTEGER,
    size_bin TEXT,
    pymeshlab_version TEXT,
    python_version TEXT,
    crash_type TEXT,
    error_message TEXT,
    model_id TEXT,
    model_fingerprint TEXT,
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
        meshprep_version = get_meshprep_version()
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        
        # Log to dedicated error log file
        if ERROR_LOGGER_AVAILABLE and log_crash:
            log_crash(
                action_name=action_name,
                error_message=error_message,
                model_id=mesh_info.model_id,
                face_count=mesh_info.face_count,
                exit_code=None,  # Not available here
            )
        
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
    
    def check_version_change(self) -> bool:
        """Check if software versions have changed since last run.
        
        If versions changed, skip recommendations from older versions
        are reset to give the new version a fresh chance.
        
        Returns:
            True if versions changed (and skips were reset)
        """
        pymeshlab_version = get_pymeshlab_version()
        meshprep_version = get_meshprep_version()
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        
        with self._get_connection() as conn:
            # Check if this version combination exists
            existing = conn.execute("""
                SELECT id FROM version_history 
                WHERE pymeshlab_version = ? AND meshprep_version = ? AND python_version = ?
            """, (pymeshlab_version, meshprep_version, python_version)).fetchone()
            
            if existing:
                # Same version, no change
                return False
            
            # New version combination - record it
            conn.execute("""
                INSERT OR IGNORE INTO version_history (pymeshlab_version, meshprep_version, python_version)
                VALUES (?, ?, ?)
            """, (pymeshlab_version, meshprep_version, python_version))
            
            # Check if there are any skip recommendations from older versions
            old_skips = conn.execute("""
                SELECT COUNT(*) FROM failure_patterns
                WHERE should_skip = 1 
                AND (pymeshlab_version != ? OR meshprep_version != ?)
            """, (pymeshlab_version, meshprep_version)).fetchone()[0]
            
            if old_skips > 0:
                logger.info(
                    f"Version change detected (MeshPrep {meshprep_version}, PyMeshLab {pymeshlab_version}). "
                    f"Resetting {old_skips} skip recommendations from older versions."
                )
                # Reset skip recommendations for old versions
                # (They remain in DB for reference but won't apply to new version)
                # We don't delete - the new version creates new entries
                return True
            
            return False
    
    def reset_skips_for_current_version(self) -> int:
        """Reset all skip recommendations for the current version.
        
        This allows retrying actions that previously failed, useful when:
        - A bug has been fixed in the current version
        - User wants to give actions another chance
        
        Returns:
            Number of skip recommendations reset
        """
        pymeshlab_version = get_pymeshlab_version()
        meshprep_version = get_meshprep_version()
        
        with self._get_connection() as conn:
            result = conn.execute("""
                UPDATE failure_patterns
                SET should_skip = 0, skip_reason = NULL
                WHERE pymeshlab_version = ? AND meshprep_version = ? AND should_skip = 1
            """, (pymeshlab_version, meshprep_version))
            
            count = result.rowcount
            if count > 0:
                logger.info(f"Reset {count} skip recommendations for MeshPrep {meshprep_version}")
            return count
    
    def get_version_info(self) -> Dict[str, Any]:
        """Get version tracking information."""
        pymeshlab_version = get_pymeshlab_version()
        meshprep_version = get_meshprep_version()
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        
        with self._get_connection() as conn:
            # Get all known versions
            versions = conn.execute("""
                SELECT pymeshlab_version, meshprep_version, python_version, first_seen
                FROM version_history
                ORDER BY first_seen DESC
            """).fetchall()
            
            # Count skips by version
            skips_by_version = conn.execute("""
                SELECT pymeshlab_version, meshprep_version, COUNT(*) as skip_count
                FROM failure_patterns
                WHERE should_skip = 1
                GROUP BY pymeshlab_version, meshprep_version
            """).fetchall()
            
            return {
                "current_version": {
                    "pymeshlab": pymeshlab_version,
                    "meshprep": meshprep_version,
                    "python": python_version,
                },
                "version_history": [
                    {
                        "pymeshlab": row["pymeshlab_version"],
                        "meshprep": row["meshprep_version"],
                        "python": row["python_version"],
                        "first_seen": row["first_seen"],
                    }
                    for row in versions
                ],
                "skips_by_version": [
                    {
                        "pymeshlab": row["pymeshlab_version"],
                        "meshprep": row["meshprep_version"],
                        "skip_count": row["skip_count"],
                    }
                    for row in skips_by_version
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


# Error message patterns for categorization
ERROR_CATEGORIES = {
    "normals_required": [
        "requires correct per vertex normals",
        "proper, not-null normal",
        "normal",
    ],
    "memory": [
        "memory",
        "out of memory",
        "allocation failed",
        "cannot allocate",
    ],
    "empty_mesh": [
        "empty mesh",
        "no faces",
        "no vertices",
        "zero faces",
    ],
    "invalid_input": [
        "invalid input",
        "invalid mesh",
        "corrupt",
        "malformed",
    ],
    "filter_failed": [
        "failed to apply filter",
        "filter failed",
        "filter error",
    ],
    "topology": [
        "non-manifold",
        "self-intersect",
        "degenerate",
        "topology",
    ],
}


def categorize_error(error_message: str) -> str:
    """Categorize an error message for learning.
    
    Args:
        error_message: The error message to categorize
        
    Returns:
        Category string like 'normals_required', 'memory', etc.
    """
    if not error_message:
        return "unknown"
    
    error_lower = error_message.lower()
    
    for category, patterns in ERROR_CATEGORIES.items():
        for pattern in patterns:
            if pattern.lower() in error_lower:
                return category
    
    return "unknown"


class ActionFailureTracker:
    """Tracks action failures (errors, not just crashes) for learning.
    
    This extends beyond crashes to track all action failures, including:
    - Python exceptions (e.g., PyMeshLab filter requirements not met)
    - Geometry validation failures
    - Timeout errors
    
    The tracker learns which actions consistently fail on certain mesh
    characteristics and can recommend skipping them.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or CRASH_DB_PATH
        self._ensure_db()
    
    def _ensure_db(self):
        """Ensure database exists with failure tracking tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(CRASH_DB_SCHEMA)
            
            # Schema migrations for existing databases
            self._migrate_db(conn)
            
            # Add failure patterns table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS failure_patterns (
                    action_name TEXT NOT NULL,
                    error_category TEXT NOT NULL,
                    size_bin TEXT NOT NULL,
                    pymeshlab_version TEXT NOT NULL,
                    meshprep_version TEXT NOT NULL DEFAULT 'unknown',
                    
                    failure_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    
                    should_skip INTEGER DEFAULT 0,
                    skip_reason TEXT,
                    
                    last_failure TEXT,
                    last_success TEXT,
                    
                    -- Track when skip was set, to reset on version change
                    skip_set_version TEXT,
                    skip_set_timestamp TEXT,
                    
                    PRIMARY KEY(action_name, error_category, size_bin, pymeshlab_version, meshprep_version)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_failure_patterns_action 
                ON failure_patterns(action_name)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_failure_patterns_version
                ON failure_patterns(pymeshlab_version, meshprep_version)
            """)
    
    def _migrate_db(self, conn) -> None:
        """Apply schema migrations for existing databases."""
        # Check if meshprep_version column exists in failure_patterns
        cursor = conn.execute("PRAGMA table_info(failure_patterns)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if "meshprep_version" not in columns and "action_name" in columns:
            # Old schema - need to add meshprep_version column
            logger.info("Migrating failure_patterns table: adding meshprep_version column")
            try:
                conn.execute("ALTER TABLE failure_patterns ADD COLUMN meshprep_version TEXT DEFAULT 'unknown'")
                conn.execute("ALTER TABLE failure_patterns ADD COLUMN skip_set_version TEXT")
                conn.execute("ALTER TABLE failure_patterns ADD COLUMN skip_set_timestamp TEXT")
                logger.info("Migration complete")
            except sqlite3.OperationalError:
                # Columns may already exist
                pass
        
        # Check if meshprep_version column exists in action_failures
        cursor = conn.execute("PRAGMA table_info(action_failures)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if "meshprep_version" not in columns and "action_name" in columns:
            logger.info("Migrating action_failures table: adding meshprep_version column")
            try:
                conn.execute("ALTER TABLE action_failures ADD COLUMN meshprep_version TEXT DEFAULT 'unknown'")
                logger.info("Migration complete")
            except sqlite3.OperationalError:
                pass
    
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
    
    def record_failure(
        self,
        action_name: str,
        mesh_info: MeshInfo,
        failure_type: str,
        error_message: str,
        pipeline_name: str = "",
        attempt_number: int = 0,
    ) -> None:
        """Record an action failure.
        
        Args:
            action_name: Name of the action that failed
            mesh_info: Mesh characteristics when failure occurred
            failure_type: Type of failure ('error', 'crash', 'timeout', 'geometry_loss')
            error_message: The error message
            pipeline_name: Name of the pipeline being executed
            attempt_number: Which attempt this was
        """
        pymeshlab_version = get_pymeshlab_version()
        meshprep_version = get_meshprep_version()
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        error_category = categorize_error(error_message)
        
        logger.debug(
            f"Recording failure: {action_name} on {mesh_info.size_bin} mesh - "
            f"{failure_type}: {error_category}"
        )
        
        # Log to dedicated error log file
        if ERROR_LOGGER_AVAILABLE and log_action_error:
            log_action_error(
                action_name=action_name,
                error_message=error_message,
                error_category=error_category,
                failure_type=failure_type,
                model_id=mesh_info.model_id,
                model_fingerprint=mesh_info.model_fingerprint,
                face_count=mesh_info.face_count,
                vertex_count=mesh_info.vertex_count,
                body_count=mesh_info.body_count,
                size_bin=mesh_info.size_bin,
                pipeline_name=pipeline_name,
                attempt_number=attempt_number,
            )
        
        with self._get_connection() as conn:
            # Record individual failure
            conn.execute("""
                INSERT INTO action_failures
                (action_name, face_count, vertex_count, body_count, size_bin,
                 pymeshlab_version, meshprep_version, python_version, failure_type, error_message,
                 error_category, model_id, model_fingerprint, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                action_name, mesh_info.face_count, mesh_info.vertex_count,
                mesh_info.body_count, mesh_info.size_bin,
                pymeshlab_version, meshprep_version, python_version, failure_type, error_message,
                error_category, mesh_info.model_id, mesh_info.model_fingerprint,
                datetime.now().isoformat()
            ))
            
            # Update failure pattern (version-specific)
            conn.execute("""
                INSERT INTO failure_patterns 
                (action_name, error_category, size_bin, pymeshlab_version, meshprep_version,
                 failure_count, last_failure)
                VALUES (?, ?, ?, ?, ?, 1, ?)
                ON CONFLICT(action_name, error_category, size_bin, pymeshlab_version, meshprep_version) 
                DO UPDATE SET
                    failure_count = failure_count + 1,
                    last_failure = excluded.last_failure
            """, (
                action_name, error_category, mesh_info.size_bin, 
                pymeshlab_version, meshprep_version, datetime.now().isoformat()
            ))
            
            # Check if we should mark for skipping (>= 3 failures of same category)
            row = conn.execute("""
                SELECT failure_count, success_count FROM failure_patterns
                WHERE action_name = ? AND error_category = ? 
                      AND size_bin = ? AND pymeshlab_version = ? AND meshprep_version = ?
            """, (action_name, error_category, mesh_info.size_bin, pymeshlab_version, meshprep_version)).fetchone()
            
            if row and row["failure_count"] >= 3:
                total = row["failure_count"] + row["success_count"]
                if total > 0:
                    failure_rate = row["failure_count"] / total
                    if failure_rate > 0.7:  # 70%+ failure rate
                        skip_reason = f"{error_category} errors on {mesh_info.size_bin} meshes ({failure_rate:.0%} fail rate)"
                        conn.execute("""
                            UPDATE failure_patterns 
                            SET should_skip = 1, skip_reason = ?,
                                skip_set_version = ?, skip_set_timestamp = ?
                            WHERE action_name = ? AND error_category = ? 
                                  AND size_bin = ? AND pymeshlab_version = ? AND meshprep_version = ?
                        """, (
                            skip_reason, meshprep_version, datetime.now().isoformat(),
                            action_name, error_category, 
                            mesh_info.size_bin, pymeshlab_version, meshprep_version
                        ))
                        logger.warning(
                            f"Marked {action_name} as should_skip for {error_category} "
                            f"on {mesh_info.size_bin} meshes (MeshPrep {meshprep_version})"
                        )
    
    def record_success(self, action_name: str, mesh_info: MeshInfo) -> None:
        """Record a successful action execution."""
        pymeshlab_version = get_pymeshlab_version()
        meshprep_version = get_meshprep_version()
        
        with self._get_connection() as conn:
            # Update all failure patterns for this action+size+version to record success
            conn.execute("""
                UPDATE failure_patterns 
                SET success_count = success_count + 1,
                    last_success = ?
                WHERE action_name = ? AND size_bin = ? 
                      AND pymeshlab_version = ? AND meshprep_version = ?
            """, (
                datetime.now().isoformat(), action_name, 
                mesh_info.size_bin, pymeshlab_version, meshprep_version
            ))
    
    def should_skip_for_error_category(
        self, 
        action_name: str, 
        mesh_info: MeshInfo,
        expected_error_category: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Check if action should be skipped based on failure history.
        
        Skip recommendations are VERSION-SPECIFIC. When MeshPrep or PyMeshLab
        is updated, old skip recommendations don't apply - the new version
        gets a fresh chance since bugs may have been fixed.
        
        Args:
            action_name: Action to check
            mesh_info: Mesh characteristics
            expected_error_category: If known, check for specific error category
            
        Returns:
            Tuple of (should_skip, reason)
        """
        pymeshlab_version = get_pymeshlab_version()
        meshprep_version = get_meshprep_version()
        
        with self._get_connection() as conn:
            if expected_error_category:
                # Check specific category for CURRENT version only
                row = conn.execute("""
                    SELECT should_skip, skip_reason FROM failure_patterns
                    WHERE action_name = ? AND error_category = ?
                          AND size_bin = ? 
                          AND pymeshlab_version = ? 
                          AND meshprep_version = ?
                          AND should_skip = 1
                """, (
                    action_name, expected_error_category, 
                    mesh_info.size_bin, pymeshlab_version, meshprep_version
                )).fetchone()
            else:
                # Check any category for CURRENT version only
                row = conn.execute("""
                    SELECT should_skip, skip_reason FROM failure_patterns
                    WHERE action_name = ? AND size_bin = ? 
                          AND pymeshlab_version = ? 
                          AND meshprep_version = ?
                          AND should_skip = 1
                    LIMIT 1
                """, (action_name, mesh_info.size_bin, pymeshlab_version, meshprep_version)).fetchone()
            
            if row and row["should_skip"]:
                reason = row["skip_reason"] or "High failure rate"
                return True, f"{reason} (MeshPrep {meshprep_version})"
        
        return False, ""
    
    def get_failure_stats(self) -> Dict[str, Any]:
        """Get failure statistics."""
        with self._get_connection() as conn:
            # Total failures
            total = conn.execute(
                "SELECT COUNT(*) FROM action_failures"
            ).fetchone()[0]
            
            # Failures by category
            by_category = conn.execute("""
                SELECT error_category, COUNT(*) as count
                FROM action_failures
                GROUP BY error_category
                ORDER BY count DESC
            """).fetchall()
            
            # Failure patterns
            patterns = conn.execute("""
                SELECT action_name, error_category, size_bin, 
                       failure_count, success_count, should_skip, skip_reason
                FROM failure_patterns
                WHERE failure_count > 0
                ORDER BY failure_count DESC
                LIMIT 50
            """).fetchall()
            
            return {
                "total_failures": total,
                "by_category": [
                    {"category": row[0], "count": row[1]}
                    for row in by_category
                ],
                "patterns": [
                    {
                        "action": row["action_name"],
                        "category": row["error_category"],
                        "size_bin": row["size_bin"],
                        "failures": row["failure_count"],
                        "successes": row["success_count"],
                        "skip": bool(row["should_skip"]),
                        "reason": row["skip_reason"],
                    }
                    for row in patterns
                ],
            }
    
    def check_version_change(self) -> bool:
        """Check if software versions have changed since last run.
        
        If versions changed, skip recommendations from older versions
        are reset to give the new version a fresh chance.
        
        Returns:
            True if versions changed (and skips were reset)
        """
        pymeshlab_version = get_pymeshlab_version()
        meshprep_version = get_meshprep_version()
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        
        with self._get_connection() as conn:
            # Check if version_history table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='version_history'"
            )
            if not cursor.fetchone():
                # Create table if it doesn't exist
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS version_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pymeshlab_version TEXT,
                        meshprep_version TEXT,
                        python_version TEXT,
                        first_seen TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(pymeshlab_version, meshprep_version, python_version)
                    )
                """)
            
            # Check if this version combination exists
            existing = conn.execute("""
                SELECT id FROM version_history 
                WHERE pymeshlab_version = ? AND meshprep_version = ? AND python_version = ?
            """, (pymeshlab_version, meshprep_version, python_version)).fetchone()
            
            if existing:
                # Same version, no change
                return False
            
            # New version combination - record it
            conn.execute("""
                INSERT OR IGNORE INTO version_history (pymeshlab_version, meshprep_version, python_version)
                VALUES (?, ?, ?)
            """, (pymeshlab_version, meshprep_version, python_version))
            
            # Check if there are any skip recommendations from older versions
            old_skips = conn.execute("""
                SELECT COUNT(*) FROM failure_patterns
                WHERE should_skip = 1 
                AND (COALESCE(pymeshlab_version, '') != ? OR COALESCE(meshprep_version, 'unknown') != ?)
            """, (pymeshlab_version, meshprep_version)).fetchone()[0]
            
            if old_skips > 0:
                logger.info(
                    f"Version change detected (MeshPrep {meshprep_version}, PyMeshLab {pymeshlab_version}). "
                    f"{old_skips} skip recommendations from older versions won't apply to new version."
                )
                return True
            
            return False
    
    def reset_skips_for_current_version(self) -> int:
        """Reset all skip recommendations for the current version.
        
        This allows retrying actions that previously failed, useful when:
        - A bug has been fixed in the current version
        - User wants to give actions another chance
        
        Returns:
            Number of skip recommendations reset
        """
        pymeshlab_version = get_pymeshlab_version()
        meshprep_version = get_meshprep_version()
        
        with self._get_connection() as conn:
            result = conn.execute("""
                UPDATE failure_patterns
                SET should_skip = 0, skip_reason = NULL
                WHERE pymeshlab_version = ? AND meshprep_version = ? AND should_skip = 1
            """, (pymeshlab_version, meshprep_version))
            
            count = result.rowcount
            if count > 0:
                logger.info(f"Reset {count} skip recommendations for MeshPrep {meshprep_version}")
            return count
    
    def get_version_info(self) -> Dict[str, Any]:
        """Get version tracking information."""
        pymeshlab_version = get_pymeshlab_version()
        meshprep_version = get_meshprep_version()
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        
        with self._get_connection() as conn:
            # Check if version_history table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='version_history'"
            )
            if not cursor.fetchone():
                return {
                    "current_version": {
                        "pymeshlab": pymeshlab_version,
                        "meshprep": meshprep_version,
                        "python": python_version,
                    },
                    "version_history": [],
                    "skips_by_version": [],
                }
            
            # Get all known versions
            versions = conn.execute("""
                SELECT pymeshlab_version, meshprep_version, python_version, first_seen
                FROM version_history
                ORDER BY first_seen DESC
            """).fetchall()
            
            # Count skips by version
            skips_by_version = conn.execute("""
                SELECT COALESCE(pymeshlab_version, 'unknown') as pv, 
                       COALESCE(meshprep_version, 'unknown') as mv, 
                       COUNT(*) as skip_count
                FROM failure_patterns
                WHERE should_skip = 1
                GROUP BY pv, mv
            """).fetchall()
            
            return {
                "current_version": {
                    "pymeshlab": pymeshlab_version,
                    "meshprep": meshprep_version,
                    "python": python_version,
                },
                "version_history": [
                    {
                        "pymeshlab": row["pymeshlab_version"],
                        "meshprep": row["meshprep_version"],
                        "python": row["python_version"],
                        "first_seen": row["first_seen"],
                    }
                    for row in versions
                ],
                "skips_by_version": [
                    {
                        "pymeshlab": row[0],
                        "meshprep": row[1],
                        "skip_count": row[2],
                    }
                    for row in skips_by_version
                ],
            }


# Global failure tracker instance
_failure_tracker: Optional[ActionFailureTracker] = None


def get_failure_tracker() -> ActionFailureTracker:
    """Get or create the global failure tracker."""
    global _failure_tracker
    if _failure_tracker is None:
        _failure_tracker = ActionFailureTracker()
    return _failure_tracker
