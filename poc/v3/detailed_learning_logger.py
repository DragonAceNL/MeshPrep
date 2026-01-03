# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Enhanced Learning Data Logger for MeshPrep.

This module provides comprehensive logging specifically for improving
the learning algorithm. It logs additional data points that are useful
for analysis but not captured by the standard learning engine.

Key additions:
- Action-level granularity (not just pipeline level)
- Mesh state changes after each action
- Failure modes and error categories
- Timing breakdowns
- Geometry preservation metrics
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def get_learning_log_db_path() -> Path:
    """Get path to the detailed learning log database."""
    data_path = Path(__file__).parent.parent / "v2" / ".." / ".." / "learning_data"
    data_path = data_path.resolve()
    data_path.mkdir(parents=True, exist_ok=True)
    return data_path / "learning_detailed_log.db"


class DetailedLearningLogger:
    """
    Logs detailed information for learning algorithm improvement.
    
    This captures data that helps answer questions like:
    - Which specific actions cause geometry loss?
    - What error patterns correlate with specific mesh characteristics?
    - How does action order affect success?
    - What are the early warning signs of a failed repair?
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or get_learning_log_db_path()
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize detailed logging tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Action-level results (most granular)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS action_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT,
                    pipeline_name TEXT,
                    attempt_number INTEGER,
                    action_index INTEGER,
                    action_name TEXT,
                    action_params_json TEXT,
                    
                    -- Timing
                    duration_ms REAL,
                    
                    -- Success/Failure
                    success INTEGER,
                    error_message TEXT,
                    error_category TEXT,  -- 'geometry_loss', 'crash', 'timeout', 'validation_fail'
                    
                    -- Mesh state BEFORE this action
                    faces_before INTEGER,
                    vertices_before INTEGER,
                    watertight_before INTEGER,
                    volume_before REAL,
                    body_count_before INTEGER,
                    
                    -- Mesh state AFTER this action
                    faces_after INTEGER,
                    vertices_after INTEGER,
                    watertight_after INTEGER,
                    volume_after REAL,
                    body_count_after INTEGER,
                    
                    -- Geometry change metrics
                    face_change_pct REAL,
                    volume_change_pct REAL,
                    geometry_preserved INTEGER,  -- 1 if change < threshold
                    
                    created_at TEXT
                )
            """)
            
            # Pipeline sequence analysis
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_sequences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT,
                    pipeline_name TEXT,
                    action_sequence TEXT,  -- JSON array of action names
                    
                    -- Outcome
                    success INTEGER,
                    final_watertight INTEGER,
                    final_manifold INTEGER,
                    
                    -- Which action fixed it (if successful)
                    winning_action_index INTEGER,
                    winning_action_name TEXT,
                    
                    -- Which action broke it (if failed)
                    breaking_action_index INTEGER,
                    breaking_action_name TEXT,
                    breaking_reason TEXT,
                    
                    created_at TEXT
                )
            """)
            
            # Failure mode analysis
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS failure_modes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT,
                    pipeline_name TEXT,
                    
                    -- Failure classification
                    failure_type TEXT,  -- 'geometry_loss', 'still_broken', 'crash', 'timeout', 'validation'
                    failure_stage TEXT,  -- 'action', 'slicer_validation', 'geometry_check'
                    
                    -- Context
                    mesh_profile TEXT,
                    issue_pattern TEXT,
                    body_count INTEGER,
                    face_count INTEGER,
                    
                    -- Details
                    error_message TEXT,
                    last_successful_action TEXT,
                    first_failed_action TEXT,
                    
                    -- Slicer feedback (if available)
                    slicer_issues_json TEXT,
                    slicer_warnings_json TEXT,
                    
                    created_at TEXT
                )
            """)
            
            # Mesh characteristic correlations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mesh_characteristics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT UNIQUE,
                    
                    -- Basic geometry
                    face_count INTEGER,
                    vertex_count INTEGER,
                    edge_count INTEGER,
                    
                    -- Topology
                    body_count INTEGER,
                    is_watertight INTEGER,
                    euler_number INTEGER,
                    has_degenerate_faces INTEGER,
                    degenerate_face_count INTEGER,
                    
                    -- Bounds and scale
                    extent_x REAL,
                    extent_y REAL,
                    extent_z REAL,
                    volume REAL,
                    surface_area REAL,
                    
                    -- Derived metrics
                    face_density REAL,  -- faces per unit volume
                    aspect_ratio REAL,  -- max extent / min extent
                    compactness REAL,  -- volume / bounding box volume
                    
                    -- Issues detected
                    issue_count INTEGER,
                    issues_json TEXT,
                    
                    -- Outcome
                    repair_success INTEGER,
                    required_blender INTEGER,
                    attempts_to_fix INTEGER,
                    winning_pipeline TEXT,
                    
                    created_at TEXT
                )
            """)
            
            # Action effectiveness by mesh type
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS action_effectiveness (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action_name TEXT,
                    action_params_json TEXT,
                    
                    -- Mesh context
                    mesh_profile TEXT,
                    issue_type TEXT,
                    body_count_bucket TEXT,
                    face_count_bucket TEXT,
                    
                    -- Effectiveness
                    times_tried INTEGER DEFAULT 0,
                    times_improved INTEGER DEFAULT 0,  -- Made mesh better
                    times_broke INTEGER DEFAULT 0,  -- Made mesh worse
                    times_fixed INTEGER DEFAULT 0,  -- Fixed the issue completely
                    times_neutral INTEGER DEFAULT 0,  -- No change
                    
                    -- Avg metrics
                    avg_face_change_pct REAL,
                    avg_volume_change_pct REAL,
                    avg_duration_ms REAL,
                    
                    updated_at TEXT,
                    UNIQUE(action_name, action_params_json, mesh_profile, issue_type)
                )
            """)
            
            # Indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_action_results_model ON action_results(model_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_action_results_action ON action_results(action_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_failure_modes_type ON failure_modes(failure_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_mesh_char_profile ON mesh_characteristics(repair_success)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_action_eff_action ON action_effectiveness(action_name)")
            
            conn.commit()
            logger.info(f"Detailed learning log database initialized at {self.db_path}")
    
    @contextmanager
    def _get_connection(self):
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def log_action_result(
        self,
        model_id: str,
        pipeline_name: str,
        attempt_number: int,
        action_index: int,
        action_name: str,
        action_params: Dict[str, Any],
        duration_ms: float,
        success: bool,
        error_message: Optional[str] = None,
        error_category: Optional[str] = None,
        mesh_before: Optional[Dict[str, Any]] = None,
        mesh_after: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log the result of a single action execution."""
        
        # Extract mesh state
        before = mesh_before or {}
        after = mesh_after or {}
        
        faces_before = before.get("faces", 0)
        faces_after = after.get("faces", 0)
        volume_before = before.get("volume", 0) or 0
        volume_after = after.get("volume", 0) or 0
        
        # Calculate changes
        face_change_pct = ((faces_after - faces_before) / faces_before * 100) if faces_before > 0 else 0
        volume_change_pct = ((volume_after - volume_before) / volume_before * 100) if volume_before > 0 else 0
        geometry_preserved = 1 if abs(face_change_pct) < 30 and abs(volume_change_pct) < 30 else 0
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO action_results (
                    model_id, pipeline_name, attempt_number, action_index, action_name, action_params_json,
                    duration_ms, success, error_message, error_category,
                    faces_before, vertices_before, watertight_before, volume_before, body_count_before,
                    faces_after, vertices_after, watertight_after, volume_after, body_count_after,
                    face_change_pct, volume_change_pct, geometry_preserved, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_id, pipeline_name, attempt_number, action_index, action_name,
                json.dumps(action_params),
                duration_ms, 1 if success else 0, error_message, error_category,
                faces_before, before.get("vertices", 0), 1 if before.get("is_watertight") else 0,
                volume_before, before.get("body_count", 1),
                faces_after, after.get("vertices", 0), 1 if after.get("is_watertight") else 0,
                volume_after, after.get("body_count", 1),
                face_change_pct, volume_change_pct, geometry_preserved,
                datetime.now().isoformat()
            ))
    
    def log_failure_mode(
        self,
        model_id: str,
        pipeline_name: str,
        failure_type: str,
        failure_stage: str,
        mesh_profile: str,
        issue_pattern: str,
        body_count: int,
        face_count: int,
        error_message: Optional[str] = None,
        last_successful_action: Optional[str] = None,
        first_failed_action: Optional[str] = None,
        slicer_issues: Optional[List[str]] = None,
        slicer_warnings: Optional[List[str]] = None,
    ) -> None:
        """Log a failure for analysis."""
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO failure_modes (
                    model_id, pipeline_name, failure_type, failure_stage,
                    mesh_profile, issue_pattern, body_count, face_count,
                    error_message, last_successful_action, first_failed_action,
                    slicer_issues_json, slicer_warnings_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_id, pipeline_name, failure_type, failure_stage,
                mesh_profile, issue_pattern, body_count, face_count,
                error_message, last_successful_action, first_failed_action,
                json.dumps(slicer_issues or []), json.dumps(slicer_warnings or []),
                datetime.now().isoformat()
            ))
    
    def log_mesh_characteristics(
        self,
        model_id: str,
        diagnostics: Dict[str, Any],
        issues: List[str],
        repair_success: bool,
        required_blender: bool,
        attempts_to_fix: int,
        winning_pipeline: Optional[str] = None,
    ) -> None:
        """Log mesh characteristics with outcome for correlation analysis."""
        
        d = diagnostics or {}
        extents = d.get("extents", [1, 1, 1]) or [1, 1, 1]
        if len(extents) < 3:
            extents = [1, 1, 1]
        
        # Calculate derived metrics
        volume = d.get("volume", 0) or 0
        extent_x, extent_y, extent_z = extents[0], extents[1], extents[2]
        bbox_volume = extent_x * extent_y * extent_z if all(e > 0 for e in [extent_x, extent_y, extent_z]) else 1
        compactness = volume / bbox_volume if bbox_volume > 0 else 0
        
        max_extent = max(extents)
        min_extent = min(e for e in extents if e > 0) if any(e > 0 for e in extents) else 1
        aspect_ratio = max_extent / min_extent if min_extent > 0 else 1
        
        face_count = d.get("faces", 0)
        face_density = face_count / volume if volume > 0 else 0
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO mesh_characteristics (
                    model_id, face_count, vertex_count, edge_count,
                    body_count, is_watertight, euler_number,
                    has_degenerate_faces, degenerate_face_count,
                    extent_x, extent_y, extent_z, volume, surface_area,
                    face_density, aspect_ratio, compactness,
                    issue_count, issues_json,
                    repair_success, required_blender, attempts_to_fix, winning_pipeline,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_id, face_count, d.get("vertices", 0), d.get("edges", 0),
                d.get("body_count", 1), 1 if d.get("is_watertight") else 0, d.get("euler_number"),
                1 if d.get("degenerate_faces", 0) > 0 else 0, d.get("degenerate_faces", 0),
                extent_x, extent_y, extent_z, volume, d.get("surface_area", 0),
                face_density, aspect_ratio, compactness,
                len(issues), json.dumps(issues),
                1 if repair_success else 0, 1 if required_blender else 0,
                attempts_to_fix, winning_pipeline,
                datetime.now().isoformat()
            ))
    
    def update_action_effectiveness(
        self,
        action_name: str,
        action_params: Dict[str, Any],
        mesh_profile: str,
        issue_type: str,
        body_count_bucket: str,
        face_count_bucket: str,
        improved: bool,
        broke: bool,
        fixed: bool,
        face_change_pct: float,
        volume_change_pct: float,
        duration_ms: float,
    ) -> None:
        """Update action effectiveness statistics."""
        
        params_json = json.dumps(action_params, sort_keys=True)
        now = datetime.now().isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get existing record
            cursor.execute("""
                SELECT times_tried, times_improved, times_broke, times_fixed, times_neutral,
                       avg_face_change_pct, avg_volume_change_pct, avg_duration_ms
                FROM action_effectiveness
                WHERE action_name = ? AND action_params_json = ? AND mesh_profile = ? AND issue_type = ?
            """, (action_name, params_json, mesh_profile, issue_type))
            
            row = cursor.fetchone()
            
            if row:
                # Update existing
                times_tried = row["times_tried"] + 1
                times_improved = row["times_improved"] + (1 if improved else 0)
                times_broke = row["times_broke"] + (1 if broke else 0)
                times_fixed = row["times_fixed"] + (1 if fixed else 0)
                times_neutral = row["times_neutral"] + (1 if not improved and not broke and not fixed else 0)
                
                # Running average
                avg_face = (row["avg_face_change_pct"] * row["times_tried"] + face_change_pct) / times_tried
                avg_vol = (row["avg_volume_change_pct"] * row["times_tried"] + volume_change_pct) / times_tried
                avg_dur = (row["avg_duration_ms"] * row["times_tried"] + duration_ms) / times_tried
                
                cursor.execute("""
                    UPDATE action_effectiveness
                    SET times_tried = ?, times_improved = ?, times_broke = ?, times_fixed = ?, times_neutral = ?,
                        avg_face_change_pct = ?, avg_volume_change_pct = ?, avg_duration_ms = ?, updated_at = ?
                    WHERE action_name = ? AND action_params_json = ? AND mesh_profile = ? AND issue_type = ?
                """, (
                    times_tried, times_improved, times_broke, times_fixed, times_neutral,
                    avg_face, avg_vol, avg_dur, now,
                    action_name, params_json, mesh_profile, issue_type
                ))
            else:
                # Insert new
                cursor.execute("""
                    INSERT INTO action_effectiveness (
                        action_name, action_params_json, mesh_profile, issue_type,
                        body_count_bucket, face_count_bucket,
                        times_tried, times_improved, times_broke, times_fixed, times_neutral,
                        avg_face_change_pct, avg_volume_change_pct, avg_duration_ms, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    action_name, params_json, mesh_profile, issue_type,
                    body_count_bucket, face_count_bucket,
                    1 if improved else 0, 1 if broke else 0, 1 if fixed else 0,
                    1 if not improved and not broke and not fixed else 0,
                    face_change_pct, volume_change_pct, duration_ms, now
                ))
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary data for learning algorithm analysis."""
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Most problematic actions (highest break rate)
            cursor.execute("""
                SELECT action_name, 
                       SUM(times_tried) as total,
                       SUM(times_broke) as broke,
                       SUM(times_fixed) as fixed,
                       CAST(SUM(times_broke) AS REAL) / NULLIF(SUM(times_tried), 0) as break_rate,
                       CAST(SUM(times_fixed) AS REAL) / NULLIF(SUM(times_tried), 0) as fix_rate
                FROM action_effectiveness
                GROUP BY action_name
                HAVING total >= 5
                ORDER BY break_rate DESC
                LIMIT 10
            """)
            problematic_actions = [dict(row) for row in cursor.fetchall()]
            
            # Best actions per issue type
            cursor.execute("""
                SELECT issue_type, action_name,
                       times_tried, times_fixed,
                       CAST(times_fixed AS REAL) / NULLIF(times_tried, 0) as fix_rate
                FROM action_effectiveness
                WHERE times_tried >= 3
                ORDER BY issue_type, fix_rate DESC
            """)
            action_by_issue = {}
            for row in cursor.fetchall():
                issue = row["issue_type"]
                if issue not in action_by_issue:
                    action_by_issue[issue] = []
                action_by_issue[issue].append({
                    "action": row["action_name"],
                    "fix_rate": row["fix_rate"],
                    "attempts": row["times_tried"],
                })
            
            # Common failure modes
            cursor.execute("""
                SELECT failure_type, failure_stage, COUNT(*) as count
                FROM failure_modes
                GROUP BY failure_type, failure_stage
                ORDER BY count DESC
                LIMIT 15
            """)
            failure_modes = [dict(row) for row in cursor.fetchall()]
            
            # Mesh characteristics that predict failure
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN body_count = 1 THEN '1 body'
                        WHEN body_count <= 5 THEN '2-5 bodies'
                        WHEN body_count <= 20 THEN '6-20 bodies'
                        ELSE '20+ bodies'
                    END as body_bucket,
                    COUNT(*) as total,
                    SUM(repair_success) as successes,
                    CAST(SUM(repair_success) AS REAL) / COUNT(*) as success_rate
                FROM mesh_characteristics
                GROUP BY body_bucket
                ORDER BY success_rate ASC
            """)
            success_by_body_count = [dict(row) for row in cursor.fetchall()]
            
            return {
                "problematic_actions": problematic_actions,
                "best_actions_by_issue": action_by_issue,
                "common_failure_modes": failure_modes,
                "success_by_body_count": success_by_body_count,
            }


# Global instance
_detailed_logger: Optional[DetailedLearningLogger] = None


def get_detailed_logger() -> DetailedLearningLogger:
    """Get or create the global detailed logger instance."""
    global _detailed_logger
    if _detailed_logger is None:
        _detailed_logger = DetailedLearningLogger()
    return _detailed_logger
