# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Action Duration Predictor for MeshPrep.

Learns expected duration for each action based on mesh characteristics,
and detects when actions are likely hanging (taking too long).

Key features:
- Predicts expected duration based on face count, vertex count, body count
- Learns from successful completions
- Detects hangs when actual time exceeds prediction by a factor
- Remembers which action+mesh characteristic combinations cause hangs
- Provides timeout recommendations for each action
"""

import json
import logging
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Database path
DB_PATH = Path(__file__).parent.parent / "v2" / "learning_data" / "action_duration.db"

# Default timeout factor - if action takes this many times longer than predicted, assume hang
DEFAULT_HANG_FACTOR = 5.0

# Minimum samples before we trust predictions
MIN_SAMPLES_FOR_PREDICTION = 3

# Default timeouts per action (seconds) - used when no learned data exists
DEFAULT_ACTION_TIMEOUTS = {
    "pymeshfix_repair": 60,
    "fill_holes": 30,
    "fix_normals": 10,
    "remove_degenerate": 10,
    "blender_remesh": 300,
    "blender_boolean_union": 180,
    "poisson_reconstruction": 120,
    "ball_pivoting": 90,
    "voxel_remesh": 60,
    "merge_close_vertices": 30,
    "default": 60,
}

# Mesh size categories for binning
SIZE_BINS = [
    (10_000, "tiny"),       # < 10k faces
    (50_000, "small"),      # 10k - 50k faces
    (100_000, "medium"),    # 50k - 100k faces
    (500_000, "large"),     # 100k - 500k faces
    (float('inf'), "huge"), # > 500k faces
]

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_info (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- Duration observations for each action
CREATE TABLE IF NOT EXISTS action_durations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action_name TEXT NOT NULL,
    
    -- Mesh characteristics
    face_count INTEGER NOT NULL,
    vertex_count INTEGER NOT NULL,
    body_count INTEGER DEFAULT 1,
    size_bin TEXT NOT NULL,
    
    -- Timing
    duration_ms REAL NOT NULL,
    success INTEGER NOT NULL,  -- 1 = completed, 0 = timed out/hung
    
    -- Context
    pipeline_name TEXT,
    model_id TEXT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexing
    UNIQUE(action_name, model_id, pipeline_name)
);

-- Learned hang patterns (action + size combinations that hang)
CREATE TABLE IF NOT EXISTS hang_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action_name TEXT NOT NULL,
    size_bin TEXT NOT NULL,
    min_faces INTEGER NOT NULL,
    
    -- Statistics
    hang_count INTEGER DEFAULT 1,
    success_count INTEGER DEFAULT 0,
    last_hang TEXT,
    
    -- Recommendation
    should_skip INTEGER DEFAULT 0,  -- 1 = skip this action for this size
    alternative_action TEXT,         -- Suggested alternative
    
    UNIQUE(action_name, size_bin)
);

-- Aggregated statistics per action+size
CREATE TABLE IF NOT EXISTS action_stats (
    action_name TEXT NOT NULL,
    size_bin TEXT NOT NULL,
    
    -- Duration stats
    sample_count INTEGER DEFAULT 0,
    avg_duration_ms REAL DEFAULT 0,
    max_duration_ms REAL DEFAULT 0,
    stddev_duration_ms REAL DEFAULT 0,
    
    -- Derived
    predicted_timeout_ms REAL DEFAULT 0,
    
    -- Success tracking
    success_count INTEGER DEFAULT 0,
    hang_count INTEGER DEFAULT 0,
    
    last_updated TEXT,
    
    PRIMARY KEY(action_name, size_bin)
);

CREATE INDEX IF NOT EXISTS idx_durations_action ON action_durations(action_name);
CREATE INDEX IF NOT EXISTS idx_durations_size ON action_durations(size_bin);
CREATE INDEX IF NOT EXISTS idx_hang_patterns_action ON hang_patterns(action_name);

INSERT OR REPLACE INTO schema_info (key, value) VALUES ('version', '1');
"""


@dataclass
class DurationPrediction:
    """Prediction for how long an action should take."""
    action_name: str
    predicted_ms: float
    timeout_ms: float
    confidence: float  # 0-1, based on sample count
    sample_count: int
    size_bin: str
    hang_risk: float  # 0-1, probability of hang based on history
    should_skip: bool
    alternative_action: Optional[str] = None


@dataclass
class MeshCharacteristics:
    """Mesh characteristics for duration prediction."""
    face_count: int
    vertex_count: int
    body_count: int = 1
    
    @property
    def size_bin(self) -> str:
        """Get size category based on face count."""
        for threshold, bin_name in SIZE_BINS:
            if self.face_count < threshold:
                return bin_name
        return "huge"
    
    @property
    def complexity_score(self) -> float:
        """Calculate complexity score for duration scaling."""
        # Faces are primary driver, bodies add complexity
        base = self.face_count / 10_000  # Normalize to 10k faces = 1.0
        body_factor = 1.0 + (self.body_count - 1) * 0.2  # Each extra body adds 20%
        return base * body_factor


class ActionDurationPredictor:
    """Predicts action durations and detects hangs based on learned patterns."""
    
    def __init__(self, db_path: Optional[Path] = None, hang_factor: float = DEFAULT_HANG_FACTOR):
        self.db_path = db_path or DB_PATH
        self.hang_factor = hang_factor
        self._ensure_db_dir()
        self._init_db()
    
    def _ensure_db_dir(self):
        """Ensure database directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(SCHEMA_SQL)
    
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
    
    def predict_duration(
        self, 
        action_name: str, 
        mesh: MeshCharacteristics
    ) -> DurationPrediction:
        """Predict how long an action should take for a given mesh.
        
        Args:
            action_name: Name of the action (e.g., 'pymeshfix_repair')
            mesh: Mesh characteristics
            
        Returns:
            DurationPrediction with expected time and timeout recommendation
        """
        size_bin = mesh.size_bin
        
        with self._get_connection() as conn:
            # Get learned statistics for this action + size
            row = conn.execute("""
                SELECT * FROM action_stats 
                WHERE action_name = ? AND size_bin = ?
            """, (action_name, size_bin)).fetchone()
            
            # Check for known hang patterns
            hang_row = conn.execute("""
                SELECT * FROM hang_patterns
                WHERE action_name = ? AND size_bin = ?
            """, (action_name, size_bin)).fetchone()
        
        # Calculate hang risk
        hang_risk = 0.0
        should_skip = False
        alternative = None
        
        if hang_row:
            total = hang_row["hang_count"] + hang_row["success_count"]
            if total > 0:
                hang_risk = hang_row["hang_count"] / total
            should_skip = bool(hang_row["should_skip"])
            alternative = hang_row["alternative_action"]
        
        if row and row["sample_count"] >= MIN_SAMPLES_FOR_PREDICTION:
            # Use learned statistics
            avg_ms = row["avg_duration_ms"]
            stddev_ms = row["stddev_duration_ms"] or avg_ms * 0.5
            sample_count = row["sample_count"]
            
            # Scale by mesh complexity relative to average
            complexity = mesh.complexity_score
            scaled_ms = avg_ms * (1 + math.log1p(complexity) * 0.3)
            
            # Timeout is predicted + hang_factor * stddev (with minimum)
            timeout_ms = max(
                scaled_ms * self.hang_factor,
                scaled_ms + stddev_ms * 3,
                5000  # Minimum 5 seconds
            )
            
            confidence = min(1.0, sample_count / 20)  # Full confidence at 20 samples
            
            return DurationPrediction(
                action_name=action_name,
                predicted_ms=scaled_ms,
                timeout_ms=timeout_ms,
                confidence=confidence,
                sample_count=sample_count,
                size_bin=size_bin,
                hang_risk=hang_risk,
                should_skip=should_skip,
                alternative_action=alternative,
            )
        else:
            # Use default timeout, scaled by complexity
            default_s = DEFAULT_ACTION_TIMEOUTS.get(action_name, DEFAULT_ACTION_TIMEOUTS["default"])
            complexity = mesh.complexity_score
            scaled_s = default_s * (1 + math.log1p(complexity) * 0.5)
            
            return DurationPrediction(
                action_name=action_name,
                predicted_ms=scaled_s * 1000 * 0.5,  # Predict half of timeout
                timeout_ms=scaled_s * 1000,
                confidence=0.0,  # No learned data
                sample_count=row["sample_count"] if row else 0,
                size_bin=size_bin,
                hang_risk=hang_risk,
                should_skip=should_skip,
                alternative_action=alternative,
            )
    
    def record_duration(
        self,
        action_name: str,
        mesh: MeshCharacteristics,
        duration_ms: float,
        success: bool,
        model_id: str = "",
        pipeline_name: str = "",
    ) -> None:
        """Record an observed action duration.
        
        Args:
            action_name: Name of the action
            mesh: Mesh characteristics
            duration_ms: How long the action took (or timeout if hung)
            success: True if completed, False if timed out/hung
            model_id: Model identifier
            pipeline_name: Pipeline that ran this action
        """
        size_bin = mesh.size_bin
        
        with self._get_connection() as conn:
            # Record individual observation
            conn.execute("""
                INSERT OR REPLACE INTO action_durations
                (action_name, face_count, vertex_count, body_count, size_bin,
                 duration_ms, success, pipeline_name, model_id, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                action_name, mesh.face_count, mesh.vertex_count, mesh.body_count,
                size_bin, duration_ms, 1 if success else 0, pipeline_name, model_id,
                datetime.now().isoformat()
            ))
            
            # Update hang patterns if this was a hang
            if not success:
                conn.execute("""
                    INSERT INTO hang_patterns (action_name, size_bin, min_faces, hang_count, last_hang)
                    VALUES (?, ?, ?, 1, ?)
                    ON CONFLICT(action_name, size_bin) DO UPDATE SET
                        hang_count = hang_count + 1,
                        min_faces = MIN(min_faces, excluded.min_faces),
                        last_hang = excluded.last_hang
                """, (action_name, size_bin, mesh.face_count, datetime.now().isoformat()))
                
                # If hang rate is high, mark as should_skip
                hang_row = conn.execute("""
                    SELECT hang_count, success_count FROM hang_patterns
                    WHERE action_name = ? AND size_bin = ?
                """, (action_name, size_bin)).fetchone()
                
                if hang_row:
                    total = hang_row["hang_count"] + hang_row["success_count"]
                    hang_rate = hang_row["hang_count"] / total if total > 0 else 0
                    
                    # If >50% hangs and at least 2 observations, mark for skipping
                    if hang_rate > 0.5 and total >= 2:
                        conn.execute("""
                            UPDATE hang_patterns SET should_skip = 1
                            WHERE action_name = ? AND size_bin = ?
                        """, (action_name, size_bin))
                        logger.warning(
                            f"Marked {action_name} as should_skip for {size_bin} meshes "
                            f"(hang rate: {hang_rate:.0%})"
                        )
            else:
                # Record success in hang patterns
                conn.execute("""
                    INSERT INTO hang_patterns (action_name, size_bin, min_faces, success_count)
                    VALUES (?, ?, ?, 1)
                    ON CONFLICT(action_name, size_bin) DO UPDATE SET
                        success_count = success_count + 1
                """, (action_name, size_bin, mesh.face_count))
            
            # Update aggregated statistics
            self._update_stats(conn, action_name, size_bin)
    
    def _update_stats(self, conn, action_name: str, size_bin: str) -> None:
        """Update aggregated statistics for an action+size combination."""
        # Calculate stats from successful observations only
        stats = conn.execute("""
            SELECT 
                COUNT(*) as count,
                AVG(duration_ms) as avg,
                MAX(duration_ms) as max,
                SUM(success) as success_count,
                COUNT(*) - SUM(success) as hang_count
            FROM action_durations
            WHERE action_name = ? AND size_bin = ? AND success = 1
        """, (action_name, size_bin)).fetchone()
        
        if stats and stats["count"] > 0:
            # Calculate stddev
            stddev_row = conn.execute("""
                SELECT AVG((duration_ms - ?) * (duration_ms - ?)) as variance
                FROM action_durations
                WHERE action_name = ? AND size_bin = ? AND success = 1
            """, (stats["avg"], stats["avg"], action_name, size_bin)).fetchone()
            
            stddev = math.sqrt(stddev_row["variance"]) if stddev_row["variance"] else stats["avg"] * 0.5
            
            # Predicted timeout: avg + 3*stddev, scaled by hang_factor
            predicted_timeout = (stats["avg"] + stddev * 3) * self.hang_factor
            
            conn.execute("""
                INSERT INTO action_stats 
                (action_name, size_bin, sample_count, avg_duration_ms, max_duration_ms,
                 stddev_duration_ms, predicted_timeout_ms, success_count, hang_count, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(action_name, size_bin) DO UPDATE SET
                    sample_count = excluded.sample_count,
                    avg_duration_ms = excluded.avg_duration_ms,
                    max_duration_ms = excluded.max_duration_ms,
                    stddev_duration_ms = excluded.stddev_duration_ms,
                    predicted_timeout_ms = excluded.predicted_timeout_ms,
                    success_count = excluded.success_count,
                    hang_count = excluded.hang_count,
                    last_updated = excluded.last_updated
            """, (
                action_name, size_bin, stats["count"], stats["avg"], stats["max"],
                stddev, predicted_timeout, stats["success_count"], stats["hang_count"],
                datetime.now().isoformat()
            ))
    
    def get_actions_to_skip(self, mesh: MeshCharacteristics) -> List[str]:
        """Get list of actions that should be skipped for this mesh size.
        
        Args:
            mesh: Mesh characteristics
            
        Returns:
            List of action names that historically hang on this size
        """
        size_bin = mesh.size_bin
        
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT action_name FROM hang_patterns
                WHERE size_bin = ? AND should_skip = 1
            """, (size_bin,)).fetchall()
        
        return [row["action_name"] for row in rows]
    
    def get_stats_summary(self) -> Dict:
        """Get summary of learned duration statistics."""
        with self._get_connection() as conn:
            # Total observations
            total = conn.execute("SELECT COUNT(*) FROM action_durations").fetchone()[0]
            
            # Unique actions tracked
            actions = conn.execute("SELECT COUNT(DISTINCT action_name) FROM action_durations").fetchone()[0]
            
            # Hang patterns
            hangs = conn.execute("SELECT COUNT(*) FROM hang_patterns WHERE should_skip = 1").fetchone()[0]
            
            # Per-action summary
            action_summary = conn.execute("""
                SELECT action_name, 
                       SUM(sample_count) as total_samples,
                       AVG(avg_duration_ms) as avg_ms,
                       SUM(hang_count) as hangs
                FROM action_stats
                GROUP BY action_name
                ORDER BY total_samples DESC
            """).fetchall()
        
        return {
            "total_observations": total,
            "actions_tracked": actions,
            "known_hang_patterns": hangs,
            "action_summary": [
                {
                    "action": row["action_name"],
                    "samples": row["total_samples"],
                    "avg_ms": row["avg_ms"],
                    "hangs": row["hangs"],
                }
                for row in action_summary
            ],
        }
    
    def set_alternative_action(self, action_name: str, size_bin: str, alternative: str) -> None:
        """Set an alternative action to use when an action hangs.
        
        Args:
            action_name: Action that hangs
            size_bin: Size category where it hangs
            alternative: Action to use instead
        """
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE hang_patterns SET alternative_action = ?
                WHERE action_name = ? AND size_bin = ?
            """, (alternative, action_name, size_bin))


# Global instance
_predictor_instance: Optional[ActionDurationPredictor] = None


def get_duration_predictor() -> ActionDurationPredictor:
    """Get or create the global duration predictor instance."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = ActionDurationPredictor()
    return _predictor_instance
