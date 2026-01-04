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
- Learns DYNAMIC timeout factors per action (not fixed multipliers)
- Detects hangs when actual time exceeds learned safe maximum
- Remembers which action+mesh characteristic combinations cause hangs
- Provides timeout recommendations for each action

The timeout factor learning works by tracking the ratio of max_duration/avg_duration
for each action. Actions with high variance get higher timeout factors automatically.
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
DB_PATH = Path(__file__).parent.parent / "learning_data" / "action_duration.db"

# Initial timeout factor before we have learned data
# This is conservative - we'll learn tighter bounds over time
INITIAL_TIMEOUT_FACTOR = 10.0

# Minimum timeout factor (even for very consistent actions)
MIN_TIMEOUT_FACTOR = 3.0

# Maximum timeout factor (cap for very inconsistent actions)
MAX_TIMEOUT_FACTOR = 20.0

# Minimum samples before we trust predictions
MIN_SAMPLES_FOR_PREDICTION = 3

# Minimum samples before we adjust timeout factor
MIN_SAMPLES_FOR_FACTOR_LEARNING = 5

# Default base timeouts per action (seconds) - used when no learned data exists
DEFAULT_ACTION_BASE_TIMEOUTS = {
    "pymeshfix_repair": 30,
    "fill_holes": 15,
    "fix_normals": 5,
    "remove_degenerate": 5,
    "blender_remesh": 120,
    "blender_boolean_union": 90,
    "poisson_reconstruction": 60,
    "ball_pivoting": 45,
    "voxel_remesh": 30,
    "merge_close_vertices": 15,
    "trimesh_basic": 5,
    "make_manifold": 20,
    "default": 30,
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
    success INTEGER NOT NULL,  -- 1 = completed normally, 0 = timed out/hung
    
    -- Context
    pipeline_name TEXT,
    model_id TEXT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexing (allow multiple observations per model for different pipelines)
    UNIQUE(action_name, model_id, pipeline_name, timestamp)
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

-- Aggregated statistics per action+size with LEARNED timeout factors
CREATE TABLE IF NOT EXISTS action_stats (
    action_name TEXT NOT NULL,
    size_bin TEXT NOT NULL,
    
    -- Duration stats (from successful completions only)
    sample_count INTEGER DEFAULT 0,
    avg_duration_ms REAL DEFAULT 0,
    min_duration_ms REAL DEFAULT 0,
    max_duration_ms REAL DEFAULT 0,
    stddev_duration_ms REAL DEFAULT 0,
    
    -- Percentile stats for better timeout estimation
    p50_duration_ms REAL DEFAULT 0,  -- Median
    p90_duration_ms REAL DEFAULT 0,  -- 90th percentile
    p99_duration_ms REAL DEFAULT 0,  -- 99th percentile
    
    -- LEARNED timeout factor for this action+size
    -- This is the key learning: ratio of safe_max/avg that works for this action
    learned_timeout_factor REAL DEFAULT 10.0,
    
    -- Derived timeout (calculated from learned factor)
    recommended_timeout_ms REAL DEFAULT 0,
    
    -- Success tracking
    success_count INTEGER DEFAULT 0,
    hang_count INTEGER DEFAULT 0,
    false_positive_count INTEGER DEFAULT 0,  -- Times we thought it hung but it was just slow
    
    last_updated TEXT,
    
    PRIMARY KEY(action_name, size_bin)
);

-- Track individual slow completions to learn timeout factors
-- If something completes after we would have killed it, that's valuable data
CREATE TABLE IF NOT EXISTS slow_completions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action_name TEXT NOT NULL,
    size_bin TEXT NOT NULL,
    duration_ms REAL NOT NULL,
    predicted_timeout_ms REAL NOT NULL,  -- What we would have used
    ratio_over_prediction REAL NOT NULL, -- How many X over prediction
    model_id TEXT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_durations_action ON action_durations(action_name);
CREATE INDEX IF NOT EXISTS idx_durations_size ON action_durations(size_bin);
CREATE INDEX IF NOT EXISTS idx_durations_success ON action_durations(success);
CREATE INDEX IF NOT EXISTS idx_hang_patterns_action ON hang_patterns(action_name);
CREATE INDEX IF NOT EXISTS idx_slow_completions_action ON slow_completions(action_name, size_bin);

INSERT OR REPLACE INTO schema_info (key, value) VALUES ('version', '2');
"""


@dataclass
class DurationPrediction:
    """Prediction for how long an action should take."""
    action_name: str
    size_bin: str
    
    # Timing predictions
    predicted_ms: float          # Expected duration (median or avg)
    soft_timeout_ms: float       # First warning threshold
    hard_timeout_ms: float       # Kill threshold
    
    # Learned factors
    timeout_factor: float        # Learned multiplier for this action
    confidence: float            # 0-1, based on sample count
    sample_count: int
    
    # Risk assessment
    hang_risk: float             # 0-1, probability of hang based on history
    variance_ratio: float        # stddev/avg - higher = more unpredictable
    
    # Recommendations
    should_skip: bool
    alternative_action: Optional[str] = None
    
    # Stats for debugging
    p90_ms: float = 0            # 90th percentile from history
    max_observed_ms: float = 0   # Maximum successful completion time


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
    """Predicts action durations and detects hangs based on learned patterns.
    
    The key innovation is learning the appropriate timeout factor per action:
    - Consistent actions (low variance) get lower factors (e.g., 3x)
    - Unpredictable actions (high variance) get higher factors (e.g., 15x)
    - The system learns from both successful completions AND slow completions
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
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
        
        Returns prediction with LEARNED timeout factors specific to this action.
        Actions with high variance automatically get higher timeout factors.
        
        Args:
            action_name: Name of the action (e.g., 'pymeshfix_repair')
            mesh: Mesh characteristics
            
        Returns:
            DurationPrediction with expected time and learned timeout factors
        """
        size_bin = mesh.size_bin
        
        with self._get_connection() as conn:
            # Get learned statistics for this action + size
            stats_row = conn.execute("""
                SELECT * FROM action_stats 
                WHERE action_name = ? AND size_bin = ?
            """, (action_name, size_bin)).fetchone()
            
            # Check for known hang patterns
            hang_row = conn.execute("""
                SELECT * FROM hang_patterns
                WHERE action_name = ? AND size_bin = ?
            """, (action_name, size_bin)).fetchone()
            
            # Get max observed successful completion for this action+size
            max_row = conn.execute("""
                SELECT MAX(duration_ms) as max_duration
                FROM action_durations
                WHERE action_name = ? AND size_bin = ? AND success = 1
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
        
        # Get max observed duration
        max_observed = max_row["max_duration"] if max_row and max_row["max_duration"] else 0
        
        if stats_row and stats_row["sample_count"] >= MIN_SAMPLES_FOR_PREDICTION:
            # Use learned statistics
            avg_ms = stats_row["avg_duration_ms"]
            stddev_ms = stats_row["stddev_duration_ms"] or avg_ms * 0.5
            p90_ms = stats_row["p90_duration_ms"] or avg_ms * 2
            sample_count = stats_row["sample_count"]
            
            # Get the LEARNED timeout factor for this action
            timeout_factor = stats_row["learned_timeout_factor"] or INITIAL_TIMEOUT_FACTOR
            
            # Clamp to reasonable bounds
            timeout_factor = max(MIN_TIMEOUT_FACTOR, min(MAX_TIMEOUT_FACTOR, timeout_factor))
            
            # Scale by mesh complexity relative to average for this size bin
            complexity = mesh.complexity_score
            complexity_scale = 1 + math.log1p(complexity) * 0.2  # Gentle scaling
            
            predicted_ms = avg_ms * complexity_scale
            
            # Soft timeout: p90 + buffer (warn but don't kill)
            soft_timeout_ms = max(p90_ms * 1.5, predicted_ms * 2) * complexity_scale
            
            # Hard timeout: use LEARNED factor
            hard_timeout_ms = max(
                predicted_ms * timeout_factor,
                max_observed * 1.5 if max_observed > 0 else predicted_ms * timeout_factor,
                soft_timeout_ms * 2,
                10000  # Minimum 10 seconds
            )
            
            # Calculate variance ratio for reporting
            variance_ratio = stddev_ms / avg_ms if avg_ms > 0 else 1.0
            
            confidence = min(1.0, sample_count / 20)  # Full confidence at 20 samples
            
            return DurationPrediction(
                action_name=action_name,
                size_bin=size_bin,
                predicted_ms=predicted_ms,
                soft_timeout_ms=soft_timeout_ms,
                hard_timeout_ms=hard_timeout_ms,
                timeout_factor=timeout_factor,
                confidence=confidence,
                sample_count=sample_count,
                hang_risk=hang_risk,
                variance_ratio=variance_ratio,
                should_skip=should_skip,
                alternative_action=alternative,
                p90_ms=p90_ms,
                max_observed_ms=max_observed,
            )
        else:
            # No learned data - use conservative defaults
            base_s = DEFAULT_ACTION_BASE_TIMEOUTS.get(action_name, DEFAULT_ACTION_BASE_TIMEOUTS["default"])
            complexity = mesh.complexity_score
            complexity_scale = 1 + math.log1p(complexity) * 0.3
            
            predicted_ms = base_s * 1000 * complexity_scale
            
            # Very conservative timeouts when we don't have data
            soft_timeout_ms = predicted_ms * 3
            hard_timeout_ms = predicted_ms * INITIAL_TIMEOUT_FACTOR
            
            return DurationPrediction(
                action_name=action_name,
                size_bin=size_bin,
                predicted_ms=predicted_ms,
                soft_timeout_ms=soft_timeout_ms,
                hard_timeout_ms=hard_timeout_ms,
                timeout_factor=INITIAL_TIMEOUT_FACTOR,
                confidence=0.0,
                sample_count=stats_row["sample_count"] if stats_row else 0,
                hang_risk=hang_risk,
                variance_ratio=1.0,  # Unknown
                should_skip=should_skip,
                alternative_action=alternative,
                p90_ms=0,
                max_observed_ms=max_observed,
            )
    
    def record_completion(
        self,
        action_name: str,
        mesh: MeshCharacteristics,
        duration_ms: float,
        model_id: str = "",
        pipeline_name: str = "",
        was_slow_warning: bool = False,
        predicted_timeout_ms: float = 0,
    ) -> None:
        """Record a successful action completion.
        
        This is called when an action completes normally (not timed out).
        If it was slower than our soft timeout, we record it as a slow completion
        to help learn better timeout factors.
        
        Args:
            action_name: Name of the action
            mesh: Mesh characteristics
            duration_ms: How long the action took
            model_id: Model identifier
            pipeline_name: Pipeline that ran this action
            was_slow_warning: True if this exceeded soft timeout but still completed
            predicted_timeout_ms: What our hard timeout was (for learning)
        """
        size_bin = mesh.size_bin
        
        with self._get_connection() as conn:
            # Record individual observation
            conn.execute("""
                INSERT INTO action_durations
                (action_name, face_count, vertex_count, body_count, size_bin,
                 duration_ms, success, pipeline_name, model_id, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
            """, (
                action_name, mesh.face_count, mesh.vertex_count, mesh.body_count,
                size_bin, duration_ms, pipeline_name, model_id,
                datetime.now().isoformat()
            ))
            
            # Record success in hang patterns
            conn.execute("""
                INSERT INTO hang_patterns (action_name, size_bin, min_faces, success_count)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(action_name, size_bin) DO UPDATE SET
                    success_count = success_count + 1
            """, (action_name, size_bin, mesh.face_count))
            
            # If this was a slow completion, record it for timeout factor learning
            if was_slow_warning and predicted_timeout_ms > 0:
                ratio = duration_ms / predicted_timeout_ms if predicted_timeout_ms > 0 else 1.0
                conn.execute("""
                    INSERT INTO slow_completions
                    (action_name, size_bin, duration_ms, predicted_timeout_ms, 
                     ratio_over_prediction, model_id, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    action_name, size_bin, duration_ms, predicted_timeout_ms,
                    ratio, model_id, datetime.now().isoformat()
                ))
                
                logger.info(
                    f"Recorded slow completion for {action_name}: "
                    f"{duration_ms/1000:.1f}s ({ratio:.1f}x of timeout)"
                )
            
            # Update aggregated statistics
            self._update_stats(conn, action_name, size_bin)
    
    def record_hang(
        self,
        action_name: str,
        mesh: MeshCharacteristics,
        timeout_ms: float,
        model_id: str = "",
        pipeline_name: str = "",
    ) -> None:
        """Record a hang (action was killed due to timeout).
        
        Args:
            action_name: Name of the action
            mesh: Mesh characteristics
            timeout_ms: The timeout that was used when we killed it
            model_id: Model identifier
            pipeline_name: Pipeline that ran this action
        """
        size_bin = mesh.size_bin
        
        with self._get_connection() as conn:
            # Record as unsuccessful
            conn.execute("""
                INSERT INTO action_durations
                (action_name, face_count, vertex_count, body_count, size_bin,
                 duration_ms, success, pipeline_name, model_id, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
            """, (
                action_name, mesh.face_count, mesh.vertex_count, mesh.body_count,
                size_bin, timeout_ms, pipeline_name, model_id,
                datetime.now().isoformat()
            ))
            
            # Update hang patterns
            conn.execute("""
                INSERT INTO hang_patterns (action_name, size_bin, min_faces, hang_count, last_hang)
                VALUES (?, ?, ?, 1, ?)
                ON CONFLICT(action_name, size_bin) DO UPDATE SET
                    hang_count = hang_count + 1,
                    min_faces = MIN(min_faces, excluded.min_faces),
                    last_hang = excluded.last_hang
            """, (action_name, size_bin, mesh.face_count, datetime.now().isoformat()))
            
            # Check if we should mark for skipping
            hang_row = conn.execute("""
                SELECT hang_count, success_count FROM hang_patterns
                WHERE action_name = ? AND size_bin = ?
            """, (action_name, size_bin)).fetchone()
            
            if hang_row:
                total = hang_row["hang_count"] + hang_row["success_count"]
                hang_rate = hang_row["hang_count"] / total if total > 0 else 0
                
                # Mark for skipping if:
                # - More than 50% hang rate AND at least 3 observations, OR
                # - More than 3 hangs regardless of success rate
                if (hang_rate > 0.5 and total >= 3) or hang_row["hang_count"] >= 3:
                    conn.execute("""
                        UPDATE hang_patterns SET should_skip = 1
                        WHERE action_name = ? AND size_bin = ?
                    """, (action_name, size_bin))
                    logger.warning(
                        f"Marked {action_name} as should_skip for {size_bin} meshes "
                        f"(hang rate: {hang_rate:.0%}, hangs: {hang_row['hang_count']})"
                    )
            
            # Update stats (won't include this in successful stats, but updates counts)
            self._update_stats(conn, action_name, size_bin)
    
    def _update_stats(self, conn, action_name: str, size_bin: str) -> None:
        """Update aggregated statistics and LEARN the timeout factor."""
        # Get all successful durations for percentile calculation
        durations_rows = conn.execute("""
            SELECT duration_ms FROM action_durations
            WHERE action_name = ? AND size_bin = ? AND success = 1
            ORDER BY duration_ms
        """, (action_name, size_bin)).fetchall()
        
        if not durations_rows:
            return
        
        durations = [row["duration_ms"] for row in durations_rows]
        n = len(durations)
        
        # Calculate statistics
        avg_ms = sum(durations) / n
        min_ms = min(durations)
        max_ms = max(durations)
        
        # Calculate percentiles
        p50_ms = durations[n // 2]
        p90_idx = int(n * 0.9)
        p99_idx = int(n * 0.99)
        p90_ms = durations[min(p90_idx, n - 1)]
        p99_ms = durations[min(p99_idx, n - 1)]
        
        # Calculate standard deviation
        if n > 1:
            variance = sum((d - avg_ms) ** 2 for d in durations) / (n - 1)
            stddev_ms = math.sqrt(variance)
        else:
            stddev_ms = avg_ms * 0.5
        
        # Get hang statistics
        hang_stats = conn.execute("""
            SELECT 
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count,
                SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as hang_count
            FROM action_durations
            WHERE action_name = ? AND size_bin = ?
        """, (action_name, size_bin)).fetchone()
        
        success_count = hang_stats["success_count"] or 0
        hang_count = hang_stats["hang_count"] or 0
        
        # LEARN THE TIMEOUT FACTOR
        # This is the key learning algorithm
        learned_factor = self._calculate_learned_timeout_factor(
            avg_ms, max_ms, stddev_ms, p99_ms, n, hang_count
        )
        
        # Calculate recommended timeout using learned factor
        recommended_timeout_ms = max(
            avg_ms * learned_factor,
            max_ms * 1.5,  # At least 1.5x the max we've seen succeed
            p99_ms * 2,    # At least 2x the 99th percentile
            10000          # Minimum 10 seconds
        )
        
        # Store updated stats
        conn.execute("""
            INSERT INTO action_stats 
            (action_name, size_bin, sample_count, avg_duration_ms, min_duration_ms,
             max_duration_ms, stddev_duration_ms, p50_duration_ms, p90_duration_ms,
             p99_duration_ms, learned_timeout_factor, recommended_timeout_ms,
             success_count, hang_count, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(action_name, size_bin) DO UPDATE SET
                sample_count = excluded.sample_count,
                avg_duration_ms = excluded.avg_duration_ms,
                min_duration_ms = excluded.min_duration_ms,
                max_duration_ms = excluded.max_duration_ms,
                stddev_duration_ms = excluded.stddev_duration_ms,
                p50_duration_ms = excluded.p50_duration_ms,
                p90_duration_ms = excluded.p90_duration_ms,
                p99_duration_ms = excluded.p99_duration_ms,
                learned_timeout_factor = excluded.learned_timeout_factor,
                recommended_timeout_ms = excluded.recommended_timeout_ms,
                success_count = excluded.success_count,
                hang_count = excluded.hang_count,
                last_updated = excluded.last_updated
        """, (
            action_name, size_bin, n, avg_ms, min_ms, max_ms, stddev_ms,
            p50_ms, p90_ms, p99_ms, learned_factor, recommended_timeout_ms,
            success_count, hang_count, datetime.now().isoformat()
        ))
        
        logger.debug(
            f"Updated stats for {action_name}/{size_bin}: "
            f"n={n}, avg={avg_ms:.0f}ms, max={max_ms:.0f}ms, "
            f"learned_factor={learned_factor:.1f}x"
        )
    
    def _calculate_learned_timeout_factor(
        self,
        avg_ms: float,
        max_ms: float,
        stddev_ms: float,
        p99_ms: float,
        sample_count: int,
        hang_count: int,
    ) -> float:
        """Calculate the appropriate timeout factor for an action.
        
        This learns from the variance in completion times:
        - Low variance (consistent timing) -> lower factor (e.g., 3-5x)
        - High variance (unpredictable) -> higher factor (e.g., 10-15x)
        - History of hangs -> even higher factor for safety
        
        Args:
            avg_ms: Average successful completion time
            max_ms: Maximum successful completion time
            stddev_ms: Standard deviation of completion times
            p99_ms: 99th percentile completion time
            sample_count: Number of successful samples
            hang_count: Number of times this action hung
            
        Returns:
            Learned timeout factor (multiplier for avg_ms)
        """
        if sample_count < MIN_SAMPLES_FOR_FACTOR_LEARNING:
            # Not enough data - use conservative default
            return INITIAL_TIMEOUT_FACTOR
        
        # Base factor from variance ratio
        # coefficient of variation (CV) = stddev / avg
        cv = stddev_ms / avg_ms if avg_ms > 0 else 1.0
        
        # Map CV to factor:
        # CV 0.0-0.3 (consistent) -> factor 3-5
        # CV 0.3-0.7 (moderate) -> factor 5-10
        # CV 0.7-1.5 (high variance) -> factor 10-15
        # CV > 1.5 (very unpredictable) -> factor 15-20
        if cv < 0.3:
            base_factor = 3.0 + cv * 6.67  # 3 to 5
        elif cv < 0.7:
            base_factor = 5.0 + (cv - 0.3) * 12.5  # 5 to 10
        elif cv < 1.5:
            base_factor = 10.0 + (cv - 0.7) * 6.25  # 10 to 15
        else:
            base_factor = 15.0 + min(cv - 1.5, 1.0) * 5  # 15 to 20
        
        # Adjust based on max/avg ratio (outlier sensitivity)
        max_ratio = max_ms / avg_ms if avg_ms > 0 else 1.0
        if max_ratio > base_factor * 0.8:
            # Our max observed is close to what we'd timeout at
            # Increase factor to give more headroom
            base_factor = max(base_factor, max_ratio * 1.3)
        
        # Adjust based on p99/avg ratio
        p99_ratio = p99_ms / avg_ms if avg_ms > 0 else 1.0
        if p99_ratio > base_factor * 0.5:
            base_factor = max(base_factor, p99_ratio * 2)
        
        # If we've had hangs, be more conservative
        if hang_count > 0:
            hang_penalty = 1 + (hang_count * 0.2)  # 20% increase per hang
            base_factor *= hang_penalty
        
        # Confidence adjustment - less confident = higher factor
        confidence = min(1.0, sample_count / 20)
        confidence_factor = 1 + (1 - confidence) * 0.5  # Up to 50% higher when uncertain
        base_factor *= confidence_factor
        
        # Clamp to bounds
        return max(MIN_TIMEOUT_FACTOR, min(MAX_TIMEOUT_FACTOR, base_factor))
    
    def get_actions_to_skip(self, mesh: MeshCharacteristics) -> List[str]:
        """Get list of actions that should be skipped for this mesh size."""
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
            
            # Successful vs hangs
            success_hang = conn.execute("""
                SELECT 
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as hangs
                FROM action_durations
            """).fetchone()
            
            # Unique actions tracked
            actions = conn.execute("SELECT COUNT(DISTINCT action_name) FROM action_durations").fetchone()[0]
            
            # Hang patterns
            hangs = conn.execute("SELECT COUNT(*) FROM hang_patterns WHERE should_skip = 1").fetchone()[0]
            
            # Per-action summary with learned factors
            action_summary = conn.execute("""
                SELECT action_name, size_bin,
                       sample_count, avg_duration_ms, max_duration_ms,
                       stddev_duration_ms, learned_timeout_factor,
                       success_count, hang_count
                FROM action_stats
                ORDER BY sample_count DESC
            """).fetchall()
            
            # Slow completions (valuable learning data)
            slow = conn.execute("""
                SELECT COUNT(*) as count, 
                       AVG(ratio_over_prediction) as avg_ratio,
                       MAX(ratio_over_prediction) as max_ratio
                FROM slow_completions
            """).fetchone()
        
        return {
            "total_observations": total,
            "successful_completions": success_hang["successes"] or 0,
            "hangs_detected": success_hang["hangs"] or 0,
            "actions_tracked": actions,
            "known_hang_patterns": hangs,
            "slow_completions": {
                "count": slow["count"] or 0,
                "avg_ratio_over_prediction": slow["avg_ratio"] or 0,
                "max_ratio_over_prediction": slow["max_ratio"] or 0,
            },
            "action_stats": [
                {
                    "action": row["action_name"],
                    "size_bin": row["size_bin"],
                    "samples": row["sample_count"],
                    "avg_ms": row["avg_duration_ms"],
                    "max_ms": row["max_duration_ms"],
                    "stddev_ms": row["stddev_duration_ms"],
                    "learned_factor": row["learned_timeout_factor"],
                    "hangs": row["hang_count"],
                }
                for row in action_summary
            ],
        }
    
    def set_alternative_action(self, action_name: str, size_bin: str, alternative: str) -> None:
        """Set an alternative action to use when an action hangs."""
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE hang_patterns SET alternative_action = ?
                WHERE action_name = ? AND size_bin = ?
            """, (alternative, action_name, size_bin))
    
    def clear_hang_pattern(self, action_name: str, size_bin: str) -> None:
        """Clear a hang pattern (e.g., after fixing a bug that caused hangs)."""
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE hang_patterns SET should_skip = 0, hang_count = 0
                WHERE action_name = ? AND size_bin = ?
            """, (action_name, size_bin))


# Global instance
_predictor_instance: Optional[ActionDurationPredictor] = None


def get_duration_predictor() -> ActionDurationPredictor:
    """Get or create the global duration predictor instance."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = ActionDurationPredictor()
    return _predictor_instance


def reset_duration_predictor() -> None:
    """Reset the global duration predictor instance."""
    global _predictor_instance
    _predictor_instance = None
