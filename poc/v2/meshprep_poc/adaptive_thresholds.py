# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Adaptive Thresholds Learning for MeshPrep.

This module learns optimal threshold values from repair outcomes:
- Volume/face loss limits that distinguish good vs bad repairs
- Decimation targets based on quality vs speed tradeoffs
- Profile boundary values (body count, face count buckets)
- Repair loop parameters (max attempts, timeouts)

Key principle: Track outcomes at different threshold values and
find the optimal point that maximizes success while minimizing
false positives/negatives.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import statistics

logger = logging.getLogger(__name__)

# Default threshold values (used when no learning data available)
DEFAULT_THRESHOLDS = {
    # Geometry loss thresholds
    "volume_loss_limit_pct": 30.0,
    "face_loss_limit_pct": 40.0,
    
    # Decimation settings
    "decimation_trigger_faces": 100000,
    "decimation_target_faces": 100000,
    
    # Profile boundary - body count
    "body_count_fragmented": 10,  # > this = fragmented
    "body_count_multi": 1,        # > this = multi-body
    
    # Profile boundary - face count buckets
    "face_count_tiny": 1000,
    "face_count_small": 10000,
    "face_count_medium": 100000,
    "face_count_large": 500000,
    
    # Repair loop settings
    "max_repair_attempts": 20,
    "repair_timeout_seconds": 120,
    
    # Escalation thresholds
    "escalation_volume_loss_pct": 30.0,
    "escalation_face_loss_pct": 40.0,
}

# Minimum samples before adjusting a threshold
MIN_SAMPLES_FOR_ADJUSTMENT = 20

# How much to adjust thresholds per iteration (conservative)
ADJUSTMENT_RATE = 0.1  # 10% adjustment per optimization


@dataclass
class ThresholdObservation:
    """An observation of a threshold being applied."""
    threshold_name: str
    threshold_value: float
    actual_value: float  # The actual measured value
    outcome_success: bool
    outcome_quality: float  # 0-1 quality score
    profile: str
    timestamp: str = ""


@dataclass
class ThresholdStats:
    """Statistics for a threshold."""
    name: str
    current_value: float
    default_value: float
    
    # Observation counts
    total_observations: int = 0
    observations_above: int = 0  # Where actual > threshold
    observations_below: int = 0  # Where actual <= threshold
    
    # Success rates
    success_rate_above: float = 0.0  # Success rate when actual > threshold
    success_rate_below: float = 0.0  # Success rate when actual <= threshold
    
    # Quality metrics
    avg_quality_above: float = 0.0
    avg_quality_below: float = 0.0
    
    # Optimal value (learned)
    optimal_value: Optional[float] = None
    confidence: float = 0.0  # 0-1 confidence in optimal value
    
    last_updated: str = ""


class AdaptiveThresholdsEngine:
    """
    Learns optimal threshold values from repair outcomes.
    
    Tracks observations of threshold applications and their outcomes,
    then computes optimal values that maximize success rate.
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """Initialize the adaptive thresholds engine."""
        self.data_path = data_path or Path(__file__).parent.parent.parent.parent / "learning_data"
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_path / "adaptive_thresholds.db"
        
        # In-memory cache of current thresholds
        self._thresholds: Dict[str, float] = dict(DEFAULT_THRESHOLDS)
        self._cache_valid = False
        
        self._init_database()
        self._load_thresholds()
    
    def _init_database(self) -> None:
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Current threshold values
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS thresholds (
                    name TEXT PRIMARY KEY,
                    current_value REAL,
                    default_value REAL,
                    optimal_value REAL,
                    confidence REAL DEFAULT 0,
                    last_updated TEXT
                )
            """)
            
            # Threshold observations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS threshold_observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    threshold_name TEXT,
                    threshold_value REAL,
                    actual_value REAL,
                    outcome_success INTEGER,
                    outcome_quality REAL,
                    profile TEXT,
                    created_at TEXT
                )
            """)
            
            # Threshold statistics by profile
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS threshold_profile_stats (
                    threshold_name TEXT,
                    profile TEXT,
                    observations INTEGER DEFAULT 0,
                    successes_above INTEGER DEFAULT 0,
                    successes_below INTEGER DEFAULT 0,
                    failures_above INTEGER DEFAULT 0,
                    failures_below INTEGER DEFAULT 0,
                    quality_sum_above REAL DEFAULT 0,
                    quality_sum_below REAL DEFAULT 0,
                    optimal_value REAL,
                    updated_at TEXT,
                    PRIMARY KEY (threshold_name, profile)
                )
            """)
            
            # Threshold adjustment history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS threshold_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    threshold_name TEXT,
                    old_value REAL,
                    new_value REAL,
                    reason TEXT,
                    observations_count INTEGER,
                    created_at TEXT
                )
            """)
            
            # Indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_obs_threshold ON threshold_observations(threshold_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_obs_profile ON threshold_observations(profile)")
            
            # Initialize default thresholds if not present
            for name, value in DEFAULT_THRESHOLDS.items():
                cursor.execute("""
                    INSERT OR IGNORE INTO thresholds (name, current_value, default_value, last_updated)
                    VALUES (?, ?, ?, ?)
                """, (name, value, value, datetime.now().isoformat()))
            
            conn.commit()
    
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
    
    def _load_thresholds(self) -> None:
        """Load current thresholds from database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, current_value FROM thresholds")
            for row in cursor.fetchall():
                self._thresholds[row["name"]] = row["current_value"]
        self._cache_valid = True
    
    def get(self, name: str, profile: Optional[str] = None) -> float:
        """Get a threshold value.
        
        Args:
            name: Threshold name
            profile: Optional profile for profile-specific thresholds
            
        Returns:
            The threshold value (learned or default)
        """
        if not self._cache_valid:
            self._load_thresholds()
        
        # First try profile-specific optimal value
        if profile:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT optimal_value FROM threshold_profile_stats
                    WHERE threshold_name = ? AND profile = ? AND optimal_value IS NOT NULL
                """, (name, profile))
                row = cursor.fetchone()
                if row and row["optimal_value"] is not None:
                    return row["optimal_value"]
        
        # Fall back to global threshold
        return self._thresholds.get(name, DEFAULT_THRESHOLDS.get(name, 0))
    
    def get_all(self) -> Dict[str, float]:
        """Get all current threshold values."""
        if not self._cache_valid:
            self._load_thresholds()
        return dict(self._thresholds)
    
    def record_observation(
        self,
        threshold_name: str,
        threshold_value: float,
        actual_value: float,
        success: bool,
        quality: float = 0.5,
        profile: str = "unknown",
    ) -> None:
        """Record an observation of a threshold being applied.
        
        Args:
            threshold_name: Name of the threshold
            threshold_value: The threshold value that was used
            actual_value: The actual measured value
            success: Whether the repair was successful
            quality: Quality score 0-1 (e.g., geometry preservation)
            profile: Model profile
        """
        now = datetime.now().isoformat()
        is_above = actual_value > threshold_value
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Record observation
            cursor.execute("""
                INSERT INTO threshold_observations
                (threshold_name, threshold_value, actual_value, outcome_success, outcome_quality, profile, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (threshold_name, threshold_value, actual_value, 1 if success else 0, quality, profile, now))
            
            # Update profile stats
            if is_above:
                if success:
                    cursor.execute("""
                        INSERT INTO threshold_profile_stats 
                        (threshold_name, profile, observations, successes_above, quality_sum_above, updated_at)
                        VALUES (?, ?, 1, 1, ?, ?)
                        ON CONFLICT(threshold_name, profile) DO UPDATE SET
                            observations = observations + 1,
                            successes_above = successes_above + 1,
                            quality_sum_above = quality_sum_above + ?,
                            updated_at = ?
                    """, (threshold_name, profile, quality, now, quality, now))
                else:
                    cursor.execute("""
                        INSERT INTO threshold_profile_stats 
                        (threshold_name, profile, observations, failures_above, updated_at)
                        VALUES (?, ?, 1, 1, ?)
                        ON CONFLICT(threshold_name, profile) DO UPDATE SET
                            observations = observations + 1,
                            failures_above = failures_above + 1,
                            updated_at = ?
                    """, (threshold_name, profile, now, now))
            else:
                if success:
                    cursor.execute("""
                        INSERT INTO threshold_profile_stats 
                        (threshold_name, profile, observations, successes_below, quality_sum_below, updated_at)
                        VALUES (?, ?, 1, 1, ?, ?)
                        ON CONFLICT(threshold_name, profile) DO UPDATE SET
                            observations = observations + 1,
                            successes_below = successes_below + 1,
                            quality_sum_below = quality_sum_below + ?,
                            updated_at = ?
                    """, (threshold_name, profile, quality, now, quality, now))
                else:
                    cursor.execute("""
                        INSERT INTO threshold_profile_stats 
                        (threshold_name, profile, observations, failures_below, updated_at)
                        VALUES (?, ?, 1, 1, ?)
                        ON CONFLICT(threshold_name, profile) DO UPDATE SET
                            observations = observations + 1,
                            failures_below = failures_below + 1,
                            updated_at = ?
                    """, (threshold_name, profile, now, now))
    
    def record_geometry_loss(
        self,
        volume_loss_pct: float,
        face_loss_pct: float,
        success: bool,
        quality: float,
        profile: str,
        escalated: bool = False,
    ) -> None:
        """Convenience method to record geometry loss observations."""
        volume_threshold = self.get("volume_loss_limit_pct", profile)
        face_threshold = self.get("face_loss_limit_pct", profile)
        
        self.record_observation(
            "volume_loss_limit_pct",
            volume_threshold,
            volume_loss_pct,
            success,
            quality,
            profile,
        )
        
        self.record_observation(
            "face_loss_limit_pct",
            face_threshold,
            face_loss_pct,
            success,
            quality,
            profile,
        )
        
        if escalated:
            self.record_observation(
                "escalation_volume_loss_pct",
                self.get("escalation_volume_loss_pct", profile),
                volume_loss_pct,
                success,
                quality,
                profile,
            )
            self.record_observation(
                "escalation_face_loss_pct",
                self.get("escalation_face_loss_pct", profile),
                face_loss_pct,
                success,
                quality,
                profile,
            )
    
    def record_decimation(
        self,
        original_faces: int,
        target_faces: int,
        result_faces: int,
        success: bool,
        quality: float,
        profile: str,
    ) -> None:
        """Record decimation operation outcome."""
        self.record_observation(
            "decimation_trigger_faces",
            self.get("decimation_trigger_faces", profile),
            float(original_faces),
            success,
            quality,
            profile,
        )
        
        self.record_observation(
            "decimation_target_faces",
            float(target_faces),
            float(result_faces),
            success,
            quality,
            profile,
        )
    
    def record_repair_attempts(
        self,
        attempts_used: int,
        duration_ms: float,
        success: bool,
        profile: str,
    ) -> None:
        """Record repair loop parameters."""
        max_attempts = self.get("max_repair_attempts", profile)
        timeout = self.get("repair_timeout_seconds", profile) * 1000  # Convert to ms
        
        # Quality based on efficiency
        attempt_efficiency = 1.0 - (attempts_used / max_attempts) if success else 0
        time_efficiency = 1.0 - min(duration_ms / timeout, 1.0) if success else 0
        quality = (attempt_efficiency + time_efficiency) / 2
        
        self.record_observation(
            "max_repair_attempts",
            max_attempts,
            float(attempts_used),
            success,
            quality,
            profile,
        )
        
        self.record_observation(
            "repair_timeout_seconds",
            timeout / 1000,
            duration_ms / 1000,
            success,
            quality,
            profile,
        )
    
    def optimize_thresholds(self, min_samples: int = MIN_SAMPLES_FOR_ADJUSTMENT) -> List[Dict[str, Any]]:
        """Optimize thresholds based on collected observations.
        
        Returns:
            List of adjustments made
        """
        adjustments = []
        now = datetime.now().isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            for threshold_name in DEFAULT_THRESHOLDS.keys():
                # Get aggregated stats across all profiles
                cursor.execute("""
                    SELECT 
                        SUM(observations) as total_obs,
                        SUM(successes_above) as succ_above,
                        SUM(successes_below) as succ_below,
                        SUM(failures_above) as fail_above,
                        SUM(failures_below) as fail_below,
                        SUM(quality_sum_above) as qual_above,
                        SUM(quality_sum_below) as qual_below
                    FROM threshold_profile_stats
                    WHERE threshold_name = ?
                """, (threshold_name,))
                
                row = cursor.fetchone()
                if not row or (row["total_obs"] or 0) < min_samples:
                    continue
                
                total_above = (row["succ_above"] or 0) + (row["fail_above"] or 0)
                total_below = (row["succ_below"] or 0) + (row["fail_below"] or 0)
                
                if total_above == 0 or total_below == 0:
                    continue
                
                success_rate_above = (row["succ_above"] or 0) / total_above
                success_rate_below = (row["succ_below"] or 0) / total_below
                
                current_value = self._thresholds.get(threshold_name, DEFAULT_THRESHOLDS[threshold_name])
                
                # Determine if threshold should be adjusted
                # If success rate is higher when actual < threshold, threshold might be too high
                # If success rate is higher when actual > threshold, threshold might be too low
                
                adjustment = None
                reason = ""
                
                if success_rate_below > success_rate_above + 0.1:
                    # Threshold is too high - lower it
                    adjustment = current_value * (1 - ADJUSTMENT_RATE)
                    reason = f"Higher success below threshold ({success_rate_below:.1%} vs {success_rate_above:.1%})"
                elif success_rate_above > success_rate_below + 0.1:
                    # Threshold is too low - raise it
                    adjustment = current_value * (1 + ADJUSTMENT_RATE)
                    reason = f"Higher success above threshold ({success_rate_above:.1%} vs {success_rate_below:.1%})"
                
                if adjustment is not None:
                    # Apply adjustment
                    cursor.execute("""
                        UPDATE thresholds 
                        SET current_value = ?, optimal_value = ?, confidence = ?, last_updated = ?
                        WHERE name = ?
                    """, (
                        adjustment,
                        adjustment,
                        min(row["total_obs"] / 100, 1.0),  # Confidence based on sample size
                        now,
                        threshold_name,
                    ))
                    
                    # Record history
                    cursor.execute("""
                        INSERT INTO threshold_history
                        (threshold_name, old_value, new_value, reason, observations_count, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (threshold_name, current_value, adjustment, reason, row["total_obs"], now))
                    
                    self._thresholds[threshold_name] = adjustment
                    
                    adjustments.append({
                        "threshold": threshold_name,
                        "old_value": current_value,
                        "new_value": adjustment,
                        "reason": reason,
                        "observations": row["total_obs"],
                    })
                    
                    logger.info(f"Adjusted {threshold_name}: {current_value:.2f} -> {adjustment:.2f} ({reason})")
        
        return adjustments
    
    def compute_optimal_for_profile(self, profile: str) -> Dict[str, Optional[float]]:
        """Compute optimal threshold values for a specific profile.
        
        Uses binary search to find the threshold value that maximizes
        success rate based on actual observations.
        """
        optimal = {}
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            for threshold_name in DEFAULT_THRESHOLDS.keys():
                # Get all observations for this threshold and profile
                cursor.execute("""
                    SELECT actual_value, outcome_success, outcome_quality
                    FROM threshold_observations
                    WHERE threshold_name = ? AND profile = ?
                    ORDER BY actual_value
                """, (threshold_name, profile))
                
                observations = cursor.fetchall()
                
                if len(observations) < MIN_SAMPLES_FOR_ADJUSTMENT:
                    optimal[threshold_name] = None
                    continue
                
                # Find optimal threshold using sliding window
                best_threshold = None
                best_score = -1
                
                values = [obs["actual_value"] for obs in observations]
                unique_values = sorted(set(values))
                
                for candidate in unique_values:
                    # Calculate success rate if we use this as threshold
                    below = [obs for obs in observations if obs["actual_value"] <= candidate]
                    above = [obs for obs in observations if obs["actual_value"] > candidate]
                    
                    if not below or not above:
                        continue
                    
                    success_below = sum(1 for obs in below if obs["outcome_success"]) / len(below)
                    success_above = sum(1 for obs in above if obs["outcome_success"]) / len(above)
                    
                    # Score: we want high success below AND appropriate behavior above
                    # For loss limits: high success below (good repairs stay under limit)
                    # For triggers: balanced success on both sides
                    score = success_below * 0.7 + (1 - success_above) * 0.3
                    
                    if score > best_score:
                        best_score = score
                        best_threshold = candidate
                
                optimal[threshold_name] = best_threshold
                
                # Store optimal value for profile
                if best_threshold is not None:
                    cursor.execute("""
                        UPDATE threshold_profile_stats
                        SET optimal_value = ?
                        WHERE threshold_name = ? AND profile = ?
                    """, (best_threshold, threshold_name, profile))
        
        return optimal
    
    def get_stats(self, threshold_name: str) -> ThresholdStats:
        """Get statistics for a threshold."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get current value
            cursor.execute("""
                SELECT current_value, default_value, optimal_value, confidence, last_updated
                FROM thresholds WHERE name = ?
            """, (threshold_name,))
            thresh_row = cursor.fetchone()
            
            if not thresh_row:
                return ThresholdStats(
                    name=threshold_name,
                    current_value=DEFAULT_THRESHOLDS.get(threshold_name, 0),
                    default_value=DEFAULT_THRESHOLDS.get(threshold_name, 0),
                )
            
            # Get aggregated stats
            cursor.execute("""
                SELECT 
                    SUM(observations) as total_obs,
                    SUM(successes_above) as succ_above,
                    SUM(successes_below) as succ_below,
                    SUM(failures_above) as fail_above,
                    SUM(failures_below) as fail_below,
                    SUM(quality_sum_above) as qual_above,
                    SUM(quality_sum_below) as qual_below
                FROM threshold_profile_stats
                WHERE threshold_name = ?
            """, (threshold_name,))
            stats_row = cursor.fetchone()
            
            total_above = ((stats_row["succ_above"] or 0) + (stats_row["fail_above"] or 0)) if stats_row else 0
            total_below = ((stats_row["succ_below"] or 0) + (stats_row["fail_below"] or 0)) if stats_row else 0
            
            return ThresholdStats(
                name=threshold_name,
                current_value=thresh_row["current_value"],
                default_value=thresh_row["default_value"],
                total_observations=stats_row["total_obs"] or 0 if stats_row else 0,
                observations_above=total_above,
                observations_below=total_below,
                success_rate_above=(stats_row["succ_above"] or 0) / total_above if total_above > 0 else 0,
                success_rate_below=(stats_row["succ_below"] or 0) / total_below if total_below > 0 else 0,
                avg_quality_above=(stats_row["qual_above"] or 0) / total_above if total_above > 0 else 0,
                avg_quality_below=(stats_row["qual_below"] or 0) / total_below if total_below > 0 else 0,
                optimal_value=thresh_row["optimal_value"],
                confidence=thresh_row["confidence"] or 0,
                last_updated=thresh_row["last_updated"] or "",
            )
    
    def get_all_stats(self) -> List[ThresholdStats]:
        """Get statistics for all thresholds."""
        return [self.get_stats(name) for name in DEFAULT_THRESHOLDS.keys()]
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """Get summary of adaptive thresholds status."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Total observations
            cursor.execute("SELECT COUNT(*) as count FROM threshold_observations")
            total_obs = cursor.fetchone()["count"]
            
            # Thresholds with adjustments
            cursor.execute("""
                SELECT COUNT(*) as count FROM thresholds
                WHERE current_value != default_value
            """)
            adjusted_count = cursor.fetchone()["count"]
            
            # Recent adjustments
            cursor.execute("""
                SELECT threshold_name, old_value, new_value, reason, created_at
                FROM threshold_history
                ORDER BY created_at DESC
                LIMIT 10
            """)
            recent_adjustments = [
                {
                    "threshold": row["threshold_name"],
                    "old": row["old_value"],
                    "new": row["new_value"],
                    "reason": row["reason"],
                    "date": row["created_at"],
                }
                for row in cursor.fetchall()
            ]
            
            # Current values vs defaults
            threshold_status = []
            for name, default in DEFAULT_THRESHOLDS.items():
                current = self._thresholds.get(name, default)
                threshold_status.append({
                    "name": name,
                    "current": current,
                    "default": default,
                    "changed": abs(current - default) > 0.01,
                    "change_pct": ((current - default) / default * 100) if default != 0 else 0,
                })
            
            return {
                "total_observations": total_obs,
                "thresholds_adjusted": adjusted_count,
                "total_thresholds": len(DEFAULT_THRESHOLDS),
                "recent_adjustments": recent_adjustments,
                "threshold_status": threshold_status,
            }
    
    def reset_to_defaults(self) -> None:
        """Reset all thresholds to default values."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            
            for name, value in DEFAULT_THRESHOLDS.items():
                cursor.execute("""
                    UPDATE thresholds
                    SET current_value = ?, optimal_value = NULL, confidence = 0, last_updated = ?
                    WHERE name = ?
                """, (value, now, name))
                
                self._thresholds[name] = value
            
            logger.info("Reset all thresholds to defaults")


# Global instance
_thresholds_engine: Optional[AdaptiveThresholdsEngine] = None


def get_adaptive_thresholds() -> AdaptiveThresholdsEngine:
    """Get or create the global adaptive thresholds engine instance."""
    global _thresholds_engine
    if _thresholds_engine is None:
        _thresholds_engine = AdaptiveThresholdsEngine()
    return _thresholds_engine
