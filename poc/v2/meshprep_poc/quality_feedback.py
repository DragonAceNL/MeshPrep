# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Visual Quality Feedback System for MeshPrep.

This module addresses a critical gap in mesh repair validation: a model can be
technically perfect (watertight, manifold, slicer-validated) but visually
unrecognizable compared to the original.

The system learns from user feedback to:
- Predict quality scores for new repairs based on similar past repairs
- Flag suspicious repairs that may need review before use
- Penalize pipelines that produce technically-valid but visually-poor results
- Learn profile-specific tolerances (organic models need more detail preservation)

Rating Scale:
- 5: Perfect - Indistinguishable from original
- 4: Good - Minor smoothing/simplification, fully usable
- 3: Acceptable - Noticeable changes but recognizable and printable
- 2: Poor - Significant detail loss, may be usable for some purposes
- 1: Rejected - Unrecognizable, destroyed, or fundamentally wrong

Binary Rating:
- Accept (1) = Rating >= 3
- Reject (0) = Rating < 3
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

# Database schema version for migrations
SCHEMA_VERSION = 1

# Minimum samples before making predictions
MIN_SAMPLES_FOR_PREDICTION = 10

# Minimum samples before retraining the model
MIN_SAMPLES_FOR_TRAINING = 50

# Quality thresholds
HIGH_QUALITY_THRESHOLD = 4  # Ratings >= 4 are "high quality"
ACCEPTABLE_THRESHOLD = 3    # Ratings >= 3 are "acceptable"


@dataclass
class QualityRating:
    """A user quality rating for a repair."""
    model_fingerprint: str
    model_filename: str
    rating_type: str  # 'binary' or 'gradational'
    rating_value: int  # 0-1 for binary, 1-5 for gradational
    user_comment: Optional[str] = None
    rated_by: Optional[str] = None
    rated_at: str = ""
    
    # Context
    pipeline_used: str = ""
    profile: str = ""
    repair_duration_ms: float = 0.0
    escalated: bool = False
    
    # Automatic metrics
    volume_change_pct: float = 0.0
    face_count_change_pct: float = 0.0
    hausdorff_distance: Optional[float] = None
    chamfer_distance: Optional[float] = None
    detail_preservation: Optional[float] = None
    silhouette_similarity: Optional[float] = None
    
    def normalized_score(self) -> float:
        """Get normalized score (0.0-1.0)."""
        if self.rating_type == 'binary':
            return float(self.rating_value)
        else:
            return (self.rating_value - 1) / 4.0  # Map 1-5 to 0-1


@dataclass
class QualityPrediction:
    """A predicted quality score."""
    score: float  # Predicted 1-5 score
    confidence: float  # 0-1 confidence in prediction
    based_on_samples: int
    warning: Optional[str] = None
    
    def is_acceptable(self) -> bool:
        """Check if predicted quality is acceptable (>= 3)."""
        return self.score >= ACCEPTABLE_THRESHOLD
    
    def is_high_quality(self) -> bool:
        """Check if predicted quality is high (>= 4)."""
        return self.score >= HIGH_QUALITY_THRESHOLD


@dataclass
class PipelineQualityStats:
    """Quality statistics for a pipeline + profile combination."""
    pipeline_name: str
    profile: str
    total_ratings: int = 0
    avg_rating: float = 0.0
    rating_stddev: float = 0.0
    ratings_5: int = 0
    ratings_4: int = 0
    ratings_3: int = 0
    ratings_2: int = 0
    ratings_1: int = 0
    
    @property
    def acceptance_rate(self) -> float:
        """Percentage of ratings >= 3."""
        if self.total_ratings == 0:
            return 0.0
        acceptable = self.ratings_5 + self.ratings_4 + self.ratings_3
        return acceptable / self.total_ratings
    
    @property
    def high_quality_rate(self) -> float:
        """Percentage of ratings >= 4."""
        if self.total_ratings == 0:
            return 0.0
        high_quality = self.ratings_5 + self.ratings_4
        return high_quality / self.total_ratings


def get_db_path(data_path: Optional[Path] = None) -> Path:
    """Get the path to the SQLite database."""
    if data_path is None:
        # Use learning_data directory at repo root
        data_path = Path(__file__).parent.parent.parent.parent / "learning_data"
    data_path.mkdir(parents=True, exist_ok=True)
    db_path = data_path / "quality_feedback.db"
    
    # Log if creating new database
    if not db_path.exists():
        logger.info(f"Creating new quality feedback database at {db_path}")
    
    return db_path


def init_database(db_path: Path) -> None:
    """Initialize the database schema."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Schema version tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_info (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        
        # Individual quality ratings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quality_ratings (
                id INTEGER PRIMARY KEY,
                model_fingerprint TEXT,
                model_filename TEXT,
                
                -- Rating data
                rating_type TEXT,
                rating_value INTEGER,
                normalized_score REAL,
                user_comment TEXT,
                rated_by TEXT,
                rated_at TEXT,
                
                -- Context
                pipeline_used TEXT,
                profile TEXT,
                repair_duration_ms REAL,
                escalated INTEGER,
                
                -- Automatic metrics
                volume_change_pct REAL,
                face_count_change_pct REAL,
                hausdorff_distance REAL,
                chamfer_distance REAL,
                detail_preservation REAL,
                silhouette_similarity REAL,
                
                -- For learning (full metrics snapshot)
                metrics_json TEXT
            )
        """)
        
        # Index for fast lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ratings_fingerprint 
            ON quality_ratings(model_fingerprint)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ratings_pipeline_profile 
            ON quality_ratings(pipeline_used, profile)
        """)
        
        # Pipeline quality statistics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_quality_stats (
                pipeline_name TEXT,
                profile TEXT,
                
                total_ratings INTEGER DEFAULT 0,
                avg_rating REAL DEFAULT 0,
                rating_stddev REAL DEFAULT 0,
                
                ratings_5 INTEGER DEFAULT 0,
                ratings_4 INTEGER DEFAULT 0,
                ratings_3 INTEGER DEFAULT 0,
                ratings_2 INTEGER DEFAULT 0,
                ratings_1 INTEGER DEFAULT 0,
                
                acceptance_rate REAL DEFAULT 0,
                high_quality_rate REAL DEFAULT 0,
                
                updated_at TEXT,
                
                PRIMARY KEY (pipeline_name, profile)
            )
        """)
        
        # Quality prediction model parameters
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quality_prediction_model (
                model_version INTEGER PRIMARY KEY,
                created_at TEXT,
                training_samples INTEGER,
                
                -- Model coefficients (JSON for flexibility)
                coefficients_json TEXT,
                
                -- Validation metrics
                mae REAL,
                rmse REAL,
                correlation REAL
            )
        """)
        
        # Profile-specific quality thresholds
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS profile_quality_thresholds (
                profile TEXT PRIMARY KEY,
                
                -- Learned acceptable ranges
                max_volume_loss_pct REAL,
                max_hausdorff_distance REAL,
                min_detail_preservation REAL,
                min_silhouette_similarity REAL,
                
                -- Based on user feedback
                samples INTEGER DEFAULT 0,
                confidence REAL DEFAULT 0,
                
                updated_at TEXT
            )
        """)
        
        # Store current schema version
        cursor.execute("""
            INSERT OR REPLACE INTO schema_info (key, value)
            VALUES ('version', ?)
        """, (str(SCHEMA_VERSION),))
        
        conn.commit()


class QualityFeedbackEngine:
    """
    Visual Quality Feedback Learning Engine.
    
    Learns from user quality ratings to predict repair quality and
    improve pipeline selection.
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize the quality feedback engine.
        
        Args:
            data_path: Path to learning data directory. If None, uses default.
        """
        self.db_path = get_db_path(data_path)
        init_database(self.db_path)
        
        # Cache for prediction model coefficients
        self._model_coefficients: Optional[Dict[str, float]] = None
        self._model_version: int = 0
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def record_rating(self, rating: QualityRating) -> None:
        """
        Record a user quality rating.
        
        Args:
            rating: The quality rating to record.
        """
        if not rating.rated_at:
            rating.rated_at = datetime.now().isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Prepare metrics JSON
            metrics = {
                "volume_change_pct": rating.volume_change_pct,
                "face_count_change_pct": rating.face_count_change_pct,
                "hausdorff_distance": rating.hausdorff_distance,
                "chamfer_distance": rating.chamfer_distance,
                "detail_preservation": rating.detail_preservation,
                "silhouette_similarity": rating.silhouette_similarity,
            }
            
            # Insert rating
            cursor.execute("""
                INSERT INTO quality_ratings (
                    model_fingerprint, model_filename, rating_type, rating_value,
                    normalized_score, user_comment, rated_by, rated_at,
                    pipeline_used, profile, repair_duration_ms, escalated,
                    volume_change_pct, face_count_change_pct, hausdorff_distance,
                    chamfer_distance, detail_preservation, silhouette_similarity,
                    metrics_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rating.model_fingerprint,
                rating.model_filename,
                rating.rating_type,
                rating.rating_value,
                rating.normalized_score(),
                rating.user_comment,
                rating.rated_by,
                rating.rated_at,
                rating.pipeline_used,
                rating.profile,
                rating.repair_duration_ms,
                1 if rating.escalated else 0,
                rating.volume_change_pct,
                rating.face_count_change_pct,
                rating.hausdorff_distance,
                rating.chamfer_distance,
                rating.detail_preservation,
                rating.silhouette_similarity,
                json.dumps(metrics),
            ))
            
            conn.commit()
            
            # Update pipeline stats
            self._update_pipeline_stats(
                conn, rating.pipeline_used, rating.profile, rating.rating_value
            )
            
            logger.info(
                f"Recorded quality rating: {rating.model_filename} = "
                f"{rating.rating_value}/5 (pipeline={rating.pipeline_used})"
            )
    
    def _update_pipeline_stats(
        self, conn: sqlite3.Connection, pipeline: str, profile: str, rating: int
    ) -> None:
        """Update pipeline quality statistics after a new rating."""
        cursor = conn.cursor()
        
        # Get all ratings for this pipeline + profile
        cursor.execute("""
            SELECT rating_value FROM quality_ratings
            WHERE pipeline_used = ? AND profile = ?
            AND rating_type = 'gradational'
        """, (pipeline, profile))
        
        ratings = [row['rating_value'] for row in cursor.fetchall()]
        
        if not ratings:
            return
        
        # Calculate statistics
        avg_rating = statistics.mean(ratings)
        stddev = statistics.stdev(ratings) if len(ratings) > 1 else 0.0
        
        ratings_5 = ratings.count(5)
        ratings_4 = ratings.count(4)
        ratings_3 = ratings.count(3)
        ratings_2 = ratings.count(2)
        ratings_1 = ratings.count(1)
        
        acceptable = ratings_5 + ratings_4 + ratings_3
        high_quality = ratings_5 + ratings_4
        
        acceptance_rate = acceptable / len(ratings) if ratings else 0.0
        high_quality_rate = high_quality / len(ratings) if ratings else 0.0
        
        # Update stats table
        cursor.execute("""
            INSERT OR REPLACE INTO pipeline_quality_stats (
                pipeline_name, profile, total_ratings, avg_rating, rating_stddev,
                ratings_5, ratings_4, ratings_3, ratings_2, ratings_1,
                acceptance_rate, high_quality_rate, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pipeline, profile, len(ratings), avg_rating, stddev,
            ratings_5, ratings_4, ratings_3, ratings_2, ratings_1,
            acceptance_rate, high_quality_rate, datetime.now().isoformat()
        ))
        
        conn.commit()
    
    def predict_quality(
        self,
        pipeline: str,
        profile: str,
        volume_change_pct: float = 0.0,
        face_count_change_pct: float = 0.0,
        hausdorff_distance: Optional[float] = None,
        detail_preservation: Optional[float] = None,
        silhouette_similarity: Optional[float] = None,
        escalated: bool = False,
    ) -> QualityPrediction:
        """
        Predict the quality score for a repair.
        
        Args:
            pipeline: Pipeline used for repair.
            profile: Model profile.
            volume_change_pct: Percentage volume change.
            face_count_change_pct: Percentage face count change.
            hausdorff_distance: Maximum surface deviation (optional).
            detail_preservation: Detail preservation score 0-1 (optional).
            silhouette_similarity: Silhouette similarity 0-1 (optional).
            escalated: Whether repair was escalated to Blender.
        
        Returns:
            QualityPrediction with score, confidence, and warnings.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get pipeline + profile stats
            cursor.execute("""
                SELECT * FROM pipeline_quality_stats
                WHERE pipeline_name = ? AND profile = ?
            """, (pipeline, profile))
            
            row = cursor.fetchone()
            
            if row and row['total_ratings'] >= MIN_SAMPLES_FOR_PREDICTION:
                # Use historical average as base prediction
                base_score = row['avg_rating']
                samples = row['total_ratings']
                confidence = min(samples / 100, 1.0)
                
                # Adjust based on geometry metrics
                adjustments = 0.0
                
                # Volume loss penalty
                if abs(volume_change_pct) > 20:
                    adjustments -= 0.5
                elif abs(volume_change_pct) > 10:
                    adjustments -= 0.2
                
                # Face count change (large reduction = potential detail loss)
                if face_count_change_pct < -50:
                    adjustments -= 0.3
                
                # Detail preservation bonus/penalty
                if detail_preservation is not None:
                    if detail_preservation < 0.5:
                        adjustments -= 0.5
                    elif detail_preservation > 0.85:
                        adjustments += 0.2
                
                # Silhouette similarity bonus/penalty
                if silhouette_similarity is not None:
                    if silhouette_similarity < 0.7:
                        adjustments -= 0.5
                    elif silhouette_similarity > 0.95:
                        adjustments += 0.2
                
                # Escalation often means aggressive repair
                if escalated:
                    adjustments -= 0.2
                
                predicted_score = max(1.0, min(5.0, base_score + adjustments))
                
                # Generate warnings
                warning = None
                if predicted_score < 2.5:
                    warning = "Predicted poor quality - consider different pipeline"
                elif predicted_score < 3.0:
                    warning = "Predicted borderline quality - manual review recommended"
                
                return QualityPrediction(
                    score=predicted_score,
                    confidence=confidence,
                    based_on_samples=samples,
                    warning=warning
                )
            
            else:
                # Fall back to heuristic prediction
                return self._heuristic_prediction(
                    volume_change_pct, face_count_change_pct,
                    hausdorff_distance, detail_preservation,
                    silhouette_similarity, escalated
                )
    
    def _heuristic_prediction(
        self,
        volume_change_pct: float,
        face_count_change_pct: float,
        hausdorff_distance: Optional[float],
        detail_preservation: Optional[float],
        silhouette_similarity: Optional[float],
        escalated: bool,
    ) -> QualityPrediction:
        """
        Make a heuristic quality prediction when insufficient historical data.
        """
        # Start with neutral score
        score = 3.5
        
        # Volume change
        if abs(volume_change_pct) < 5:
            score += 0.3
        elif abs(volume_change_pct) > 20:
            score -= 0.5
        elif abs(volume_change_pct) > 10:
            score -= 0.2
        
        # Face count change
        if face_count_change_pct > -20 and face_count_change_pct < 20:
            score += 0.2
        elif face_count_change_pct < -50:
            score -= 0.4
        
        # Detail preservation
        if detail_preservation is not None:
            if detail_preservation > 0.9:
                score += 0.4
            elif detail_preservation > 0.7:
                score += 0.1
            elif detail_preservation < 0.5:
                score -= 0.5
        
        # Silhouette similarity
        if silhouette_similarity is not None:
            if silhouette_similarity > 0.95:
                score += 0.3
            elif silhouette_similarity < 0.7:
                score -= 0.5
        
        # Escalation penalty
        if escalated:
            score -= 0.2
        
        score = max(1.0, min(5.0, score))
        
        warning = None
        if score < 3.0:
            warning = "Heuristic prediction suggests potential quality issues"
        
        return QualityPrediction(
            score=score,
            confidence=0.3,  # Low confidence for heuristic
            based_on_samples=0,
            warning=warning
        )
    
    def get_pipeline_quality_stats(
        self, pipeline: str, profile: Optional[str] = None
    ) -> List[PipelineQualityStats]:
        """
        Get quality statistics for a pipeline.
        
        Args:
            pipeline: Pipeline name.
            profile: Optional profile filter.
        
        Returns:
            List of PipelineQualityStats.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if profile:
                cursor.execute("""
                    SELECT * FROM pipeline_quality_stats
                    WHERE pipeline_name = ? AND profile = ?
                """, (pipeline, profile))
            else:
                cursor.execute("""
                    SELECT * FROM pipeline_quality_stats
                    WHERE pipeline_name = ?
                """, (pipeline,))
            
            results = []
            for row in cursor.fetchall():
                results.append(PipelineQualityStats(
                    pipeline_name=row['pipeline_name'],
                    profile=row['profile'],
                    total_ratings=row['total_ratings'],
                    avg_rating=row['avg_rating'],
                    rating_stddev=row['rating_stddev'],
                    ratings_5=row['ratings_5'],
                    ratings_4=row['ratings_4'],
                    ratings_3=row['ratings_3'],
                    ratings_2=row['ratings_2'],
                    ratings_1=row['ratings_1'],
                ))
            
            return results
    
    def get_all_pipeline_quality_stats(self) -> Dict[str, Dict[str, PipelineQualityStats]]:
        """
        Get quality statistics for all pipelines.
        
        Returns:
            Nested dict: {pipeline_name: {profile: stats}}
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM pipeline_quality_stats")
            
            results: Dict[str, Dict[str, PipelineQualityStats]] = {}
            for row in cursor.fetchall():
                pipeline = row['pipeline_name']
                profile = row['profile']
                
                if pipeline not in results:
                    results[pipeline] = {}
                
                results[pipeline][profile] = PipelineQualityStats(
                    pipeline_name=pipeline,
                    profile=profile,
                    total_ratings=row['total_ratings'],
                    avg_rating=row['avg_rating'],
                    rating_stddev=row['rating_stddev'],
                    ratings_5=row['ratings_5'],
                    ratings_4=row['ratings_4'],
                    ratings_3=row['ratings_3'],
                    ratings_2=row['ratings_2'],
                    ratings_1=row['ratings_1'],
                )
            
            return results
    
    def get_rating_by_fingerprint(self, fingerprint: str) -> Optional[QualityRating]:
        """
        Get the most recent rating for a model by fingerprint.
        
        Args:
            fingerprint: Model fingerprint (e.g., "MP:42f3729aa758")
        
        Returns:
            QualityRating if found, None otherwise.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM quality_ratings
                WHERE model_fingerprint = ?
                ORDER BY rated_at DESC
                LIMIT 1
            """, (fingerprint,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return QualityRating(
                model_fingerprint=row['model_fingerprint'],
                model_filename=row['model_filename'],
                rating_type=row['rating_type'],
                rating_value=row['rating_value'],
                user_comment=row['user_comment'],
                rated_by=row['rated_by'],
                rated_at=row['rated_at'],
                pipeline_used=row['pipeline_used'],
                profile=row['profile'],
                repair_duration_ms=row['repair_duration_ms'],
                escalated=bool(row['escalated']),
                volume_change_pct=row['volume_change_pct'],
                face_count_change_pct=row['face_count_change_pct'],
                hausdorff_distance=row['hausdorff_distance'],
                chamfer_distance=row['chamfer_distance'],
                detail_preservation=row['detail_preservation'],
                silhouette_similarity=row['silhouette_similarity'],
            )
    
    def get_unrated_repairs(
        self,
        limit: int = 100,
        sort_by: str = "predicted_quality"
    ) -> List[Dict[str, Any]]:
        """
        Get list of repairs that haven't been rated yet.
        
        This requires integration with the main repair database to find
        completed repairs without ratings. Returns empty list if no
        integration is available.
        
        Args:
            limit: Maximum number to return.
            sort_by: Sort order ("predicted_quality", "timestamp", "pipeline").
        
        Returns:
            List of repair records awaiting rating.
        """
        # TODO: Integrate with main repair tracking
        # This would query the model_results table in meshprep_learning.db
        # and find entries without corresponding quality_ratings
        logger.warning("get_unrated_repairs not yet integrated with repair tracking")
        return []
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for the quality feedback system.
        
        Returns:
            Dictionary with summary stats.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Total ratings
            cursor.execute("SELECT COUNT(*) as count FROM quality_ratings")
            total_ratings = cursor.fetchone()['count']
            
            # Ratings by score
            cursor.execute("""
                SELECT rating_value, COUNT(*) as count 
                FROM quality_ratings 
                WHERE rating_type = 'gradational'
                GROUP BY rating_value
            """)
            rating_distribution = {
                row['rating_value']: row['count'] 
                for row in cursor.fetchall()
            }
            
            # Average rating
            cursor.execute("""
                SELECT AVG(rating_value) as avg 
                FROM quality_ratings 
                WHERE rating_type = 'gradational'
            """)
            avg_row = cursor.fetchone()
            avg_rating = avg_row['avg'] if avg_row['avg'] else 0.0
            
            # Pipelines with quality data
            cursor.execute("""
                SELECT COUNT(DISTINCT pipeline_name || profile) as count 
                FROM pipeline_quality_stats
            """)
            pipeline_profile_count = cursor.fetchone()['count']
            
            # Acceptance rate overall
            cursor.execute("""
                SELECT 
                    SUM(CASE WHEN rating_value >= 3 THEN 1 ELSE 0 END) as acceptable,
                    COUNT(*) as total
                FROM quality_ratings
                WHERE rating_type = 'gradational'
            """)
            row = cursor.fetchone()
            acceptance_rate = row['acceptable'] / row['total'] if row['total'] > 0 else 0.0
            
            return {
                "total_ratings": total_ratings,
                "rating_distribution": rating_distribution,
                "avg_rating": round(avg_rating, 2),
                "pipeline_profile_combinations": pipeline_profile_count,
                "overall_acceptance_rate": round(acceptance_rate * 100, 1),
                "ready_for_prediction": total_ratings >= MIN_SAMPLES_FOR_PREDICTION,
            }
    
    def update_profile_quality_thresholds(self, profile: str) -> None:
        """
        Update learned quality thresholds for a profile based on ratings.
        
        Args:
            profile: Profile name to update thresholds for.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get high-quality ratings (>= 4) for this profile
            cursor.execute("""
                SELECT volume_change_pct, hausdorff_distance, 
                       detail_preservation, silhouette_similarity
                FROM quality_ratings
                WHERE profile = ? AND rating_value >= 4
                AND rating_type = 'gradational'
            """, (profile,))
            
            high_quality_rows = cursor.fetchall()
            
            if len(high_quality_rows) < 10:
                logger.debug(f"Not enough high-quality samples for {profile}")
                return
            
            # Calculate thresholds from high-quality repairs
            volume_losses = [abs(r['volume_change_pct']) for r in high_quality_rows]
            max_volume_loss = max(volume_losses) if volume_losses else 30.0
            
            hausdorff_vals = [
                r['hausdorff_distance'] for r in high_quality_rows 
                if r['hausdorff_distance'] is not None
            ]
            max_hausdorff = max(hausdorff_vals) if hausdorff_vals else None
            
            detail_vals = [
                r['detail_preservation'] for r in high_quality_rows
                if r['detail_preservation'] is not None
            ]
            min_detail = min(detail_vals) if detail_vals else None
            
            silhouette_vals = [
                r['silhouette_similarity'] for r in high_quality_rows
                if r['silhouette_similarity'] is not None
            ]
            min_silhouette = min(silhouette_vals) if silhouette_vals else None
            
            # Calculate confidence
            samples = len(high_quality_rows)
            confidence = min(samples / 50, 1.0)
            
            # Update thresholds
            cursor.execute("""
                INSERT OR REPLACE INTO profile_quality_thresholds (
                    profile, max_volume_loss_pct, max_hausdorff_distance,
                    min_detail_preservation, min_silhouette_similarity,
                    samples, confidence, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                profile, max_volume_loss, max_hausdorff,
                min_detail, min_silhouette,
                samples, confidence, datetime.now().isoformat()
            ))
            
            conn.commit()
            logger.info(f"Updated quality thresholds for profile: {profile}")
    
    def get_profile_quality_thresholds(self, profile: str) -> Optional[Dict[str, Any]]:
        """
        Get learned quality thresholds for a profile.
        
        Args:
            profile: Profile name.
        
        Returns:
            Dict with threshold values, or None if not learned.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM profile_quality_thresholds
                WHERE profile = ?
            """, (profile,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return {
                "profile": row['profile'],
                "max_volume_loss_pct": row['max_volume_loss_pct'],
                "max_hausdorff_distance": row['max_hausdorff_distance'],
                "min_detail_preservation": row['min_detail_preservation'],
                "min_silhouette_similarity": row['min_silhouette_similarity'],
                "samples": row['samples'],
                "confidence": row['confidence'],
            }
    
    def should_flag_for_review(
        self,
        pipeline: str,
        profile: str,
        volume_change_pct: float,
        face_count_change_pct: float,
        detail_preservation: Optional[float] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a repair should be flagged for manual quality review.
        
        Args:
            pipeline: Pipeline used.
            profile: Model profile.
            volume_change_pct: Volume change percentage.
            face_count_change_pct: Face count change percentage.
            detail_preservation: Detail preservation score (optional).
        
        Returns:
            Tuple of (should_flag, reason).
        """
        # Get prediction
        prediction = self.predict_quality(
            pipeline=pipeline,
            profile=profile,
            volume_change_pct=volume_change_pct,
            face_count_change_pct=face_count_change_pct,
            detail_preservation=detail_preservation,
        )
        
        # Check prediction
        if prediction.score < 2.0 and prediction.confidence > 0.6:
            return True, f"Predicted low quality ({prediction.score:.1f}/5)"
        
        if prediction.score < 3.0 and prediction.confidence > 0.8:
            return True, f"Predicted borderline quality ({prediction.score:.1f}/5)"
        
        # Check profile thresholds
        thresholds = self.get_profile_quality_thresholds(profile)
        if thresholds and thresholds['confidence'] > 0.5:
            if abs(volume_change_pct) > thresholds['max_volume_loss_pct'] * 1.5:
                return True, f"Volume loss ({abs(volume_change_pct):.1f}%) exceeds profile threshold"
            
            if detail_preservation is not None:
                min_detail = thresholds.get('min_detail_preservation')
                if min_detail and detail_preservation < min_detail * 0.8:
                    return True, f"Detail preservation ({detail_preservation:.2f}) below profile threshold"
        
        # Check pipeline history
        stats = self.get_pipeline_quality_stats(pipeline, profile)
        if stats and stats[0].total_ratings >= 20:
            if stats[0].acceptance_rate < 0.5:
                return True, f"Pipeline has low historical quality for this profile ({stats[0].acceptance_rate:.0%})"
        
        return False, None


# Singleton instance
_quality_engine: Optional[QualityFeedbackEngine] = None


def get_quality_engine(data_path: Optional[Path] = None) -> QualityFeedbackEngine:
    """
    Get the singleton quality feedback engine instance.
    
    Args:
        data_path: Optional path to learning data directory.
    
    Returns:
        QualityFeedbackEngine instance.
    """
    global _quality_engine
    if _quality_engine is None:
        _quality_engine = QualityFeedbackEngine(data_path)
    return _quality_engine


def reset_quality_engine() -> None:
    """Reset the singleton instance (mainly for testing)."""
    global _quality_engine
    _quality_engine = None


# =============================================================================
# Quality-Aware Pipeline Selection
# =============================================================================

def get_quality_adjusted_pipeline_order(
    pipelines: List[str],
    profile: str,
    issues: Optional[List[str]] = None,
) -> List[str]:
    """
    Reorder pipelines based on USER quality ratings.
    
    This function uses historical USER ratings (not auto-estimated) to rank
    pipelines. Pipelines that received higher quality ratings from users
    for similar profiles are ranked higher.
    
    The system learns from manual ratings via --rate command:
        python run_full_test.py --rate MP:abc123 --rating 4
    
    Args:
        pipelines: Original ordered list of pipelines to try
        profile: Mesh profile (e.g., "standard", "fragmented")
        issues: Optional list of issues detected in the mesh
    
    Returns:
        Reordered list of pipelines with quality-aware ranking
    """
    try:
        engine = get_quality_engine()
        all_stats = engine.get_all_pipeline_quality_stats()
        
        if not all_stats:
            return pipelines  # No user ratings yet
        
        # Calculate quality scores for each pipeline based on USER ratings
        pipeline_quality_scores: Dict[str, float] = {}
        
        for pipeline in pipelines:
            if pipeline not in all_stats:
                pipeline_quality_scores[pipeline] = 3.0  # Neutral score (no data)
                continue
            
            profile_stats = all_stats[pipeline]
            
            # Get stats for this profile if available
            if profile in profile_stats:
                stats = profile_stats[profile]
                if stats.total_ratings >= 3:  # Minimum samples for confidence
                    pipeline_quality_scores[pipeline] = stats.avg_rating
                else:
                    pipeline_quality_scores[pipeline] = 3.0  # Not enough user ratings
            elif profile_stats:
                # Use average across all profiles (user rated)
                total_ratings = sum(s.total_ratings for s in profile_stats.values())
                if total_ratings >= 3:
                    weighted_avg = sum(
                        s.avg_rating * s.total_ratings 
                        for s in profile_stats.values()
                    ) / total_ratings
                    pipeline_quality_scores[pipeline] = weighted_avg
                else:
                    pipeline_quality_scores[pipeline] = 3.0
            else:
                pipeline_quality_scores[pipeline] = 3.0
        
        # Sort pipelines by user quality score (highest first)
        # Preserve original order for ties
        sorted_pipelines = sorted(
            pipelines,
            key=lambda p: (pipeline_quality_scores.get(p, 3.0), -pipelines.index(p)),
            reverse=True
        )
        
        # Log if quality data influenced the order
        if any(s != 3.0 for s in pipeline_quality_scores.values()):
            logger.debug(
                f"Quality-adjusted pipeline order for {profile}: "
                f"{sorted_pipelines[:3]} (scores: {[pipeline_quality_scores.get(p, 3.0) for p in sorted_pipelines[:3]]})"
            )
        
        return sorted_pipelines
        
    except Exception as e:
        logger.debug(f"Quality-adjusted pipeline ordering failed: {e}")
        return pipelines  # Return original order on error


def should_warn_low_quality(
    pipeline: str,
    profile: str,
    volume_change_pct: float,
    face_change_pct: float,
) -> Tuple[bool, Optional[str]]:
    """
    Check if user ratings suggest this pipeline produces poor quality for this profile.
    
    This is used to warn users when a repair completes but historical user
    ratings suggest the result may have quality issues.
    
    Args:
        pipeline: Pipeline that was used
        profile: Mesh profile
        volume_change_pct: Volume change percentage
        face_change_pct: Face change percentage
    
    Returns:
        Tuple of (should_warn, warning_message)
    """
    try:
        engine = get_quality_engine()
        stats_list = engine.get_pipeline_quality_stats(pipeline, profile)
        
        if not stats_list or stats_list[0].total_ratings < 5:
            return False, None  # Not enough user ratings to warn
        
        stats = stats_list[0]
        
        # Warn if users have consistently rated this pipeline poorly
        if stats.avg_rating < 2.5:
            return True, (
                f"Users have rated '{pipeline}' repairs for '{profile}' models "
                f"poorly (avg {stats.avg_rating:.1f}/5 from {stats.total_ratings} ratings). "
                f"Consider manual review."
            )
        
        # Warn if acceptance rate is low
        if stats.acceptance_rate < 0.5:
            return True, (
                f"Only {stats.acceptance_rate:.0%} of users accepted '{pipeline}' repairs "
                f"for '{profile}' models. Consider manual review."
            )
        
        return False, None
        
    except Exception as e:
        logger.debug(f"Quality warning check failed: {e}")
        return False, None
