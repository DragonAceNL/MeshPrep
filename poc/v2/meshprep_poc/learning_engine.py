# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Self-Learning Repair Engine for MeshPrep (SQLite Version).

This engine learns from each repair attempt to improve future repairs:
- Tracks which pipelines work best for specific mesh characteristics
- Optimizes pipeline order based on success rates and speed
- Refines model profile detection based on outcomes
- Predicts the best repair strategy for new models

Uses SQLite for efficient storage and querying of large datasets.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

# Database schema version for migrations
SCHEMA_VERSION = 1


def get_db_path(data_path: Optional[Path] = None) -> Path:
    """Get the path to the SQLite database."""
    if data_path is None:
        data_path = Path(__file__).parent.parent.parent.parent / "learning_data"
    data_path.mkdir(parents=True, exist_ok=True)
    return data_path / "meshprep_learning.db"


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
        
        # Pipeline statistics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_stats (
                pipeline_name TEXT PRIMARY KEY,
                total_attempts INTEGER DEFAULT 0,
                successes INTEGER DEFAULT 0,
                failures INTEGER DEFAULT 0,
                total_duration_ms REAL DEFAULT 0,
                updated_at TEXT
            )
        """)
        
        # Pipeline success by issue type
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_issue_stats (
                pipeline_name TEXT,
                issue_type TEXT,
                successes INTEGER DEFAULT 0,
                failures INTEGER DEFAULT 0,
                PRIMARY KEY (pipeline_name, issue_type)
            )
        """)
        
        # Pipeline success by mesh characteristics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_mesh_stats (
                pipeline_name TEXT,
                characteristic TEXT,  -- e.g., 'body_count:1', 'face_count:medium'
                successes INTEGER DEFAULT 0,
                failures INTEGER DEFAULT 0,
                PRIMARY KEY (pipeline_name, characteristic)
            )
        """)
        
        # Issue pattern statistics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS issue_pattern_stats (
                pattern_key TEXT PRIMARY KEY,  -- Sorted, comma-separated issues
                total_models INTEGER DEFAULT 0,
                fixed_models INTEGER DEFAULT 0,
                failed_models INTEGER DEFAULT 0,
                best_pipeline TEXT,
                best_pipeline_success_rate REAL DEFAULT 0,
                updated_at TEXT
            )
        """)
        
        # Issue pattern to pipeline results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_pipeline_results (
                pattern_key TEXT,
                pipeline_name TEXT,
                successes INTEGER DEFAULT 0,
                failures INTEGER DEFAULT 0,
                PRIMARY KEY (pattern_key, pipeline_name)
            )
        """)
        
        # Mesh profile statistics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS profile_stats (
                profile_name TEXT PRIMARY KEY,
                total_models INTEGER DEFAULT 0,
                clean_models INTEGER DEFAULT 0,
                fixed_models INTEGER DEFAULT 0,
                failed_models INTEGER DEFAULT 0,
                escalated_models INTEGER DEFAULT 0,
                total_attempts_to_fix INTEGER DEFAULT 0,  -- For calculating average
                best_first_pipeline TEXT,
                updated_at TEXT
            )
        """)
        
        # Profile common issues
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS profile_issues (
                profile_name TEXT,
                issue_type TEXT,
                count INTEGER DEFAULT 0,
                PRIMARY KEY (profile_name, issue_type)
            )
        """)
        
        # Individual model results (for detailed analysis)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_results (
                model_id TEXT PRIMARY KEY,
                fingerprint TEXT,
                profile TEXT,
                issue_pattern TEXT,
                success INTEGER,
                escalated INTEGER,
                precheck_passed INTEGER,
                total_attempts INTEGER,
                winning_pipeline TEXT,
                total_duration_ms REAL,
                faces_before INTEGER,
                faces_after INTEGER,
                body_count INTEGER,
                created_at TEXT
            )
        """)
        
        # Individual repair attempts (for detailed pipeline analysis)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS repair_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT,
                attempt_number INTEGER,
                pipeline_name TEXT,
                success INTEGER,
                duration_ms REAL,
                created_at TEXT,
                FOREIGN KEY (model_id) REFERENCES model_results(model_id)
            )
        """)
        
        # Create indexes for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_profile ON model_results(profile)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_pattern ON model_results(issue_pattern)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_success ON model_results(success)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_attempts_pipeline ON repair_attempts(pipeline_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_attempts_model ON repair_attempts(model_id)")
        
        # Set schema version
        cursor.execute(
            "INSERT OR REPLACE INTO schema_info (key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION))
        )
        
        conn.commit()
        logger.info(f"Database initialized at {db_path}")


class LearningEngine:
    """
    Self-learning engine that improves repair strategies over time.
    
    Uses SQLite for efficient storage and querying of learning data.
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """Initialize the learning engine.
        
        Args:
            data_path: Path to store learning data. If None, uses default location.
        """
        self.data_path = data_path or Path(__file__).parent.parent.parent.parent / "learning_data"
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_path / "meshprep_learning.db"
        
        # Initialize database
        init_database(self.db_path)
        
        # Cache for frequently accessed data
        self._optimal_pipeline_order: Optional[List[str]] = None
        self._issue_to_pipeline_map: Optional[Dict[str, List[str]]] = None
        self._cache_valid = False
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper error handling."""
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
    
    def _get_body_count_bucket(self, body_count: int) -> str:
        """Categorize body count into buckets."""
        if body_count == 1:
            return "body_count:1"
        elif body_count <= 5:
            return "body_count:2-5"
        elif body_count <= 20:
            return "body_count:6-20"
        else:
            return "body_count:20+"
    
    def _get_face_count_bucket(self, face_count: int) -> str:
        """Categorize face count into buckets."""
        if face_count < 1000:
            return "face_count:tiny"
        elif face_count < 10000:
            return "face_count:small"
        elif face_count < 100000:
            return "face_count:medium"
        elif face_count < 500000:
            return "face_count:large"
        else:
            return "face_count:huge"
    
    def _get_issue_pattern_key(self, issues: List[str]) -> str:
        """Create a consistent key from a list of issues."""
        return ",".join(sorted(set(issues))) if issues else "none"
    
    def _detect_profile(self, diagnostics: Optional[Dict[str, Any]]) -> str:
        """Detect mesh profile from diagnostics."""
        if not diagnostics:
            return "unknown"
        
        body_count = diagnostics.get("body_count", 1)
        face_count = diagnostics.get("faces", 0)
        is_watertight = diagnostics.get("is_watertight", False)
        
        if body_count > 10:
            return "fragmented"
        elif body_count > 1:
            return "multi-body"
        elif not is_watertight and face_count < 1000:
            return "simple-broken"
        elif not is_watertight:
            return "complex-broken"
        elif face_count > 500000:
            return "high-poly"
        else:
            return "standard"
    
    def record_result(self, filter_data: Dict[str, Any]) -> None:
        """
        Record a repair result to learn from it.
        
        Args:
            filter_data: The complete filter data dict saved for each model
        """
        try:
            success = filter_data.get("success", False)
            escalated = filter_data.get("escalated_to_blender", False)
            precheck = filter_data.get("precheck", {})
            repair_attempts = filter_data.get("repair_attempts", {})
            diagnostics = filter_data.get("diagnostics", {})
            before_diag = diagnostics.get("before", {}) or {}
            
            # Extract key info
            model_id = filter_data.get("model_id", f"model_{datetime.now().timestamp()}")
            fingerprint = filter_data.get("model_fingerprint", "")
            profile = self._detect_profile(before_diag)
            
            precheck_info = precheck.get("mesh_info", {}) or {}
            issues = precheck_info.get("issues", []) if precheck_info else []
            issue_pattern = self._get_issue_pattern_key(issues)
            
            attempts = repair_attempts.get("attempts", [])
            total_attempts = repair_attempts.get("total_attempts", len(attempts))
            total_duration = repair_attempts.get("total_duration_ms", 0)
            
            # Find winning pipeline
            winning_pipeline = ""
            for attempt in attempts:
                if attempt.get("success", False):
                    winning_pipeline = attempt.get("pipeline_name", "")
                    break
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                # 1. Record model result
                cursor.execute("""
                    INSERT OR REPLACE INTO model_results 
                    (model_id, fingerprint, profile, issue_pattern, success, escalated,
                     precheck_passed, total_attempts, winning_pipeline, total_duration_ms,
                     faces_before, faces_after, body_count, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_id, fingerprint, profile, issue_pattern,
                    1 if success else 0,
                    1 if escalated else 0,
                    1 if precheck.get("passed", False) else 0,
                    total_attempts, winning_pipeline, total_duration,
                    before_diag.get("faces", 0),
                    diagnostics.get("after", {}).get("faces", 0) if diagnostics.get("after") else 0,
                    before_diag.get("body_count", 1),
                    now
                ))
                
                # 2. Record individual attempts
                for attempt in attempts:
                    cursor.execute("""
                        INSERT INTO repair_attempts 
                        (model_id, attempt_number, pipeline_name, success, duration_ms, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        model_id,
                        attempt.get("attempt_number", 0),
                        attempt.get("pipeline_name", "unknown"),
                        1 if attempt.get("success", False) else 0,
                        attempt.get("duration_ms", 0),
                        now
                    ))
                    
                    # 3. Update pipeline stats
                    pipeline_name = attempt.get("pipeline_name", "unknown")
                    attempt_success = attempt.get("success", False)
                    duration = attempt.get("duration_ms", 0)
                    
                    cursor.execute("""
                        INSERT INTO pipeline_stats (pipeline_name, total_attempts, successes, failures, total_duration_ms, updated_at)
                        VALUES (?, 1, ?, ?, ?, ?)
                        ON CONFLICT(pipeline_name) DO UPDATE SET
                            total_attempts = total_attempts + 1,
                            successes = successes + ?,
                            failures = failures + ?,
                            total_duration_ms = total_duration_ms + ?,
                            updated_at = ?
                    """, (
                        pipeline_name,
                        1 if attempt_success else 0,
                        0 if attempt_success else 1,
                        duration, now,
                        1 if attempt_success else 0,
                        0 if attempt_success else 1,
                        duration, now
                    ))
                    
                    # 4. Update pipeline-issue stats
                    for issue in issues:
                        cursor.execute("""
                            INSERT INTO pipeline_issue_stats (pipeline_name, issue_type, successes, failures)
                            VALUES (?, ?, ?, ?)
                            ON CONFLICT(pipeline_name, issue_type) DO UPDATE SET
                                successes = successes + ?,
                                failures = failures + ?
                        """, (
                            pipeline_name, issue,
                            1 if attempt_success else 0,
                            0 if attempt_success else 1,
                            1 if attempt_success else 0,
                            0 if attempt_success else 1
                        ))
                    
                    # 5. Update pipeline-mesh characteristic stats
                    for char in [self._get_body_count_bucket(before_diag.get("body_count", 1)),
                                 self._get_face_count_bucket(before_diag.get("faces", 0))]:
                        cursor.execute("""
                            INSERT INTO pipeline_mesh_stats (pipeline_name, characteristic, successes, failures)
                            VALUES (?, ?, ?, ?)
                            ON CONFLICT(pipeline_name, characteristic) DO UPDATE SET
                                successes = successes + ?,
                                failures = failures + ?
                        """, (
                            pipeline_name, char,
                            1 if attempt_success else 0,
                            0 if attempt_success else 1,
                            1 if attempt_success else 0,
                            0 if attempt_success else 1
                        ))
                
                # 6. Update issue pattern stats
                cursor.execute("""
                    INSERT INTO issue_pattern_stats (pattern_key, total_models, fixed_models, failed_models, updated_at)
                    VALUES (?, 1, ?, ?, ?)
                    ON CONFLICT(pattern_key) DO UPDATE SET
                        total_models = total_models + 1,
                        fixed_models = fixed_models + ?,
                        failed_models = failed_models + ?,
                        updated_at = ?
                """, (
                    issue_pattern,
                    1 if success else 0,
                    0 if success else 1,
                    now,
                    1 if success else 0,
                    0 if success else 1,
                    now
                ))
                
                # 7. Update pattern-pipeline results
                for attempt in attempts:
                    pipeline_name = attempt.get("pipeline_name", "unknown")
                    attempt_success = attempt.get("success", False)
                    
                    cursor.execute("""
                        INSERT INTO pattern_pipeline_results (pattern_key, pipeline_name, successes, failures)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(pattern_key, pipeline_name) DO UPDATE SET
                            successes = successes + ?,
                            failures = failures + ?
                    """, (
                        issue_pattern, pipeline_name,
                        1 if attempt_success else 0,
                        0 if attempt_success else 1,
                        1 if attempt_success else 0,
                        0 if attempt_success else 1
                    ))
                    
                    if attempt_success:
                        break  # Only record up to successful attempt
                
                # 8. Update profile stats
                cursor.execute("""
                    INSERT INTO profile_stats 
                    (profile_name, total_models, clean_models, fixed_models, failed_models, 
                     escalated_models, total_attempts_to_fix, updated_at)
                    VALUES (?, 1, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(profile_name) DO UPDATE SET
                        total_models = total_models + 1,
                        clean_models = clean_models + ?,
                        fixed_models = fixed_models + ?,
                        failed_models = failed_models + ?,
                        escalated_models = escalated_models + ?,
                        total_attempts_to_fix = total_attempts_to_fix + ?,
                        updated_at = ?
                """, (
                    profile,
                    1 if precheck.get("passed", False) else 0,
                    1 if success and not precheck.get("passed", False) else 0,
                    0 if success else 1,
                    1 if escalated else 0,
                    total_attempts if success else 0,
                    now,
                    1 if precheck.get("passed", False) else 0,
                    1 if success and not precheck.get("passed", False) else 0,
                    0 if success else 1,
                    1 if escalated else 0,
                    total_attempts if success else 0,
                    now
                ))
                
                # 9. Update profile issues
                for issue in issues:
                    cursor.execute("""
                        INSERT INTO profile_issues (profile_name, issue_type, count)
                        VALUES (?, ?, 1)
                        ON CONFLICT(profile_name, issue_type) DO UPDATE SET
                            count = count + 1
                    """, (profile, issue))
            
            # Invalidate cache
            self._cache_valid = False
            
        except Exception as e:
            logger.warning(f"Learning engine error: {e}")
    
    def _refresh_cache(self) -> None:
        """Refresh the cached optimal orderings."""
        if self._cache_valid:
            return
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get pipelines sorted by efficiency (success_rate * 1000 / avg_duration)
            cursor.execute("""
                SELECT pipeline_name,
                       CAST(successes AS REAL) / NULLIF(total_attempts, 0) as success_rate,
                       total_duration_ms / NULLIF(total_attempts, 0) as avg_duration
                FROM pipeline_stats
                WHERE total_attempts >= 5
                ORDER BY (CAST(successes AS REAL) / NULLIF(total_attempts, 0) * 1000) / 
                         NULLIF(total_duration_ms / NULLIF(total_attempts, 0), 0.001) DESC
            """)
            self._optimal_pipeline_order = [row["pipeline_name"] for row in cursor.fetchall()]
            
            # Build issue-to-pipeline map from patterns with good success rates
            self._issue_to_pipeline_map = {}
            cursor.execute("""
                SELECT DISTINCT p.pattern_key, r.pipeline_name,
                       CAST(r.successes AS REAL) / NULLIF(r.successes + r.failures, 0) as success_rate
                FROM issue_pattern_stats p
                JOIN pattern_pipeline_results r ON p.pattern_key = r.pattern_key
                WHERE r.successes + r.failures >= 3
                  AND CAST(r.successes AS REAL) / NULLIF(r.successes + r.failures, 0) > 0.5
                ORDER BY success_rate DESC
            """)
            
            for row in cursor.fetchall():
                issues = row["pattern_key"].split(",")
                for issue in issues:
                    if issue and issue != "none":
                        if issue not in self._issue_to_pipeline_map:
                            self._issue_to_pipeline_map[issue] = []
                        if row["pipeline_name"] not in self._issue_to_pipeline_map[issue]:
                            self._issue_to_pipeline_map[issue].append(row["pipeline_name"])
        
        self._cache_valid = True
        logger.debug(f"Refreshed cache: {len(self._optimal_pipeline_order)} pipelines ranked")
    
    def get_recommended_pipeline_order(
        self,
        issues: List[str],
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Get recommended pipeline order for a model based on its characteristics.
        
        Args:
            issues: List of issues detected in the model
            diagnostics: Optional mesh diagnostics
            
        Returns:
            Ordered list of pipeline names to try
        """
        self._refresh_cache()
        
        if not self._optimal_pipeline_order:
            return []
        
        # Start with issue-specific recommendations
        recommended = []
        if self._issue_to_pipeline_map:
            for issue in issues:
                if issue in self._issue_to_pipeline_map:
                    for pipeline in self._issue_to_pipeline_map[issue]:
                        if pipeline not in recommended:
                            recommended.append(pipeline)
        
        # Add remaining pipelines in optimal order
        for pipeline in self._optimal_pipeline_order:
            if pipeline not in recommended:
                recommended.append(pipeline)
        
        return recommended
    
    def get_success_probability(
        self,
        pipeline_name: str,
        issues: List[str],
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Estimate probability of success for a pipeline given the issues.
        
        Returns:
            Estimated probability of success (0.0 to 1.0)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get base success rate
            cursor.execute("""
                SELECT CAST(successes AS REAL) / NULLIF(total_attempts, 0) as success_rate,
                       total_attempts
                FROM pipeline_stats
                WHERE pipeline_name = ?
            """, (pipeline_name,))
            
            row = cursor.fetchone()
            if not row or row["total_attempts"] < 5:
                return 0.5  # Not enough data
            
            base_rate = row["success_rate"] or 0.5
            
            # Adjust based on issue-specific data
            if issues:
                cursor.execute("""
                    SELECT CAST(successes AS REAL) / NULLIF(successes + failures, 0) as rate
                    FROM pipeline_issue_stats
                    WHERE pipeline_name = ? AND issue_type IN ({})
                      AND successes + failures >= 3
                """.format(",".join("?" * len(issues))), [pipeline_name] + issues)
                
                issue_rates = [row["rate"] for row in cursor.fetchall() if row["rate"] is not None]
                if issue_rates:
                    base_rate = min(base_rate, min(issue_rates))
            
            # Adjust based on mesh characteristics
            if diagnostics:
                body_bucket = self._get_body_count_bucket(diagnostics.get("body_count", 1))
                cursor.execute("""
                    SELECT successes, failures
                    FROM pipeline_mesh_stats
                    WHERE pipeline_name = ? AND characteristic = ?
                """, (pipeline_name, body_bucket))
                
                row = cursor.fetchone()
                if row and "20+" in body_bucket:
                    base_rate *= 0.7  # Fragmented models are harder
                elif row and "6-20" in body_bucket:
                    base_rate *= 0.85
            
            return min(max(base_rate, 0.0), 1.0)
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """Get a summary of learning statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Total models
            cursor.execute("SELECT COUNT(*) as count FROM model_results")
            total_models = cursor.fetchone()["count"]
            
            # Pipeline count
            cursor.execute("SELECT COUNT(*) as count FROM pipeline_stats")
            pipeline_count = cursor.fetchone()["count"]
            
            # Issue pattern count
            cursor.execute("SELECT COUNT(*) as count FROM issue_pattern_stats")
            pattern_count = cursor.fetchone()["count"]
            
            # Profile count
            cursor.execute("SELECT COUNT(*) as count FROM profile_stats")
            profile_count = cursor.fetchone()["count"]
            
            # Top pipelines
            cursor.execute("""
                SELECT pipeline_name,
                       CAST(successes AS REAL) / NULLIF(total_attempts, 0) as success_rate,
                       total_duration_ms / NULLIF(total_attempts, 0) as avg_duration_ms,
                       total_attempts
                FROM pipeline_stats
                WHERE total_attempts >= 3
                ORDER BY success_rate DESC
                LIMIT 5
            """)
            top_pipelines = [
                {
                    "name": row["pipeline_name"],
                    "success_rate": row["success_rate"] or 0,
                    "avg_duration_ms": row["avg_duration_ms"] or 0,
                    "attempts": row["total_attempts"],
                }
                for row in cursor.fetchall()
            ]
            
            # Profile summary
            cursor.execute("""
                SELECT profile_name, total_models,
                       CAST(fixed_models AS REAL) / NULLIF(total_models, 0) as fix_rate
                FROM profile_stats
                ORDER BY total_models DESC
            """)
            profile_summary = {
                row["profile_name"]: {
                    "total": row["total_models"],
                    "fix_rate": row["fix_rate"] or 0,
                }
                for row in cursor.fetchall()
            }
            
            # Last update
            cursor.execute("""
                SELECT MAX(created_at) as last_update FROM model_results
            """)
            last_update = cursor.fetchone()["last_update"]
            
            # Refresh cache and get optimal order
            self._refresh_cache()
            
            return {
                "version": "2.0.0-sqlite",
                "total_models_processed": total_models,
                "last_updated": last_update,
                "pipelines_tracked": pipeline_count,
                "issue_patterns_tracked": pattern_count,
                "profiles_tracked": profile_count,
                "optimal_pipeline_order": (self._optimal_pipeline_order or [])[:5],
                "top_pipelines": top_pipelines,
                "profile_summary": profile_summary,
            }
    
    def get_detailed_pipeline_stats(self) -> List[Dict[str, Any]]:
        """Get detailed statistics for all pipelines."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT pipeline_name,
                       total_attempts,
                       successes,
                       failures,
                       CAST(successes AS REAL) / NULLIF(total_attempts, 0) as success_rate,
                       total_duration_ms / NULLIF(total_attempts, 0) as avg_duration_ms
                FROM pipeline_stats
                ORDER BY success_rate DESC, total_attempts DESC
            """)
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_issue_pattern_analysis(self, min_samples: int = 5) -> List[Dict[str, Any]]:
        """Get analysis of issue patterns and their best pipelines."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT p.pattern_key,
                       p.total_models,
                       p.fixed_models,
                       CAST(p.fixed_models AS REAL) / NULLIF(p.total_models, 0) as fix_rate,
                       (SELECT pipeline_name 
                        FROM pattern_pipeline_results r 
                        WHERE r.pattern_key = p.pattern_key 
                        ORDER BY CAST(r.successes AS REAL) / NULLIF(r.successes + r.failures, 0) DESC 
                        LIMIT 1) as best_pipeline
                FROM issue_pattern_stats p
                WHERE p.total_models >= ?
                ORDER BY p.total_models DESC
            """, (min_samples,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def query_similar_models(
        self,
        profile: str = None,
        issue_pattern: str = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Query for similar models that were successfully fixed."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            conditions = ["success = 1"]
            params = []
            
            if profile:
                conditions.append("profile = ?")
                params.append(profile)
            
            if issue_pattern:
                conditions.append("issue_pattern = ?")
                params.append(issue_pattern)
            
            query = f"""
                SELECT model_id, fingerprint, profile, issue_pattern,
                       winning_pipeline, total_attempts, total_duration_ms
                FROM model_results
                WHERE {" AND ".join(conditions)}
                ORDER BY created_at DESC
                LIMIT ?
            """
            params.append(limit)
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def export_to_json(self, output_path: Path = None) -> Path:
        """Export all learning data to JSON for backup/sharing."""
        if output_path is None:
            output_path = self.data_path / "learning_export.json"
        
        data = {
            "version": "2.0.0-sqlite",
            "exported_at": datetime.now().isoformat(),
            "stats_summary": self.get_stats_summary(),
            "pipeline_stats": self.get_detailed_pipeline_stats(),
            "issue_patterns": self.get_issue_pattern_analysis(min_samples=1),
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported learning data to {output_path}")
        return output_path
    
    def force_save(self) -> None:
        """Force save - with SQLite this is a no-op since all writes are immediate."""
        pass  # SQLite commits are automatic


# Global instance
_learning_engine: Optional[LearningEngine] = None


def get_learning_engine(data_path: Optional[Path] = None) -> LearningEngine:
    """Get or create the global learning engine instance."""
    global _learning_engine
    if _learning_engine is None:
        _learning_engine = LearningEngine(data_path)
    return _learning_engine
