# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Automatic Profile Discovery for MeshPrep.

This module discovers new mesh profiles by analyzing repair outcomes and
clustering meshes with similar characteristics. It helps the learning
engine adapt to new types of meshes that don't fit existing profiles.

Key features:
- Clusters meshes by normalized characteristics (face count, body count, etc.)
- Identifies profiles with poor success rates that need splitting
- Discovers new profiles from failed repairs
- Suggests optimal pipelines for each discovered profile
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import math

logger = logging.getLogger(__name__)

# Minimum samples needed before discovering profiles
MIN_SAMPLES_FOR_DISCOVERY = 50

# Minimum samples in a cluster to create a profile
MIN_CLUSTER_SIZE = 10

# Success rate threshold - profiles below this may need splitting
LOW_SUCCESS_THRESHOLD = 0.5

# High variance threshold - profiles with high variance may need splitting
HIGH_VARIANCE_THRESHOLD = 0.3


@dataclass
class MeshCharacteristics:
    """Normalized mesh characteristics for clustering."""
    face_count_bucket: str  # tiny, small, medium, large, huge
    body_count_bucket: str  # 1, 2-5, 6-20, 20+
    is_watertight: bool
    has_degenerate_faces: bool
    aspect_ratio_bucket: str  # flat, normal, elongated
    issue_signature: str  # Sorted issues as string
    
    def to_key(self) -> str:
        """Create a unique key for this characteristic combination."""
        return f"{self.face_count_bucket}|{self.body_count_bucket}|{self.is_watertight}|{self.has_degenerate_faces}|{self.aspect_ratio_bucket}|{self.issue_signature}"


@dataclass
class DiscoveredProfile:
    """A profile discovered through clustering."""
    name: str
    description: str
    characteristics: Dict[str, Any]
    
    # Statistics
    total_models: int = 0
    successful_repairs: int = 0
    failed_repairs: int = 0
    avg_attempts_to_fix: float = 0
    
    # Best strategies
    best_pipeline: Optional[str] = None
    best_pipeline_success_rate: float = 0
    recommended_pipelines: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: str = ""
    updated_at: str = ""
    is_promoted: bool = False  # True if manually promoted to standard profile
    
    @property
    def success_rate(self) -> float:
        total = self.total_models
        return self.successful_repairs / total if total > 0 else 0


class ProfileDiscoveryEngine:
    """
    Discovers new mesh profiles by analyzing repair outcomes.
    
    Uses clustering on mesh characteristics to find groups of similar
    meshes that may benefit from specialized repair strategies.
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """Initialize the profile discovery engine."""
        if data_path is None:
            # Resolve the path first to handle relative imports correctly
            data_path = Path(__file__).resolve().parent.parent.parent.parent / "learning_data"
        self.data_path = data_path
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_path / "profile_discovery.db"
        
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Discovered profiles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS discovered_profiles (
                    profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE,
                    description TEXT,
                    characteristics_json TEXT,
                    
                    total_models INTEGER DEFAULT 0,
                    successful_repairs INTEGER DEFAULT 0,
                    failed_repairs INTEGER DEFAULT 0,
                    avg_attempts_to_fix REAL DEFAULT 0,
                    
                    best_pipeline TEXT,
                    best_pipeline_success_rate REAL DEFAULT 0,
                    recommended_pipelines_json TEXT,
                    
                    created_at TEXT,
                    updated_at TEXT,
                    is_promoted INTEGER DEFAULT 0,
                    is_active INTEGER DEFAULT 1
                )
            """)
            
            # Profile membership - which models belong to which discovered profile
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS profile_membership (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT,
                    profile_id INTEGER,
                    characteristics_key TEXT,
                    success INTEGER,
                    pipeline_used TEXT,
                    attempts INTEGER,
                    created_at TEXT,
                    FOREIGN KEY (profile_id) REFERENCES discovered_profiles(profile_id)
                )
            """)
            
            # Characteristic clusters - raw clustering data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS characteristic_clusters (
                    cluster_key TEXT PRIMARY KEY,
                    
                    face_count_bucket TEXT,
                    body_count_bucket TEXT,
                    is_watertight INTEGER,
                    has_degenerate_faces INTEGER,
                    aspect_ratio_bucket TEXT,
                    issue_signature TEXT,
                    
                    total_models INTEGER DEFAULT 0,
                    successful_models INTEGER DEFAULT 0,
                    failed_models INTEGER DEFAULT 0,
                    
                    assigned_profile_id INTEGER,
                    
                    updated_at TEXT,
                    FOREIGN KEY (assigned_profile_id) REFERENCES discovered_profiles(profile_id)
                )
            """)
            
            # Pipeline effectiveness per cluster
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cluster_pipeline_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cluster_key TEXT,
                    pipeline_name TEXT,
                    attempts INTEGER DEFAULT 0,
                    successes INTEGER DEFAULT 0,
                    total_duration_ms REAL DEFAULT 0,
                    updated_at TEXT,
                    UNIQUE(cluster_key, pipeline_name)
                )
            """)
            
            # Discovery history - when profiles were created/merged
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS discovery_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT,  -- 'created', 'merged', 'split', 'promoted', 'deactivated'
                    profile_name TEXT,
                    details_json TEXT,
                    created_at TEXT
                )
            """)
            
            # Indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_membership_model ON profile_membership(model_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_membership_profile ON profile_membership(profile_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_clusters_profile ON characteristic_clusters(assigned_profile_id)")
            
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
    
    def _get_face_count_bucket(self, face_count: int) -> str:
        """Categorize face count into buckets."""
        if face_count < 1000:
            return "tiny"
        elif face_count < 10000:
            return "small"
        elif face_count < 100000:
            return "medium"
        elif face_count < 500000:
            return "large"
        else:
            return "huge"
    
    def _get_body_count_bucket(self, body_count: int) -> str:
        """Categorize body count into buckets."""
        if body_count == 1:
            return "1"
        elif body_count <= 5:
            return "2-5"
        elif body_count <= 20:
            return "6-20"
        else:
            return "20+"
    
    def _get_aspect_ratio_bucket(self, extents: List[float]) -> str:
        """Categorize aspect ratio into buckets."""
        if not extents or len(extents) < 3:
            return "unknown"
        
        max_ext = max(extents)
        min_ext = min(e for e in extents if e > 0) if any(e > 0 for e in extents) else 1
        ratio = max_ext / min_ext if min_ext > 0 else 1
        
        if ratio < 2:
            return "compact"
        elif ratio < 5:
            return "normal"
        elif ratio < 10:
            return "elongated"
        else:
            return "extreme"
    
    def _get_issue_signature(self, issues: List[str]) -> str:
        """Create a signature from issues."""
        if not issues:
            return "none"
        # Sort and join first 3 most common issue types
        sorted_issues = sorted(set(issues))[:3]
        return ",".join(sorted_issues) if sorted_issues else "none"
    
    def extract_characteristics(self, diagnostics: Dict[str, Any], issues: List[str] = None) -> MeshCharacteristics:
        """Extract normalized characteristics from mesh diagnostics."""
        face_count = diagnostics.get("faces", 0)
        body_count = diagnostics.get("body_count", 1)
        is_watertight = diagnostics.get("is_watertight", False)
        degenerate = diagnostics.get("degenerate_faces", 0) > 0
        extents = diagnostics.get("extents", [1, 1, 1])
        
        return MeshCharacteristics(
            face_count_bucket=self._get_face_count_bucket(face_count),
            body_count_bucket=self._get_body_count_bucket(body_count),
            is_watertight=is_watertight,
            has_degenerate_faces=degenerate,
            aspect_ratio_bucket=self._get_aspect_ratio_bucket(extents),
            issue_signature=self._get_issue_signature(issues or []),
        )
    
    def record_model(
        self,
        model_id: str,
        diagnostics: Dict[str, Any],
        issues: List[str],
        success: bool,
        pipeline_used: str,
        attempts: int,
    ) -> None:
        """Record a model's characteristics and repair outcome."""
        chars = self.extract_characteristics(diagnostics, issues)
        cluster_key = chars.to_key()
        now = datetime.now().isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Update or insert cluster
            cursor.execute("""
                INSERT INTO characteristic_clusters (
                    cluster_key, face_count_bucket, body_count_bucket, is_watertight,
                    has_degenerate_faces, aspect_ratio_bucket, issue_signature,
                    total_models, successful_models, failed_models, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                ON CONFLICT(cluster_key) DO UPDATE SET
                    total_models = total_models + 1,
                    successful_models = successful_models + ?,
                    failed_models = failed_models + ?,
                    updated_at = ?
            """, (
                cluster_key, chars.face_count_bucket, chars.body_count_bucket,
                1 if chars.is_watertight else 0, 1 if chars.has_degenerate_faces else 0,
                chars.aspect_ratio_bucket, chars.issue_signature,
                1 if success else 0, 0 if success else 1, now,
                1 if success else 0, 0 if success else 1, now
            ))
            
            # Update pipeline stats for this cluster
            cursor.execute("""
                INSERT INTO cluster_pipeline_stats (
                    cluster_key, pipeline_name, attempts, successes, updated_at
                ) VALUES (?, ?, 1, ?, ?)
                ON CONFLICT(cluster_key, pipeline_name) DO UPDATE SET
                    attempts = attempts + 1,
                    successes = successes + ?,
                    updated_at = ?
            """, (
                cluster_key, pipeline_used, 1 if success else 0, now,
                1 if success else 0, now
            ))
            
            # Get assigned profile for this cluster
            cursor.execute(
                "SELECT assigned_profile_id FROM characteristic_clusters WHERE cluster_key = ?",
                (cluster_key,)
            )
            row = cursor.fetchone()
            profile_id = row["assigned_profile_id"] if row and row["assigned_profile_id"] else None
            
            # Record membership
            cursor.execute("""
                INSERT INTO profile_membership (
                    model_id, profile_id, characteristics_key, success, pipeline_used, attempts, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                model_id, profile_id, cluster_key, 1 if success else 0, pipeline_used, attempts, now
            ))
    
    def run_discovery(self, min_samples: int = MIN_SAMPLES_FOR_DISCOVERY) -> List[DiscoveredProfile]:
        """
        Run profile discovery on accumulated data.
        
        Returns list of newly discovered or updated profiles.
        """
        discovered = []
        now = datetime.now().isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get total sample count
            cursor.execute("SELECT SUM(total_models) as total FROM characteristic_clusters")
            total = cursor.fetchone()["total"] or 0
            
            if total < min_samples:
                logger.info(f"Not enough samples for discovery ({total} < {min_samples})")
                return []
            
            # Find clusters that aren't assigned to any profile yet
            cursor.execute("""
                SELECT cluster_key, face_count_bucket, body_count_bucket, is_watertight,
                       has_degenerate_faces, aspect_ratio_bucket, issue_signature,
                       total_models, successful_models, failed_models
                FROM characteristic_clusters
                WHERE assigned_profile_id IS NULL
                  AND total_models >= ?
                ORDER BY total_models DESC
            """, (MIN_CLUSTER_SIZE,))
            
            unassigned_clusters = cursor.fetchall()
            
            # Group similar clusters into profiles
            cluster_groups = self._group_similar_clusters(unassigned_clusters)
            
            for group_key, clusters in cluster_groups.items():
                # Calculate aggregate stats
                total_models = sum(c["total_models"] for c in clusters)
                successful = sum(c["successful_models"] for c in clusters)
                failed = sum(c["failed_models"] for c in clusters)
                
                if total_models < MIN_CLUSTER_SIZE:
                    continue
                
                success_rate = successful / total_models if total_models > 0 else 0
                
                # Generate profile name and description
                sample_cluster = clusters[0]
                profile_name = self._generate_profile_name(sample_cluster, len(discovered))
                description = self._generate_profile_description(clusters, success_rate)
                
                # Get best pipeline for this group
                cluster_keys = [c["cluster_key"] for c in clusters]
                best_pipeline, best_rate, recommended = self._get_best_pipelines_for_clusters(
                    cursor, cluster_keys
                )
                
                # Create the profile
                characteristics = {
                    "face_count_buckets": list(set(c["face_count_bucket"] for c in clusters)),
                    "body_count_buckets": list(set(c["body_count_bucket"] for c in clusters)),
                    "is_watertight": any(c["is_watertight"] for c in clusters),
                    "has_degenerate_faces": any(c["has_degenerate_faces"] for c in clusters),
                    "aspect_ratio_buckets": list(set(c["aspect_ratio_bucket"] for c in clusters)),
                    "issue_signatures": list(set(c["issue_signature"] for c in clusters)),
                    "cluster_keys": cluster_keys,
                }
                
                # Insert profile
                cursor.execute("""
                    INSERT INTO discovered_profiles (
                        name, description, characteristics_json,
                        total_models, successful_repairs, failed_repairs,
                        best_pipeline, best_pipeline_success_rate, recommended_pipelines_json,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    profile_name, description, json.dumps(characteristics),
                    total_models, successful, failed,
                    best_pipeline, best_rate, json.dumps(recommended),
                    now, now
                ))
                
                profile_id = cursor.lastrowid
                
                # Assign clusters to this profile
                for cluster_key in cluster_keys:
                    cursor.execute("""
                        UPDATE characteristic_clusters 
                        SET assigned_profile_id = ? 
                        WHERE cluster_key = ?
                    """, (profile_id, cluster_key))
                
                # Update membership records
                cursor.execute("""
                    UPDATE profile_membership
                    SET profile_id = ?
                    WHERE characteristics_key IN ({})
                """.format(",".join("?" * len(cluster_keys))),
                    [profile_id] + cluster_keys
                )
                
                # Record discovery history
                cursor.execute("""
                    INSERT INTO discovery_history (action, profile_name, details_json, created_at)
                    VALUES ('created', ?, ?, ?)
                """, (profile_name, json.dumps({"clusters": len(clusters), "models": total_models}), now))
                
                profile = DiscoveredProfile(
                    name=profile_name,
                    description=description,
                    characteristics=characteristics,
                    total_models=total_models,
                    successful_repairs=successful,
                    failed_repairs=failed,
                    best_pipeline=best_pipeline,
                    best_pipeline_success_rate=best_rate,
                    recommended_pipelines=recommended,
                    created_at=now,
                    updated_at=now,
                )
                discovered.append(profile)
                
                logger.info(f"Discovered profile: {profile_name} ({total_models} models, {success_rate*100:.1f}% success)")
            
            # Also check existing profiles that may need splitting
            self._check_for_profile_splits(cursor, now)
        
        return discovered
    
    def _group_similar_clusters(self, clusters: List[sqlite3.Row]) -> Dict[str, List[sqlite3.Row]]:
        """Group similar clusters together for profile creation."""
        groups = {}
        
        for cluster in clusters:
            # Create a simplified grouping key (less granular than cluster_key)
            # Group by: body count bucket + watertight status + primary issue
            group_key = f"{cluster['body_count_bucket']}|{cluster['is_watertight']}|{cluster['issue_signature'].split(',')[0] if cluster['issue_signature'] != 'none' else 'clean'}"
            
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(cluster)
        
        return groups
    
    def _generate_profile_name(self, sample_cluster: sqlite3.Row, index: int) -> str:
        """Generate a descriptive profile name."""
        parts = []
        
        # Body count
        body = sample_cluster["body_count_bucket"]
        if body == "1":
            parts.append("single")
        elif body == "2-5":
            parts.append("multi")
        elif body in ("6-20", "20+"):
            parts.append("fragmented")
        
        # Watertight status
        if not sample_cluster["is_watertight"]:
            parts.append("broken")
        
        # Face count
        faces = sample_cluster["face_count_bucket"]
        if faces in ("large", "huge"):
            parts.append("highpoly")
        elif faces == "tiny":
            parts.append("lowpoly")
        
        # Issue type
        issues = sample_cluster["issue_signature"]
        if issues != "none":
            primary_issue = issues.split(",")[0].replace(" ", "-").lower()
            if primary_issue not in parts:
                parts.append(primary_issue)
        
        if not parts:
            parts.append("standard")
        
        name = f"discovered-{'-'.join(parts)}"
        
        # Add index if needed to ensure uniqueness
        return f"{name}-{index}" if index > 0 else name
    
    def _generate_profile_description(self, clusters: List[sqlite3.Row], success_rate: float) -> str:
        """Generate a human-readable profile description."""
        total = sum(c["total_models"] for c in clusters)
        
        # Collect characteristics
        face_buckets = set(c["face_count_bucket"] for c in clusters)
        body_buckets = set(c["body_count_bucket"] for c in clusters)
        issues = set()
        for c in clusters:
            if c["issue_signature"] != "none":
                issues.update(c["issue_signature"].split(","))
        
        parts = []
        
        # Face count
        if "huge" in face_buckets or "large" in face_buckets:
            parts.append("high-poly")
        elif "tiny" in face_buckets:
            parts.append("low-poly")
        
        # Body count
        if "20+" in body_buckets:
            parts.append("highly fragmented")
        elif "6-20" in body_buckets:
            parts.append("fragmented")
        elif "2-5" in body_buckets:
            parts.append("multi-body")
        
        # Issues
        if issues:
            parts.append(f"with {', '.join(sorted(issues)[:2])}")
        
        desc = " ".join(parts) if parts else "Standard"
        return f"{desc.capitalize()} meshes ({total} samples, {success_rate*100:.0f}% success rate)"
    
    def _get_best_pipelines_for_clusters(
        self, cursor: sqlite3.Cursor, cluster_keys: List[str]
    ) -> Tuple[Optional[str], float, List[str]]:
        """Get the best performing pipelines for a set of clusters."""
        placeholders = ",".join("?" * len(cluster_keys))
        
        cursor.execute(f"""
            SELECT pipeline_name,
                   SUM(attempts) as total_attempts,
                   SUM(successes) as total_successes,
                   CAST(SUM(successes) AS REAL) / NULLIF(SUM(attempts), 0) as success_rate
            FROM cluster_pipeline_stats
            WHERE cluster_key IN ({placeholders})
            GROUP BY pipeline_name
            HAVING total_attempts >= 3
            ORDER BY success_rate DESC, total_attempts DESC
            LIMIT 5
        """, cluster_keys)
        
        results = cursor.fetchall()
        
        if not results:
            return None, 0, []
        
        best = results[0]
        recommended = [r["pipeline_name"] for r in results[:3]]
        
        return best["pipeline_name"], best["success_rate"] or 0, recommended
    
    def _check_for_profile_splits(self, cursor: sqlite3.Cursor, now: str) -> None:
        """Check existing profiles for high variance that may need splitting."""
        cursor.execute("""
            SELECT profile_id, name, total_models, successful_repairs, failed_repairs
            FROM discovered_profiles
            WHERE is_active = 1
              AND total_models >= ?
        """, (MIN_CLUSTER_SIZE * 2,))
        
        for profile in cursor.fetchall():
            success_rate = profile["successful_repairs"] / profile["total_models"] if profile["total_models"] > 0 else 0
            
            # Check if profile has low success rate - may need splitting
            if success_rate < LOW_SUCCESS_THRESHOLD:
                logger.info(f"Profile '{profile['name']}' has low success rate ({success_rate*100:.1f}%) - may need splitting")
                # TODO: Implement profile splitting logic
    
    def get_profile_for_mesh(self, diagnostics: Dict[str, Any], issues: List[str] = None) -> Optional[DiscoveredProfile]:
        """Get the best matching discovered profile for a mesh."""
        chars = self.extract_characteristics(diagnostics, issues)
        cluster_key = chars.to_key()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # First try exact cluster match
            cursor.execute("""
                SELECT dp.*
                FROM discovered_profiles dp
                JOIN characteristic_clusters cc ON cc.assigned_profile_id = dp.profile_id
                WHERE cc.cluster_key = ?
                  AND dp.is_active = 1
            """, (cluster_key,))
            
            row = cursor.fetchone()
            
            if row:
                return self._row_to_profile(row)
            
            # Try fuzzy match based on key characteristics
            cursor.execute("""
                SELECT dp.*, 
                       (CASE WHEN cc.body_count_bucket = ? THEN 2 ELSE 0 END +
                        CASE WHEN cc.is_watertight = ? THEN 1 ELSE 0 END +
                        CASE WHEN cc.face_count_bucket = ? THEN 1 ELSE 0 END) as match_score
                FROM discovered_profiles dp
                JOIN characteristic_clusters cc ON cc.assigned_profile_id = dp.profile_id
                WHERE dp.is_active = 1
                ORDER BY match_score DESC, dp.total_models DESC
                LIMIT 1
            """, (chars.body_count_bucket, 1 if chars.is_watertight else 0, chars.face_count_bucket))
            
            row = cursor.fetchone()
            
            if row and row["match_score"] >= 2:
                return self._row_to_profile(row)
        
        return None
    
    def _row_to_profile(self, row: sqlite3.Row) -> DiscoveredProfile:
        """Convert a database row to a DiscoveredProfile."""
        return DiscoveredProfile(
            name=row["name"],
            description=row["description"],
            characteristics=json.loads(row["characteristics_json"]) if row["characteristics_json"] else {},
            total_models=row["total_models"],
            successful_repairs=row["successful_repairs"],
            failed_repairs=row["failed_repairs"],
            best_pipeline=row["best_pipeline"],
            best_pipeline_success_rate=row["best_pipeline_success_rate"] or 0,
            recommended_pipelines=json.loads(row["recommended_pipelines_json"]) if row["recommended_pipelines_json"] else [],
            created_at=row["created_at"] or "",
            updated_at=row["updated_at"] or "",
            is_promoted=bool(row["is_promoted"]),
        )
    
    def get_all_profiles(self, include_inactive: bool = False) -> List[DiscoveredProfile]:
        """Get all discovered profiles."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM discovered_profiles"
            if not include_inactive:
                query += " WHERE is_active = 1"
            query += " ORDER BY total_models DESC"
            
            cursor.execute(query)
            return [self._row_to_profile(row) for row in cursor.fetchall()]
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """Get summary statistics for profile discovery."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Total clusters
            cursor.execute("SELECT COUNT(*) as count, SUM(total_models) as models FROM characteristic_clusters")
            cluster_stats = cursor.fetchone()
            
            # Active profiles
            cursor.execute("""
                SELECT COUNT(*) as count, 
                       SUM(total_models) as models,
                       AVG(CAST(successful_repairs AS REAL) / NULLIF(total_models, 0)) as avg_success_rate
                FROM discovered_profiles
                WHERE is_active = 1
            """)
            profile_stats = cursor.fetchone()
            
            # Unassigned clusters
            cursor.execute("""
                SELECT COUNT(*) as count, SUM(total_models) as models
                FROM characteristic_clusters
                WHERE assigned_profile_id IS NULL
            """)
            unassigned = cursor.fetchone()
            
            # Top profiles
            cursor.execute("""
                SELECT name, total_models, successful_repairs, failed_repairs, best_pipeline
                FROM discovered_profiles
                WHERE is_active = 1
                ORDER BY total_models DESC
                LIMIT 10
            """)
            top_profiles = [
                {
                    "name": r["name"],
                    "total_models": r["total_models"],
                    "success_rate": r["successful_repairs"] / r["total_models"] if r["total_models"] > 0 else 0,
                    "best_pipeline": r["best_pipeline"],
                }
                for r in cursor.fetchall()
            ]
            
            return {
                "total_clusters": cluster_stats["count"] or 0,
                "total_models_clustered": cluster_stats["models"] or 0,
                "active_profiles": profile_stats["count"] or 0,
                "models_with_profiles": profile_stats["models"] or 0,
                "avg_profile_success_rate": profile_stats["avg_success_rate"] or 0,
                "unassigned_clusters": unassigned["count"] or 0,
                "unassigned_models": unassigned["models"] or 0,
                "top_profiles": top_profiles,
            }
    
    def promote_profile(self, profile_name: str) -> bool:
        """Mark a discovered profile as promoted (for inclusion in standard profiles)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            
            cursor.execute("""
                UPDATE discovered_profiles
                SET is_promoted = 1, updated_at = ?
                WHERE name = ?
            """, (now, profile_name))
            
            if cursor.rowcount > 0:
                cursor.execute("""
                    INSERT INTO discovery_history (action, profile_name, details_json, created_at)
                    VALUES ('promoted', ?, '{}', ?)
                """, (profile_name, now))
                return True
            return False


# Global instance
_discovery_engine: Optional[ProfileDiscoveryEngine] = None


def get_discovery_engine() -> ProfileDiscoveryEngine:
    """Get or create the global profile discovery engine instance."""
    global _discovery_engine
    if _discovery_engine is None:
        _discovery_engine = ProfileDiscoveryEngine()
    return _discovery_engine
