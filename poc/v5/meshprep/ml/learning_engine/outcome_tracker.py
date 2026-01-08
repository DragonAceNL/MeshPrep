# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Outcome Tracker - Records repair outcomes for learning.

Stores:
- Original mesh features
- Actions applied
- Repair outcome (success, quality score)
- Fidelity metrics (volume change, Hausdorff distance)

This data is used to train the repair predictor via the learning loop.
"""

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class RepairOutcome:
    """Record of a single repair attempt."""
    
    # Identification
    mesh_id: str
    timestamp: str = ""
    
    # Input mesh features
    input_features: Dict[str, float] = field(default_factory=dict)
    input_feature_vector: Optional[List[float]] = None
    
    # Actions applied
    actions: List[str] = field(default_factory=list)
    parameters: Dict[str, Dict] = field(default_factory=dict)
    
    # Outcome
    success: bool = False
    is_printable: bool = False
    quality_score: float = 0.0  # 1-5
    
    # Fidelity metrics
    volume_change_pct: float = 0.0
    hausdorff_relative: float = 0.0
    bbox_change_pct: float = 0.0
    
    # Timing
    duration_ms: float = 0.0
    
    # For learning
    target_actions: Optional[List[str]] = None  # Ground truth (if known)
    
    def to_training_sample(self) -> Dict:
        """Convert to training sample format."""
        return {
            "features": self.input_feature_vector,
            "actions": self.actions,
            "parameters": self.parameters,
            "quality_score": self.quality_score,
            "is_printable": self.is_printable,
            "reward": self._compute_reward(),
        }
    
    def _compute_reward(self) -> float:
        """
        Compute reward signal for reinforcement learning.
        
        Reward structure:
        - Base: quality_score / 5 (0-1)
        - Bonus for printability: +0.3
        - Penalty for large changes: -0.2 * change_severity
        """
        reward = self.quality_score / 5.0
        
        # Printability bonus
        if self.is_printable:
            reward += 0.3
        
        # Fidelity penalty
        change_severity = min(abs(self.volume_change_pct) / 50, 1.0)
        reward -= 0.2 * change_severity
        
        # Hausdorff penalty
        hausdorff_penalty = min(self.hausdorff_relative * 10, 0.2)
        reward -= hausdorff_penalty
        
        return max(0.0, min(1.0, reward))


class OutcomeTracker:
    """
    SQLite-backed storage for repair outcomes.
    
    Provides:
    - Recording new outcomes
    - Querying outcomes for training
    - Statistics and analysis
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = Path("learning_data/repair_outcomes.db")
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mesh_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    
                    -- Features (stored as JSON)
                    input_features TEXT,
                    input_feature_vector TEXT,
                    
                    -- Actions (stored as JSON)
                    actions TEXT,
                    parameters TEXT,
                    
                    -- Outcome
                    success INTEGER,
                    is_printable INTEGER,
                    quality_score REAL,
                    
                    -- Fidelity
                    volume_change_pct REAL,
                    hausdorff_relative REAL,
                    bbox_change_pct REAL,
                    
                    -- Timing
                    duration_ms REAL,
                    
                    -- Computed reward
                    reward REAL
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_outcomes_quality
                ON outcomes(quality_score DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_outcomes_printable
                ON outcomes(is_printable)
            """)
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def record(self, outcome: RepairOutcome) -> int:
        """
        Record a repair outcome.
        
        Returns:
            ID of the recorded outcome
        """
        if not outcome.timestamp:
            outcome.timestamp = datetime.now().isoformat()
        
        sample = outcome.to_training_sample()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO outcomes (
                    mesh_id, timestamp,
                    input_features, input_feature_vector,
                    actions, parameters,
                    success, is_printable, quality_score,
                    volume_change_pct, hausdorff_relative, bbox_change_pct,
                    duration_ms, reward
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                outcome.mesh_id,
                outcome.timestamp,
                json.dumps(outcome.input_features),
                json.dumps(outcome.input_feature_vector) if outcome.input_feature_vector else None,
                json.dumps(outcome.actions),
                json.dumps(outcome.parameters),
                int(outcome.success),
                int(outcome.is_printable),
                outcome.quality_score,
                outcome.volume_change_pct,
                outcome.hausdorff_relative,
                outcome.bbox_change_pct,
                outcome.duration_ms,
                sample["reward"],
            ))
            
            conn.commit()
            return cursor.lastrowid
    
    def get_training_data(
        self,
        min_quality: float = 0.0,
        only_printable: bool = False,
        limit: int = 10000,
    ) -> List[Dict]:
        """
        Get training samples from recorded outcomes.
        
        Args:
            min_quality: Minimum quality score to include
            only_printable: Only include outcomes that achieved printability
            limit: Maximum number of samples
            
        Returns:
            List of training sample dicts
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT * FROM outcomes
                WHERE quality_score >= ?
            """
            params = [min_quality]
            
            if only_printable:
                query += " AND is_printable = 1"
            
            query += " ORDER BY reward DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            
            samples = []
            for row in cursor.fetchall():
                feature_vector = json.loads(row["input_feature_vector"]) if row["input_feature_vector"] else None
                
                samples.append({
                    "features": feature_vector,
                    "actions": json.loads(row["actions"]),
                    "parameters": json.loads(row["parameters"]) if row["parameters"] else {},
                    "quality_score": row["quality_score"],
                    "is_printable": bool(row["is_printable"]),
                    "reward": row["reward"],
                })
            
            return samples
    
    def get_best_actions_for_features(
        self,
        feature_vector: List[float],
        k: int = 5,
    ) -> List[Dict]:
        """
        Find the best-performing actions for similar meshes.
        
        Uses simple nearest-neighbor lookup in feature space.
        
        Args:
            feature_vector: Query mesh features
            k: Number of neighbors to consider
            
        Returns:
            List of action sequences that worked well for similar meshes
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get all outcomes with feature vectors
            cursor.execute("""
                SELECT input_feature_vector, actions, parameters, reward
                FROM outcomes
                WHERE input_feature_vector IS NOT NULL
                AND is_printable = 1
                ORDER BY reward DESC
            """)
            
            query_vec = np.array(feature_vector)
            
            neighbors = []
            for row in cursor.fetchall():
                stored_vec = np.array(json.loads(row["input_feature_vector"]))
                
                # Compute distance
                if len(stored_vec) == len(query_vec):
                    dist = np.linalg.norm(query_vec - stored_vec)
                    neighbors.append({
                        "distance": dist,
                        "actions": json.loads(row["actions"]),
                        "parameters": json.loads(row["parameters"]) if row["parameters"] else {},
                        "reward": row["reward"],
                    })
            
            # Sort by distance and return top k
            neighbors.sort(key=lambda x: x["distance"])
            return neighbors[:k]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about recorded outcomes."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Total count
            cursor.execute("SELECT COUNT(*) FROM outcomes")
            total = cursor.fetchone()[0]
            
            # Success rate
            cursor.execute("SELECT COUNT(*) FROM outcomes WHERE is_printable = 1")
            printable = cursor.fetchone()[0]
            
            # Average quality
            cursor.execute("SELECT AVG(quality_score) FROM outcomes")
            avg_quality = cursor.fetchone()[0] or 0
            
            # Action frequency
            cursor.execute("SELECT actions FROM outcomes")
            action_counts = {}
            for row in cursor.fetchall():
                for action in json.loads(row["actions"]):
                    action_counts[action] = action_counts.get(action, 0) + 1
            
            return {
                "total_outcomes": total,
                "printable_count": printable,
                "success_rate": printable / max(total, 1),
                "avg_quality_score": round(avg_quality, 2),
                "action_frequency": action_counts,
            }
    
    def clear(self):
        """Clear all recorded outcomes (for testing)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM outcomes")
            conn.commit()
