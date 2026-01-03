# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Evolutionary Pipeline Discovery for MeshPrep.

This module implements a genetic algorithm-inspired approach to discover
new repair pipeline combinations by:
1. Tracking individual action success rates
2. Combining successful actions from different pipelines
3. Occasionally mutating pipelines (add/remove/swap actions)
4. Saving winning combinations as new pipelines

The goal is to discover better repair strategies over time through
exploration and exploitation of the action space.
"""

import hashlib
import json
import logging
import random
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

# All available repair actions that can be combined
AVAILABLE_ACTIONS = [
    # Basic cleanup
    {"action": "trimesh_basic", "params": {}},
    {"action": "remove_degenerate", "params": {}},
    {"action": "fix_normals", "params": {}},
    {"action": "fix_winding", "params": {}},
    
    # Hole filling variants
    {"action": "fill_holes", "params": {"max_hole_size": 50}},
    {"action": "fill_holes", "params": {"max_hole_size": 100}},
    {"action": "fill_holes", "params": {"max_hole_size": 500}},
    {"action": "fill_holes", "params": {"max_hole_size": 1000}},
    
    # PyMeshFix variants
    {"action": "pymeshfix_repair", "params": {}},
    {"action": "pymeshfix_clean", "params": {}},
    {"action": "pymeshfix_repair_conservative", "params": {}},
    
    # Manifold repair
    {"action": "make_manifold", "params": {}},
    
    # Placement
    {"action": "place_on_bed", "params": {}},
    
    # Blender actions (expensive, use sparingly)
    {"action": "blender_remesh", "params": {"voxel_size": "auto"}},
    {"action": "blender_remesh", "params": {"voxel_size": 0.5}},
    {"action": "blender_remesh", "params": {"voxel_size": 1.0}},
]

# Actions that should typically come first (preparation)
PREP_ACTIONS = ["trimesh_basic", "remove_degenerate", "fix_normals", "fix_winding", "place_on_bed"]

# Actions that should typically come last (finalization)
FINAL_ACTIONS = ["blender_remesh"]

# Maximum actions in a pipeline (to prevent bloat)
MAX_PIPELINE_LENGTH = 5

# Minimum actions in a pipeline
MIN_PIPELINE_LENGTH = 1


@dataclass
class EvolvedPipeline:
    """A pipeline created through evolution."""
    name: str
    actions: List[Dict[str, Any]]
    parent_pipelines: List[str] = field(default_factory=list)
    generation: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Performance tracking
    attempts: int = 0
    successes: int = 0
    total_duration_ms: float = 0
    
    @property
    def success_rate(self) -> float:
        return self.successes / self.attempts if self.attempts > 0 else 0.0
    
    @property
    def avg_duration_ms(self) -> float:
        return self.total_duration_ms / self.attempts if self.attempts > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "actions": self.actions,
            "parent_pipelines": self.parent_pipelines,
            "generation": self.generation,
            "created_at": self.created_at,
            "attempts": self.attempts,
            "successes": self.successes,
            "total_duration_ms": self.total_duration_ms,
            "success_rate": self.success_rate,
            "avg_duration_ms": self.avg_duration_ms,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvolvedPipeline":
        return cls(
            name=data["name"],
            actions=data["actions"],
            parent_pipelines=data.get("parent_pipelines", []),
            generation=data.get("generation", 0),
            created_at=data.get("created_at", datetime.now().isoformat()),
            attempts=data.get("attempts", 0),
            successes=data.get("successes", 0),
            total_duration_ms=data.get("total_duration_ms", 0),
        )


class PipelineEvolution:
    """
    Evolutionary pipeline discovery engine.
    
    Uses genetic algorithm concepts to discover new effective pipeline combinations:
    - Selection: Choose successful pipelines/actions as parents
    - Crossover: Combine actions from different successful pipelines
    - Mutation: Randomly add/remove/swap actions
    - Fitness: Track success rate and speed
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """Initialize the evolution engine.
        
        Args:
            data_path: Path to store evolution data. If None, uses default location.
        """
        self.data_path = data_path or Path(__file__).parent.parent.parent.parent / "learning_data"
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_path / "pipeline_evolution.db"
        
        self._init_database()
        
        # Cache for evolved pipelines
        self._evolved_pipelines: Dict[str, EvolvedPipeline] = {}
        self._load_evolved_pipelines()
    
    def _init_database(self) -> None:
        """Initialize the SQLite database for evolution tracking."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Track individual action performance
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS action_stats (
                    action_key TEXT PRIMARY KEY,
                    action_name TEXT,
                    params_json TEXT,
                    total_attempts INTEGER DEFAULT 0,
                    successes INTEGER DEFAULT 0,
                    failures INTEGER DEFAULT 0,
                    total_duration_ms REAL DEFAULT 0,
                    updated_at TEXT
                )
            """)
            
            # Track action performance by issue type
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS action_issue_stats (
                    action_key TEXT,
                    issue_type TEXT,
                    successes INTEGER DEFAULT 0,
                    failures INTEGER DEFAULT 0,
                    PRIMARY KEY (action_key, issue_type)
                )
            """)
            
            # Evolved pipelines
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evolved_pipelines (
                    name TEXT PRIMARY KEY,
                    actions_json TEXT,
                    parent_pipelines_json TEXT,
                    generation INTEGER DEFAULT 0,
                    attempts INTEGER DEFAULT 0,
                    successes INTEGER DEFAULT 0,
                    total_duration_ms REAL DEFAULT 0,
                    created_at TEXT,
                    updated_at TEXT,
                    is_promoted INTEGER DEFAULT 0
                )
            """)
            
            # Evolution history (for analysis)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evolution_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pipeline_name TEXT,
                    event_type TEXT,
                    details_json TEXT,
                    created_at TEXT
                )
            """)
            
            conn.commit()
            logger.debug(f"Evolution database initialized at {self.db_path}")
    
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
    
    def _get_action_key(self, action: Dict[str, Any]) -> str:
        """Generate a unique key for an action + params combination."""
        params_str = json.dumps(action.get("params", {}), sort_keys=True)
        return f"{action['action']}:{params_str}"
    
    def _generate_pipeline_name(self, actions: List[Dict[str, Any]]) -> str:
        """Generate a unique name for a pipeline based on its actions."""
        action_str = "|".join(self._get_action_key(a) for a in actions)
        hash_suffix = hashlib.md5(action_str.encode()).hexdigest()[:8]
        
        # Create a readable prefix
        if len(actions) == 1:
            prefix = actions[0]["action"]
        elif len(actions) == 2:
            prefix = f"{actions[0]['action']}-{actions[1]['action']}"
        else:
            prefix = f"{actions[0]['action']}-combo"
        
        return f"evolved-{prefix}-{hash_suffix}"
    
    def _load_evolved_pipelines(self) -> None:
        """Load evolved pipelines from database into cache."""
        self._evolved_pipelines = {}
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM evolved_pipelines")
            
            for row in cursor.fetchall():
                pipeline = EvolvedPipeline(
                    name=row["name"],
                    actions=json.loads(row["actions_json"]),
                    parent_pipelines=json.loads(row["parent_pipelines_json"] or "[]"),
                    generation=row["generation"],
                    created_at=row["created_at"],
                    attempts=row["attempts"],
                    successes=row["successes"],
                    total_duration_ms=row["total_duration_ms"],
                )
                self._evolved_pipelines[pipeline.name] = pipeline
    
    def record_action_result(
        self,
        action: Dict[str, Any],
        success: bool,
        duration_ms: float,
        issues: List[str] = None,
    ) -> None:
        """Record the result of an individual action execution.
        
        Args:
            action: The action dict with 'action' and 'params' keys
            success: Whether this action contributed to a successful repair
            duration_ms: How long the action took
            issues: List of issues the model had (for issue-specific tracking)
        """
        action_key = self._get_action_key(action)
        action_name = action["action"]
        params_json = json.dumps(action.get("params", {}), sort_keys=True)
        now = datetime.now().isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Update action stats
            cursor.execute("""
                INSERT INTO action_stats 
                (action_key, action_name, params_json, total_attempts, successes, failures, total_duration_ms, updated_at)
                VALUES (?, ?, ?, 1, ?, ?, ?, ?)
                ON CONFLICT(action_key) DO UPDATE SET
                    total_attempts = total_attempts + 1,
                    successes = successes + ?,
                    failures = failures + ?,
                    total_duration_ms = total_duration_ms + ?,
                    updated_at = ?
            """, (
                action_key, action_name, params_json,
                1 if success else 0,
                0 if success else 1,
                duration_ms, now,
                1 if success else 0,
                0 if success else 1,
                duration_ms, now
            ))
            
            # Update issue-specific stats
            if issues:
                for issue in issues:
                    cursor.execute("""
                        INSERT INTO action_issue_stats (action_key, issue_type, successes, failures)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(action_key, issue_type) DO UPDATE SET
                            successes = successes + ?,
                            failures = failures + ?
                    """, (
                        action_key, issue,
                        1 if success else 0,
                        0 if success else 1,
                        1 if success else 0,
                        0 if success else 1
                    ))
    
    def record_pipeline_result(
        self,
        pipeline_name: str,
        success: bool,
        duration_ms: float,
    ) -> None:
        """Record the result of an evolved pipeline execution."""
        if pipeline_name not in self._evolved_pipelines:
            return
        
        pipeline = self._evolved_pipelines[pipeline_name]
        pipeline.attempts += 1
        if success:
            pipeline.successes += 1
        pipeline.total_duration_ms += duration_ms
        
        now = datetime.now().isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE evolved_pipelines
                SET attempts = ?, successes = ?, total_duration_ms = ?, updated_at = ?
                WHERE name = ?
            """, (pipeline.attempts, pipeline.successes, pipeline.total_duration_ms, now, pipeline_name))
    
    def get_best_actions_for_issues(
        self,
        issues: List[str],
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get the best-performing actions for the given issues.
        
        Args:
            issues: List of issue types to consider
            top_n: Maximum number of actions to return
            
        Returns:
            List of action dicts, sorted by success rate for these issues
        """
        if not issues:
            issues = ["unknown"]
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get actions that have worked well for these issues
            placeholders = ",".join("?" * len(issues))
            cursor.execute(f"""
                SELECT a.action_name, a.params_json,
                       SUM(i.successes) as total_successes,
                       SUM(i.failures) as total_failures,
                       CAST(SUM(i.successes) AS REAL) / NULLIF(SUM(i.successes) + SUM(i.failures), 0) as success_rate
                FROM action_stats a
                JOIN action_issue_stats i ON a.action_key = i.action_key
                WHERE i.issue_type IN ({placeholders})
                  AND (i.successes + i.failures) >= 3
                GROUP BY a.action_key
                HAVING success_rate > 0.3
                ORDER BY success_rate DESC, total_successes DESC
                LIMIT ?
            """, issues + [top_n])
            
            actions = []
            for row in cursor.fetchall():
                actions.append({
                    "action": row["action_name"],
                    "params": json.loads(row["params_json"]),
                    "_success_rate": row["success_rate"],
                })
            
            # If not enough issue-specific actions, add globally good actions
            if len(actions) < top_n:
                cursor.execute("""
                    SELECT action_name, params_json,
                           CAST(successes AS REAL) / NULLIF(total_attempts, 0) as success_rate
                    FROM action_stats
                    WHERE total_attempts >= 5
                      AND CAST(successes AS REAL) / NULLIF(total_attempts, 0) > 0.4
                    ORDER BY success_rate DESC
                    LIMIT ?
                """, (top_n - len(actions),))
                
                existing_keys = {self._get_action_key(a) for a in actions}
                for row in cursor.fetchall():
                    action = {
                        "action": row["action_name"],
                        "params": json.loads(row["params_json"]),
                    }
                    if self._get_action_key(action) not in existing_keys:
                        actions.append(action)
            
            return actions
    
    def _is_valid_action_order(self, actions: List[Dict[str, Any]]) -> bool:
        """Check if the action order makes sense."""
        action_names = [a["action"] for a in actions]
        
        # Blender should be last if present
        blender_indices = [i for i, name in enumerate(action_names) if "blender" in name]
        if blender_indices and blender_indices[0] != len(actions) - 1:
            return False
        
        # Don't have duplicate actions (same action + params)
        seen = set()
        for action in actions:
            key = self._get_action_key(action)
            if key in seen:
                return False
            seen.add(key)
        
        return True
    
    def _reorder_actions(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reorder actions to a sensible sequence."""
        prep = []
        middle = []
        final = []
        
        for action in actions:
            name = action["action"]
            if name in PREP_ACTIONS:
                prep.append(action)
            elif name in FINAL_ACTIONS or "blender" in name:
                final.append(action)
            else:
                middle.append(action)
        
        return prep + middle + final
    
    def generate_evolved_pipeline(
        self,
        issues: List[str],
        diagnostics: Optional[Dict[str, Any]] = None,
        exploration_rate: float = 0.2,
    ) -> Optional[EvolvedPipeline]:
        """Generate a new evolved pipeline for the given issues.
        
        Uses genetic algorithm concepts:
        - Selection: Get best actions for these issues
        - Crossover: Combine actions from multiple sources
        - Mutation: Randomly modify with some probability
        
        Args:
            issues: List of issues to address
            diagnostics: Optional mesh diagnostics for smarter selection
            exploration_rate: Probability of including random/experimental actions
            
        Returns:
            New EvolvedPipeline, or None if generation failed
        """
        # Get best-performing actions for these issues
        best_actions = self.get_best_actions_for_issues(issues, top_n=15)
        
        if not best_actions:
            # No learned data yet - use random selection from available actions
            best_actions = random.sample(AVAILABLE_ACTIONS, min(5, len(AVAILABLE_ACTIONS)))
        
        # Determine pipeline length
        num_actions = random.randint(MIN_PIPELINE_LENGTH, MAX_PIPELINE_LENGTH)
        
        # Select actions with bias toward best performers
        selected_actions = []
        
        # Always try to include a core repair action
        core_repair_actions = [a for a in best_actions if a["action"] in ["pymeshfix_repair", "fill_holes", "make_manifold"]]
        if core_repair_actions:
            selected_actions.append(random.choice(core_repair_actions))
        
        # Add more actions
        remaining_pool = [a for a in best_actions if a not in selected_actions]
        
        while len(selected_actions) < num_actions and remaining_pool:
            if random.random() < exploration_rate:
                # Exploration: pick random action from all available
                candidates = [a for a in AVAILABLE_ACTIONS 
                             if self._get_action_key(a) not in {self._get_action_key(s) for s in selected_actions}]
                if candidates:
                    selected_actions.append(random.choice(candidates))
            else:
                # Exploitation: pick from best performers
                # Weight by index (earlier = better)
                weights = [1.0 / (i + 1) for i in range(len(remaining_pool))]
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]
                
                idx = random.choices(range(len(remaining_pool)), weights=weights, k=1)[0]
                selected_actions.append(remaining_pool.pop(idx))
        
        if not selected_actions:
            return None
        
        # Reorder actions to sensible sequence
        selected_actions = self._reorder_actions(selected_actions)
        
        # Validate order
        if not self._is_valid_action_order(selected_actions):
            return None
        
        # Check if this pipeline already exists
        pipeline_name = self._generate_pipeline_name(selected_actions)
        if pipeline_name in self._evolved_pipelines:
            return self._evolved_pipelines[pipeline_name]
        
        # Create new evolved pipeline
        pipeline = EvolvedPipeline(
            name=pipeline_name,
            actions=selected_actions,
            parent_pipelines=[],  # Could track parent pipeline names if doing crossover
            generation=self._get_current_generation() + 1,
        )
        
        # Save to database
        self._save_evolved_pipeline(pipeline)
        self._evolved_pipelines[pipeline_name] = pipeline
        
        logger.info(f"Generated new evolved pipeline: {pipeline_name} with {len(selected_actions)} actions")
        
        return pipeline
    
    def _get_current_generation(self) -> int:
        """Get the current maximum generation number."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(generation) FROM evolved_pipelines")
            result = cursor.fetchone()[0]
            return result or 0
    
    def _save_evolved_pipeline(self, pipeline: EvolvedPipeline) -> None:
        """Save an evolved pipeline to the database."""
        now = datetime.now().isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO evolved_pipelines
                (name, actions_json, parent_pipelines_json, generation, attempts, successes, 
                 total_duration_ms, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pipeline.name,
                json.dumps(pipeline.actions),
                json.dumps(pipeline.parent_pipelines),
                pipeline.generation,
                pipeline.attempts,
                pipeline.successes,
                pipeline.total_duration_ms,
                pipeline.created_at,
                now
            ))
            
            # Log creation event
            cursor.execute("""
                INSERT INTO evolution_history (pipeline_name, event_type, details_json, created_at)
                VALUES (?, 'created', ?, ?)
            """, (pipeline.name, json.dumps(pipeline.to_dict()), now))
    
    def get_evolved_pipelines_for_issues(
        self,
        issues: List[str],
        max_pipelines: int = 3,
        min_success_rate: float = 0.3,
        min_attempts: int = 3,
    ) -> List[EvolvedPipeline]:
        """Get evolved pipelines that have worked well for similar issues.
        
        Args:
            issues: List of issues to consider
            max_pipelines: Maximum number of pipelines to return
            min_success_rate: Minimum success rate threshold
            min_attempts: Minimum attempts before considering
            
        Returns:
            List of EvolvedPipeline objects, sorted by success rate
        """
        # Get pipelines with good success rates
        candidates = [
            p for p in self._evolved_pipelines.values()
            if p.attempts >= min_attempts and p.success_rate >= min_success_rate
        ]
        
        # Sort by success rate (descending), then by avg duration (ascending)
        candidates.sort(key=lambda p: (-p.success_rate, p.avg_duration_ms))
        
        return candidates[:max_pipelines]
    
    def should_try_evolution(
        self,
        issues: List[str],
        failed_pipelines: List[str],
        attempt_number: int,
    ) -> bool:
        """Determine if we should try an evolved pipeline.
        
        Args:
            issues: Current issues to fix
            failed_pipelines: List of pipeline names that have already failed
            attempt_number: Current attempt number
            
        Returns:
            True if we should try generating/using an evolved pipeline
        """
        # Don't try evolution too early - let standard pipelines have a chance
        if attempt_number < 5:
            return False
        
        # More likely to try evolution if many pipelines have failed
        failure_factor = min(len(failed_pipelines) / 10.0, 1.0)
        
        # Random chance increases with failures
        evolution_chance = 0.1 + (0.4 * failure_factor)
        
        return random.random() < evolution_chance
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the evolution engine."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Count evolved pipelines
            cursor.execute("SELECT COUNT(*) FROM evolved_pipelines")
            total_evolved = cursor.fetchone()[0]
            
            # Count successful evolved pipelines
            cursor.execute("""
                SELECT COUNT(*) FROM evolved_pipelines 
                WHERE attempts >= 3 AND CAST(successes AS REAL) / attempts > 0.5
            """)
            successful_evolved = cursor.fetchone()[0]
            
            # Get top evolved pipelines
            cursor.execute("""
                SELECT name, attempts, successes, 
                       CAST(successes AS REAL) / NULLIF(attempts, 0) as success_rate,
                       total_duration_ms / NULLIF(attempts, 0) as avg_duration_ms
                FROM evolved_pipelines
                WHERE attempts >= 3
                ORDER BY success_rate DESC
                LIMIT 5
            """)
            top_evolved = [
                {
                    "name": row["name"],
                    "attempts": row["attempts"],
                    "successes": row["successes"],
                    "success_rate": row["success_rate"] or 0,
                    "avg_duration_ms": row["avg_duration_ms"] or 0,
                }
                for row in cursor.fetchall()
            ]
            
            # Count tracked actions
            cursor.execute("SELECT COUNT(*) FROM action_stats")
            tracked_actions = cursor.fetchone()[0]
            
            # Current generation
            cursor.execute("SELECT MAX(generation) FROM evolved_pipelines")
            current_generation = cursor.fetchone()[0] or 0
            
            return {
                "total_evolved_pipelines": total_evolved,
                "successful_evolved_pipelines": successful_evolved,
                "tracked_actions": tracked_actions,
                "current_generation": current_generation,
                "top_evolved_pipelines": top_evolved,
            }
    
    def export_successful_pipelines(self, output_path: Optional[Path] = None) -> Path:
        """Export successful evolved pipelines to JSON for manual review.
        
        Pipelines with high success rates can be promoted to the standard
        pipeline library after human review.
        """
        if output_path is None:
            output_path = self.data_path / "evolved_pipelines_export.json"
        
        successful = [
            p.to_dict() for p in self._evolved_pipelines.values()
            if p.attempts >= 5 and p.success_rate >= 0.5
        ]
        
        successful.sort(key=lambda p: -p["success_rate"])
        
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "pipelines": successful,
            "promotion_note": "Pipelines with success_rate >= 0.7 are candidates for promotion to standard library",
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(successful)} successful evolved pipelines to {output_path}")
        return output_path


# Global instance
_evolution_engine: Optional[PipelineEvolution] = None


def get_evolution_engine(data_path: Optional[Path] = None) -> PipelineEvolution:
    """Get or create the global evolution engine instance."""
    global _evolution_engine
    if _evolution_engine is None:
        _evolution_engine = PipelineEvolution(data_path)
    return _evolution_engine
