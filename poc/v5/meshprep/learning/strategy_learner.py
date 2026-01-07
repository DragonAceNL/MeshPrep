# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Strategy learner for recommending optimal pipelines."""

from typing import List, Tuple, Optional, Dict, Any
import logging

from .history_tracker import HistoryTracker

logger = logging.getLogger(__name__)


class StrategyLearner:
    """
    Learns optimal repair strategies from history.
    
    Analyzes repair history to recommend best pipelines
    based on success rates, quality scores, and efficiency.
    """
    
    def __init__(self, tracker: Optional[HistoryTracker] = None):
        """
        Initialize learner.
        
        Args:
            tracker: HistoryTracker instance (creates new if None)
        """
        self.tracker = tracker or HistoryTracker()
        logger.info("StrategyLearner initialized")
    
    def recommend_pipelines(
        self,
        mesh_fingerprint: Optional[str] = None,
        top_k: int = 5,
        min_attempts: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Recommend best pipelines based on history.
        
        Args:
            mesh_fingerprint: Optional mesh fingerprint for specific recommendations
            top_k: Number of pipelines to recommend
            min_attempts: Minimum attempts required for pipeline to be considered
            
        Returns:
            List of (pipeline_name, score) tuples, sorted by score
        """
        stats = self.tracker.get_all_pipeline_stats()
        
        # Filter by minimum attempts
        stats = [s for s in stats if s['total_attempts'] >= min_attempts]
        
        if not stats:
            logger.warning("No pipeline statistics available")
            return []
        
        # Compute scores
        scored_pipelines = []
        for stat in stats:
            score = self._compute_pipeline_score(stat)
            scored_pipelines.append((stat['pipeline_name'], score))
        
        # Sort by score
        scored_pipelines.sort(key=lambda x: x[1], reverse=True)
        
        return scored_pipelines[:top_k]
    
    def _compute_pipeline_score(self, stats: Dict[str, Any]) -> float:
        """
        Compute overall score for a pipeline.
        
        Considers:
        - Success rate (40%)
        - Average quality (30%)
        - Efficiency/speed (20%)
        - Recency (10%)
        """
        total = stats['total_attempts']
        successes = stats['successes']
        avg_quality = stats.get('avg_quality')
        avg_duration = stats.get('avg_duration_ms')
        
        # Success rate (0-1)
        success_rate = successes / total if total > 0 else 0.0
        
        # Quality score (0-1, normalized from 1-5)
        quality_score = (avg_quality - 1) / 4 if avg_quality else 0.5
        
        # Efficiency score (0-1, faster is better)
        # Normalize to 0-1 where <1s=1.0, >10s=0.0
        if avg_duration:
            efficiency = max(0.0, min(1.0, 1.0 - (avg_duration - 1000) / 9000))
        else:
            efficiency = 0.5
        
        # Weighted score
        score = (
            success_rate * 0.4 +
            quality_score * 0.3 +
            efficiency * 0.2 +
            0.1  # Baseline recency bonus
        )
        
        return score
    
    def get_best_pipeline_for_profile(
        self,
        profile: str,
    ) -> Optional[str]:
        """
        Get best pipeline for a mesh profile.
        
        Args:
            profile: Mesh profile name
            
        Returns:
            Best pipeline name or None
        """
        # TODO: Track profile-specific success rates
        # For now, return overall best
        recommendations = self.recommend_pipelines(top_k=1)
        
        if recommendations:
            return recommendations[0][0]
        return None
    
    def get_statistics_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        stats = self.tracker.get_all_pipeline_stats()
        
        if not stats:
            return {
                "total_pipelines": 0,
                "total_attempts": 0,
                "overall_success_rate": 0.0,
            }
        
        total_attempts = sum(s['total_attempts'] for s in stats)
        total_successes = sum(s['successes'] for s in stats)
        
        return {
            "total_pipelines": len(stats),
            "total_attempts": total_attempts,
            "total_successes": total_successes,
            "overall_success_rate": total_successes / total_attempts if total_attempts > 0 else 0.0,
            "pipelines": stats,
        }
    
    def analyze_failures(
        self,
        pipeline_name: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Analyze recent failures.
        
        Args:
            pipeline_name: Filter by pipeline (None for all)
            limit: Maximum failures to return
            
        Returns:
            List of failure records
        """
        # Get recent repairs from tracker
        repairs = self.tracker.get_recent_repairs(limit=limit * 2)
        
        # Filter failures
        failures = [r for r in repairs if not r['success']]
        
        if pipeline_name:
            failures = [f for f in failures if f['pipeline_name'] == pipeline_name]
        
        return failures[:limit]
    
    def suggest_improvements(self) -> List[str]:
        """Suggest improvements based on analysis."""
        suggestions = []
        
        stats = self.tracker.get_all_pipeline_stats()
        
        if not stats:
            suggestions.append("No data collected yet. Run more repairs to build history.")
            return suggestions
        
        # Check for low success rates
        for stat in stats:
            success_rate = stat['successes'] / stat['total_attempts'] if stat['total_attempts'] > 0 else 0
            
            if success_rate < 0.5 and stat['total_attempts'] >= 5:
                suggestions.append(
                    f"Pipeline '{stat['pipeline_name']}' has low success rate "
                    f"({success_rate:.1%}). Consider reviewing or removing."
                )
        
        # Check for slow pipelines
        for stat in stats:
            if stat.get('avg_duration_ms', 0) > 30000:  # >30 seconds
                suggestions.append(
                    f"Pipeline '{stat['pipeline_name']}' is slow "
                    f"({stat['avg_duration_ms']/1000:.1f}s average). Consider optimization."
                )
        
        if not suggestions:
            suggestions.append("All pipelines performing well! Keep collecting data.")
        
        return suggestions
