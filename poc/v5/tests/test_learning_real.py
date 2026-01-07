# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Test learning system with real database operations."""

import pytest
from meshprep.learning import HistoryTracker, StrategyLearner
from meshprep.core import Mesh
import trimesh


class TestHistoryTracker:
    """Test history tracking with real database."""
    
    def test_record_repair(self, temp_db):
        """Record repair in database."""
        tracker = HistoryTracker(temp_db)
        
        # Create test mesh
        mesh = Mesh(trimesh.primitives.Box())
        fingerprint = tracker.compute_mesh_fingerprint(mesh)
        
        # Record repair
        tracker.record_repair(
            mesh_fingerprint=fingerprint,
            pipeline_name="test-pipeline",
            success=True,
            vertex_count=8,
            face_count=12,
            quality_score=4.5,
            duration_ms=1500.0
        )
        
        # Verify recorded
        stats = tracker.get_pipeline_stats("test-pipeline")
        assert stats is not None
        assert stats["total_attempts"] == 1
        assert stats["successes"] == 1
        assert stats["avg_quality"] == 4.5
    
    def test_multiple_repairs_aggregate(self, temp_db):
        """Multiple repairs aggregate correctly."""
        tracker = HistoryTracker(temp_db)
        
        # Record 10 repairs
        for i in range(10):
            tracker.record_repair(
                mesh_fingerprint=f"mesh-{i}",
                pipeline_name="test-pipeline",
                success=i % 2 == 0,  # 5 successes, 5 failures
                quality_score=4.0,
                duration_ms=1000.0
            )
        
        stats = tracker.get_pipeline_stats("test-pipeline")
        
        assert stats["total_attempts"] == 10
        assert stats["successes"] == 5
        assert stats["avg_quality"] == 4.0
    
    def test_mesh_fingerprint_consistent(self):
        """Mesh fingerprint is consistent."""
        tracker = HistoryTracker()
        
        mesh1 = Mesh(trimesh.primitives.Box())
        mesh2 = Mesh(trimesh.primitives.Box())
        
        fp1 = tracker.compute_mesh_fingerprint(mesh1)
        fp2 = tracker.compute_mesh_fingerprint(mesh2)
        
        # Same mesh should give same fingerprint
        assert fp1 == fp2
        assert fp1.startswith("MP:")


class TestStrategyLearner:
    """Test strategy learning with real data."""
    
    def test_recommend_pipelines(self, temp_db):
        """Get pipeline recommendations from real data."""
        tracker = HistoryTracker(temp_db)
        
        # Record multiple pipelines with different success rates
        for i in range(10):
            tracker.record_repair(
                mesh_fingerprint=f"mesh-{i}",
                pipeline_name="pipeline-A",
                success=True,
                quality_score=4.5,
                duration_ms=1000.0
            )
        
        for i in range(5):
            tracker.record_repair(
                mesh_fingerprint=f"mesh-{i}",
                pipeline_name="pipeline-B",
                success=False,
                quality_score=2.0,
                duration_ms=3000.0
            )
        
        learner = StrategyLearner(tracker)
        recommendations = learner.recommend_pipelines(top_k=5, min_attempts=3)
        
        assert len(recommendations) > 0
        # Pipeline-A should be recommended over Pipeline-B
        assert recommendations[0][0] == "pipeline-A"
    
    def test_analyze_failures(self, temp_db):
        """Analyze failures from real data."""
        tracker = HistoryTracker(temp_db)
        
        # Record some failures
        for i in range(5):
            tracker.record_repair(
                mesh_fingerprint=f"mesh-{i}",
                pipeline_name="failing-pipeline",
                success=False,
                error_message=f"Test error {i}"
            )
        
        learner = StrategyLearner(tracker)
        failures = learner.analyze_failures(pipeline_name="failing-pipeline")
        
        assert len(failures) == 5
        assert all(f["success"] == 0 for f in failures)
    
    def test_suggest_improvements(self, temp_db):
        """Get improvement suggestions."""
        tracker = HistoryTracker(temp_db)
        
        # Record slow pipeline
        for i in range(10):
            tracker.record_repair(
                mesh_fingerprint=f"mesh-{i}",
                pipeline_name="slow-pipeline",
                success=True,
                duration_ms=35000.0  # Very slow
            )
        
        learner = StrategyLearner(tracker)
        suggestions = learner.suggest_improvements()
        
        assert len(suggestions) > 0
        # Should mention slow pipeline
        suggestions_text = " ".join(suggestions).lower()
        assert "slow" in suggestions_text or "optimization" in suggestions_text
    
    def test_statistics_summary(self, temp_db):
        """Get complete statistics summary."""
        tracker = HistoryTracker(temp_db)
        
        # Record varied data
        for i in range(20):
            tracker.record_repair(
                mesh_fingerprint=f"mesh-{i}",
                pipeline_name=f"pipeline-{i % 3}",
                success=i % 4 != 0,  # 75% success rate
                quality_score=4.0
            )
        
        learner = StrategyLearner(tracker)
        summary = learner.get_statistics_summary()
        
        assert summary["total_attempts"] == 20
        assert summary["total_successes"] == 15
        assert summary["overall_success_rate"] == 0.75
        assert summary["total_pipelines"] == 3
