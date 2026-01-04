# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""Tests for the Visual Quality Feedback System."""

import tempfile
from pathlib import Path

import pytest

from meshprep_poc.quality_feedback import (
    QualityRating,
    QualityPrediction,
    PipelineQualityStats,
    QualityFeedbackEngine,
    get_quality_engine,
    reset_quality_engine,
)


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def quality_engine(temp_data_dir):
    """Create a quality feedback engine with isolated data."""
    reset_quality_engine()
    engine = QualityFeedbackEngine(temp_data_dir)
    yield engine
    reset_quality_engine()


class TestQualityRating:
    """Tests for QualityRating dataclass."""
    
    def test_normalized_score_binary_accept(self):
        """Binary accept should normalize to 1.0."""
        rating = QualityRating(
            model_fingerprint="MP:test123",
            model_filename="test.stl",
            rating_type="binary",
            rating_value=1,
        )
        assert rating.normalized_score() == 1.0
    
    def test_normalized_score_binary_reject(self):
        """Binary reject should normalize to 0.0."""
        rating = QualityRating(
            model_fingerprint="MP:test123",
            model_filename="test.stl",
            rating_type="binary",
            rating_value=0,
        )
        assert rating.normalized_score() == 0.0
    
    def test_normalized_score_gradational(self):
        """Gradational ratings should normalize to 0-1 range."""
        for value in range(1, 6):
            rating = QualityRating(
                model_fingerprint="MP:test123",
                model_filename="test.stl",
                rating_type="gradational",
                rating_value=value,
            )
            expected = (value - 1) / 4.0
            assert rating.normalized_score() == expected


class TestQualityPrediction:
    """Tests for QualityPrediction dataclass."""
    
    def test_is_acceptable_true(self):
        """Scores >= 3 should be acceptable."""
        for score in [3.0, 3.5, 4.0, 5.0]:
            pred = QualityPrediction(score=score, confidence=0.8, based_on_samples=10)
            assert pred.is_acceptable()
    
    def test_is_acceptable_false(self):
        """Scores < 3 should not be acceptable."""
        for score in [1.0, 2.0, 2.9]:
            pred = QualityPrediction(score=score, confidence=0.8, based_on_samples=10)
            assert not pred.is_acceptable()
    
    def test_is_high_quality_true(self):
        """Scores >= 4 should be high quality."""
        for score in [4.0, 4.5, 5.0]:
            pred = QualityPrediction(score=score, confidence=0.8, based_on_samples=10)
            assert pred.is_high_quality()
    
    def test_is_high_quality_false(self):
        """Scores < 4 should not be high quality."""
        for score in [1.0, 2.0, 3.0, 3.9]:
            pred = QualityPrediction(score=score, confidence=0.8, based_on_samples=10)
            assert not pred.is_high_quality()


class TestPipelineQualityStats:
    """Tests for PipelineQualityStats dataclass."""
    
    def test_acceptance_rate(self):
        """Test acceptance rate calculation."""
        stats = PipelineQualityStats(
            pipeline_name="test",
            profile="standard",
            total_ratings=10,
            ratings_5=3,
            ratings_4=2,
            ratings_3=2,
            ratings_2=2,
            ratings_1=1,
        )
        # Acceptable = 5+4+3 ratings = 7 out of 10
        assert stats.acceptance_rate == 0.7
    
    def test_high_quality_rate(self):
        """Test high quality rate calculation."""
        stats = PipelineQualityStats(
            pipeline_name="test",
            profile="standard",
            total_ratings=10,
            ratings_5=3,
            ratings_4=2,
            ratings_3=2,
            ratings_2=2,
            ratings_1=1,
        )
        # High quality = 5+4 ratings = 5 out of 10
        assert stats.high_quality_rate == 0.5
    
    def test_rates_with_zero_ratings(self):
        """Test rate calculations with no ratings."""
        stats = PipelineQualityStats(
            pipeline_name="test",
            profile="standard",
            total_ratings=0,
        )
        assert stats.acceptance_rate == 0.0
        assert stats.high_quality_rate == 0.0


class TestQualityFeedbackEngine:
    """Tests for QualityFeedbackEngine."""
    
    def test_record_rating(self, quality_engine):
        """Test recording a quality rating."""
        rating = QualityRating(
            model_fingerprint="MP:abc123def456",
            model_filename="test_model.stl",
            rating_type="gradational",
            rating_value=4,
            pipeline_used="trimesh-basic",
            profile="standard",
            volume_change_pct=-5.0,
            face_count_change_pct=-10.0,
        )
        
        quality_engine.record_rating(rating)
        
        # Verify it was stored
        stored = quality_engine.get_rating_by_fingerprint("MP:abc123def456")
        assert stored is not None
        assert stored.rating_value == 4
        assert stored.pipeline_used == "trimesh-basic"
    
    def test_predict_quality_no_data(self, quality_engine):
        """Test prediction with no historical data (heuristic mode)."""
        pred = quality_engine.predict_quality(
            pipeline="unknown-pipeline",
            profile="unknown-profile",
            volume_change_pct=-5.0,
            face_count_change_pct=-10.0,
        )
        
        # Should return a heuristic prediction with low confidence
        assert pred is not None
        assert 1.0 <= pred.score <= 5.0
        assert pred.confidence < 0.5  # Low confidence for heuristic
        assert pred.based_on_samples == 0
    
    def test_predict_quality_with_data(self, quality_engine):
        """Test prediction with sufficient historical data."""
        # Record 15 ratings (above MIN_SAMPLES_FOR_PREDICTION)
        for i in range(15):
            rating = QualityRating(
                model_fingerprint=f"MP:test{i:03d}",
                model_filename=f"test_{i}.stl",
                rating_type="gradational",
                rating_value=4,  # Consistently good
                pipeline_used="test-pipeline",
                profile="test-profile",
                volume_change_pct=-3.0,
                face_count_change_pct=-5.0,
            )
            quality_engine.record_rating(rating)
        
        # Now prediction should use historical data
        pred = quality_engine.predict_quality(
            pipeline="test-pipeline",
            profile="test-profile",
            volume_change_pct=-3.0,
            face_count_change_pct=-5.0,
        )
        
        assert pred is not None
        assert pred.score >= 3.5  # Should be close to historical average (4)
        assert pred.confidence > 0.1  # Should have some confidence
        assert pred.based_on_samples == 15
    
    def test_get_pipeline_quality_stats(self, quality_engine):
        """Test retrieving pipeline quality statistics."""
        # Record some ratings
        for value in [5, 4, 4, 3, 3]:
            rating = QualityRating(
                model_fingerprint=f"MP:stat{value}",
                model_filename=f"stat_{value}.stl",
                rating_type="gradational",
                rating_value=value,
                pipeline_used="stats-pipeline",
                profile="stats-profile",
            )
            quality_engine.record_rating(rating)
        
        stats = quality_engine.get_pipeline_quality_stats("stats-pipeline", "stats-profile")
        
        assert len(stats) == 1
        assert stats[0].total_ratings == 5
        assert stats[0].avg_rating == 3.8  # (5+4+4+3+3)/5
        assert stats[0].ratings_5 == 1
        assert stats[0].ratings_4 == 2
        assert stats[0].ratings_3 == 2
    
    def test_get_summary_stats(self, quality_engine):
        """Test getting summary statistics."""
        # Record a few ratings
        for i in range(5):
            rating = QualityRating(
                model_fingerprint=f"MP:sum{i}",
                model_filename=f"sum_{i}.stl",
                rating_type="gradational",
                rating_value=3 + (i % 3),  # Values 3, 4, 5, 3, 4
                pipeline_used="sum-pipeline",
                profile="sum-profile",
            )
            quality_engine.record_rating(rating)
        
        summary = quality_engine.get_summary_stats()
        
        assert summary["total_ratings"] == 5
        assert summary["avg_rating"] > 0
        assert "rating_distribution" in summary
        assert summary["pipeline_profile_combinations"] >= 1
    
    def test_should_flag_for_review_low_prediction(self, quality_engine):
        """Test flagging when predicted quality is low."""
        # Record poor quality history for a pipeline
        for i in range(20):
            rating = QualityRating(
                model_fingerprint=f"MP:poor{i:03d}",
                model_filename=f"poor_{i}.stl",
                rating_type="gradational",
                rating_value=2,  # Poor quality
                pipeline_used="poor-pipeline",
                profile="poor-profile",
                volume_change_pct=-30.0,
            )
            quality_engine.record_rating(rating)
        
        # Should flag for review
        should_flag, reason = quality_engine.should_flag_for_review(
            pipeline="poor-pipeline",
            profile="poor-profile",
            volume_change_pct=-25.0,
            face_count_change_pct=-40.0,
        )
        
        assert should_flag
        assert reason is not None


class TestHeuristicPrediction:
    """Tests for heuristic quality prediction."""
    
    def test_small_volume_change_bonus(self, quality_engine):
        """Small volume change should improve score."""
        pred_small = quality_engine._heuristic_prediction(
            volume_change_pct=-2.0,
            face_count_change_pct=0.0,
            hausdorff_distance=None,
            detail_preservation=None,
            silhouette_similarity=None,
            escalated=False,
        )
        
        pred_large = quality_engine._heuristic_prediction(
            volume_change_pct=-25.0,
            face_count_change_pct=0.0,
            hausdorff_distance=None,
            detail_preservation=None,
            silhouette_similarity=None,
            escalated=False,
        )
        
        assert pred_small.score > pred_large.score
    
    def test_detail_preservation_bonus(self, quality_engine):
        """High detail preservation should improve score."""
        pred_high = quality_engine._heuristic_prediction(
            volume_change_pct=0.0,
            face_count_change_pct=0.0,
            hausdorff_distance=None,
            detail_preservation=0.95,
            silhouette_similarity=None,
            escalated=False,
        )
        
        pred_low = quality_engine._heuristic_prediction(
            volume_change_pct=0.0,
            face_count_change_pct=0.0,
            hausdorff_distance=None,
            detail_preservation=0.3,
            silhouette_similarity=None,
            escalated=False,
        )
        
        assert pred_high.score > pred_low.score
    
    def test_escalation_penalty(self, quality_engine):
        """Escalation should slightly reduce score."""
        pred_normal = quality_engine._heuristic_prediction(
            volume_change_pct=0.0,
            face_count_change_pct=0.0,
            hausdorff_distance=None,
            detail_preservation=None,
            silhouette_similarity=None,
            escalated=False,
        )
        
        pred_escalated = quality_engine._heuristic_prediction(
            volume_change_pct=0.0,
            face_count_change_pct=0.0,
            hausdorff_distance=None,
            detail_preservation=None,
            silhouette_similarity=None,
            escalated=True,
        )
        
        assert pred_normal.score > pred_escalated.score
