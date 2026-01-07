# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""End-to-end integration tests with complete system."""

import pytest
from pathlib import Path
from meshprep.core import Mesh, RepairEngine, Validator, Pipeline
from meshprep.learning import HistoryTracker, StrategyLearner


class TestCompleteRepairWorkflow:
    """Test complete repair workflows from load to save."""
    
    def test_load_repair_validate_save(self, test_meshes_dir, tmp_path):
        """Complete workflow: load → repair → validate → save."""
        # Load broken mesh
        broken_path = test_meshes_dir / "broken_holes.stl"
        mesh = Mesh.load(broken_path)
        
        # Verify it's broken
        assert mesh.metadata.is_watertight == False
        
        # Repair with pipeline
        pipeline = Pipeline(
            name="repair-workflow",
            actions=[
                {"name": "remove_duplicates"},
                {"name": "fix_normals"},
                {"name": "fill_holes"},
                {"name": "make_watertight"},
            ]
        )
        
        result = pipeline.execute(mesh)
        assert result.success == True
        
        # Validate repaired
        validator = Validator()
        validation = validator.validate_geometry(result.mesh)
        assert validation.is_watertight == True
        assert validation.is_printable == True or len(validation.issues) < 3
        
        # Save result
        output_path = tmp_path / "repaired.stl"
        result.mesh.trimesh.export(str(output_path))
        assert output_path.exists()
        
        # Verify saved mesh is loadable
        repaired = Mesh.load(output_path)
        assert repaired.metadata.face_count > 0
        assert repaired.metadata.is_watertight == True


class TestRepairEngineIntegration:
    """Test RepairEngine with learning system."""
    
    def test_repair_engine_with_tracking(self, test_meshes_dir, tmp_path):
        """RepairEngine with history tracking."""
        db_path = tmp_path / "engine_test.db"
        tracker = HistoryTracker(db_path)
        engine = RepairEngine(tracker=tracker)
        
        # Repair mesh
        broken_path = test_meshes_dir / "broken_holes.stl"
        result = engine.repair(broken_path)
        
        assert result.success == True
        
        # Verify tracked
        stats = tracker.get_all_pipeline_stats()
        assert len(stats) > 0
    
    def test_repair_engine_progressive_strategies(self, test_meshes_dir, tmp_path):
        """RepairEngine tries multiple strategies."""
        tracker = HistoryTracker(tmp_path / "progressive.db")
        engine = RepairEngine(tracker=tracker)
        
        # Create pipelines
        pipelines = [
            Pipeline("light", [
                {"name": "remove_duplicates"},
                {"name": "fix_normals"},
            ]),
            Pipeline("medium", [
                {"name": "pymeshfix_clean"},
                {"name": "fill_holes"},
            ]),
            Pipeline("aggressive", [
                {"name": "pymeshfix_repair"},
                {"name": "make_watertight"},
            ]),
        ]
        
        # Try each pipeline
        mesh_path = test_meshes_dir / "broken_holes.stl"
        mesh = Mesh.load(mesh_path)
        
        for pipeline in pipelines:
            result = pipeline.execute(mesh)
            
            if result.success:
                validator = Validator()
                validation = validator.validate_geometry(result.mesh)
                
                # Record result
                fingerprint = tracker.compute_mesh_fingerprint(mesh)
                tracker.record_repair(
                    mesh_fingerprint=fingerprint,
                    pipeline_name=pipeline.name,
                    success=result.success,
                    quality_score=validation.quality_score,
                    duration_ms=result.duration_ms
                )
                
                if validation.is_printable:
                    break  # Success!
        
        # Verify at least one strategy worked
        stats = tracker.get_all_pipeline_stats()
        assert any(s["successes"] > 0 for s in stats)


class TestLearningImprovement:
    """Test learning system improves over time."""
    
    def test_learning_improves_recommendations(self, test_meshes_dir, tmp_path):
        """Learning system improves recommendations over time."""
        db_path = tmp_path / "learning_improvement.db"
        tracker = HistoryTracker(db_path)
        learner = StrategyLearner(tracker)
        
        # Simulate repairs on multiple meshes
        test_files = [
            "broken_holes.stl",
            "broken_fragments.stl",
            "broken_normals.stl",
        ]
        
        pipelines = [
            Pipeline("cleanup", [
                {"name": "remove_duplicates"},
                {"name": "fix_normals"},
                {"name": "fill_holes"},
            ]),
            Pipeline("aggressive", [
                {"name": "pymeshfix_repair"},
                {"name": "make_watertight"},
            ]),
        ]
        
        # Run repairs and track results
        for mesh_file in test_files:
            mesh_path = test_meshes_dir / mesh_file
            mesh = Mesh.load(mesh_path)
            fingerprint = tracker.compute_mesh_fingerprint(mesh)
            
            for pipeline in pipelines:
                result = pipeline.execute(mesh)
                
                validator = Validator()
                validation = validator.validate_geometry(result.mesh)
                
                tracker.record_repair(
                    mesh_fingerprint=fingerprint,
                    pipeline_name=pipeline.name,
                    success=result.success and validation.is_watertight,
                    quality_score=validation.quality_score,
                    duration_ms=result.duration_ms
                )
        
        # Get statistics
        summary = learner.get_statistics_summary()
        assert summary["total_attempts"] >= len(test_files) * len(pipelines)
        
        # Get recommendations
        recommendations = learner.recommend_pipelines(top_k=2, min_attempts=2)
        assert len(recommendations) > 0
        
        # Best pipeline should have high score
        best_pipeline, best_score = recommendations[0]
        assert best_score > 0.3  # Reasonable threshold


class TestEndToEndWithAllComponents:
    """Test complete system integration."""
    
    def test_complete_system_workflow(self, test_meshes_dir, tmp_path):
        """Test every component working together."""
        # Setup
        db_path = tmp_path / "complete_system.db"
        tracker = HistoryTracker(db_path)
        learner = StrategyLearner(tracker)
        validator = Validator()
        
        # Load mesh
        mesh_path = test_meshes_dir / "broken_holes.stl"
        mesh = Mesh.load(mesh_path)
        
        # Validate before
        validation_before = validator.validate_geometry(mesh)
        assert validation_before.is_printable == False
        
        # Repair with best available pipeline
        pipeline = Pipeline(
            name="best-effort",
            actions=[
                {"name": "validate"},  # Checkpoint
                {"name": "remove_duplicates"},
                {"name": "fix_normals"},
                {"name": "fill_holes"},
                {"name": "make_watertight"},
                {"name": "validate"},  # Final check
            ]
        )
        
        result = pipeline.execute(mesh)
        assert result.success == True
        
        # Validate after
        validation_after = validator.validate_geometry(result.mesh)
        assert validation_after.is_printable == True
        
        # Record in learning system
        fingerprint = tracker.compute_mesh_fingerprint(mesh)
        tracker.record_repair(
            mesh_fingerprint=fingerprint,
            pipeline_name=pipeline.name,
            success=result.success,
            vertex_count=result.mesh.metadata.vertex_count,
            face_count=result.mesh.metadata.face_count,
            quality_score=validation_after.quality_score,
            duration_ms=result.duration_ms
        )
        
        # Verify learning system updated
        stats = tracker.get_pipeline_stats(pipeline.name)
        assert stats is not None
        assert stats["total_attempts"] >= 1
        
        # Get recommendations
        recommendations = learner.recommend_pipelines(top_k=3, min_attempts=1)
        assert len(recommendations) > 0
        
        # Save repaired mesh
        output_path = tmp_path / "final_repaired.stl"
        result.mesh.trimesh.export(str(output_path))
        
        # Verify saved and reloadable
        final_mesh = Mesh.load(output_path)
        final_validation = validator.validate_geometry(final_mesh)
        assert final_validation.is_printable == True
        
        print("\n✓ Complete system test passed!")
        print(f"  Repaired: {mesh_path.name}")
        print(f"  Quality: {validation_after.quality_score:.2f}/5")
        print(f"  Duration: {result.duration_ms:.0f}ms")
        print(f"  Output: {output_path}")
