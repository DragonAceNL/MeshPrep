# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Test core classes with real meshes - no mocking."""

import pytest
from meshprep.core import Mesh, Pipeline, Validator, ActionRegistry


class TestMeshWithRealData:
    """Test Mesh class with real mesh files."""
    
    def test_load_valid_mesh(self, valid_mesh):
        """Load valid mesh and check properties."""
        assert valid_mesh.metadata.vertex_count > 0
        assert valid_mesh.metadata.face_count == 12  # Box has 12 triangular faces
        assert valid_mesh.metadata.is_watertight == True
        assert valid_mesh.metadata.volume > 0
    
    def test_load_broken_mesh(self, holed_mesh):
        """Load broken mesh and detect issues."""
        assert holed_mesh.metadata.face_count < 12  # Some faces removed
        assert holed_mesh.metadata.is_watertight == False  # Has holes
    
    def test_mesh_copy_preserves_metadata(self, valid_mesh):
        """Copy preserves all metadata."""
        copy = valid_mesh.copy()
        
        assert copy.metadata.vertex_count == valid_mesh.metadata.vertex_count
        assert copy.metadata.face_count == valid_mesh.metadata.face_count
        assert copy.metadata.is_watertight == valid_mesh.metadata.is_watertight
    
    def test_mesh_sample_points(self, valid_mesh):
        """Sample points with normals."""
        sample = valid_mesh.sample_points(n_points=1000, include_normals=True)
        
        assert "points" in sample
        assert "normals" in sample
        assert sample["points"].shape == (1000, 3)
        assert sample["normals"].shape == (1000, 3)
    
    def test_fragmented_mesh_components(self, fragmented_mesh):
        """Detect multiple components."""
        # Should have 3 disconnected components
        assert fragmented_mesh.metadata.body_count == 3


class TestValidatorWithRealMeshes:
    """Test Validator with real meshes."""
    
    def test_validate_valid_mesh(self, valid_mesh):
        """Valid mesh passes validation."""
        validator = Validator()
        result = validator.validate_geometry(valid_mesh)
        
        assert result.is_watertight == True
        assert result.is_manifold == True
        assert result.is_printable == True
        assert len(result.issues) == 0
    
    def test_validate_broken_mesh(self, holed_mesh):
        """Broken mesh fails validation."""
        validator = Validator()
        result = validator.validate_geometry(holed_mesh)
        
        assert result.is_watertight == False
        assert result.is_printable == False
        assert len(result.issues) > 0
        
        # Check issues are meaningful
        issues_str = " ".join(result.issues).lower()
        assert "watertight" in issues_str or "hole" in issues_str


class TestPipelineWithRealMeshes:
    """Test Pipeline execution with real meshes."""
    
    def test_pipeline_execution(self, valid_mesh):
        """Execute pipeline on real mesh."""
        pipeline = Pipeline(
            name="test-pipeline",
            actions=[
                {"name": "remove_duplicates"},
                {"name": "fix_normals"},
                {"name": "validate"},
            ]
        )
        
        result = pipeline.execute(valid_mesh)
        
        assert result.success == True
        assert result.actions_executed == 3
        assert result.mesh is not None
        assert result.duration_ms > 0
    
    def test_pipeline_stops_on_failure(self, valid_mesh):
        """Pipeline stops when action fails."""
        # Create pipeline with an action that will fail
        pipeline = Pipeline(
            name="failing-pipeline",
            actions=[
                {"name": "fix_normals"},
                {"name": "nonexistent_action"},  # This will fail
                {"name": "validate"},
            ]
        )
        
        result = pipeline.execute(valid_mesh, stop_on_failure=True)
        
        assert result.success == False
        assert result.actions_executed < 3  # Stopped before last action
    
    def test_pipeline_continues_on_failure(self, valid_mesh):
        """Pipeline continues when stop_on_failure=False."""
        pipeline = Pipeline(
            name="continue-pipeline",
            actions=[
                {"name": "fix_normals"},
                {"name": "validate"},
            ]
        )
        
        result = pipeline.execute(valid_mesh, stop_on_failure=False)
        
        # Even if one action fails, should try all
        assert result.actions_executed == 2


class TestActionRegistryWithRealActions:
    """Test ActionRegistry with real actions."""
    
    def test_list_all_actions(self):
        """List returns all registered actions."""
        from meshprep.actions import trimesh, pymeshfix, blender, open3d, core
        
        actions = ActionRegistry.list_actions()
        
        assert len(actions) >= 20
        assert "fix_normals" in actions
        assert "fill_holes" in actions
        assert "pymeshfix_repair" in actions
    
    def test_execute_real_action(self, valid_mesh):
        """Execute real action via registry."""
        result = ActionRegistry.execute("fix_normals", valid_mesh)
        
        assert result.success == True
        assert result.mesh is not None
        assert result.duration_ms >= 0
    
    def test_execute_with_params(self, high_poly_mesh):
        """Execute action with parameters."""
        original_faces = high_poly_mesh.metadata.face_count
        
        result = ActionRegistry.execute(
            "decimate",
            high_poly_mesh,
            {"face_count": 100}
        )
        
        assert result.success == True
        assert result.mesh.metadata.face_count <= 100
        assert result.mesh.metadata.face_count < original_faces
