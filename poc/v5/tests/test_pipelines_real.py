# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Test complete repair pipelines with real meshes."""

import pytest
from meshprep.core import Pipeline

# Import actions to register them
from meshprep.actions import trimesh, pymeshfix, blender, open3d, core


class TestRepairPipelines:
    """Test complete repair workflows."""
    
    def test_cleanup_pipeline(self, holed_mesh):
        """Cleanup pipeline fixes common issues."""
        pipeline = Pipeline(
            name="cleanup",
            actions=[
                {"name": "remove_duplicates"},
                {"name": "fix_normals"},
                {"name": "fill_holes"},
                {"name": "make_watertight"},
            ]
        )
        
        result = pipeline.execute(holed_mesh)
        
        assert result.success == True
        assert result.actions_executed == 4
        # Mesh should be improved
        result.mesh._update_metadata_from_mesh()
        assert result.mesh.metadata.is_watertight == True
    
    def test_aggressive_pipeline(self, holed_mesh):
        """Aggressive repair pipeline."""
        pipeline = Pipeline(
            name="aggressive",
            actions=[
                {"name": "pymeshfix_clean"},
                {"name": "fill_holes"},
                {"name": "pymeshfix_repair"},
                {"name": "make_watertight"},
            ]
        )
        
        result = pipeline.execute(holed_mesh)
        
        assert result.success == True
        assert result.mesh.metadata.is_watertight == True
    
    def test_fragment_cleanup_pipeline(self, fragmented_mesh):
        """Pipeline to handle fragmented meshes."""
        pipeline = Pipeline(
            name="defragment",
            actions=[
                {"name": "pymeshfix_remove_small"},
                {"name": "keep_largest"},
                {"name": "pymeshfix_repair"},
            ]
        )
        
        result = pipeline.execute(fragmented_mesh)
        
        assert result.success == True
        # Should have single component now
        assert result.mesh.metadata.body_count == 1
    
    def test_quality_enhancement_pipeline(self, valid_mesh):
        """Pipeline to enhance quality."""
        pipeline = Pipeline(
            name="enhance",
            actions=[
                {"name": "subdivide", "params": {"iterations": 1}},
                {"name": "smooth", "params": {"iterations": 2}},
                {"name": "fix_normals"},
            ]
        )
        
        original_faces = valid_mesh.metadata.face_count
        result = pipeline.execute(valid_mesh)
        
        assert result.success == True
        # Subdivision should have increased faces
        assert result.mesh.metadata.face_count > original_faces
    
    def test_optimization_pipeline(self, high_poly_mesh):
        """Pipeline to optimize mesh."""
        pipeline = Pipeline(
            name="optimize",
            actions=[
                {"name": "remove_duplicates"},
                {"name": "decimate", "params": {"face_count": 200}},
                {"name": "smooth"},
                {"name": "fix_normals"},
            ]
        )
        
        result = pipeline.execute(high_poly_mesh)
        
        assert result.success == True
        assert result.mesh.metadata.face_count <= 200
    
    def test_pipeline_with_validation(self, valid_mesh):
        """Pipeline with validation checkpoints."""
        pipeline = Pipeline(
            name="validated-repair",
            actions=[
                {"name": "validate"},
                {"name": "remove_duplicates"},
                {"name": "fix_normals"},
                {"name": "validate"},
            ]
        )
        
        result = pipeline.execute(valid_mesh)
        
        assert result.success == True
        assert result.actions_executed == 4


class TestReconstructionPipelines:
    """Test reconstruction workflows (Open3D/Blender)."""
    
    def test_poisson_reconstruction_pipeline(self, holed_mesh):
        """Complete Poisson reconstruction workflow."""
        pipeline = Pipeline(
            name="reconstruct-poisson",
            actions=[
                {"name": "poisson_reconstruction", "params": {"depth": 8}},
                {"name": "smooth"},
                {"name": "pymeshfix_repair"},
            ]
        )
        
        result = pipeline.execute(holed_mesh)
        
        assert result.success == True
        assert result.mesh.metadata.is_watertight == True
    
    @pytest.mark.slow
    def test_blender_reconstruction_pipeline(self, fragmented_mesh):
        """Complete Blender-based reconstruction."""
        from meshprep.core.bootstrap import get_bootstrap_manager
        manager = get_bootstrap_manager()
        
        if not manager._check_blender():
            pytest.skip("Blender not available")
        
        pipeline = Pipeline(
            name="reconstruct-blender",
            actions=[
                {"name": "blender_boolean_union"},
                {"name": "blender_remesh", "params": {"voxel_size": 0.3}},
                {"name": "smooth"},
            ]
        )
        
        result = pipeline.execute(fragmented_mesh)
        
        assert result.success == True
        assert result.mesh.metadata.is_watertight == True
        assert result.mesh.metadata.body_count == 1
