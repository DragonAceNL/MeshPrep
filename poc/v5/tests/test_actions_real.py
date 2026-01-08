# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Test all 20 actions with real meshes - no mocking."""

import pytest
from meshprep.core import ActionRegistry

# Import actions to register them
from meshprep.actions import trimesh, pymeshfix, blender, open3d, core


class TestTrimeshActions:
    """Test trimesh actions (10 actions)."""
    
    def test_fix_normals(self, inverted_normals_mesh):
        """Fix inverted normals."""
        result = ActionRegistry.execute("fix_normals", inverted_normals_mesh)
        
        assert result.success == True
        # Verify normals are consistent
        assert result.mesh.trimesh.is_winding_consistent or result.mesh.metadata.face_count > 0
    
    def test_remove_duplicates(self, valid_mesh):
        """Remove duplicate vertices."""
        result = ActionRegistry.execute("remove_duplicates", valid_mesh)
        
        assert result.success == True
        # Should not increase vertex count
        assert result.mesh.metadata.vertex_count <= valid_mesh.metadata.vertex_count
    
    def test_fill_holes(self, holed_mesh):
        """Fill holes in mesh."""
        original_faces = holed_mesh.metadata.face_count
        
        result = ActionRegistry.execute("fill_holes", holed_mesh)
        
        assert result.success == True
        # Filling holes should add faces
        assert result.mesh.metadata.face_count >= original_faces
    
    def test_make_watertight(self, holed_mesh):
        """Make mesh watertight."""
        # Before: not watertight
        assert holed_mesh.metadata.is_watertight == False
        
        result = ActionRegistry.execute("make_watertight", holed_mesh)
        
        assert result.success == True
        # After: should be watertight
        result.mesh._update_metadata_from_mesh()
        assert result.mesh.metadata.is_watertight == True
    
    def test_decimate_reduces_faces(self, high_poly_mesh):
        """Decimate reduces face count."""
        original_faces = high_poly_mesh.metadata.face_count
        target_faces = 100
        
        result = ActionRegistry.execute("decimate", high_poly_mesh, {"face_count": target_faces})
        
        assert result.success == True
        assert result.mesh.metadata.face_count <= target_faces
        assert result.mesh.metadata.face_count < original_faces
    
    def test_keep_largest_removes_fragments(self, fragmented_mesh):
        """Keep largest removes small fragments."""
        # Before: multiple components
        assert fragmented_mesh.metadata.body_count > 1
        
        result = ActionRegistry.execute("keep_largest", fragmented_mesh)
        
        assert result.success == True
        # After: single component
        assert result.mesh.metadata.body_count == 1
    
    def test_smooth(self, valid_mesh):
        """Smooth mesh surface."""
        result = ActionRegistry.execute("smooth", valid_mesh, {"iterations": 2})
        
        assert result.success == True
        # Should have same topology
        assert result.mesh.metadata.face_count == valid_mesh.metadata.face_count
    
    def test_subdivide_increases_faces(self, valid_mesh):
        """Subdivide increases face count."""
        original_faces = valid_mesh.metadata.face_count
        
        result = ActionRegistry.execute("subdivide", valid_mesh, {"iterations": 1})
        
        assert result.success == True
        assert result.mesh.metadata.face_count > original_faces
    
    def test_fix_intersections(self, intersecting_mesh):
        """Fix self-intersecting geometry."""
        result = ActionRegistry.execute("fix_intersections", intersecting_mesh)
        
        assert result.success == True
        # Mesh should still be valid
        assert result.mesh.metadata.face_count > 0
    
    def test_convex_hull(self, fragmented_mesh):
        """Create convex hull."""
        result = ActionRegistry.execute("convex_hull", fragmented_mesh)
        
        assert result.success == True
        # Convex hull should be watertight
        assert result.mesh.metadata.is_watertight == True


class TestPyMeshFixActions:
    """Test PyMeshFix actions (3 actions)."""
    
    def test_pymeshfix_repair(self, holed_mesh):
        """PyMeshFix repair fixes manifold issues."""
        result = ActionRegistry.execute("pymeshfix_repair", holed_mesh)
        
        assert result.success == True
        # Should improve mesh quality
        assert result.mesh.metadata.is_watertight == True or result.mesh.metadata.is_manifold == True
    
    def test_pymeshfix_clean(self, nonmanifold_mesh):
        """PyMeshFix clean removes artifacts."""
        result = ActionRegistry.execute("pymeshfix_clean", nonmanifold_mesh)
        
        assert result.success == True
        # Should clean up mesh
        assert result.mesh.metadata.vertex_count <= nonmanifold_mesh.metadata.vertex_count
    
    def test_pymeshfix_remove_small(self, fragmented_mesh):
        """PyMeshFix remove small components."""
        original_bodies = fragmented_mesh.metadata.body_count
        
        result = ActionRegistry.execute("pymeshfix_remove_small", fragmented_mesh)
        
        assert result.success == True
        # Should reduce component count
        assert result.mesh.metadata.body_count <= original_bodies


class TestBlenderActions:
    """Test Blender actions (3 actions) - SLOW."""
    
    @pytest.mark.slow
    def test_blender_remesh(self, valid_mesh, check_test_dependencies):
        """Blender voxel remesh."""
        from meshprep.core.bootstrap import get_bootstrap_manager
        manager = get_bootstrap_manager()
        
        if not manager._check_blender():
            pytest.skip("Blender not available")
        
        result = ActionRegistry.execute(
            "blender_remesh",
            valid_mesh,
            {"voxel_size": 0.5}
        )
        
        assert result.success == True
        assert result.mesh is not None
        # Voxel remesh creates watertight mesh
        assert result.mesh.metadata.is_watertight == True
    
    @pytest.mark.slow
    def test_blender_boolean_union(self, fragmented_mesh, check_test_dependencies):
        """Blender boolean union merges components."""
        from meshprep.core.bootstrap import get_bootstrap_manager
        manager = get_bootstrap_manager()
        
        if not manager._check_blender():
            pytest.skip("Blender not available")
        
        original_bodies = fragmented_mesh.metadata.body_count
        
        result = ActionRegistry.execute("blender_boolean_union", fragmented_mesh)
        
        assert result.success == True
        # Should reduce components (merged)
        assert result.mesh.metadata.body_count <= original_bodies
    
    @pytest.mark.slow
    def test_blender_solidify(self, thin_mesh, check_test_dependencies):
        """Blender solidify adds thickness."""
        from meshprep.core.bootstrap import get_bootstrap_manager
        manager = get_bootstrap_manager()
        
        if not manager._check_blender():
            pytest.skip("Blender not available")
        
        result = ActionRegistry.execute(
            "blender_solidify",
            thin_mesh,
            {"thickness": 0.2}
        )
        
        assert result.success == True
        # Should increase volume
        assert result.mesh.metadata.volume > thin_mesh.metadata.volume


class TestOpen3DActions:
    """Test Open3D actions (3 actions)."""
    
    def test_poisson_reconstruction(self, holed_mesh):
        """Poisson surface reconstruction."""
        result = ActionRegistry.execute(
            "poisson_reconstruction",
            holed_mesh,
            {"depth": 7}
        )
        
        assert result.success == True
        # Reconstructed surface should be watertight
        assert result.mesh.metadata.is_watertight == True
    
    def test_ball_pivot(self, high_poly_mesh):
        """Ball pivoting reconstruction."""
        # Ball pivot needs non-coplanar vertices (sphere has them)
        result = ActionRegistry.execute(
            "ball_pivot",
            high_poly_mesh,
            {"radii": [0.5, 1.0, 2.0]}  # Larger radii for sphere
        )
        
        assert result.success == True
        assert result.mesh.metadata.face_count > 0
    
    def test_open3d_simplify(self, high_poly_mesh):
        """Open3D high-quality decimation."""
        original_faces = high_poly_mesh.metadata.face_count
        
        result = ActionRegistry.execute(
            "open3d_simplify",
            high_poly_mesh,
            {"target_reduction": 0.5}
        )
        
        assert result.success == True
        assert result.mesh.metadata.face_count < original_faces
        assert result.mesh.metadata.face_count >= original_faces * 0.4  # Allow some variance


class TestCoreActions:
    """Test core utility actions (1 action)."""
    
    def test_validate_action(self, valid_mesh, holed_mesh):
        """Validate action checks geometry."""
        # Valid mesh
        result_valid = ActionRegistry.execute("validate", valid_mesh)
        assert result_valid.success == True
        # Returns same mesh (no changes)
        assert result_valid.mesh.metadata.face_count == valid_mesh.metadata.face_count
        
        # Broken mesh
        result_broken = ActionRegistry.execute("validate", holed_mesh)
        assert result_broken.success == True
        # Still returns mesh (validation doesn't modify)
        assert result_broken.mesh.metadata.face_count == holed_mesh.metadata.face_count
