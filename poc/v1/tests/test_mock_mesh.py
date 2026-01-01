# Copyright 2025 Dragon Ace
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""Tests for mock mesh operations."""

import pytest
from pathlib import Path

from meshprep.core.mock_mesh import (
    MockMesh, load_mock_stl, save_mock_stl,
    MockTrimesh, MockPyMeshFix, MockBlender,
)


class TestMockMesh:
    """Tests for MockMesh class."""
    
    def test_create_default_mesh(self):
        """Test creating a default mock mesh."""
        mesh = MockMesh()
        assert mesh.vertex_count == 1000
        assert mesh.face_count == 2000
        assert not mesh.is_watertight
        assert mesh.fingerprint != ""
    
    def test_mesh_copy(self):
        """Test copying a mesh."""
        original = MockMesh(vertex_count=500, is_watertight=True)
        copy = original.copy()
        
        assert copy.vertex_count == original.vertex_count
        assert copy.is_watertight == original.is_watertight
        assert copy is not original
    
    def test_bbox_properties(self):
        """Test bounding box computed properties."""
        mesh = MockMesh(
            bbox_min=(0, 0, 0),
            bbox_max=(10, 20, 30),
        )
        
        assert mesh.bbox == (10, 20, 30)
        assert mesh.bbox_volume == 6000
        assert mesh.aspect_ratio == 3.0


class TestLoadSaveMockStl:
    """Tests for STL loading and saving."""
    
    def test_load_clean_model(self, tmp_path):
        """Test loading a model with 'clean' in the name."""
        stl_path = tmp_path / "clean_model.stl"
        stl_path.write_text("solid test\nendsolid test")
        
        mesh = load_mock_stl(stl_path)
        
        assert mesh.is_watertight
        assert mesh.hole_count == 0
        assert mesh.non_manifold_edge_count == 0
    
    def test_load_holes_model(self, tmp_path):
        """Test loading a model with 'holes' in the name."""
        stl_path = tmp_path / "holes_model.stl"
        stl_path.write_text("solid test\nendsolid test")
        
        mesh = load_mock_stl(stl_path)
        
        assert not mesh.is_watertight
        assert mesh.hole_count > 0
    
    def test_save_mock_stl(self, tmp_path):
        """Test saving a mock STL."""
        mesh = MockMesh(vertex_count=100, face_count=200)
        output_path = tmp_path / "output.stl"
        
        result = save_mock_stl(mesh, output_path)
        
        assert result
        assert output_path.exists()
        content = output_path.read_text()
        assert "MockMesh" in content


class TestMockTrimesh:
    """Tests for mock trimesh operations."""
    
    def test_basic_cleanup(self):
        """Test basic cleanup."""
        mesh = MockMesh(degenerate_face_count=10)
        result = MockTrimesh.basic_cleanup(mesh)
        
        assert result.degenerate_face_count < mesh.degenerate_face_count
        assert "trimesh_basic" in result.modifications
    
    def test_merge_vertices(self):
        """Test vertex merging."""
        mesh = MockMesh(duplicate_vertex_ratio=0.05)
        result = MockTrimesh.merge_vertices(mesh, eps=1e-6)
        
        assert result.duplicate_vertex_ratio == 0.0
        assert "merge_vertices" in result.modifications[0]
    
    def test_fill_holes(self):
        """Test hole filling."""
        mesh = MockMesh(is_watertight=False, hole_count=5)
        result = MockTrimesh.fill_holes(mesh, max_hole_size=1000)
        
        assert result.hole_count < mesh.hole_count
        assert "fill_holes" in result.modifications[0]
    
    def test_fix_normals(self):
        """Test normal fixing."""
        mesh = MockMesh(normal_consistency=0.5)
        result = MockTrimesh.fix_normals(mesh)
        
        assert result.normal_consistency == 1.0


class TestMockPyMeshFix:
    """Tests for mock pymeshfix operations."""
    
    def test_repair(self):
        """Test pymeshfix repair."""
        mesh = MockMesh(
            non_manifold_edge_count=10,
            hole_count=3,
            self_intersections=True,
        )
        
        result = MockPyMeshFix.repair(mesh)
        
        assert result.non_manifold_edge_count == 0
        assert not result.self_intersections
        assert "pymeshfix_repair" in result.modifications


class TestMockBlender:
    """Tests for mock Blender operations."""
    
    def test_remesh(self):
        """Test Blender remesh."""
        mesh = MockMesh(
            is_watertight=False,
            non_manifold_edge_count=5,
        )
        
        result = MockBlender.remesh(mesh, voxel_size=0.1)
        
        assert result.is_watertight
        assert result.non_manifold_edge_count == 0
        assert "blender_remesh" in result.modifications[0]
    
    def test_boolean_union(self):
        """Test Blender boolean union."""
        mesh = MockMesh(component_count=3)
        result = MockBlender.boolean_union(mesh)
        
        assert result.component_count == 1
    
    def test_solidify(self):
        """Test Blender solidify."""
        mesh = MockMesh(estimated_min_thickness=0.3)
        result = MockBlender.solidify(mesh, thickness=1.5)
        
        assert result.estimated_min_thickness >= 1.5
