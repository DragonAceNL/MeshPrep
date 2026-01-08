# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Pytest configuration and fixtures for POC v5 tests - SIMPLIFIED."""

import pytest
import sys
from pathlib import Path
import numpy as np

# Add poc/v5 to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import trimesh
from meshprep.core import Mesh

# NO BOOTSTRAP IN TESTS - assumes dev environment is set up
# If dependencies missing, tests fail fast with clear error


@pytest.fixture(scope="session")
def check_test_dependencies():
    """
    Quick check that test dependencies are available.
    DOES NOT INSTALL - just checks and fails fast with helpful message.
    """
    missing = []
    
    try:
        import pymeshfix
    except ImportError:
        missing.append("pymeshfix")
    
    try:
        import open3d
    except ImportError:
        missing.append("open3d")
    
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    if missing:
        pytest.exit(
            f"\n❌ Missing test dependencies: {', '.join(missing)}\n"
            f"Install with: pip install {' '.join(missing)}\n"
            f"Or: pip install -e \".[all]\"\n"
        )
    
    print("\n✓ All test dependencies available")


@pytest.fixture(scope="session")
def test_meshes_dir(tmp_path_factory):
    """Generate real test meshes with actual geometric issues."""
    meshes_dir = tmp_path_factory.mktemp("test_meshes")
    
    print(f"\nGenerating test meshes in {meshes_dir}")
    
    # 1. Valid cube (baseline)
    cube = trimesh.primitives.Box(extents=[10, 10, 10])
    cube.export(meshes_dir / "valid_cube.stl")
    
    # 2. Mesh with holes (use sphere for better geometry)
    # Spheres have non-coplanar vertices, needed for Poisson/Ball Pivot
    sphere_temp = trimesh.primitives.Sphere(radius=5, subdivisions=3)
    faces_keep = [i for i in range(len(sphere_temp.faces)) if i % 10 != 0]  # Remove ~10% of faces
    holed = trimesh.Trimesh(
        vertices=sphere_temp.vertices,
        faces=sphere_temp.faces[faces_keep]
    )
    holed.export(meshes_dir / "broken_holes.stl")
    
    # 3. Fragmented (3 OVERLAPPING parts for boolean union testing)
    cube1 = trimesh.primitives.Box(extents=[5, 5, 5])
    cube2 = trimesh.primitives.Box(extents=[4, 4, 4])
    cube2.apply_translation([2, 0, 0])  # Overlaps with cube1
    cube3 = trimesh.primitives.Sphere(radius=2)
    cube3.apply_translation([0, 2, 0])  # Overlaps with cube1
    fragmented = trimesh.util.concatenate([cube1, cube2, cube3])
    fragmented.export(meshes_dir / "broken_fragments.stl")
    
    # 4. Inverted normals
    inv_temp = trimesh.primitives.Box(extents=[10, 10, 10])
    inverted = trimesh.Trimesh(
        vertices=inv_temp.vertices,
        faces=inv_temp.faces[:, ::-1]  # Flip winding
    )
    inverted.export(meshes_dir / "broken_normals.stl")
    
    # 5. High poly
    sphere = trimesh.primitives.Sphere(radius=10, subdivisions=4)
    sphere.export(meshes_dir / "high_poly.stl")
    
    # 6. Self-intersecting
    cube_a = trimesh.primitives.Box(extents=[10, 10, 10])
    cube_b = trimesh.primitives.Box(extents=[10, 10, 10])
    cube_b.apply_translation([5, 0, 0])
    intersecting = trimesh.util.concatenate([cube_a, cube_b])
    intersecting.export(meshes_dir / "broken_intersections.stl")
    
    # 7. Thin sheet (plane) - for solidify test
    # Create a simple plane (thin sheet with no thickness)
    thin_sheet = trimesh.creation.box(extents=[10, 10, 0.01])  # Very thin box
    thin_sheet.export(meshes_dir / "thin_walls.stl")
    
    # 8. Non-manifold
    nm_temp = trimesh.primitives.Box(extents=[10, 10, 10])
    extra_verts = np.array([[20, 20, 20], [25, 25, 25]])
    nonmanifold = trimesh.Trimesh(
        vertices=np.vstack([nm_temp.vertices, extra_verts]),
        faces=nm_temp.faces
    )
    nonmanifold.export(meshes_dir / "broken_manifold.stl")
    
    print("✓ Generated 8 test meshes")
    
    return meshes_dir


# Simple fixtures - no complexity
@pytest.fixture
def valid_mesh(test_meshes_dir):
    """Load valid cube mesh."""
    return Mesh.load(test_meshes_dir / "valid_cube.stl")


@pytest.fixture
def holed_mesh(test_meshes_dir):
    """Load mesh with holes."""
    return Mesh.load(test_meshes_dir / "broken_holes.stl")


@pytest.fixture
def fragmented_mesh(test_meshes_dir):
    """Load fragmented mesh."""
    return Mesh.load(test_meshes_dir / "broken_fragments.stl")


@pytest.fixture
def inverted_normals_mesh(test_meshes_dir):
    """Load mesh with inverted normals."""
    return Mesh.load(test_meshes_dir / "broken_normals.stl")


@pytest.fixture
def high_poly_mesh(test_meshes_dir):
    """Load high poly mesh."""
    return Mesh.load(test_meshes_dir / "high_poly.stl")


@pytest.fixture
def intersecting_mesh(test_meshes_dir):
    """Load self-intersecting mesh."""
    return Mesh.load(test_meshes_dir / "broken_intersections.stl")


@pytest.fixture
def thin_mesh(test_meshes_dir):
    """Load thin walls mesh."""
    return Mesh.load(test_meshes_dir / "thin_walls.stl")


@pytest.fixture
def nonmanifold_mesh(test_meshes_dir):
    """Load non-manifold mesh."""
    return Mesh.load(test_meshes_dir / "broken_manifold.stl")


@pytest.fixture
def temp_db(tmp_path):
    """Provide temporary database."""
    return tmp_path / "test_history.db"
