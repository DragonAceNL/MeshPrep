# Copyright 2025 Dragon Ace
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""Mock mesh implementation for POC - simulates trimesh, pymeshfix, meshio, and Blender."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import random
import hashlib
import time


@dataclass
class MockMesh:
    """Mock mesh object that simulates trimesh.Trimesh behavior."""

    # Geometry stats
    vertex_count: int = 1000
    face_count: int = 2000
    
    # Validation state
    is_watertight: bool = False
    hole_count: int = 5
    component_count: int = 1
    non_manifold_edge_count: int = 3
    non_manifold_vertex_count: int = 2
    degenerate_face_count: int = 10
    normal_consistency: float = 0.75
    self_intersections: bool = True
    self_intersection_count: int = 15
    
    # Dimensions
    bbox_min: tuple = (-50.0, -50.0, 0.0)
    bbox_max: tuple = (50.0, 50.0, 100.0)
    volume: float = 125000.0
    
    # Computed properties
    duplicate_vertex_ratio: float = 0.02
    avg_edge_length: float = 2.5
    triangle_density: float = 0.016
    estimated_min_thickness: float = 1.2
    genus: int = 0
    
    # Additional stats for profiles
    largest_component_pct: float = 0.95
    nested_shell_count: int = 0
    overhang_face_ratio: float = 0.15
    
    # Metadata
    source_path: Optional[Path] = None
    fingerprint: str = ""
    
    # Modification tracking
    modifications: list = field(default_factory=list)
    
    def __post_init__(self):
        """Generate fingerprint if not provided."""
        if not self.fingerprint:
            self.fingerprint = hashlib.sha256(
                f"{self.vertex_count}_{self.face_count}_{time.time()}".encode()
            ).hexdigest()[:16]
    
    @property
    def bbox(self) -> tuple:
        """Return bounding box dimensions."""
        return (
            self.bbox_max[0] - self.bbox_min[0],
            self.bbox_max[1] - self.bbox_min[1],
            self.bbox_max[2] - self.bbox_min[2],
        )
    
    @property
    def bbox_volume(self) -> float:
        """Return bounding box volume."""
        dims = self.bbox
        return dims[0] * dims[1] * dims[2]
    
    @property
    def aspect_ratio(self) -> float:
        """Return aspect ratio (max dimension / min dimension)."""
        dims = self.bbox
        return max(dims) / max(min(dims), 0.001)
    
    def copy(self) -> "MockMesh":
        """Create a copy of the mesh."""
        return MockMesh(
            vertex_count=self.vertex_count,
            face_count=self.face_count,
            is_watertight=self.is_watertight,
            hole_count=self.hole_count,
            component_count=self.component_count,
            non_manifold_edge_count=self.non_manifold_edge_count,
            non_manifold_vertex_count=self.non_manifold_vertex_count,
            degenerate_face_count=self.degenerate_face_count,
            normal_consistency=self.normal_consistency,
            self_intersections=self.self_intersections,
            self_intersection_count=self.self_intersection_count,
            bbox_min=self.bbox_min,
            bbox_max=self.bbox_max,
            volume=self.volume,
            duplicate_vertex_ratio=self.duplicate_vertex_ratio,
            avg_edge_length=self.avg_edge_length,
            triangle_density=self.triangle_density,
            estimated_min_thickness=self.estimated_min_thickness,
            genus=self.genus,
            largest_component_pct=self.largest_component_pct,
            nested_shell_count=self.nested_shell_count,
            overhang_face_ratio=self.overhang_face_ratio,
            source_path=self.source_path,
            fingerprint=self.fingerprint,
            modifications=self.modifications.copy(),
        )


def load_mock_stl(path: Path) -> MockMesh:
    """
    Load a mock STL file.
    
    In POC mode, this generates a random mesh with issues based on filename hints.
    
    Args:
        path: Path to the STL file.
        
    Returns:
        MockMesh with simulated properties.
    """
    path = Path(path)
    filename = path.stem.lower()
    
    # Base mesh
    mesh = MockMesh(source_path=path)
    
    # Simulate different mesh conditions based on filename hints
    if "clean" in filename:
        mesh.is_watertight = True
        mesh.hole_count = 0
        mesh.non_manifold_edge_count = 0
        mesh.non_manifold_vertex_count = 0
        mesh.degenerate_face_count = 0
        mesh.normal_consistency = 1.0
        mesh.self_intersections = False
        mesh.self_intersection_count = 0
    elif "holes" in filename:
        mesh.is_watertight = False
        mesh.hole_count = random.randint(3, 15)
        mesh.non_manifold_edge_count = 0
        mesh.normal_consistency = 0.9
    elif "fragmented" in filename:
        mesh.component_count = random.randint(5, 20)
        mesh.largest_component_pct = random.uniform(0.3, 0.6)
    elif "manifold" in filename or "non-manifold" in filename:
        mesh.non_manifold_edge_count = random.randint(10, 50)
        mesh.non_manifold_vertex_count = random.randint(5, 25)
    elif "normals" in filename:
        mesh.normal_consistency = random.uniform(0.3, 0.6)
    elif "thin" in filename:
        mesh.estimated_min_thickness = random.uniform(0.1, 0.5)
    elif "noisy" in filename or "scan" in filename:
        mesh.triangle_density = random.uniform(0.1, 0.5)
        mesh.degenerate_face_count = random.randint(50, 200)
        mesh.vertex_count = random.randint(50000, 200000)
        mesh.face_count = mesh.vertex_count * 2
    elif "intersect" in filename:
        mesh.self_intersections = True
        mesh.self_intersection_count = random.randint(20, 100)
    elif "hollow" in filename:
        mesh.nested_shell_count = random.randint(1, 3)
    elif "complex" in filename or "genus" in filename:
        mesh.genus = random.randint(5, 20)
    else:
        # Random mesh with various issues
        mesh.is_watertight = random.choice([True, False])
        mesh.hole_count = random.randint(0, 10) if not mesh.is_watertight else 0
        mesh.non_manifold_edge_count = random.randint(0, 20)
        mesh.degenerate_face_count = random.randint(0, 30)
        mesh.normal_consistency = random.uniform(0.6, 1.0)
        mesh.self_intersections = random.choice([True, False])
    
    # Generate fingerprint from path
    mesh.fingerprint = hashlib.sha256(str(path).encode()).hexdigest()[:16]
    
    return mesh


def save_mock_stl(mesh: MockMesh, path: Path, ascii_format: bool = False) -> bool:
    """
    Save a mock STL file.
    
    In POC mode, this writes a simple text file with mesh stats.
    
    Args:
        mesh: The MockMesh to save.
        path: Output path.
        ascii_format: Whether to use ASCII format (ignored in mock).
        
    Returns:
        True if successful.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write a mock STL (just metadata for POC)
    content = f"""solid MockMesh
# POC Mock STL - Not a real STL file
# Source: {mesh.source_path}
# Fingerprint: {mesh.fingerprint}
# Vertices: {mesh.vertex_count}
# Faces: {mesh.face_count}
# Watertight: {mesh.is_watertight}
# Holes: {mesh.hole_count}
# Components: {mesh.component_count}
# Non-manifold edges: {mesh.non_manifold_edge_count}
# Degenerate faces: {mesh.degenerate_face_count}
# Normal consistency: {mesh.normal_consistency:.2f}
# Self-intersections: {mesh.self_intersection_count}
# Modifications: {', '.join(mesh.modifications) if mesh.modifications else 'None'}
endsolid MockMesh
"""
    path.write_text(content)
    return True


# Mock implementations of mesh operations
class MockTrimesh:
    """Mock trimesh module."""
    
    @staticmethod
    def basic_cleanup(mesh: MockMesh) -> MockMesh:
        """Simulate trimesh basic cleanup."""
        mesh = mesh.copy()
        mesh.degenerate_face_count = max(0, mesh.degenerate_face_count - 5)
        mesh.duplicate_vertex_ratio = max(0, mesh.duplicate_vertex_ratio - 0.01)
        mesh.modifications.append("trimesh_basic")
        return mesh
    
    @staticmethod
    def merge_vertices(mesh: MockMesh, eps: float = 1e-8) -> MockMesh:
        """Simulate vertex merging."""
        mesh = mesh.copy()
        mesh.duplicate_vertex_ratio = 0.0
        mesh.vertex_count = int(mesh.vertex_count * 0.95)
        mesh.modifications.append(f"merge_vertices(eps={eps})")
        return mesh
    
    @staticmethod
    def remove_degenerate_faces(mesh: MockMesh) -> MockMesh:
        """Simulate degenerate face removal."""
        mesh = mesh.copy()
        removed = mesh.degenerate_face_count
        mesh.face_count -= removed
        mesh.degenerate_face_count = 0
        mesh.modifications.append("remove_degenerate_faces")
        return mesh
    
    @staticmethod
    def fill_holes(mesh: MockMesh, max_hole_size: int = 1000) -> MockMesh:
        """Simulate hole filling."""
        mesh = mesh.copy()
        filled = min(mesh.hole_count, max_hole_size // 100)
        mesh.hole_count = max(0, mesh.hole_count - filled)
        mesh.face_count += filled * 10  # Add faces for filled holes
        if mesh.hole_count == 0:
            mesh.is_watertight = True
        mesh.modifications.append(f"fill_holes(max_size={max_hole_size})")
        return mesh
    
    @staticmethod
    def recalculate_normals(mesh: MockMesh) -> MockMesh:
        """Simulate normal recalculation."""
        mesh = mesh.copy()
        mesh.normal_consistency = min(1.0, mesh.normal_consistency + 0.2)
        mesh.modifications.append("recalculate_normals")
        return mesh
    
    @staticmethod
    def fix_normals(mesh: MockMesh) -> MockMesh:
        """Simulate full normal fix."""
        mesh = mesh.copy()
        mesh.normal_consistency = 1.0
        mesh.modifications.append("fix_normals")
        return mesh
    
    @staticmethod
    def remove_small_components(mesh: MockMesh, min_faces: int = 100) -> MockMesh:
        """Simulate small component removal."""
        mesh = mesh.copy()
        if mesh.component_count > 1:
            removed = mesh.component_count - 1
            mesh.component_count = 1
            mesh.largest_component_pct = 1.0
            mesh.face_count = int(mesh.face_count * mesh.largest_component_pct)
        mesh.modifications.append(f"remove_small_components(min_faces={min_faces})")
        return mesh
    
    @staticmethod
    def decimate(mesh: MockMesh, target_faces: Optional[int] = None, 
                 target_ratio: float = 0.5) -> MockMesh:
        """Simulate decimation."""
        mesh = mesh.copy()
        if target_faces:
            mesh.face_count = min(mesh.face_count, target_faces)
        else:
            mesh.face_count = int(mesh.face_count * target_ratio)
        mesh.vertex_count = mesh.face_count // 2
        mesh.modifications.append(f"decimate(ratio={target_ratio})")
        return mesh
    
    @staticmethod
    def smooth_laplacian(mesh: MockMesh, iterations: int = 1) -> MockMesh:
        """Simulate Laplacian smoothing."""
        mesh = mesh.copy()
        mesh.triangle_density = max(0.01, mesh.triangle_density * 0.9)
        mesh.modifications.append(f"smooth_laplacian(iter={iterations})")
        return mesh


class MockPyMeshFix:
    """Mock pymeshfix module."""
    
    @staticmethod
    def repair(mesh: MockMesh) -> MockMesh:
        """Simulate pymeshfix repair."""
        mesh = mesh.copy()
        # Fix most issues
        mesh.non_manifold_edge_count = 0
        mesh.non_manifold_vertex_count = 0
        mesh.hole_count = max(0, mesh.hole_count - 3)
        if mesh.hole_count == 0:
            mesh.is_watertight = True
        mesh.self_intersections = False
        mesh.self_intersection_count = 0
        mesh.modifications.append("pymeshfix_repair")
        return mesh


class MockBlender:
    """Mock Blender operations."""
    
    @staticmethod
    def remesh(mesh: MockMesh, voxel_size: float = 0.1) -> MockMesh:
        """Simulate Blender voxel remesh."""
        mesh = mesh.copy()
        # Remesh fixes everything but changes geometry
        mesh.is_watertight = True
        mesh.hole_count = 0
        mesh.non_manifold_edge_count = 0
        mesh.non_manifold_vertex_count = 0
        mesh.degenerate_face_count = 0
        mesh.normal_consistency = 1.0
        mesh.self_intersections = False
        mesh.self_intersection_count = 0
        mesh.component_count = 1
        # Uniform triangle distribution
        mesh.triangle_density = 1.0 / (voxel_size ** 3)
        mesh.modifications.append(f"blender_remesh(voxel={voxel_size})")
        return mesh
    
    @staticmethod
    def boolean_union(mesh: MockMesh) -> MockMesh:
        """Simulate Blender boolean union."""
        mesh = mesh.copy()
        mesh.component_count = 1
        mesh.largest_component_pct = 1.0
        mesh.self_intersections = False
        mesh.self_intersection_count = 0
        mesh.modifications.append("blender_boolean_union")
        return mesh
    
    @staticmethod
    def solidify(mesh: MockMesh, thickness: float = 1.0) -> MockMesh:
        """Simulate Blender solidify."""
        mesh = mesh.copy()
        mesh.estimated_min_thickness = max(mesh.estimated_min_thickness, thickness)
        mesh.modifications.append(f"blender_solidify(thickness={thickness})")
        return mesh
