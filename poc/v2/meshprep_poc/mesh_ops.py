# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Mesh loading and diagnostics using real trimesh operations.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np

import trimesh


@dataclass
class MeshDiagnostics:
    """Diagnostic information about a mesh."""
    
    # Basic geometry
    vertex_count: int = 0
    face_count: int = 0
    edge_count: int = 0
    
    # Volume and area
    volume: float = 0.0
    surface_area: float = 0.0
    
    # Bounding box
    bbox_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bbox_max: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bbox_diagonal: float = 0.0
    
    # Quality flags
    is_watertight: bool = False
    is_volume: bool = False  # trimesh's manifold check
    is_winding_consistent: bool = False
    
    # Defect counts
    hole_count: int = 0
    boundary_edge_count: int = 0
    component_count: int = 1
    degenerate_face_count: int = 0
    duplicate_face_count: int = 0
    
    # Computed flags
    has_holes: bool = False
    has_multiple_components: bool = False
    has_degenerate_faces: bool = False
    
    # Euler characteristic
    euler_characteristic: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "edge_count": self.edge_count,
            "volume": self.volume,
            "surface_area": self.surface_area,
            "bbox_diagonal": self.bbox_diagonal,
            "is_watertight": self.is_watertight,
            "is_volume": self.is_volume,
            "is_winding_consistent": self.is_winding_consistent,
            "hole_count": self.hole_count,
            "boundary_edge_count": self.boundary_edge_count,
            "component_count": self.component_count,
            "degenerate_face_count": self.degenerate_face_count,
            "has_holes": self.has_holes,
            "has_multiple_components": self.has_multiple_components,
            "euler_characteristic": self.euler_characteristic,
        }


def load_mesh(path: Path) -> trimesh.Trimesh:
    """
    Load a mesh from file.
    
    Args:
        path: Path to STL/OBJ/PLY file
        
    Returns:
        trimesh.Trimesh object
    """
    mesh = trimesh.load(str(path), force='mesh')
    
    # If it's a Scene, extract the geometry
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 1:
            mesh = list(mesh.geometry.values())[0]
        else:
            # Concatenate all geometries
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    
    return mesh


def save_mesh(mesh: trimesh.Trimesh, path: Path, file_type: str = "stl") -> None:
    """
    Save a mesh to file.
    
    Args:
        mesh: The mesh to save
        path: Output file path
        file_type: File format (stl, obj, ply, etc.)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path), file_type=file_type)


def compute_diagnostics(mesh: trimesh.Trimesh) -> MeshDiagnostics:
    """
    Compute comprehensive diagnostics for a mesh.
    
    Args:
        mesh: The mesh to analyze
        
    Returns:
        MeshDiagnostics with all computed values
    """
    diag = MeshDiagnostics()
    
    # Basic geometry
    diag.vertex_count = len(mesh.vertices)
    diag.face_count = len(mesh.faces)
    diag.edge_count = len(mesh.edges_unique)
    
    # Volume and area
    try:
        diag.volume = float(mesh.volume) if mesh.is_volume else 0.0
    except Exception:
        diag.volume = 0.0
    
    diag.surface_area = float(mesh.area)
    
    # Bounding box
    if len(mesh.vertices) > 0:
        bounds = mesh.bounds
        diag.bbox_min = tuple(bounds[0])
        diag.bbox_max = tuple(bounds[1])
        diag.bbox_diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))
    
    # Quality flags
    diag.is_watertight = mesh.is_watertight
    diag.is_volume = mesh.is_volume
    diag.is_winding_consistent = mesh.is_winding_consistent
    
    # Holes / boundary edges
    # In trimesh, boundary edges are edges that belong to only one face
    try:
        # Get edges that are only used by one face (boundary edges)
        edges_face_count = mesh.edges_face
        boundary_mask = edges_face_count == 1
        diag.boundary_edge_count = int(np.sum(boundary_mask))
        
        # Estimate hole count from boundary loops
        # This is approximate - a proper implementation would trace boundary loops
        if diag.boundary_edge_count > 0:
            # Simple heuristic: assume each hole has ~10 edges on average
            diag.hole_count = max(1, diag.boundary_edge_count // 10)
        else:
            diag.hole_count = 0
    except Exception:
        diag.boundary_edge_count = 0
        diag.hole_count = 0
    
    diag.has_holes = diag.boundary_edge_count > 0 or not mesh.is_watertight
    
    # Components
    try:
        components = mesh.split(only_watertight=False)
        diag.component_count = len(components)
    except Exception:
        diag.component_count = 1
    
    diag.has_multiple_components = diag.component_count > 1
    
    # Degenerate faces (zero area)
    try:
        face_areas = mesh.area_faces
        diag.degenerate_face_count = int(np.sum(face_areas < 1e-10))
    except Exception:
        diag.degenerate_face_count = 0
    
    diag.has_degenerate_faces = diag.degenerate_face_count > 0
    
    # Duplicate faces
    try:
        # Check for duplicate face indices
        face_set = set(map(tuple, np.sort(mesh.faces, axis=1)))
        diag.duplicate_face_count = len(mesh.faces) - len(face_set)
    except Exception:
        diag.duplicate_face_count = 0
    
    # Euler characteristic: V - E + F
    diag.euler_characteristic = (
        diag.vertex_count - diag.edge_count + diag.face_count
    )
    
    return diag


def print_diagnostics(diag: MeshDiagnostics, title: str = "Mesh Diagnostics") -> None:
    """Print diagnostics in a readable format."""
    print(f"\n{title}")
    print("=" * 50)
    print(f"Vertices: {diag.vertex_count:,}")
    print(f"Faces: {diag.face_count:,}")
    print(f"Edges: {diag.edge_count:,}")
    print(f"Volume: {diag.volume:.4f}")
    print(f"Surface Area: {diag.surface_area:.4f}")
    print(f"Bbox Diagonal: {diag.bbox_diagonal:.4f}")
    print()
    print(f"Watertight: {diag.is_watertight}")
    print(f"Is Volume (Manifold): {diag.is_volume}")
    print(f"Winding Consistent: {diag.is_winding_consistent}")
    print()
    print(f"Boundary Edges: {diag.boundary_edge_count}")
    print(f"Hole Count (est): {diag.hole_count}")
    print(f"Components: {diag.component_count}")
    print(f"Degenerate Faces: {diag.degenerate_face_count}")
    print(f"Euler Characteristic: {diag.euler_characteristic}")
    print("=" * 50)
