# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Mesh loading, saving, and diagnostics using trimesh.

This module provides the core mesh I/O operations and diagnostic
computations used throughout MeshPrep.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union
import hashlib
import logging

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


@dataclass
class MeshDiagnostics:
    """
    Comprehensive diagnostic information about a mesh.
    
    This dataclass captures all the metrics needed for:
    - Profile detection (see profiles.py)
    - Validation checks (see validation.py)
    - Reporting
    
    Attributes:
        vertex_count: Number of vertices in the mesh
        face_count: Number of faces (triangles)
        edge_count: Number of unique edges
        volume: Mesh volume (0 if not watertight)
        surface_area: Total surface area
        bbox_min: Minimum corner of bounding box (x, y, z)
        bbox_max: Maximum corner of bounding box (x, y, z)
        bbox_diagonal: Length of bounding box diagonal
        is_watertight: True if mesh is closed (no holes)
        is_volume: True if mesh is a valid volume (manifold)
        is_winding_consistent: True if all face normals point consistently
        hole_count: Estimated number of holes
        boundary_edge_count: Number of edges on mesh boundary
        component_count: Number of disconnected components
        degenerate_face_count: Number of zero-area faces
        duplicate_face_count: Number of duplicate faces
        has_holes: Convenience flag for hole_count > 0
        has_multiple_components: Convenience flag for component_count > 1
        has_degenerate_faces: Convenience flag for degenerate_face_count > 0
        euler_characteristic: V - E + F (2 for closed sphere)
    """
    
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
    is_volume: bool = False
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
        """Convert to dictionary for JSON serialization."""
        return {
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "edge_count": self.edge_count,
            "volume": self.volume,
            "surface_area": self.surface_area,
            "bbox_min": list(self.bbox_min),
            "bbox_max": list(self.bbox_max),
            "bbox_diagonal": self.bbox_diagonal,
            "is_watertight": self.is_watertight,
            "is_volume": self.is_volume,
            "is_winding_consistent": self.is_winding_consistent,
            "hole_count": self.hole_count,
            "boundary_edge_count": self.boundary_edge_count,
            "component_count": self.component_count,
            "degenerate_face_count": self.degenerate_face_count,
            "duplicate_face_count": self.duplicate_face_count,
            "has_holes": self.has_holes,
            "has_multiple_components": self.has_multiple_components,
            "has_degenerate_faces": self.has_degenerate_faces,
            "euler_characteristic": self.euler_characteristic,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "MeshDiagnostics":
        """Create from dictionary."""
        diag = cls()
        for key, value in data.items():
            if hasattr(diag, key):
                if key in ("bbox_min", "bbox_max") and isinstance(value, list):
                    value = tuple(value)
                setattr(diag, key, value)
        return diag


def load_mesh(path: Union[str, Path]) -> trimesh.Trimesh:
    """
    Load a mesh from file.
    
    Supports STL (ASCII and binary), OBJ, PLY, and other formats
    supported by trimesh.
    
    Args:
        path: Path to mesh file
        
    Returns:
        trimesh.Trimesh object
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file cannot be loaded as a mesh
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")
    
    logger.info(f"Loading mesh from: {path}")
    
    try:
        mesh = trimesh.load(str(path), force='mesh')
    except Exception as e:
        raise ValueError(f"Failed to load mesh: {e}") from e
    
    # If it's a Scene (multiple objects), concatenate them
    if isinstance(mesh, trimesh.Scene):
        geometries = list(mesh.geometry.values())
        if len(geometries) == 0:
            raise ValueError("No geometry found in file")
        elif len(geometries) == 1:
            mesh = geometries[0]
        else:
            logger.info(f"Concatenating {len(geometries)} geometries from scene")
            mesh = trimesh.util.concatenate(geometries)
    
    logger.info(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    return mesh


def save_mesh(
    mesh: trimesh.Trimesh, 
    path: Union[str, Path], 
    file_type: str = "stl",
    ascii_format: bool = False
) -> None:
    """
    Save a mesh to file.
    
    Args:
        mesh: The mesh to save
        path: Output file path
        file_type: File format (stl, obj, ply, etc.)
        ascii_format: For STL, use ASCII format instead of binary
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving mesh to: {path}")
    
    if file_type.lower() == "stl" and ascii_format:
        mesh.export(str(path), file_type="stl_ascii")
    else:
        mesh.export(str(path), file_type=file_type)


def compute_fingerprint(mesh: trimesh.Trimesh) -> str:
    """
    Compute a SHA256 fingerprint of the mesh geometry.
    
    This can be used to identify meshes for reproducibility
    and preset matching.
    
    Args:
        mesh: The mesh to fingerprint
        
    Returns:
        SHA256 hex digest string
    """
    # Combine vertex and face data
    data = mesh.vertices.tobytes() + mesh.faces.tobytes()
    return hashlib.sha256(data).hexdigest()


def compute_diagnostics(mesh: trimesh.Trimesh) -> MeshDiagnostics:
    """
    Compute comprehensive diagnostics for a mesh.
    
    This is the primary function for analyzing mesh quality
    before and after repair operations.
    
    Args:
        mesh: The mesh to analyze
        
    Returns:
        MeshDiagnostics with all computed values
    """
    diag = MeshDiagnostics()
    
    # Basic geometry
    diag.vertex_count = len(mesh.vertices)
    diag.face_count = len(mesh.faces)
    
    try:
        diag.edge_count = len(mesh.edges_unique)
    except Exception:
        diag.edge_count = 0
    
    # Volume and area
    try:
        diag.volume = float(mesh.volume) if mesh.is_volume else 0.0
    except Exception:
        diag.volume = 0.0
    
    try:
        diag.surface_area = float(mesh.area)
    except Exception:
        diag.surface_area = 0.0
    
    # Bounding box
    if len(mesh.vertices) > 0:
        try:
            bounds = mesh.bounds
            diag.bbox_min = tuple(float(x) for x in bounds[0])
            diag.bbox_max = tuple(float(x) for x in bounds[1])
            diag.bbox_diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))
        except Exception:
            pass
    
    # Quality flags
    try:
        diag.is_watertight = bool(mesh.is_watertight)
    except Exception:
        diag.is_watertight = False
    
    try:
        diag.is_volume = bool(mesh.is_volume)
    except Exception:
        diag.is_volume = False
    
    try:
        diag.is_winding_consistent = bool(mesh.is_winding_consistent)
    except Exception:
        diag.is_winding_consistent = False
    
    # Boundary edges (indicate holes)
    try:
        edges_face_count = mesh.edges_face
        boundary_mask = edges_face_count == 1
        diag.boundary_edge_count = int(np.sum(boundary_mask))
        
        # Estimate hole count (heuristic)
        if diag.boundary_edge_count > 0:
            diag.hole_count = max(1, diag.boundary_edge_count // 10)
        else:
            diag.hole_count = 0
    except Exception:
        diag.boundary_edge_count = 0
        diag.hole_count = 0
    
    diag.has_holes = diag.boundary_edge_count > 0 or not diag.is_watertight
    
    # Connected components
    try:
        components = mesh.split(only_watertight=False)
        diag.component_count = len(components) if components else 1
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
        face_set = set(map(tuple, np.sort(mesh.faces, axis=1)))
        diag.duplicate_face_count = len(mesh.faces) - len(face_set)
    except Exception:
        diag.duplicate_face_count = 0
    
    # Euler characteristic: V - E + F
    diag.euler_characteristic = (
        diag.vertex_count - diag.edge_count + diag.face_count
    )
    
    return diag


def format_diagnostics(diag: MeshDiagnostics, title: str = "Mesh Diagnostics") -> str:
    """
    Format diagnostics as a human-readable string.
    
    Args:
        diag: The diagnostics to format
        title: Title for the output
        
    Returns:
        Formatted string
    """
    lines = [
        f"\n{title}",
        "=" * 50,
        f"Vertices: {diag.vertex_count:,}",
        f"Faces: {diag.face_count:,}",
        f"Edges: {diag.edge_count:,}",
        f"Volume: {diag.volume:.4f}",
        f"Surface Area: {diag.surface_area:.4f}",
        f"Bbox Diagonal: {diag.bbox_diagonal:.4f}",
        "",
        f"Watertight: {diag.is_watertight}",
        f"Is Volume (Manifold): {diag.is_volume}",
        f"Winding Consistent: {diag.is_winding_consistent}",
        "",
        f"Boundary Edges: {diag.boundary_edge_count}",
        f"Hole Count (est): {diag.hole_count}",
        f"Components: {diag.component_count}",
        f"Degenerate Faces: {diag.degenerate_face_count}",
        f"Euler Characteristic: {diag.euler_characteristic}",
        "=" * 50,
    ]
    return "\n".join(lines)


def print_diagnostics(diag: MeshDiagnostics, title: str = "Mesh Diagnostics") -> None:
    """Print diagnostics in a readable format."""
    print(format_diagnostics(diag, title))
