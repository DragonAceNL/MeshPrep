# Copyright 2025 Dragon Ace
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""Diagnostics computation for mesh analysis."""

from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path

from .mock_mesh import MockMesh


@dataclass
class Diagnostics:
    """Complete diagnostics vector for a mesh."""
    
    # Basic stats
    vertex_count: int = 0
    face_count: int = 0
    
    # Validation state
    is_watertight: bool = False
    hole_count: int = 0
    component_count: int = 1
    largest_component_pct: float = 1.0
    non_manifold_edge_count: int = 0
    non_manifold_vertex_count: int = 0
    degenerate_face_count: int = 0
    normal_consistency: float = 1.0
    
    # Bounding box
    bbox_x: float = 0.0
    bbox_y: float = 0.0
    bbox_z: float = 0.0
    bbox_volume: float = 0.0
    
    # Computed metrics
    volume: float = 0.0
    avg_edge_length: float = 0.0
    triangle_density: float = 0.0
    duplicate_vertex_ratio: float = 0.0
    aspect_ratio: float = 1.0
    
    # Self-intersection
    self_intersections: bool = False
    self_intersection_count: int = 0
    
    # Advanced
    estimated_min_thickness: float = 0.0
    genus: int = 0
    nested_shell_count: int = 0
    overhang_face_ratio: float = 0.0
    
    # Euler characteristic: V - E + F = 2 - 2g for closed surface
    euler_characteristic: int = 2
    
    def to_dict(self) -> dict:
        """Convert diagnostics to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "Diagnostics":
        """Create diagnostics from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=== Mesh Diagnostics ===",
            f"Vertices: {self.vertex_count:,}",
            f"Faces: {self.face_count:,}",
            f"",
            f"Watertight: {'✓' if self.is_watertight else '✗'} ({self.hole_count} holes)",
            f"Components: {self.component_count} (largest: {self.largest_component_pct:.1%})",
            f"Non-manifold: {self.non_manifold_edge_count} edges, {self.non_manifold_vertex_count} vertices",
            f"Degenerate faces: {self.degenerate_face_count}",
            f"Normal consistency: {self.normal_consistency:.1%}",
            f"Self-intersections: {'Yes' if self.self_intersections else 'No'} ({self.self_intersection_count})",
            f"",
            f"Bounding box: {self.bbox_x:.1f} × {self.bbox_y:.1f} × {self.bbox_z:.1f}",
            f"Volume: {self.volume:.1f}",
            f"Min thickness: {self.estimated_min_thickness:.2f}",
            f"Genus: {self.genus}",
        ]
        return "\n".join(lines)
    
    def is_printable(self) -> bool:
        """Check if mesh is likely printable."""
        return (
            self.is_watertight
            and self.non_manifold_edge_count == 0
            and self.non_manifold_vertex_count == 0
            and self.component_count == 1
            and not self.self_intersections
        )
    
    def issues(self) -> list[str]:
        """List all detected issues."""
        problems = []
        if not self.is_watertight:
            problems.append(f"Not watertight ({self.hole_count} holes)")
        if self.non_manifold_edge_count > 0:
            problems.append(f"Non-manifold edges ({self.non_manifold_edge_count})")
        if self.non_manifold_vertex_count > 0:
            problems.append(f"Non-manifold vertices ({self.non_manifold_vertex_count})")
        if self.degenerate_face_count > 0:
            problems.append(f"Degenerate faces ({self.degenerate_face_count})")
        if self.normal_consistency < 0.9:
            problems.append(f"Inconsistent normals ({self.normal_consistency:.1%})")
        if self.self_intersections:
            problems.append(f"Self-intersections ({self.self_intersection_count})")
        if self.component_count > 1:
            problems.append(f"Multiple components ({self.component_count})")
        if self.estimated_min_thickness < 0.8:
            problems.append(f"Thin walls ({self.estimated_min_thickness:.2f}mm)")
        return problems


def compute_diagnostics(mesh: MockMesh) -> Diagnostics:
    """
    Compute full diagnostics vector for a mesh.
    
    Args:
        mesh: The MockMesh to analyze.
        
    Returns:
        Diagnostics object with all computed values.
    """
    bbox = mesh.bbox
    
    return Diagnostics(
        vertex_count=mesh.vertex_count,
        face_count=mesh.face_count,
        is_watertight=mesh.is_watertight,
        hole_count=mesh.hole_count,
        component_count=mesh.component_count,
        largest_component_pct=mesh.largest_component_pct,
        non_manifold_edge_count=mesh.non_manifold_edge_count,
        non_manifold_vertex_count=mesh.non_manifold_vertex_count,
        degenerate_face_count=mesh.degenerate_face_count,
        normal_consistency=mesh.normal_consistency,
        bbox_x=bbox[0],
        bbox_y=bbox[1],
        bbox_z=bbox[2],
        bbox_volume=mesh.bbox_volume,
        volume=mesh.volume,
        avg_edge_length=mesh.avg_edge_length,
        triangle_density=mesh.triangle_density,
        duplicate_vertex_ratio=mesh.duplicate_vertex_ratio,
        aspect_ratio=mesh.aspect_ratio,
        self_intersections=mesh.self_intersections,
        self_intersection_count=mesh.self_intersection_count,
        estimated_min_thickness=mesh.estimated_min_thickness,
        genus=mesh.genus,
        nested_shell_count=mesh.nested_shell_count,
        overhang_face_ratio=mesh.overhang_face_ratio,
        euler_characteristic=mesh.vertex_count - (mesh.face_count * 3 // 2) + mesh.face_count,
    )
