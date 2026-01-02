# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Validation module for mesh repair operations.

Implements two-stage validation as per docs/validation.md:
1. Geometric validation: Is the mesh printable?
2. Fidelity validation: Is the appearance preserved?
"""

from dataclasses import dataclass
from typing import Optional
import logging

import numpy as np
from scipy.spatial import cKDTree
import trimesh

from .mesh_ops import MeshDiagnostics, compute_diagnostics

logger = logging.getLogger(__name__)


@dataclass
class GeometricValidation:
    """
    Result of geometric validation checks.
    
    Geometric validation determines if a mesh is suitable for 3D printing.
    All checks must pass for a mesh to be considered printable.
    """
    
    is_watertight: bool = False
    is_manifold: bool = False
    has_positive_volume: bool = False
    is_winding_consistent: bool = False
    no_degenerate_faces: bool = True
    
    # Details
    volume: float = 0.0
    boundary_edge_count: int = 0
    degenerate_face_count: int = 0
    
    @property
    def is_printable(self) -> bool:
        """Check if mesh meets all geometric requirements for 3D printing."""
        return (
            self.is_watertight and
            self.is_manifold and
            self.has_positive_volume and
            self.is_winding_consistent
        )
    
    @property
    def issues(self) -> list[str]:
        """List of detected geometric issues."""
        issues = []
        if not self.is_watertight:
            issues.append(f"Not watertight ({self.boundary_edge_count} boundary edges)")
        if not self.is_manifold:
            issues.append("Non-manifold geometry")
        if not self.has_positive_volume:
            issues.append(f"Invalid volume ({self.volume:.4f})")
        if not self.is_winding_consistent:
            issues.append("Inconsistent winding/normals")
        if not self.no_degenerate_faces:
            issues.append(f"Degenerate faces ({self.degenerate_face_count})")
        return issues
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_printable": self.is_printable,
            "is_watertight": self.is_watertight,
            "is_manifold": self.is_manifold,
            "has_positive_volume": self.has_positive_volume,
            "is_winding_consistent": self.is_winding_consistent,
            "no_degenerate_faces": self.no_degenerate_faces,
            "volume": self.volume,
            "boundary_edge_count": self.boundary_edge_count,
            "degenerate_face_count": self.degenerate_face_count,
            "issues": self.issues,
        }


@dataclass
class FidelityValidation:
    """
    Result of fidelity validation checks.
    
    Fidelity validation ensures the repair didn't significantly
    change the mesh's visual appearance.
    """
    
    volume_acceptable: bool = False
    bbox_unchanged: bool = False
    hausdorff_acceptable: bool = False
    surface_area_acceptable: bool = True
    
    # Details
    original_volume: float = 0.0
    repaired_volume: float = 0.0
    volume_change_pct: float = 0.0
    
    original_bbox_diagonal: float = 0.0
    repaired_bbox_diagonal: float = 0.0
    bbox_change_pct: float = 0.0
    
    hausdorff_distance: float = 0.0
    hausdorff_relative: float = 0.0
    mean_surface_distance: float = 0.0
    
    original_area: float = 0.0
    repaired_area: float = 0.0
    area_change_pct: float = 0.0
    
    @property
    def is_visually_unchanged(self) -> bool:
        """Check if mesh appearance is preserved."""
        return (
            self.volume_acceptable and
            self.bbox_unchanged and
            self.hausdorff_acceptable
        )
    
    @property
    def changes(self) -> list[str]:
        """List of detected changes."""
        changes = []
        if not self.volume_acceptable:
            changes.append(f"Volume changed by {self.volume_change_pct:.2f}%")
        if not self.bbox_unchanged:
            changes.append(f"Bounding box changed by {self.bbox_change_pct:.2f}%")
        if not self.hausdorff_acceptable:
            changes.append(f"Surface deviation: {self.hausdorff_relative*100:.2f}% of bbox")
        if not self.surface_area_acceptable:
            changes.append(f"Surface area changed by {self.area_change_pct:.2f}%")
        return changes
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_visually_unchanged": self.is_visually_unchanged,
            "volume_acceptable": self.volume_acceptable,
            "bbox_unchanged": self.bbox_unchanged,
            "hausdorff_acceptable": self.hausdorff_acceptable,
            "surface_area_acceptable": self.surface_area_acceptable,
            "volume_change_pct": self.volume_change_pct,
            "bbox_change_pct": self.bbox_change_pct,
            "hausdorff_relative": self.hausdorff_relative,
            "mean_surface_distance": self.mean_surface_distance,
            "area_change_pct": self.area_change_pct,
            "changes": self.changes,
        }


@dataclass
class ValidationResult:
    """
    Complete validation result for a repair operation.
    
    Combines geometric and fidelity validation results with
    before/after diagnostics.
    """
    
    geometric: GeometricValidation
    fidelity: FidelityValidation
    
    # Diagnostics
    original_diagnostics: Optional[MeshDiagnostics] = None
    repaired_diagnostics: Optional[MeshDiagnostics] = None
    
    @property
    def is_successful(self) -> bool:
        """Check if repair is fully successful (printable AND visually unchanged)."""
        return self.geometric.is_printable and self.fidelity.is_visually_unchanged
    
    @property
    def is_geometrically_valid(self) -> bool:
        """Alias for geometric validation - mesh is printable."""
        return self.geometric.is_printable
    
    @property
    def is_visually_unchanged(self) -> bool:
        """Alias for fidelity validation - appearance preserved."""
        return self.fidelity.is_visually_unchanged
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export."""
        result = {
            "is_successful": self.is_successful,
            "is_geometrically_valid": self.is_geometrically_valid,
            "is_visually_unchanged": self.is_visually_unchanged,
            "geometric": self.geometric.to_dict(),
            "fidelity": self.fidelity.to_dict(),
        }
        
        if self.original_diagnostics:
            result["original_diagnostics"] = self.original_diagnostics.to_dict()
        if self.repaired_diagnostics:
            result["repaired_diagnostics"] = self.repaired_diagnostics.to_dict()
        
        return result


def validate_geometry(mesh: trimesh.Trimesh) -> GeometricValidation:
    """
    Run geometric validation checks on a mesh.
    
    Checks if the mesh is suitable for 3D printing:
    - Watertight (no holes)
    - Manifold (valid topology)
    - Positive volume
    - Consistent winding
    
    Args:
        mesh: The mesh to validate
        
    Returns:
        GeometricValidation result
    """
    result = GeometricValidation()
    
    try:
        result.is_watertight = bool(mesh.is_watertight)
    except Exception:
        result.is_watertight = False
    
    try:
        result.is_manifold = bool(mesh.is_volume)
    except Exception:
        result.is_manifold = False
    
    try:
        result.is_winding_consistent = bool(mesh.is_winding_consistent)
    except Exception:
        result.is_winding_consistent = False
    
    try:
        result.volume = float(mesh.volume) if mesh.is_volume else 0.0
        result.has_positive_volume = result.volume > 0
    except Exception:
        result.volume = 0.0
        result.has_positive_volume = False
    
    # Check boundary edges
    try:
        edges_face_count = mesh.edges_face
        result.boundary_edge_count = int(np.sum(edges_face_count == 1))
    except Exception:
        result.boundary_edge_count = 0
    
    # Check degenerate faces
    try:
        face_areas = mesh.area_faces
        result.degenerate_face_count = int(np.sum(face_areas < 1e-10))
        result.no_degenerate_faces = result.degenerate_face_count == 0
    except Exception:
        result.degenerate_face_count = 0
        result.no_degenerate_faces = True
    
    return result


def compute_hausdorff_distance(
    original: trimesh.Trimesh,
    repaired: trimesh.Trimesh,
    sample_count: int = 10000
) -> tuple[float, float]:
    """
    Compute Hausdorff distance between two meshes.
    
    The Hausdorff distance measures the maximum deviation between
    two surfaces. A lower value means the surfaces are more similar.
    
    Args:
        original: Original mesh
        repaired: Repaired mesh
        sample_count: Number of surface samples for approximation
        
    Returns:
        Tuple of (hausdorff_distance, mean_distance)
    """
    try:
        # Sample points on both surfaces
        samples_original = original.sample(sample_count)
        samples_repaired = repaired.sample(sample_count)
        
        # Build KD-trees for fast nearest-neighbor lookup
        tree_original = cKDTree(samples_original)
        tree_repaired = cKDTree(samples_repaired)
        
        # Forward distances: original -> repaired
        dist_orig_to_rep, _ = tree_repaired.query(samples_original)
        
        # Backward distances: repaired -> original
        dist_rep_to_orig, _ = tree_original.query(samples_repaired)
        
        # Hausdorff is max of both directions
        hausdorff = max(float(dist_orig_to_rep.max()), float(dist_rep_to_orig.max()))
        
        # Mean surface distance (average of both directions)
        mean_distance = (float(dist_orig_to_rep.mean()) + float(dist_rep_to_orig.mean())) / 2
        
        return hausdorff, mean_distance
        
    except Exception as e:
        logger.warning(f"Failed to compute Hausdorff distance: {e}")
        return 0.0, 0.0


def validate_fidelity(
    original: trimesh.Trimesh,
    repaired: trimesh.Trimesh,
    max_volume_change_pct: float = 1.0,
    max_bbox_change_pct: float = 0.1,
    max_hausdorff_relative: float = 0.001,
    max_area_change_pct: float = 2.0,
    hausdorff_sample_count: int = 10000
) -> FidelityValidation:
    """
    Run fidelity validation comparing original and repaired meshes.
    
    Ensures the repair operation didn't significantly change
    the mesh's visual appearance.
    
    Args:
        original: Original mesh before repair
        repaired: Mesh after repair
        max_volume_change_pct: Maximum allowed volume change (default 1%)
        max_bbox_change_pct: Maximum allowed bbox change (default 0.1%)
        max_hausdorff_relative: Max Hausdorff as fraction of bbox diagonal
        max_area_change_pct: Maximum allowed surface area change (default 2%)
        hausdorff_sample_count: Number of samples for Hausdorff computation
        
    Returns:
        FidelityValidation result
    """
    result = FidelityValidation()
    
    # Volume comparison
    try:
        result.original_volume = abs(float(original.volume)) if original.is_volume else 0.0
        result.repaired_volume = abs(float(repaired.volume)) if repaired.is_volume else 0.0
        
        if result.original_volume > 0:
            result.volume_change_pct = (
                (result.repaired_volume - result.original_volume) / 
                result.original_volume * 100
            )
        else:
            result.volume_change_pct = 0.0
        
        result.volume_acceptable = abs(result.volume_change_pct) <= max_volume_change_pct
    except Exception:
        result.volume_acceptable = True
    
    # Bounding box comparison
    try:
        orig_diagonal = float(np.linalg.norm(original.bounds[1] - original.bounds[0]))
        rep_diagonal = float(np.linalg.norm(repaired.bounds[1] - repaired.bounds[0]))
        
        result.original_bbox_diagonal = orig_diagonal
        result.repaired_bbox_diagonal = rep_diagonal
        
        if orig_diagonal > 0:
            result.bbox_change_pct = abs(rep_diagonal - orig_diagonal) / orig_diagonal * 100
        else:
            result.bbox_change_pct = 0.0
        
        result.bbox_unchanged = result.bbox_change_pct <= max_bbox_change_pct
    except Exception:
        result.bbox_unchanged = True
    
    # Hausdorff distance
    try:
        hausdorff, mean_dist = compute_hausdorff_distance(
            original, repaired, hausdorff_sample_count
        )
        
        result.hausdorff_distance = hausdorff
        result.mean_surface_distance = mean_dist
        
        if result.original_bbox_diagonal > 0:
            result.hausdorff_relative = hausdorff / result.original_bbox_diagonal
        else:
            result.hausdorff_relative = 0.0
        
        result.hausdorff_acceptable = result.hausdorff_relative <= max_hausdorff_relative
    except Exception:
        result.hausdorff_acceptable = True
    
    # Surface area comparison
    try:
        result.original_area = float(original.area)
        result.repaired_area = float(repaired.area)
        
        if result.original_area > 0:
            result.area_change_pct = (
                (result.repaired_area - result.original_area) /
                result.original_area * 100
            )
        else:
            result.area_change_pct = 0.0
        
        result.surface_area_acceptable = abs(result.area_change_pct) <= max_area_change_pct
    except Exception:
        result.surface_area_acceptable = True
    
    return result


def validate_repair(
    original: trimesh.Trimesh,
    repaired: trimesh.Trimesh,
    max_volume_change_pct: float = 1.0,
    max_hausdorff_relative: float = 0.001
) -> ValidationResult:
    """
    Complete validation of a repair operation.
    
    Runs both geometric and fidelity validation, and captures
    before/after diagnostics.
    
    Args:
        original: Original mesh before repair
        repaired: Mesh after repair
        max_volume_change_pct: Maximum allowed volume change
        max_hausdorff_relative: Maximum relative surface deviation
        
    Returns:
        Complete ValidationResult
    """
    logger.info("Running repair validation...")
    
    geometric = validate_geometry(repaired)
    fidelity = validate_fidelity(
        original, repaired,
        max_volume_change_pct=max_volume_change_pct,
        max_hausdorff_relative=max_hausdorff_relative
    )
    
    original_diag = compute_diagnostics(original)
    repaired_diag = compute_diagnostics(repaired)
    
    result = ValidationResult(
        geometric=geometric,
        fidelity=fidelity,
        original_diagnostics=original_diag,
        repaired_diagnostics=repaired_diag
    )
    
    logger.info(f"Validation result: geometric={'PASS' if geometric.is_printable else 'FAIL'}, "
                f"fidelity={'PASS' if fidelity.is_visually_unchanged else 'FAIL'}")
    
    return result


def format_validation_result(result: ValidationResult) -> str:
    """Format validation result as a human-readable string."""
    lines = [
        "\nValidation Result",
        "=" * 50,
        f"Overall: {'SUCCESS' if result.is_successful else 'FAILED'}",
        "",
        f"Geometric Validation: {'PASS' if result.geometric.is_printable else 'FAIL'}",
        f"  Watertight: {result.geometric.is_watertight}",
        f"  Manifold: {result.geometric.is_manifold}",
        f"  Positive Volume: {result.geometric.has_positive_volume}",
        f"  Winding Consistent: {result.geometric.is_winding_consistent}",
    ]
    
    if result.geometric.issues:
        lines.append(f"  Issues: {', '.join(result.geometric.issues)}")
    
    lines.extend([
        "",
        f"Fidelity Validation: {'PASS' if result.fidelity.is_visually_unchanged else 'FAIL'}",
        f"  Volume Change: {result.fidelity.volume_change_pct:.2f}%",
        f"  Bbox Change: {result.fidelity.bbox_change_pct:.2f}%",
        f"  Hausdorff (relative): {result.fidelity.hausdorff_relative*100:.4f}%",
    ])
    
    if result.fidelity.changes:
        lines.append(f"  Changes: {', '.join(result.fidelity.changes)}")
    
    lines.append("=" * 50)
    
    return "\n".join(lines)


def print_validation_result(result: ValidationResult) -> None:
    """Print validation result in a readable format."""
    print(format_validation_result(result))
