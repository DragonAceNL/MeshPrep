# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Validation module for mesh repair operations.

Implements two-stage validation:
1. Geometric validation: Is the mesh printable?
2. Fidelity validation: Is the appearance preserved?
3. Auto-quality scoring: Automatic 1-5 quality rating from metrics
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.spatial import cKDTree

import trimesh

from .mesh_ops import MeshDiagnostics, compute_diagnostics


@dataclass
class GeometricValidation:
    """Result of geometric validation checks."""
    
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


@dataclass
class FidelityValidation:
    """Result of fidelity validation checks."""
    
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


@dataclass
class ValidationResult:
    """Complete validation result for a repair operation."""
    
    geometric: GeometricValidation
    fidelity: FidelityValidation
    
    # Diagnostics
    original_diagnostics: Optional[MeshDiagnostics] = None
    repaired_diagnostics: Optional[MeshDiagnostics] = None
    
    @property
    def is_successful(self) -> bool:
        """Check if repair is fully successful."""
        return self.geometric.is_printable and self.fidelity.is_visually_unchanged
    
    @property
    def is_geometrically_valid(self) -> bool:
        """Alias for geometric validation."""
        return self.geometric.is_printable
    
    @property
    def is_visually_unchanged(self) -> bool:
        """Alias for fidelity validation."""
        return self.fidelity.is_visually_unchanged
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export."""
        return {
            "is_successful": self.is_successful,
            "geometric": {
                "is_printable": self.geometric.is_printable,
                "is_watertight": self.geometric.is_watertight,
                "is_manifold": self.geometric.is_manifold,
                "has_positive_volume": self.geometric.has_positive_volume,
                "is_winding_consistent": self.geometric.is_winding_consistent,
                "issues": self.geometric.issues
            },
            "fidelity": {
                "is_visually_unchanged": self.fidelity.is_visually_unchanged,
                "volume_change_pct": self.fidelity.volume_change_pct,
                "bbox_change_pct": self.fidelity.bbox_change_pct,
                "hausdorff_relative": self.fidelity.hausdorff_relative,
                "changes": self.fidelity.changes
            }
        }


def validate_geometry(mesh: trimesh.Trimesh) -> GeometricValidation:
    """
    Run geometric validation checks on a mesh.
    
    Args:
        mesh: The mesh to validate
        
    Returns:
        GeometricValidation result
    """
    result = GeometricValidation()
    
    result.is_watertight = mesh.is_watertight
    result.is_manifold = mesh.is_volume
    result.is_winding_consistent = mesh.is_winding_consistent
    
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
    
    Args:
        original: Original mesh
        repaired: Repaired mesh
        sample_count: Number of surface samples
        
    Returns:
        Tuple of (hausdorff_distance, mean_distance)
    """
    try:
        # Sample points on both surfaces
        samples_original = original.sample(sample_count)
        samples_repaired = repaired.sample(sample_count)
        
        # Build KD-trees
        tree_original = cKDTree(samples_original)
        tree_repaired = cKDTree(samples_repaired)
        
        # Forward distances: original -> repaired
        dist_orig_to_rep, _ = tree_repaired.query(samples_original)
        
        # Backward distances: repaired -> original
        dist_rep_to_orig, _ = tree_original.query(samples_repaired)
        
        # Hausdorff is max of both directions
        hausdorff = max(dist_orig_to_rep.max(), dist_rep_to_orig.max())
        
        # Mean surface distance
        mean_distance = (dist_orig_to_rep.mean() + dist_rep_to_orig.mean()) / 2
        
        return float(hausdorff), float(mean_distance)
        
    except Exception:
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
    Run fidelity validation checks comparing original and repaired meshes.
    
    Args:
        original: Original mesh
        repaired: Repaired mesh
        max_volume_change_pct: Maximum allowed volume change percentage
        max_bbox_change_pct: Maximum allowed bbox change percentage
        max_hausdorff_relative: Maximum relative Hausdorff distance (fraction of bbox diagonal)
        max_area_change_pct: Maximum allowed surface area change percentage
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
        
        result.volume_acceptable = abs(result.volume_change_pct) < max_volume_change_pct
    except Exception:
        result.volume_acceptable = True  # Can't compute, assume OK
    
    # Bounding box comparison
    try:
        orig_diagonal = np.linalg.norm(original.bounds[1] - original.bounds[0])
        rep_diagonal = np.linalg.norm(repaired.bounds[1] - repaired.bounds[0])
        
        result.original_bbox_diagonal = float(orig_diagonal)
        result.repaired_bbox_diagonal = float(rep_diagonal)
        
        if orig_diagonal > 0:
            result.bbox_change_pct = abs(rep_diagonal - orig_diagonal) / orig_diagonal * 100
        else:
            result.bbox_change_pct = 0.0
        
        result.bbox_unchanged = result.bbox_change_pct < max_bbox_change_pct
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
        
        result.hausdorff_acceptable = result.hausdorff_relative < max_hausdorff_relative
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
        
        result.surface_area_acceptable = abs(result.area_change_pct) < max_area_change_pct
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
    
    Args:
        original: Original mesh before repair
        repaired: Mesh after repair
        max_volume_change_pct: Maximum allowed volume change
        max_hausdorff_relative: Maximum relative surface deviation
        
    Returns:
        Complete ValidationResult
    """
    geometric = validate_geometry(repaired)
    fidelity = validate_fidelity(
        original, repaired,
        max_volume_change_pct=max_volume_change_pct,
        max_hausdorff_relative=max_hausdorff_relative
    )
    
    original_diag = compute_diagnostics(original)
    repaired_diag = compute_diagnostics(repaired)
    
    return ValidationResult(
        geometric=geometric,
        fidelity=fidelity,
        original_diagnostics=original_diag,
        repaired_diagnostics=repaired_diag
    )


def print_validation_result(result: ValidationResult) -> None:
    """Print validation result in a readable format."""
    print("\nValidation Result")
    print("=" * 50)
    
    status = "SUCCESS" if result.is_successful else "FAILED"
    print(f"Overall: {status}")
    
    print(f"\nGeometric Validation: {'PASS' if result.geometric.is_printable else 'FAIL'}")
    print(f"  Watertight: {result.geometric.is_watertight}")
    print(f"  Manifold: {result.geometric.is_manifold}")
    print(f"  Positive Volume: {result.geometric.has_positive_volume}")
    print(f"  Winding Consistent: {result.geometric.is_winding_consistent}")
    
    if result.geometric.issues:
        print(f"  Issues: {', '.join(result.geometric.issues)}")
    
    print(f"\nFidelity Validation: {'PASS' if result.fidelity.is_visually_unchanged else 'FAIL'}")
    print(f"  Volume Change: {result.fidelity.volume_change_pct:.2f}%")
    print(f"  Bbox Change: {result.fidelity.bbox_change_pct:.2f}%")
    print(f"  Hausdorff (relative): {result.fidelity.hausdorff_relative*100:.4f}%")
    
    if result.fidelity.changes:
        print(f"  Changes: {', '.join(result.fidelity.changes)}")
    
    print("=" * 50)


def compute_auto_quality_score(
    original: trimesh.Trimesh,
    repaired: trimesh.Trimesh,
    fidelity: FidelityValidation = None,
) -> tuple[int, dict]:
    """
    Compute automatic quality score (1-5) from geometric fidelity metrics.
    
    This function provides an objective quality assessment based on how well
    the repaired mesh preserves the original geometry. It can be used for:
    - Automatic training of the learning system on large datasets
    - Quick quality checks without manual review
    - Flagging repairs that need human verification
    
    The scoring is based on:
    - Volume change (most important - indicates shape preservation)
    - Hausdorff distance (surface deviation from original)
    - Bounding box change (overall size preservation)
    - Surface area change (detail preservation indicator)
    
    Args:
        original: Original mesh before repair
        repaired: Repaired mesh
        fidelity: Pre-computed FidelityValidation (computed if None)
        
    Returns:
        Tuple of (score, details) where:
        - score: Quality score 1-5
            5 = Perfect (indistinguishable from original)
            4 = Good (minor smoothing, fully usable)
            3 = Acceptable (noticeable changes but recognizable)
            2 = Poor (significant detail loss)
            1 = Rejected (unrecognizable or destroyed)
        - details: Dict with breakdown of score components
    """
    if fidelity is None:
        fidelity = validate_fidelity(original, repaired)
    
    score = 5.0  # Start with perfect score
    penalties = {}
    
    # ==========================================================================
    # Volume change penalties (most important - indicates shape preservation)
    # ==========================================================================
    vol_change = abs(fidelity.volume_change_pct)
    if vol_change > 50:
        penalties["volume"] = -3.0
        score -= 3.0
    elif vol_change > 30:
        penalties["volume"] = -2.0
        score -= 2.0
    elif vol_change > 15:
        penalties["volume"] = -1.0
        score -= 1.0
    elif vol_change > 5:
        penalties["volume"] = -0.5
        score -= 0.5
    elif vol_change > 2:
        penalties["volume"] = -0.25
        score -= 0.25
    else:
        penalties["volume"] = 0.0
    
    # ==========================================================================
    # Hausdorff distance penalties (surface deviation from original)
    # hausdorff_relative is fraction of bbox diagonal
    # ==========================================================================
    hausdorff_pct = fidelity.hausdorff_relative * 100  # Convert to percentage
    if hausdorff_pct > 10:
        penalties["hausdorff"] = -2.0
        score -= 2.0
    elif hausdorff_pct > 5:
        penalties["hausdorff"] = -1.5
        score -= 1.5
    elif hausdorff_pct > 2:
        penalties["hausdorff"] = -1.0
        score -= 1.0
    elif hausdorff_pct > 1:
        penalties["hausdorff"] = -0.5
        score -= 0.5
    elif hausdorff_pct > 0.5:
        penalties["hausdorff"] = -0.25
        score -= 0.25
    else:
        penalties["hausdorff"] = 0.0
    
    # ==========================================================================
    # Bounding box change (should be minimal for shape preservation)
    # ==========================================================================
    bbox_change = fidelity.bbox_change_pct
    if bbox_change > 10:
        penalties["bbox"] = -1.5
        score -= 1.5
    elif bbox_change > 5:
        penalties["bbox"] = -1.0
        score -= 1.0
    elif bbox_change > 2:
        penalties["bbox"] = -0.5
        score -= 0.5
    elif bbox_change > 0.5:
        penalties["bbox"] = -0.25
        score -= 0.25
    else:
        penalties["bbox"] = 0.0
    
    # ==========================================================================
    # Surface area change (secondary indicator of detail changes)
    # ==========================================================================
    area_change = abs(fidelity.area_change_pct)
    if area_change > 50:
        penalties["area"] = -1.0
        score -= 1.0
    elif area_change > 30:
        penalties["area"] = -0.5
        score -= 0.5
    elif area_change > 15:
        penalties["area"] = -0.25
        score -= 0.25
    else:
        penalties["area"] = 0.0
    
    # Clamp score to valid range
    final_score = max(1, min(5, round(score)))
    
    details = {
        "raw_score": score,
        "final_score": final_score,
        "penalties": penalties,
        "metrics": {
            "volume_change_pct": fidelity.volume_change_pct,
            "hausdorff_relative_pct": hausdorff_pct,
            "bbox_change_pct": fidelity.bbox_change_pct,
            "area_change_pct": fidelity.area_change_pct,
            "mean_surface_distance": fidelity.mean_surface_distance,
        },
        "thresholds": {
            "volume_excellent": 2,
            "volume_good": 5,
            "volume_acceptable": 15,
            "hausdorff_excellent": 0.5,
            "hausdorff_good": 1,
            "hausdorff_acceptable": 2,
        },
        "interpretation": _interpret_quality_score(final_score),
    }
    
    return final_score, details


def _interpret_quality_score(score: int) -> str:
    """Get human-readable interpretation of quality score."""
    interpretations = {
        5: "Perfect - indistinguishable from original",
        4: "Good - minor smoothing, fully usable for printing",
        3: "Acceptable - noticeable changes but recognizable",
        2: "Poor - significant detail loss, may need review",
        1: "Rejected - unrecognizable or fundamentally changed",
    }
    return interpretations.get(score, "Unknown")


def compute_auto_quality_with_geometric(
    original: trimesh.Trimesh,
    repaired: trimesh.Trimesh,
    geometric: GeometricValidation = None,
    fidelity: FidelityValidation = None,
) -> tuple[int, dict]:
    """
    Compute auto-quality score with geometric validation adjustments.
    
    This version also considers whether the repair achieved geometric
    validity (watertight, manifold) when scoring.
    
    A repair that makes a mesh printable but with some visual changes
    may still be acceptable, while a repair that doesn't achieve
    printability is less useful regardless of visual preservation.
    
    Args:
        original: Original mesh before repair
        repaired: Repaired mesh
        geometric: Pre-computed GeometricValidation (computed if None)
        fidelity: Pre-computed FidelityValidation (computed if None)
        
    Returns:
        Tuple of (score, details) with geometric context
    """
    if geometric is None:
        geometric = validate_geometry(repaired)
    
    if fidelity is None:
        fidelity = validate_fidelity(original, repaired)
    
    # Get base score from fidelity
    base_score, details = compute_auto_quality_score(original, repaired, fidelity)
    
    # Adjust based on geometric validity
    adjustments = {}
    adjusted_score = float(base_score)
    
    # Bonus for achieving printability
    if geometric.is_printable:
        adjustments["printable_bonus"] = 0.5
        adjusted_score += 0.5
    else:
        # Penalty for not achieving printability
        adjustments["not_printable_penalty"] = -0.5
        adjusted_score -= 0.5
    
    # Penalty for introducing issues
    if not geometric.is_watertight:
        adjustments["not_watertight_penalty"] = -0.25
        adjusted_score -= 0.25
    
    if not geometric.is_manifold:
        adjustments["not_manifold_penalty"] = -0.25
        adjusted_score -= 0.25
    
    final_adjusted_score = max(1, min(5, round(adjusted_score)))
    
    details["geometric_adjustments"] = adjustments
    details["geometric_valid"] = geometric.is_printable
    details["adjusted_score"] = adjusted_score
    details["final_score"] = final_adjusted_score
    details["interpretation"] = _interpret_quality_score(final_adjusted_score)
    
    return final_adjusted_score, details
