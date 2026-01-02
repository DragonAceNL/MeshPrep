# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Trimesh-based actions for mesh repair.

These are the core repair operations using trimesh's built-in functions.
See docs/filter_actions.md for the complete action catalog.
"""

import numpy as np
import trimesh
from trimesh import repair

from .registry import register_action


# =============================================================================
# Category: Loading & Basic Cleanup
# =============================================================================

@register_action(
    name="trimesh_basic",
    description="Basic trimesh cleanup: merge vertices, remove degenerate faces, fix normals",
    parameters={},
    risk_level="low",
    tool="trimesh",
    category="Loading & Basic Cleanup"
)
def action_trimesh_basic(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Basic cleanup using trimesh.
    
    Operations:
    - Merge duplicate vertices
    - Remove degenerate faces
    - Remove duplicate faces
    - Remove unreferenced vertices
    - Fix normals
    """
    mesh = mesh.copy()
    
    # Merge close vertices
    try:
        mesh.merge_vertices()
    except Exception:
        pass
    
    # Remove degenerate faces (zero area)
    try:
        if hasattr(mesh, 'remove_degenerate_faces'):
            mesh.remove_degenerate_faces()
        else:
            mask = mesh.nondegenerate_faces()
            if not mask.all():
                mesh.update_faces(mask)
    except Exception:
        pass
    
    # Remove duplicate faces
    try:
        if hasattr(mesh, 'remove_duplicate_faces'):
            mesh.remove_duplicate_faces()
        else:
            _, unique_idx = np.unique(
                np.sort(mesh.faces, axis=1), 
                axis=0, 
                return_index=True
            )
            if len(unique_idx) < len(mesh.faces):
                mesh.update_faces(np.sort(unique_idx))
    except Exception:
        pass
    
    # Remove unreferenced vertices
    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass
    
    # Fix normals
    try:
        mesh.fix_normals()
    except Exception:
        pass
    
    return mesh


@register_action(
    name="merge_vertices",
    description="Weld duplicate vertices within a tolerance",
    parameters={"eps": {"type": "float", "default": 1e-8, "description": "Distance threshold"}},
    risk_level="low",
    tool="trimesh",
    category="Loading & Basic Cleanup"
)
def action_merge_vertices(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """Merge close vertices within tolerance."""
    mesh = mesh.copy()
    try:
        mesh.merge_vertices()
    except Exception:
        pass
    return mesh


@register_action(
    name="remove_degenerate_faces",
    description="Remove faces with zero area or invalid topology",
    parameters={},
    risk_level="low",
    tool="trimesh",
    category="Loading & Basic Cleanup"
)
def action_remove_degenerate_faces(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """Remove degenerate (zero-area) faces."""
    mesh = mesh.copy()
    try:
        if hasattr(mesh, 'remove_degenerate_faces'):
            mesh.remove_degenerate_faces()
        else:
            mask = mesh.nondegenerate_faces()
            if not mask.all():
                mesh.update_faces(mask)
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass
    return mesh


@register_action(
    name="remove_duplicate_faces",
    description="Remove faces that are exact duplicates",
    parameters={},
    risk_level="low",
    tool="trimesh",
    category="Loading & Basic Cleanup"
)
def action_remove_duplicate_faces(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """Remove duplicate faces."""
    mesh = mesh.copy()
    try:
        if hasattr(mesh, 'remove_duplicate_faces'):
            mesh.remove_duplicate_faces()
    except Exception:
        pass
    return mesh


@register_action(
    name="remove_unreferenced_vertices",
    description="Remove vertices not referenced by any face",
    parameters={},
    risk_level="low",
    tool="trimesh",
    category="Loading & Basic Cleanup"
)
def action_remove_unreferenced_vertices(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """Remove unreferenced vertices."""
    mesh = mesh.copy()
    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass
    return mesh


# =============================================================================
# Category: Hole Filling
# =============================================================================

@register_action(
    name="fill_holes",
    description="Fill holes in the mesh up to a maximum size",
    parameters={},
    risk_level="medium",
    tool="trimesh",
    category="Hole Filling"
)
def action_fill_holes(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Fill holes in the mesh.
    
    Uses trimesh's repair.fill_holes function.
    """
    mesh = mesh.copy()
    repair.fill_holes(mesh)
    return mesh


# =============================================================================
# Category: Normal Correction
# =============================================================================

@register_action(
    name="fix_normals",
    description="Fix face normals to be consistent and outward-pointing",
    parameters={},
    risk_level="low",
    tool="trimesh",
    category="Normal Correction"
)
def action_fix_normals(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """Fix face normals for consistent winding and outward direction."""
    mesh = mesh.copy()
    mesh.fix_normals()
    return mesh


@register_action(
    name="fix_winding",
    description="Make face winding consistent",
    parameters={},
    risk_level="low",
    tool="trimesh",
    category="Normal Correction"
)
def action_fix_winding(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """Make face winding consistent."""
    mesh = mesh.copy()
    repair.fix_winding(mesh)
    return mesh


@register_action(
    name="fix_inversion",
    description="Fix inverted normals (inside-out mesh)",
    parameters={},
    risk_level="low",
    tool="trimesh",
    category="Normal Correction"
)
def action_fix_inversion(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """Fix inverted normals if mesh is inside-out."""
    mesh = mesh.copy()
    repair.fix_inversion(mesh)
    return mesh


@register_action(
    name="flip_normals",
    description="Invert all face normals",
    parameters={},
    risk_level="low",
    tool="trimesh",
    category="Normal Correction"
)
def action_flip_normals(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """Invert all face normals (useful if model is inside-out)."""
    mesh = mesh.copy()
    mesh.invert()
    return mesh


# =============================================================================
# Category: Component Management
# =============================================================================

@register_action(
    name="keep_largest_component",
    description="Keep only the largest connected component",
    parameters={},
    risk_level="high",
    tool="trimesh",
    category="Component Management"
)
def action_keep_largest_component(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Keep only the largest connected component.
    
    WARNING: This removes all other components!
    """
    components = mesh.split(only_watertight=False)
    
    if not components or len(components) <= 1:
        return mesh.copy()
    
    # Find largest by face count
    largest = max(components, key=lambda m: len(m.faces))
    return largest


@register_action(
    name="remove_small_components",
    description="Remove disconnected components below a threshold",
    parameters={
        "min_faces": {"type": "int", "default": 100, "description": "Minimum face count"},
        "min_volume_ratio": {"type": "float", "default": 0.01, "description": "Min ratio of largest volume"}
    },
    risk_level="medium",
    tool="trimesh",
    category="Component Management"
)
def action_remove_small_components(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """Remove small disconnected components."""
    min_faces = params.get("min_faces", 100)
    min_ratio = params.get("min_volume_ratio", 0.01)
    
    components = mesh.split(only_watertight=False)
    
    if not components or len(components) <= 1:
        return mesh.copy()
    
    # Find the largest component for reference
    largest_faces = max(len(c.faces) for c in components)
    threshold = max(min_faces, int(largest_faces * min_ratio))
    
    # Keep components above threshold
    kept = [c for c in components if len(c.faces) >= threshold]
    
    if not kept:
        return mesh.copy()
    
    if len(kept) == 1:
        return kept[0]
    
    return trimesh.util.concatenate(kept)


# =============================================================================
# Category: Simplification & Remeshing
# =============================================================================

@register_action(
    name="simplify",
    description="Simplify mesh by reducing face count",
    parameters={
        "target_faces": {"type": "int", "default": None, "description": "Target face count"},
        "ratio": {"type": "float", "default": 0.5, "description": "Ratio to reduce to (if target_faces not set)"}
    },
    risk_level="medium",
    tool="trimesh",
    category="Simplification & Remeshing"
)
def action_simplify(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """Simplify mesh using quadric decimation."""
    target_faces = params.get("target_faces")
    ratio = params.get("ratio", 0.5)
    
    if target_faces is None:
        target_faces = max(4, int(len(mesh.faces) * ratio))
    
    return mesh.simplify_quadric_decimation(target_faces)


@register_action(
    name="subdivide",
    description="Subdivide faces to increase mesh resolution",
    parameters={"iterations": {"type": "int", "default": 1, "description": "Number of subdivisions"}},
    risk_level="medium",
    tool="trimesh",
    category="Simplification & Remeshing"
)
def action_subdivide(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """Subdivide mesh faces."""
    mesh = mesh.copy()
    iterations = params.get("iterations", 1)
    
    for _ in range(iterations):
        mesh = mesh.subdivide()
    
    return mesh


@register_action(
    name="smooth_laplacian",
    description="Apply Laplacian smoothing to reduce noise",
    parameters={
        "iterations": {"type": "int", "default": 1, "description": "Number of smoothing passes"},
        "lamb": {"type": "float", "default": 0.5, "description": "Smoothing factor (0-1)"}
    },
    risk_level="medium",
    tool="trimesh",
    category="Simplification & Remeshing"
)
def action_smooth_laplacian(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """Apply Laplacian smoothing."""
    mesh = mesh.copy()
    iterations = params.get("iterations", 1)
    lamb = params.get("lamb", 0.5)
    
    trimesh.smoothing.filter_laplacian(mesh, iterations=iterations, lamb=lamb)
    return mesh


# =============================================================================
# Category: Boolean & Advanced
# =============================================================================

@register_action(
    name="convex_hull",
    description="Replace mesh with its convex hull",
    parameters={},
    risk_level="high",
    tool="trimesh",
    category="Boolean & Advanced"
)
def action_convex_hull(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Replace mesh with its convex hull.
    
    WARNING: Dramatically changes mesh shape!
    """
    return mesh.convex_hull


# =============================================================================
# Category: Validation & Diagnostics
# =============================================================================

@register_action(
    name="validate",
    description="Validation checkpoint - does not modify mesh",
    parameters={},
    risk_level="low",
    tool="internal",
    category="Validation & Diagnostics"
)
def action_validate(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Validation checkpoint.
    
    This action doesn't modify the mesh - it's a marker for
    when to run validation checks in the pipeline.
    """
    return mesh.copy()
