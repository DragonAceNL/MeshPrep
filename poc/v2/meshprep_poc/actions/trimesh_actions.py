# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Trimesh-based actions for mesh repair.

These are the core repair operations using trimesh's built-in functions.
"""

import numpy as np
import trimesh
from trimesh import repair

from .registry import register_action


@register_action(
    name="trimesh_basic",
    description="Basic trimesh cleanup: merge vertices, remove degenerate faces",
    parameters={"merge_threshold": 1e-8},
    risk_level="low"
)
def action_trimesh_basic(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Basic cleanup using trimesh.
    
    Operations:
    - Merge duplicate vertices
    - Remove degenerate faces
    - Remove duplicate faces
    - Remove unreferenced vertices
    """
    # Work on a copy
    mesh = mesh.copy()
    
    # Merge close vertices
    # Note: trimesh API varies by version, try different approaches
    try:
        mesh.merge_vertices()
    except Exception:
        pass  # Continue if merge fails
    
    # Remove degenerate faces (zero area)
    # In newer trimesh, use nondegenerate_faces property to filter
    try:
        if hasattr(mesh, 'remove_degenerate_faces'):
            mesh.remove_degenerate_faces()
        else:
            # Newer trimesh - filter using nondegenerate mask
            mask = mesh.nondegenerate_faces()
            if not mask.all():
                mesh.update_faces(mask)
    except Exception:
        pass
    
    # Remove duplicate faces (update_faces with unique)
    try:
        if hasattr(mesh, 'remove_duplicate_faces'):
            mesh.remove_duplicate_faces()
        else:
            # Newer trimesh - use unique_faces
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
    
    # Update face normals
    try:
        mesh.fix_normals()
    except Exception:
        pass
    
    return mesh


@register_action(
    name="fill_holes",
    description="Fill holes in the mesh using trimesh",
    parameters={},
    risk_level="medium"
)
def action_fill_holes(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Fill holes in the mesh.
    
    Uses trimesh's repair.fill_holes function which attempts to
    fill all holes in the mesh by adding triangular faces.
    """
    mesh = mesh.copy()
    
    # trimesh's fill_holes modifies the mesh in place
    repair.fill_holes(mesh)
    
    return mesh


@register_action(
    name="fix_normals",
    description="Fix face normals to be consistent and outward-pointing",
    parameters={},
    risk_level="low"
)
def action_fix_normals(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Fix face normals.
    
    Ensures consistent winding order and outward-pointing normals.
    """
    mesh = mesh.copy()
    
    # Fix winding to be consistent
    mesh.fix_normals()
    
    return mesh


@register_action(
    name="fix_winding",
    description="Make face winding consistent",
    parameters={},
    risk_level="low"
)
def action_fix_winding(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Make face winding consistent.
    
    All faces will have the same winding direction.
    """
    mesh = mesh.copy()
    repair.fix_winding(mesh)
    return mesh


@register_action(
    name="fix_inversion",
    description="Fix inverted normals (inside-out mesh)",
    parameters={},
    risk_level="low"
)
def action_fix_inversion(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Fix inverted normals.
    
    Detects if the mesh is inside-out and flips normals if needed.
    """
    mesh = mesh.copy()
    repair.fix_inversion(mesh)
    return mesh


@register_action(
    name="remove_degenerate",
    description="Remove degenerate (zero-area) faces",
    parameters={},
    risk_level="low"
)
def action_remove_degenerate(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Remove degenerate faces.
    
    Removes faces with zero or near-zero area.
    """
    mesh = mesh.copy()
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    return mesh


@register_action(
    name="merge_vertices",
    description="Merge vertices that are very close together",
    parameters={"threshold": 1e-8},
    risk_level="low"
)
def action_merge_vertices(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Merge close vertices.
    
    Vertices within the threshold distance are merged into one.
    """
    mesh = mesh.copy()
    threshold = params.get("threshold", 1e-8)
    mesh.merge_vertices(merge_threshold=threshold)
    return mesh


@register_action(
    name="remove_duplicate_faces",
    description="Remove duplicate faces",
    parameters={},
    risk_level="low"
)
def action_remove_duplicate_faces(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Remove duplicate faces.
    
    Removes faces that reference the same vertices.
    """
    mesh = mesh.copy()
    mesh.remove_duplicate_faces()
    return mesh


@register_action(
    name="keep_largest_component",
    description="Keep only the largest connected component",
    parameters={},
    risk_level="high"
)
def action_keep_largest_component(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Keep only the largest connected component.
    
    WARNING: This removes all other components!
    """
    components = mesh.split(only_watertight=False)
    
    if len(components) <= 1:
        return mesh.copy()
    
    # Find largest by volume or face count
    largest = max(components, key=lambda m: len(m.faces))
    
    return largest


@register_action(
    name="subdivide",
    description="Subdivide faces to increase mesh resolution",
    parameters={"iterations": 1},
    risk_level="medium"
)
def action_subdivide(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Subdivide mesh faces.
    
    Each face is split into smaller faces.
    """
    mesh = mesh.copy()
    iterations = params.get("iterations", 1)
    
    for _ in range(iterations):
        mesh = mesh.subdivide()
    
    return mesh


@register_action(
    name="simplify",
    description="Simplify mesh by reducing face count",
    parameters={"target_faces": None, "ratio": 0.5},
    risk_level="medium"
)
def action_simplify(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Simplify mesh by reducing faces.
    
    Uses quadric decimation to reduce complexity while
    preserving shape.
    """
    mesh = mesh.copy()
    
    target_faces = params.get("target_faces")
    ratio = params.get("ratio", 0.5)
    
    if target_faces is None:
        target_faces = int(len(mesh.faces) * ratio)
    
    # trimesh uses simplify_quadric_decimation
    simplified = mesh.simplify_quadric_decimation(target_faces)
    
    return simplified


@register_action(
    name="convex_hull",
    description="Replace mesh with its convex hull",
    parameters={},
    risk_level="high"
)
def action_convex_hull(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Replace mesh with its convex hull.
    
    WARNING: This dramatically changes the mesh shape!
    Only use for specific purposes.
    """
    return mesh.convex_hull


@register_action(
    name="smooth_laplacian",
    description="Apply Laplacian smoothing",
    parameters={"iterations": 1, "lamb": 0.5},
    risk_level="medium"
)
def action_smooth_laplacian(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Apply Laplacian smoothing.
    
    Smooths the mesh by averaging vertex positions with neighbors.
    """
    mesh = mesh.copy()
    iterations = params.get("iterations", 1)
    lamb = params.get("lamb", 0.5)
    
    # trimesh's smoothing
    trimesh.smoothing.filter_laplacian(mesh, iterations=iterations, lamb=lamb)
    
    return mesh


@register_action(
    name="validate",
    description="Validation checkpoint - does not modify mesh",
    parameters={},
    risk_level="low"
)
def action_validate(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Validation checkpoint.
    
    This action doesn't modify the mesh - it's a marker for
    when to run validation checks in the pipeline.
    """
    return mesh.copy()


@register_action(
    name="place_on_bed",
    description="Move mesh so its lowest point is at Z=0 (on build plate)",
    parameters={},
    risk_level="low"
)
def action_place_on_bed(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Move mesh to the build plate.
    
    Translates the mesh so its lowest Z coordinate is at Z=0,
    ensuring it sits on the build plate.
    """
    mesh = mesh.copy()
    
    # Get the minimum Z value
    min_z = mesh.bounds[0][2]  # bounds[0] is min, [2] is Z
    
    # Translate to place on bed (Z=0)
    if min_z != 0:
        translation = [0, 0, -min_z]
        mesh.apply_translation(translation)
    
    return mesh


@register_action(
    name="center_mesh",
    description="Center mesh at origin (X=0, Y=0) while keeping on bed",
    parameters={},
    risk_level="low"
)
def action_center_mesh(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Center mesh at origin.
    
    Centers the mesh on X and Y axes while keeping it on the build plate.
    """
    mesh = mesh.copy()
    
    # Get centroid
    centroid = mesh.centroid
    
    # Keep Z at minimum (on bed)
    min_z = mesh.bounds[0][2]
    
    # Translate to center X/Y and place on bed
    translation = [-centroid[0], -centroid[1], -min_z]
    mesh.apply_translation(translation)
    
    return mesh
