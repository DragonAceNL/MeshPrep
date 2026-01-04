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


@register_action(
    name="flip_normals",
    description="Invert all face normals (useful if model is inside-out)",
    parameters={},
    risk_level="low"
)
def action_flip_normals(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Flip all face normals.
    
    Inverts the winding order of all faces, effectively flipping
    all normals to point in the opposite direction. Useful when
    a model appears inside-out.
    """
    mesh = mesh.copy()
    
    # Flip faces by reversing the vertex order
    mesh.faces = np.fliplr(mesh.faces)
    
    # Invalidate cached normals so they get recalculated
    mesh._cache.clear()
    
    return mesh


@register_action(
    name="remove_small_components",
    description="Remove disconnected components below a volume or face-count threshold",
    parameters={"min_faces": 50, "min_volume": None},
    risk_level="medium"
)
def action_remove_small_components(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Remove small disconnected components.
    
    Splits the mesh into connected components and removes those
    that are below the specified thresholds. Useful for removing
    debris, floating particles, or detached geometry.
    
    Args:
        mesh: Input mesh
        params: 
            - min_faces: Minimum face count to keep (default: 50)
            - min_volume: Minimum volume to keep (optional)
            
    Returns:
        Mesh with small components removed
    """
    min_faces = params.get("min_faces", 50)
    min_volume = params.get("min_volume")
    
    # Split into connected components
    components = mesh.split(only_watertight=False)
    
    if len(components) <= 1:
        return mesh.copy()
    
    # Filter components
    kept_components = []
    for comp in components:
        # Check face count threshold
        if len(comp.faces) < min_faces:
            continue
        
        # Check volume threshold if specified
        if min_volume is not None:
            try:
                if comp.is_watertight and abs(comp.volume) < min_volume:
                    continue
            except Exception:
                pass  # Skip volume check if it fails
        
        kept_components.append(comp)
    
    if not kept_components:
        # If all components would be removed, keep the largest
        largest = max(components, key=lambda c: len(c.faces))
        return largest
    
    # Concatenate remaining components
    if len(kept_components) == 1:
        return kept_components[0]
    
    return trimesh.util.concatenate(kept_components)


@register_action(
    name="decimate",
    description="Reduce face count while preserving shape using quadric decimation",
    parameters={"target_faces": None, "ratio": 0.5},
    risk_level="medium"
)
def action_decimate(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Decimate mesh to reduce face count.
    
    Uses quadric error decimation to reduce the number of faces
    while attempting to preserve the overall shape. Specify either
    target_faces (absolute count) or ratio (fraction to keep).
    
    Args:
        mesh: Input mesh
        params:
            - target_faces: Target number of faces (takes precedence)
            - ratio: Fraction of faces to keep (default: 0.5 = 50%)
            
    Returns:
        Decimated mesh with fewer faces
    """
    target_faces = params.get("target_faces")
    ratio = params.get("ratio", 0.5)
    
    current_faces = len(mesh.faces)
    
    if target_faces is None:
        target_faces = int(current_faces * ratio)
    
    # Don't decimate if already below target
    if current_faces <= target_faces:
        return mesh.copy()
    
    # Use trimesh's simplify_quadric_decimation
    try:
        simplified = mesh.simplify_quadric_decimation(target_faces)
        return simplified
    except Exception as e:
        # Fall back to original if decimation fails
        import logging
        logging.getLogger(__name__).warning(f"Decimation failed: {e}")
        return mesh.copy()


@register_action(
    name="smooth",
    description="Apply smoothing to reduce noise and surface irregularities",
    parameters={"iterations": 1, "lamb": 0.5, "method": "laplacian"},
    risk_level="medium"
)
def action_smooth(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Apply smoothing to the mesh.
    
    Smooths the mesh surface to reduce noise, sharp edges, and
    irregularities. Supports Laplacian and Taubin smoothing methods.
    
    Args:
        mesh: Input mesh
        params:
            - iterations: Number of smoothing passes (default: 1)
            - lamb: Smoothing factor 0-1 (default: 0.5)
            - method: "laplacian" or "taubin" (default: "laplacian")
            
    Returns:
        Smoothed mesh
    """
    mesh = mesh.copy()
    
    iterations = params.get("iterations", 1)
    lamb = params.get("lamb", 0.5)
    method = params.get("method", "laplacian")
    
    try:
        if method == "taubin":
            # Taubin smoothing reduces shrinkage compared to Laplacian
            mu = params.get("mu", -0.53)  # Typical value to counteract shrinkage
            trimesh.smoothing.filter_taubin(
                mesh, 
                iterations=iterations, 
                lamb=lamb, 
                mu=mu
            )
        else:
            # Default: Laplacian smoothing
            trimesh.smoothing.filter_laplacian(
                mesh, 
                iterations=iterations, 
                lamb=lamb
            )
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Smoothing failed: {e}")
    
    return mesh


@register_action(
    name="smooth_humphrey",
    description="Apply Humphrey smoothing (better feature preservation)",
    parameters={"iterations": 1, "alpha": 0.1, "beta": 0.5},
    risk_level="medium"
)
def action_smooth_humphrey(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Apply Humphrey's Classes smoothing.
    
    A smoothing algorithm that better preserves sharp features
    and edges compared to standard Laplacian smoothing.
    
    Args:
        mesh: Input mesh
        params:
            - iterations: Number of smoothing passes (default: 1)
            - alpha: Feature preservation factor (default: 0.1)
            - beta: Smoothing factor (default: 0.5)
    """
    mesh = mesh.copy()
    
    iterations = params.get("iterations", 1)
    alpha = params.get("alpha", 0.1)
    beta = params.get("beta", 0.5)
    
    try:
        trimesh.smoothing.filter_humphrey(
            mesh,
            iterations=iterations,
            alpha=alpha,
            beta=beta
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Humphrey smoothing failed: {e}")
    
    return mesh


@register_action(
    name="smooth_mut_dif",
    description="Apply Mutually Diffused smoothing (volume-preserving)",
    parameters={"iterations": 1, "lamb": 0.5},
    risk_level="medium"
)
def action_smooth_mut_dif(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Apply Mutually Diffused Laplacian smoothing.
    
    A volume-preserving smoothing method that better maintains
    the overall shape of the mesh.
    """
    mesh = mesh.copy()
    
    iterations = params.get("iterations", 1)
    lamb = params.get("lamb", 0.5)
    
    try:
        trimesh.smoothing.filter_mut_dif_laplacian(
            mesh,
            iterations=iterations,
            lamb=lamb
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Mut-dif smoothing failed: {e}")
    
    return mesh


@register_action(
    name="subdivide_loop",
    description="Loop subdivision for smooth surface refinement",
    parameters={"iterations": 1},
    risk_level="medium"
)
def action_subdivide_loop(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Apply Loop subdivision.
    
    Loop subdivision creates a smooth surface by subdividing
    each triangle into four and smoothing vertex positions.
    Good for creating smooth organic shapes.
    """
    iterations = params.get("iterations", 1)
    
    result = mesh.copy()
    for _ in range(iterations):
        try:
            result = result.subdivide_loop()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Loop subdivision failed: {e}")
            break
    
    return result


@register_action(
    name="subdivide_to_size",
    description="Subdivide until edges are below a target length",
    parameters={"max_edge": 1.0},
    risk_level="medium"
)
def action_subdivide_to_size(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Subdivide mesh to achieve a maximum edge length.
    
    Iteratively subdivides faces until no edge is longer than
    the specified maximum length.
    
    Args:
        mesh: Input mesh
        params:
            - max_edge: Maximum edge length (default: 1.0)
    """
    max_edge = params.get("max_edge", 1.0)
    
    try:
        vertices, faces = trimesh.remesh.subdivide_to_size(
            mesh.vertices,
            mesh.faces,
            max_edge=max_edge
        )
        return trimesh.Trimesh(vertices=vertices, faces=faces)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Subdivide to size failed: {e}")
        return mesh.copy()


@register_action(
    name="remove_infinite_values",
    description="Remove vertices and faces with NaN or infinite values",
    parameters={},
    risk_level="low"
)
def action_remove_infinite_values(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Remove infinite/NaN values from mesh.
    
    Removes any vertices that contain NaN or infinite coordinate
    values, and any faces that reference them.
    """
    mesh = mesh.copy()
    
    try:
        mesh.remove_infinite_values()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Remove infinite values failed: {e}")
    
    return mesh


@register_action(
    name="remove_unreferenced_vertices",
    description="Remove vertices not used by any face",
    parameters={},
    risk_level="low"
)
def action_remove_unreferenced_vertices(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Remove unreferenced vertices.
    
    Removes vertices that are not connected to any face.
    Reduces file size without affecting geometry.
    """
    mesh = mesh.copy()
    
    try:
        mesh.remove_unreferenced_vertices()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Remove unreferenced vertices failed: {e}")
    
    return mesh


@register_action(
    name="unmerge_vertices",
    description="Unmerge vertices so each face has unique vertices",
    parameters={},
    risk_level="low"
)
def action_unmerge_vertices(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Unmerge vertices.
    
    Creates a mesh where each face has its own unique vertices.
    This can be useful before certain operations that need
    per-face vertex attributes.
    """
    mesh = mesh.copy()
    
    try:
        mesh.unmerge_vertices()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Unmerge vertices failed: {e}")
    
    return mesh


@register_action(
    name="split_components",
    description="Split mesh into separate connected components",
    parameters={"only_watertight": False},
    risk_level="low"
)
def action_split_components(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Split mesh into connected components.
    
    Returns the mesh split into separate components. If there's
    only one component, returns a copy of the original.
    
    Note: This action returns all components concatenated back together,
    but with the split performed. Use for diagnostics or when combined
    with other component-based operations.
    """
    only_watertight = params.get("only_watertight", False)
    
    components = mesh.split(only_watertight=only_watertight)
    
    if len(components) <= 1:
        return mesh.copy()
    
    # Return all components concatenated
    return trimesh.util.concatenate(components)


@register_action(
    name="boolean_union",
    description="Boolean union of all mesh components",
    parameters={},
    risk_level="high"
)
def action_boolean_union(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Boolean union of all mesh components.
    
    Merges all overlapping components into a single watertight mesh.
    Requires multiple components in the mesh.
    """
    components = mesh.split(only_watertight=False)
    
    if len(components) <= 1:
        return mesh.copy()
    
    try:
        # Start with first component
        result = components[0]
        
        # Union with remaining components
        for comp in components[1:]:
            try:
                result = result.union(comp)
            except Exception:
                # If union fails, try to continue
                pass
        
        return result
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Boolean union failed: {e}")
        return mesh.copy()


@register_action(
    name="boolean_difference",
    description="Boolean difference: subtract second component from first",
    parameters={},
    risk_level="high"
)
def action_boolean_difference(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Boolean difference of mesh components.
    
    Subtracts smaller components from the largest component.
    """
    components = mesh.split(only_watertight=False)
    
    if len(components) <= 1:
        return mesh.copy()
    
    try:
        # Sort by volume (largest first)
        components = sorted(components, key=lambda c: len(c.faces), reverse=True)
        
        result = components[0]
        for comp in components[1:]:
            try:
                result = result.difference(comp)
            except Exception:
                pass
        
        return result
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Boolean difference failed: {e}")
        return mesh.copy()


@register_action(
    name="boolean_intersection",
    description="Boolean intersection of mesh components",
    parameters={},
    risk_level="high"
)
def action_boolean_intersection(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Boolean intersection of mesh components.
    
    Keeps only the overlapping volume of all components.
    """
    components = mesh.split(only_watertight=False)
    
    if len(components) <= 1:
        return mesh.copy()
    
    try:
        result = components[0]
        for comp in components[1:]:
            try:
                result = result.intersection(comp)
            except Exception:
                pass
        
        return result
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Boolean intersection failed: {e}")
        return mesh.copy()


@register_action(
    name="voxelize_and_reconstruct",
    description="Voxelize mesh and reconstruct surface (aggressive repair)",
    parameters={"pitch": None, "fill": True},
    risk_level="high"
)
def action_voxelize_and_reconstruct(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Voxelize and reconstruct mesh.
    
    Converts mesh to voxels and back, creating a watertight mesh.
    This is an aggressive repair that may lose fine detail but
    guarantees a valid closed mesh.
    
    Args:
        mesh: Input mesh
        params:
            - pitch: Voxel size (auto/None for auto-calculation, or numeric value)
            - fill: Fill internal voids (default: True)
    """
    pitch = params.get("pitch")
    fill = params.get("fill", True)
    
    # Auto-calculate pitch based on mesh size if pitch is None or "auto"
    if pitch is None or pitch == "auto" or not isinstance(pitch, (int, float)):
        bbox = mesh.bounds
        bbox_size = bbox[1] - bbox[0]
        pitch = max(bbox_size) / 100  # 1% of largest dimension
    
    try:
        # Voxelize
        voxels = mesh.voxelized(pitch=pitch)
        
        if fill:
            voxels = voxels.fill()
        
        # Convert back to mesh
        result = voxels.marching_cubes
        
        return result
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Voxelize and reconstruct failed: {e}")
        return mesh.copy()


@register_action(
    name="convex_decomposition",
    description="Decompose mesh into convex parts",
    parameters={"max_parts": 16},
    risk_level="high"
)
def action_convex_decomposition(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Approximate convex decomposition.
    
    Decomposes the mesh into approximately convex parts.
    Useful for physics simulation or simplifying complex geometry.
    
    Note: Returns all convex parts merged back into one mesh.
    """
    max_parts = params.get("max_parts", 16)
    
    try:
        # Perform convex decomposition
        convex_parts = mesh.convex_decomposition(maxhulls=max_parts)
        
        if convex_parts and len(convex_parts) > 0:
            return trimesh.util.concatenate(convex_parts)
        return mesh.copy()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Convex decomposition failed: {e}")
        return mesh.copy()


@register_action(
    name="sample_surface",
    description="Sample points on mesh surface (for point cloud conversion)",
    parameters={"count": 10000},
    risk_level="low"
)
def action_sample_surface(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Sample points on mesh surface.
    
    Creates a point cloud by sampling the mesh surface.
    Note: Returns the original mesh unchanged - this is primarily
    for diagnostics or when combined with reconstruction.
    """
    # This action is mainly informational - return original mesh
    return mesh.copy()


@register_action(
    name="scale_to_unit",
    description="Scale mesh to fit in a unit bounding box",
    parameters={"uniform": True},
    risk_level="low"
)
def action_scale_to_unit(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Scale mesh to fit in unit bounding box.
    
    Normalizes mesh size by scaling to fit within a 1x1x1 box.
    
    Args:
        mesh: Input mesh
        params:
            - uniform: Keep aspect ratio (default: True)
    """
    mesh = mesh.copy()
    uniform = params.get("uniform", True)
    
    bbox = mesh.bounds
    size = bbox[1] - bbox[0]
    
    if uniform:
        scale_factor = 1.0 / max(size)
    else:
        scale_factor = 1.0 / size  # Non-uniform scaling
    
    if uniform:
        mesh.apply_scale(scale_factor)
    else:
        mesh.apply_scale(scale_factor)
    
    return mesh


@register_action(
    name="fix_broken_faces",
    description="Attempt to fix broken/invalid face references",
    parameters={},
    risk_level="medium"
)
def action_fix_broken_faces(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Fix broken face references.
    
    Identifies and attempts to repair faces with invalid
    vertex references.
    """
    mesh = mesh.copy()
    
    try:
        broken = repair.broken_faces(mesh)
        if len(broken) > 0:
            # Remove broken faces
            mask = np.ones(len(mesh.faces), dtype=bool)
            mask[broken] = False
            mesh.update_faces(mask)
            mesh.remove_unreferenced_vertices()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Fix broken faces failed: {e}")
    
    return mesh


@register_action(
    name="stitch_boundaries",
    description="Stitch open boundary edges that are close together",
    parameters={"distance": 0.001},
    risk_level="medium"
)
def action_stitch_boundaries(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Stitch open boundary edges.
    
    Attempts to connect nearby open edges to close holes
    caused by small gaps in the mesh.
    
    Args:
        mesh: Input mesh  
        params:
            - distance: Maximum distance between edges to stitch (default: 0.001)
    """
    mesh = mesh.copy()
    distance = params.get("distance", 0.001)
    
    try:
        # Use trimesh's stitch function
        trimesh.repair.stitch(mesh, distance=distance)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Stitch boundaries failed: {e}")
    
    return mesh
