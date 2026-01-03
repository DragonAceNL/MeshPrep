# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
PyMeshLab-based actions for mesh repair.

PyMeshLab provides Python bindings for MeshLab, offering access to 
MeshLab's extensive collection of mesh processing filters. These are
often more robust than trimesh equivalents for difficult meshes.

Note: PyMeshLab must be installed: pip install pymeshlab
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

from .registry import register_action

logger = logging.getLogger(__name__)

# Try to import pymeshlab
try:
    import pymeshlab
    PYMESHLAB_AVAILABLE = True
except ImportError:
    PYMESHLAB_AVAILABLE = False
    logger.warning("pymeshlab not available - PyMeshLab actions will be disabled")


def is_pymeshlab_available() -> bool:
    """Check if PyMeshLab is available."""
    return PYMESHLAB_AVAILABLE


def trimesh_to_pymeshlab(mesh: trimesh.Trimesh) -> "pymeshlab.MeshSet":
    """
    Convert a trimesh mesh to a PyMeshLab MeshSet.
    
    Args:
        mesh: trimesh.Trimesh object
        
    Returns:
        pymeshlab.MeshSet with the mesh loaded
    """
    if not PYMESHLAB_AVAILABLE:
        raise ImportError("pymeshlab is required for this action")
    
    # Export to temp file and load in pymeshlab
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "mesh.ply"
        mesh.export(str(tmp_path))
        
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(tmp_path))
        
    return ms


def pymeshlab_to_trimesh(ms: "pymeshlab.MeshSet") -> trimesh.Trimesh:
    """
    Convert a PyMeshLab MeshSet back to trimesh.
    
    Args:
        ms: pymeshlab.MeshSet object
        
    Returns:
        trimesh.Trimesh object
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "mesh.ply"
        ms.save_current_mesh(str(tmp_path))
        
        result = trimesh.load(str(tmp_path), force='mesh')
        
        # Handle Scene objects
        if isinstance(result, trimesh.Scene):
            result = trimesh.util.concatenate(list(result.geometry.values()))
        
    return result


# =============================================================================
# CLEANING & REPAIR FILTERS
# =============================================================================

@register_action(
    name="meshlab_repair",
    description="Comprehensive mesh repair using MeshLab filters (remove duplicates, unreferenced, non-manifold)",
    parameters={
        "remove_duplicate_faces": True,
        "remove_duplicate_vertices": True,
        "remove_unreferenced_vertices": True,
        "remove_zero_area_faces": True,
        "repair_non_manifold_edges": True,
        "repair_non_manifold_vertices": True,
    },
    risk_level="medium"
)
def action_meshlab_repair(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Comprehensive mesh repair using MeshLab's cleaning filters.
    
    Applies multiple cleaning operations in sequence:
    1. Remove duplicate vertices
    2. Remove duplicate faces
    3. Remove unreferenced vertices
    4. Remove zero-area faces
    5. Repair non-manifold edges (by removing)
    6. Repair non-manifold vertices (by splitting)
    
    This is often more robust than pymeshfix for complex meshes.
    """
    if not PYMESHLAB_AVAILABLE:
        logger.warning("pymeshlab not available, falling back to trimesh")
        mesh = mesh.copy()
        mesh.merge_vertices()
        mesh.remove_degenerate_faces()
        return mesh
    
    try:
        ms = trimesh_to_pymeshlab(mesh)
        
        # Remove duplicate vertices
        if params.get("remove_duplicate_vertices", True):
            try:
                ms.meshing_remove_duplicate_vertices()
            except Exception as e:
                logger.debug(f"Remove duplicate vertices failed: {e}")
        
        # Remove duplicate faces
        if params.get("remove_duplicate_faces", True):
            try:
                ms.meshing_remove_duplicate_faces()
            except Exception as e:
                logger.debug(f"Remove duplicate faces failed: {e}")
        
        # Remove unreferenced vertices
        if params.get("remove_unreferenced_vertices", True):
            try:
                ms.meshing_remove_unreferenced_vertices()
            except Exception as e:
                logger.debug(f"Remove unreferenced vertices failed: {e}")
        
        # Remove zero-area faces
        if params.get("remove_zero_area_faces", True):
            try:
                ms.meshing_remove_null_faces()
            except Exception as e:
                logger.debug(f"Remove zero-area faces failed: {e}")
        
        # Repair non-manifold edges by removing faces
        if params.get("repair_non_manifold_edges", True):
            try:
                ms.meshing_repair_non_manifold_edges()
            except Exception as e:
                logger.debug(f"Repair non-manifold edges failed: {e}")
        
        # Repair non-manifold vertices by splitting
        if params.get("repair_non_manifold_vertices", True):
            try:
                ms.meshing_repair_non_manifold_vertices()
            except Exception as e:
                logger.debug(f"Repair non-manifold vertices failed: {e}")
        
        return pymeshlab_to_trimesh(ms)
        
    except Exception as e:
        logger.error(f"meshlab_repair failed: {e}")
        return mesh.copy()


@register_action(
    name="meshlab_close_holes",
    description="Close holes using MeshLab's hole-filling algorithm",
    parameters={
        "max_hole_size": 30,
        "self_intersection": True,
    },
    risk_level="medium"
)
def action_meshlab_close_holes(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Close holes using MeshLab's hole-filling algorithm.
    
    MeshLab's hole filling is often more robust than trimesh's,
    especially for non-planar holes.
    
    Args:
        mesh: Input mesh
        params:
            - max_hole_size: Maximum number of edges in holes to fill (default: 30)
            - self_intersection: Check for self-intersection (default: True)
    """
    if not PYMESHLAB_AVAILABLE:
        logger.warning("pymeshlab not available, falling back to trimesh")
        mesh = mesh.copy()
        trimesh.repair.fill_holes(mesh)
        return mesh
    
    max_hole_size = params.get("max_hole_size", 30)
    self_intersection = params.get("self_intersection", True)
    
    try:
        ms = trimesh_to_pymeshlab(mesh)
        
        ms.meshing_close_holes(
            maxholesize=max_hole_size,
            selfintersection=self_intersection
        )
        
        return pymeshlab_to_trimesh(ms)
        
    except Exception as e:
        logger.error(f"meshlab_close_holes failed: {e}")
        return mesh.copy()


@register_action(
    name="meshlab_remove_isolated",
    description="Remove isolated/floating components by diameter percentage",
    parameters={
        "diameter_percentage": 10.0,
    },
    risk_level="medium"
)
def action_meshlab_remove_isolated(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Remove isolated (floating) mesh components.
    
    Removes connected components whose diameter is smaller than
    the specified percentage of the main mesh diameter.
    
    Args:
        mesh: Input mesh
        params:
            - diameter_percentage: Remove components smaller than this % of main diameter
    """
    if not PYMESHLAB_AVAILABLE:
        logger.warning("pymeshlab not available, falling back to trimesh")
        # Trimesh fallback: keep largest component
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            return max(components, key=lambda c: len(c.faces))
        return mesh.copy()
    
    diameter_pct = params.get("diameter_percentage", 10.0)
    
    try:
        ms = trimesh_to_pymeshlab(mesh)
        
        ms.meshing_remove_connected_component_by_diameter(
            mincomponentdiag=pymeshlab.PercentageValue(diameter_pct)
        )
        
        return pymeshlab_to_trimesh(ms)
        
    except Exception as e:
        logger.error(f"meshlab_remove_isolated failed: {e}")
        return mesh.copy()


@register_action(
    name="meshlab_remove_small_components",
    description="Remove small connected components by face count",
    parameters={
        "min_face_count": 50,
    },
    risk_level="medium"
)
def action_meshlab_remove_small_components(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Remove small connected components by face count.
    
    Args:
        mesh: Input mesh
        params:
            - min_face_count: Minimum faces to keep a component (default: 50)
    """
    if not PYMESHLAB_AVAILABLE:
        logger.warning("pymeshlab not available, falling back to trimesh")
        from .trimesh_actions import action_remove_small_components
        return action_remove_small_components(mesh, params)
    
    min_faces = params.get("min_face_count", 50)
    
    try:
        ms = trimesh_to_pymeshlab(mesh)
        
        ms.meshing_remove_connected_component_by_face_number(
            mincomponentsize=min_faces
        )
        
        return pymeshlab_to_trimesh(ms)
        
    except Exception as e:
        logger.error(f"meshlab_remove_small_components failed: {e}")
        return mesh.copy()


# =============================================================================
# REMESHING FILTERS
# =============================================================================

@register_action(
    name="meshlab_remesh_isotropic",
    description="Isotropic explicit remeshing for uniform triangle distribution",
    parameters={
        "target_edge_length": None,  # Auto-calculate if None
        "iterations": 3,
        "adaptive": False,
    },
    risk_level="medium"
)
def action_meshlab_remesh_isotropic(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Isotropic explicit remeshing.
    
    Creates a new mesh with uniform, well-shaped triangles.
    Good for cleaning up noisy or irregular meshes.
    
    Args:
        mesh: Input mesh
        params:
            - target_edge_length: Target edge length (auto if None)
            - iterations: Number of remeshing iterations (default: 3)
            - adaptive: Use adaptive remeshing (default: False)
    """
    if not PYMESHLAB_AVAILABLE:
        logger.warning("pymeshlab not available, cannot perform isotropic remesh")
        return mesh.copy()
    
    target_length = params.get("target_edge_length")
    iterations = params.get("iterations", 3)
    adaptive = params.get("adaptive", False)
    
    try:
        ms = trimesh_to_pymeshlab(mesh)
        
        # Auto-calculate target edge length if not specified
        if target_length is None:
            # Use average edge length or bbox-based calculation
            bbox = mesh.bounds
            bbox_diagonal = np.linalg.norm(bbox[1] - bbox[0])
            target_length = bbox_diagonal / 100  # 1% of bbox diagonal
        
        # Use percentage value for targetlen
        ms.meshing_isotropic_explicit_remeshing(
            targetlen=pymeshlab.AbsoluteValue(target_length),
            iterations=iterations,
            adaptive=adaptive
        )
        
        return pymeshlab_to_trimesh(ms)
        
    except Exception as e:
        logger.error(f"meshlab_remesh_isotropic failed: {e}")
        return mesh.copy()


@register_action(
    name="meshlab_simplify_quadric",
    description="Simplify mesh using quadric edge collapse decimation",
    parameters={
        "target_faces": None,
        "target_percentage": 50.0,
        "quality_threshold": 0.3,
        "preserve_boundary": True,
        "preserve_topology": True,
    },
    risk_level="medium"
)
def action_meshlab_simplify_quadric(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Simplify mesh using quadric edge collapse decimation.
    
    MeshLab's implementation is often more robust than trimesh's
    and offers more control over the simplification.
    
    Args:
        mesh: Input mesh
        params:
            - target_faces: Target number of faces (overrides percentage)
            - target_percentage: Target percentage of faces to keep (default: 50%)
            - quality_threshold: Quality threshold 0-1 (default: 0.3)
            - preserve_boundary: Preserve mesh boundaries (default: True)
            - preserve_topology: Preserve mesh topology (default: True)
    """
    if not PYMESHLAB_AVAILABLE:
        logger.warning("pymeshlab not available, falling back to trimesh")
        from .trimesh_actions import action_decimate
        return action_decimate(mesh, params)
    
    target_faces = params.get("target_faces")
    target_pct = params.get("target_percentage", 50.0)
    quality = params.get("quality_threshold", 0.3)
    preserve_boundary = params.get("preserve_boundary", True)
    preserve_topology = params.get("preserve_topology", True)
    
    try:
        ms = trimesh_to_pymeshlab(mesh)
        
        # Calculate target faces
        if target_faces is None:
            current_faces = len(mesh.faces)
            target_faces = int(current_faces * target_pct / 100)
        
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_faces,
            qualitythr=quality,
            preserveboundary=preserve_boundary,
            preservetopology=preserve_topology
        )
        
        return pymeshlab_to_trimesh(ms)
        
    except Exception as e:
        logger.error(f"meshlab_simplify_quadric failed: {e}")
        return mesh.copy()


@register_action(
    name="meshlab_subdivide_midpoint",
    description="Subdivide mesh using midpoint subdivision",
    parameters={
        "iterations": 1,
    },
    risk_level="low"
)
def action_meshlab_subdivide_midpoint(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Subdivide mesh using midpoint subdivision.
    
    Each triangle is split into 4 triangles by adding midpoint vertices.
    """
    if not PYMESHLAB_AVAILABLE:
        logger.warning("pymeshlab not available, falling back to trimesh")
        from .trimesh_actions import action_subdivide
        return action_subdivide(mesh, params)
    
    iterations = params.get("iterations", 1)
    
    try:
        ms = trimesh_to_pymeshlab(mesh)
        
        for _ in range(iterations):
            ms.meshing_surface_subdivision_midpoint()
        
        return pymeshlab_to_trimesh(ms)
        
    except Exception as e:
        logger.error(f"meshlab_subdivide_midpoint failed: {e}")
        return mesh.copy()


# =============================================================================
# SMOOTHING FILTERS
# =============================================================================

@register_action(
    name="meshlab_smooth_laplacian",
    description="Laplacian smoothing using MeshLab",
    parameters={
        "iterations": 3,
        "boundary_smoothing": True,
        "cotangent_weight": True,
    },
    risk_level="medium"
)
def action_meshlab_smooth_laplacian(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Apply Laplacian smoothing using MeshLab.
    
    Args:
        mesh: Input mesh
        params:
            - iterations: Number of smoothing iterations (default: 3)
            - boundary_smoothing: Smooth boundary vertices too (default: True)
            - cotangent_weight: Use cotangent weighting (default: True)
    """
    if not PYMESHLAB_AVAILABLE:
        logger.warning("pymeshlab not available, falling back to trimesh")
        from .trimesh_actions import action_smooth
        return action_smooth(mesh, {"iterations": params.get("iterations", 3)})
    
    iterations = params.get("iterations", 3)
    boundary = params.get("boundary_smoothing", True)
    cotangent = params.get("cotangent_weight", True)
    
    try:
        ms = trimesh_to_pymeshlab(mesh)
        
        ms.apply_coord_laplacian_smoothing(
            stepsmoothnum=iterations,
            boundary=boundary,
            cotangentweight=cotangent
        )
        
        return pymeshlab_to_trimesh(ms)
        
    except Exception as e:
        logger.error(f"meshlab_smooth_laplacian failed: {e}")
        return mesh.copy()


@register_action(
    name="meshlab_smooth_taubin",
    description="Taubin smoothing (reduces shrinkage compared to Laplacian)",
    parameters={
        "lambda_factor": 0.5,
        "mu_factor": -0.53,
        "iterations": 10,
    },
    risk_level="medium"
)
def action_meshlab_smooth_taubin(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Apply Taubin smoothing using MeshLab.
    
    Taubin smoothing alternates between positive and negative
    smoothing factors to reduce the shrinkage effect of
    standard Laplacian smoothing.
    """
    if not PYMESHLAB_AVAILABLE:
        logger.warning("pymeshlab not available, falling back to trimesh")
        from .trimesh_actions import action_smooth
        return action_smooth(mesh, {"method": "taubin", "iterations": params.get("iterations", 10)})
    
    lambda_val = params.get("lambda_factor", 0.5)
    mu_val = params.get("mu_factor", -0.53)
    iterations = params.get("iterations", 10)
    
    try:
        ms = trimesh_to_pymeshlab(mesh)
        
        ms.apply_coord_taubin_smoothing(
            lambda_=lambda_val,
            mu=mu_val,
            stepsmoothnum=iterations
        )
        
        return pymeshlab_to_trimesh(ms)
        
    except Exception as e:
        logger.error(f"meshlab_smooth_taubin failed: {e}")
        return mesh.copy()


# =============================================================================
# NORMALS FILTERS
# =============================================================================

@register_action(
    name="meshlab_recompute_normals",
    description="Recompute vertex and face normals",
    parameters={
        "face_normals": True,
        "vertex_normals": True,
    },
    risk_level="low"
)
def action_meshlab_recompute_normals(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Recompute face and/or vertex normals using MeshLab.
    """
    if not PYMESHLAB_AVAILABLE:
        logger.warning("pymeshlab not available, falling back to trimesh")
        mesh = mesh.copy()
        mesh.fix_normals()
        return mesh
    
    face_normals = params.get("face_normals", True)
    vertex_normals = params.get("vertex_normals", True)
    
    try:
        ms = trimesh_to_pymeshlab(mesh)
        
        if face_normals:
            ms.meshing_re_orient_faces_coherentely()
        
        if vertex_normals:
            ms.compute_normal_per_vertex()
        
        return pymeshlab_to_trimesh(ms)
        
    except Exception as e:
        logger.error(f"meshlab_recompute_normals failed: {e}")
        return mesh.copy()


@register_action(
    name="meshlab_flip_normals",
    description="Flip/invert all face normals",
    parameters={},
    risk_level="low"
)
def action_meshlab_flip_normals(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Flip all face normals using MeshLab.
    """
    if not PYMESHLAB_AVAILABLE:
        logger.warning("pymeshlab not available, falling back to trimesh")
        from .trimesh_actions import action_flip_normals
        return action_flip_normals(mesh, params)
    
    try:
        ms = trimesh_to_pymeshlab(mesh)
        
        ms.meshing_invert_face_orientation()
        
        return pymeshlab_to_trimesh(ms)
        
    except Exception as e:
        logger.error(f"meshlab_flip_normals failed: {e}")
        return mesh.copy()


# =============================================================================
# SELECTION & ANALYSIS FILTERS
# =============================================================================

@register_action(
    name="meshlab_select_self_intersecting",
    description="Select and optionally remove self-intersecting faces",
    parameters={
        "remove": False,
    },
    risk_level="medium"
)
def action_meshlab_select_self_intersecting(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Select self-intersecting faces and optionally remove them.
    
    Args:
        mesh: Input mesh
        params:
            - remove: If True, remove the selected faces (default: False)
    """
    if not PYMESHLAB_AVAILABLE:
        logger.warning("pymeshlab not available")
        return mesh.copy()
    
    remove = params.get("remove", False)
    
    try:
        ms = trimesh_to_pymeshlab(mesh)
        
        # Select self-intersecting faces
        ms.compute_selection_by_self_intersections_per_face()
        
        if remove:
            ms.meshing_remove_selected_faces()
            ms.meshing_remove_unreferenced_vertices()
        
        return pymeshlab_to_trimesh(ms)
        
    except Exception as e:
        logger.error(f"meshlab_select_self_intersecting failed: {e}")
        return mesh.copy()


@register_action(
    name="meshlab_select_non_manifold",
    description="Select and optionally remove non-manifold edges/vertices",
    parameters={
        "remove": False,
    },
    risk_level="medium"
)
def action_meshlab_select_non_manifold(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Select non-manifold edges and vertices, optionally remove them.
    """
    if not PYMESHLAB_AVAILABLE:
        logger.warning("pymeshlab not available")
        return mesh.copy()
    
    remove = params.get("remove", False)
    
    try:
        ms = trimesh_to_pymeshlab(mesh)
        
        # Select non-manifold edges
        ms.compute_selection_by_non_manifold_edges_per_face()
        
        if remove:
            ms.meshing_remove_selected_faces()
            ms.meshing_remove_unreferenced_vertices()
        
        return pymeshlab_to_trimesh(ms)
        
    except Exception as e:
        logger.error(f"meshlab_select_non_manifold failed: {e}")
        return mesh.copy()


# =============================================================================
# TRANSFORMATION FILTERS
# =============================================================================

@register_action(
    name="meshlab_transform_align_to_axes",
    description="Align mesh principal axes to coordinate axes",
    parameters={},
    risk_level="low"
)
def action_meshlab_transform_align_to_axes(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Align mesh to coordinate axes using PCA.
    
    Rotates the mesh so its principal axes align with X, Y, Z axes.
    Useful for normalizing mesh orientation.
    """
    if not PYMESHLAB_AVAILABLE:
        logger.warning("pymeshlab not available")
        return mesh.copy()
    
    try:
        ms = trimesh_to_pymeshlab(mesh)
        
        ms.compute_matrix_by_principal_axis()
        ms.apply_matrix_freeze()
        
        return pymeshlab_to_trimesh(ms)
        
    except Exception as e:
        logger.error(f"meshlab_transform_align_to_axes failed: {e}")
        return mesh.copy()


@register_action(
    name="meshlab_transform_center",
    description="Center mesh at origin",
    parameters={
        "center_on_bbox": True,
    },
    risk_level="low"
)
def action_meshlab_transform_center(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Center mesh at origin.
    
    Args:
        mesh: Input mesh
        params:
            - center_on_bbox: Use bounding box center (True) or barycenter (False)
    """
    if not PYMESHLAB_AVAILABLE:
        logger.warning("pymeshlab not available, falling back to trimesh")
        from .trimesh_actions import action_center_mesh
        return action_center_mesh(mesh, params)
    
    use_bbox = params.get("center_on_bbox", True)
    
    try:
        ms = trimesh_to_pymeshlab(mesh)
        
        if use_bbox:
            ms.compute_matrix_from_translation(
                traslmethod='Center on Layer BBox'
            )
        else:
            ms.compute_matrix_from_translation(
                traslmethod='Center on Scene BBox'
            )
        ms.apply_matrix_freeze()
        
        return pymeshlab_to_trimesh(ms)
        
    except Exception as e:
        logger.error(f"meshlab_transform_center failed: {e}")
        return mesh.copy()


@register_action(
    name="meshlab_transform_scale_to_unit",
    description="Scale mesh to fit in unit bounding box",
    parameters={
        "uniform": True,
    },
    risk_level="low"
)
def action_meshlab_transform_scale_to_unit(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Scale mesh to fit in a unit bounding box.
    
    Useful for normalizing mesh size before processing.
    """
    if not PYMESHLAB_AVAILABLE:
        logger.warning("pymeshlab not available")
        # Manual scaling fallback
        mesh = mesh.copy()
        bbox = mesh.bounds
        scale = 1.0 / max(bbox[1] - bbox[0])
        mesh.apply_scale(scale)
        return mesh
    
    uniform = params.get("uniform", True)
    
    try:
        ms = trimesh_to_pymeshlab(mesh)
        
        ms.compute_matrix_from_scaling_or_normalization(
            scalecenter='barycenter',
            unitflag=True,
            uniformflag=uniform
        )
        ms.apply_matrix_freeze()
        
        return pymeshlab_to_trimesh(ms)
        
    except Exception as e:
        logger.error(f"meshlab_transform_scale_to_unit failed: {e}")
        return mesh.copy()


# =============================================================================
# ADDITIONAL SUBDIVISION FILTERS
# =============================================================================

@register_action(
    name="meshlab_subdivide_butterfly",
    description="Butterfly subdivision for smooth surface refinement",
    parameters={
        "iterations": 1,
        "threshold": 1.0,
    },
    risk_level="medium"
)
def action_meshlab_subdivide_butterfly(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Apply Butterfly subdivision.
    
    Butterfly subdivision creates smooth surfaces while
    interpolating original vertices (they don't move).
    Good for creating smooth surfaces from coarse meshes.
    """
    if not PYMESHLAB_AVAILABLE:
        logger.warning("pymeshlab not available")
        return mesh.copy()
    
    iterations = params.get("iterations", 1)
    threshold = params.get("threshold", 1.0)
    
    try:
        ms = trimesh_to_pymeshlab(mesh)
        
        for _ in range(iterations):
            ms.meshing_surface_subdivision_butterfly(
                threshold=pymeshlab.PercentageValue(threshold)
            )
        
        return pymeshlab_to_trimesh(ms)
        
    except Exception as e:
        logger.error(f"meshlab_subdivide_butterfly failed: {e}")
        return mesh.copy()


@register_action(
    name="meshlab_subdivide_catmull_clark",
    description="Catmull-Clark subdivision for smooth quad-based refinement",
    parameters={
        "iterations": 1,
    },
    risk_level="medium"
)
def action_meshlab_subdivide_catmull_clark(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Apply Catmull-Clark subdivision.
    
    Creates smooth surfaces using quad-based subdivision.
    The result is re-triangulated for STL compatibility.
    """
    if not PYMESHLAB_AVAILABLE:
        logger.warning("pymeshlab not available")
        return mesh.copy()
    
    iterations = params.get("iterations", 1)
    
    try:
        ms = trimesh_to_pymeshlab(mesh)
        
        for _ in range(iterations):
            ms.meshing_surface_subdivision_catmull_clark()
        
        return pymeshlab_to_trimesh(ms)
        
    except Exception as e:
        logger.error(f"meshlab_subdivide_catmull_clark failed: {e}")
        return mesh.copy()


@register_action(
    name="meshlab_subdivide_loop",
    description="Loop subdivision for smooth triangle-based refinement",
    parameters={
        "iterations": 1,
        "threshold": 1.0,
    },
    risk_level="medium"
)
def action_meshlab_subdivide_loop(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Apply Loop subdivision using MeshLab.
    
    Loop subdivision is designed for triangle meshes and
    creates smooth surfaces by subdividing each triangle into four.
    """
    if not PYMESHLAB_AVAILABLE:
        logger.warning("pymeshlab not available")
        return mesh.copy()
    
    iterations = params.get("iterations", 1)
    threshold = params.get("threshold", 1.0)
    
    try:
        ms = trimesh_to_pymeshlab(mesh)
        
        for _ in range(iterations):
            ms.meshing_surface_subdivision_loop(
                threshold=pymeshlab.PercentageValue(threshold)
            )
        
        return pymeshlab_to_trimesh(ms)
        
    except Exception as e:
        logger.error(f"meshlab_subdivide_loop failed: {e}")
        return mesh.copy()


@register_action(
    name="meshlab_subdivide_ls3",
    description="LS3 Loop subdivision (better handling of boundaries)",
    parameters={
        "iterations": 1,
    },
    risk_level="medium"
)
def action_meshlab_subdivide_ls3(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Apply LS3 Loop subdivision.
    
    An improved version of Loop subdivision with better
    handling of boundary edges and irregular vertices.
    """
    if not PYMESHLAB_AVAILABLE:
        logger.warning("pymeshlab not available")
        return mesh.copy()
    
    iterations = params.get("iterations", 1)
    
    try:
        ms = trimesh_to_pymeshlab(mesh)
        
        for _ in range(iterations):
            ms.meshing_surface_subdivision_ls3_loop()
        
        return pymeshlab_to_trimesh(ms)
        
    except Exception as e:
        logger.error(f"meshlab_subdivide_ls3 failed: {e}")
        return mesh.copy()


# =============================================================================
# ADDITIONAL CLEANING FILTERS
# =============================================================================

@register_action(
    name="meshlab_remove_folded_faces",
    description="Remove folded/overlapping faces",
    parameters={},
    risk_level="medium"
)
def action_meshlab_remove_folded_faces(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Remove folded (overlapping) faces.
    
    Removes faces that are folded over other faces,
    which can cause rendering and slicing issues.
    """
    if not PYMESHLAB_AVAILABLE:
        logger.warning("pymeshlab not available")
        return mesh.copy()
    
    try:
        ms = trimesh_to_pymeshlab(mesh)
        ms.meshing_remove_folded_faces()
        return pymeshlab_to_trimesh(ms)
        
    except Exception as e:
        logger.error(f"meshlab_remove_folded_faces failed: {e}")
        return mesh.copy()


@register_action(
    name="meshlab_remove_t_vertices",
    description="Remove T-vertices (vertices on edges of other faces)",
    parameters={},
    risk_level="medium"
)
def action_meshlab_remove_t_vertices(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Remove T-vertices.
    
    T-vertices occur when a vertex lies on the edge of another
    face rather than at a corner. They can cause non-manifold issues.
    """
    if not PYMESHLAB_AVAILABLE:
        logger.warning("pymeshlab not available")
        return mesh.copy()
    
    try:
        ms = trimesh_to_pymeshlab(mesh)
        ms.meshing_remove_t_vertices()
        return pymeshlab_to_trimesh(ms)
        
    except Exception as e:
        logger.error(f"meshlab_remove_t_vertices failed: {e}")
        return mesh.copy()


@register_action(
    name="meshlab_snap_borders",
    description="Snap mismatched border edges together",
    parameters={
        "threshold": 1.0,
    },
    risk_level="medium"
)
def action_meshlab_snap_borders(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Snap mismatched border edges.
    
    Attempts to close small gaps between nearly-matching
    border edges by snapping them together.
    """
    if not PYMESHLAB_AVAILABLE:
        logger.warning("pymeshlab not available")
        return mesh.copy()
    
    threshold = params.get("threshold", 1.0)
    
    try:
        ms = trimesh_to_pymeshlab(mesh)
        ms.meshing_snap_mismatched_borders(
            edgedistratio=threshold
        )
        return pymeshlab_to_trimesh(ms)
        
    except Exception as e:
        logger.error(f"meshlab_snap_borders failed: {e}")
        return mesh.copy()


@register_action(
    name="meshlab_merge_close_vertices",
    description="Merge vertices that are very close together",
    parameters={
        "threshold": 0.001,
    },
    risk_level="medium"
)
def action_meshlab_merge_close_vertices(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Merge close vertices.
    
    Merges vertices that are within the specified threshold distance.
    Useful for closing small gaps and reducing vertex count.
    """
    if not PYMESHLAB_AVAILABLE:
        logger.warning("pymeshlab not available")
        return mesh.copy()
    
    threshold = params.get("threshold", 0.001)
    
    try:
        ms = trimesh_to_pymeshlab(mesh)
        ms.meshing_merge_close_vertices(
            threshold=pymeshlab.AbsoluteValue(threshold)
        )
        return pymeshlab_to_trimesh(ms)
        
    except Exception as e:
        logger.error(f"meshlab_merge_close_vertices failed: {e}")
        return mesh.copy()


# =============================================================================
# DECIMATION VARIANTS
# =============================================================================

@register_action(
    name="meshlab_decimate_clustering",
    description="Fast clustering-based decimation",
    parameters={
        "threshold": 1.0,
    },
    risk_level="medium"
)
def action_meshlab_decimate_clustering(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Clustering-based decimation.
    
    A fast decimation method that clusters nearby vertices.
    Less accurate than quadric decimation but faster.
    """
    if not PYMESHLAB_AVAILABLE:
        logger.warning("pymeshlab not available")
        return mesh.copy()
    
    threshold = params.get("threshold", 1.0)
    
    try:
        ms = trimesh_to_pymeshlab(mesh)
        ms.meshing_decimation_clustering(
            threshold=pymeshlab.PercentageValue(threshold)
        )
        return pymeshlab_to_trimesh(ms)
        
    except Exception as e:
        logger.error(f"meshlab_decimate_clustering failed: {e}")
        return mesh.copy()


# =============================================================================
# SURFACE RECONSTRUCTION
# =============================================================================

@register_action(
    name="meshlab_reconstruct_ball_pivoting",
    description="Surface reconstruction using ball pivoting algorithm",
    parameters={
        "ball_radius": 0.0,
        "clustering": 20.0,
        "angle_threshold": 90.0,
    },
    risk_level="high"
)
def action_meshlab_reconstruct_ball_pivoting(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Ball pivoting surface reconstruction.
    
    Reconstructs the mesh surface using the ball pivoting algorithm.
    Good for point clouds or meshes with many holes.
    """
    if not PYMESHLAB_AVAILABLE:
        logger.warning("pymeshlab not available")
        return mesh.copy()
    
    ball_radius = params.get("ball_radius", 0.0)  # 0 = auto
    clustering = params.get("clustering", 20.0)
    angle = params.get("angle_threshold", 90.0)
    
    try:
        ms = trimesh_to_pymeshlab(mesh)
        
        # Ball pivoting needs normals
        ms.compute_normal_per_vertex()
        
        ms.generate_surface_reconstruction_ball_pivoting(
            ballradius=pymeshlab.PercentageValue(ball_radius) if ball_radius > 0 else pymeshlab.Percentage(0),
            clustering=clustering,
            creasethr=angle
        )
        
        return pymeshlab_to_trimesh(ms)
        
    except Exception as e:
        logger.error(f"meshlab_reconstruct_ball_pivoting failed: {e}")
        return mesh.copy()


@register_action(
    name="meshlab_reconstruct_poisson",
    description="Screened Poisson surface reconstruction",
    parameters={
        "depth": 8,
        "fullDepth": 5,
        "scale": 1.1,
    },
    risk_level="high"
)
def action_meshlab_reconstruct_poisson(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Screened Poisson surface reconstruction.
    
    Creates a watertight mesh from the input using Poisson reconstruction.
    This can create a completely new surface that approximates the original.
    """
    if not PYMESHLAB_AVAILABLE:
        logger.warning("pymeshlab not available")
        return mesh.copy()
    
    depth = params.get("depth", 8)
    full_depth = params.get("fullDepth", 5)
    scale = params.get("scale", 1.1)
    
    try:
        ms = trimesh_to_pymeshlab(mesh)
        
        # Poisson needs good normals
        ms.compute_normal_per_vertex()
        
        ms.generate_surface_reconstruction_screened_poisson(
            depth=depth,
            fulldepth=full_depth,
            scale=scale
        )
        
        return pymeshlab_to_trimesh(ms)
        
    except Exception as e:
        logger.error(f"meshlab_reconstruct_poisson failed: {e}")
        return mesh.copy()


@register_action(
    name="meshlab_alpha_wrap",
    description="Alpha wrap for watertight mesh generation",
    parameters={
        "alpha": 1.0,
        "offset": 1.0,
    },
    risk_level="high"
)
def action_meshlab_alpha_wrap(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Generate alpha wrap.
    
    Creates a watertight outer hull using alpha wrapping.
    Very effective for creating printable meshes from complex geometry.
    """
    if not PYMESHLAB_AVAILABLE:
        logger.warning("pymeshlab not available")
        return mesh.copy()
    
    alpha = params.get("alpha", 1.0)
    offset = params.get("offset", 1.0)
    
    try:
        ms = trimesh_to_pymeshlab(mesh)
        
        ms.generate_alpha_wrap(
            alpha=pymeshlab.PercentageValue(alpha),
            offset=pymeshlab.PercentageValue(offset)
        )
        
        return pymeshlab_to_trimesh(ms)
        
    except Exception as e:
        logger.error(f"meshlab_alpha_wrap failed: {e}")
        return mesh.copy()


# =============================================================================
# CONVEX HULL AND OTHER GEOMETRY
# =============================================================================

@register_action(
    name="meshlab_convex_hull",
    description="Generate convex hull of mesh",
    parameters={},
    risk_level="high"
)
def action_meshlab_convex_hull(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Generate convex hull.
    
    Creates the convex hull of the mesh - the smallest convex
    shape that contains all vertices.
    """
    if not PYMESHLAB_AVAILABLE:
        logger.warning("pymeshlab not available, using trimesh")
        try:
            return mesh.convex_hull
        except Exception:
            return mesh.copy()
    
    try:
        ms = trimesh_to_pymeshlab(mesh)
        ms.generate_convex_hull()
        return pymeshlab_to_trimesh(ms)
        
    except Exception as e:
        logger.error(f"meshlab_convex_hull failed: {e}")
        return mesh.copy()


@register_action(
    name="meshlab_alpha_shape",
    description="Generate alpha shape (tight-fitting hull)",
    parameters={
        "alpha": 1.0,
    },
    risk_level="high"
)
def action_meshlab_alpha_shape(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Generate alpha shape.
    
    Creates an alpha shape - a hull that can follow concavities
    more closely than a convex hull.
    """
    if not PYMESHLAB_AVAILABLE:
        logger.warning("pymeshlab not available")
        return mesh.copy()
    
    alpha = params.get("alpha", 1.0)
    
    try:
        ms = trimesh_to_pymeshlab(mesh)
        ms.generate_alpha_shape(
            alpha=pymeshlab.PercentageValue(alpha)
        )
        return pymeshlab_to_trimesh(ms)
        
    except Exception as e:
        logger.error(f"meshlab_alpha_shape failed: {e}")
        return mesh.copy()


@register_action(
    name="meshlab_solid_wireframe",
    description="Create solid wireframe version of mesh edges",
    parameters={
        "edge_radius": 0.02,
        "vertex_radius": 0.04,
    },
    risk_level="high"
)
def action_meshlab_solid_wireframe(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Generate solid wireframe.
    
    Creates a printable wireframe model by replacing edges with
    cylinders and vertices with spheres.
    """
    if not PYMESHLAB_AVAILABLE:
        logger.warning("pymeshlab not available")
        return mesh.copy()
    
    edge_radius = params.get("edge_radius", 0.02)
    vertex_radius = params.get("vertex_radius", 0.04)
    
    try:
        ms = trimesh_to_pymeshlab(mesh)
        
        ms.generate_solid_wireframe(
            edgecylradius=pymeshlab.PercentageValue(edge_radius * 100),
            vertexsphradius=pymeshlab.PercentageValue(vertex_radius * 100)
        )
        
        return pymeshlab_to_trimesh(ms)
        
    except Exception as e:
        logger.error(f"meshlab_solid_wireframe failed: {e}")
        return mesh.copy()


# =============================================================================
# EDGE OPTIMIZATION
# =============================================================================

@register_action(
    name="meshlab_optimize_edges_curvature",
    description="Optimize mesh by flipping edges based on curvature",
    parameters={
        "iterations": 10,
    },
    risk_level="medium"
)
def action_meshlab_optimize_edges_curvature(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Edge flip optimization by curvature.
    
    Improves mesh quality by flipping edges to better align
    with surface curvature.
    """
    if not PYMESHLAB_AVAILABLE:
        logger.warning("pymeshlab not available")
        return mesh.copy()
    
    iterations = params.get("iterations", 10)
    
    try:
        ms = trimesh_to_pymeshlab(mesh)
        
        for _ in range(iterations):
            ms.meshing_edge_flip_by_curvature_optimization()
        
        return pymeshlab_to_trimesh(ms)
        
    except Exception as e:
        logger.error(f"meshlab_optimize_edges_curvature failed: {e}")
        return mesh.copy()


@register_action(
    name="meshlab_optimize_edges_planar",
    description="Optimize mesh by flipping edges for planar optimization",
    parameters={
        "iterations": 10,
        "threshold": 1.0,
    },
    risk_level="medium"
)
def action_meshlab_optimize_edges_planar(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Edge flip optimization for planarity.
    
    Improves mesh quality by flipping edges to make faces
    more coplanar where appropriate.
    """
    if not PYMESHLAB_AVAILABLE:
        logger.warning("pymeshlab not available")
        return mesh.copy()
    
    iterations = params.get("iterations", 10)
    threshold = params.get("threshold", 1.0)
    
    try:
        ms = trimesh_to_pymeshlab(mesh)
        
        for _ in range(iterations):
            ms.meshing_edge_flip_by_planar_optimization(
                planartype='area',
                threshold=threshold
            )
        
        return pymeshlab_to_trimesh(ms)
        
    except Exception as e:
        logger.error(f"meshlab_optimize_edges_planar failed: {e}")
        return mesh.copy()


# =============================================================================
# TRI TO QUAD CONVERSION
# =============================================================================

@register_action(
    name="meshlab_tri_to_quad",
    description="Convert triangles to quads where possible",
    parameters={},
    risk_level="medium"
)
def action_meshlab_tri_to_quad(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Convert triangles to quads.
    
    Pairs adjacent triangles into quads where possible.
    Useful for modeling workflows. Note: Output is re-triangulated
    for STL export.
    """
    if not PYMESHLAB_AVAILABLE:
        logger.warning("pymeshlab not available")
        return mesh.copy()
    
    try:
        ms = trimesh_to_pymeshlab(mesh)
        ms.meshing_tri_to_quad_by_smart_triangle_pairing()
        # Re-triangulate for STL compatibility
        ms.meshing_poly_to_tri()
        return pymeshlab_to_trimesh(ms)
        
    except Exception as e:
        logger.error(f"meshlab_tri_to_quad failed: {e}")
        return mesh.copy()


# =============================================================================
# CURVATURE-BASED OPERATIONS
# =============================================================================

@register_action(
    name="meshlab_compute_curvature",
    description="Compute and optionally visualize surface curvature",
    parameters={
        "method": "mean",  # mean, gaussian, rms, abs
    },
    risk_level="low"
)
def action_meshlab_compute_curvature(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Compute surface curvature.
    
    Calculates surface curvature values. This is primarily a
    diagnostic/analysis action - the mesh is returned unchanged.
    """
    # This is primarily for analysis - return mesh unchanged
    return mesh.copy()


# =============================================================================
# RESAMPLING
# =============================================================================

@register_action(
    name="meshlab_resample_uniform",
    description="Resample mesh to uniform vertex distribution",
    parameters={
        "cell_size": 1.0,
        "offset": 50.0,
    },
    risk_level="high"
)
def action_meshlab_resample_uniform(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Uniform mesh resampling.
    
    Creates a new mesh with uniformly distributed vertices.
    Can improve mesh quality but may change topology.
    """
    if not PYMESHLAB_AVAILABLE:
        logger.warning("pymeshlab not available")
        return mesh.copy()
    
    cell_size = params.get("cell_size", 1.0)
    offset = params.get("offset", 50.0)
    
    try:
        ms = trimesh_to_pymeshlab(mesh)
        
        ms.generate_resampled_uniform_mesh(
            cellsize=pymeshlab.PercentageValue(cell_size),
            offset=pymeshlab.PercentageValue(offset)
        )
        
        return pymeshlab_to_trimesh(ms)
        
    except Exception as e:
        logger.error(f"meshlab_resample_uniform failed: {e}")
        return mesh.copy()


# =============================================================================
# BOOLEAN OPERATIONS
# =============================================================================

@register_action(
    name="meshlab_boolean_union",
    description="Boolean union of all mesh components",
    parameters={},
    risk_level="high"
)
def action_meshlab_boolean_union(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Boolean union of all mesh components using MeshLab.
    
    Note: This requires multiple components in the mesh.
    """
    if not PYMESHLAB_AVAILABLE:
        logger.warning("pymeshlab not available")
        return mesh.copy()
    
    try:
        # Split mesh into components
        components = mesh.split(only_watertight=False)
        
        if len(components) < 2:
            logger.info("meshlab_boolean_union: Only one component, nothing to union")
            return mesh.copy()
        
        ms = pymeshlab.MeshSet()
        
        # Load each component as a separate mesh
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, comp in enumerate(components):
                tmp_path = Path(tmpdir) / f"comp_{i}.ply"
                comp.export(str(tmp_path))
                ms.load_new_mesh(str(tmp_path))
        
        # Perform boolean union iteratively
        # Set mesh 0 as current
        ms.set_current_mesh(0)
        
        for i in range(1, len(components)):
            try:
                ms.generate_boolean_union(first_mesh=0, second_mesh=i)
            except Exception as e:
                logger.warning(f"Boolean union step {i} failed: {e}")
        
        return pymeshlab_to_trimesh(ms)
        
    except Exception as e:
        logger.error(f"meshlab_boolean_union failed: {e}")
        return mesh.copy()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_pymeshlab_version() -> Optional[str]:
    """Get PyMeshLab version string."""
    if not PYMESHLAB_AVAILABLE:
        return None
    try:
        return pymeshlab.__version__
    except Exception:
        return "unknown"


def list_pymeshlab_filters() -> list:
    """List available PyMeshLab filters (for debugging)."""
    if not PYMESHLAB_AVAILABLE:
        return []
    try:
        ms = pymeshlab.MeshSet()
        return [name for name in dir(ms) if not name.startswith('_')]
    except Exception:
        return []
