# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
CGAL-based mesh repair actions.

Uses CGAL's Alpha Wrap for guaranteed watertight mesh reconstruction.
"""

import logging
import tempfile
import os
import numpy as np
import trimesh

from . import register_action

logger = logging.getLogger(__name__)

# Check for CGAL availability
try:
    from CGAL.CGAL_Alpha_wrap_3 import alpha_wrap_3
    from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
    from CGAL.CGAL_Kernel import Point_3
    CGAL_AVAILABLE = True
    logger.info("CGAL Alpha Wrap available")
except ImportError:
    CGAL_AVAILABLE = False
    logger.debug("CGAL not available - alpha wrap action disabled")


def _trimesh_to_cgal_via_off(mesh: trimesh.Trimesh) -> "Polyhedron_3":
    """Convert trimesh to CGAL Polyhedron_3 via OFF file format."""
    if not CGAL_AVAILABLE:
        raise ImportError("CGAL is required for this operation")
    
    from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
    
    # Write to temp OFF file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.off', delete=False) as f:
        temp_path = f.name
        # Write OFF format manually
        f.write("OFF\n")
        f.write(f"{len(mesh.vertices)} {len(mesh.faces)} 0\n")
        for v in mesh.vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for face in mesh.faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    
    try:
        # Load into CGAL
        poly = Polyhedron_3(temp_path)
        return poly
    finally:
        # Clean up temp file
        os.unlink(temp_path)


def _cgal_to_trimesh_via_off(poly: "Polyhedron_3") -> trimesh.Trimesh:
    """Convert CGAL Polyhedron_3 to trimesh via OFF file format."""
    # Write to temp OFF file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.off', delete=False) as f:
        temp_path = f.name
    
    try:
        poly.write_to_file(temp_path)
        # Load into trimesh
        result = trimesh.load(temp_path)
        return result
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@register_action(
    name="cgal_alpha_wrap",
    description="CGAL Alpha Wrap - guaranteed watertight envelope",
    parameters={"relative_alpha": 2000.0, "relative_offset": 4000.0},
    risk_level="high"
)
def action_cgal_alpha_wrap(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    CGAL Alpha Wrap surface reconstruction.
    
    Creates a guaranteed watertight envelope around the input geometry.
    Excellent for extreme fragmentation where other methods fail.
    
    Parameter guide (from CGAL docs):
    - relative_alpha: Divisor for bbox diagonal to get alpha value
      Higher = smaller alpha = more detail preserved (but slower)
    - relative_offset: Divisor for bbox diagonal to get offset value  
      Higher = smaller offset = tighter wrap
    
    Quality presets:
    - Quick/coarse: alpha=20-100, offset=1000 (fast preview)
    - Good quality: alpha=500, offset=1200 (balanced)
    - High detail: alpha=1000, offset=2000 (good quality)
    - Ultra detail: alpha=2000, offset=4000 (maximum quality, default)
    
    Args:
        mesh: Input mesh (can be severely fragmented)
        params:
            - relative_alpha: Alpha divisor (default 2000, higher = more detail)
            - relative_offset: Offset divisor (default 4000, higher = tighter)
    
    Returns:
        Watertight wrapped mesh
    """
    if not CGAL_AVAILABLE:
        logger.warning("CGAL not available, falling back to original mesh")
        return mesh.copy()
    
    relative_alpha = params.get("relative_alpha", 2000.0)  # Ultra detail default
    relative_offset = params.get("relative_offset", 4000.0)  # Ultra tight wrap default
    
    try:
        from CGAL.CGAL_Alpha_wrap_3 import alpha_wrap_3
        from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
        
        # Calculate bounding box diagonal
        bbox = mesh.bounds
        bbox_diag = np.linalg.norm(bbox[1] - bbox[0])
        
        # Convert relative values to absolute
        alpha = bbox_diag / relative_alpha
        offset = bbox_diag / relative_offset
        
        logger.info(f"Running CGAL Alpha Wrap (alpha={alpha:.4f}, offset={offset:.4f})")
        logger.info(f"  BBox diagonal: {bbox_diag:.2f}")
        logger.info(f"  Input: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Convert to CGAL via OFF file
        logger.info("  Converting to CGAL format...")
        poly_in = _trimesh_to_cgal_via_off(mesh)
        
        logger.info(f"  CGAL Polyhedron: {poly_in.size_of_vertices()} vertices, {poly_in.size_of_facets()} facets")
        
        # Create output polyhedron
        poly_out = Polyhedron_3()
        
        # Run alpha wrap
        logger.info("  Running alpha_wrap_3...")
        alpha_wrap_3(poly_in, alpha, offset, poly_out)
        
        logger.info(f"  Output: {poly_out.size_of_vertices()} vertices, {poly_out.size_of_facets()} facets")
        
        if poly_out.size_of_vertices() == 0:
            logger.warning("Alpha wrap produced empty mesh, returning original")
            return mesh.copy()
        
        # Convert back to trimesh
        result = _cgal_to_trimesh_via_off(poly_out)
        
        logger.info(f"Alpha wrap complete: {len(result.vertices)} vertices, {len(result.faces)} faces")
        logger.info(f"  Is watertight: {result.is_watertight}")
        logger.info(f"  Result bounds: {result.bounds}")
        
        return result
        
    except Exception as e:
        logger.error(f"CGAL Alpha Wrap failed: {e}")
        import traceback
        traceback.print_exc()
        return mesh.copy()


def is_cgal_available() -> bool:
    """Check if CGAL is available."""
    return CGAL_AVAILABLE


@register_action(
    name="hc_laplacian_smooth",
    description="HC Laplacian smoothing - volume preserving",
    parameters={"iterations": 3},
    risk_level="medium"
)
def action_hc_laplacian_smooth(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Apply HC Laplacian smoothing to reduce stair-stepping artifacts.
    
    HC Laplacian is better than standard Laplacian for preserving
    volume and sharp features while smoothing the surface.
    
    Args:
        mesh: Input mesh
        params:
            - iterations: Number of smoothing passes (default 3)
    
    Returns:
        Smoothed mesh
    """
    iterations = params.get("iterations", 3)
    
    try:
        import pymeshlab
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
            temp_in = f.name
        mesh.export(temp_in)
        
        # Load into PyMeshLab
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(temp_in)
        
        # Apply HC Laplacian smoothing
        logger.info(f"Applying HC Laplacian smoothing ({iterations} iterations)...")
        for i in range(iterations):
            ms.apply_coord_hc_laplacian_smoothing()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
            temp_out = f.name
        ms.save_current_mesh(temp_out)
        
        # Load result
        result = trimesh.load(temp_out)
        
        # Cleanup
        os.unlink(temp_in)
        os.unlink(temp_out)
        
        logger.info(f"Smoothing complete: {len(result.vertices)} vertices, {len(result.faces)} faces")
        return result
        
    except Exception as e:
        logger.error(f"HC Laplacian smoothing failed: {e}")
        return mesh.copy()


@register_action(
    name="cgal_alpha_wrap_smooth",
    description="CGAL Alpha Wrap + HC Laplacian smoothing for best quality",
    parameters={"relative_alpha": 2000.0, "relative_offset": 4000.0, "smooth_iterations": 10},
    risk_level="high"
)
def action_cgal_alpha_wrap_smooth(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Combined CGAL Alpha Wrap + HC Laplacian smoothing.
    
    This produces the best quality results for fragmented meshes:
    1. Alpha wrap creates watertight envelope with maximum detail
    2. HC Laplacian smoothing removes stair-stepping artifacts
    
    See docs/extreme_fragmentation_guide.md for detailed parameter guidance.
    
    Args:
        mesh: Input mesh (can be severely fragmented)
        params:
            - relative_alpha: Alpha divisor (default 2000, higher = more detail)
            - relative_offset: Offset divisor (default 4000, higher = tighter wrap)
            - smooth_iterations: HC Laplacian iterations (default 10)
    
    Returns:
        Watertight, smooth mesh
    """
    # First do alpha wrap
    wrapped = action_cgal_alpha_wrap(mesh, {
        "relative_alpha": params.get("relative_alpha", 2000.0),
        "relative_offset": params.get("relative_offset", 4000.0)
    })
    
    # Then smooth (10 iterations by default for good stair-step removal)
    smoothed = action_hc_laplacian_smooth(wrapped, {
        "iterations": params.get("smooth_iterations", 10)
    })
    
    return smoothed


@register_action(
    name="isotropic_remesh",
    description="Isotropic remeshing - redistributes triangles evenly",
    parameters={"target_edge_percent": 0.5, "iterations": 3},
    risk_level="medium"
)
def action_isotropic_remesh(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Apply isotropic explicit remeshing to redistribute triangles evenly.
    
    This is useful after alpha wrap to:
    - Remove stair-stepping patterns
    - Reduce face count while preserving quality
    - Create more efficient meshes for slicing
    
    Args:
        mesh: Input mesh
        params:
            - target_edge_percent: Target edge length as % of bbox diagonal (default 0.5)
            - iterations: Number of remeshing iterations (default 3)
    
    Returns:
        Remeshed mesh with evenly distributed triangles
    """
    target_edge_percent = params.get("target_edge_percent", 0.5)
    iterations = params.get("iterations", 3)
    
    try:
        import pymeshlab
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
            temp_in = f.name
        mesh.export(temp_in)
        
        # Load into PyMeshLab
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(temp_in)
        
        logger.info(f"Applying isotropic remeshing (target edge: {target_edge_percent}% of bbox, {iterations} iterations)...")
        logger.info(f"  Input: {ms.current_mesh().vertex_number()} vertices, {ms.current_mesh().face_number()} faces")
        
        # Apply isotropic remeshing
        ms.meshing_isotropic_explicit_remeshing(
            iterations=iterations,
            targetlen=pymeshlab.PercentageValue(target_edge_percent)
        )
        
        logger.info(f"  Output: {ms.current_mesh().vertex_number()} vertices, {ms.current_mesh().face_number()} faces")
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
            temp_out = f.name
        ms.save_current_mesh(temp_out)
        
        # Load result
        result = trimesh.load(temp_out)
        
        # Cleanup
        os.unlink(temp_in)
        os.unlink(temp_out)
        
        return result
        
    except Exception as e:
        logger.error(f"Isotropic remeshing failed: {e}")
        return mesh.copy()


@register_action(
    name="cgal_alpha_wrap_remesh_smooth",
    description="CGAL Alpha Wrap + remesh + smooth - balanced quality and file size",
    parameters={"relative_alpha": 2000.0, "relative_offset": 4000.0, "target_edge_percent": 0.5, "smooth_iterations": 3},
    risk_level="high"
)
def action_cgal_alpha_wrap_remesh_smooth(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Combined CGAL Alpha Wrap + isotropic remeshing + HC Laplacian smoothing.
    
    This pipeline produces:
    1. Watertight envelope from alpha wrap
    2. Even triangle distribution from remeshing (reduces file size)
    3. Smooth surface from HC Laplacian
    
    Best for 3D printing where file size matters.
    
    Args:
        mesh: Input mesh (can be severely fragmented)
        params:
            - relative_alpha: Alpha divisor (default 2000)
            - relative_offset: Offset divisor (default 4000)
            - target_edge_percent: Remesh target edge % (default 0.5)
            - smooth_iterations: HC Laplacian iterations (default 3)
    
    Returns:
        Watertight, efficient, smooth mesh
    """
    # First do alpha wrap
    wrapped = action_cgal_alpha_wrap(mesh, {
        "relative_alpha": params.get("relative_alpha", 2000.0),
        "relative_offset": params.get("relative_offset", 4000.0)
    })
    
    # Then remesh for even triangle distribution
    remeshed = action_isotropic_remesh(wrapped, {
        "target_edge_percent": params.get("target_edge_percent", 0.5),
        "iterations": 3
    })
    
    # Then smooth
    smoothed = action_hc_laplacian_smooth(remeshed, {
        "iterations": params.get("smooth_iterations", 3)
    })
    
    return smoothed
