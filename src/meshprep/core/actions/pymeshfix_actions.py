# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
PyMeshFix-based actions for mesh repair.

PyMeshFix is a Python wrapper for MeshFix, which can repair
non-manifold meshes and fill holes robustly.

Note: pymeshfix requires Python 3.11 or 3.12 (no pre-built wheels for 3.13+).
"""

import numpy as np
import trimesh
import logging

from .registry import register_action

logger = logging.getLogger(__name__)

# Track pymeshfix availability
try:
    import pymeshfix
    PYMESHFIX_AVAILABLE = True
except ImportError:
    PYMESHFIX_AVAILABLE = False
    logger.warning("pymeshfix not available - some repair actions will be limited")


def is_pymeshfix_available() -> bool:
    """Check if pymeshfix is available."""
    return PYMESHFIX_AVAILABLE


def trimesh_to_pymeshfix(mesh: trimesh.Trimesh):
    """
    Convert trimesh to pymeshfix MeshFix object.
    
    Args:
        mesh: Input trimesh object
        
    Returns:
        pymeshfix.MeshFix object
        
    Raises:
        ImportError: If pymeshfix is not installed
    """
    if not PYMESHFIX_AVAILABLE:
        raise ImportError(
            "pymeshfix is required for this action. "
            "Install with: pip install pymeshfix "
            "(requires Python 3.11 or 3.12)"
        )
    
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    
    return pymeshfix.MeshFix(vertices, faces)


def pymeshfix_to_trimesh(meshfix) -> trimesh.Trimesh:
    """
    Convert pymeshfix result back to trimesh.
    
    Args:
        meshfix: pymeshfix.MeshFix object with repaired mesh
        
    Returns:
        trimesh.Trimesh object
    """
    vertices = meshfix.v
    faces = meshfix.f
    
    return trimesh.Trimesh(vertices=vertices, faces=faces)


@register_action(
    name="pymeshfix_repair",
    description="Full mesh repair using PyMeshFix - fixes holes and non-manifold geometry",
    parameters={
        "verbose": {"type": "bool", "default": False, "description": "Show repair progress"},
        "joincomp": {"type": "bool", "default": True, "description": "Join components into one"},
        "remove_smallest_components": {"type": "bool", "default": True, "description": "Remove small components"}
    },
    risk_level="medium",
    tool="pymeshfix",
    category="Repair & Manifold Fixes"
)
def action_pymeshfix_repair(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Full mesh repair using PyMeshFix.
    
    This is a powerful repair operation that can:
    - Fill holes
    - Fix non-manifold edges and vertices
    - Make the mesh watertight
    - Optionally join components
    
    Note: PyMeshFix may modify the mesh topology significantly.
    Falls back to trimesh repair if pymeshfix is not available.
    """
    if not PYMESHFIX_AVAILABLE:
        logger.warning("pymeshfix not available, falling back to trimesh repair")
        mesh = mesh.copy()
        trimesh.repair.fill_holes(mesh)
        mesh.fix_normals()
        return mesh
    
    verbose = params.get("verbose", False)
    joincomp = params.get("joincomp", True)
    remove_smallest = params.get("remove_smallest_components", True)
    
    try:
        # Convert to pymeshfix
        meshfix = trimesh_to_pymeshfix(mesh)
        
        # Run repair
        meshfix.repair(verbose=verbose, joincomp=joincomp)
        
        # Convert back
        result = pymeshfix_to_trimesh(meshfix)
        
        # Optionally remove small components
        if remove_smallest and len(result.faces) > 0:
            components = result.split(only_watertight=False)
            if components and len(components) > 1:
                # Keep only components larger than 1% of total faces
                total_faces = len(result.faces)
                threshold = max(4, total_faces * 0.01)  # At least 4 faces
                kept = [c for c in components if len(c.faces) >= threshold]
                if kept:
                    if len(kept) == 1:
                        result = kept[0]
                    else:
                        result = trimesh.util.concatenate(kept)
        
        return result
        
    except Exception as e:
        logger.error(f"pymeshfix repair failed: {e}")
        # Return original mesh on failure
        return mesh.copy()


@register_action(
    name="pymeshfix_clean",
    description="Light cleanup using PyMeshFix - removes self-intersections",
    parameters={"verbose": {"type": "bool", "default": False, "description": "Show progress"}},
    risk_level="low",
    tool="pymeshfix",
    category="Loading & Basic Cleanup"
)
def action_pymeshfix_clean(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Light cleanup using PyMeshFix.
    
    Performs basic cleaning without full repair.
    Falls back to trimesh cleanup if pymeshfix is not available.
    """
    if not PYMESHFIX_AVAILABLE:
        logger.warning("pymeshfix not available, using trimesh cleanup")
        mesh = mesh.copy()
        try:
            mesh.remove_degenerate_faces()
            mesh.remove_duplicate_faces()
            mesh.fix_normals()
        except Exception:
            pass
        return mesh
    
    verbose = params.get("verbose", False)
    
    try:
        meshfix = trimesh_to_pymeshfix(mesh)
        meshfix.repair(verbose=verbose, joincomp=False)
        return pymeshfix_to_trimesh(meshfix)
        
    except Exception as e:
        logger.error(f"pymeshfix clean failed: {e}")
        return mesh.copy()


@register_action(
    name="make_manifold",
    description="Make mesh manifold using PyMeshFix",
    parameters={},
    risk_level="medium",
    tool="pymeshfix",
    category="Repair & Manifold Fixes"
)
def action_make_manifold(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Make mesh manifold.
    
    Fixes non-manifold edges and vertices to create a valid
    2-manifold surface suitable for 3D printing.
    Falls back to basic trimesh repair if pymeshfix is not available.
    """
    if not PYMESHFIX_AVAILABLE:
        logger.warning("pymeshfix not available for manifold repair")
        mesh = mesh.copy()
        try:
            mesh.fix_normals()
            trimesh.repair.fill_holes(mesh)
        except Exception:
            pass
        return mesh
    
    try:
        meshfix = trimesh_to_pymeshfix(mesh)
        meshfix.repair(verbose=False, joincomp=True)
        return pymeshfix_to_trimesh(meshfix)
        
    except Exception as e:
        logger.error(f"make_manifold failed: {e}")
        return mesh.copy()


@register_action(
    name="fill_holes_pymeshfix",
    description="Use pymeshfix's hole-filling algorithm for robust repairs",
    parameters={},
    risk_level="medium",
    tool="pymeshfix",
    category="Hole Filling"
)
def action_fill_holes_pymeshfix(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Fill holes using PyMeshFix.
    
    More robust than trimesh's fill_holes for complex cases.
    Falls back to trimesh fill_holes if pymeshfix is not available.
    """
    if not PYMESHFIX_AVAILABLE:
        logger.warning("pymeshfix not available, using trimesh fill_holes")
        mesh = mesh.copy()
        trimesh.repair.fill_holes(mesh)
        return mesh
    
    try:
        meshfix = trimesh_to_pymeshfix(mesh)
        meshfix.repair(verbose=False, joincomp=False)
        return pymeshfix_to_trimesh(meshfix)
        
    except Exception as e:
        logger.error(f"pymeshfix fill_holes failed: {e}")
        mesh = mesh.copy()
        trimesh.repair.fill_holes(mesh)
        return mesh
