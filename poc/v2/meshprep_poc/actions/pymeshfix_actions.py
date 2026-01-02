# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
PyMeshFix-based actions for mesh repair.

PyMeshFix is a Python wrapper for MeshFix, which can repair
non-manifold meshes and fill holes robustly.
"""

import numpy as np
import trimesh
import logging

from .registry import register_action

logger = logging.getLogger(__name__)

# Try to import pymeshfix
try:
    import pymeshfix
    PYMESHFIX_AVAILABLE = True
except ImportError:
    PYMESHFIX_AVAILABLE = False
    logger.warning("pymeshfix not available - some repair actions will be limited")


def trimesh_to_pymeshfix(mesh: trimesh.Trimesh):
    """Convert trimesh to pymeshfix MeshFix object."""
    if not PYMESHFIX_AVAILABLE:
        raise ImportError("pymeshfix is required for this action")
    
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    
    return pymeshfix.MeshFix(vertices, faces)


def pymeshfix_to_trimesh(meshfix) -> trimesh.Trimesh:
    """Convert pymeshfix result back to trimesh."""
    vertices = meshfix.v
    faces = meshfix.f
    
    return trimesh.Trimesh(vertices=vertices, faces=faces)


@register_action(
    name="pymeshfix_repair",
    description="Full mesh repair using PyMeshFix - fixes holes and non-manifold geometry",
    parameters={"verbose": False, "joincomp": True, "remove_smallest_components": True},
    risk_level="medium"
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
    """
    if not PYMESHFIX_AVAILABLE:
        logger.warning("pymeshfix not available, falling back to trimesh repair")
        # Fallback to basic trimesh operations
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
        if remove_smallest:
            components = result.split(only_watertight=False)
            if len(components) > 1:
                # Keep only components larger than 1% of total faces
                total_faces = len(result.faces)
                threshold = total_faces * 0.01
                kept = [c for c in components if len(c.faces) >= threshold]
                if kept:
                    result = trimesh.util.concatenate(kept)
        
        return result
        
    except Exception as e:
        logger.error(f"pymeshfix repair failed: {e}")
        # Return original mesh on failure
        return mesh.copy()


@register_action(
    name="pymeshfix_clean",
    description="Light cleanup using PyMeshFix - removes self-intersections",
    parameters={"verbose": False},
    risk_level="low"
)
def action_pymeshfix_clean(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Light cleanup using PyMeshFix.
    
    Performs basic cleaning without full repair.
    """
    if not PYMESHFIX_AVAILABLE:
        logger.warning("pymeshfix not available, using trimesh cleanup")
        mesh = mesh.copy()
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.fix_normals()
        return mesh
    
    verbose = params.get("verbose", False)
    
    try:
        meshfix = trimesh_to_pymeshfix(mesh)
        
        # Just clean, don't do full repair
        # This removes self-intersections and small components
        meshfix.repair(verbose=verbose, joincomp=False)
        
        return pymeshfix_to_trimesh(meshfix)
        
    except Exception as e:
        logger.error(f"pymeshfix clean failed: {e}")
        return mesh.copy()


@register_action(
    name="make_manifold",
    description="Make mesh manifold using PyMeshFix",
    parameters={},
    risk_level="medium"
)
def action_make_manifold(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Make mesh manifold.
    
    Fixes non-manifold edges and vertices to create a valid
    2-manifold surface.
    """
    if not PYMESHFIX_AVAILABLE:
        logger.warning("pymeshfix not available for manifold repair")
        # Basic trimesh fallback
        mesh = mesh.copy()
        mesh.fix_normals()
        return mesh
    
    try:
        meshfix = trimesh_to_pymeshfix(mesh)
        meshfix.repair(verbose=False, joincomp=True)
        return pymeshfix_to_trimesh(meshfix)
        
    except Exception as e:
        logger.error(f"make_manifold failed: {e}")
        return mesh.copy()
