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

# Try to import error logging
try:
    from .error_logging import log_action_failure
    ERROR_LOGGING_AVAILABLE = True
except ImportError:
    ERROR_LOGGING_AVAILABLE = False
    log_action_failure = None


def _log_pymeshfix_failure(action_name: str, error_message: str, mesh: trimesh.Trimesh) -> None:
    """Log a pymeshfix action failure."""
    if ERROR_LOGGING_AVAILABLE and log_action_failure:
        log_action_failure(
            action_name=action_name,
            error_message=error_message,
            mesh=mesh,
            action_type="pymeshfix",
        )


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
    
    WARNING: With joincomp=True, multi-component models may lose geometry!
    For models with intentional multiple parts, use joincomp=False or
    use pymeshfix_repair_conservative instead.
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
    
    # Store original metrics for comparison
    original_faces = len(mesh.faces)
    original_vertices = len(mesh.vertices)
    original_bounds = mesh.bounds
    original_bbox_diagonal = float(np.linalg.norm(original_bounds[1] - original_bounds[0]))
    
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
        
        # Check for significant data loss
        result_faces = len(result.faces)
        result_bounds = result.bounds
        result_bbox_diagonal = float(np.linalg.norm(result_bounds[1] - result_bounds[0]))
        
        face_loss_pct = (original_faces - result_faces) / original_faces * 100 if original_faces > 0 else 0
        bbox_change_pct = abs(original_bbox_diagonal - result_bbox_diagonal) / original_bbox_diagonal * 100 if original_bbox_diagonal > 0 else 0
        
        if face_loss_pct > 50:
            logger.warning(f"pymeshfix_repair: {face_loss_pct:.1f}% face loss detected! Model may have lost significant geometry.")
        if bbox_change_pct > 30:
            logger.warning(f"pymeshfix_repair: Bounding box changed by {bbox_change_pct:.1f}%! Model dimensions significantly altered.")
        
        return result
        
    except Exception as e:
        logger.error(f"pymeshfix repair failed: {e}")
        _log_pymeshfix_failure("pymeshfix_repair", str(e), mesh)
        # Return original mesh on failure
        return mesh.copy()


@register_action(
    name="pymeshfix_repair_conservative",
    description="Conservative mesh repair - preserves geometry when possible",
    parameters={"verbose": False, "max_face_loss_pct": 50.0},
    risk_level="low"
)
def action_pymeshfix_repair_conservative(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Conservative mesh repair using PyMeshFix.
    
    This repairs each component separately, preserving multi-part models.
    If repair would destroy too much geometry (>50% faces), it keeps the original.
    
    Use this for models with intentional multiple components (e.g., assemblies)
    or when aggressive repair destroys too much geometry.
    """
    if not PYMESHFIX_AVAILABLE:
        logger.warning("pymeshfix not available, falling back to trimesh repair")
        mesh = mesh.copy()
        trimesh.repair.fill_holes(mesh)
        mesh.fix_normals()
        return mesh
    
    verbose = params.get("verbose", False)
    max_face_loss_pct = params.get("max_face_loss_pct", 50.0)
    
    # Store original metrics
    original_faces = len(mesh.faces)
    original_vertices = len(mesh.vertices)
    original_bbox_diagonal = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    
    try:
        # Split into components
        components = mesh.split(only_watertight=False)
        
        if len(components) <= 1:
            # Single component - try repair but check for destruction
            meshfix = trimesh_to_pymeshfix(mesh)
            meshfix.repair(verbose=verbose, joincomp=False)
            result = pymeshfix_to_trimesh(meshfix)
            
            # Check if repair destroyed the mesh
            result_faces = len(result.faces)
            result_bbox_diagonal = float(np.linalg.norm(result.bounds[1] - result.bounds[0]))
            
            face_loss_pct = (original_faces - result_faces) / original_faces * 100 if original_faces > 0 else 0
            bbox_change_pct = abs(original_bbox_diagonal - result_bbox_diagonal) / original_bbox_diagonal * 100 if original_bbox_diagonal > 0 else 0
            
            if face_loss_pct > max_face_loss_pct or bbox_change_pct > 50:
                logger.warning(
                    f"pymeshfix_repair_conservative: Repair would destroy {face_loss_pct:.1f}% faces "
                    f"(bbox change {bbox_change_pct:.1f}%). Keeping original mesh."
                )
                # Return original but with minor fixes
                result = mesh.copy()
                result.fix_normals()
                return result
            
            return result
        
        logger.info(f"pymeshfix_repair_conservative: Repairing {len(components)} components separately")
        
        # Repair each component separately
        repaired_components = []
        for i, comp in enumerate(components):
            if len(comp.faces) < 4:  # Skip tiny fragments (less than a tetrahedron)
                logger.debug(f"  Skipping tiny component {i} with {len(comp.faces)} faces")
                continue
            
            comp_faces_before = len(comp.faces)
            comp_bbox_before = float(np.linalg.norm(comp.bounds[1] - comp.bounds[0]))
            
            try:
                meshfix = trimesh_to_pymeshfix(comp)
                meshfix.repair(verbose=verbose, joincomp=False)
                repaired = pymeshfix_to_trimesh(meshfix)
                
                # Check if repair destroyed this component
                comp_faces_after = len(repaired.faces)
                face_loss = (comp_faces_before - comp_faces_after) / comp_faces_before * 100 if comp_faces_before > 0 else 0
                
                if face_loss > max_face_loss_pct:
                    logger.warning(f"  Component {i}: repair lost {face_loss:.1f}% faces, keeping original")
                    repaired_components.append(comp)
                else:
                    repaired_components.append(repaired)
            except Exception as e:
                logger.warning(f"  Component {i} repair failed: {e}, keeping original")
                repaired_components.append(comp)
        
        if repaired_components:
            result = trimesh.util.concatenate(repaired_components)
            logger.info(f"pymeshfix_repair_conservative: Preserved {len(repaired_components)} components")
            return result
        else:
            logger.warning("pymeshfix_repair_conservative: No components survived, returning original")
            return mesh.copy()
        
    except Exception as e:
        logger.error(f"pymeshfix_repair_conservative failed: {e}")
        _log_pymeshfix_failure("pymeshfix_repair_conservative", str(e), mesh)
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
        _log_pymeshfix_failure("pymeshfix_clean", str(e), mesh)
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
        _log_pymeshfix_failure("make_manifold", str(e), mesh)
        return mesh.copy()
