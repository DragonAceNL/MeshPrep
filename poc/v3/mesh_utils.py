# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Mesh processing utilities for POC v3 batch processing.

Contains functions for mesh analysis, geometry checking, decimation,
and diagnostic extraction.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import adaptive thresholds
try:
    from meshprep_poc.adaptive_thresholds import get_adaptive_thresholds
    ADAPTIVE_THRESHOLDS_AVAILABLE = True
except ImportError:
    ADAPTIVE_THRESHOLDS_AVAILABLE = False
    get_adaptive_thresholds = None

from config import (
    DEFAULT_VOLUME_LOSS_LIMIT_PCT,
    DEFAULT_FACE_LOSS_LIMIT_PCT,
    DEFAULT_DECIMATION_TARGET_FACES,
)


def check_geometry_loss(
    original_diag, 
    result_mesh, 
    profile: str = "unknown"
) -> Tuple[bool, float, float]:
    """Check if repair caused significant geometry loss.
    
    Uses adaptive thresholds if available, otherwise falls back to defaults.
    
    Args:
        original_diag: Diagnostics from the original mesh
        result_mesh: The repaired trimesh object
        profile: Model profile for threshold lookup
        
    Returns:
        Tuple of (significant_loss, volume_loss_pct, face_loss_pct)
    """
    # Get thresholds (adaptive or defaults)
    if ADAPTIVE_THRESHOLDS_AVAILABLE:
        thresholds = get_adaptive_thresholds()
        volume_limit = thresholds.get("volume_loss_limit_pct", profile)
        face_limit = thresholds.get("face_loss_limit_pct", profile)
    else:
        volume_limit = DEFAULT_VOLUME_LOSS_LIMIT_PCT
        face_limit = DEFAULT_FACE_LOSS_LIMIT_PCT
    
    original_volume = original_diag.volume if original_diag.volume > 0 else 0
    result_volume = result_mesh.volume if result_mesh.is_volume else 0
    
    volume_loss_pct = 0.0
    if original_volume > 0:
        volume_loss_pct = abs(original_volume - result_volume) / original_volume * 100
    
    original_faces = original_diag.face_count
    result_faces = len(result_mesh.faces)
    face_loss_pct = 0.0
    if original_faces > 0:
        face_loss_pct = (original_faces - result_faces) / original_faces * 100
    
    significant_loss = volume_loss_pct > volume_limit or face_loss_pct > face_limit
    
    return significant_loss, volume_loss_pct, face_loss_pct


def decimate_mesh(mesh, target_faces: Optional[int] = None, profile: str = "unknown"):
    """Decimate mesh to reduce face count while preserving shape.
    
    Uses adaptive thresholds for target if not specified.
    
    Args:
        mesh: Trimesh object to decimate
        target_faces: Target face count (uses adaptive threshold if None)
        profile: Model profile for threshold lookup
        
    Returns:
        Decimated mesh (or original if already small enough)
    """
    # Get target from adaptive thresholds if not specified
    if target_faces is None:
        if ADAPTIVE_THRESHOLDS_AVAILABLE:
            thresholds = get_adaptive_thresholds()
            target_faces = int(thresholds.get("decimation_target_faces", profile))
        else:
            target_faces = DEFAULT_DECIMATION_TARGET_FACES
    
    if len(mesh.faces) <= target_faces:
        return mesh
    
    try:
        # Try fast_simplification first (best quality)
        import fast_simplification
        
        # Get vertices and faces as numpy arrays
        vertices = mesh.vertices.copy()
        faces = mesh.faces.copy()
        
        # Calculate target ratio
        ratio = target_faces / len(faces)
        
        # Simplify
        new_verts, new_faces = fast_simplification.simplify(
            vertices, faces, 
            target_reduction=1.0 - ratio,
            agg=5  # Aggression level (1-10)
        )
        
        # Create new mesh
        import trimesh
        decimated = trimesh.Trimesh(vertices=new_verts, faces=new_faces)
        
        if len(decimated.faces) > 0:
            logger.info(f"  Decimated: {len(mesh.faces):,} -> {len(decimated.faces):,} faces")
            return decimated
            
    except ImportError:
        logger.warning("  fast_simplification not installed, trying trimesh method")
    except Exception as e:
        logger.warning(f"  fast_simplification failed: {e}")
    
    # Fallback to trimesh's built-in method
    try:
        decimated = mesh.simplify_quadric_decimation(target_faces)
        if decimated is not None and len(decimated.faces) > 0:
            logger.info(f"  Decimated (trimesh): {len(mesh.faces):,} -> {len(decimated.faces):,} faces")
            return decimated
    except Exception as e:
        logger.warning(f"  Trimesh decimation failed: {e}")
    
    return mesh


def extract_mesh_diagnostics(mesh, label: str = "") -> Optional[Dict[str, Any]]:
    """Extract comprehensive mesh diagnostics for analysis.
    
    This captures mesh characteristics useful for:
    - Model profile detection
    - Filter script selection
    - Understanding repair outcomes
    
    Args:
        mesh: Trimesh object to analyze
        label: Optional label for logging
        
    Returns:
        Dictionary of diagnostics, or None if mesh is None
    """
    if mesh is None:
        return None
    
    try:
        diagnostics = {
            # Basic geometry
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "edges": len(mesh.edges_unique) if hasattr(mesh, 'edges_unique') else 0,
            
            # Volume and bounds
            "volume": float(mesh.volume) if mesh.is_watertight else None,
            "bounding_box": {
                "min": mesh.bounds[0].tolist() if mesh.bounds is not None else None,
                "max": mesh.bounds[1].tolist() if mesh.bounds is not None else None,
            },
            "extents": mesh.extents.tolist() if hasattr(mesh, 'extents') else None,
            
            # Topology
            "is_watertight": bool(mesh.is_watertight),
            "is_winding_consistent": bool(mesh.is_winding_consistent) if hasattr(mesh, 'is_winding_consistent') else None,
            "euler_number": int(mesh.euler_number) if hasattr(mesh, 'euler_number') else None,
            
            # Body count (fragmentation)
            "body_count": len(mesh.split(only_watertight=False)) if hasattr(mesh, 'split') else 1,
            
            # Face analysis
            "degenerate_faces": int(mesh.degenerate_faces.sum()) if hasattr(mesh, 'degenerate_faces') else 0,
            
            # Area
            "surface_area": float(mesh.area) if hasattr(mesh, 'area') else None,
        }
        
        # Try to get additional diagnostics
        try:
            # Check for non-manifold edges
            if hasattr(mesh, 'edges_unique') and hasattr(mesh, 'faces'):
                # Simple heuristic: non-manifold if we have unusual edge-face relationships
                edges = mesh.edges_sorted.reshape(-1, 2)
                edge_counts = {}
                for e in map(tuple, edges):
                    edge_counts[e] = edge_counts.get(e, 0) + 1
                non_manifold_edges = sum(1 for c in edge_counts.values() if c > 2)
                diagnostics["non_manifold_edges"] = non_manifold_edges
        except:
            pass
        
        return diagnostics
        
    except Exception as e:
        return {
            "error": f"Failed to extract diagnostics: {str(e)}",
            "vertices": len(mesh.vertices) if hasattr(mesh, 'vertices') else 0,
            "faces": len(mesh.faces) if hasattr(mesh, 'faces') else 0,
        }


def render_mesh_image(mesh, output_path: Path, title: str = "") -> bool:
    """Render mesh to image using matplotlib.
    
    Args:
        mesh: Trimesh object to render
        output_path: Path for the output image
        title: Optional title for the image
        
    Returns:
        True if rendering succeeded, False otherwise
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import numpy as np
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get mesh data
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Limit faces for performance
        max_faces = 50000
        if len(faces) > max_faces:
            indices = np.random.choice(len(faces), max_faces, replace=False)
            faces = faces[indices]
        
        # Create polygon collection
        mesh_faces = vertices[faces]
        collection = Poly3DCollection(mesh_faces, alpha=0.8, linewidth=0.1, edgecolor='gray')
        collection.set_facecolor([0.3, 0.6, 0.9])
        ax.add_collection3d(collection)
        
        # Set axis limits
        scale = vertices.max() - vertices.min()
        center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
        ax.set_xlim(center[0] - scale/2, center[0] + scale/2)
        ax.set_ylim(center[1] - scale/2, center[1] + scale/2)
        ax.set_zlim(center[2] - scale/2, center[2] + scale/2)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        if title:
            ax.set_title(title)
        
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        return True
    except Exception as e:
        logger.warning(f"Failed to render image: {e}")
        return False
