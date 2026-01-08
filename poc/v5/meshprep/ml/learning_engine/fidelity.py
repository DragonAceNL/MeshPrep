# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Fidelity Validation - Measures how well repairs preserve original geometry.

Computes:
- Hausdorff distance (max surface deviation)
- Volume change percentage
- Bounding box change
- Quality score (1-5)

Used by the learning loop to evaluate repair outcomes.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from scipy.spatial import cKDTree
import logging

logger = logging.getLogger(__name__)


@dataclass
class FidelityMetrics:
    """Fidelity metrics comparing original and repaired mesh."""
    
    # Volume
    original_volume: float = 0.0
    repaired_volume: float = 0.0
    volume_change_pct: float = 0.0
    
    # Surface distance
    hausdorff_distance: float = 0.0
    hausdorff_relative: float = 0.0  # Relative to bbox diagonal
    mean_surface_distance: float = 0.0
    
    # Bounding box
    bbox_change_pct: float = 0.0
    
    # Surface area
    area_change_pct: float = 0.0
    
    # Printability
    is_printable: bool = False
    is_watertight: bool = False
    is_manifold: bool = False
    
    @property
    def quality_score(self) -> int:
        """Compute quality score 1-5 from metrics."""
        return compute_quality_score(self)


def compute_hausdorff_distance(
    original,
    repaired,
    num_samples: int = 10000,
) -> Tuple[float, float]:
    """
    Compute Hausdorff distance between two meshes.
    
    Args:
        original: Original trimesh
        repaired: Repaired trimesh
        num_samples: Number of surface samples
        
    Returns:
        Tuple of (hausdorff_distance, mean_distance)
    """
    try:
        # Sample points from both surfaces
        samples_orig = original.sample(num_samples)
        samples_rep = repaired.sample(num_samples)
        
        # Build KD-trees
        tree_orig = cKDTree(samples_orig)
        tree_rep = cKDTree(samples_rep)
        
        # Forward distances: original -> repaired
        dist_o2r, _ = tree_rep.query(samples_orig)
        
        # Backward distances: repaired -> original  
        dist_r2o, _ = tree_orig.query(samples_rep)
        
        # Hausdorff is max of both directions
        hausdorff = max(dist_o2r.max(), dist_r2o.max())
        
        # Mean surface distance (symmetric)
        mean_dist = (dist_o2r.mean() + dist_r2o.mean()) / 2
        
        return float(hausdorff), float(mean_dist)
        
    except Exception as e:
        logger.warning(f"Could not compute Hausdorff distance: {e}")
        return 0.0, 0.0


def compute_fidelity_metrics(
    original,
    repaired,
    num_samples: int = 10000,
) -> FidelityMetrics:
    """
    Compute all fidelity metrics between original and repaired mesh.
    
    Args:
        original: Original mesh (Mesh or trimesh.Trimesh)
        repaired: Repaired mesh (Mesh or trimesh.Trimesh)
        num_samples: Number of samples for Hausdorff computation
        
    Returns:
        FidelityMetrics with all measurements
    """
    # Handle Mesh wrapper
    orig_tm = original.trimesh if hasattr(original, 'trimesh') else original
    rep_tm = repaired.trimesh if hasattr(repaired, 'trimesh') else repaired
    
    metrics = FidelityMetrics()
    
    # Volume comparison
    try:
        metrics.original_volume = abs(float(orig_tm.volume)) if orig_tm.is_volume else 0.0
        metrics.repaired_volume = abs(float(rep_tm.volume)) if rep_tm.is_volume else 0.0
        
        if metrics.original_volume > 0:
            metrics.volume_change_pct = (
                (metrics.repaired_volume - metrics.original_volume) /
                metrics.original_volume * 100
            )
    except Exception as e:
        logger.debug(f"Could not compute volume: {e}")
    
    # Bounding box
    try:
        orig_diagonal = np.linalg.norm(orig_tm.bounds[1] - orig_tm.bounds[0])
        rep_diagonal = np.linalg.norm(rep_tm.bounds[1] - rep_tm.bounds[0])
        
        if orig_diagonal > 0:
            metrics.bbox_change_pct = abs(rep_diagonal - orig_diagonal) / orig_diagonal * 100
    except Exception as e:
        logger.debug(f"Could not compute bbox: {e}")
    
    # Hausdorff distance
    try:
        orig_diagonal = np.linalg.norm(orig_tm.bounds[1] - orig_tm.bounds[0])
        
        hausdorff, mean_dist = compute_hausdorff_distance(orig_tm, rep_tm, num_samples)
        metrics.hausdorff_distance = hausdorff
        metrics.mean_surface_distance = mean_dist
        
        if orig_diagonal > 0:
            metrics.hausdorff_relative = hausdorff / orig_diagonal
    except Exception as e:
        logger.debug(f"Could not compute Hausdorff: {e}")
    
    # Surface area
    try:
        orig_area = float(orig_tm.area)
        rep_area = float(rep_tm.area)
        
        if orig_area > 0:
            metrics.area_change_pct = (rep_area - orig_area) / orig_area * 100
    except Exception as e:
        logger.debug(f"Could not compute area: {e}")
    
    # Printability checks
    metrics.is_watertight = rep_tm.is_watertight
    metrics.is_manifold = rep_tm.is_volume
    metrics.is_printable = metrics.is_watertight and metrics.is_manifold
    
    return metrics


def compute_quality_score(metrics: FidelityMetrics) -> int:
    """
    Compute quality score 1-5 from fidelity metrics.
    
    Scoring:
    - 5: Perfect - indistinguishable from original
    - 4: Good - minor changes, fully usable
    - 3: Acceptable - noticeable changes but recognizable
    - 2: Poor - significant changes, may need review
    - 1: Rejected - unrecognizable or destroyed
    
    Args:
        metrics: Computed fidelity metrics
        
    Returns:
        Quality score 1-5
    """
    score = 5.0
    
    # Volume change penalty (most important)
    vol_change = abs(metrics.volume_change_pct)
    if vol_change > 50:
        score -= 3.0
    elif vol_change > 30:
        score -= 2.0
    elif vol_change > 15:
        score -= 1.0
    elif vol_change > 5:
        score -= 0.5
    elif vol_change > 2:
        score -= 0.25
    
    # Hausdorff distance penalty
    hausdorff_pct = metrics.hausdorff_relative * 100
    if hausdorff_pct > 10:
        score -= 2.0
    elif hausdorff_pct > 5:
        score -= 1.5
    elif hausdorff_pct > 2:
        score -= 1.0
    elif hausdorff_pct > 1:
        score -= 0.5
    elif hausdorff_pct > 0.5:
        score -= 0.25
    
    # Bounding box change
    if metrics.bbox_change_pct > 10:
        score -= 1.5
    elif metrics.bbox_change_pct > 5:
        score -= 1.0
    elif metrics.bbox_change_pct > 2:
        score -= 0.5
    
    # Area change
    area_change = abs(metrics.area_change_pct)
    if area_change > 50:
        score -= 1.0
    elif area_change > 30:
        score -= 0.5
    
    # Printability bonus/penalty
    if metrics.is_printable:
        score += 0.5
    else:
        score -= 0.5
    
    return max(1, min(5, round(score)))


def quick_quality_check(original, repaired) -> Tuple[int, bool, str]:
    """
    Quick quality check for a repair.
    
    Args:
        original: Original mesh
        repaired: Repaired mesh
        
    Returns:
        Tuple of (quality_score, is_printable, summary_message)
    """
    metrics = compute_fidelity_metrics(original, repaired, num_samples=5000)
    
    score = metrics.quality_score
    
    # Build summary
    parts = []
    
    if metrics.is_printable:
        parts.append("✓ Printable")
    else:
        parts.append("✗ Not printable")
    
    if abs(metrics.volume_change_pct) > 5:
        parts.append(f"Vol: {metrics.volume_change_pct:+.1f}%")
    
    if metrics.hausdorff_relative > 0.01:
        parts.append(f"Hausdorff: {metrics.hausdorff_relative*100:.2f}%")
    
    summary = " | ".join(parts)
    
    return score, metrics.is_printable, summary
