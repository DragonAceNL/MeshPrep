# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Smart filter pipeline combinations based on model profiles.

Each profile has recommended pipelines (ordered sequences of actions) that are
most likely to succeed. Pipelines are tried in order - each starting fresh
from the original mesh.

Design principles:
1. Each pipeline starts fresh from the ORIGINAL mesh (no stacking damage)
2. Pipelines are ordered by likelihood of success (try safest first)
3. Profile-specific pipelines are tried before generic fallbacks
4. Blender is only used as last resort (slow, can destroy fragmented models)
5. Each action in a pipeline builds on the previous action's result
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class FilterPipeline:
    """A sequence of filter actions to apply in order."""
    name: str
    description: str
    actions: List[Dict]  # List of {"action": str, "params": dict}
    priority: int = 1  # Lower = try first
    avoid_for_fragmented: bool = False  # Skip if model is fragmented
    
    def __repr__(self):
        action_names = " -> ".join(a["action"] for a in self.actions)
        return f"Pipeline({self.name}: {action_names})"


# =============================================================================
# PROFILE-SPECIFIC PIPELINES
# Organized by model profile category from docs/model_profiles.md
# =============================================================================

PROFILE_PIPELINES: Dict[str, List[FilterPipeline]] = {
    
    # =========================================================================
    # CLEAN / MINIMAL REPAIR
    # =========================================================================
    "clean": [
        FilterPipeline(
            name="validate-only",
            description="Model is already clean, just validate",
            actions=[
                {"action": "validate", "params": {}},
            ],
            priority=1,
        ),
    ],
    
    "clean-minor-issues": [
        FilterPipeline(
            name="basic-cleanup",
            description="Basic cleanup for minor issues",
            actions=[
                {"action": "trimesh_basic", "params": {}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=1,
        ),
    ],
    
    # =========================================================================
    # HOLES AND BOUNDARIES
    # =========================================================================
    "holes": [
        FilterPipeline(
            name="fill-small-holes",
            description="Fill small holes with trimesh",
            actions=[
                {"action": "trimesh_basic", "params": {}},
                {"action": "fill_holes", "params": {"max_hole_size": 100}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=1,
        ),
        FilterPipeline(
            name="fill-medium-holes",
            description="Fill medium holes",
            actions=[
                {"action": "trimesh_basic", "params": {}},
                {"action": "fill_holes", "params": {"max_hole_size": 500}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=2,
        ),
        FilterPipeline(
            name="fill-large-holes",
            description="Fill large holes",
            actions=[
                {"action": "trimesh_basic", "params": {}},
                {"action": "fill_holes", "params": {"max_hole_size": 1000}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=3,
        ),
        FilterPipeline(
            name="pymeshfix-holes",
            description="Use pymeshfix for stubborn holes",
            actions=[
                {"action": "trimesh_basic", "params": {}},
                {"action": "pymeshfix_repair", "params": {}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=4,
            avoid_for_fragmented=True,
        ),
        FilterPipeline(
            name="blender-remesh-holes",
            description="Blender remesh for very difficult holes",
            actions=[
                {"action": "blender_remesh", "params": {"voxel_size": "auto"}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=5,
            avoid_for_fragmented=True,
        ),
    ],
    
    # Combined profile: holes + non-manifold (common combination)
    # Based on model_profiles.md: mesh-with-holes-and-non-manifold
    "mesh-with-holes-and-non-manifold": [
        FilterPipeline(
            name="holes-and-nonmanifold-pymeshfix",
            description="Pymeshfix handles both holes and non-manifold",
            actions=[
                {"action": "trimesh_basic", "params": {}},
                {"action": "pymeshfix_repair", "params": {}},
                {"action": "fill_holes", "params": {"max_hole_size": 500}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=1,
            avoid_for_fragmented=True,
        ),
        FilterPipeline(
            name="holes-and-nonmanifold-conservative",
            description="Conservative repair for holes + non-manifold",
            actions=[
                {"action": "trimesh_basic", "params": {}},
                {"action": "pymeshfix_repair_conservative", "params": {}},
                {"action": "fill_holes", "params": {"max_hole_size": 500}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=2,
        ),
        FilterPipeline(
            name="holes-and-nonmanifold-blender",
            description="Blender remesh for severe holes + non-manifold",
            actions=[
                {"action": "blender_remesh", "params": {"voxel_size": "auto"}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=3,
            avoid_for_fragmented=True,
        ),
    ],
    
    "holes-only": [
        # Same as holes but simpler model
        FilterPipeline(
            name="simple-hole-fill",
            description="Simple hole filling for single-component model",
            actions=[
                {"action": "fill_holes", "params": {"max_hole_size": 500}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=1,
        ),
        FilterPipeline(
            name="pymeshfix-simple",
            description="Pymeshfix for clean single-component with holes",
            actions=[
                {"action": "pymeshfix_repair", "params": {}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=2,
        ),
    ],
    
    "many-small-holes": [
        FilterPipeline(
            name="fill-tiny-holes",
            description="Fill many tiny holes",
            actions=[
                {"action": "trimesh_basic", "params": {}},
                {"action": "fill_holes", "params": {"max_hole_size": 50}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=1,
        ),
        FilterPipeline(
            name="pymeshfix-many-holes",
            description="Pymeshfix handles many holes well",
            actions=[
                {"action": "trimesh_basic", "params": {}},
                {"action": "pymeshfix_repair", "params": {}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=2,
            avoid_for_fragmented=True,
        ),
    ],
    
    "open-bottom": [
        FilterPipeline(
            name="fill-planar-hole",
            description="Fill single large planar hole (open bottom)",
            actions=[
                {"action": "fill_holes", "params": {"max_hole_size": 10000}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=1,
        ),
    ],
    
    # =========================================================================
    # FRAGMENTED / MULTI-COMPONENT
    # These profiles must NOT use blender_remesh or aggressive pymeshfix
    # EXCEPT for extreme fragmentation where reconstruction is the only option
    # =========================================================================
    "fragmented": [
        FilterPipeline(
            name="conservative-per-component",
            description="Conservative repair preserving all components",
            actions=[
                {"action": "trimesh_basic", "params": {}},
                {"action": "pymeshfix_repair_conservative", "params": {}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=1,
        ),
        FilterPipeline(
            name="minimal-fragmented",
            description="Minimal repair for fragmented models",
            actions=[
                {"action": "trimesh_basic", "params": {}},
                {"action": "fill_holes", "params": {"max_hole_size": 50}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=2,
        ),
        FilterPipeline(
            name="normals-only-fragmented",
            description="Just fix normals, preserve structure",
            actions=[
                {"action": "fix_normals", "params": {}},
                {"action": "fix_winding", "params": {}},
            ],
            priority=3,
        ),
        # For extreme fragmentation (>1000 bodies), reconstruction is the only option
        FilterPipeline(
            name="voxel-reconstruct-fragmented",
            description="Voxelize and reconstruct for extreme fragmentation",
            actions=[
                {"action": "voxelize_and_reconstruct", "params": {"pitch": "auto"}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=4,
        ),
        FilterPipeline(
            name="meshlab-poisson-reconstruct",
            description="Poisson surface reconstruction from point cloud",
            actions=[
                {"action": "meshlab_reconstruct_poisson", "params": {"depth": 8}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=5,
        ),
        FilterPipeline(
            name="meshlab-ball-pivoting-reconstruct",
            description="Ball pivoting reconstruction",
            actions=[
                {"action": "meshlab_reconstruct_ball_pivoting", "params": {}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=6,
        ),
        FilterPipeline(
            name="blender-remesh-aggressive",
            description="Aggressive Blender remesh as last resort for extreme fragmentation",
            actions=[
                {"action": "blender_remesh", "params": {"voxel_size": 0.01}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=7,
            # NOTE: This CAN destroy fragmented models, but for extreme cases
            # (>1000 bodies) it may be the only option to get a printable mesh
        ),
    ],
    
    # NEW: Extreme fragmentation profile (>1000 bodies) - reconstruction only
    "extreme-fragmented": [
        FilterPipeline(
            name="voxel-reconstruct-extreme",
            description="Voxelize and reconstruct for 1000+ body meshes",
            actions=[
                {"action": "voxelize_and_reconstruct", "params": {"pitch": "auto"}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=1,
        ),
        FilterPipeline(
            name="voxel-reconstruct-fine",
            description="Finer voxel reconstruction",
            actions=[
                {"action": "voxelize_and_reconstruct", "params": {"pitch": 0.5}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=2,
        ),
        FilterPipeline(
            name="meshlab-poisson-extreme",
            description="High-detail Poisson reconstruction",
            actions=[
                {"action": "meshlab_reconstruct_poisson", "params": {"depth": 10}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=3,
        ),
        FilterPipeline(
            name="meshlab-alpha-wrap",
            description="Alpha wrap reconstruction",
            actions=[
                {"action": "meshlab_alpha_wrap", "params": {}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=4,
        ),
        FilterPipeline(
            name="blender-remesh-extreme",
            description="Blender voxel remesh for extreme fragmentation",
            actions=[
                {"action": "blender_remesh", "params": {"voxel_size": 0.005}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=5,
        ),
        FilterPipeline(
            name="convex-hull-extreme",
            description="Convex hull as absolute last resort",
            actions=[
                {"action": "convex_hull", "params": {}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=6,
        ),
    ],
    
    "multiple-disconnected-large": [
        FilterPipeline(
            name="conservative-multi-component",
            description="Conservative repair for multiple large components",
            actions=[
                {"action": "trimesh_basic", "params": {}},
                {"action": "pymeshfix_repair_conservative", "params": {}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=1,
        ),
    ],
    
    "floating-components": [
        FilterPipeline(
            name="remove-floating-small",
            description="Remove small floating components",
            actions=[
                {"action": "trimesh_basic", "params": {}},
                {"action": "remove_small_components", "params": {"min_faces": 50}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=1,
        ),
        FilterPipeline(
            name="conservative-floating",
            description="Conservative repair keeping floating parts",
            actions=[
                {"action": "trimesh_basic", "params": {}},
                {"action": "pymeshfix_repair_conservative", "params": {}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=2,
        ),
    ],
    
    "debris-particles": [
        FilterPipeline(
            name="remove-debris",
            description="Remove tiny debris particles",
            actions=[
                {"action": "trimesh_basic", "params": {}},
                {"action": "remove_small_components", "params": {"min_faces": 10}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=1,
        ),
    ],
    
    # =========================================================================
    # TOPOLOGY ERRORS
    # =========================================================================
    "non-manifold": [
        FilterPipeline(
            name="make-manifold-simple",
            description="Simple manifold repair",
            actions=[
                {"action": "trimesh_basic", "params": {}},
                {"action": "make_manifold", "params": {}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=1,
        ),
        FilterPipeline(
            name="pymeshfix-manifold",
            description="Pymeshfix for non-manifold repair",
            actions=[
                {"action": "trimesh_basic", "params": {}},
                {"action": "pymeshfix_repair", "params": {}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=2,
            avoid_for_fragmented=True,
        ),
        FilterPipeline(
            name="blender-manifold",
            description="Blender for severe non-manifold issues",
            actions=[
                {"action": "blender_remesh", "params": {"voxel_size": "auto"}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=3,
            avoid_for_fragmented=True,
        ),
    ],
    
    "non_manifold": [  # Alias for slicer-detected issue
        FilterPipeline(
            name="pymeshfix-non-manifold",
            description="Pymeshfix for non-manifold edges",
            actions=[
                {"action": "trimesh_basic", "params": {}},
                {"action": "pymeshfix_repair", "params": {}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=1,
            avoid_for_fragmented=True,
        ),
        FilterPipeline(
            name="make-manifold",
            description="Make mesh manifold",
            actions=[
                {"action": "trimesh_basic", "params": {}},
                {"action": "make_manifold", "params": {}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=2,
        ),
        FilterPipeline(
            name="blender-non-manifold",
            description="Blender remesh for stubborn non-manifold",
            actions=[
                {"action": "blender_remesh", "params": {"voxel_size": "auto"}},
            ],
            priority=3,
            avoid_for_fragmented=True,
        ),
    ],
    
    "degenerate-heavy": [
        FilterPipeline(
            name="remove-degenerate",
            description="Remove degenerate faces",
            actions=[
                {"action": "trimesh_basic", "params": {}},
                {"action": "remove_degenerate", "params": {}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=1,
        ),
        FilterPipeline(
            name="pymeshfix-degenerate",
            description="Pymeshfix after removing degenerate faces",
            actions=[
                {"action": "trimesh_basic", "params": {}},
                {"action": "remove_degenerate", "params": {}},
                {"action": "pymeshfix_repair", "params": {}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=2,
            avoid_for_fragmented=True,
        ),
    ],
    
    # =========================================================================
    # NORMAL ISSUES
    # =========================================================================
    "normals": [
        FilterPipeline(
            name="fix-normals-simple",
            description="Simple normal fix",
            actions=[
                {"action": "fix_normals", "params": {}},
            ],
            priority=1,
        ),
        FilterPipeline(
            name="fix-winding-and-normals",
            description="Fix winding order and normals",
            actions=[
                {"action": "fix_winding", "params": {}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=2,
        ),
        FilterPipeline(
            name="full-normal-repair",
            description="Full cleanup with normal repair",
            actions=[
                {"action": "trimesh_basic", "params": {}},
                {"action": "fix_winding", "params": {}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=3,
        ),
    ],
    
    "normals-inconsistent": [
        FilterPipeline(
            name="unify-normals",
            description="Unify inconsistent normals",
            actions=[
                {"action": "trimesh_basic", "params": {}},
                {"action": "fix_winding", "params": {}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=1,
        ),
    ],
    
    "inverted-normals": [
        FilterPipeline(
            name="flip-normals",
            description="Flip inverted normals",
            actions=[
                {"action": "flip_normals", "params": {}},
            ],
            priority=1,
        ),
    ],
    
    # =========================================================================
    # SELF-INTERSECTION
    # =========================================================================
    "self-intersecting": [
        FilterPipeline(
            name="pymeshfix-intersections",
            description="Pymeshfix for self-intersections",
            actions=[
                {"action": "trimesh_basic", "params": {}},
                {"action": "pymeshfix_clean", "params": {}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=1,
            avoid_for_fragmented=True,
        ),
        FilterPipeline(
            name="full-repair-intersections",
            description="Full repair for self-intersections",
            actions=[
                {"action": "trimesh_basic", "params": {}},
                {"action": "pymeshfix_repair", "params": {}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=2,
            avoid_for_fragmented=True,
        ),
        FilterPipeline(
            name="blender-intersections",
            description="Blender remesh for severe intersections",
            actions=[
                {"action": "blender_remesh", "params": {"voxel_size": "auto"}},
            ],
            priority=3,
            avoid_for_fragmented=True,
        ),
    ],
    
    "self_intersections": [  # Alias
        FilterPipeline(
            name="pymeshfix-clean-intersections",
            description="Clean self-intersections with pymeshfix",
            actions=[
                {"action": "pymeshfix_clean", "params": {}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=1,
            avoid_for_fragmented=True,
        ),
        FilterPipeline(
            name="full-repair-intersect",
            description="Full pymeshfix repair",
            actions=[
                {"action": "trimesh_basic", "params": {}},
                {"action": "pymeshfix_repair", "params": {}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=2,
            avoid_for_fragmented=True,
        ),
    ],
    
    # =========================================================================
    # SCAN / NOISY MESH
    # =========================================================================
    "noisy-scan": [
        FilterPipeline(
            name="decimate-and-repair",
            description="Decimate noisy scan then repair",
            actions=[
                {"action": "trimesh_basic", "params": {}},
                {"action": "decimate", "params": {"target_faces": 100000}},
                {"action": "pymeshfix_repair", "params": {}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=1,
            avoid_for_fragmented=True,
        ),
        FilterPipeline(
            name="smooth-and-repair",
            description="Smooth noisy mesh then repair",
            actions=[
                {"action": "trimesh_basic", "params": {}},
                {"action": "smooth", "params": {"iterations": 2}},
                {"action": "pymeshfix_repair", "params": {}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=2,
            avoid_for_fragmented=True,
        ),
    ],
    
    "high-triangle-density": [
        FilterPipeline(
            name="decimate-high-density",
            description="Decimate extremely dense mesh",
            actions=[
                {"action": "trimesh_basic", "params": {}},
                {"action": "decimate", "params": {"target_faces": 100000}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=1,
        ),
    ],
    
    # =========================================================================
    # COMPLEX TOPOLOGY
    # =========================================================================
    "complex-high-genus": [
        FilterPipeline(
            name="pymeshfix-complex",
            description="Pymeshfix for complex topology",
            actions=[
                {"action": "trimesh_basic", "params": {}},
                {"action": "pymeshfix_repair", "params": {}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=1,
            avoid_for_fragmented=True,
        ),
        FilterPipeline(
            name="blender-complex",
            description="Blender remesh for very complex topology",
            actions=[
                {"action": "blender_remesh", "params": {"voxel_size": "auto"}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=2,
            avoid_for_fragmented=True,
        ),
    ],
    
    # =========================================================================
    # GEOMETRY ISSUES (generic)
    # =========================================================================
    "geometry_issue": [
        FilterPipeline(
            name="place-on-bed-first",
            description="Place on bed then repair",
            actions=[
                {"action": "place_on_bed", "params": {}},
                {"action": "trimesh_basic", "params": {}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=1,
        ),
        FilterPipeline(
            name="full-geometry-repair",
            description="Full repair for geometry issues",
            actions=[
                {"action": "trimesh_basic", "params": {}},
                {"action": "pymeshfix_repair", "params": {}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=2,
            avoid_for_fragmented=True,
        ),
    ],
    
    "bed_adhesion": [
        FilterPipeline(
            name="place-on-bed",
            description="Place model on build plate",
            actions=[
                {"action": "place_on_bed", "params": {}},
            ],
            priority=1,
        ),
    ],
    
    # =========================================================================
    # DEGENERATE (slicer-detected)
    # =========================================================================
    "degenerate": [
        FilterPipeline(
            name="basic-degenerate-fix",
            description="Basic cleanup removes degenerate faces",
            actions=[
                {"action": "trimesh_basic", "params": {}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=1,
        ),
        FilterPipeline(
            name="remove-degenerate-explicit",
            description="Explicitly remove degenerate faces",
            actions=[
                {"action": "remove_degenerate", "params": {}},
                {"action": "trimesh_basic", "params": {}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=2,
        ),
    ],
    
    # =========================================================================
    # UNKNOWN / FALLBACK
    # =========================================================================
    "unknown": [
        FilterPipeline(
            name="basic-cleanup-unknown",
            description="Basic cleanup for unknown issues",
            actions=[
                {"action": "trimesh_basic", "params": {}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=1,
        ),
        FilterPipeline(
            name="fill-and-repair-unknown",
            description="Fill holes and repair",
            actions=[
                {"action": "trimesh_basic", "params": {}},
                {"action": "fill_holes", "params": {"max_hole_size": 500}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=2,
        ),
        FilterPipeline(
            name="pymeshfix-unknown",
            description="Full pymeshfix repair",
            actions=[
                {"action": "trimesh_basic", "params": {}},
                {"action": "pymeshfix_repair", "params": {}},
                {"action": "fix_normals", "params": {}},
            ],
            priority=3,
            avoid_for_fragmented=True,
        ),
        FilterPipeline(
            name="blender-unknown",
            description="Blender remesh as last resort",
            actions=[
                {"action": "blender_remesh", "params": {"voxel_size": "auto"}},
            ],
            priority=4,
            avoid_for_fragmented=True,
        ),
    ],
}


# =============================================================================
# GENERIC FALLBACK PIPELINES
# Used when profile-specific pipelines are exhausted
# =============================================================================

GENERIC_PIPELINES: List[FilterPipeline] = [
    FilterPipeline(
        name="generic-basic",
        description="Generic basic cleanup",
        actions=[
            {"action": "trimesh_basic", "params": {}},
            {"action": "fix_normals", "params": {}},
        ],
        priority=1,
    ),
    FilterPipeline(
        name="generic-fill-holes",
        description="Generic hole filling",
        actions=[
            {"action": "trimesh_basic", "params": {}},
            {"action": "fill_holes", "params": {"max_hole_size": 500}},
            {"action": "fix_normals", "params": {}},
        ],
        priority=2,
    ),
    FilterPipeline(
        name="generic-pymeshfix",
        description="Generic pymeshfix repair",
        actions=[
            {"action": "trimesh_basic", "params": {}},
            {"action": "pymeshfix_repair", "params": {}},
            {"action": "fix_normals", "params": {}},
        ],
        priority=3,
        avoid_for_fragmented=True,
    ),
    FilterPipeline(
        name="generic-conservative",
        description="Generic conservative repair (safe for multi-component)",
        actions=[
            {"action": "trimesh_basic", "params": {}},
            {"action": "pymeshfix_repair_conservative", "params": {}},
            {"action": "fix_normals", "params": {}},
        ],
        priority=4,
    ),
    FilterPipeline(
        name="generic-blender",
        description="Generic Blender remesh (last resort)",
        actions=[
            {"action": "blender_remesh", "params": {"voxel_size": "auto"}},
        ],
        priority=5,
        avoid_for_fragmented=True,
    ),
]


def get_pipelines_for_issues(
    issues: List[str],
    is_fragmented: bool = False,
    use_learning: bool = True,
) -> List[FilterPipeline]:
    """
    Get recommended pipelines for the given issues.
    
    This function:
    1. Detects combined profiles (e.g., holes + non-manifold)
    2. Collects pipelines from all matching issue categories
    3. Adds combined profile pipelines if applicable
    4. Deduplicates and sorts by priority
    5. Filters out dangerous pipelines for fragmented models
    6. **NEW**: Reorders based on learning engine recommendations
    
    Args:
        issues: List of issue categories (e.g., ["holes", "normals"])
        is_fragmented: If True, skip pipelines that destroy fragmented models
        use_learning: If True, use learning engine to optimize pipeline order
        
    Returns:
        List of FilterPipeline objects, ordered by priority/learning
    """
    pipelines = []
    seen_names = set()
    
    # Check for combined profiles first (higher priority)
    combined_profiles = detect_combined_profiles(issues)
    for profile in combined_profiles:
        profile_pipelines = PROFILE_PIPELINES.get(profile, [])
        for pipeline in profile_pipelines:
            if pipeline.name not in seen_names:
                if is_fragmented and pipeline.avoid_for_fragmented:
                    logger.debug(f"Skipping pipeline '{pipeline.name}' - not safe for fragmented")
                    continue
                pipelines.append(pipeline)
                seen_names.add(pipeline.name)
    
    # Collect pipelines from all matching issue categories
    for issue in issues:
        issue_pipelines = PROFILE_PIPELINES.get(issue, [])
        for pipeline in issue_pipelines:
            if pipeline.name not in seen_names:
                # Skip dangerous pipelines for fragmented models
                if is_fragmented and pipeline.avoid_for_fragmented:
                    logger.debug(f"Skipping pipeline '{pipeline.name}' - not safe for fragmented")
                    continue
                pipelines.append(pipeline)
                seen_names.add(pipeline.name)
    
    # Add generic fallbacks that weren't already included
    for pipeline in GENERIC_PIPELINES:
        if pipeline.name not in seen_names:
            if is_fragmented and pipeline.avoid_for_fragmented:
                continue
            pipelines.append(pipeline)
            seen_names.add(pipeline.name)
    
    # =========================================================================
    # NEW: Reorder pipelines based on learning engine recommendations
    # =========================================================================
    if use_learning and pipelines:
        try:
            from .learning_engine import get_learning_engine
            engine = get_learning_engine()
            stats = engine.get_stats_summary()
            
            # Only use learning if we have enough data (at least 50 models)
            if stats.get("total_models_processed", 0) >= 50:
                # Get recommended order from learning engine
                recommended_order = engine.get_recommended_pipeline_order(issues)
                
                if recommended_order:
                    # Create lookup for recommended positions
                    order_lookup = {name: idx for idx, name in enumerate(recommended_order)}
                    
                    # Reorder pipelines: learned order first, then by original priority
                    def sort_key(p):
                        if p.name in order_lookup:
                            # Learned pipelines get negative priority (come first)
                            return (-1000 + order_lookup[p.name], p.priority)
                        else:
                            # Unknown pipelines keep original priority
                            return (0, p.priority)
                    
                    pipelines.sort(key=sort_key)
                    logger.debug(f"Reordered pipelines using learning engine (top: {pipelines[0].name if pipelines else 'none'})")
        except Exception as e:
            # Learning engine not available or error - fall back to default order
            logger.debug(f"Learning engine unavailable for pipeline ordering: {e}")
            pipelines.sort(key=lambda p: p.priority)
    else:
        # Sort by priority (default)
        pipelines.sort(key=lambda p: p.priority)
    
    return pipelines


def detect_combined_profiles(issues: List[str]) -> List[str]:
    """
    Detect combined profiles based on issue combinations.
    
    Based on model_profiles.md, some issue combinations have specific
    profiles with optimized repair strategies.
    
    Args:
        issues: List of detected issues
        
    Returns:
        List of combined profile names to add
    """
    combined = []
    issue_set = set(issues)
    
    # mesh-with-holes-and-non-manifold: holes + non-manifold
    if ("holes" in issue_set or "open_edges" in issue_set) and \
       ("non_manifold" in issue_set or "non-manifold" in issue_set):
        combined.append("mesh-with-holes-and-non-manifold")
    
    # self-intersecting with holes is common
    if ("holes" in issue_set) and ("self_intersections" in issue_set or "self-intersecting" in issue_set):
        combined.append("self-intersecting")  # Use self-intersecting profile which handles both
    
    # degenerate + holes often occur together
    if ("degenerate" in issue_set) and ("holes" in issue_set):
        combined.append("degenerate-heavy")  # Handle degenerate first
    
    return combined


def analyze_fragmented_model(mesh) -> dict:
    """
    Analyze a fragmented model to determine the type of fragmentation.
    
    This helps decide whether blender_remesh would be destructive:
    - debris-particles: Many tiny fragments -> Blender DESTROYS
    - multi-part-assembly: Multiple large parts -> Blender DESTROYS  
    - split-seam: Parts that should connect -> Blender might HELP
    
    Args:
        mesh: trimesh.Trimesh object
        
    Returns:
        dict with analysis results
    """
    try:
        components = mesh.split(only_watertight=False)
        num_components = len(components)
        
        if num_components <= 1:
            return {
                "is_fragmented": False,
                "fragmentation_type": None,
                "blender_safe": True,
            }
        
        # Analyze component sizes
        component_faces = [len(c.faces) for c in components]
        total_faces = sum(component_faces)
        
        tiny_count = sum(1 for f in component_faces if f < 10)  # < 10 faces
        small_count = sum(1 for f in component_faces if 10 <= f < 50)
        large_count = sum(1 for f in component_faces if f >= 50)
        
        largest_pct = max(component_faces) / total_faces * 100 if total_faces > 0 else 0
        
        # Determine fragmentation type
        # Type 1: Debris/particles - many tiny fragments
        if tiny_count > num_components * 0.7:  # >70% are tiny
            return {
                "is_fragmented": True,
                "fragmentation_type": "debris-particles",
                "blender_safe": False,  # Would merge debris into main part
                "reason": f"{tiny_count}/{num_components} components are tiny (<10 faces)",
                "component_count": num_components,
                "tiny_count": tiny_count,
                "largest_pct": largest_pct,
            }
        
        # Type 2: Multi-part assembly - multiple large components
        if large_count >= 2 and largest_pct < 80:
            return {
                "is_fragmented": True,
                "fragmentation_type": "multi-part-assembly",
                "blender_safe": False,  # Would merge separate parts
                "reason": f"{large_count} large components, largest is {largest_pct:.1f}% of total",
                "component_count": num_components,
                "large_count": large_count,
                "largest_pct": largest_pct,
            }
        
        # Type 3: Sparse/wireframe - few faces spread across large bbox
        import numpy as np
        bbox_size = mesh.bounds[1] - mesh.bounds[0]
        bbox_volume = np.prod(bbox_size)
        faces_per_volume = total_faces / (bbox_volume + 0.001)
        
        if faces_per_volume < 0.001 and num_components > 20:
            return {
                "is_fragmented": True,
                "fragmentation_type": "sparse-wireframe",
                "blender_safe": False,  # Would fill in the gaps
                "reason": f"Very sparse: {faces_per_volume:.6f} faces/unit volume",
                "component_count": num_components,
                "faces_per_volume": faces_per_volume,
            }
        
        # Type 4: Split seam - components might need to be reconnected
        # This is the case where Blender MIGHT help
        if num_components <= 10 and largest_pct > 50:
            return {
                "is_fragmented": True,
                "fragmentation_type": "split-seam",
                "blender_safe": True,  # Might help reconnect parts
                "reason": f"Few components ({num_components}), might be split seam",
                "component_count": num_components,
                "largest_pct": largest_pct,
            }
        
        # Default: assume fragmented but unknown type
        return {
            "is_fragmented": True,
            "fragmentation_type": "unknown",
            "blender_safe": num_components < 10,  # Conservative: only safe if few components
            "reason": f"{num_components} components, unclear structure",
            "component_count": num_components,
        }
        
    except Exception as e:
        logger.debug(f"Could not analyze fragmented model: {e}")
        return {
            "is_fragmented": False,
            "fragmentation_type": None,
            "blender_safe": True,
            "error": str(e),
        }


def get_pipeline_count() -> dict:
    """Get statistics about available pipelines."""
    total_profile_pipelines = sum(len(p) for p in PROFILE_PIPELINES.values())
    unique_profiles = len(PROFILE_PIPELINES)
    
    return {
        "unique_profiles": unique_profiles,
        "total_profile_pipelines": total_profile_pipelines,
        "generic_pipelines": len(GENERIC_PIPELINES),
        "total_pipelines": total_profile_pipelines + len(GENERIC_PIPELINES),
    }


def print_pipeline_summary():
    """Print a summary of all available pipelines."""
    print("=" * 70)
    print("FILTER PIPELINE SUMMARY")
    print("=" * 70)
    
    stats = get_pipeline_count()
    print(f"\nProfiles with pipelines: {stats['unique_profiles']}")
    print(f"Profile-specific pipelines: {stats['total_profile_pipelines']}")
    print(f"Generic fallback pipelines: {stats['generic_pipelines']}")
    print(f"Total unique pipelines: {stats['total_pipelines']}")
    
    print("\n" + "-" * 70)
    print("PROFILE-SPECIFIC PIPELINES")
    print("-" * 70)
    
    for profile, pipelines in sorted(PROFILE_PIPELINES.items()):
        print(f"\n{profile}: ({len(pipelines)} pipelines)")
        for p in pipelines:
            actions = " -> ".join(a["action"] for a in p.actions)
            fragmented_note = " [!fragmented]" if p.avoid_for_fragmented else ""
            print(f"  {p.priority}. {p.name}{fragmented_note}")
            print(f"     {actions}")
    
    print("\n" + "-" * 70)
    print("GENERIC FALLBACK PIPELINES")
    print("-" * 70)
    
    for p in GENERIC_PIPELINES:
        actions = " -> ".join(a["action"] for a in p.actions)
        fragmented_note = " [!fragmented]" if p.avoid_for_fragmented else ""
        print(f"  {p.priority}. {p.name}{fragmented_note}")
        print(f"     {actions}")


if __name__ == "__main__":
    print_pipeline_summary()
