# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Iterative slicer-driven repair loop.

When slicer validation fails, this module attempts automatic repair using
available filter actions, iterating until the model passes or no repair
options remain.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
import time

import trimesh

from .actions import ActionRegistry
from .actions.slicer_actions import (
    validate_mesh,
    validate_stl_file,
    get_mesh_info_prusa,
    SlicerResult,
    MeshInfo,
    is_slicer_available,
    SLICER_ERROR_PATTERNS,
)
from .filter_script import FilterScript, FilterScriptRunner, create_filter_script
from .validation import validate_geometry

logger = logging.getLogger(__name__)


@dataclass
class RepairStrategy:
    """A repair strategy that maps slicer issues to actions."""
    issue_type: str
    action: str
    params: dict = field(default_factory=dict)
    priority: int = 1
    description: str = ""


# Mapping of slicer issues to repair strategies
# Based on functional spec: docs/functional_spec.md
SLICER_ISSUE_MAPPINGS: Dict[str, List[RepairStrategy]] = {
    "non_manifold": [
        RepairStrategy("non_manifold", "pymeshfix_repair", {}, 1, "Full repair with PyMeshFix"),
        RepairStrategy("non_manifold", "make_manifold", {}, 2, "Make mesh manifold"),
        RepairStrategy("non_manifold", "blender_remesh", {"voxel_size": 0.05}, 3, "Blender voxel remesh"),
    ],
    "holes": [
        RepairStrategy("holes", "fill_holes", {"max_hole_size": 100}, 1, "Fill small holes"),
        RepairStrategy("holes", "fill_holes", {"max_hole_size": 1000}, 2, "Fill larger holes"),
        RepairStrategy("holes", "pymeshfix_repair", {}, 3, "Full repair with PyMeshFix"),
        RepairStrategy("holes", "blender_remesh", {"voxel_size": 0.05}, 4, "Blender voxel remesh"),
    ],
    "self_intersections": [
        RepairStrategy("self_intersections", "pymeshfix_clean", {}, 1, "Clean self-intersections"),
        RepairStrategy("self_intersections", "pymeshfix_repair", {}, 2, "Full repair"),
        RepairStrategy("self_intersections", "blender_remesh", {"voxel_size": 0.05}, 3, "Blender remesh"),
    ],
    "degenerate": [
        RepairStrategy("degenerate", "trimesh_basic", {}, 1, "Basic trimesh cleanup"),
        RepairStrategy("degenerate", "remove_degenerate", {}, 2, "Remove degenerate faces"),
    ],
    "normals": [
        RepairStrategy("normals", "fix_normals", {}, 1, "Fix face normals"),
        RepairStrategy("normals", "fix_winding", {}, 2, "Fix face winding"),
    ],
    # Bed adhesion issues - model needs to be placed on the build plate
    "bed_adhesion": [
        RepairStrategy("bed_adhesion", "place_on_bed", {}, 1, "Move mesh to build plate"),
        RepairStrategy("bed_adhesion", "convex_hull", {}, 2, "Replace with convex hull"),
        RepairStrategy("bed_adhesion", "blender_remesh", {"voxel_size": 0.05}, 3, "Blender remesh"),
    ],
    # Geometry issues - general mesh problems
    "geometry_issue": [
        RepairStrategy("geometry_issue", "place_on_bed", {}, 1, "Move mesh to build plate"),
        RepairStrategy("geometry_issue", "pymeshfix_repair", {}, 2, "Full repair with PyMeshFix"),
        RepairStrategy("geometry_issue", "blender_remesh", {"voxel_size": 0.05}, 3, "Blender remesh"),
        RepairStrategy("geometry_issue", "blender_remesh", {"voxel_size": 0.02}, 4, "Finer Blender remesh"),
    ],
    # Generic fallback strategies
    "unknown": [
        RepairStrategy("unknown", "trimesh_basic", {}, 1, "Basic cleanup"),
        RepairStrategy("unknown", "pymeshfix_repair", {}, 2, "Full repair"),
        RepairStrategy("unknown", "blender_remesh", {"voxel_size": 0.05}, 3, "Blender remesh"),
    ],
}


@dataclass
class RepairAttempt:
    """Record of a single repair attempt."""
    attempt_number: int
    strategy: RepairStrategy
    success: bool
    duration_ms: float
    error: Optional[str] = None
    slicer_result: Optional[SlicerResult] = None
    geometry_valid: bool = False


@dataclass
class SlicerRepairResult:
    """Result of the iterative slicer repair loop."""
    success: bool
    final_mesh: Optional[trimesh.Trimesh]
    total_attempts: int
    total_duration_ms: float
    attempts: List[RepairAttempt] = field(default_factory=list)
    issues_found: List[str] = field(default_factory=list)
    issues_resolved: List[str] = field(default_factory=list)
    final_slicer_result: Optional[SlicerResult] = None
    error: Optional[str] = None
    
    # Pre-check results (STRICT validation before any repair)
    precheck_passed: bool = False  # True if model was already clean
    precheck_skipped: bool = False  # True if we skipped repair due to precheck
    precheck_mesh_info: Optional[MeshInfo] = None  # Mesh info from pre-check
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export."""
        return {
            "success": self.success,
            "total_attempts": self.total_attempts,
            "total_duration_ms": self.total_duration_ms,
            "precheck_passed": self.precheck_passed,
            "precheck_skipped": self.precheck_skipped,
            "precheck_mesh_info": {
                "manifold": self.precheck_mesh_info.manifold,
                "open_edges": self.precheck_mesh_info.open_edges,
                "is_clean": self.precheck_mesh_info.is_clean,
                "issues": self.precheck_mesh_info.issues,
            } if self.precheck_mesh_info else None,
            "issues_found": self.issues_found,
            "issues_resolved": self.issues_resolved,
            "attempts": [
                {
                    "attempt": a.attempt_number,
                    "action": a.strategy.action,
                    "params": a.strategy.params,
                    "success": a.success,
                    "duration_ms": a.duration_ms,
                    "error": a.error,
                }
                for a in self.attempts
            ],
            "error": self.error,
        }


def categorize_slicer_issues(slicer_result: SlicerResult) -> List[str]:
    """
    Analyze slicer output to categorize issues.
    
    Args:
        slicer_result: Result from slicer validation
        
    Returns:
        List of issue categories found
    """
    categories = set()
    
    # Check parsed issues
    for issue in slicer_result.issues:
        categories.add(issue.get("type", "unknown"))
    
    # Analyze error/warning text for additional patterns
    all_text = " ".join(slicer_result.errors + slicer_result.warnings).lower()
    all_text += " " + slicer_result.stdout.lower() + " " + slicer_result.stderr.lower()
    
    for category, patterns in SLICER_ERROR_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in all_text:
                categories.add(category)
                break
    
    # Check for specific common issues
    if "no extrusion" in all_text or "no material" in all_text or "empty" in all_text:
        categories.add("geometry_issue")
    
    if "first layer" in all_text:
        categories.add("bed_adhesion")
    
    return list(categories) if categories else ["unknown"]


def get_repair_strategies(issue_categories: List[str]) -> List[RepairStrategy]:
    """
    Get repair strategies for the given issue categories.
    
    Args:
        issue_categories: List of issue category strings
        
    Returns:
        Prioritized list of repair strategies to try
    """
    strategies = []
    seen_actions = set()
    
    for category in issue_categories:
        category_strategies = SLICER_ISSUE_MAPPINGS.get(category, [])
        for strategy in category_strategies:
            # Avoid duplicate actions
            action_key = (strategy.action, str(strategy.params))
            if action_key not in seen_actions:
                strategies.append(strategy)
                seen_actions.add(action_key)
    
    # Add fallback strategies if nothing found
    if not strategies:
        strategies = SLICER_ISSUE_MAPPINGS.get("unknown", [])
    
    # Sort by priority
    strategies.sort(key=lambda s: s.priority)
    
    return strategies


def run_slicer_repair_loop(
    mesh: trimesh.Trimesh,
    slicer: str = "auto",
    max_attempts: int = 10,
    escalate_to_blender_after: int = 5,
    timeout: int = 120,
    skip_if_clean: bool = True,
    progress_callback: Optional[callable] = None,
) -> SlicerRepairResult:
    """
    Run the iterative slicer-driven repair loop.
    
    This function:
    1. Pre-checks the mesh with STRICT slicer validation (no auto-repair)
    2. If already clean, returns immediately without repair
    3. If it fails, parses the errors and maps to repair actions
    4. Tries repair actions in priority order
    5. Repeats until success or max attempts reached
    
    Args:
        mesh: Input mesh to repair
        slicer: Slicer to use ('prusa', 'orca', 'auto')
        max_attempts: Maximum repair attempts
        escalate_to_blender_after: Number of attempts before escalating to Blender
        timeout: Slicer timeout in seconds
        skip_if_clean: If True, skip repair if pre-check passes (default: True)
        progress_callback: Optional callback(attempt_num, action_name, total_attempts)
        
    Returns:
        SlicerRepairResult with repair details
    """
    start_time = time.perf_counter()
    result = SlicerRepairResult(
        success=False,
        final_mesh=None,
        total_attempts=0,
        total_duration_ms=0,
    )
    
    if not is_slicer_available(slicer):
        result.error = f"No slicer available (tried: {slicer})"
        return result
    
    # =========================================================================
    # STEP 0: PRE-CHECK - Run STRICT slicer validation BEFORE any repair
    # This detects models that are already clean and don't need repair
    # =========================================================================
    if skip_if_clean:
        logger.info("Running STRICT pre-check (no auto-repair)...")
        precheck_result = validate_mesh(mesh, slicer=slicer, timeout=timeout, strict=True)
        result.precheck_mesh_info = precheck_result.mesh_info
        
        if precheck_result.success and precheck_result.mesh_info and precheck_result.mesh_info.is_clean:
            # Model is already clean! Skip repair entirely.
            logger.info("  PRE-CHECK PASSED: Model already clean (manifold, no holes, no reversed facets)")
            result.success = True
            result.precheck_passed = True
            result.precheck_skipped = True
            result.final_mesh = mesh.copy()
            result.final_slicer_result = precheck_result
            result.total_duration_ms = (time.perf_counter() - start_time) * 1000
            return result
        else:
            # Model has issues, will proceed with repair
            issues = precheck_result.mesh_info.issues if precheck_result.mesh_info else ["unknown"]
            logger.info(f"  PRE-CHECK: Issues detected: {issues}")
            result.precheck_passed = False
            result.issues_found.extend(issues)
    
    # =========================================================================
    # STEP 1+: Iterative repair loop
    # =========================================================================
    current_mesh = mesh.copy()
    attempted_actions = set()
    
    logger.info(f"Starting slicer repair loop (max_attempts={max_attempts})")
    
    for attempt_num in range(1, max_attempts + 1):
        result.total_attempts = attempt_num
        
        # Step 1: Validate with slicer (STRICT mode - no auto-repair)
        logger.info(f"  Attempt {attempt_num}: Running slicer validation...")
        slicer_result = validate_mesh(current_mesh, slicer=slicer, timeout=timeout, strict=True)
        
        if slicer_result.success:
            # Success! We're done
            logger.info(f"  Slicer validation PASSED on attempt {attempt_num}")
            result.success = True
            result.final_mesh = current_mesh
            result.final_slicer_result = slicer_result
            break
        
        # Step 2: Parse slicer errors and categorize issues
        issues = categorize_slicer_issues(slicer_result)
        logger.info(f"  Slicer validation FAILED. Issues: {issues}")
        result.issues_found.extend([i for i in issues if i not in result.issues_found])
        
        # Step 3: Get repair strategies for these issues
        strategies = get_repair_strategies(issues)
        
        # Filter out already-tried strategies
        strategies = [
            s for s in strategies 
            if (s.action, str(s.params)) not in attempted_actions
        ]
        
        if not strategies:
            logger.warning(f"  No more repair strategies available")
            result.error = "Exhausted all repair strategies"
            result.final_mesh = current_mesh
            result.final_slicer_result = slicer_result
            break
        
        # Should we escalate to Blender?
        if attempt_num >= escalate_to_blender_after:
            # Move Blender strategies to front
            blender_strategies = [s for s in strategies if "blender" in s.action]
            other_strategies = [s for s in strategies if "blender" not in s.action]
            strategies = blender_strategies + other_strategies
        
        # Step 4: Try the next repair strategy
        strategy = strategies[0]
        logger.info(f"  Trying repair: {strategy.action} {strategy.params}")
        
        # Call progress callback if provided
        if progress_callback:
            progress_callback(attempt_num, strategy.action, max_attempts)
        
        attempted_actions.add((strategy.action, str(strategy.params)))
        
        attempt_start = time.perf_counter()
        attempt_result = RepairAttempt(
            attempt_number=attempt_num,
            strategy=strategy,
            success=False,
            duration_ms=0,
        )
        
        try:
            # Execute the repair action
            repaired_mesh = ActionRegistry.execute(
                strategy.action, current_mesh, strategy.params
            )
            
            # Step 5: Validate geometry before slicer re-check
            geom_valid = validate_geometry(repaired_mesh)
            attempt_result.geometry_valid = geom_valid.is_printable
            
            if geom_valid.is_printable:
                logger.info(f"    Repair applied, geometry valid")
                current_mesh = repaired_mesh
                attempt_result.success = True
                result.issues_resolved.append(strategy.issue_type)
            else:
                logger.warning(f"    Repair broke geometry: {geom_valid.issues}")
                attempt_result.error = f"Geometry invalid: {geom_valid.issues}"
            
        except Exception as e:
            logger.error(f"    Repair action failed: {e}")
            attempt_result.error = str(e)
        
        attempt_result.duration_ms = (time.perf_counter() - attempt_start) * 1000
        result.attempts.append(attempt_result)
    
    result.total_duration_ms = (time.perf_counter() - start_time) * 1000
    
    if not result.success:
        result.final_mesh = current_mesh
        if not result.error:
            result.error = f"Failed after {result.total_attempts} attempts"
    
    return result
