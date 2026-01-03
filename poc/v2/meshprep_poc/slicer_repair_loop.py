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
from .filter_pipelines import (
    FilterPipeline,
    get_pipelines_for_issues,
    analyze_fragmented_model,
    PROFILE_PIPELINES,
    GENERIC_PIPELINES,
)
from .validation import validate_geometry

logger = logging.getLogger(__name__)


# Geometry loss thresholds
MAX_FACE_LOSS_PERCENT = 50  # Reject if more than 50% faces lost
MAX_VOLUME_LOSS_PERCENT = 40  # Reject if more than 40% volume lost
MIN_FACES_ABSOLUTE = 20  # Reject if result has fewer than 20 faces
MAX_FACE_INCREASE_FACTOR = 100  # Reject if faces increase by more than 100x


def check_geometry_loss(original_mesh: trimesh.Trimesh, repaired_mesh: trimesh.Trimesh) -> tuple[bool, str]:
    """
    Check if repair caused unacceptable geometry loss or excessive increase.
    
    Args:
        original_mesh: Original mesh before repair
        repaired_mesh: Mesh after repair
        
    Returns:
        Tuple of (is_acceptable, reason_if_not)
    """
    original_faces = len(original_mesh.faces)
    result_faces = len(repaired_mesh.faces)
    
    # Check absolute minimum
    if result_faces < MIN_FACES_ABSOLUTE:
        return False, f"Result has only {result_faces} faces (minimum: {MIN_FACES_ABSOLUTE})"
    
    # Check face loss percentage
    if original_faces > 0:
        face_loss_pct = (original_faces - result_faces) / original_faces * 100
        if face_loss_pct > MAX_FACE_LOSS_PERCENT:
            return False, f"Face loss {face_loss_pct:.1f}% exceeds threshold {MAX_FACE_LOSS_PERCENT}%"
        
        # Check excessive face increase (e.g., Blender remesh creating millions of faces)
        face_increase_factor = result_faces / original_faces
        if face_increase_factor > MAX_FACE_INCREASE_FACTOR:
            return False, f"Face count increased {face_increase_factor:.0f}x (max: {MAX_FACE_INCREASE_FACTOR}x)"
    
    # Check volume loss (if both have valid volumes)
    try:
        original_volume = original_mesh.volume if original_mesh.is_volume else 0
        result_volume = repaired_mesh.volume if repaired_mesh.is_volume else 0
        
        if original_volume > 0 and result_volume > 0:
            volume_loss_pct = abs(original_volume - result_volume) / original_volume * 100
            if volume_loss_pct > MAX_VOLUME_LOSS_PERCENT:
                return False, f"Volume change {volume_loss_pct:.1f}% exceeds threshold {MAX_VOLUME_LOSS_PERCENT}%"
    except Exception:
        pass  # Volume check is optional
    
    return True, ""


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
        RepairStrategy("non_manifold", "blender_remesh", {"voxel_size": "auto"}, 3, "Blender voxel remesh (auto)"),
    ],
    "holes": [
        RepairStrategy("holes", "fill_holes", {"max_hole_size": 100}, 1, "Fill small holes"),
        RepairStrategy("holes", "fill_holes", {"max_hole_size": 1000}, 2, "Fill larger holes"),
        RepairStrategy("holes", "pymeshfix_repair", {}, 3, "Full repair with PyMeshFix"),
        RepairStrategy("holes", "blender_remesh", {"voxel_size": "auto"}, 4, "Blender voxel remesh (auto)"),
    ],
    "self_intersections": [
        RepairStrategy("self_intersections", "pymeshfix_clean", {}, 1, "Clean self-intersections"),
        RepairStrategy("self_intersections", "pymeshfix_repair", {}, 2, "Full repair"),
        RepairStrategy("self_intersections", "blender_remesh", {"voxel_size": "auto"}, 3, "Blender remesh (auto)"),
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
        RepairStrategy("bed_adhesion", "blender_remesh", {"voxel_size": "auto"}, 3, "Blender remesh (auto)"),
    ],
    # Geometry issues - general mesh problems
    "geometry_issue": [
        RepairStrategy("geometry_issue", "place_on_bed", {}, 1, "Move mesh to build plate"),
        RepairStrategy("geometry_issue", "pymeshfix_repair", {}, 2, "Full repair with PyMeshFix"),
        RepairStrategy("geometry_issue", "blender_remesh", {"voxel_size": "auto"}, 3, "Blender remesh (auto)"),
    ],
    # Fragmented models - many components, sparse geometry
    # These need special handling - don't use blender_remesh which destroys structure
    "fragmented": [
        RepairStrategy("fragmented", "pymeshfix_repair_conservative", {}, 1, "Conservative per-component repair"),
        RepairStrategy("fragmented", "fill_holes", {"max_hole_size": 50}, 2, "Fill tiny holes only"),
        RepairStrategy("fragmented", "fix_normals", {}, 3, "Fix normals"),
        # NOTE: blender_remesh intentionally NOT included - it destroys fragmented models
    ],
    # Generic fallback strategies
    "unknown": [
        RepairStrategy("unknown", "trimesh_basic", {}, 1, "Basic cleanup"),
        RepairStrategy("unknown", "pymeshfix_repair", {}, 2, "Full repair"),
        RepairStrategy("unknown", "blender_remesh", {"voxel_size": "auto"}, 3, "Blender remesh (auto)"),
    ],
}


@dataclass
class RepairAttempt:
    """Record of a single repair attempt (can be a pipeline with multiple actions)."""
    attempt_number: int
    pipeline_name: str
    pipeline_actions: List[str]  # List of action names in the pipeline
    success: bool
    duration_ms: float
    error: Optional[str] = None
    slicer_result: Optional[SlicerResult] = None
    geometry_valid: bool = False
    
    # Legacy compatibility
    @property
    def strategy(self):
        """Legacy compatibility - return a strategy-like object."""
        return type('Strategy', (), {
            'action': self.pipeline_name,
            'params': {},
            'issue_type': 'pipeline',
        })()


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
                    "pipeline": a.pipeline_name,
                    "actions": a.pipeline_actions,
                    "success": a.success,
                    "duration_ms": a.duration_ms,
                    "error": a.error,
                }
                for a in self.attempts
            ],
            "error": self.error,
        }


def categorize_slicer_issues(slicer_result: SlicerResult, mesh: trimesh.Trimesh = None) -> List[str]:
    """
    Analyze slicer output to categorize issues.
    
    Args:
        slicer_result: Result from slicer validation
        mesh: Optional mesh to analyze for additional issues (e.g., fragmented)
        
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
    
    # Check for fragmented models if mesh is provided
    if mesh is not None:
        try:
            components = mesh.split(only_watertight=False)
            num_components = len(components)
            num_faces = len(mesh.faces)
            
            # Fragmented: many components with sparse geometry
            # This is important because blender_remesh destroys fragmented models!
            if num_components > 20 and num_faces < 1000:
                categories.add("fragmented")
                logger.info(f"  Detected FRAGMENTED model: {num_components} components, {num_faces} faces")
            elif num_components > 50:
                categories.add("fragmented")
                logger.info(f"  Detected FRAGMENTED model: {num_components} components")
        except Exception:
            pass
    
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
    max_attempts: int = 20,
    timeout: int = 120,
    skip_if_clean: bool = True,
    progress_callback: Optional[callable] = None,
) -> SlicerRepairResult:
    """
    Run the iterative slicer-driven repair loop using smart filter pipelines.
    
    This function:
    1. Pre-checks the mesh with STRICT slicer validation (no auto-repair)
    2. If already clean, returns immediately without repair
    3. Detects model profile (fragmented, holes, non-manifold, etc.)
    4. Gets profile-specific pipelines (multi-action sequences)
    5. Tries each pipeline FROM THE ORIGINAL MESH until one succeeds
    
    IMPORTANT: Each pipeline is applied to the ORIGINAL mesh, not to
    the result of a previous failed pipeline. This prevents "stacking" damage.
    
    Args:
        mesh: Input mesh to repair
        slicer: Slicer to use ('prusa', 'orca', 'auto')
        max_attempts: Maximum pipeline attempts (default 20 for more combinations)
        timeout: Slicer timeout in seconds
        skip_if_clean: If True, skip repair if pre-check passes (default: True)
        progress_callback: Optional callback(attempt_num, pipeline_name, total_attempts)
        
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
    
    # Keep the original mesh for fresh starts on each attempt
    original_mesh = mesh.copy()
    
    # =========================================================================
    # STEP 0: PRE-CHECK - Run STRICT slicer validation BEFORE any repair
    # =========================================================================
    if skip_if_clean:
        logger.info("Running STRICT pre-check (no auto-repair)...")
        precheck_result = validate_mesh(mesh, slicer=slicer, timeout=timeout, strict=True)
        result.precheck_mesh_info = precheck_result.mesh_info
        
        if precheck_result.success and precheck_result.mesh_info and precheck_result.mesh_info.is_clean:
            logger.info("  PRE-CHECK PASSED: Model already clean")
            result.success = True
            result.precheck_passed = True
            result.precheck_skipped = True
            result.final_mesh = mesh.copy()
            result.final_slicer_result = precheck_result
            result.total_duration_ms = (time.perf_counter() - start_time) * 1000
            return result
        else:
            issues = precheck_result.mesh_info.issues if precheck_result.mesh_info else ["unknown"]
            logger.info(f"  PRE-CHECK: Issues detected: {issues}")
            result.precheck_passed = False
            result.issues_found.extend(issues)
    
    # =========================================================================
    # STEP 1: Analyze fragmented models (critical for pipeline selection)
    # Use smart analysis to determine if blender_remesh would be destructive
    # =========================================================================
    frag_analysis = analyze_fragmented_model(original_mesh)
    blender_unsafe = False  # Will be True if blender_remesh should be skipped
    
    if frag_analysis["is_fragmented"]:
        if "fragmented" not in result.issues_found:
            result.issues_found.append("fragmented")
        
        frag_type = frag_analysis.get("fragmentation_type", "unknown")
        blender_safe = frag_analysis.get("blender_safe", False)
        reason = frag_analysis.get("reason", "")
        
        logger.warning(f"  FRAGMENTED MODEL DETECTED")
        logger.warning(f"    Type: {frag_type}")
        logger.warning(f"    Reason: {reason}")
        logger.warning(f"    Components: {frag_analysis.get('component_count', '?')}")
        
        if blender_safe:
            logger.info(f"    Blender remesh: ALLOWED (may help reconnect parts)")
            blender_unsafe = False
        else:
            logger.warning(f"    Blender remesh: BLOCKED (would destroy structure)")
            blender_unsafe = True
    
    # =========================================================================
    # STEP 2: Get smart pipelines for the detected issues
    # =========================================================================
    issues = result.issues_found if result.issues_found else ["unknown"]
    pipelines = get_pipelines_for_issues(issues, is_fragmented=blender_unsafe)
    
    logger.info(f"Starting pipeline-based repair loop")
    logger.info(f"  Issues: {issues}")
    logger.info(f"  Available pipelines: {len(pipelines)}")
    logger.info(f"  Max attempts: {max_attempts}")
    logger.info(f"  Blender remesh allowed: {not blender_unsafe}")
    logger.info(f"  NOTE: Each pipeline starts fresh from the original mesh")
    
    # =========================================================================
    # STEP 3: Try each pipeline until one succeeds
    # =========================================================================
    best_mesh = None
    best_score = -1
    attempted_pipelines = set()
    
    for attempt_num in range(1, min(max_attempts, len(pipelines)) + 1):
        result.total_attempts = attempt_num
        
        # Get next untried pipeline
        pipeline = None
        for p in pipelines:
            if p.name not in attempted_pipelines:
                pipeline = p
                break
        
        if pipeline is None:
            logger.warning(f"  No more pipelines available")
            result.error = "Exhausted all repair pipelines"
            break
        
        attempted_pipelines.add(pipeline.name)
        
        action_names = [a["action"] for a in pipeline.actions]
        logger.info(f"  Attempt {attempt_num}: {pipeline.name}")
        logger.info(f"    Actions: {' → '.join(action_names)}")
        logger.info(f"    Starting from ORIGINAL mesh")
        
        # Progress callback
        if progress_callback:
            progress_callback(attempt_num, pipeline.name, min(max_attempts, len(pipelines)))
        
        attempt_start = time.perf_counter()
        attempt_result = RepairAttempt(
            attempt_number=attempt_num,
            pipeline_name=pipeline.name,
            pipeline_actions=action_names,
            success=False,
            duration_ms=0,
        )
        
        try:
            # ALWAYS start from the ORIGINAL mesh!
            working_mesh = original_mesh.copy()
            
            # Execute each action in the pipeline sequentially
            for action_def in pipeline.actions:
                action_name = action_def["action"]
                action_params = action_def.get("params", {})
                
                logger.debug(f"      Executing: {action_name} {action_params}")
                working_mesh = ActionRegistry.execute(action_name, working_mesh, action_params)
            
            repaired_mesh = working_mesh
            
            # Validate geometry
            geom_valid = validate_geometry(repaired_mesh)
            attempt_result.geometry_valid = geom_valid.is_printable
            
            # Check for geometry loss
            geom_acceptable, geom_loss_reason = check_geometry_loss(original_mesh, repaired_mesh)
            
            if geom_valid.is_printable and geom_acceptable:
                logger.info(f"    Pipeline completed, geometry valid")
                
                # Run slicer validation
                logger.info(f"    Running slicer validation...")
                slicer_result = validate_mesh(repaired_mesh, slicer=slicer, timeout=timeout, strict=True)
                attempt_result.slicer_result = slicer_result
                
                if slicer_result.success:
                    logger.info(f"    ✓ Slicer validation PASSED!")
                    result.success = True
                    result.final_mesh = repaired_mesh
                    result.final_slicer_result = slicer_result
                    result.issues_resolved.append(pipeline.name)
                    attempt_result.success = True
                    attempt_result.duration_ms = (time.perf_counter() - attempt_start) * 1000
                    result.attempts.append(attempt_result)
                    break  # Success!
                else:
                    logger.info(f"    ✗ Slicer validation FAILED")
                    
                    # Track best result
                    score = (1 if repaired_mesh.is_watertight else 0) + (1 if repaired_mesh.is_volume else 0)
                    if score > best_score:
                        best_score = score
                        best_mesh = repaired_mesh
                        logger.info(f"    New best mesh (score: {score})")
                    
                    attempt_result.error = f"Slicer issues remain"
            elif geom_valid.is_printable and not geom_acceptable:
                logger.warning(f"    ✗ Geometry loss: {geom_loss_reason}")
                attempt_result.error = f"Geometry loss: {geom_loss_reason}"
                attempt_result.geometry_valid = False
            else:
                logger.warning(f"    ✗ Geometry invalid: {geom_valid.issues}")
                attempt_result.error = f"Geometry invalid: {geom_valid.issues}"
            
        except Exception as e:
            logger.error(f"    ✗ Pipeline failed: {e}")
            attempt_result.error = str(e)
        
        attempt_result.duration_ms = (time.perf_counter() - attempt_start) * 1000
        result.attempts.append(attempt_result)
    
    result.total_duration_ms = (time.perf_counter() - start_time) * 1000
    
    if not result.success:
        result.final_mesh = best_mesh if best_mesh is not None else original_mesh
        if not result.error:
            result.error = f"Failed after {result.total_attempts} pipeline attempts"
    
    return result
