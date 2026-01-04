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

# Import pipeline evolution (optional - won't fail if not available)
try:
    from .pipeline_evolution import get_evolution_engine, EvolvedPipeline
    EVOLUTION_AVAILABLE = True
except ImportError:
    EVOLUTION_AVAILABLE = False
    get_evolution_engine = None
    EvolvedPipeline = None

# Import detailed learning logger (optional)
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "v3"))
    from detailed_learning_logger import get_detailed_logger
    DETAILED_LOGGING_AVAILABLE = True
except ImportError:
    DETAILED_LOGGING_AVAILABLE = False
    get_detailed_logger = None

logger = logging.getLogger(__name__)


# Geometry loss thresholds (defaults - can be overridden by adaptive thresholds)
MAX_FACE_LOSS_PERCENT = 50  # Reject if more than 50% faces lost
MAX_VOLUME_LOSS_PERCENT = 40  # Reject if more than 40% volume lost
MIN_FACES_ABSOLUTE = 20  # Reject if result has fewer than 20 faces
MAX_FACE_INCREASE_FACTOR = 100  # Reject if faces increase by more than 100x

# Reconstruction mode thresholds (for extreme-fragmented meshes)
# These are much more lenient because reconstruction creates new geometry
RECONSTRUCTION_FACE_LOSS_PERCENT = 95  # Allow up to 95% face loss for reconstruction
RECONSTRUCTION_VOLUME_LOSS_PERCENT = 80  # Allow up to 80% volume loss for reconstruction


def _extract_mesh_state(mesh: trimesh.Trimesh) -> dict:
    """Extract mesh state for logging."""
    try:
        body_count = len(mesh.split(only_watertight=False))
    except:
        body_count = 1
    
    return {
        "faces": len(mesh.faces),
        "vertices": len(mesh.vertices),
        "is_watertight": mesh.is_watertight,
        "volume": float(mesh.volume) if mesh.is_watertight else 0,
        "body_count": body_count,
    }


def _log_action_execution(
    model_id: str,
    pipeline_name: str,
    attempt_number: int,
    action_index: int,
    action_def: dict,
    mesh_before: trimesh.Trimesh,
    mesh_after: trimesh.Trimesh,
    duration_ms: float,
    success: bool,
    error_message: str = None,
) -> None:
    """Log detailed action execution for learning algorithm improvement."""
    if not DETAILED_LOGGING_AVAILABLE:
        return
    
    try:
        detailed_logger = get_detailed_logger()
        
        before_state = _extract_mesh_state(mesh_before)
        after_state = _extract_mesh_state(mesh_after)
        
        # Determine error category
        error_category = None
        if error_message:
            if "timeout" in error_message.lower():
                error_category = "timeout"
            elif "memory" in error_message.lower():
                error_category = "memory"
            elif "geometry" in error_message.lower() or "loss" in error_message.lower():
                error_category = "geometry_loss"
            else:
                error_category = "crash"
        
        detailed_logger.log_action_result(
            model_id=model_id,
            pipeline_name=pipeline_name,
            attempt_number=attempt_number,
            action_index=action_index,
            action_name=action_def["action"],
            action_params=action_def.get("params", {}),
            duration_ms=duration_ms,
            success=success,
            error_message=error_message,
            error_category=error_category,
            mesh_before=before_state,
            mesh_after=after_state,
        )
    except Exception as e:
        logger.debug(f"Failed to log action details: {e}")

def check_geometry_loss(
    original_mesh: trimesh.Trimesh, 
    repaired_mesh: trimesh.Trimesh,
    is_reconstruction_mode: bool = False,
    profile: str = "unknown",
) -> tuple[bool, str, float]:
    """
    Check if repair caused unacceptable geometry loss or excessive increase.
    
    For reconstruction mode (extreme-fragmented meshes), much higher geometry
    loss is acceptable because we're creating new geometry from scattered fragments.
    
    Args:
        original_mesh: Original mesh before repair
        repaired_mesh: Mesh after repair
        is_reconstruction_mode: If True, use lenient reconstruction thresholds
        profile: Mesh profile for adaptive threshold lookup
        
    Returns:
        Tuple of (is_acceptable, reason_if_not, face_loss_pct)
    """
    original_faces = len(original_mesh.faces)
    result_faces = len(repaired_mesh.faces)
    face_loss_pct = 0.0
    
    # Get adaptive thresholds if available
    try:
        from .adaptive_thresholds import get_adaptive_thresholds
        thresholds = get_adaptive_thresholds()
        
        if is_reconstruction_mode:
            max_face_loss = thresholds.get("reconstruction_face_loss_limit_pct", profile)
            max_volume_loss = thresholds.get("reconstruction_volume_loss_limit_pct", profile)
        else:
            max_face_loss = thresholds.get("face_loss_limit_pct", profile)
            max_volume_loss = thresholds.get("volume_loss_limit_pct", profile)
    except Exception:
        # Fallback to hardcoded defaults
        if is_reconstruction_mode:
            max_face_loss = RECONSTRUCTION_FACE_LOSS_PERCENT
            max_volume_loss = RECONSTRUCTION_VOLUME_LOSS_PERCENT
        else:
            max_face_loss = MAX_FACE_LOSS_PERCENT
            max_volume_loss = MAX_VOLUME_LOSS_PERCENT
    
    # Check absolute minimum
    if result_faces < MIN_FACES_ABSOLUTE:
        return False, f"Result has only {result_faces} faces (minimum: {MIN_FACES_ABSOLUTE})", 100.0
    
    # Check face loss percentage
    if original_faces > 0:
        face_loss_pct = (original_faces - result_faces) / original_faces * 100
        if face_loss_pct > 0:  # Only check loss, not gain
            if face_loss_pct > max_face_loss:
                return False, f"Face loss {face_loss_pct:.1f}% exceeds threshold {max_face_loss:.0f}%", face_loss_pct
        
        # Check excessive face increase (e.g., Blender remesh creating millions of faces)
        face_increase_factor = result_faces / original_faces
        if face_increase_factor > MAX_FACE_INCREASE_FACTOR:
            return False, f"Face count increased {face_increase_factor:.0f}x (max: {MAX_FACE_INCREASE_FACTOR}x)", face_loss_pct
    
    # Check volume loss (if both have valid volumes)
    try:
        original_volume = original_mesh.volume if original_mesh.is_volume else 0
        result_volume = repaired_mesh.volume if repaired_mesh.is_volume else 0
        
        if original_volume > 0 and result_volume > 0:
            volume_loss_pct = abs(original_volume - result_volume) / original_volume * 100
            if volume_loss_pct > max_volume_loss:
                return False, f"Volume change {volume_loss_pct:.1f}% exceeds threshold {max_volume_loss:.0f}%", face_loss_pct
    except Exception:
        pass  # Volume check is optional
    
    return True, "", face_loss_pct


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
    
    # Reconstruction mode (for extreme-fragmented meshes)
    # When True, the mesh was reconstructed rather than repaired
    # This means geometry changed significantly but result is printable
    is_reconstruction: bool = False  # True if result is a reconstruction
    reconstruction_method: Optional[str] = None  # Pipeline/action that succeeded
    geometry_loss_pct: float = 0.0  # Actual face loss percentage
    
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
            "is_reconstruction": self.is_reconstruction,
            "reconstruction_method": self.reconstruction_method,
            "geometry_loss_pct": self.geometry_loss_pct,
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
    is_extreme_fragmented = False  # For 1000+ body meshes that need reconstruction
    
    # Get learned thresholds for fragmentation detection
    try:
        from .adaptive_thresholds import get_adaptive_thresholds
        thresholds = get_adaptive_thresholds()
        body_count_extreme_threshold = thresholds.get("body_count_extreme_fragmented")
    except Exception:
        body_count_extreme_threshold = 1000  # Default fallback
    
    if frag_analysis["is_fragmented"]:
        component_count = frag_analysis.get("component_count", 0)
        
        # Check for extreme fragmentation - threshold is LEARNED, not hardcoded!
        if component_count > body_count_extreme_threshold:
            is_extreme_fragmented = True
            if "extreme-fragmented" not in result.issues_found:
                result.issues_found.append("extreme-fragmented")
            logger.warning(f"  EXTREME FRAGMENTATION DETECTED: {component_count} components (threshold: {body_count_extreme_threshold})")
            logger.warning(f"    This mesh requires RECONSTRUCTION, not repair")
            # For extreme fragmentation, we WANT to try reconstruction methods
            blender_unsafe = False  # Allow blender and other reconstruction
        else:
            if "fragmented" not in result.issues_found:
                result.issues_found.append("fragmented")
        
            frag_type = frag_analysis.get("fragmentation_type", "unknown")
            blender_safe = frag_analysis.get("blender_safe", False)
            reason = frag_analysis.get("reason", "")
            
            logger.warning(f"  FRAGMENTED MODEL DETECTED")
            logger.warning(f"    Type: {frag_type}")
            logger.warning(f"    Reason: {reason}")
            logger.warning(f"    Components: {component_count}")
            
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
    
    # Determine the profile for adaptive thresholds
    profile = "extreme-fragmented" if is_extreme_fragmented else "fragmented" if blender_unsafe else "standard"
    
    logger.info(f"Starting pipeline-based repair loop")
    logger.info(f"  Issues: {issues}")
    logger.info(f"  Available pipelines: {len(pipelines)}")
    logger.info(f"  Max attempts: {max_attempts}")
    logger.info(f"  Blender remesh allowed: {not blender_unsafe}")
    logger.info(f"  Reconstruction mode: {is_extreme_fragmented}")
    logger.info(f"  NOTE: Each pipeline starts fresh from the original mesh")
    
    # =========================================================================
    # STEP 3: Try each pipeline until one succeeds
    # =========================================================================
    best_mesh = None
    best_score = -1
    best_face_loss_pct = 100.0
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
        logger.info(f"    Actions: {' -> '.join(action_names)}")
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
            
            # Generate a model_id for logging
            model_id = f"model_{hash(id(original_mesh))}"
            
            # Execute each action in the pipeline sequentially
            for action_idx, action_def in enumerate(pipeline.actions):
                action_name = action_def["action"]
                action_params = action_def.get("params", {})
                
                logger.info(f"Executing action: {action_name} with params: {action_params}")
                
                # Capture mesh state before action
                mesh_before = working_mesh.copy()
                action_start = time.perf_counter()
                action_error = None
                action_success = True
                
                try:
                    working_mesh = ActionRegistry.execute(action_name, working_mesh, action_params)
                except Exception as e:
                    action_error = str(e)
                    action_success = False
                    raise
                finally:
                    action_duration = (time.perf_counter() - action_start) * 1000
                    # Log action execution details
                    _log_action_execution(
                        model_id=model_id,
                        pipeline_name=pipeline.name,
                        attempt_number=attempt_num,
                        action_index=action_idx,
                        action_def=action_def,
                        mesh_before=mesh_before,
                        mesh_after=working_mesh if action_success else mesh_before,
                        duration_ms=action_duration,
                        success=action_success,
                        error_message=action_error,
                    )
            
            repaired_mesh = working_mesh
            
            # Validate geometry
            geom_valid = validate_geometry(repaired_mesh)
            attempt_result.geometry_valid = geom_valid.is_printable
            
            # Check for geometry loss - use reconstruction mode for extreme-fragmented
            geom_acceptable, geom_loss_reason, face_loss_pct = check_geometry_loss(
                original_mesh, repaired_mesh, 
                is_reconstruction_mode=is_extreme_fragmented,
                profile=profile
            )
            
            if geom_valid.is_printable and geom_acceptable:
                logger.info(f"    Pipeline completed, geometry valid (face loss: {face_loss_pct:.1f}%)")
                
                # Run slicer validation
                logger.info(f"    Running slicer validation...")
                slicer_result = validate_mesh(repaired_mesh, slicer=slicer, timeout=timeout, strict=True)
                attempt_result.slicer_result = slicer_result
                
                if slicer_result.success:
                    logger.info(f"    [OK] Slicer validation PASSED!")
                    result.success = True
                    result.final_mesh = repaired_mesh
                    result.final_slicer_result = slicer_result
                    result.issues_resolved.append(pipeline.name)
                    result.geometry_loss_pct = face_loss_pct
                    
                    # Mark as reconstruction if significant geometry loss
                    if face_loss_pct > 50:
                        result.is_reconstruction = True
                        result.reconstruction_method = pipeline.name
                        logger.info(f"    [RECONSTRUCTION] Model reconstructed with {face_loss_pct:.1f}% face loss")
                    
                    attempt_result.success = True
                    attempt_result.duration_ms = (time.perf_counter() - attempt_start) * 1000
                    result.attempts.append(attempt_result)
                    break  # Success!
                else:
                    logger.info(f"    [X] Slicer validation FAILED")
                    
                    # Track best result (prefer lower face loss when scores tie)
                    score = (1 if repaired_mesh.is_watertight else 0) + (1 if repaired_mesh.is_volume else 0)
                    if score > best_score or (score == best_score and face_loss_pct < best_face_loss_pct):
                        best_score = score
                        best_mesh = repaired_mesh
                        best_face_loss_pct = face_loss_pct
                        logger.info(f"    New best mesh (score: {score}, face_loss: {face_loss_pct:.1f}%)")
                    
                    attempt_result.error = f"Slicer issues remain"
            elif geom_valid.is_printable and not geom_acceptable:
                logger.warning(f"    [X] Geometry loss: {geom_loss_reason}")
                attempt_result.error = f"Geometry loss: {geom_loss_reason}"
                attempt_result.geometry_valid = False
                
                # For reconstruction mode, track this as potential best even if exceeds threshold
                if is_extreme_fragmented and repaired_mesh.is_watertight:
                    score = 2  # watertight + volume
                    if score > best_score or (score == best_score and face_loss_pct < best_face_loss_pct):
                        best_score = score
                        best_mesh = repaired_mesh
                        best_face_loss_pct = face_loss_pct
                        logger.info(f"    Tracking as potential reconstruction candidate (face_loss: {face_loss_pct:.1f}%)")
            else:
                logger.warning(f"    [X] Geometry invalid: {geom_valid.issues}")
                attempt_result.error = f"Geometry invalid: {geom_valid.issues}"
            
        except Exception as e:
            logger.error(f"    [X] Pipeline failed: {e}")
            attempt_result.error = str(e)
        
        attempt_result.duration_ms = (time.perf_counter() - attempt_start) * 1000
        result.attempts.append(attempt_result)
    
    # =========================================================================
    # STEP 4: Try evolved pipelines if standard ones failed
    # =========================================================================
    if not result.success and EVOLUTION_AVAILABLE:
        evolution_engine = get_evolution_engine()
        
        # Check if we should try evolution
        if evolution_engine.should_try_evolution(issues, list(attempted_pipelines), result.total_attempts):
            logger.info(f"  Standard pipelines exhausted, trying evolved pipelines...")
            
            # Try existing successful evolved pipelines first
            evolved_candidates = evolution_engine.get_evolved_pipelines_for_issues(
                issues, max_pipelines=3, min_success_rate=0.4
            )
            
            for evolved in evolved_candidates:
                if evolved.name in attempted_pipelines:
                    continue
                
                result.total_attempts += 1
                attempted_pipelines.add(evolved.name)
                
                action_names = [a["action"] for a in evolved.actions]
                logger.info(f"  Attempt {result.total_attempts}: [EVOLVED] {evolved.name}")
                logger.info(f"    Actions: {'  '.join(action_names)}")
                logger.info(f"    Success rate: {evolved.success_rate*100:.1f}%")
                
                if progress_callback:
                    progress_callback(result.total_attempts, f"[EVOLVED] {evolved.name}", max_attempts)
                
                attempt_start = time.perf_counter()
                attempt_result = RepairAttempt(
                    attempt_number=result.total_attempts,
                    pipeline_name=f"[EVOLVED] {evolved.name}",
                    pipeline_actions=action_names,
                    success=False,
                    duration_ms=0,
                )
                
                try:
                    working_mesh = original_mesh.copy()
                    
                    for action_def in evolved.actions:
                        action_name = action_def["action"]
                        action_params = action_def.get("params", {})
                        working_mesh = ActionRegistry.execute(action_name, working_mesh, action_params)
                    
                    repaired_mesh = working_mesh
                    geom_valid = validate_geometry(repaired_mesh)
                    geom_acceptable, _, face_loss_pct = check_geometry_loss(
                        original_mesh, repaired_mesh,
                        is_reconstruction_mode=is_extreme_fragmented,
                        profile=profile
                    )
                    
                    if geom_valid.is_printable and geom_acceptable:
                        slicer_result = validate_mesh(repaired_mesh, slicer=slicer, timeout=timeout, strict=True)
                        
                        if slicer_result.success:
                            logger.info(f"    [OK] EVOLVED PIPELINE SUCCESS!")
                            result.success = True
                            result.final_mesh = repaired_mesh
                            result.final_slicer_result = slicer_result
                            result.issues_resolved.append(f"[EVOLVED] {evolved.name}")
                            result.geometry_loss_pct = face_loss_pct
                            
                            # Mark as reconstruction if significant geometry loss
                            if face_loss_pct > 50:
                                result.is_reconstruction = True
                                result.reconstruction_method = f"[EVOLVED] {evolved.name}"
                            
                            attempt_result.success = True
                            
                            # Record success for learning
                            evolution_engine.record_pipeline_result(
                                evolved.name, success=True,
                                duration_ms=(time.perf_counter() - attempt_start) * 1000
                            )
                            break
                        else:
                            evolution_engine.record_pipeline_result(
                                evolved.name, success=False,
                                duration_ms=(time.perf_counter() - attempt_start) * 1000
                            )
                except Exception as e:
                    logger.warning(f"    Evolved pipeline failed: {e}")
                    attempt_result.error = str(e)
                
                attempt_result.duration_ms = (time.perf_counter() - attempt_start) * 1000
                result.attempts.append(attempt_result)
                
                if result.success:
                    break
            
            # If still not successful, try generating a NEW evolved pipeline
            if not result.success and result.total_attempts < max_attempts:
                logger.info(f"  Generating new evolved pipeline...")
                
                # Get diagnostics for smarter generation
                diagnostics = {
                    "faces": len(original_mesh.faces),
                    "vertices": len(original_mesh.vertices),
                    "body_count": len(original_mesh.split(only_watertight=False)) if hasattr(original_mesh, 'split') else 1,
                    "is_watertight": original_mesh.is_watertight,
                }
                
                new_evolved = evolution_engine.generate_evolved_pipeline(
                    issues=issues,
                    diagnostics=diagnostics,
                    exploration_rate=0.3,  # Higher exploration for failed cases
                )
                
                if new_evolved and new_evolved.name not in attempted_pipelines:
                    result.total_attempts += 1
                    attempted_pipelines.add(new_evolved.name)
                    
                    action_names = [a["action"] for a in new_evolved.actions]
                    logger.info(f"  Attempt {result.total_attempts}: [NEW EVOLVED] {new_evolved.name}")
                    logger.info(f"    Actions: {'  '.join(action_names)}")
                    
                    if progress_callback:
                        progress_callback(result.total_attempts, f"[NEW] {new_evolved.name}", max_attempts)
                    
                    attempt_start = time.perf_counter()
                    attempt_result = RepairAttempt(
                        attempt_number=result.total_attempts,
                        pipeline_name=f"[NEW EVOLVED] {new_evolved.name}",
                        pipeline_actions=action_names,
                        success=False,
                        duration_ms=0,
                    )
                    
                    try:
                        working_mesh = original_mesh.copy()
                        
                        for action_def in new_evolved.actions:
                            action_name = action_def["action"]
                            action_params = action_def.get("params", {})
                            working_mesh = ActionRegistry.execute(action_name, working_mesh, action_params)
                            
                            # Record individual action results for learning
                            evolution_engine.record_action_result(
                                action_def, success=True, duration_ms=0, issues=issues
                            )
                        
                        repaired_mesh = working_mesh
                        geom_valid = validate_geometry(repaired_mesh)
                        geom_acceptable, _, face_loss_pct = check_geometry_loss(
                            original_mesh, repaired_mesh,
                            is_reconstruction_mode=is_extreme_fragmented,
                            profile=profile
                        )
                        
                        if geom_valid.is_printable and geom_acceptable:
                            slicer_result = validate_mesh(repaired_mesh, slicer=slicer, timeout=timeout, strict=True)
                            
                            if slicer_result.success:
                                logger.info(f"    [OK] NEW EVOLVED PIPELINE SUCCESS!")
                                result.success = True
                                result.final_mesh = repaired_mesh
                                result.final_slicer_result = slicer_result
                                result.issues_resolved.append(f"[NEW EVOLVED] {new_evolved.name}")
                                result.geometry_loss_pct = face_loss_pct
                                
                                # Mark as reconstruction if significant geometry loss
                                if face_loss_pct > 50:
                                    result.is_reconstruction = True
                                    result.reconstruction_method = f"[NEW EVOLVED] {new_evolved.name}"
                                
                                attempt_result.success = True
                                
                                evolution_engine.record_pipeline_result(
                                    new_evolved.name, success=True,
                                    duration_ms=(time.perf_counter() - attempt_start) * 1000
                                )
                            else:
                                evolution_engine.record_pipeline_result(
                                    new_evolved.name, success=False,
                                    duration_ms=(time.perf_counter() - attempt_start) * 1000
                                )
                    except Exception as e:
                        logger.warning(f"    New evolved pipeline failed: {e}")
                        attempt_result.error = str(e)
                        # Record action failure
                        for action_def in new_evolved.actions:
                            evolution_engine.record_action_result(
                                action_def, success=False, duration_ms=0, issues=issues
                            )
                    
                    attempt_result.duration_ms = (time.perf_counter() - attempt_start) * 1000
                    result.attempts.append(attempt_result)
    
    result.total_duration_ms = (time.perf_counter() - start_time) * 1000
    
    if not result.success:
        result.final_mesh = best_mesh if best_mesh is not None else original_mesh
        result.geometry_loss_pct = best_face_loss_pct if best_mesh is not None else 0
        if not result.error:
            result.error = f"Failed after {result.total_attempts} pipeline attempts"
    
    return result
