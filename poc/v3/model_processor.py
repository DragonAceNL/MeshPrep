# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Single model processing for POC v3 batch processing.

Contains the core logic for processing a single mesh file through
the repair pipeline, including fingerprinting, repair, validation,
and report generation.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Add POC v2 to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "v2"))

from meshprep_poc.mesh_ops import load_mesh, save_mesh, compute_diagnostics
from meshprep_poc.validation import validate_repair
from meshprep_poc.filter_script import FilterScriptRunner, get_preset
from meshprep_poc.slicer_repair_loop import run_slicer_repair_loop
from meshprep_poc.actions.blender_actions import is_blender_available
from meshprep_poc.fingerprint import compute_file_fingerprint, compute_full_file_hash
from meshprep_poc.learning_engine import get_learning_engine

from config import (
    REPORTS_PATH, FIXED_OUTPUT_PATH, THINGI10K_PATH,
    DEFAULT_MAX_REPAIR_ATTEMPTS, DEFAULT_REPAIR_TIMEOUT,
    DEFAULT_DECIMATION_TRIGGER_FACES,
)
from test_result import TestResult
from progress_tracker import Progress, save_progress
from mesh_utils import (
    check_geometry_loss, decimate_mesh, extract_mesh_diagnostics, 
    render_mesh_image, ADAPTIVE_THRESHOLDS_AVAILABLE,
)
from filter_persistence import save_filter_info
from report_generator import generate_model_report

# Try to import adaptive thresholds
if ADAPTIVE_THRESHOLDS_AVAILABLE:
    from meshprep_poc.adaptive_thresholds import get_adaptive_thresholds

logger = logging.getLogger(__name__)

# Global progress reference for callback
_current_progress: Optional[Progress] = None
_progress_file: Optional[Path] = None


def set_progress_file(progress_file: Path) -> None:
    """Set the progress file path for saving updates."""
    global _progress_file
    _progress_file = progress_file


def update_action_progress(action_index: int, action_name: str, total_actions: int) -> None:
    """Callback to update progress with current action.
    
    Args:
        action_index: Zero-based index of current action
        action_name: Name of the action being executed
        total_actions: Total number of actions in the filter
    """
    global _current_progress, _progress_file
    if _current_progress and _progress_file:
        _current_progress.current_action = action_name
        _current_progress.current_step = action_index + 1
        _current_progress.total_steps = total_actions
        save_progress(_current_progress, _progress_file)


def get_best_filter(mesh):
    """Select the best filter based on mesh analysis.
    
    Args:
        mesh: Trimesh object to analyze
        
    Returns:
        FilterScript preset to use
    """
    try:
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            logger.debug(f"Multi-component model ({len(components)} parts) - using conservative")
            return get_preset("conservative-repair")
    except Exception:
        pass
    
    return get_preset("conservative-repair")  # Safe default


def process_single_model(
    stl_path: Path, 
    skip_if_clean: bool = True, 
    progress: Optional[Progress] = None
) -> TestResult:
    """Process a single model using the POC v2 slicer repair loop.
    
    This is a thin wrapper around the v2 repair loop that:
    1. Loads the mesh
    2. Runs the slicer repair loop (which includes STRICT pre-check)
    3. Captures metrics and generates reports
    
    Args:
        stl_path: Path to the mesh file
        skip_if_clean: If True, skip repair if model already passes STRICT slicer check
        progress: Optional Progress object for live updates
        
    Returns:
        TestResult with all metrics and status
    """
    global _current_progress
    _current_progress = progress
    file_id = stl_path.stem
    start_time = time.time()
    
    result = TestResult(
        file_id=file_id,
        file_path=str(stl_path),
    )
    
    try:
        # Compute fingerprint FIRST (from original file bytes)
        result.model_fingerprint = compute_file_fingerprint(stl_path)
        result.original_file_hash = compute_full_file_hash(stl_path)
        logger.info(f"  Fingerprint: {result.model_fingerprint}")
        
        # Load mesh
        original = load_mesh(stl_path)
        original_diag = compute_diagnostics(original)
        
        result.original_vertices = original_diag.vertex_count
        result.original_faces = original_diag.face_count
        result.original_volume = original_diag.volume
        result.original_watertight = original_diag.is_watertight
        result.original_manifold = original_diag.is_volume
        result.original_file_size = stl_path.stat().st_size
        
        # Additional diagnostics
        try:
            result.original_components = len(original.split(only_watertight=False))
        except:
            result.original_components = 1
        
        # Run slicer repair loop (includes STRICT pre-check)
        repair_result = run_slicer_repair_loop(
            mesh=original,
            slicer="auto",
            max_attempts=DEFAULT_MAX_REPAIR_ATTEMPTS,
            timeout=DEFAULT_REPAIR_TIMEOUT,
            skip_if_clean=skip_if_clean,
            progress_callback=update_action_progress,
        )
        
        # Capture pre-check results
        result.precheck_passed = repair_result.precheck_passed
        result.precheck_skipped = repair_result.precheck_skipped
        
        # Capture reconstruction info
        result.is_reconstruction = repair_result.is_reconstruction
        result.reconstruction_method = repair_result.reconstruction_method or ""
        result.geometry_loss_pct = repair_result.geometry_loss_pct
        
        if repair_result.precheck_skipped:
            # Model was already clean - no repair needed
            logger.info(f"  PRE-CHECK PASSED: Model already clean (from v2 repair loop)")
            result.success = True
            result.filter_used = "none (already clean)"
            result.duration_ms = repair_result.total_duration_ms
            
            # Result = Original (no changes)
            result.result_vertices = result.original_vertices
            result.result_faces = result.original_faces
            result.result_volume = result.original_volume
            result.result_watertight = result.original_watertight
            result.result_manifold = result.original_manifold
            
            # Generate report showing it was skipped
            _generate_report(stl_path, original, original, result, None)
            return result
        
        # Repair was attempted
        if repair_result.success and repair_result.final_mesh is not None:
            repaired = repair_result.final_mesh
            result.filter_used = "slicer-repair-loop"
            
            # Check if Blender escalation was used
            for attempt in repair_result.attempts:
                if "blender" in attempt.pipeline_name.lower():
                    result.escalation_used = True
                    result.filter_used = "slicer-repair-loop (blender)"
                    break
        else:
            # Repair failed, fall back to conservative filter script approach
            logger.info(f"  Slicer repair loop failed, trying filter script approach...")
            filter_script = get_best_filter(original)
            result.filter_used = filter_script.name
            
            runner = FilterScriptRunner(stop_on_error=False)
            filter_result = runner.run(filter_script, original, progress_callback=update_action_progress)
            
            if not filter_result.success or filter_result.final_mesh is None:
                result.success = False
                result.error = filter_result.error or repair_result.error or "Repair failed"
                result.duration_ms = (time.time() - start_time) * 1000
                
                # Generate failed report
                _generate_report(stl_path, original, original, result, None)
                return result
            
            repaired = filter_result.final_mesh
            
            # Check for geometry loss and escalate if needed
            validation = validate_repair(original, repaired)
            significant_loss, vol_loss, face_loss = check_geometry_loss(original_diag, repaired)
            
            needs_escalation = (
                not validation.geometric.is_printable or
                significant_loss
            )
            
            if needs_escalation and is_blender_available():
                logger.info(f"  Escalating to Blender...")
                escalation_script = get_preset("blender-remesh")
                escalation_result = runner.run(escalation_script, original, progress_callback=update_action_progress)
                
                if escalation_result.success and escalation_result.final_mesh is not None:
                    repaired = escalation_result.final_mesh
                    result.filter_used = "blender-remesh"
                    result.escalation_used = True
        
        # Calculate geometry changes
        significant_loss, vol_loss, face_loss = check_geometry_loss(original_diag, repaired)
        result.volume_change_pct = vol_loss
        result.face_change_pct = face_loss
        
        # Capture printability status BEFORE decimation
        validation = validate_repair(original, repaired)
        is_printable_before_decimate = validation.geometric.is_printable
        
        # Decimate if mesh is too large
        if ADAPTIVE_THRESHOLDS_AVAILABLE:
            thresholds = get_adaptive_thresholds()
            decimation_trigger = int(thresholds.get("decimation_trigger_faces", "unknown"))
        else:
            decimation_trigger = DEFAULT_DECIMATION_TRIGGER_FACES
        
        if len(repaired.faces) > decimation_trigger:
            original_repaired = repaired.copy()
            repaired = decimate_mesh(repaired, profile="unknown")
            
            # Check if decimation broke manifold - if so, keep original large mesh
            if is_printable_before_decimate and not repaired.is_watertight:
                logger.warning("  Decimation broke manifold status - keeping original large mesh")
                repaired = original_repaired
        
        # Final metrics (after potential decimation)
        result.result_vertices = len(repaired.vertices)
        result.result_faces = len(repaired.faces)
        result.result_volume = repaired.volume if repaired.is_volume else 0
        result.result_watertight = repaired.is_watertight
        result.result_manifold = repaired.is_volume
        
        # Result diagnostics
        try:
            result.result_components = len(repaired.split(only_watertight=False))
        except:
            result.result_components = 1
        
        # Success based on FINAL mesh state
        result.success = repaired.is_watertight and repaired.is_volume
        result.duration_ms = (time.time() - start_time) * 1000
        
        # Save repaired model to fixed directory
        fixed_path = FIXED_OUTPUT_PATH / f"{file_id}.stl"
        if result.success:
            save_mesh(repaired, fixed_path)
            result.fixed_file_size = fixed_path.stat().st_size
            logger.info(f"  Saved fixed model to {fixed_path}")
        
        # Extract diagnostics for analysis
        before_diagnostics = extract_mesh_diagnostics(original, "before")
        after_diagnostics = extract_mesh_diagnostics(repaired, "after") if repaired else None
        
        # Save detailed filter info for later analysis
        save_filter_info(
            file_id=file_id,
            filter_used=result.filter_used,
            escalated=result.escalation_used,
            repair_result=repair_result,
            model_fingerprint=result.model_fingerprint,
            original_filename=stl_path.name,
            original_format=stl_path.suffix.lstrip('.').lower(),
            before_diagnostics=before_diagnostics,
            after_diagnostics=after_diagnostics,
        )
        
        # Feed result to learning engine
        _update_learning_engine(result, repair_result, before_diagnostics, after_diagnostics)
        
        # Feed observations to adaptive thresholds engine
        _update_adaptive_thresholds(
            result, repair_result, before_diagnostics, 
            vol_loss, face_loss, decimation_trigger
        )
        
        # Generate report
        _generate_report(stl_path, original, repaired, result, fixed_path if result.success else None)
        
        return result
        
    except Exception as e:
        result.success = False
        result.error = f"{type(e).__name__}: {str(e)}"
        result.duration_ms = (time.time() - start_time) * 1000
        logger.error(f"  Error: {result.error}")
        return result


def _generate_report(stl_path: Path, original, repaired, result: TestResult, fixed_path: Optional[Path] = None):
    """Generate HTML report with before/after images."""
    images_dir = REPORTS_PATH / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Render images
    before_img = images_dir / f"{stl_path.stem}_before.png"
    after_img = images_dir / f"{stl_path.stem}_after.png"
    
    render_mesh_image(original, before_img, "Before")
    render_mesh_image(repaired, after_img, "After")
    
    # Use the dedicated report generator module
    generate_model_report(
        stl_path=stl_path,
        result=result,
        reports_path=REPORTS_PATH,
        thingi10k_path=THINGI10K_PATH,
        fixed_path=fixed_path,
    )


def _update_learning_engine(result: TestResult, repair_result, before_diagnostics, after_diagnostics):
    """Feed result to learning engine for continuous improvement."""
    try:
        learning_engine = get_learning_engine()
        filter_data = {
            "success": result.success,
            "escalated_to_blender": result.escalation_used,
            "precheck": {
                "passed": repair_result.precheck_passed if repair_result else False,
                "skipped": repair_result.precheck_skipped if repair_result else False,
                "mesh_info": {
                    "issues": repair_result.issues_found if repair_result else [],
                } if repair_result else None,
            },
            "repair_attempts": {
                "total_attempts": repair_result.total_attempts if repair_result else 0,
                "attempts": [
                    {
                        "pipeline_name": a.pipeline_name,
                        "success": a.success,
                        "duration_ms": a.duration_ms,
                    }
                    for a in (repair_result.attempts if repair_result else [])
                ],
            },
            "diagnostics": {
                "before": before_diagnostics,
                "after": after_diagnostics,
            },
        }
        learning_engine.record_result(filter_data)
    except Exception as le_error:
        logger.debug(f"Learning engine update failed: {le_error}")


def _update_adaptive_thresholds(
    result: TestResult, 
    repair_result, 
    before_diagnostics, 
    vol_loss: float, 
    face_loss: float,
    decimation_trigger: int
):
    """Feed observations to adaptive thresholds engine."""
    try:
        if not ADAPTIVE_THRESHOLDS_AVAILABLE:
            return
            
        adaptive = get_adaptive_thresholds()
        profile = "unknown"
        
        # Calculate quality score based on geometry preservation
        quality = 1.0
        if vol_loss > 0:
            quality -= min(vol_loss / 100, 0.5)
        if face_loss > 0:
            quality -= min(face_loss / 100, 0.3)
        quality = max(0, quality)
        
        # Record geometry loss observations
        adaptive.record_geometry_loss(
            volume_loss_pct=vol_loss,
            face_loss_pct=face_loss,
            success=result.success,
            quality=quality,
            profile=profile,
            escalated=result.escalation_used,
        )
        
        # Record decimation observations if decimation was performed
        if result.original_faces > decimation_trigger:
            decimation_success = result.success and result.result_watertight
            adaptive.record_decimation(
                original_faces=result.original_faces,
                target_faces=int(adaptive.get("decimation_target_faces", profile)),
                result_faces=result.result_faces,
                success=decimation_success,
                quality=quality,
                profile=profile,
            )
        
        # Record repair attempt observations
        if repair_result:
            adaptive.record_repair_attempts(
                attempts_used=repair_result.total_attempts,
                duration_ms=repair_result.total_duration_ms,
                success=result.success,
                profile=profile,
            )
        
        # Record body count threshold observations
        if before_diagnostics and "body_count" in before_diagnostics:
            body_count = before_diagnostics["body_count"]
            
            extreme_threshold = adaptive.get("body_count_extreme_fragmented")
            adaptive.record_observation(
                threshold_name="body_count_extreme_fragmented",
                threshold_value=extreme_threshold,
                actual_value=body_count,
                success=result.success,
                quality=quality,
                profile=profile,
            )
            
            frag_threshold = adaptive.get("body_count_fragmented")
            adaptive.record_observation(
                threshold_name="body_count_fragmented",
                threshold_value=frag_threshold,
                actual_value=body_count,
                success=result.success,
                quality=quality,
                profile=profile,
            )
    except Exception as at_error:
        logger.debug(f"Adaptive thresholds update failed: {at_error}")
