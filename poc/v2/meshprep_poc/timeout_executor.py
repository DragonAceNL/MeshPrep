# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Timeout-aware action executor for MeshPrep.

Wraps action execution with intelligent timeout detection based on learned durations.
Uses a two-stage timeout system:
1. Soft timeout: Warn but continue (logs that action is running slow)
2. Hard timeout: Kill process only after giving it ample time

The hard timeout factor is LEARNED per action based on observed variance:
- Consistent actions (low variance) get tighter timeouts
- Unpredictable actions (high variance) get more generous timeouts

This uses multiprocessing to truly kill hung operations (threads can't be killed).
"""

import logging
import multiprocessing as mp
from multiprocessing import Process, Queue
import queue
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Any, Callable
import traceback

import trimesh
import numpy as np

logger = logging.getLogger(__name__)

# Import the duration predictor
try:
    from .action_duration_predictor import (
        get_duration_predictor,
        MeshCharacteristics,
        DurationPrediction,
    )
    PREDICTOR_AVAILABLE = True
except ImportError:
    PREDICTOR_AVAILABLE = False
    logger.warning("Action duration predictor not available")


@dataclass
class TimeoutActionResult:
    """Result of a timeout-aware action execution."""
    success: bool
    mesh: Optional[trimesh.Trimesh]
    duration_ms: float
    timed_out: bool
    
    # Timeout info
    soft_timeout_exceeded: bool = False  # Exceeded soft timeout but completed
    hard_timeout_exceeded: bool = False  # Was killed by hard timeout
    
    error: Optional[str] = None
    prediction: Optional[DurationPrediction] = None


def _run_action_in_process(
    action_name: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    params: dict,
    result_queue: Queue,
) -> None:
    """Run an action in a separate process (for timeout capability).
    
    This function runs in a subprocess and puts results in the queue.
    We pass vertices/faces arrays instead of the mesh to avoid pickling issues.
    """
    try:
        # Reconstruct mesh from arrays
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Import and run the action
        from .actions import ActionRegistry
        
        start = time.perf_counter()
        result_mesh = ActionRegistry.execute(action_name, mesh, params)
        duration_ms = (time.perf_counter() - start) * 1000
        
        # Send back the result (as arrays to avoid pickling issues)
        result_queue.put({
            "success": True,
            "vertices": result_mesh.vertices.copy(),
            "faces": result_mesh.faces.copy(),
            "duration_ms": duration_ms,
        })
    except Exception as e:
        result_queue.put({
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}",
            "traceback": traceback.format_exc(),
        })


def execute_with_timeout(
    action_name: str,
    mesh: trimesh.Trimesh,
    params: dict,
    model_id: str = "",
    pipeline_name: str = "",
    progress_callback: Optional[Callable[[str], None]] = None,
) -> TimeoutActionResult:
    """Execute an action with intelligent timeout based on learned duration patterns.
    
    Uses a two-stage timeout system:
    1. Soft timeout: Log warning but continue waiting
    2. Hard timeout: Kill the process (learned factor specific to this action)
    
    The hard timeout factor is learned per action:
    - Consistent actions (low timing variance) get tighter timeouts (e.g., 3x average)
    - Unpredictable actions (high variance) get generous timeouts (e.g., 15x average)
    
    Args:
        action_name: Name of the action to execute
        mesh: Input mesh
        params: Action parameters
        model_id: Model identifier for learning
        pipeline_name: Pipeline name for learning
        progress_callback: Optional callback for status updates
        
    Returns:
        TimeoutActionResult with mesh or timeout info
    """
    # Get mesh characteristics
    try:
        body_count = len(mesh.split(only_watertight=False))
    except:
        body_count = 1
    
    mesh_chars = MeshCharacteristics(
        face_count=len(mesh.faces),
        vertex_count=len(mesh.vertices),
        body_count=body_count,
    )
    
    # Get prediction
    prediction = None
    soft_timeout_s = 30  # Default soft timeout
    hard_timeout_s = 300  # Default hard timeout (5 minutes)
    
    if PREDICTOR_AVAILABLE:
        predictor = get_duration_predictor()
        prediction = predictor.predict_duration(action_name, mesh_chars)
        
        # Check if we should skip this action entirely
        if prediction.should_skip:
            logger.warning(
                f"Skipping {action_name} for {mesh_chars.size_bin} mesh "
                f"(hang risk: {prediction.hang_risk:.0%}, learned from history)"
            )
            return TimeoutActionResult(
                success=False,
                mesh=None,
                duration_ms=0,
                timed_out=False,
                error=f"Action skipped: {prediction.hang_risk:.0%} hang risk for {mesh_chars.size_bin} meshes",
                prediction=prediction,
            )
        
        # Use learned timeouts
        soft_timeout_s = prediction.soft_timeout_ms / 1000
        hard_timeout_s = prediction.hard_timeout_ms / 1000
        
        logger.info(
            f"Executing {action_name} on {mesh_chars.size_bin} mesh ({mesh_chars.face_count:,} faces)"
        )
        logger.debug(
            f"  Timeouts: soft={soft_timeout_s:.1f}s, hard={hard_timeout_s:.1f}s "
            f"(factor={prediction.timeout_factor:.1f}x, confidence={prediction.confidence:.0%})"
        )
    
    # Decide execution strategy based on expected duration
    if hard_timeout_s < 60 and mesh_chars.face_count < 100000:
        # Short timeout and small mesh - run directly for efficiency
        return _execute_direct_with_monitoring(
            action_name, mesh, params, soft_timeout_s, hard_timeout_s,
            prediction, model_id, pipeline_name, mesh_chars, progress_callback
        )
    else:
        # Large mesh or long timeout - use subprocess for killability
        return _execute_in_subprocess(
            action_name, mesh, params, soft_timeout_s, hard_timeout_s,
            prediction, model_id, pipeline_name, mesh_chars, progress_callback
        )


def _execute_direct_with_monitoring(
    action_name: str,
    mesh: trimesh.Trimesh,
    params: dict,
    soft_timeout_s: float,
    hard_timeout_s: float,
    prediction: Optional[DurationPrediction],
    model_id: str,
    pipeline_name: str,
    mesh_chars: MeshCharacteristics,
    progress_callback: Optional[Callable[[str], None]],
) -> TimeoutActionResult:
    """Execute action directly with timing monitoring.
    
    This is faster but can't truly kill hung operations.
    Used for small meshes where we trust the action won't hang.
    """
    from .actions import ActionRegistry
    
    start = time.perf_counter()
    soft_timeout_exceeded = False
    
    try:
        result_mesh = ActionRegistry.execute(action_name, mesh, params)
        duration_ms = (time.perf_counter() - start) * 1000
        duration_s = duration_ms / 1000
        
        # Check if it was slower than soft timeout
        if duration_s > soft_timeout_s:
            soft_timeout_exceeded = True
            logger.info(
                f"  {action_name} completed but was slow: {duration_s:.1f}s "
                f"(soft timeout: {soft_timeout_s:.1f}s)"
            )
        
        # Record successful completion for learning
        if PREDICTOR_AVAILABLE:
            predictor = get_duration_predictor()
            predictor.record_completion(
                action_name, mesh_chars, duration_ms,
                model_id=model_id, pipeline_name=pipeline_name,
                was_slow_warning=soft_timeout_exceeded,
                predicted_timeout_ms=hard_timeout_s * 1000 if prediction else 0,
            )
        
        return TimeoutActionResult(
            success=True,
            mesh=result_mesh,
            duration_ms=duration_ms,
            timed_out=False,
            soft_timeout_exceeded=soft_timeout_exceeded,
            prediction=prediction,
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - start) * 1000
        return TimeoutActionResult(
            success=False,
            mesh=None,
            duration_ms=duration_ms,
            timed_out=False,
            error=str(e),
            prediction=prediction,
        )


def _execute_in_subprocess(
    action_name: str,
    mesh: trimesh.Trimesh,
    params: dict,
    soft_timeout_s: float,
    hard_timeout_s: float,
    prediction: Optional[DurationPrediction],
    model_id: str,
    pipeline_name: str,
    mesh_chars: MeshCharacteristics,
    progress_callback: Optional[Callable[[str], None]],
) -> TimeoutActionResult:
    """Execute action in subprocess with two-stage timeout.
    
    Stage 1: Wait until soft_timeout, if exceeded log warning but continue
    Stage 2: Wait until hard_timeout, if exceeded kill the process
    """
    result_queue = mp.Queue()
    
    # Start subprocess
    process = Process(
        target=_run_action_in_process,
        args=(action_name, mesh.vertices.copy(), mesh.faces.copy(), params, result_queue),
    )
    process.start()
    
    start = time.perf_counter()
    soft_timeout_exceeded = False
    
    # Stage 1: Wait until soft timeout
    try:
        result = result_queue.get(timeout=soft_timeout_s)
        # Completed within soft timeout
        duration_ms = (time.perf_counter() - start) * 1000
        return _handle_subprocess_result(
            result, duration_ms, prediction, action_name, mesh_chars,
            model_id, pipeline_name, soft_timeout_exceeded=False,
            hard_timeout_s=hard_timeout_s, progress_callback=progress_callback
        )
    except queue.Empty:
        # Exceeded soft timeout, but don't kill yet
        soft_timeout_exceeded = True
        elapsed = time.perf_counter() - start
        
        if progress_callback:
            progress_callback(f"{action_name} running slow ({elapsed:.0f}s)...")
        
        logger.warning(
            f"  {action_name} exceeded soft timeout ({soft_timeout_s:.1f}s), "
            f"continuing until hard timeout ({hard_timeout_s:.1f}s)..."
        )
    
    # Stage 2: Wait for remaining time until hard timeout
    remaining_s = hard_timeout_s - (time.perf_counter() - start)
    
    if remaining_s > 0:
        try:
            result = result_queue.get(timeout=remaining_s)
            # Completed (slowly) before hard timeout
            duration_ms = (time.perf_counter() - start) * 1000
            
            logger.info(
                f"  {action_name} completed after {duration_ms/1000:.1f}s "
                f"(exceeded soft timeout but finished before hard timeout)"
            )
            
            return _handle_subprocess_result(
                result, duration_ms, prediction, action_name, mesh_chars,
                model_id, pipeline_name, soft_timeout_exceeded=True,
                hard_timeout_s=hard_timeout_s, progress_callback=progress_callback
            )
        except queue.Empty:
            pass  # Will handle below
    
    # Hard timeout exceeded - kill the process
    duration_ms = hard_timeout_s * 1000
    
    logger.error(
        f"  {action_name} KILLED after {hard_timeout_s:.1f}s hard timeout "
        f"(learned factor: {prediction.timeout_factor:.1f}x)" if prediction else
        f"  {action_name} KILLED after {hard_timeout_s:.1f}s hard timeout"
    )
    
    # Kill the hung process
    process.terminate()
    process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)
    
    # Record as hang for learning
    if PREDICTOR_AVAILABLE:
        predictor = get_duration_predictor()
        predictor.record_hang(
            action_name, mesh_chars, duration_ms,
            model_id=model_id, pipeline_name=pipeline_name,
        )
    
    return TimeoutActionResult(
        success=False,
        mesh=None,
        duration_ms=duration_ms,
        timed_out=True,
        soft_timeout_exceeded=True,
        hard_timeout_exceeded=True,
        error=f"Action killed after {hard_timeout_s:.1f}s hard timeout",
        prediction=prediction,
    )


def _handle_subprocess_result(
    result: dict,
    duration_ms: float,
    prediction: Optional[DurationPrediction],
    action_name: str,
    mesh_chars: MeshCharacteristics,
    model_id: str,
    pipeline_name: str,
    soft_timeout_exceeded: bool,
    hard_timeout_s: float,
    progress_callback: Optional[Callable[[str], None]],
) -> TimeoutActionResult:
    """Handle result from subprocess execution."""
    if result["success"]:
        # Reconstruct mesh from result
        result_mesh = trimesh.Trimesh(
            vertices=result["vertices"],
            faces=result["faces"],
        )
        
        # Record successful duration for learning
        if PREDICTOR_AVAILABLE:
            predictor = get_duration_predictor()
            predictor.record_completion(
                action_name, mesh_chars, duration_ms,
                model_id=model_id, pipeline_name=pipeline_name,
                was_slow_warning=soft_timeout_exceeded,
                predicted_timeout_ms=hard_timeout_s * 1000 if prediction else 0,
            )
        
        return TimeoutActionResult(
            success=True,
            mesh=result_mesh,
            duration_ms=duration_ms,
            timed_out=False,
            soft_timeout_exceeded=soft_timeout_exceeded,
            prediction=prediction,
        )
    else:
        return TimeoutActionResult(
            success=False,
            mesh=None,
            duration_ms=duration_ms,
            timed_out=False,
            soft_timeout_exceeded=soft_timeout_exceeded,
            error=result.get("error", "Unknown error"),
            prediction=prediction,
        )


def get_timeout_info(action_name: str, mesh: trimesh.Trimesh) -> dict:
    """Get timeout information for an action on a specific mesh.
    
    Useful for logging/debugging timeout decisions.
    
    Args:
        action_name: Action name
        mesh: Input mesh
        
    Returns:
        Dict with timeout info including learned factors
    """
    if not PREDICTOR_AVAILABLE:
        return {
            "soft_timeout_s": 30,
            "hard_timeout_s": 300,
            "learned_factor": None,
            "confidence": 0,
        }
    
    try:
        body_count = len(mesh.split(only_watertight=False))
    except:
        body_count = 1
    
    mesh_chars = MeshCharacteristics(
        face_count=len(mesh.faces),
        vertex_count=len(mesh.vertices),
        body_count=body_count,
    )
    
    predictor = get_duration_predictor()
    prediction = predictor.predict_duration(action_name, mesh_chars)
    
    return {
        "action_name": action_name,
        "size_bin": prediction.size_bin,
        "predicted_ms": prediction.predicted_ms,
        "soft_timeout_s": prediction.soft_timeout_ms / 1000,
        "hard_timeout_s": prediction.hard_timeout_ms / 1000,
        "learned_factor": prediction.timeout_factor,
        "confidence": prediction.confidence,
        "sample_count": prediction.sample_count,
        "hang_risk": prediction.hang_risk,
        "variance_ratio": prediction.variance_ratio,
        "should_skip": prediction.should_skip,
        "p90_ms": prediction.p90_ms,
        "max_observed_ms": prediction.max_observed_ms,
    }


def should_skip_action(action_name: str, mesh: trimesh.Trimesh) -> Tuple[bool, Optional[str], float]:
    """Check if an action should be skipped for this mesh due to hang risk.
    
    Args:
        action_name: Action name
        mesh: Input mesh
        
    Returns:
        Tuple of (should_skip, alternative_action_name, hang_risk)
    """
    if not PREDICTOR_AVAILABLE:
        return False, None, 0.0
    
    try:
        body_count = len(mesh.split(only_watertight=False))
    except:
        body_count = 1
    
    mesh_chars = MeshCharacteristics(
        face_count=len(mesh.faces),
        vertex_count=len(mesh.vertices),
        body_count=body_count,
    )
    
    predictor = get_duration_predictor()
    prediction = predictor.predict_duration(action_name, mesh_chars)
    
    return prediction.should_skip, prediction.alternative_action, prediction.hang_risk
