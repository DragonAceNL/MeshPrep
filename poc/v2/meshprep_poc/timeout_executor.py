# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Timeout-aware action executor for MeshPrep.

Wraps action execution with timeout detection based on learned durations.
When an action exceeds its predicted timeout, it's killed and recorded as a hang.

This uses multiprocessing to truly kill hung operations (threads can't be killed).
"""

import logging
import multiprocessing as mp
from multiprocessing import Process, Queue
import queue
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Any
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
    timeout_ms: Optional[float] = None,
    model_id: str = "",
    pipeline_name: str = "",
) -> TimeoutActionResult:
    """Execute an action with timeout based on predicted duration.
    
    If the action takes longer than the predicted timeout (based on mesh
    characteristics and learned history), it's killed and recorded as a hang.
    
    Args:
        action_name: Name of the action to execute
        mesh: Input mesh
        params: Action parameters
        timeout_ms: Override timeout in milliseconds (uses prediction if None)
        model_id: Model identifier for learning
        pipeline_name: Pipeline name for learning
        
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
    if PREDICTOR_AVAILABLE:
        predictor = get_duration_predictor()
        prediction = predictor.predict_duration(action_name, mesh_chars)
        
        # Check if we should skip this action entirely
        if prediction.should_skip:
            logger.warning(
                f"Skipping {action_name} for {mesh_chars.size_bin} mesh "
                f"(hang risk: {prediction.hang_risk:.0%})"
            )
            return TimeoutActionResult(
                success=False,
                mesh=None,
                duration_ms=0,
                timed_out=False,
                error=f"Action skipped due to high hang risk ({prediction.hang_risk:.0%})",
                prediction=prediction,
            )
        
        # Use predicted timeout if not overridden
        if timeout_ms is None:
            timeout_ms = prediction.timeout_ms
    
    # Default timeout if no prediction
    if timeout_ms is None:
        timeout_ms = 60000  # 60 seconds default
    
    timeout_s = timeout_ms / 1000
    
    logger.debug(
        f"Executing {action_name} with {timeout_s:.1f}s timeout "
        f"(predicted: {prediction.predicted_ms/1000:.1f}s)" if prediction else
        f"Executing {action_name} with {timeout_s:.1f}s timeout"
    )
    
    # For short timeouts or simple actions, run directly without subprocess overhead
    if timeout_s > 300 or mesh_chars.face_count > 100000:
        # Large mesh or long timeout - use subprocess for killability
        return _execute_in_subprocess(
            action_name, mesh, params, timeout_s, 
            prediction, model_id, pipeline_name, mesh_chars
        )
    else:
        # Small mesh - run directly with timing check
        return _execute_direct_with_timeout(
            action_name, mesh, params, timeout_s,
            prediction, model_id, pipeline_name, mesh_chars
        )


def _execute_direct_with_timeout(
    action_name: str,
    mesh: trimesh.Trimesh,
    params: dict,
    timeout_s: float,
    prediction: Optional[DurationPrediction],
    model_id: str,
    pipeline_name: str,
    mesh_chars: MeshCharacteristics,
) -> TimeoutActionResult:
    """Execute action directly (no subprocess) with timing.
    
    This is faster for small meshes but can't truly kill hung operations.
    We rely on the predictor to avoid known-hanging combinations.
    """
    from .actions import ActionRegistry
    
    start = time.perf_counter()
    try:
        result_mesh = ActionRegistry.execute(action_name, mesh, params)
        duration_ms = (time.perf_counter() - start) * 1000
        
        # Record successful duration for learning
        if PREDICTOR_AVAILABLE:
            predictor = get_duration_predictor()
            predictor.record_duration(
                action_name, mesh_chars, duration_ms, 
                success=True, model_id=model_id, pipeline_name=pipeline_name
            )
        
        return TimeoutActionResult(
            success=True,
            mesh=result_mesh,
            duration_ms=duration_ms,
            timed_out=False,
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
    timeout_s: float,
    prediction: Optional[DurationPrediction],
    model_id: str,
    pipeline_name: str,
    mesh_chars: MeshCharacteristics,
) -> TimeoutActionResult:
    """Execute action in subprocess with hard timeout capability."""
    
    result_queue = mp.Queue()
    
    # Start subprocess
    process = Process(
        target=_run_action_in_process,
        args=(action_name, mesh.vertices.copy(), mesh.faces.copy(), params, result_queue),
    )
    process.start()
    
    start = time.perf_counter()
    
    # Wait for result with timeout
    try:
        result = result_queue.get(timeout=timeout_s)
        duration_ms = (time.perf_counter() - start) * 1000
        
        if result["success"]:
            # Reconstruct mesh from result
            result_mesh = trimesh.Trimesh(
                vertices=result["vertices"],
                faces=result["faces"],
            )
            
            # Record successful duration
            if PREDICTOR_AVAILABLE:
                predictor = get_duration_predictor()
                predictor.record_duration(
                    action_name, mesh_chars, duration_ms,
                    success=True, model_id=model_id, pipeline_name=pipeline_name
                )
            
            return TimeoutActionResult(
                success=True,
                mesh=result_mesh,
                duration_ms=duration_ms,
                timed_out=False,
                prediction=prediction,
            )
        else:
            return TimeoutActionResult(
                success=False,
                mesh=None,
                duration_ms=duration_ms,
                timed_out=False,
                error=result.get("error", "Unknown error"),
                prediction=prediction,
            )
            
    except queue.Empty:
        # Timeout! Kill the process
        duration_ms = timeout_s * 1000
        
        logger.warning(
            f"Action {action_name} TIMED OUT after {timeout_s:.1f}s on "
            f"{mesh_chars.size_bin} mesh ({mesh_chars.face_count:,} faces)"
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
            predictor.record_duration(
                action_name, mesh_chars, duration_ms,
                success=False, model_id=model_id, pipeline_name=pipeline_name
            )
        
        return TimeoutActionResult(
            success=False,
            mesh=None,
            duration_ms=duration_ms,
            timed_out=True,
            error=f"Action timed out after {timeout_s:.1f}s",
            prediction=prediction,
        )
    
    finally:
        # Ensure process is cleaned up
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)


def get_recommended_timeout(action_name: str, mesh: trimesh.Trimesh) -> float:
    """Get recommended timeout for an action based on mesh characteristics.
    
    Args:
        action_name: Action name
        mesh: Input mesh
        
    Returns:
        Recommended timeout in seconds
    """
    if not PREDICTOR_AVAILABLE:
        return 60.0  # Default 60 seconds
    
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
    
    return prediction.timeout_ms / 1000


def should_skip_action(action_name: str, mesh: trimesh.Trimesh) -> Tuple[bool, Optional[str]]:
    """Check if an action should be skipped for this mesh due to hang risk.
    
    Args:
        action_name: Action name
        mesh: Input mesh
        
    Returns:
        Tuple of (should_skip, alternative_action_name)
    """
    if not PREDICTOR_AVAILABLE:
        return False, None
    
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
    
    return prediction.should_skip, prediction.alternative_action
