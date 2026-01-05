# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Shared error logging utility for all action modules.

This module provides a unified way to log action failures to both:
1. Text log file (daily error logs)
2. SQLite database (for learning patterns)
"""

import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import trimesh

logger = logging.getLogger(__name__)

# Try to import error logging modules
_ERROR_LOGGING_AVAILABLE = False
_log_action_error = None
_get_failure_tracker = None
_MeshInfo = None
_categorize_error = None

try:
    from ..error_logger import log_action_error as _log_action_error
    from ..subprocess_executor import (
        get_failure_tracker as _get_failure_tracker,
        MeshInfo as _MeshInfo,
        categorize_error as _categorize_error,
    )
    _ERROR_LOGGING_AVAILABLE = True
except ImportError:
    pass


def log_action_failure(
    action_name: str,
    error_message: str,
    mesh: "trimesh.Trimesh",
    action_type: str = "unknown",
    filter_name: str = "",
) -> None:
    """Log an action failure to both text log and SQLite database.
    
    This is the unified error logging function that should be called
    from all action modules when an action fails.
    
    Args:
        action_name: Name of the action (e.g., 'pymeshfix_repair')
        error_message: Error message string
        mesh: The mesh that was being processed (for characteristics)
        action_type: Type of action ('pymeshlab', 'trimesh', 'blender', etc.)
        filter_name: Specific filter/function that failed (optional)
    """
    if not _ERROR_LOGGING_AVAILABLE:
        return
    
    try:
        # Get mesh characteristics
        face_count = len(mesh.faces) if hasattr(mesh, 'faces') else 0
        vertex_count = len(mesh.vertices) if hasattr(mesh, 'vertices') else 0
        
        try:
            body_count = len(mesh.split(only_watertight=False))
        except:
            body_count = 1
        
        # Categorize the error
        error_category = _categorize_error(error_message) if _categorize_error else "unknown"
        
        # Log to text file
        if _log_action_error:
            _log_action_error(
                action_name=action_name,
                error_message=error_message,
                error_category=error_category,
                failure_type="error",
                model_id="",  # Not available at action level
                model_fingerprint="",
                face_count=face_count,
                vertex_count=vertex_count,
                body_count=body_count,
                size_bin=_get_size_bin(face_count),
                pipeline_name="",
                attempt_number=0,
                extra_data={"action_type": action_type, "filter_name": filter_name} if filter_name else {"action_type": action_type},
            )
        
        # Log to SQLite database for learning
        if _get_failure_tracker and _MeshInfo:
            mesh_info = _MeshInfo(
                face_count=face_count,
                vertex_count=vertex_count,
                body_count=body_count,
                model_id="",
                model_fingerprint="",
            )
            
            tracker = _get_failure_tracker()
            tracker.record_failure(
                action_name=action_name,
                mesh_info=mesh_info,
                failure_type="error",
                error_message=error_message,
            )
            
    except Exception as e:
        logger.debug(f"Failed to log action error: {e}")


def _get_size_bin(face_count: int) -> str:
    """Get size category from face count."""
    if face_count < 10_000:
        return "tiny"
    elif face_count < 50_000:
        return "small"
    elif face_count < 100_000:
        return "medium"
    elif face_count < 500_000:
        return "large"
    return "huge"


def is_error_logging_available() -> bool:
    """Check if error logging is available."""
    return _ERROR_LOGGING_AVAILABLE
