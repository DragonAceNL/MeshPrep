# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Dedicated error/crash logging for action failures.

This module provides a separate log file for tracking all action failures,
crashes, and errors. This makes it easy to analyze failure patterns without
searching through the main batch processing log.

Log files are stored in: poc/v2/learning_data/error_logs/
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Log directory
ERROR_LOG_DIR = Path(__file__).parent.parent / "learning_data" / "error_logs"

# Ensure directory exists
ERROR_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Current log file (one per day)
_current_log_date: Optional[str] = None
_error_logger: Optional[logging.Logger] = None


def _get_error_logger() -> logging.Logger:
    """Get or create the error logger for today."""
    global _current_log_date, _error_logger
    
    today = datetime.now().strftime("%Y-%m-%d")
    
    if _current_log_date != today or _error_logger is None:
        _current_log_date = today
        
        # Create a dedicated logger
        _error_logger = logging.getLogger("meshprep.errors")
        _error_logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        _error_logger.handlers.clear()
        
        # Create file handler for today's log
        log_file = ERROR_LOG_DIR / f"errors_{today}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        
        # Format: timestamp | level | message
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        _error_logger.addHandler(file_handler)
        
        # Don't propagate to root logger
        _error_logger.propagate = False
    
    return _error_logger


def log_action_error(
    action_name: str,
    error_message: str,
    error_category: str,
    failure_type: str,
    model_id: str = "",
    model_fingerprint: str = "",
    face_count: int = 0,
    vertex_count: int = 0,
    body_count: int = 0,
    size_bin: str = "",
    pipeline_name: str = "",
    attempt_number: int = 0,
    extra_data: Optional[Dict[str, Any]] = None,
) -> None:
    """Log an action error/failure.
    
    Args:
        action_name: Name of the action that failed
        error_message: The error message
        error_category: Categorized error type (normals_required, memory, etc.)
        failure_type: Type of failure (crash, error, timeout, geometry_loss)
        model_id: Model ID being processed
        model_fingerprint: Model fingerprint
        face_count: Number of faces in mesh
        vertex_count: Number of vertices in mesh
        body_count: Number of bodies/components
        size_bin: Size category (tiny, small, medium, large, huge)
        pipeline_name: Name of the pipeline being executed
        attempt_number: Which attempt this was
        extra_data: Any additional data to log
    """
    logger = _get_error_logger()
    
    # Build log entry
    entry = {
        "action": action_name,
        "error": error_message[:500],  # Truncate long errors
        "category": error_category,
        "type": failure_type,
        "model_id": model_id,
        "fingerprint": model_fingerprint,
        "faces": face_count,
        "vertices": vertex_count,
        "bodies": body_count,
        "size": size_bin,
        "pipeline": pipeline_name,
        "attempt": attempt_number,
    }
    
    if extra_data:
        entry["extra"] = extra_data
    
    # Log as JSON for easy parsing
    log_line = json.dumps(entry, ensure_ascii=False)
    
    # Use appropriate log level based on failure type
    if failure_type == "crash":
        logger.critical(log_line)
    elif failure_type == "timeout":
        logger.error(log_line)
    else:
        logger.warning(log_line)


def log_crash(
    action_name: str,
    error_message: str,
    model_id: str = "",
    face_count: int = 0,
    exit_code: Optional[int] = None,
) -> None:
    """Log a process crash (convenience function).
    
    Args:
        action_name: Name of the action that crashed
        error_message: Crash details
        model_id: Model ID being processed
        face_count: Number of faces in mesh
        exit_code: Process exit code if available
    """
    logger = _get_error_logger()
    
    entry = {
        "type": "CRASH",
        "action": action_name,
        "model_id": model_id,
        "faces": face_count,
        "exit_code": exit_code,
        "error": error_message,
    }
    
    logger.critical(json.dumps(entry, ensure_ascii=False))


def log_pymeshlab_error(
    action_name: str,
    filter_name: str,
    error_message: str,
    model_id: str = "",
    face_count: int = 0,
) -> None:
    """Log a PyMeshLab filter error (convenience function).
    
    Args:
        action_name: Name of the action
        filter_name: PyMeshLab filter that failed
        error_message: Error details
        model_id: Model ID being processed
        face_count: Number of faces in mesh
    """
    logger = _get_error_logger()
    
    entry = {
        "type": "PYMESHLAB_ERROR",
        "action": action_name,
        "filter": filter_name,
        "model_id": model_id,
        "faces": face_count,
        "error": error_message[:500],
    }
    
    logger.warning(json.dumps(entry, ensure_ascii=False))


def get_error_log_path() -> Path:
    """Get the path to today's error log."""
    today = datetime.now().strftime("%Y-%m-%d")
    return ERROR_LOG_DIR / f"errors_{today}.log"


def get_all_error_logs() -> list:
    """Get list of all error log files."""
    return sorted(ERROR_LOG_DIR.glob("errors_*.log"), reverse=True)


def parse_error_log(log_path: Path) -> list:
    """Parse an error log file and return entries as dicts.
    
    Args:
        log_path: Path to the log file
        
    Returns:
        List of parsed log entries
    """
    entries = []
    
    if not log_path.exists():
        return entries
    
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            # Format: HH:MM:SS | LEVEL | JSON
            parts = line.strip().split(" | ", 2)
            if len(parts) >= 3:
                try:
                    timestamp = parts[0]
                    level = parts[1]
                    data = json.loads(parts[2])
                    data["_timestamp"] = timestamp
                    data["_level"] = level
                    entries.append(data)
                except json.JSONDecodeError:
                    pass  # Skip malformed lines
    
    return entries


def get_error_summary(log_path: Optional[Path] = None) -> Dict[str, Any]:
    """Get a summary of errors from a log file.
    
    Args:
        log_path: Path to log file (defaults to today's log)
        
    Returns:
        Summary dict with counts by category, action, etc.
    """
    if log_path is None:
        log_path = get_error_log_path()
    
    entries = parse_error_log(log_path)
    
    if not entries:
        return {"total": 0, "by_category": {}, "by_action": {}, "by_type": {}}
    
    by_category = {}
    by_action = {}
    by_type = {}
    
    for entry in entries:
        cat = entry.get("category", "unknown")
        action = entry.get("action", "unknown")
        ftype = entry.get("type", "unknown")
        
        by_category[cat] = by_category.get(cat, 0) + 1
        by_action[action] = by_action.get(action, 0) + 1
        by_type[ftype] = by_type.get(ftype, 0) + 1
    
    return {
        "total": len(entries),
        "by_category": dict(sorted(by_category.items(), key=lambda x: -x[1])),
        "by_action": dict(sorted(by_action.items(), key=lambda x: -x[1])),
        "by_type": dict(sorted(by_type.items(), key=lambda x: -x[1])),
    }
