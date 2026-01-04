# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Filter and repair info persistence for POC v3 batch processing.

Contains functions for saving filter scripts and detailed repair
information to JSON files for later analysis.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from config import FILTERS_PATH


def save_filter_script(
    file_id: str, 
    filter_script, 
    escalated: bool = False
) -> None:
    """Save the filter script used for a model (legacy format).
    
    Args:
        file_id: Model identifier (filename without extension)
        filter_script: FilterScript object that was used
        escalated: Whether Blender escalation was used
    """
    # Ensure filters directory exists
    FILTERS_PATH.mkdir(parents=True, exist_ok=True)
    
    filter_path = FILTERS_PATH / f"{file_id}.json"
    
    # Build filter data
    filter_data = {
        "model_id": file_id,
        "filter_name": filter_script.name,
        "filter_version": getattr(filter_script, 'version', '1.0.0'),
        "escalated_to_blender": escalated,
        "actions": [
            {
                "name": action.name,
                "params": action.params
            }
            for action in filter_script.actions
        ],
        "timestamp": datetime.now().isoformat()
    }
    
    with open(filter_path, "w", encoding="utf-8") as f:
        json.dump(filter_data, f, indent=2)


def save_filter_info(
    file_id: str,
    filter_used: str,
    escalated: bool,
    repair_result = None,
    model_fingerprint: str = "",
    original_filename: str = "",
    original_format: str = "",
    before_diagnostics: Optional[Dict[str, Any]] = None,
    after_diagnostics: Optional[Dict[str, Any]] = None,
) -> None:
    """Save detailed filter/repair info for a model.
    
    This saves comprehensive data for later analysis to:
    - Refine filter script selection
    - Improve model profile detection
    - Optimize repair pipeline performance
    
    The saved filter script includes the model fingerprint to enable
    community sharing and discovery. Search "MP:xxxxxxxxxxxx" on Reddit
    to find filter scripts for a specific model.
    
    Args:
        file_id: Model identifier (filename without extension)
        filter_used: Name of the filter/pipeline used
        escalated: Whether Blender escalation was used
        repair_result: SlicerRepairResult object (optional)
        model_fingerprint: Searchable fingerprint (MP:xxxx format)
        original_filename: Original filename with extension
        original_format: File format (stl, ctm, obj, etc.)
        before_diagnostics: Mesh diagnostics before repair
        after_diagnostics: Mesh diagnostics after repair
    """
    # Ensure filters directory exists
    FILTERS_PATH.mkdir(parents=True, exist_ok=True)
    
    filter_path = FILTERS_PATH / f"{file_id}.json"
    
    # Extract detailed attempt info for analysis
    attempts_detail = []
    if repair_result and repair_result.attempts:
        for attempt in repair_result.attempts:
            attempt_info = {
                "attempt_number": attempt.attempt_number,
                "pipeline_name": attempt.pipeline_name,
                "actions": attempt.pipeline_actions,
                "success": attempt.success,
                "duration_ms": attempt.duration_ms,
                "error": attempt.error,
                "geometry_valid": attempt.geometry_valid,
            }
            # Include slicer validation results if available
            if attempt.slicer_result:
                attempt_info["slicer_validation"] = {
                    "success": attempt.slicer_result.success,
                    "issues": attempt.slicer_result.issues,
                    "warnings": attempt.slicer_result.warnings[:5] if attempt.slicer_result.warnings else [],
                    "errors": attempt.slicer_result.errors[:5] if attempt.slicer_result.errors else [],
                }
            attempts_detail.append(attempt_info)
    
    # Extract precheck info for analysis
    precheck_info = None
    if repair_result and repair_result.precheck_mesh_info:
        precheck_info = {
            "manifold": repair_result.precheck_mesh_info.manifold,
            "open_edges": repair_result.precheck_mesh_info.open_edges,
            "reversed_facets": getattr(repair_result.precheck_mesh_info, 'reversed_facets', 0),
            "is_clean": repair_result.precheck_mesh_info.is_clean,
            "issues": repair_result.precheck_mesh_info.issues,
        }
    
    # Build comprehensive filter data for analysis
    filter_data = {
        # Model identification
        "model_id": file_id,
        "model_fingerprint": model_fingerprint,
        "original_filename": original_filename,
        "original_format": original_format,
        
        # Repair outcome
        "filter_name": filter_used,
        "success": repair_result.success if repair_result else False,
        "escalated_to_blender": escalated,
        
        # Precheck results (for analyzing which models need repair)
        "precheck": {
            "passed": repair_result.precheck_passed if repair_result else False,
            "skipped": repair_result.precheck_skipped if repair_result else False,
            "mesh_info": precheck_info,
        },
        
        # Detailed attempt history (for pipeline optimization)
        "repair_attempts": {
            "total_attempts": repair_result.total_attempts if repair_result else 0,
            "total_duration_ms": repair_result.total_duration_ms if repair_result else 0,
            "issues_found": repair_result.issues_found if repair_result else [],
            "issues_resolved": repair_result.issues_resolved if repair_result else [],
            "attempts": attempts_detail,
        },
        
        # Mesh diagnostics (for model profile analysis)
        "diagnostics": {
            "before": before_diagnostics,
            "after": after_diagnostics,
        },
        
        # Metadata
        "timestamp": datetime.now().isoformat(),
        "meshprep_version": "0.1.0",
        "meshprep_url": "https://github.com/DragonAceNL/MeshPrep",
        "sharing_note": f"Search '{model_fingerprint}' on Reddit to find/share filter scripts for this model",
    }
    
    with open(filter_path, "w", encoding="utf-8") as f:
        json.dump(filter_data, f, indent=2)
