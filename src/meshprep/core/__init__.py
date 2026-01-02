# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Core logic for mesh processing, actions, profiles, and validation.

This module provides the fundamental building blocks for mesh repair:
- mesh_ops: Load, save, and analyze meshes
- validation: Geometric and fidelity validation
- filter_script: Filter script loading and execution
- actions: Action registry and implementations
"""

from .mesh_ops import (
    MeshDiagnostics,
    load_mesh,
    save_mesh,
    compute_diagnostics,
    compute_fingerprint,
    format_diagnostics,
    print_diagnostics,
)

from .validation import (
    GeometricValidation,
    FidelityValidation,
    ValidationResult,
    validate_geometry,
    validate_fidelity,
    validate_repair,
    format_validation_result,
    print_validation_result,
)

from .filter_script import (
    FilterAction,
    FilterScript,
    FilterScriptMeta,
    FilterScriptRunner,
    FilterScriptResult,
    ActionResult,
    create_filter_script,
    get_preset,
    list_presets,
    load_presets_from_directory,
    PRESETS,
)

from .actions import ActionRegistry

__all__ = [
    # Mesh operations
    "MeshDiagnostics",
    "load_mesh",
    "save_mesh",
    "compute_diagnostics",
    "compute_fingerprint",
    "format_diagnostics",
    "print_diagnostics",
    # Validation
    "GeometricValidation",
    "FidelityValidation", 
    "ValidationResult",
    "validate_geometry",
    "validate_fidelity",
    "validate_repair",
    "format_validation_result",
    "print_validation_result",
    # Filter scripts
    "FilterAction",
    "FilterScript",
    "FilterScriptMeta",
    "FilterScriptRunner",
    "FilterScriptResult",
    "ActionResult",
    "create_filter_script",
    "get_preset",
    "list_presets",
    "load_presets_from_directory",
    "PRESETS",
    # Actions
    "ActionRegistry",
]
