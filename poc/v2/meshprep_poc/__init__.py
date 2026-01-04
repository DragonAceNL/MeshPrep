# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""MeshPrep POC v2 - Real mesh operations proof of concept."""

__version__ = "0.2.0"

from .mesh_ops import MeshDiagnostics, load_mesh, save_mesh, compute_diagnostics
from .validation import validate_repair
from .filter_script import FilterScript, FilterScriptRunner, get_preset, list_presets
from .report import (
    RepairReport,
    create_repair_report,
    generate_markdown_report,
    generate_json_report,
    generate_report_index,
    render_mesh_image,
)
from .fingerprint import (
    compute_file_fingerprint,
    compute_full_file_hash,
    is_valid_fingerprint,
)
from .reproducibility import (
    ReproducibilityLevel,
    EnvironmentSnapshot,
    CompatibilityMatrix,
    CompatibilityResult,
    capture_environment,
    check_compatibility,
    check_filter_script_compatibility,
    create_reproducibility_block,
    export_environment,
    import_environment,
    get_meshprep_version,
    get_package_version,
    print_environment_check,
)
from .quality_feedback import (
    QualityRating,
    QualityPrediction,
    PipelineQualityStats,
    QualityFeedbackEngine,
    get_quality_engine,
    reset_quality_engine,
)

__all__ = [
    # Mesh operations
    "MeshDiagnostics",
    "load_mesh",
    "save_mesh",
    "compute_diagnostics",
    "validate_repair",
    # Filter scripts
    "FilterScript",
    "FilterScriptRunner",
    "get_preset",
    "list_presets",
    # Reports
    "RepairReport",
    "create_repair_report",
    "generate_markdown_report",
    "generate_json_report",
    "generate_report_index",
    "render_mesh_image",
    # Fingerprinting
    "compute_file_fingerprint",
    "compute_full_file_hash",
    "is_valid_fingerprint",
    # Reproducibility
    "ReproducibilityLevel",
    "EnvironmentSnapshot",
    "CompatibilityMatrix",
    "CompatibilityResult",
    "capture_environment",
    "check_compatibility",
    "check_filter_script_compatibility",
    "create_reproducibility_block",
    "export_environment",
    "import_environment",
    "get_meshprep_version",
    "get_package_version",
    "print_environment_check",
    # Quality Feedback
    "QualityRating",
    "QualityPrediction",
    "PipelineQualityStats",
    "QualityFeedbackEngine",
    "get_quality_engine",
    "reset_quality_engine",
]
