# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""MeshPrep POC v2 - Real mesh operations proof of concept."""

__version__ = "0.2.1"

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

__all__ = [
    "MeshDiagnostics",
    "load_mesh",
    "save_mesh",
    "compute_diagnostics",
    "validate_repair",
    "FilterScript",
    "FilterScriptRunner",
    "get_preset",
    "list_presets",
    "RepairReport",
    "create_repair_report",
    "generate_markdown_report",
    "generate_json_report",
    "generate_report_index",
    "render_mesh_image",
]
