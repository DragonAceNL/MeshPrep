# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""
MeshPrep v5 - Automated mesh repair system.

Zero-setup mesh repair with ML and learning.
"""

__version__ = "5.0.0"

# Bootstrap environment (auto-install dependencies)
from .core.bootstrap import ensure_environment

_env_ready = ensure_environment()

if not _env_ready:
    import sys
    print("\n⚠ Environment setup incomplete. Please install dependencies manually:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

# Core exports
from .core.mesh import Mesh
from .core.action import Action, ActionRegistry, ActionRiskLevel
from .core.pipeline import Pipeline
from .core.repair_engine import RepairEngine
from .core.validator import Validator

__all__ = [
    "Mesh",
    "Action",
    "ActionRegistry",
    "ActionRiskLevel",
    "Pipeline",
    "RepairEngine",
    "Validator",
]
