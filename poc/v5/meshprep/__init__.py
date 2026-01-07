# MeshPrep v5 - Production-Ready Mesh Repair System
# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0

"""
MeshPrep v5: Clean, Production-Ready Architecture
==================================================

Usage:
    from meshprep import RepairEngine
    
    engine = RepairEngine()
    result = engine.repair("broken_model.stl")
"""

__version__ = "5.0.0"
__author__ = "Allard Peper (Dragon Ace)"

from .core.mesh import Mesh
from .core.action import Action, ActionRegistry, ActionResult
from .core.pipeline import Pipeline, PipelineResult

__all__ = [
    "Mesh",
    "Action",
    "ActionRegistry", 
    "ActionResult",
    "Pipeline",
    "PipelineResult",
]
