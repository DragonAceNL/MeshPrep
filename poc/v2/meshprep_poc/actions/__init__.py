# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Action implementations for mesh repair.

Each action is a function that takes a mesh and parameters,
and returns a modified mesh.
"""

from .registry import ActionRegistry, register_action
from .trimesh_actions import *
from .pymeshfix_actions import *

__all__ = ["ActionRegistry", "register_action"]
