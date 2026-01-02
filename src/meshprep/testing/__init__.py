# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""Testing utilities for MeshPrep."""

from .thingi10k_manager import (
    Thingi10KDatabase,
    Thingi10KManager,
    ModelMetadata,
    categorize_model,
    map_to_meshprep_profile,
    PERMISSIVE_LICENSES,
    RESTRICTIVE_LICENSES,
)

__all__ = [
    "Thingi10KDatabase",
    "Thingi10KManager", 
    "ModelMetadata",
    "categorize_model",
    "map_to_meshprep_profile",
    "PERMISSIVE_LICENSES",
    "RESTRICTIVE_LICENSES",
]
