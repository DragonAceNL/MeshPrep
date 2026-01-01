# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""Resource path utilities for MeshPrep.

Handles resource paths correctly for both development and installed environments.
Resources are stored in the meshprep/data/ directory within the package.
"""

import sys
from pathlib import Path

# Package data directory (meshprep/data/)
_DATA_DIR = Path(__file__).parent / "data"


def get_resource_path(relative_path: str) -> Path:
    """Get absolute path to a resource.
    
    Args:
        relative_path: Path relative to the data directory 
                       (e.g., "images/MeshPrepLogo.svg")
    
    Returns:
        Absolute path to the resource.
    """
    # For PyInstaller bundles
    if hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS) / "meshprep" / "data" / relative_path
    
    return _DATA_DIR / relative_path


def get_logo_path() -> Path:
    """Get path to the MeshPrep logo."""
    return get_resource_path("images/MeshPrepLogo.svg")


def get_config_path(filename: str) -> Path:
    """Get path to a config file."""
    return get_resource_path(f"config/{filename}")
