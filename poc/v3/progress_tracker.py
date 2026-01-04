# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Progress tracking for MeshPrep batch processing.

This module provides backward-compatible functions that now use SQLite
via progress_db.py instead of JSON files.
"""

from pathlib import Path
from typing import Optional

# Re-export Progress class from progress_db for backward compatibility
from progress_db import Progress, get_progress_db, ModelResult


def load_progress(progress_file: Optional[Path] = None) -> Progress:
    """Load progress from database.
    
    Args:
        progress_file: Ignored (kept for backward compatibility)
        
    Returns:
        Progress object from database
    """
    db = get_progress_db()
    return db.get_progress()


def save_progress(progress: Progress, progress_file: Optional[Path] = None) -> None:
    """Save progress to database.
    
    Args:
        progress: Progress object to save
        progress_file: Ignored (kept for backward compatibility)
    """
    db = get_progress_db()
    db.save_progress(progress)


def reset_progress() -> None:
    """Reset progress for a new batch run."""
    db = get_progress_db()
    db.reset_progress()


# Re-export for convenience
__all__ = [
    "Progress",
    "ModelResult", 
    "load_progress",
    "save_progress",
    "reset_progress",
    "get_progress_db",
]
