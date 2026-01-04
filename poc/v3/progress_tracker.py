# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Progress tracking for MeshPrep batch processing.

Contains the Progress dataclass for tracking batch processing status,
along with functions for loading and saving progress to disk.
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class Progress:
    """Track overall progress of batch processing."""
    total_files: int = 0
    processed: int = 0
    successful: int = 0
    failed: int = 0
    escalations: int = 0
    skipped: int = 0
    precheck_skipped: int = 0  # Models skipped because already clean
    reconstructed: int = 0  # Models reconstructed (significant geometry change)
    
    start_time: str = ""
    last_update: str = ""
    current_file: str = ""
    current_action: str = ""  # Current action being executed
    current_step: int = 0  # Current step number (e.g., 1 of 4)
    total_steps: int = 0  # Total steps in current filter
    
    # Timing
    total_duration_ms: float = 0
    avg_duration_ms: float = 0
    
    # ETA
    eta_seconds: float = 0
    
    @property
    def percent_complete(self) -> float:
        """Calculate percentage of files processed."""
        if self.total_files == 0:
            return 0
        return (self.processed / self.total_files) * 100
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.processed == 0:
            return 0
        return (self.successful / self.processed) * 100


def load_progress(progress_file: Path) -> Progress:
    """Load progress from file.
    
    Args:
        progress_file: Path to the progress JSON file
        
    Returns:
        Progress object (empty if file doesn't exist or is invalid)
    """
    if progress_file.exists():
        try:
            with open(progress_file) as f:
                data = json.load(f)
                return Progress(**data)
        except Exception:
            pass
    return Progress()


def save_progress(progress: Progress, progress_file: Path) -> None:
    """Save progress to file.
    
    Args:
        progress: Progress object to save
        progress_file: Path to save the progress JSON file
    """
    progress.last_update = datetime.now().isoformat()
    with open(progress_file, "w") as f:
        json.dump(asdict(progress), f, indent=2)
