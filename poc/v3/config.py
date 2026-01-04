# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Configuration and paths for POC v3 batch processing.

Centralizes all path definitions and constants used across the batch
processing modules.
"""

from pathlib import Path

# =============================================================================
# Paths
# =============================================================================

# Input paths
THINGI10K_PATH = Path(r"C:\Users\Dragon Ace\Source\repos\Thingi10K\raw_meshes")
CTM_MESHES_PATH = Path(r"C:\Users\Dragon Ace\Source\repos\Thingi10K\ctm_meshes")

# Output base path (shared for STL and CTM)
OUTPUT_BASE_PATH = Path(r"C:\Users\Dragon Ace\Source\repos\Thingi10K")
REPORTS_PATH = OUTPUT_BASE_PATH / "reports"
FILTERS_PATH = OUTPUT_BASE_PATH / "reports" / "filters"
FIXED_OUTPUT_PATH = OUTPUT_BASE_PATH / "fixed"

# POC v3 local paths
POC_V3_PATH = Path(__file__).parent
PROGRESS_FILE = POC_V3_PATH / "progress.json"
SUMMARY_FILE = POC_V3_PATH / "summary.json"
RESULTS_CSV = POC_V3_PATH / "results.csv"
DASHBOARD_FILE = POC_V3_PATH / "dashboard.html"
LOG_DIR = POC_V3_PATH / "logs"

# =============================================================================
# Constants
# =============================================================================

# Supported mesh formats for processing
SUPPORTED_FORMATS = {".stl", ".ctm", ".obj", ".ply", ".3mf", ".off"}

# Default thresholds (used when adaptive thresholds not available)
DEFAULT_VOLUME_LOSS_LIMIT_PCT = 30.0
DEFAULT_FACE_LOSS_LIMIT_PCT = 40.0
DEFAULT_DECIMATION_TARGET_FACES = 100000
DEFAULT_DECIMATION_TRIGGER_FACES = 100000

# Batch processing settings
DEFAULT_MAX_REPAIR_ATTEMPTS = 20
DEFAULT_REPAIR_TIMEOUT = 120
DASHBOARD_UPDATE_INTERVAL = 10  # Update dashboard every N files
THRESHOLD_OPTIMIZE_INTERVAL = 100  # Optimize thresholds every N models


# =============================================================================
# Directory initialization
# =============================================================================

def ensure_directories_exist():
    """Create all required output directories if they don't exist."""
    REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    FILTERS_PATH.mkdir(parents=True, exist_ok=True)
    FIXED_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)


# Auto-create directories on import
ensure_directories_exist()
