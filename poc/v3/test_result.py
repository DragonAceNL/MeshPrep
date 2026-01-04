# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Test result data class for MeshPrep batch processing.

Contains the TestResult dataclass that holds all metrics and status
for a single model repair operation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class TestResult:
    """Result of a single model test."""
    file_id: str
    file_path: str
    success: bool = False
    error: Optional[str] = None
    filter_used: str = ""
    escalation_used: bool = False
    duration_ms: float = 0
    
    # Model fingerprint for filter script discovery
    model_fingerprint: str = ""  # Searchable fingerprint: MP:xxxxxxxxxxxx
    original_file_hash: str = ""  # Full SHA256 for exact matching
    
    # Pre-check result (slicer --info before repair)
    precheck_passed: bool = False  # True if model was already clean
    precheck_skipped: bool = False  # True if we skipped repair due to precheck
    
    # Reconstruction mode (for extreme-fragmented meshes)
    is_reconstruction: bool = False  # True if mesh was reconstructed (not repaired)
    reconstruction_method: str = ""  # Pipeline that performed reconstruction
    geometry_loss_pct: float = 0  # Face loss percentage
    
    # Original metrics
    original_vertices: int = 0
    original_faces: int = 0
    original_volume: float = 0
    original_watertight: bool = False
    original_manifold: bool = False
    
    # Result metrics
    result_vertices: int = 0
    result_faces: int = 0
    result_volume: float = 0
    result_watertight: bool = False
    result_manifold: bool = False
    
    # Geometry change
    volume_change_pct: float = 0
    face_change_pct: float = 0
    
    # Additional diagnostics
    original_components: int = 0
    original_holes: int = 0
    result_components: int = 0
    result_holes: int = 0
    
    # File sizes (bytes)
    original_file_size: int = 0
    fixed_file_size: int = 0
    
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
