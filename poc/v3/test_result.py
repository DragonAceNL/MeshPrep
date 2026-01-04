# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Test result data class for POC v3 batch processing.

Contains the TestResult dataclass that captures all metrics
from processing a single mesh file.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class TestResult:
    """Result of processing a single model."""
    
    # Identification
    file_id: str
    file_path: str = ""
    model_fingerprint: str = ""
    original_file_hash: str = ""
    
    # Status
    success: bool = False
    filter_used: str = ""
    escalation_used: bool = False
    error: Optional[str] = None
    
    # Pre-check
    precheck_passed: bool = False
    precheck_skipped: bool = False
    
    # Reconstruction (for extreme-fragmented meshes)
    is_reconstruction: bool = False
    reconstruction_method: str = ""
    geometry_loss_pct: float = 0.0
    
    # Original mesh metrics
    original_vertices: int = 0
    original_faces: int = 0
    original_volume: float = 0.0
    original_watertight: bool = False
    original_manifold: bool = False
    original_components: int = 1
    original_holes: int = 0
    original_file_size: int = 0
    
    # Result mesh metrics
    result_vertices: int = 0
    result_faces: int = 0
    result_volume: float = 0.0
    result_watertight: bool = False
    result_manifold: bool = False
    result_components: int = 1
    result_holes: int = 0
    fixed_file_size: int = 0
    
    # Change metrics
    volume_change_pct: float = 0.0
    face_change_pct: float = 0.0
    
    # Timing
    duration_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "file_id": self.file_id,
            "file_path": self.file_path,
            "model_fingerprint": self.model_fingerprint,
            "original_file_hash": self.original_file_hash,
            "success": self.success,
            "filter_used": self.filter_used,
            "escalation_used": self.escalation_used,
            "error": self.error,
            "precheck_passed": self.precheck_passed,
            "precheck_skipped": self.precheck_skipped,
            "is_reconstruction": self.is_reconstruction,
            "reconstruction_method": self.reconstruction_method,
            "geometry_loss_pct": self.geometry_loss_pct,
            "original_vertices": self.original_vertices,
            "original_faces": self.original_faces,
            "original_volume": self.original_volume,
            "original_watertight": self.original_watertight,
            "original_manifold": self.original_manifold,
            "original_components": self.original_components,
            "original_holes": self.original_holes,
            "original_file_size": self.original_file_size,
            "result_vertices": self.result_vertices,
            "result_faces": self.result_faces,
            "result_volume": self.result_volume,
            "result_watertight": self.result_watertight,
            "result_manifold": self.result_manifold,
            "result_components": self.result_components,
            "result_holes": self.result_holes,
            "fixed_file_size": self.fixed_file_size,
            "volume_change_pct": self.volume_change_pct,
            "face_change_pct": self.face_change_pct,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
        }
