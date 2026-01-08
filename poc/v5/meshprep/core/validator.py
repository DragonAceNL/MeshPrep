# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Validation logic for mesh repair quality."""

from dataclasses import dataclass
from typing import Optional
import logging
import numpy as np

from .mesh import Mesh

logger = logging.getLogger(__name__)


@dataclass
class GeometricValidation:
    """Result of geometric validation checks."""
    is_watertight: bool = False
    is_manifold: bool = False
    has_positive_volume: bool = False
    volume: float = 0.0
    issues: list = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []
    
    @property
    def is_printable(self) -> bool:
        """Check if mesh is printable."""
        return self.is_watertight and self.is_manifold and self.has_positive_volume
    
    @property
    def quality_score(self) -> float:
        """Simple quality score (1-5) based on geometric properties."""
        score = 1.0
        if self.is_watertight:
            score += 1.5
        if self.is_manifold:
            score += 1.5
        if self.has_positive_volume:
            score += 1.0
        return min(5.0, score)


@dataclass
class FidelityValidation:
    """Result of fidelity validation checks."""
    volume_change_pct: float = 0.0
    face_count_change_pct: float = 0.0
    hausdorff_distance: float = 0.0
    volume_acceptable: bool = True
    changes: list = None
    
    def __post_init__(self):
        if self.changes is None:
            self.changes = []


@dataclass
class ValidationResult:
    """Complete validation result."""
    geometric: GeometricValidation
    fidelity: Optional[FidelityValidation] = None
    quality_score: Optional[int] = None
    
    @property
    def is_successful(self) -> bool:
        """Check if repair is successful."""
        return self.geometric.is_printable


class Validator:
    """Validates mesh repair quality."""
    
    def validate_geometry(self, mesh: Mesh) -> GeometricValidation:
        """Run geometric validation checks."""
        result = GeometricValidation()
        
        try:
            result.is_watertight = mesh.metadata.is_watertight
            result.is_manifold = mesh.metadata.is_manifold
            result.volume = mesh.metadata.volume
            result.has_positive_volume = result.volume > 0
            
            if not result.is_watertight:
                result.issues.append("Not watertight")
            if not result.is_manifold:
                result.issues.append("Non-manifold geometry")
            if not result.has_positive_volume:
                result.issues.append(f"Invalid volume ({result.volume:.4f})")
                
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            result.issues.append(f"Validation error: {e}")
        
        return result
    
    def validate_fidelity(self, original: Mesh, repaired: Mesh, 
                         volume_threshold: float = 30.0) -> FidelityValidation:
        """Validate fidelity between original and repaired mesh."""
        result = FidelityValidation()
        
        try:
            # Volume change
            orig_vol = original.metadata.volume
            rep_vol = repaired.metadata.volume
            if orig_vol > 0:
                result.volume_change_pct = ((rep_vol - orig_vol) / orig_vol) * 100
                result.volume_acceptable = abs(result.volume_change_pct) <= volume_threshold
                
                if not result.volume_acceptable:
                    result.changes.append(f"Volume changed by {result.volume_change_pct:.2f}%")
            
            # Face count change
            orig_faces = original.metadata.face_count
            rep_faces = repaired.metadata.face_count
            if orig_faces > 0:
                result.face_count_change_pct = ((rep_faces - orig_faces) / orig_faces) * 100
                
        except Exception as e:
            logger.warning(f"Fidelity validation failed: {e}")
        
        return result
    
    def compute_quality_score(self, original: Mesh, repaired: Mesh) -> int:
        """Compute auto-quality score (1-5)."""
        fidelity = self.validate_fidelity(original, repaired)
        geometric = self.validate_geometry(repaired)
        
        score = 5.0
        
        # Volume change penalties
        vol_change = abs(fidelity.volume_change_pct)
        if vol_change > 30:
            score -= 2.0
        elif vol_change > 15:
            score -= 1.0
        elif vol_change > 5:
            score -= 0.5
        
        # Geometric validity bonus/penalty
        if geometric.is_printable:
            score += 0.5
        else:
            score -= 0.5
        
        return max(1, min(5, round(score)))
    
    def validate(self, original: Mesh, repaired: Mesh) -> ValidationResult:
        """Complete validation."""
        geometric = self.validate_geometry(repaired)
        fidelity = self.validate_fidelity(original, repaired)
        quality = self.compute_quality_score(original, repaired)
        
        return ValidationResult(
            geometric=geometric,
            fidelity=fidelity,
            quality_score=quality,
        )
