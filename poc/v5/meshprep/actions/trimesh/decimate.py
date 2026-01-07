# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Decimate mesh to reduce face count."""

from typing import Dict, Any, Optional

from meshprep.core.action import Action, ActionRiskLevel, register_action
from meshprep.core.mesh import Mesh


@register_action
class DecimateAction(Action):
    """Reduce mesh face count while preserving shape."""
    
    name = "decimate"
    description = "Reduce face count (decimation)"
    risk_level = ActionRiskLevel.MEDIUM
    default_params = {"face_count": 10000}
    
    def execute(self, mesh: Mesh, params: Optional[Dict[str, Any]] = None) -> Mesh:
        """Decimate mesh to target face count."""
        params = self.get_params(params)
        target_faces = params["face_count"]
        
        result = mesh.copy()
        
        current_faces = len(result.trimesh.faces)
        if current_faces <= target_faces:
            self.logger.info(f"Mesh already has {current_faces} faces (target: {target_faces})")
            return result
        
        try:
            # Calculate face ratio
            face_ratio = target_faces / current_faces
            
            # Decimate
            result.trimesh = result.trimesh.simplify_quadric_decimation(target_faces)
            
            result._update_metadata_from_mesh()
            
            self.logger.info(f"Decimated from {current_faces} to {len(result.trimesh.faces)} faces")
            
        except Exception as e:
            self.logger.error(f"Decimation failed: {e}")
            raise
        
        return result
