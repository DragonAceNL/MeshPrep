# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Subdivide mesh to add detail."""

from typing import Dict, Any, Optional

from meshprep.core.action import Action, ActionRiskLevel, register_action
from meshprep.core.mesh import Mesh


@register_action
class SubdivideAction(Action):
    """Subdivide mesh faces to add detail."""
    
    name = "subdivide"
    description = "Subdivide mesh (increases face count)"
    risk_level = ActionRiskLevel.LOW
    default_params = {"iterations": 1}
    
    def execute(self, mesh: Mesh, params: Optional[Dict[str, Any]] = None) -> Mesh:
        """Subdivide mesh."""
        params = self.get_params(params)
        iterations = params["iterations"]
        
        result = mesh.copy()
        
        try:
            original_faces = len(result.trimesh.faces)
            
            # Subdivide using trimesh
            result.trimesh = result.trimesh.subdivide(iterations=iterations)
            
            result._update_metadata_from_mesh()
            
            new_faces = len(result.trimesh.faces)
            self.logger.info(f"Subdivided mesh: {original_faces} → {new_faces} faces "
                           f"({iterations} iterations)")
            
        except Exception as e:
            self.logger.error(f"Subdivision failed: {e}")
            raise
        
        return result
