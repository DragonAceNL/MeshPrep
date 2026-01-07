# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Remove duplicate vertices."""

from typing import Dict, Any, Optional

from meshprep.core.action import Action, ActionRiskLevel, register_action
from meshprep.core.mesh import Mesh


@register_action
class RemoveDuplicatesAction(Action):
    """Remove duplicate vertices and degenerate faces."""
    
    name = "remove_duplicates"
    description = "Remove duplicate vertices and degenerate faces"
    risk_level = ActionRiskLevel.LOW
    default_params = {}
    
    def execute(self, mesh: Mesh, params: Optional[Dict[str, Any]] = None) -> Mesh:
        """Remove duplicates."""
        params = self.get_params(params)
        
        result = mesh.copy()
        
        try:
            original_vertices = len(result.trimesh.vertices)
            original_faces = len(result.trimesh.faces)
            
            # Merge duplicate vertices
            result.trimesh.merge_vertices()
            
            # Remove degenerate faces
            result.trimesh.remove_degenerate_faces()
            
            result._update_metadata_from_mesh()
            
            removed_vertices = original_vertices - len(result.trimesh.vertices)
            removed_faces = original_faces - len(result.trimesh.faces)
            
            self.logger.info(f"Removed {removed_vertices} duplicate vertices, "
                           f"{removed_faces} degenerate faces")
            
        except Exception as e:
            self.logger.warning(f"Could not remove duplicates: {e}")
            return mesh
        
        return result
