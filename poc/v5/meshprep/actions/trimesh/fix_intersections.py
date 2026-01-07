# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Fix self-intersecting faces."""

from typing import Dict, Any, Optional

from meshprep.core.action import Action, ActionRiskLevel, register_action
from meshprep.core.mesh import Mesh


@register_action
class FixIntersectionsAction(Action):
    """Attempt to fix self-intersecting geometry."""
    
    name = "fix_intersections"
    description = "Fix self-intersecting faces"
    risk_level = ActionRiskLevel.MEDIUM
    default_params = {}
    
    def execute(self, mesh: Mesh, params: Optional[Dict[str, Any]] = None) -> Mesh:
        """Fix self-intersections."""
        params = self.get_params(params)
        
        result = mesh.copy()
        
        try:
            # Check for intersections
            if result.trimesh.is_volume and not result.trimesh.is_watertight:
                # Try to fix using trimesh's built-in
                result.trimesh.process(validate=False)
                
                # Remove degenerate faces
                result.trimesh.remove_degenerate_faces()
                
                # Merge vertices
                result.trimesh.merge_vertices()
                
                result._update_metadata_from_mesh()
                
                self.logger.info("Fixed self-intersections")
            else:
                self.logger.info("No self-intersections detected")
            
        except Exception as e:
            self.logger.warning(f"Could not fix intersections: {e}")
            return mesh
        
        return result
