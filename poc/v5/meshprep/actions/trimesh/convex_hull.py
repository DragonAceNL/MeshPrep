# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Create convex hull of mesh."""

from typing import Dict, Any, Optional

from meshprep.core.action import Action, ActionRiskLevel, register_action
from meshprep.core.mesh import Mesh


@register_action
class ConvexHullAction(Action):
    """Create convex hull (wraps mesh in simplest printable shape)."""
    
    name = "convex_hull"
    description = "Create convex hull (extreme simplification)"
    risk_level = ActionRiskLevel.HIGH
    default_params = {}
    
    def execute(self, mesh: Mesh, params: Optional[Dict[str, Any]] = None) -> Mesh:
        """Create convex hull."""
        params = self.get_params(params)
        
        result = mesh.copy()
        
        try:
            # Create convex hull
            hull = result.trimesh.convex_hull
            
            result._mesh = hull
            result._update_metadata_from_mesh()
            
            self.logger.info(f"Created convex hull: {len(hull.faces)} faces "
                           f"(was {mesh.metadata.face_count})")
            
        except Exception as e:
            self.logger.error(f"Convex hull failed: {e}")
            raise
        
        return result
