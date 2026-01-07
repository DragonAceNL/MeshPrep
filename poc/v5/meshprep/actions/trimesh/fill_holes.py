# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Fill holes action using trimesh."""

from typing import Dict, Any, Optional
import trimesh.repair

from meshprep.core.action import Action, ActionRiskLevel, register_action
from meshprep.core.mesh import Mesh


@register_action
class FillHolesAction(Action):
    """Fill holes in a mesh using trimesh."""
    
    name = "fill_holes"
    description = "Fill holes in the mesh"
    risk_level = ActionRiskLevel.MEDIUM
    default_params = {"max_hole_size": 1000}
    
    def execute(self, mesh: Mesh, params: Optional[Dict[str, Any]] = None) -> Mesh:
        """Fill holes in the mesh."""
        params = self.get_params(params)
        
        # Work on a copy
        result = mesh.copy()
        
        # Fill holes using trimesh
        trimesh.repair.fill_holes(result.trimesh)
        
        # Update metadata
        result._update_metadata_from_mesh()
        
        return result
