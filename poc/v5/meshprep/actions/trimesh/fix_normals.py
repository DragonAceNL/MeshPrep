# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Fix face normals action using trimesh."""

from typing import Dict, Any, Optional
import numpy as np

from meshprep.core.action import Action, ActionRiskLevel, register_action
from meshprep.core.mesh import Mesh


@register_action
class FixNormalsAction(Action):
    """Fix face normal directions to ensure consistent winding."""
    
    name = "fix_normals"
    description = "Fix face normal directions"
    risk_level = ActionRiskLevel.LOW
    default_params = {}
    
    def execute(self, mesh: Mesh, params: Optional[Dict[str, Any]] = None) -> Mesh:
        """Fix face normals."""
        params = self.get_params(params)
        
        # Work on a copy
        result = mesh.copy()
        
        try:
            # Fix normals using trimesh
            result.trimesh.fix_normals()
            
            # Update metadata
            result._update_metadata_from_mesh()
            
            self.logger.info("Fixed face normals")
            
        except Exception as e:
            self.logger.warning(f"Could not fix normals: {e}")
            # Return original if fix fails
            return mesh
        
        return result
