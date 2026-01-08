# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Make mesh watertight using trimesh."""

from typing import Dict, Any, Optional

from meshprep.core.action import Action, ActionRiskLevel, register_action
from meshprep.core.mesh import Mesh


@register_action
class MakeWatertightAction(Action):
    """Make mesh watertight (closes all gaps)."""
    
    name = "make_watertight"
    description = "Make mesh watertight (aggressive hole filling)"
    risk_level = ActionRiskLevel.MEDIUM
    default_params = {}
    
    def execute(self, mesh: Mesh, params: Optional[Dict[str, Any]] = None) -> Mesh:
        """Make mesh watertight."""
        params = self.get_params(params)
        
        result = mesh.copy()
        
        if result.metadata.is_watertight:
            self.logger.info("Mesh already watertight")
            return result
        
        try:
            # Fill all holes
            import trimesh.repair
            trimesh.repair.fill_holes(result.trimesh)
            
            # Attempt to make watertight
            result.trimesh.fill_holes()
            
            # Fix normals
            result.trimesh.fix_normals()
            
            result._update_metadata_from_mesh()
            
            if result.metadata.is_watertight:
                self.logger.info("Successfully made watertight")
            else:
                self.logger.warning("Could not make fully watertight")
            
        except Exception as e:
            self.logger.error(f"Make watertight failed: {e}")
            raise
        
        return result
