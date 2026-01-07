# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Keep only the largest mesh component."""

from typing import Dict, Any, Optional

from meshprep.core.action import Action, ActionRiskLevel, register_action
from meshprep.core.mesh import Mesh


@register_action
class KeepLargestAction(Action):
    """Keep only the largest mesh component."""
    
    name = "keep_largest"
    description = "Keep only largest component (removes small fragments)"
    risk_level = ActionRiskLevel.MEDIUM
    default_params = {}
    
    def execute(self, mesh: Mesh, params: Optional[Dict[str, Any]] = None) -> Mesh:
        """Keep largest component."""
        params = self.get_params(params)
        
        result = mesh.copy()
        
        try:
            # Split into components
            components = result.trimesh.split(only_watertight=False)
            
            if len(components) <= 1:
                self.logger.info("Only 1 component, nothing to remove")
                return result
            
            # Find largest by face count
            largest = max(components, key=lambda c: len(c.faces))
            
            result._mesh = largest
            result._update_metadata_from_mesh()
            
            removed = len(components) - 1
            self.logger.info(f"Kept largest component, removed {removed} fragments")
            
        except Exception as e:
            self.logger.error(f"Keep largest failed: {e}")
            raise
        
        return result
