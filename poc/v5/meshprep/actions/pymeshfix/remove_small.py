# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Remove smallest components using PyMeshFix."""

from typing import Dict, Any, Optional

from meshprep.core.action import Action, ActionRiskLevel, register_action
from meshprep.core.mesh import Mesh


@register_action
class PyMeshFixRemoveSmallAction(Action):
    """Remove smallest mesh components (fragments)."""
    
    name = "pymeshfix_remove_small"
    description = "Remove smallest components with PyMeshFix"
    risk_level = ActionRiskLevel.LOW
    default_params = {}
    
    def execute(self, mesh: Mesh, params: Optional[Dict[str, Any]] = None) -> Mesh:
        """Remove small components."""
        params = self.get_params(params)
        
        try:
            import pymeshfix
        except ImportError:
            raise RuntimeError("pymeshfix not installed")
        
        result = mesh.copy()
        
        try:
            meshfix = pymeshfix.MeshFix(
                result.trimesh.vertices,
                result.trimesh.faces
            )
            
            # Repair with remove smallest
            meshfix.repair(
                verbose=False,
                joincomp=False,
                remove_smallest_components=True
            )
            
            import trimesh
            result._mesh = trimesh.Trimesh(
                vertices=meshfix.v,
                faces=meshfix.f
            )
            
            result._update_metadata_from_mesh()
            
            self.logger.info("Removed smallest components")
            
        except Exception as e:
            self.logger.error(f"Remove small failed: {e}")
            raise
        
        return result
