# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Clean mesh using PyMeshFix."""

from typing import Dict, Any, Optional

from meshprep.core.action import Action, ActionRiskLevel, register_action
from meshprep.core.mesh import Mesh


@register_action
class PyMeshFixCleanAction(Action):
    """Clean mesh (remove unreferenced vertices, fix orientation)."""
    
    name = "pymeshfix_clean"
    description = "Clean mesh with PyMeshFix (lightweight)"
    risk_level = ActionRiskLevel.LOW
    default_params = {}
    
    def execute(self, mesh: Mesh, params: Optional[Dict[str, Any]] = None) -> Mesh:
        """Clean mesh."""
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
            
            # Light clean (no aggressive repair)
            meshfix.repair(
                verbose=False,
                joincomp=False,
                remove_smallest_components=False
            )
            
            # Update mesh
            import trimesh
            result._mesh = trimesh.Trimesh(
                vertices=meshfix.v,
                faces=meshfix.f
            )
            
            result._update_metadata_from_mesh()
            
            self.logger.info("PyMeshFix clean complete")
            
        except Exception as e:
            self.logger.error(f"PyMeshFix clean failed: {e}")
            raise
        
        return result
