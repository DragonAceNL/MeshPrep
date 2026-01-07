# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""PyMeshFix repair action for robust mesh fixing."""

from typing import Dict, Any, Optional
import tempfile
from pathlib import Path

from meshprep.core.action import Action, ActionRiskLevel, register_action
from meshprep.core.mesh import Mesh


@register_action
class PyMeshFixRepairAction(Action):
    """Robust mesh repair using PyMeshFix."""
    
    name = "pymeshfix_repair"
    description = "Repair mesh using PyMeshFix (fills holes, fixes non-manifold)"
    risk_level = ActionRiskLevel.MEDIUM
    default_params = {
        "verbose": False,
        "joincomp": False,  # Join components
        "remove_smallest_components": False,
    }
    
    def execute(self, mesh: Mesh, params: Optional[Dict[str, Any]] = None) -> Mesh:
        """Repair mesh using PyMeshFix."""
        params = self.get_params(params)
        
        try:
            import pymeshfix
        except ImportError:
            raise RuntimeError("pymeshfix not installed. Install with: pip install pymeshfix")
        
        # Work on a copy
        result = mesh.copy()
        
        try:
            # Create meshfix wrapper
            meshfix = pymeshfix.MeshFix(
                result.trimesh.vertices,
                result.trimesh.faces
            )
            
            # Repair
            self.logger.info("Running PyMeshFix repair...")
            meshfix.repair(
                verbose=params["verbose"],
                joincomp=params["joincomp"],
                remove_smallest_components=params["remove_smallest_components"]
            )
            
            # Get repaired mesh
            vertices, faces = meshfix.v, meshfix.f
            
            # Update trimesh object
            import trimesh
            result._mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # Update metadata
            result._update_metadata_from_mesh()
            
            self.logger.info(f"PyMeshFix repair complete: "
                           f"{len(faces)} faces (was {mesh.metadata.face_count})")
            
        except Exception as e:
            self.logger.error(f"PyMeshFix repair failed: {e}")
            raise
        
        return result
