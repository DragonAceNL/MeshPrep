# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Simplify mesh using Open3D quadric decimation."""

from typing import Dict, Any, Optional
import numpy as np

from meshprep.core.action import Action, ActionRiskLevel, register_action
from meshprep.core.mesh import Mesh


@register_action
class Open3DSimplifyAction(Action):
    """Simplify mesh using Open3D (high-quality decimation)."""
    
    name = "open3d_simplify"
    description = "High-quality mesh simplification (Open3D)"
    risk_level = ActionRiskLevel.MEDIUM
    default_params = {"target_reduction": 0.5}
    
    def execute(self, mesh: Mesh, params: Optional[Dict[str, Any]] = None) -> Mesh:
        """Simplify mesh."""
        params = self.get_params(params)
        
        try:
            import open3d as o3d
        except ImportError:
            raise RuntimeError("Open3D not installed")
        
        try:
            # Convert to Open3D
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.trimesh.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.trimesh.faces)
            
            # Calculate target triangles
            current_triangles = len(mesh.trimesh.faces)
            target_triangles = int(current_triangles * params["target_reduction"])
            
            self.logger.info(f"Simplifying: {current_triangles} → {target_triangles} triangles")
            
            # Simplify
            simplified = o3d_mesh.simplify_quadric_decimation(target_triangles)
            
            # Convert back
            import trimesh
            vertices = np.asarray(simplified.vertices)
            faces = np.asarray(simplified.triangles)
            
            result_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            result = Mesh(result_mesh, mesh.metadata)
            result._update_metadata_from_mesh()
            
            self.logger.info(f"Simplified to {len(faces)} triangles")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Simplification failed: {e}")
            raise
