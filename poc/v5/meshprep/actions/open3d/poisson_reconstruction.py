# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Poisson surface reconstruction using Open3D."""

from typing import Dict, Any, Optional
import numpy as np

from meshprep.core.action import Action, ActionRiskLevel, register_action
from meshprep.core.mesh import Mesh


@register_action
class PoissonReconstructionAction(Action):
    """Reconstruct surface using screened Poisson."""
    
    name = "poisson_reconstruction"
    description = "Poisson surface reconstruction (for point clouds/broken surfaces)"
    risk_level = ActionRiskLevel.HIGH
    default_params = {"depth": 9, "width": 0, "scale": 1.1}
    
    def execute(self, mesh: Mesh, params: Optional[Dict[str, Any]] = None) -> Mesh:
        """Reconstruct surface with Poisson."""
        params = self.get_params(params)
        
        try:
            import open3d as o3d
        except ImportError:
            raise RuntimeError("Open3D not installed. Install with: pip install open3d")
        
        try:
            # Convert to Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(mesh.trimesh.vertices)
            
            # Estimate normals if not present
            pcd.estimate_normals()
            pcd.orient_normals_consistent_tangent_plane(k=30)
            
            # Poisson reconstruction
            self.logger.info(f"Running Poisson reconstruction (depth={params['depth']})...")
            
            o3d_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd,
                depth=params["depth"],
                width=params["width"],
                scale=params["scale"],
            )
            
            # Convert back to trimesh
            import trimesh
            vertices = np.asarray(o3d_mesh.vertices)
            faces = np.asarray(o3d_mesh.triangles)
            
            result_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            result = Mesh(result_mesh, mesh.metadata)
            result._update_metadata_from_mesh()
            
            self.logger.info(f"Poisson reconstruction complete: {len(faces)} faces")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Poisson reconstruction failed: {e}")
            raise
