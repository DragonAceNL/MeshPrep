# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Ball pivoting algorithm for mesh reconstruction."""

from typing import Dict, Any, Optional
import numpy as np

from meshprep.core.action import Action, ActionRiskLevel, register_action
from meshprep.core.mesh import Mesh


@register_action
class BallPivotAction(Action):
    """Reconstruct mesh from point cloud using ball pivoting."""
    
    name = "ball_pivot"
    description = "Ball pivoting reconstruction (for point clouds)"
    risk_level = ActionRiskLevel.HIGH
    default_params = {"radii": [0.005, 0.01, 0.02, 0.04]}
    
    def execute(self, mesh: Mesh, params: Optional[Dict[str, Any]] = None) -> Mesh:
        """Reconstruct with ball pivoting."""
        params = self.get_params(params)
        
        try:
            import open3d as o3d
        except ImportError:
            raise RuntimeError("Open3D not installed")
        
        try:
            # Convert to point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(mesh.trimesh.vertices)
            
            # Estimate normals
            pcd.estimate_normals()
            pcd.orient_normals_consistent_tangent_plane(k=30)
            
            # Ball pivoting
            radii = params["radii"]
            self.logger.info(f"Ball pivoting with radii: {radii}")
            
            o3d_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd,
                o3d.utility.DoubleVector(radii)
            )
            
            # Convert back to trimesh
            import trimesh
            vertices = np.asarray(o3d_mesh.vertices)
            faces = np.asarray(o3d_mesh.triangles)
            
            if len(faces) == 0:
                raise RuntimeError("Ball pivoting produced no faces")
            
            result_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            result = Mesh(result_mesh, mesh.metadata)
            result._update_metadata_from_mesh()
            
            self.logger.info(f"Ball pivoting complete: {len(faces)} faces")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ball pivoting failed: {e}")
            raise
