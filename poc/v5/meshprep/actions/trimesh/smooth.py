# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Smooth mesh using Laplacian smoothing."""

from typing import Dict, Any, Optional
import numpy as np

from meshprep.core.action import Action, ActionRiskLevel, register_action
from meshprep.core.mesh import Mesh


@register_action
class SmoothAction(Action):
    """Apply Laplacian smoothing to mesh surface."""
    
    name = "smooth"
    description = "Smooth mesh surface (reduces noise)"
    risk_level = ActionRiskLevel.MEDIUM
    default_params = {"iterations": 3}
    
    def execute(self, mesh: Mesh, params: Optional[Dict[str, Any]] = None) -> Mesh:
        """Smooth mesh surface."""
        params = self.get_params(params)
        iterations = params["iterations"]
        
        result = mesh.copy()
        
        try:
            # Trimesh doesn't have built-in smoothing, so we do simple Laplacian
            vertices = result.trimesh.vertices.copy()
            
            for _ in range(iterations):
                # Build adjacency
                vertex_adjacency = result.trimesh.vertex_neighbors
                
                # Apply Laplacian smoothing
                new_vertices = vertices.copy()
                for i, neighbors in enumerate(vertex_adjacency):
                    if len(neighbors) > 0:
                        new_vertices[i] = np.mean(vertices[neighbors], axis=0)
                
                vertices = 0.5 * vertices + 0.5 * new_vertices  # Lambda = 0.5
            
            # Update mesh
            result.trimesh.vertices = vertices
            result._update_metadata_from_mesh()
            
            self.logger.info(f"Applied {iterations} smoothing iterations")
            
        except Exception as e:
            self.logger.error(f"Smoothing failed: {e}")
            raise
        
        return result
