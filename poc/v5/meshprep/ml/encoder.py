# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Mesh Feature Encoder - Extracts numerical features from 3D meshes.

Converts mesh geometry into a fixed-size feature vector suitable
for neural network input. Features capture:
- Geometry: vertex/face count, volume, surface area
- Topology: components, holes, manifoldness
- Problems: degenerate faces, non-manifold edges
- Scale: bounding box size for parameter scaling
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class MeshFeatures:
    """Numerical features extracted from a mesh."""
    
    # Geometry (log-scaled for neural network)
    vertex_count_log: float = 0.0
    face_count_log: float = 0.0
    volume_normalized: float = 0.0
    surface_area_normalized: float = 0.0
    
    # Scale (raw values for parameter calculation)
    bbox_diagonal: float = 1.0
    bbox_aspect_ratio: float = 1.0
    
    # Topology
    num_components: int = 1
    num_components_log: float = 0.0
    is_watertight: bool = False
    is_manifold: bool = False
    
    # Problem indicators (0-1 scale)
    hole_ratio: float = 0.0
    degeneracy_ratio: float = 0.0
    component_imbalance: float = 0.0
    
    # Derived flags
    is_fragmented: bool = False      # >10 components
    is_very_fragmented: bool = False  # >100 components
    
    def to_vector(self) -> np.ndarray:
        """Convert to fixed-size vector (16 features)."""
        return np.array([
            self.vertex_count_log,
            self.face_count_log,
            self.volume_normalized,
            self.surface_area_normalized,
            np.log10(max(self.bbox_diagonal, 0.001)),
            min(self.bbox_aspect_ratio / 10.0, 1.0),
            self.num_components_log,
            float(self.is_watertight),
            float(self.is_manifold),
            self.hole_ratio,
            self.degeneracy_ratio,
            self.component_imbalance,
            float(self.is_fragmented),
            float(self.is_very_fragmented),
            # Composite problem scores
            float(not self.is_watertight) * 0.5 + self.hole_ratio * 0.5,
            self.component_imbalance * float(self.num_components > 1),
        ], dtype=np.float32)
    
    @property
    def state_dim(self) -> int:
        """Dimension of state vector."""
        return 16


class MeshEncoder:
    """Extracts features from meshes for RL state representation."""
    
    def encode(self, mesh) -> MeshFeatures:
        """
        Extract features from a mesh.
        
        Args:
            mesh: Mesh object or trimesh.Trimesh
            
        Returns:
            MeshFeatures with all geometric properties
        """
        # Get trimesh object
        tm = mesh.trimesh if hasattr(mesh, 'trimesh') else mesh
        
        features = MeshFeatures()
        
        # Basic counts (log scale)
        features.vertex_count_log = np.log10(max(len(tm.vertices), 1))
        features.face_count_log = np.log10(max(len(tm.faces), 1))
        
        # Bounding box
        try:
            bbox_size = tm.bounds[1] - tm.bounds[0]
            features.bbox_diagonal = float(np.linalg.norm(bbox_size))
            if bbox_size.min() > 0:
                features.bbox_aspect_ratio = float(bbox_size.max() / bbox_size.min())
        except Exception:
            pass
        
        # Normalized volume and area
        if features.bbox_diagonal > 0:
            try:
                if tm.is_volume:
                    features.volume_normalized = abs(tm.volume) / (features.bbox_diagonal ** 3)
                features.surface_area_normalized = tm.area / (features.bbox_diagonal ** 2)
            except Exception:
                pass
        
        # Topology
        features.is_watertight = bool(tm.is_watertight)
        features.is_manifold = bool(tm.is_volume)
        
        # Components
        try:
            components = tm.split(only_watertight=False)
            features.num_components = len(components)
            features.num_components_log = np.log10(max(features.num_components, 1))
            features.is_fragmented = features.num_components > 10
            features.is_very_fragmented = features.num_components > 100
            
            if features.num_components > 1:
                sizes = [len(c.faces) for c in components]
                features.component_imbalance = 1.0 - max(sizes) / sum(sizes)
        except Exception:
            pass
        
        # Holes (boundary edges)
        try:
            edge_counts = tm.edges_face
            boundary = int(np.sum(edge_counts == 1))
            total = len(tm.edges_unique)
            features.hole_ratio = boundary / max(total, 1)
        except Exception:
            pass
        
        # Degenerate faces
        try:
            areas = tm.area_faces
            degenerate = int(np.sum(areas < 1e-10))
            features.degeneracy_ratio = degenerate / max(len(tm.faces), 1)
        except Exception:
            pass
        
        return features
