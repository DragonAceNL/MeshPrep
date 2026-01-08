# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Mesh Geometry Encoder - Extracts rich features from 3D meshes.

This encoder captures geometric properties that are relevant for
predicting repair strategies:
- Global shape features (volume, area, bbox)
- Topology features (holes, components, manifoldness)
- Local geometry features (curvature distribution, edge lengths)
- Problem indicators (degenerate faces, non-manifold edges)
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class MeshFeatures:
    """Rich feature representation of a mesh."""
    
    # Global features (normalized)
    vertex_count_log: float = 0.0
    face_count_log: float = 0.0
    volume_normalized: float = 0.0
    surface_area_normalized: float = 0.0
    bbox_aspect_ratio: float = 1.0
    bbox_diagonal: float = 0.0  # Raw diagonal for parameter scaling
    bbox_diagonal_log: float = 0.0  # Log scale for neural network
    
    # Topology features
    num_components: int = 1
    num_components_log: float = 0.0  # Log scale to handle extreme fragmentation
    num_boundary_edges: int = 0
    num_non_manifold_edges: int = 0
    genus: int = 0
    is_watertight: bool = False
    is_manifold: bool = False
    
    # Problem indicators (0-1 scale)
    hole_ratio: float = 0.0  # boundary_edges / total_edges
    degeneracy_ratio: float = 0.0  # degenerate_faces / total_faces
    component_imbalance: float = 0.0  # 1 - (largest_component / total)
    is_extremely_fragmented: bool = False  # >100 components
    
    # Local geometry statistics
    mean_edge_length: float = 0.0
    edge_length_std: float = 0.0
    mean_face_area: float = 0.0
    face_area_std: float = 0.0
    mean_dihedral_angle: float = 0.0
    
    # Point cloud features (sampled)
    point_cloud: Optional[np.ndarray] = None  # (N, 3)
    normals: Optional[np.ndarray] = None  # (N, 3)
    
    def to_vector(self) -> np.ndarray:
        """Convert to fixed-size feature vector (28 features)."""
        return np.array([
            self.vertex_count_log,
            self.face_count_log,
            self.volume_normalized,
            self.surface_area_normalized,
            self.bbox_aspect_ratio,
            self.bbox_diagonal_log,  # NEW: scale information
            self.num_components_log,  # NEW: log scale for fragmentation
            float(self.num_boundary_edges) / 1000,
            float(self.num_non_manifold_edges) / 100,
            float(self.genus) / 10,
            float(self.is_watertight),
            float(self.is_manifold),
            self.hole_ratio,
            self.degeneracy_ratio,
            self.component_imbalance,
            float(self.is_extremely_fragmented),  # NEW: binary flag
            self.mean_edge_length,
            self.edge_length_std,
            self.mean_face_area,
            self.face_area_std,
            self.mean_dihedral_angle / np.pi,  # Normalize to 0-1
            # Problem severity indicators (derived)
            float(not self.is_watertight) * 0.5 + self.hole_ratio * 0.5,
            float(not self.is_manifold) * 0.5 + float(self.num_non_manifold_edges > 0) * 0.5,
            self.component_imbalance,
            self.degeneracy_ratio,
            float(self.num_components > 1) * self.component_imbalance,
            float(self.num_boundary_edges > 0) * self.hole_ratio,
            min(self.num_components_log / 5.0, 1.0),  # NEW: fragmentation severity 0-1
        ], dtype=np.float32)


class MeshGeometryEncoder:
    """
    Extracts rich geometric features from meshes.
    
    Two modes:
    1. Feature vector mode: Fast 25-dimensional feature vector
    2. Point cloud mode: Full point cloud + normals for neural network
    """
    
    def __init__(self, num_sample_points: int = 2048):
        self.num_sample_points = num_sample_points
    
    def encode(self, mesh) -> MeshFeatures:
        """
        Extract features from a mesh.
        
        Args:
            mesh: Either a Mesh object or trimesh.Trimesh
            
        Returns:
            MeshFeatures with all geometric properties
        """
        # Handle both Mesh wrapper and raw trimesh
        if hasattr(mesh, 'trimesh'):
            tm = mesh.trimesh
        else:
            tm = mesh
        
        features = MeshFeatures()
        
        # Global features
        features.vertex_count_log = np.log10(max(len(tm.vertices), 1))
        features.face_count_log = np.log10(max(len(tm.faces), 1))
        
        # Volume and area (normalized by bbox diagonal)
        bbox_diagonal = 1.0
        try:
            bbox_diagonal = np.linalg.norm(tm.bounds[1] - tm.bounds[0])
            features.bbox_diagonal = bbox_diagonal
            features.bbox_diagonal_log = np.log10(max(bbox_diagonal, 0.001))
            
            if bbox_diagonal > 0:
                features.volume_normalized = abs(tm.volume) / (bbox_diagonal ** 3) if tm.is_volume else 0
                features.surface_area_normalized = tm.area / (bbox_diagonal ** 2)
            
            # Aspect ratio
            bbox_size = tm.bounds[1] - tm.bounds[0]
            if bbox_size.min() > 0:
                features.bbox_aspect_ratio = bbox_size.max() / bbox_size.min()
        except Exception as e:
            logger.debug(f"Could not compute volume/area: {e}")
        
        # Topology features
        features.is_watertight = tm.is_watertight
        features.is_manifold = tm.is_volume
        
        try:
            # Components
            components = tm.split(only_watertight=False)
            features.num_components = len(components)
            features.num_components_log = np.log10(max(features.num_components, 1))
            features.is_extremely_fragmented = features.num_components > 100
            
            if features.num_components > 1:
                component_sizes = [len(c.faces) for c in components]
                largest = max(component_sizes)
                total = sum(component_sizes)
                features.component_imbalance = 1.0 - (largest / total) if total > 0 else 0
        except Exception:
            features.num_components = 1
            features.num_components_log = 0.0
        
        # Boundary edges (holes)
        try:
            edges_face_count = tm.edges_face
            features.num_boundary_edges = int(np.sum(edges_face_count == 1))
            total_edges = len(tm.edges_unique)
            features.hole_ratio = features.num_boundary_edges / max(total_edges, 1)
        except Exception:
            pass
        
        # Non-manifold edges
        try:
            edges_face_count = tm.edges_face
            features.num_non_manifold_edges = int(np.sum(edges_face_count > 2))
        except Exception:
            pass
        
        # Degenerate faces
        try:
            face_areas = tm.area_faces
            degenerate = np.sum(face_areas < 1e-10)
            features.degeneracy_ratio = degenerate / max(len(tm.faces), 1)
        except Exception:
            pass
        
        # Edge length statistics
        try:
            edge_lengths = tm.edges_unique_length
            features.mean_edge_length = np.mean(edge_lengths) / bbox_diagonal if bbox_diagonal > 0 else 0
            features.edge_length_std = np.std(edge_lengths) / bbox_diagonal if bbox_diagonal > 0 else 0
        except Exception:
            pass
        
        # Face area statistics
        try:
            face_areas = tm.area_faces
            features.mean_face_area = np.mean(face_areas) / (bbox_diagonal ** 2) if bbox_diagonal > 0 else 0
            features.face_area_std = np.std(face_areas) / (bbox_diagonal ** 2) if bbox_diagonal > 0 else 0
        except Exception:
            pass
        
        # Dihedral angles
        try:
            face_adjacency_angles = tm.face_adjacency_angles
            if len(face_adjacency_angles) > 0:
                features.mean_dihedral_angle = np.mean(face_adjacency_angles)
        except Exception:
            pass
        
        # Sample point cloud
        try:
            if len(tm.faces) > 0:
                samples, face_idx = tm.sample(self.num_sample_points, return_index=True)
                features.point_cloud = samples
                features.normals = tm.face_normals[face_idx]
        except Exception as e:
            logger.debug(f"Could not sample point cloud: {e}")
        
        return features
    
    def encode_to_tensor(self, mesh, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode mesh to PyTorch tensors for neural network.
        
        Returns:
            Tuple of (feature_vector, point_cloud_with_normals)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        features = self.encode(mesh)
        
        # Feature vector
        feature_vec = torch.from_numpy(features.to_vector()).float().to(device)
        
        # Point cloud with normals (N, 6)
        if features.point_cloud is not None and features.normals is not None:
            points_normals = np.concatenate([features.point_cloud, features.normals], axis=1)
            point_cloud = torch.from_numpy(points_normals).float().to(device)
        else:
            point_cloud = torch.zeros(self.num_sample_points, 6, device=device)
        
        return feature_vec, point_cloud


class NeuralMeshEncoder(nn.Module):
    """
    Neural network encoder for mesh geometry.
    
    Combines:
    1. Global feature MLP
    2. Point cloud PointNet encoder
    3. Fusion layer
    """
    
    def __init__(self, latent_dim: int = 128, num_points: int = 2048):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_points = num_points
        
        # Global feature encoder (28 features -> latent_dim/2)
        self.global_encoder = nn.Sequential(
            nn.Linear(28, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, latent_dim // 2),
        )
        
        # Point cloud encoder (simplified PointNet)
        self.point_encoder = nn.Sequential(
            nn.Conv1d(6, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, latent_dim // 2, 1),
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.BatchNorm1d(latent_dim),
        )
    
    def forward(self, global_features: torch.Tensor, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        Encode mesh to latent vector.
        
        Args:
            global_features: (B, 25) global feature vectors
            point_cloud: (B, N, 6) point cloud with normals
            
        Returns:
            (B, latent_dim) latent vectors
        """
        # Encode global features
        global_latent = self.global_encoder(global_features)  # (B, latent_dim/2)
        
        # Encode point cloud
        points = point_cloud.transpose(1, 2)  # (B, 6, N)
        point_features = self.point_encoder(points)  # (B, latent_dim/2, N)
        point_latent = point_features.max(dim=2)[0]  # (B, latent_dim/2) - global max pool
        
        # Fuse
        combined = torch.cat([global_latent, point_latent], dim=1)  # (B, latent_dim)
        latent = self.fusion(combined)
        
        return latent
