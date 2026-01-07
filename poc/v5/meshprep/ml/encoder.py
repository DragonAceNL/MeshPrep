# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Mesh encoder using PointNet++ architecture."""

from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. ML features disabled.")


class PointNetSetAbstraction(nn.Module):
    """PointNet++ Set Abstraction layer."""
    
    def __init__(self, in_channel: int, mlp: list):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
    
    def forward(self, xyz: torch.Tensor, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            xyz: (B, N, 3) point coordinates
            features: (B, C, N) point features
        Returns:
            (B, mlp[-1]) global features
        """
        if features is None:
            features = xyz.transpose(1, 2)  # (B, 3, N)
        else:
            features = torch.cat([xyz.transpose(1, 2), features], dim=1)
        
        # MLP layers
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            features = F.relu(bn(conv(features)))
        
        # Global max pooling
        features = torch.max(features, 2)[0]  # (B, mlp[-1])
        
        return features


class MeshEncoder(nn.Module):
    """
    PointNet++ style encoder for mesh geometry.
    
    Encodes mesh as point cloud with normals to latent vector.
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        num_points: int = 2048,
    ):
        """
        Initialize encoder.
        
        Args:
            latent_dim: Dimension of latent vector
            num_points: Number of points to sample from mesh
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not installed. Install with: pip install torch")
        
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_points = num_points
        
        # PointNet++ architecture
        # Input: (B, N, 6) - points (3) + normals (3)
        self.sa1 = PointNetSetAbstraction(6, [64, 64, 128])
        self.sa2 = PointNetSetAbstraction(128, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(256, [256, 512, latent_dim])
        
        logger.debug(f"MeshEncoder initialized: latent_dim={latent_dim}, "
                    f"num_points={num_points}")
    
    def forward(
        self,
        points: torch.Tensor,
        normals: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode mesh to latent vector.
        
        Args:
            points: (B, N, 3) point coordinates
            normals: (B, N, 3) point normals
            
        Returns:
            (B, latent_dim) latent vectors
        """
        # Concatenate points and normals
        features = torch.cat([points, normals], dim=2)  # (B, N, 6)
        
        # PointNet++ layers
        x1 = self.sa1(features[:, :, :3], features[:, :, 3:].transpose(1, 2))
        x2 = self.sa2(features[:, :, :3], x1.unsqueeze(2).expand(-1, -1, features.size(1)))
        x3 = self.sa3(features[:, :, :3], x2.unsqueeze(2).expand(-1, -1, features.size(1)))
        
        return x3
    
    def encode_mesh(self, mesh) -> torch.Tensor:
        """
        Encode a Mesh object to latent vector.
        
        Args:
            mesh: Mesh object from meshprep.core.mesh
            
        Returns:
            (latent_dim,) latent vector (on same device as model)
        """
        # Sample points from mesh
        sample = mesh.sample_points(self.num_points, include_normals=True)
        
        points = torch.from_numpy(sample["points"]).float()
        normals = torch.from_numpy(sample["normals"]).float()
        
        # Add batch dimension
        points = points.unsqueeze(0)  # (1, N, 3)
        normals = normals.unsqueeze(0)  # (1, N, 3)
        
        # Move to same device as model
        device = next(self.parameters()).device
        points = points.to(device)
        normals = normals.to(device)
        
        # Encode
        with torch.no_grad():
            latent = self.forward(points, normals)
        
        return latent.squeeze(0)  # (latent_dim,)
