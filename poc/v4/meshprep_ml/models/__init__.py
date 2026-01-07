# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Neural network architectures for mesh repair prediction.

This module implements deep learning models for:
1. Mesh encoding (geometry ? latent vector)
2. Pipeline selection (which repair to try first)
3. Quality prediction (expected quality score)
4. Failure prediction (will this action crash?)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MeshEncoder(nn.Module):
    """
    Encode mesh geometry into a fixed-size latent vector.
    
    Uses a PointNet++-inspired architecture that works on sampled
    surface points. This is more robust than face-based methods for
    broken/non-manifold meshes.
    
    Input: Point cloud (N, 3) + normals (N, 3)
    Output: Latent vector (256,)
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        num_points: int = 2048,
        use_normals: bool = True,
    ):
        """
        Args:
            latent_dim: Size of output latent vector
            num_points: Number of points to sample from mesh
            use_normals: Whether to use surface normals as features
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_points = num_points
        self.use_normals = use_normals
        
        # Input: (x, y, z) + optional (nx, ny, nz)
        input_dim = 6 if use_normals else 3
        
        # PointNet-style feature extraction
        # Each layer processes points independently, then max-pools
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Global feature aggregation
        self.fc1 = nn.Linear(256, latent_dim)
        self.bn_fc = nn.BatchNorm1d(latent_dim)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, points: torch.Tensor, normals: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode point cloud to latent vector.
        
        Args:
            points: (B, N, 3) tensor of point coordinates
            normals: (B, N, 3) tensor of point normals (optional)
            
        Returns:
            latent: (B, latent_dim) tensor
        """
        batch_size = points.size(0)
        
        # Concatenate points and normals if available
        if self.use_normals and normals is not None:
            x = torch.cat([points, normals], dim=2)  # (B, N, 6)
        else:
            x = points  # (B, N, 3)
        
        # Transpose for Conv1d: (B, C, N)
        x = x.transpose(1, 2)
        
        # Point-wise feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global max pooling across all points
        x = torch.max(x, 2)[0]  # (B, 256)
        
        # Final projection to latent space
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        
        return x


class PipelineSelector(nn.Module):
    """
    Select the best repair pipeline for a given mesh.
    
    Input: Mesh latent vector + mesh statistics
    Output: Probability distribution over pipelines
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        num_pipelines: int = 50,  # Number of available pipelines
        hidden_dim: int = 128,
    ):
        """
        Args:
            latent_dim: Size of mesh encoding
            num_pipelines: Number of repair pipelines to choose from
            hidden_dim: Hidden layer size
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_pipelines = num_pipelines
        
        # Additional mesh statistics (watertight, face count, etc.)
        stat_dim = 10  # [is_watertight, face_count_log, body_count, ...]
        
        # Classifier network
        self.fc1 = nn.Linear(latent_dim + stat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_pipelines)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, latent: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        """
        Predict pipeline probabilities.
        
        Args:
            latent: (B, latent_dim) mesh encoding
            stats: (B, stat_dim) mesh statistics
            
        Returns:
            probs: (B, num_pipelines) probability distribution
        """
        # Concatenate geometry and statistics
        x = torch.cat([latent, stats], dim=1)
        
        # Multi-layer classifier
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        # Softmax for probability distribution
        probs = F.softmax(x, dim=1)
        
        return probs


class QualityPredictor(nn.Module):
    """
    Predict repair quality score (1-5) for a given pipeline.
    
    Input: Mesh latent vector + pipeline embedding
    Output: Quality score (1-5) and confidence
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        num_pipelines: int = 50,
        pipeline_embed_dim: int = 32,
        hidden_dim: int = 128,
    ):
        """
        Args:
            latent_dim: Size of mesh encoding
            num_pipelines: Number of pipelines
            pipeline_embed_dim: Size of pipeline embedding
            hidden_dim: Hidden layer size
        """
        super().__init__()
        
        # Learn embeddings for each pipeline
        self.pipeline_embedding = nn.Embedding(num_pipelines, pipeline_embed_dim)
        
        # Regressor network
        self.fc1 = nn.Linear(latent_dim + pipeline_embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Two heads: quality score + confidence
        self.quality_head = nn.Linear(hidden_dim, 1)
        self.confidence_head = nn.Linear(hidden_dim, 1)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, latent: torch.Tensor, pipeline_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict quality and confidence for a pipeline.
        
        Args:
            latent: (B, latent_dim) mesh encoding
            pipeline_ids: (B,) pipeline indices
            
        Returns:
            quality: (B,) predicted quality score (1-5)
            confidence: (B,) confidence in prediction (0-1)
        """
        # Embed pipeline IDs
        pipeline_embed = self.pipeline_embedding(pipeline_ids)  # (B, embed_dim)
        
        # Concatenate mesh and pipeline features
        x = torch.cat([latent, pipeline_embed], dim=1)
        
        # Shared layers
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        
        # Predict quality (1-5)
        quality = self.quality_head(x).squeeze(-1)
        quality = torch.clamp(quality, 1.0, 5.0)  # Enforce range
        
        # Predict confidence (0-1)
        confidence = torch.sigmoid(self.confidence_head(x).squeeze(-1))
        
        return quality, confidence


class MeshRepairNet(nn.Module):
    """
    Combined model for mesh repair prediction.
    
    Integrates:
    - Mesh encoder
    - Pipeline selector
    - Quality predictor
    
    This is the main model for POC v4.
    """
    
    def __init__(
        self,
        num_pipelines: int = 50,
        latent_dim: int = 256,
        num_points: int = 2048,
    ):
        """
        Args:
            num_pipelines: Number of available repair pipelines
            latent_dim: Size of mesh latent representation
            num_points: Points to sample from each mesh
        """
        super().__init__()
        
        self.encoder = MeshEncoder(latent_dim=latent_dim, num_points=num_points)
        self.pipeline_selector = PipelineSelector(latent_dim=latent_dim, num_pipelines=num_pipelines)
        self.quality_predictor = QualityPredictor(latent_dim=latent_dim, num_pipelines=num_pipelines)
    
    def forward(
        self,
        points: torch.Tensor,
        normals: torch.Tensor,
        stats: torch.Tensor,
        pipeline_ids: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Full forward pass for mesh repair prediction.
        
        Args:
            points: (B, N, 3) sampled surface points
            normals: (B, N, 3) surface normals
            stats: (B, 10) mesh statistics
            pipeline_ids: (B,) pipeline IDs for quality prediction (optional)
            
        Returns:
            dict with:
                - latent: (B, latent_dim) mesh encoding
                - pipeline_probs: (B, num_pipelines) pipeline selection probabilities
                - quality: (B,) predicted quality (if pipeline_ids provided)
                - confidence: (B,) quality prediction confidence
        """
        # Encode mesh geometry
        latent = self.encoder(points, normals)
        
        # Select best pipeline
        pipeline_probs = self.pipeline_selector(latent, stats)
        
        # Predict quality for specified pipelines
        if pipeline_ids is not None:
            quality, confidence = self.quality_predictor(latent, pipeline_ids)
        else:
            quality, confidence = None, None
        
        return {
            "latent": latent,
            "pipeline_probs": pipeline_probs,
            "quality": quality,
            "confidence": confidence,
        }
    
    def predict_best_pipeline(
        self,
        points: torch.Tensor,
        normals: torch.Tensor,
        stats: torch.Tensor,
        top_k: int = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict top-k best pipelines for a mesh.
        
        Args:
            points: (B, N, 3) sampled surface points
            normals: (B, N, 3) surface normals
            stats: (B, 10) mesh statistics
            top_k: Number of top pipelines to return
            
        Returns:
            pipeline_ids: (B, k) top pipeline indices
            probs: (B, k) corresponding probabilities
        """
        with torch.no_grad():
            output = self.forward(points, normals, stats)
            pipeline_probs = output["pipeline_probs"]
            
            # Get top-k pipelines
            probs, pipeline_ids = torch.topk(pipeline_probs, k=top_k, dim=1)
        
        return pipeline_ids, probs


# =============================================================================
# Loss Functions
# =============================================================================

class PipelineSelectorLoss(nn.Module):
    """
    Loss for training pipeline selector.
    
    Uses cross-entropy with label smoothing to prevent overconfidence.
    """
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred_probs: torch.Tensor, target_pipeline: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_probs: (B, num_pipelines) predicted probabilities
            target_pipeline: (B,) ground truth pipeline indices
            
        Returns:
            loss: Scalar loss value
        """
        num_classes = pred_probs.size(1)
        
        # One-hot encode targets
        one_hot = F.one_hot(target_pipeline, num_classes).float()
        
        # Apply label smoothing
        smooth_labels = one_hot * (1 - self.smoothing) + self.smoothing / num_classes
        
        # Cross-entropy loss
        log_probs = torch.log(pred_probs + 1e-10)
        loss = -(smooth_labels * log_probs).sum(dim=1).mean()
        
        return loss


class QualityPredictorLoss(nn.Module):
    """
    Loss for training quality predictor.
    
    Combines MSE for quality prediction with confidence calibration.
    """
    
    def __init__(self, quality_weight: float = 1.0, confidence_weight: float = 0.1):
        super().__init__()
        self.quality_weight = quality_weight
        self.confidence_weight = confidence_weight
    
    def forward(
        self,
        pred_quality: torch.Tensor,
        pred_confidence: torch.Tensor,
        target_quality: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_quality: (B,) predicted quality scores
            pred_confidence: (B,) predicted confidence values
            target_quality: (B,) ground truth quality scores
            
        Returns:
            loss: Scalar loss value
        """
        # Quality prediction loss (MSE)
        quality_loss = F.mse_loss(pred_quality, target_quality)
        
        # Confidence calibration: confidence should match absolute error
        abs_error = torch.abs(pred_quality - target_quality)
        max_error = 4.0  # Max possible error (5-1)
        expected_confidence = 1.0 - (abs_error / max_error)
        confidence_loss = F.mse_loss(pred_confidence, expected_confidence)
        
        # Combined loss
        total_loss = (
            self.quality_weight * quality_loss +
            self.confidence_weight * confidence_loss
        )
        
        return total_loss


def create_model(num_pipelines: int = 50, device: str = "cuda") -> MeshRepairNet:
    """
    Create and initialize a MeshRepairNet model.
    
    Args:
        num_pipelines: Number of available repair pipelines
        device: Device to place model on ("cuda" or "cpu")
        
    Returns:
        Initialized model
    """
    model = MeshRepairNet(num_pipelines=num_pipelines)
    model = model.to(device)
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    
    logger.info(f"Created MeshRepairNet with {num_pipelines} pipelines")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model
