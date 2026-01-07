# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Quality scorer for predicting repair quality."""

from typing import Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .encoder import MeshEncoder


class QualityRegressor(nn.Module):
    """Regressor to predict repair quality (1-5)."""
    
    def __init__(self, latent_dim: int, pipeline_embedding_dim: int = 32):
        """
        Initialize regressor.
        
        Args:
            latent_dim: Dimension of mesh latent vector
            pipeline_embedding_dim: Dimension of pipeline embedding
        """
        super().__init__()
        
        # Pipeline embedding (learns representation of each pipeline)
        self.num_pipelines = 20  # Max number of pipelines
        self.pipeline_embed = nn.Embedding(self.num_pipelines, pipeline_embedding_dim)
        
        input_dim = latent_dim + pipeline_embedding_dim
        
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        
        # Output: quality score (1-5) + confidence
        self.fc_quality = nn.Linear(64, 1)
        self.fc_confidence = nn.Linear(64, 1)
    
    def forward(
        self,
        latent: torch.Tensor,
        pipeline_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict quality and confidence.
        
        Args:
            latent: (B, latent_dim) mesh latent vectors
            pipeline_id: (B,) pipeline IDs
            
        Returns:
            quality: (B, 1) quality scores (1-5)
            confidence: (B, 1) confidence scores (0-1)
        """
        # Embed pipeline
        pipeline_features = self.pipeline_embed(pipeline_id)  # (B, pipeline_embedding_dim)
        
        # Concatenate
        x = torch.cat([latent, pipeline_features], dim=1)
        
        # MLP
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        
        # Predict quality (1-5) and confidence (0-1)
        quality = torch.sigmoid(self.fc_quality(x)) * 4 + 1  # Scale to [1, 5]
        confidence = torch.sigmoid(self.fc_confidence(x))
        
        return quality, confidence


class QualityScorer:
    """
    Predicts repair quality for mesh + pipeline combination.
    
    Returns quality score (1-5) and confidence (0-1).
    """
    
    def __init__(
        self,
        encoder: Optional[MeshEncoder] = None,
        regressor: Optional[QualityRegressor] = None,
        device: str = "auto",
    ):
        """
        Initialize quality scorer.
        
        Args:
            encoder: Pre-trained encoder
            regressor: Pre-trained regressor
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not installed")
        
        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Default encoder
        if encoder is None:
            encoder = MeshEncoder(latent_dim=256, num_points=2048)
        
        self.encoder = encoder.to(self.device)
        self.encoder.eval()
        
        # Default regressor
        if regressor is None:
            regressor = QualityRegressor(latent_dim=256, pipeline_embedding_dim=32)
        
        self.regressor = regressor.to(self.device)
        self.regressor.eval()
        
        # Pipeline ID mapping
        self.pipeline_to_id = {}
        
        logger.info(f"QualityScorer initialized on {self.device}")
    
    def predict_quality(
        self,
        mesh,
        pipeline_name: str,
    ) -> Tuple[float, float]:
        """
        Predict quality for mesh + pipeline.
        
        Args:
            mesh: Mesh object
            pipeline_name: Name of pipeline
            
        Returns:
            (quality, confidence) where quality is 1-5, confidence is 0-1
        """
        # Get pipeline ID
        pipeline_id = self.pipeline_to_id.get(pipeline_name, 0)
        pipeline_id_tensor = torch.tensor([pipeline_id], device=self.device)
        
        # Encode mesh
        with torch.no_grad():
            latent = self.encoder.encode_mesh(mesh)
            latent = latent.unsqueeze(0)  # (1, latent_dim)
            
            # Predict
            quality, confidence = self.regressor(latent, pipeline_id_tensor)
        
        quality_value = quality.item()
        confidence_value = confidence.item()
        
        logger.debug(f"Predicted quality: {quality_value:.2f}/5 "
                    f"(confidence: {confidence_value:.2%})")
        
        return quality_value, confidence_value
    
    def save(self, path: Path):
        """Save model to disk."""
        torch.save({
            "encoder_state": self.encoder.state_dict(),
            "regressor_state": self.regressor.state_dict(),
            "pipeline_to_id": self.pipeline_to_id,
            "latent_dim": self.encoder.latent_dim,
            "num_points": self.encoder.num_points,
        }, path)
        
        logger.info(f"Saved model to {path}")
    
    @classmethod
    def load(cls, path: Path, device: str = "auto") -> "QualityScorer":
        """Load model from disk."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not installed")
        
        checkpoint = torch.load(path, map_location="cpu")
        
        # Create encoder
        encoder = MeshEncoder(
            latent_dim=checkpoint["latent_dim"],
            num_points=checkpoint["num_points"],
        )
        encoder.load_state_dict(checkpoint["encoder_state"])
        
        # Create regressor
        regressor = QualityRegressor(latent_dim=checkpoint["latent_dim"])
        regressor.load_state_dict(checkpoint["regressor_state"])
        
        # Create scorer
        scorer = cls(
            encoder=encoder,
            regressor=regressor,
            device=device,
        )
        scorer.pipeline_to_id = checkpoint["pipeline_to_id"]
        
        logger.info(f"Loaded model from {path}")
        
        return scorer
