# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Pipeline predictor using mesh encoder."""

from typing import List, Tuple, Optional
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


class PipelineClassifier(nn.Module):
    """Classifier to predict best pipeline from latent vector."""
    
    def __init__(self, latent_dim: int, num_pipelines: int):
        """
        Initialize classifier.
        
        Args:
            latent_dim: Dimension of input latent vector
            num_pipelines: Number of pipeline classes to predict
        """
        super().__init__()
        
        self.fc1 = nn.Linear(latent_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(256, num_pipelines)
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Predict pipeline probabilities.
        
        Args:
            latent: (B, latent_dim) latent vectors
            
        Returns:
            (B, num_pipelines) logits
        """
        x = F.relu(self.bn1(self.fc1(latent)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x


class PipelinePredictor:
    """
    Predicts best repair pipeline for a given mesh.
    
    Uses MeshEncoder + PipelineClassifier.
    """
    
    def __init__(
        self,
        encoder: Optional[MeshEncoder] = None,
        classifier: Optional[PipelineClassifier] = None,
        pipeline_names: Optional[List[str]] = None,
        device: str = "auto",
    ):
        """
        Initialize predictor.
        
        Args:
            encoder: Pre-trained encoder
            classifier: Pre-trained classifier
            pipeline_names: List of pipeline names (in order)
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
        
        # Default classifier (must be provided or loaded for actual use)
        if classifier is None:
            # Placeholder - will be replaced when loading model
            classifier = PipelineClassifier(latent_dim=256, num_pipelines=10)
        
        self.classifier = classifier.to(self.device)
        self.classifier.eval()
        
        # Pipeline names
        if pipeline_names is None:
            pipeline_names = [
                "cleanup", "standard", "aggressive", "reconstruction",
                "optimize", "thin_wall", "fragments", "geometry_fix",
                "enhance", "nuclear"
            ]
        
        self.pipeline_names = pipeline_names
        
        logger.info(f"PipelinePredictor initialized on {self.device}")
    
    def predict(
        self,
        mesh,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Predict best pipelines for a mesh.
        
        Args:
            mesh: Mesh object
            top_k: Number of top predictions to return
            
        Returns:
            List of (pipeline_name, confidence) tuples
        """
        # Encode mesh
        with torch.no_grad():
            latent = self.encoder.encode_mesh(mesh)
            latent = latent.unsqueeze(0)  # (1, latent_dim)
            
            # Predict
            logits = self.classifier(latent)
            probs = F.softmax(logits, dim=1).squeeze(0)  # (num_pipelines,)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, min(top_k, len(self.pipeline_names)))
        
        results = []
        for prob, idx in zip(top_probs, top_indices):
            pipeline_name = self.pipeline_names[idx.item()]
            confidence = prob.item()
            results.append((pipeline_name, confidence))
        
        logger.info(f"Top prediction: {results[0][0]} ({results[0][1]:.2%} confidence)")
        
        return results
    
    def save(self, path: Path):
        """Save model to disk."""
        torch.save({
            "encoder_state": self.encoder.state_dict(),
            "classifier_state": self.classifier.state_dict(),
            "pipeline_names": self.pipeline_names,
            "latent_dim": self.encoder.latent_dim,
            "num_points": self.encoder.num_points,
        }, path)
        
        logger.info(f"Saved model to {path}")
    
    @classmethod
    def load(cls, path: Path, device: str = "auto") -> "PipelinePredictor":
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
        
        # Create classifier
        num_pipelines = len(checkpoint["pipeline_names"])
        classifier = PipelineClassifier(
            latent_dim=checkpoint["latent_dim"],
            num_pipelines=num_pipelines,
        )
        classifier.load_state_dict(checkpoint["classifier_state"])
        
        # Create predictor
        predictor = cls(
            encoder=encoder,
            classifier=classifier,
            pipeline_names=checkpoint["pipeline_names"],
            device=device,
        )
        
        logger.info(f"Loaded model from {path}")
        
        return predictor
