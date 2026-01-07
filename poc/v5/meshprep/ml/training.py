# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Training utilities for ML models."""

from typing import List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class MeshDataset(Dataset):
    """Dataset for mesh repair training."""
    
    def __init__(self, samples: List[Tuple]):
        """
        Initialize dataset.
        
        Args:
            samples: List of (mesh, pipeline_id, quality_score) tuples
        """
        self.samples = samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        mesh, pipeline_id, quality = self.samples[idx]
        
        # Sample points from mesh
        sample = mesh.sample_points(n_points=2048, include_normals=True)
        
        return {
            "points": torch.from_numpy(sample["points"]).float(),
            "normals": torch.from_numpy(sample["normals"]).float(),
            "pipeline_id": torch.tensor(pipeline_id, dtype=torch.long),
            "quality": torch.tensor(quality, dtype=torch.float),
        }


class Trainer:
    """Training utilities for ML models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "auto",
        learning_rate: float = 1e-4,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            device: Device to use
            learning_rate: Learning rate
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not installed")
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        logger.info(f"Trainer initialized on {self.device}")
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            loss = criterion(self.model, batch)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
    ) -> float:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Forward
                loss = criterion(self.model, batch)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    path: Path,
):
    """Save training checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, path)
    
    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
) -> int:
    """Load training checkpoint."""
    checkpoint = torch.load(path)
    
    model.load_state_dict(checkpoint["model_state"])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    
    epoch = checkpoint["epoch"]
    
    logger.info(f"Loaded checkpoint from {path} (epoch {epoch})")
    
    return epoch
