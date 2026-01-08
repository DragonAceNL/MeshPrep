# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Learning Loop - The core training system that learns from repair outcomes.

This is the heart of the ML engine. It:
1. Collects repair experiences (mesh -> actions -> outcome)
2. Trains the neural network on successful repairs
3. Updates predictions based on new data
4. Continuously improves repair quality

Training strategies:
- Supervised: Learn from high-quality outcomes
- Reinforcement: Reward signal based on quality + printability
- Imitation: Learn from manually curated "perfect" repairs
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .mesh_encoder import MeshGeometryEncoder, NeuralMeshEncoder, MeshFeatures
from .repair_predictor import RepairPredictor, ACTIONS, ACTION_TO_IDX
from .outcome_tracker import OutcomeTracker, RepairOutcome


@dataclass
class TrainingConfig:
    """Configuration for the learning loop."""
    
    # Model architecture
    latent_dim: int = 128
    num_sample_points: int = 2048
    
    # Training
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs_per_update: int = 10
    min_samples_to_train: int = 50
    
    # Reward weighting
    quality_weight: float = 0.5
    printability_weight: float = 0.3
    fidelity_weight: float = 0.2
    
    # Saving
    save_interval: int = 100  # Save every N updates
    model_dir: Path = Path("models")


class RepairDataset(Dataset):
    """Dataset of repair experiences for training."""
    
    def __init__(self, samples: List[Dict]):
        self.samples = samples
        self.num_actions = len(ACTIONS)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Feature vector
        features = torch.tensor(sample["features"], dtype=torch.float32)
        
        # Action labels (multi-hot encoding)
        action_labels = torch.zeros(self.num_actions)
        for action in sample["actions"]:
            if action in ACTION_TO_IDX:
                action_labels[ACTION_TO_IDX[action]] = 1.0
        
        # Reward as weight
        reward = torch.tensor(sample["reward"], dtype=torch.float32)
        
        return features, action_labels, reward


class LearningLoop:
    """
    Main learning system that trains from repair outcomes.
    
    Usage:
        loop = LearningLoop()
        
        # After each repair:
        loop.record_outcome(mesh, actions, result)
        
        # Periodically train:
        loop.train_step()
        
        # Get predictions:
        actions = loop.predict(mesh)
    """
    
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        device: str = "auto",
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for learning loop")
        
        self.config = config or TrainingConfig()
        
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"LearningLoop using device: {self.device}")
        
        # Components
        self.feature_encoder = MeshGeometryEncoder(self.config.num_sample_points)
        
        self.neural_encoder = NeuralMeshEncoder(
            latent_dim=self.config.latent_dim,
            num_points=self.config.num_sample_points,
        ).to(self.device)
        
        self.predictor = RepairPredictor(
            latent_dim=self.config.latent_dim,
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.neural_encoder.parameters()) + list(self.predictor.parameters()),
            lr=self.config.learning_rate,
        )
        
        # Outcome tracker
        self.tracker = OutcomeTracker()
        
        # Training state
        self.update_count = 0
        self.total_samples_seen = 0
        
        # Try to load existing model
        self._try_load_model()
    
    def _try_load_model(self):
        """Try to load existing model weights."""
        model_path = self.config.model_dir / "learning_loop_model.pt"
        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.neural_encoder.load_state_dict(checkpoint["encoder"])
                self.predictor.load_state_dict(checkpoint["predictor"])
                self.update_count = checkpoint.get("update_count", 0)
                self.total_samples_seen = checkpoint.get("total_samples", 0)
                logger.info(f"Loaded model from {model_path} (update {self.update_count})")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
    
    def save_model(self):
        """Save model weights."""
        self.config.model_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.config.model_dir / "learning_loop_model.pt"
        
        torch.save({
            "encoder": self.neural_encoder.state_dict(),
            "predictor": self.predictor.state_dict(),
            "update_count": self.update_count,
            "total_samples": self.total_samples_seen,
        }, model_path)
        
        logger.info(f"Saved model to {model_path}")
    
    def record_outcome(
        self,
        mesh,
        actions: List[str],
        parameters: Dict[str, Dict],
        is_printable: bool,
        quality_score: float,
        volume_change_pct: float = 0.0,
        hausdorff_relative: float = 0.0,
        duration_ms: float = 0.0,
        mesh_id: Optional[str] = None,
    ) -> int:
        """
        Record a repair outcome for learning.
        
        Args:
            mesh: Original mesh (before repair)
            actions: List of actions applied
            parameters: Parameters used for each action
            is_printable: Whether result is printable
            quality_score: Quality score 1-5
            volume_change_pct: Volume change percentage
            hausdorff_relative: Hausdorff distance relative to bbox
            duration_ms: Repair duration
            mesh_id: Optional identifier for the mesh
            
        Returns:
            ID of the recorded outcome
        """
        # Extract features
        features = self.feature_encoder.encode(mesh)
        feature_vector = features.to_vector().tolist()
        
        outcome = RepairOutcome(
            mesh_id=mesh_id or f"mesh_{self.total_samples_seen}",
            input_features={
                "vertex_count": 10 ** features.vertex_count_log,
                "face_count": 10 ** features.face_count_log,
                "num_components": features.num_components,
                "is_watertight": features.is_watertight,
                "hole_ratio": features.hole_ratio,
            },
            input_feature_vector=feature_vector,
            actions=actions,
            parameters=parameters,
            success=is_printable,
            is_printable=is_printable,
            quality_score=quality_score,
            volume_change_pct=volume_change_pct,
            hausdorff_relative=hausdorff_relative,
            duration_ms=duration_ms,
        )
        
        outcome_id = self.tracker.record(outcome)
        self.total_samples_seen += 1
        
        logger.debug(f"Recorded outcome {outcome_id}: actions={actions}, quality={quality_score}")
        
        return outcome_id
    
    def train_step(self, force: bool = False) -> Optional[Dict]:
        """
        Perform a training step if enough data is available.
        
        Args:
            force: Train even if not enough samples
            
        Returns:
            Training metrics if training occurred, None otherwise
        """
        # Get training data
        samples = self.tracker.get_training_data(
            min_quality=2.0,  # Only learn from decent repairs
            only_printable=False,
        )
        
        if len(samples) < self.config.min_samples_to_train and not force:
            logger.debug(f"Not enough samples to train ({len(samples)} < {self.config.min_samples_to_train})")
            return None
        
        # Filter samples with feature vectors
        valid_samples = [s for s in samples if s["features"] is not None]
        
        if len(valid_samples) < 10:
            logger.debug("Not enough valid samples with features")
            return None
        
        # Create dataset
        dataset = RepairDataset(valid_samples)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        
        # Training loop
        self.neural_encoder.train()
        self.predictor.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.config.epochs_per_update):
            for features, action_labels, rewards in dataloader:
                features = features.to(self.device)
                action_labels = action_labels.to(self.device)
                rewards = rewards.to(self.device)
                
                # Create dummy point cloud (zeros for now - feature vector is primary)
                batch_size = features.size(0)
                point_cloud = torch.zeros(batch_size, self.config.num_sample_points, 6, device=self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                latent = self.neural_encoder(features, point_cloud)
                action_logits, _, _, confidence = self.predictor(latent)
                
                # Weighted BCE loss (weight by reward)
                bce_loss = nn.functional.binary_cross_entropy_with_logits(
                    action_logits,
                    action_labels,
                    reduction='none'
                )
                
                # Weight by reward (better outcomes have higher weight)
                weighted_loss = (bce_loss * rewards.unsqueeze(1)).mean()
                
                # Add confidence regularization
                confidence_loss = 0.1 * (confidence.squeeze() - rewards).pow(2).mean()
                
                loss = weighted_loss + confidence_loss
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        self.update_count += 1
        avg_loss = total_loss / max(num_batches, 1)
        
        # Save periodically
        if self.update_count % self.config.save_interval == 0:
            self.save_model()
        
        metrics = {
            "loss": avg_loss,
            "samples_used": len(valid_samples),
            "update_count": self.update_count,
            "epochs": self.config.epochs_per_update,
        }
        
        logger.info(f"Training step {self.update_count}: loss={avg_loss:.4f}, samples={len(valid_samples)}")
        
        return metrics
    
    def predict(self, mesh, threshold: float = 0.4) -> Tuple[List[str], Dict[str, Dict], float]:
        """
        Predict repair strategy for a mesh.
        
        Args:
            mesh: Mesh to repair
            threshold: Minimum action probability threshold
            
        Returns:
            Tuple of (actions, parameters, confidence)
        """
        self.neural_encoder.eval()
        self.predictor.eval()
        
        # Encode mesh
        features = self.feature_encoder.encode(mesh)
        feature_vector = torch.tensor(features.to_vector(), dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Create point cloud tensor
        if features.point_cloud is not None and features.normals is not None:
            points_normals = np.concatenate([features.point_cloud, features.normals], axis=1)
            point_cloud = torch.tensor(points_normals, dtype=torch.float32).unsqueeze(0).to(self.device)
        else:
            point_cloud = torch.zeros(1, self.config.num_sample_points, 6, device=self.device)
        
        with torch.no_grad():
            latent = self.neural_encoder(feature_vector, point_cloud)
            prediction = self.predictor.predict(latent, threshold=threshold)
        
        return prediction.actions, prediction.parameters, prediction.confidence
    
    def predict_with_fallback(self, mesh) -> Tuple[List[str], Dict[str, Dict], float]:
        """
        Predict with fallback to nearest neighbor if model is untrained.
        
        Args:
            mesh: Mesh to repair
            
        Returns:
            Tuple of (actions, parameters, confidence)
        """
        # Try neural prediction first
        actions, params, confidence = self.predict(mesh)
        
        # If confidence is low and we have historical data, use nearest neighbor
        if confidence < 0.3 and self.total_samples_seen > 10:
            features = self.feature_encoder.encode(mesh)
            feature_vector = features.to_vector().tolist()
            
            neighbors = self.tracker.get_best_actions_for_features(feature_vector, k=3)
            
            if neighbors:
                best = neighbors[0]
                logger.debug(f"Using nearest neighbor (distance={best['distance']:.3f})")
                return best["actions"], best["parameters"], 0.5
        
        # If still no good prediction, use default pipeline
        if confidence < 0.2 or not actions:
            logger.debug("Using default pipeline (low confidence)")
            return self._default_pipeline(mesh), {}, 0.1
        
        return actions, params, confidence
    
    def _default_pipeline(self, mesh) -> List[str]:
        """Get default repair pipeline based on mesh problems."""
        features = self.feature_encoder.encode(mesh)
        
        actions = []
        
        # Always start with cleanup
        actions.append("fix_normals")
        
        # Handle multiple components
        if features.num_components > 1:
            if features.component_imbalance > 0.5:
                actions.append("keep_largest")
            else:
                actions.append("blender_boolean_union")
        
        # Handle holes
        if features.hole_ratio > 0:
            actions.append("fill_holes")
        
        # Main repair
        if not features.is_manifold or not features.is_watertight:
            actions.append("pymeshfix_repair")
        
        # Final watertight check
        actions.append("make_watertight")
        
        return actions
    
    def get_statistics(self) -> Dict:
        """Get learning statistics."""
        tracker_stats = self.tracker.get_statistics()
        
        return {
            **tracker_stats,
            "model_updates": self.update_count,
            "device": str(self.device),
            "latent_dim": self.config.latent_dim,
        }
