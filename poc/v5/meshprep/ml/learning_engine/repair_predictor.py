# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Repair Predictor - Neural network that predicts optimal repair strategies.

Given a mesh encoding, predicts:
1. Which actions to apply
2. In what order
3. With what parameters

This is trained via the learning loop from repair outcomes.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Available actions and their parameter ranges
ACTIONS = [
    "fix_normals",
    "remove_duplicates",
    "fill_holes",
    "make_watertight",
    "keep_largest",
    "smooth",
    "decimate",
    "pymeshfix_repair",
    "pymeshfix_clean",
    "blender_remesh",
    "blender_boolean_union",
    "poisson_reconstruction",
]

ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
IDX_TO_ACTION = {i: a for i, a in enumerate(ACTIONS)}


@dataclass
class RepairPrediction:
    """Predicted repair strategy."""
    actions: List[str]
    action_probs: List[float]
    parameters: Dict[str, Dict]
    confidence: float
    latent_vector: Optional[np.ndarray] = None


class RepairPredictor(nn.Module):
    """
    Predicts repair strategy from mesh encoding.
    
    Architecture:
    - Input: Mesh latent vector (128-dim)
    - Hidden layers for strategy reasoning
    - Multi-head output:
      - Action sequence (which actions to apply)
      - Action ordering (priority scores)
      - Parameter prediction (per-action parameters)
    """
    
    def __init__(
        self,
        latent_dim: int = 128,
        num_actions: int = len(ACTIONS),
        max_sequence_length: int = 6,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        self.max_sequence_length = max_sequence_length
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )
        
        # Action selection head (which actions to apply)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )
        
        # Action priority head (ordering)
        self.priority_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )
        
        # Parameter heads (one per action with parameters)
        self.param_heads = nn.ModuleDict({
            "smooth": nn.Linear(hidden_dim, 2),  # iterations, factor
            "decimate": nn.Linear(hidden_dim, 1),  # target_ratio
            "blender_remesh": nn.Linear(hidden_dim, 1),  # voxel_size
            "poisson_reconstruction": nn.Linear(hidden_dim, 1),  # depth
        })
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Predict repair strategy.
        
        Args:
            latent: (B, latent_dim) mesh encodings
            
        Returns:
            Tuple of:
            - action_logits: (B, num_actions) probability of applying each action
            - priorities: (B, num_actions) priority scores for ordering
            - params: Dict of parameter predictions
            - confidence: (B, 1) confidence in prediction
        """
        features = self.backbone(latent)
        
        action_logits = self.action_head(features)
        priorities = self.priority_head(features)
        
        params = {}
        for name, head in self.param_heads.items():
            params[name] = torch.sigmoid(head(features))  # Normalize to 0-1
        
        confidence = self.confidence_head(features)
        
        return action_logits, priorities, params, confidence
    
    def predict(self, latent: torch.Tensor, threshold: float = 0.5) -> RepairPrediction:
        """
        Get repair prediction from latent vector.
        
        Args:
            latent: (latent_dim,) or (1, latent_dim) mesh encoding
            threshold: Minimum probability to include action
            
        Returns:
            RepairPrediction with ordered actions and parameters
        """
        self.eval()
        
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        
        with torch.no_grad():
            action_logits, priorities, params, confidence = self.forward(latent)
            
            # Get action probabilities
            action_probs = torch.sigmoid(action_logits).squeeze(0).cpu().numpy()
            priority_scores = priorities.squeeze(0).cpu().numpy()
            
            # Select actions above threshold
            selected_indices = np.where(action_probs > threshold)[0]
            
            if len(selected_indices) == 0:
                # Fall back to top action
                selected_indices = [np.argmax(action_probs)]
            
            # Sort by priority
            sorted_indices = sorted(
                selected_indices,
                key=lambda i: priority_scores[i],
                reverse=True
            )[:self.max_sequence_length]
            
            # Build action list
            actions = [IDX_TO_ACTION[i] for i in sorted_indices]
            probs = [float(action_probs[i]) for i in sorted_indices]
            
            # Get parameters
            parameters = {}
            for action in actions:
                if action in params:
                    param_values = params[action].squeeze(0).cpu().numpy()
                    parameters[action] = self._decode_params(action, param_values)
            
            return RepairPrediction(
                actions=actions,
                action_probs=probs,
                parameters=parameters,
                confidence=float(confidence.item()),
                latent_vector=latent.squeeze(0).cpu().numpy(),
            )
    
    def _decode_params(self, action: str, values: np.ndarray) -> Dict:
        """Decode parameter values to actual parameters."""
        if action == "smooth":
            return {
                "iterations": int(1 + values[0] * 9),  # 1-10
                "factor": float(values[1] * 0.5),  # 0-0.5
            }
        elif action == "decimate":
            return {
                "target_ratio": float(0.1 + values[0] * 0.8),  # 0.1-0.9
            }
        elif action == "blender_remesh":
            return {
                "voxel_size": float(0.1 + values[0] * 1.9),  # 0.1-2.0
            }
        elif action == "poisson_reconstruction":
            return {
                "depth": int(6 + values[0] * 4),  # 6-10
            }
        return {}


class RepairPredictorEnsemble:
    """
    Ensemble of repair predictors for more robust predictions.
    
    Uses multiple models and aggregates their predictions.
    """
    
    def __init__(self, num_models: int = 3, **kwargs):
        self.models = [RepairPredictor(**kwargs) for _ in range(num_models)]
        self.num_models = num_models
    
    def predict(self, latent: torch.Tensor, threshold: float = 0.4) -> RepairPrediction:
        """Aggregate predictions from all models."""
        predictions = [m.predict(latent, threshold) for m in self.models]
        
        # Aggregate action votes
        action_votes = {}
        for pred in predictions:
            for action, prob in zip(pred.actions, pred.action_probs):
                if action not in action_votes:
                    action_votes[action] = []
                action_votes[action].append(prob)
        
        # Average probabilities
        action_avg_probs = {
            action: np.mean(probs)
            for action, probs in action_votes.items()
        }
        
        # Sort by average probability
        sorted_actions = sorted(
            action_avg_probs.keys(),
            key=lambda a: action_avg_probs[a],
            reverse=True
        )
        
        # Aggregate parameters (use first model's params)
        parameters = predictions[0].parameters
        
        # Average confidence
        avg_confidence = np.mean([p.confidence for p in predictions])
        
        return RepairPrediction(
            actions=sorted_actions[:6],
            action_probs=[action_avg_probs[a] for a in sorted_actions[:6]],
            parameters=parameters,
            confidence=float(avg_confidence),
            latent_vector=predictions[0].latent_vector,
        )
    
    def to(self, device):
        """Move all models to device."""
        for model in self.models:
            model.to(device)
        return self
    
    def train(self, mode: bool = True):
        """Set training mode for all models."""
        for model in self.models:
            model.train(mode)
        return self
    
    def eval(self):
        """Set eval mode for all models."""
        return self.train(False)
    
    def parameters(self):
        """Get all parameters for optimization."""
        for model in self.models:
            yield from model.parameters()
