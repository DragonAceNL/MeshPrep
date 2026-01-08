# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Policy Network - Actor-Critic neural network for PPO.

Shared backbone with separate heads for:
- Policy (actor): Action probabilities
- Value (critic): State value estimate
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .environment import MeshRepairEnv


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    
    Architecture:
        Input (16) -> Shared (128) -> Policy head (13)
                                   -> Value head (1)
    """
    
    def __init__(
        self,
        state_dim: int = MeshRepairEnv.STATE_DIM,
        num_actions: int = MeshRepairEnv.NUM_ACTIONS,
        hidden_dim: int = 128,
    ):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        
        # Policy head (actor)
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )
        
        # Value head (critic)
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        
        # Small init for policy output (more uniform initial policy)
        nn.init.orthogonal_(self.policy[-1].weight, gain=0.01)
        nn.init.zeros_(self.policy[-1].bias)
    
    def forward(self, state: torch.Tensor):
        """Forward pass returning logits and value."""
        features = self.shared(state)
        logits = self.policy(features)
        value = self.value(features)
        return logits, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """
        Sample action from policy.
        
        Returns: (action_idx, log_prob, value)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        
        return (
            action.item(),
            dist.log_prob(action).item(),
            value.squeeze(-1).item(),
        )
    
    def evaluate(self, states: torch.Tensor, actions: torch.Tensor):
        """
        Evaluate actions for PPO update.
        
        Returns: (log_probs, values, entropy)
        """
        logits, values = self.forward(states)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        
        return (
            dist.log_prob(actions),
            values.squeeze(-1),
            dist.entropy(),
        )
