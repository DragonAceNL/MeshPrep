# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Policy and Value Networks for the RL Agent.

Actor-Critic architecture:
- Policy Network (Actor): Outputs action probabilities
- Value Network (Critic): Estimates state value
"""

import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class PolicyNetwork(nn.Module):
    """
    Policy network that outputs action probabilities.
    
    Given a state (mesh features), outputs probability distribution
    over actions. Uses softmax output for discrete action space.
    """
    
    def __init__(
        self,
        state_dim: int = 28,
        num_actions: int = 13,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.num_actions = num_actions
        
        # Network layers
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        
        # Smaller init for output layer (more uniform initial policy)
        nn.init.orthogonal_(self.network[-1].weight, gain=0.01)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: (B, state_dim) state observations
            
        Returns:
            (B, num_actions) action logits
        """
        return self.network(state)
    
    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            state: (state_dim,) or (1, state_dim) state observation
            deterministic: If True, take argmax instead of sampling
            
        Returns:
            Tuple of (action_idx, log_prob, entropy)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        
        dist = Categorical(probs)
        
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action.item(), log_prob, entropy
    
    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and entropy for given state-action pairs.
        
        Args:
            states: (B, state_dim) batch of states
            actions: (B,) batch of actions
            
        Returns:
            Tuple of (log_probs, entropy)
        """
        logits = self.forward(states)
        probs = F.softmax(logits, dim=-1)
        
        dist = Categorical(probs)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, entropy


class ValueNetwork(nn.Module):
    """
    Value network that estimates state value.
    
    Given a state, outputs estimated cumulative reward.
    Used as baseline to reduce variance in policy gradient.
    """
    
    def __init__(
        self,
        state_dim: int = 28,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: (B, state_dim) state observations
            
        Returns:
            (B, 1) value estimates
        """
        return self.network(state)


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network with shared features.
    
    More efficient than separate networks when features are similar.
    """
    
    def __init__(
        self,
        state_dim: int = 28,
        num_actions: int = 13,
        hidden_dim: int = 256,
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
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        
        # Smaller init for policy output
        nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.01)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            Tuple of (action_logits, value)
        """
        features = self.shared(state)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value
    
    def get_action_and_value(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log prob, entropy, and value in one forward pass.
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
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action.item(), log_prob, entropy, value.squeeze(-1)
