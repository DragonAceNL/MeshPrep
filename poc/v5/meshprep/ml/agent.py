# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
PPO Agent - Proximal Policy Optimization for mesh repair.

Clean implementation of PPO with:
- Rollout collection
- GAE advantage estimation
- Clipped surrogate objective
- Value function loss
- Entropy bonus
"""

from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass
import logging

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .environment import MeshRepairEnv, IDX_TO_ACTION
from .policy import ActorCritic

logger = logging.getLogger(__name__)


@dataclass
class Rollout:
    """Collected experience for PPO update."""
    states: List[np.ndarray]
    actions: List[int]
    rewards: List[float]
    dones: List[bool]
    log_probs: List[float]
    values: List[float]
    
    def __len__(self):
        return len(self.states)


class PPOAgent:
    """
    PPO Agent for mesh repair.
    
    Learns optimal repair strategies through experience.
    """
    
    def __init__(
        self,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "auto",
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required")
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        self.network = ActorCritic().to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        self.env = MeshRepairEnv()
        
        # Stats
        self.total_episodes = 0
        self.total_steps = 0
        
        logger.info(f"PPOAgent on {self.device}")
    
    def select_action(self, state: np.ndarray, deterministic: bool = False):
        """Select action given state."""
        state_t = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.network.get_action(state_t, deterministic)
        
        return action, log_prob, value
    
    def collect_rollout(self, meshes: List) -> Rollout:
        """Collect experience from meshes."""
        rollout = Rollout([], [], [], [], [], [])
        
        for mesh in meshes:
            state = self.env.reset(mesh)
            done = False
            
            while not done:
                action, log_prob, value = self.select_action(state)
                
                result = self.env.step(action)
                
                rollout.states.append(state)
                rollout.actions.append(action)
                rollout.rewards.append(result.reward)
                rollout.dones.append(result.done)
                rollout.log_probs.append(log_prob)
                rollout.values.append(value)
                
                state = result.state
                done = result.done
            
            self.total_episodes += 1
        
        return rollout
    
    def compute_advantages(self, rollout: Rollout, last_value: float = 0.0):
        """Compute GAE advantages and returns."""
        rewards = rollout.rewards
        values = rollout.values
        dones = rollout.dones
        
        advantages = []
        returns = []
        
        gae = 0.0
        next_value = last_value
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0.0
                gae = 0.0
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
            
            next_value = values[t]
        
        return advantages, returns
    
    def update(self, rollout: Rollout, epochs: int = 4, batch_size: int = 32) -> Dict:
        """PPO update step."""
        # Compute advantages
        _, _, last_value = self.select_action(
            rollout.states[-1] if rollout.states else np.zeros(16),
            deterministic=True
        )
        advantages, returns = self.compute_advantages(rollout, last_value)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(rollout.states)).to(self.device)
        actions = torch.LongTensor(rollout.actions).to(self.device)
        old_log_probs = torch.FloatTensor(rollout.log_probs).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        # Training loop
        total_loss = 0.0
        num_updates = 0
        
        indices = np.arange(len(rollout))
        
        for _ in range(epochs):
            np.random.shuffle(indices)
            
            for start in range(0, len(indices), batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                
                # Get batch
                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_old_log_probs = old_log_probs[batch_idx]
                b_returns = returns_t[batch_idx]
                b_advantages = advantages_t[batch_idx]
                
                # Forward pass
                log_probs, values, entropy = self.network.evaluate(b_states, b_actions)
                
                # Policy loss (clipped)
                ratio = torch.exp(log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values, b_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                num_updates += 1
        
        self.total_steps += len(rollout)
        
        return {
            "loss": total_loss / max(num_updates, 1),
            "episodes": self.total_episodes,
            "steps": self.total_steps,
        }
    
    def save(self, path: Path):
        """Save agent."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "episodes": self.total_episodes,
            "steps": self.total_steps,
        }, path)
    
    def load(self, path: Path):
        """Load agent."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_episodes = checkpoint.get("episodes", 0)
        self.total_steps = checkpoint.get("steps", 0)
