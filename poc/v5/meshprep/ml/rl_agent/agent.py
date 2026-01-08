# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
RL Repair Agent - The main agent that learns to repair meshes.

Uses PPO (Proximal Policy Optimization) algorithm for stable training.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .environment import MeshRepairEnv, ACTIONS, IDX_TO_ACTION
from .policy_network import ActorCritic
from .replay_buffer import RolloutBuffer


@dataclass
class RepairResult:
    """Result of agent repairing a mesh."""
    success: bool
    mesh: Any
    actions_taken: List[str]
    total_reward: float
    num_steps: int
    is_printable: bool


class RepairAgent:
    """
    RL Agent that learns to repair meshes through experience.
    
    Uses Actor-Critic architecture with PPO updates.
    No hardcoded rules - everything learned from rewards.
    """
    
    def __init__(
        self,
        state_dim: int = 28,
        num_actions: int = len(ACTIONS),
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "auto",
        model_path: Optional[Path] = None,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for RL agent")
        
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        # Network
        self.network = ActorCritic(state_dim, num_actions, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Environment
        self.env = MeshRepairEnv()
        
        # Training stats
        self.total_episodes = 0
        self.total_steps = 0
        
        # Try load existing model
        if model_path and model_path.exists():
            self.load(model_path)
        
        logger.info(f"RepairAgent initialized on {self.device}")
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """
        Select action given current state.
        
        Returns:
            Tuple of (action_idx, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, entropy, value = self.network.get_action_and_value(
                state_tensor, deterministic
            )
        
        return action, log_prob.item(), value.item()
    
    def repair(
        self,
        mesh,
        deterministic: bool = True,
        verbose: bool = False,
    ) -> RepairResult:
        """
        Repair a mesh using learned policy.
        
        Args:
            mesh: Mesh to repair
            deterministic: Use greedy action selection
            verbose: Print progress
            
        Returns:
            RepairResult with repaired mesh and stats
        """
        state = self.env.reset(mesh)
        
        total_reward = 0.0
        done = False
        
        while not done:
            action_idx, _, _ = self.select_action(state, deterministic)
            
            result = self.env.step(action_idx)
            
            if verbose:
                print(f"  Action: {IDX_TO_ACTION[action_idx]} -> reward={result.reward:.2f}")
            
            total_reward += result.reward
            state = result.state
            done = result.done
        
        # Get final mesh
        final_mesh = self.env.current_mesh
        tm = final_mesh.trimesh
        is_printable = tm.is_watertight and tm.is_volume
        
        return RepairResult(
            success=is_printable,
            mesh=final_mesh,
            actions_taken=self.env.actions_taken,
            total_reward=total_reward,
            num_steps=self.env.current_step,
            is_printable=is_printable,
        )
    
    def collect_rollout(
        self,
        meshes: List,
        steps_per_mesh: int = 10,
    ) -> RolloutBuffer:
        """
        Collect experiences from multiple meshes.
        
        Args:
            meshes: List of meshes to train on
            steps_per_mesh: Max steps per mesh
            
        Returns:
            RolloutBuffer with collected experiences
        """
        buffer = RolloutBuffer()
        
        for mesh in meshes:
            state = self.env.reset(mesh)
            done = False
            
            while not done:
                action, log_prob, value = self.select_action(state, deterministic=False)
                
                result = self.env.step(action)
                
                buffer.push(
                    state=state,
                    action=action,
                    reward=result.reward,
                    done=result.done,
                    log_prob=log_prob,
                    value=value,
                )
                
                state = result.state
                done = result.done
            
            self.total_episodes += 1
        
        # Compute returns and advantages
        _, _, last_value = self.select_action(state, deterministic=True)
        buffer.compute_returns_and_advantages(last_value, self.gamma, self.gae_lambda)
        
        return buffer
    
    def update(self, buffer: RolloutBuffer, epochs: int = 4, batch_size: int = 64) -> Dict:
        """
        Update policy using PPO.
        
        Args:
            buffer: Rollout buffer with experiences
            epochs: Number of optimization epochs
            batch_size: Mini-batch size
            
        Returns:
            Training metrics
        """
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0
        
        for _ in range(epochs):
            for batch in buffer.get_batches(batch_size):
                states, actions, returns, advantages, old_log_probs, old_values = batch
                
                # Move to device
                states = states.to(self.device)
                actions = actions.to(self.device)
                returns = returns.to(self.device)
                advantages = advantages.to(self.device)
                old_log_probs = old_log_probs.to(self.device)
                
                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Get current policy outputs
                logits, values = self.network(states)
                values = values.squeeze(-1)
                
                # Calculate log probs
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values, returns)
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_updates += 1
        
        self.total_steps += len(buffer)
        
        return {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
        }
    
    def save(self, path: Path):
        """Save agent state."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
        }, path)
        logger.info(f"Agent saved to {path}")
    
    def load(self, path: Path):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_episodes = checkpoint.get("total_episodes", 0)
        self.total_steps = checkpoint.get("total_steps", 0)
        logger.info(f"Agent loaded from {path}")
