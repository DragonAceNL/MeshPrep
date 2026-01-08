# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Replay Buffer for storing and sampling experiences.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import random


@dataclass
class Experience:
    """Single experience tuple."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float = 0.0
    value: float = 0.0


class ReplayBuffer:
    """
    Experience replay buffer for off-policy learning.
    """
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, experience: Experience):
        """Add experience to buffer."""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch of experiences."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()


class RolloutBuffer:
    """
    Buffer for on-policy algorithms (PPO, A2C).
    Stores complete episodes/rollouts.
    """
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.returns = []
        self.advantages = []
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ):
        """Add step to buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Compute returns and GAE advantages.
        
        Args:
            last_value: Value estimate of final state
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        self.returns = []
        self.advantages = []
        
        gae = 0
        next_value = last_value
        
        # Iterate backwards through buffer
        for t in reversed(range(len(self.rewards))):
            if self.dones[t]:
                next_value = 0
                gae = 0
            
            delta = self.rewards[t] + gamma * next_value - self.values[t]
            gae = delta + gamma * gae_lambda * gae
            
            self.advantages.insert(0, gae)
            self.returns.insert(0, gae + self.values[t])
            
            next_value = self.values[t]
    
    def get_batches(self, batch_size: int):
        """
        Yield random batches from buffer.
        """
        import torch
        
        indices = np.arange(len(self.states))
        np.random.shuffle(indices)
        
        for start in range(0, len(indices), batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            yield (
                torch.FloatTensor(np.array([self.states[i] for i in batch_indices])),
                torch.LongTensor([self.actions[i] for i in batch_indices]),
                torch.FloatTensor([self.returns[i] for i in batch_indices]),
                torch.FloatTensor([self.advantages[i] for i in batch_indices]),
                torch.FloatTensor([self.log_probs[i] for i in batch_indices]),
                torch.FloatTensor([self.values[i] for i in batch_indices]),
            )
    
    def clear(self):
        """Clear the buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.returns = []
        self.advantages = []
    
    def __len__(self) -> int:
        return len(self.states)
