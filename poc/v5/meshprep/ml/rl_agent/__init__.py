# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Reinforcement Learning Agent for Mesh Repair.

This module implements a true RL-based repair system:
- State: Mesh features (geometry, topology, problems)
- Actions: Repair operations (20 available)
- Reward: Quality score + printability bonus
- Policy: Neural network that learns optimal sequences

No hardcoded rules - everything is learned from experience.
"""

from .environment import MeshRepairEnv
from .policy_network import PolicyNetwork, ValueNetwork
from .agent import RepairAgent
from .trainer import RLTrainer
from .replay_buffer import ReplayBuffer, Experience

__all__ = [
    "MeshRepairEnv",
    "PolicyNetwork",
    "ValueNetwork", 
    "RepairAgent",
    "RLTrainer",
    "ReplayBuffer",
    "Experience",
]
