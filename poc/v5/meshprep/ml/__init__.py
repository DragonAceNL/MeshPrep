# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
MeshPrep ML Module - Reinforcement Learning for Mesh Repair.

This module provides a clean RL-based system for learning optimal
mesh repair strategies. The agent learns through experience which
actions work best for different types of broken meshes.

Architecture:
    meshprep/ml/
    ├── __init__.py          # Public API
    ├── encoder.py           # Mesh feature extraction
    ├── environment.py       # RL environment
    ├── policy.py            # Neural network policy
    ├── agent.py             # RL agent (PPO)
    └── repair_agent.py      # High-level repair interface

Usage:
    from meshprep.ml import RepairAgent
    
    agent = RepairAgent()
    result = agent.repair("broken_model.stl")
    
    # Train on dataset
    agent.train(mesh_dir="path/to/meshes", iterations=1000)
"""

from .repair_agent import RepairAgent, RepairResult

__all__ = ["RepairAgent", "RepairResult"]
