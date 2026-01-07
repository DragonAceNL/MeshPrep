# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
MeshPrep POC v4: ML-Based Mesh Repair
======================================

This POC explores using PyTorch + PyTorch3D for learning-based mesh repair
strategy selection and quality prediction.

Key Innovations
---------------
1. **Neural Pipeline Selector**: Learns which pipeline to try first
2. **Quality Predictor**: Predicts repair quality before execution
3. **Mesh Encoder**: Embeds mesh geometry into learned feature space
4. **Failure Predictor**: Learns which actions crash/fail on which meshes

Compared to POC v2/v3
---------------------
- POC v2: Rule-based repair with manual pipeline ordering
- POC v3: Statistical learning from repair history
- POC v4: Deep learning on mesh geometry for smart predictions

Architecture
------------
Input: 3D Mesh (vertices, faces, normals)
  ?
Mesh Encoder (PointNet++ / MeshCNN)
  ?
Latent Representation (256D vector)
  ?
Multi-Task Heads:
  - Pipeline Selector (classification)
  - Quality Predictor (regression)
  - Failure Predictor (binary)
  ?
Repair Strategy + Expected Quality

Training Data
-------------
Uses repair history from POC v3:
- 10,000+ repaired models from Thingi10K
- Quality scores (auto + manual)
- Success/failure per pipeline
- Geometry changes (Hausdorff, volume, etc.)
"""

__version__ = "0.1.0"
__author__ = "Allard Peper (Dragon Ace)"

from .models import MeshEncoder, PipelineSelector, QualityPredictor
from .data import MeshDataset, prepare_training_data
from .training import train_model, evaluate_model

__all__ = [
    "MeshEncoder",
    "PipelineSelector", 
    "QualityPredictor",
    "MeshDataset",
    "prepare_training_data",
    "train_model",
    "evaluate_model",
]
