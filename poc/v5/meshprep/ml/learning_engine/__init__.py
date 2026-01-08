# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Real ML Learning Engine for MeshPrep.

This module implements a true learning system that:
1. Encodes mesh geometry into rich feature vectors
2. Predicts optimal repair strategies
3. Learns from repair outcomes (feedback loop)
4. Improves predictions over time

Unlike rule-based systems, this actually trains neural networks
on mesh geometry and repair outcomes.

Quick Start:
    from meshprep.ml.learning_engine import SmartRepairEngine
    
    engine = SmartRepairEngine()
    result = engine.repair("model.stl")
    
    # Check results
    print(f"Quality: {result.quality_score}/5")
    print(f"Printable: {result.is_printable}")
    
    # Engine learns automatically!
    print(engine.get_statistics())
"""

from .mesh_encoder import MeshGeometryEncoder, MeshFeatures, NeuralMeshEncoder
from .repair_predictor import RepairPredictor, RepairPrediction, ACTIONS
from .outcome_tracker import OutcomeTracker, RepairOutcome
from .learning_loop import LearningLoop, TrainingConfig
from .fidelity import compute_fidelity_metrics, FidelityMetrics, quick_quality_check
from .smart_engine import SmartRepairEngine, SmartRepairResult, smart_repair

__all__ = [
    # Main interface
    "SmartRepairEngine",
    "SmartRepairResult",
    "smart_repair",
    
    # Core components
    "LearningLoop",
    "TrainingConfig",
    
    # Encoding
    "MeshGeometryEncoder",
    "MeshFeatures", 
    "NeuralMeshEncoder",
    
    # Prediction
    "RepairPredictor",
    "RepairPrediction",
    "ACTIONS",
    
    # Tracking
    "OutcomeTracker",
    "RepairOutcome",
    
    # Validation
    "compute_fidelity_metrics",
    "FidelityMetrics",
    "quick_quality_check",
]
