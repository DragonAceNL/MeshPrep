# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""
ML components for MeshPrep v5.

Uses PyTorch + PyTorch3D for mesh encoding and prediction.
"""

__version__ = "5.0.0"

# Try to import ML components
try:
    from .encoder import MeshEncoder
    from .predictor import PipelinePredictor
    from .quality_scorer import QualityScorer
    
    __all__ = [
        "MeshEncoder",
        "PipelinePredictor",
        "QualityScorer",
    ]
    
    ML_AVAILABLE = True
    
except ImportError as e:
    ML_AVAILABLE = False
    
    class MLNotAvailable:
        """Placeholder when ML not available."""
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "ML components not available. Install dependencies:\n"
                "  pip install torch torchvision\n"
                "  conda install pytorch3d -c pytorch3d  # Optional but recommended"
            )
    
    # Placeholder classes
    MeshEncoder = MLNotAvailable
    PipelinePredictor = MLNotAvailable
    QualityScorer = MLNotAvailable
    
    __all__ = []


def check_ml_available() -> bool:
    """Check if ML components are available."""
    return ML_AVAILABLE
