# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""
Learning system for MeshPrep v5.

Tracks repair history and learns optimal strategies over time.
"""

__version__ = "5.0.0"

from .history_tracker import HistoryTracker
from .strategy_learner import StrategyLearner

__all__ = [
    "HistoryTracker",
    "StrategyLearner",
]
