# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Mesh Repair Environment - Gym-style RL environment.

Models mesh repair as a Markov Decision Process:
- State: Mesh features (16-dim vector)
- Actions: Repair operations (12) + STOP (1)
- Reward: +5 for printable, penalties for failures
- Terminal: Printable OR max steps reached
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from .encoder import MeshEncoder, MeshFeatures

logger = logging.getLogger(__name__)

# Action space
ACTIONS = [
    "fix_normals",
    "remove_duplicates", 
    "fill_holes",
    "make_watertight",
    "keep_largest",
    "smooth",
    "decimate",
    "pymeshfix_repair",
    "pymeshfix_clean",
    "blender_remesh",
    "blender_boolean_union",
    "poisson_reconstruction",
    "STOP",
]

ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
IDX_TO_ACTION = {i: a for i, a in enumerate(ACTIONS)}


@dataclass
class StepResult:
    """Result of environment step."""
    state: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]


class MeshRepairEnv:
    """
    RL Environment for mesh repair.
    
    The agent observes mesh features and selects repair actions.
    Goal: Make the mesh 3D-printable (watertight + manifold).
    """
    
    # Class constants
    STATE_DIM = 16
    NUM_ACTIONS = len(ACTIONS)
    
    def __init__(
        self,
        max_steps: int = 8,
        printable_reward: float = 10.0,
        step_cost: float = 0.1,
        failure_penalty: float = 1.0,
    ):
        self.max_steps = max_steps
        self.printable_reward = printable_reward
        self.step_cost = step_cost
        self.failure_penalty = failure_penalty
        
        self.encoder = MeshEncoder()
        self._registry = None
        
        # Episode state
        self.mesh = None
        self.original_mesh = None
        self.features: Optional[MeshFeatures] = None
        self.step_count = 0
        self.actions_taken: List[str] = []
    
    def _get_registry(self):
        """Lazy load action registry."""
        if self._registry is None:
            from meshprep.core.action import ActionRegistry
            from meshprep import actions
            self._registry = ActionRegistry
        return self._registry
    
    def reset(self, mesh) -> np.ndarray:
        """Start new episode with given mesh."""
        from meshprep.core import Mesh
        
        self.original_mesh = mesh.copy() if hasattr(mesh, 'copy') else Mesh(mesh.trimesh.copy())
        self.mesh = mesh.copy() if hasattr(mesh, 'copy') else Mesh(mesh.trimesh.copy())
        self.step_count = 0
        self.actions_taken = []
        
        self.features = self.encoder.encode(self.mesh)
        return self._get_state()
    
    def step(self, action_idx: int) -> StepResult:
        """Execute action and return result."""
        action = IDX_TO_ACTION[action_idx]
        
        if action == "STOP":
            return self._handle_stop()
        
        # Execute action
        success = self._execute(action)
        self.actions_taken.append(action)
        self.step_count += 1
        
        # Update features
        self.features = self.encoder.encode(self.mesh)
        
        # Calculate reward
        reward = self._compute_reward(success)
        
        # Check termination
        done = self._is_terminal()
        
        return StepResult(
            state=self._get_state(),
            reward=reward,
            done=done,
            info={
                "action": action,
                "success": success,
                "is_printable": self._is_printable(),
            }
        )
    
    def _get_state(self) -> np.ndarray:
        """Get current state vector."""
        state = self.features.to_vector()
        return np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
    
    def _execute(self, action: str) -> bool:
        """Execute repair action."""
        if len(self.mesh.trimesh.faces) == 0:
            return False
        
        params = self._get_params(action)
        
        try:
            registry = self._get_registry()
            result = registry.execute(action, self.mesh, params)
            
            new_mesh = result.mesh if hasattr(result, 'mesh') else result
            
            if new_mesh is None or len(new_mesh.trimesh.faces) == 0:
                return False
            
            self.mesh = new_mesh
            return True
            
        except Exception as e:
            logger.debug(f"Action {action} failed: {e}")
            return False
    
    def _get_params(self, action: str) -> Dict:
        """Get scale-aware parameters for action."""
        params = {}
        diag = max(self.features.bbox_diagonal, 1.0)
        
        if action == "blender_remesh":
            # Voxel size: ~1% of diagonal, clamped
            voxel = max(0.5, min(diag / 100, 100.0))
            params["voxel_size"] = voxel
            
        elif action == "smooth":
            params["iterations"] = 2
            params["factor"] = 0.3
            
        elif action == "decimate":
            params["target_ratio"] = 0.7
            
        elif action == "poisson_reconstruction":
            params["depth"] = 7
        
        return params
    
    def _compute_reward(self, action_success: bool) -> float:
        """Compute reward for current state."""
        reward = -self.step_cost
        
        if not action_success:
            reward -= self.failure_penalty
            return reward
        
        if self._is_printable():
            reward += self.printable_reward
        else:
            # Partial credit for progress
            if self.features.num_components == 1:
                reward += 0.3
            if self.features.hole_ratio < 0.1:
                reward += 0.2
        
        return reward
    
    def _is_printable(self) -> bool:
        """Check if mesh is 3D-printable."""
        tm = self.mesh.trimesh
        return tm.is_watertight and tm.is_volume
    
    def _is_terminal(self) -> bool:
        """Check if episode should end."""
        if self.step_count >= self.max_steps:
            return True
        if self._is_printable():
            return True
        return False
    
    def _handle_stop(self) -> StepResult:
        """Handle STOP action."""
        if self._is_printable():
            reward = self.printable_reward
        else:
            reward = -self.failure_penalty
        
        return StepResult(
            state=self._get_state(),
            reward=reward,
            done=True,
            info={"action": "STOP", "success": self._is_printable(), "is_printable": self._is_printable()}
        )
