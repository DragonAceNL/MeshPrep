# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Mesh Repair Environment - Gym-style environment for RL training.

The environment models mesh repair as a sequential decision process:
- State: Current mesh features (28-dim vector)
- Actions: Discrete repair operations + STOP action
- Reward: Based on quality improvement and printability
- Done: When mesh is printable OR max steps reached
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Available actions (must match action registry)
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
    "STOP",  # Special action to end episode
]

NUM_ACTIONS = len(ACTIONS)
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
IDX_TO_ACTION = {i: a for i, a in enumerate(ACTIONS)}


@dataclass
class StepResult:
    """Result of taking an action in the environment."""
    state: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]


class MeshRepairEnv:
    """
    Gym-style environment for mesh repair.
    
    The agent observes mesh features and selects repair actions.
    Episodes end when the mesh is printable or max steps reached.
    """
    
    def __init__(
        self,
        max_steps: int = 10,
        quality_reward_scale: float = 1.0,
        printable_bonus: float = 5.0,
        step_penalty: float = 0.1,
        failure_penalty: float = 2.0,
    ):
        self.max_steps = max_steps
        self.quality_reward_scale = quality_reward_scale
        self.printable_bonus = printable_bonus
        self.step_penalty = step_penalty
        self.failure_penalty = failure_penalty
        
        # State
        self.current_mesh = None
        self.original_mesh = None
        self.current_step = 0
        self.actions_taken = []
        self.encoder = None
        
        # Lazy import to avoid circular deps
        self._action_registry = None
    
    @property
    def state_dim(self) -> int:
        """Dimension of state vector."""
        return 28  # From MeshFeatures.to_vector()
    
    @property
    def num_actions(self) -> int:
        """Number of available actions."""
        return NUM_ACTIONS
    
    def _get_encoder(self):
        """Lazy load encoder."""
        if self.encoder is None:
            from ..learning_engine.mesh_encoder import MeshGeometryEncoder
            self.encoder = MeshGeometryEncoder()
        return self.encoder
    
    def _get_action_registry(self):
        """Lazy load action registry."""
        if self._action_registry is None:
            from meshprep.core.action import ActionRegistry
            from meshprep import actions  # Register actions
            self._action_registry = ActionRegistry
        return self._action_registry
    
    def reset(self, mesh) -> np.ndarray:
        """
        Reset environment with a new mesh.
        
        Args:
            mesh: Mesh object to repair
            
        Returns:
            Initial state observation
        """
        from meshprep.core import Mesh
        
        # Store original for comparison
        self.original_mesh = mesh.copy() if hasattr(mesh, 'copy') else Mesh(mesh.trimesh.copy())
        self.current_mesh = mesh.copy() if hasattr(mesh, 'copy') else Mesh(mesh.trimesh.copy())
        
        self.current_step = 0
        self.actions_taken = []
        
        # Get initial state
        state = self._get_state()
        
        return state
    
    def step(self, action_idx: int) -> StepResult:
        """
        Take an action in the environment.
        
        Args:
            action_idx: Index of action to take
            
        Returns:
            StepResult with new state, reward, done flag, and info
        """
        action_name = IDX_TO_ACTION[action_idx]
        
        # Check for STOP action
        if action_name == "STOP":
            return self._handle_stop()
        
        # Execute action
        success, error = self._execute_action(action_name)
        
        self.actions_taken.append(action_name)
        self.current_step += 1
        
        # Calculate reward
        reward = self._calculate_reward(success, error)
        
        # Check if done
        done = self._is_done()
        
        # Get new state
        state = self._get_state()
        
        info = {
            "action": action_name,
            "success": success,
            "error": error,
            "step": self.current_step,
            "actions_taken": self.actions_taken.copy(),
        }
        
        return StepResult(state, reward, done, info)
    
    def _get_state(self) -> np.ndarray:
        """Get current state observation."""
        encoder = self._get_encoder()
        try:
            features = encoder.encode(self.current_mesh)
            state = features.to_vector()
            # Handle NaN/Inf values
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
            return state
        except Exception as e:
            logger.warning(f"Could not encode state: {e}")
            return np.zeros(28, dtype=np.float32)
    
    def _execute_action(self, action_name: str) -> Tuple[bool, Optional[str]]:
        """Execute a repair action."""
        registry = self._get_action_registry()
        
        # Check if mesh is valid
        if len(self.current_mesh.trimesh.faces) == 0:
            return False, "Mesh has no faces"
        
        # Get scale-aware parameters
        params = self._get_action_params(action_name)
        
        try:
            result = registry.execute(action_name, self.current_mesh, params)
            
            # Extract mesh from result
            if hasattr(result, 'mesh'):
                new_mesh = result.mesh
            else:
                new_mesh = result
            
            # Validate result
            if new_mesh is None or len(new_mesh.trimesh.faces) == 0:
                return False, "Action produced empty mesh"
            
            self.current_mesh = new_mesh
            return True, None
            
        except Exception as e:
            logger.debug(f"Action {action_name} failed: {e}")
            return False, str(e)
    
    def _get_action_params(self, action_name: str) -> Dict:
        """Get scale-aware parameters for an action."""
        encoder = self._get_encoder()
        features = encoder.encode(self.current_mesh)
        
        params = {}
        
        if action_name == "blender_remesh":
            # Voxel size proportional to mesh size, with sensible bounds
            bbox_diag = max(features.bbox_diagonal, 1.0)
            voxel_size = bbox_diag / 100  # Coarser remesh for stability
            voxel_size = max(0.5, min(voxel_size, 50.0))  # Clamp to reasonable range
            params["voxel_size"] = voxel_size
        
        elif action_name == "smooth":
            params["iterations"] = 2
            params["factor"] = 0.3
        
        elif action_name == "decimate":
            params["target_ratio"] = 0.7  # Less aggressive
        
        elif action_name == "poisson_reconstruction":
            params["depth"] = 7  # Lower depth for speed
        
        return params
    
    def _calculate_reward(self, action_success: bool, error: Optional[str]) -> float:
        """Calculate reward for the current state."""
        reward = 0.0
        
        # Penalty for each step (encourage efficiency)
        reward -= self.step_penalty
        
        # Penalty for failed action
        if not action_success:
            reward -= self.failure_penalty
            return reward
        
        # Get current mesh quality
        tm = self.current_mesh.trimesh
        
        # Printability bonus
        if tm.is_watertight and tm.is_volume:
            reward += self.printable_bonus
            
            # Additional bonus for maintaining quality
            quality = self._compute_quality()
            reward += quality * self.quality_reward_scale
        
        # Partial progress rewards
        else:
            # Reward for reducing problems
            features = self._get_encoder().encode(self.current_mesh)
            
            if features.num_components == 1:
                reward += 0.5  # Single component is good
            
            if features.hole_ratio < 0.1:
                reward += 0.3  # Few holes is good
        
        return reward
    
    def _compute_quality(self) -> float:
        """Compute quality score comparing to original."""
        try:
            from ..learning_engine.fidelity import compute_fidelity_metrics
            metrics = compute_fidelity_metrics(self.original_mesh, self.current_mesh)
            return metrics.quality_score / 5.0  # Normalize to 0-1
        except Exception:
            return 0.5  # Default if can't compute
    
    def _is_done(self) -> bool:
        """Check if episode is done."""
        # Max steps reached
        if self.current_step >= self.max_steps:
            return True
        
        # Mesh is printable
        tm = self.current_mesh.trimesh
        if tm.is_watertight and tm.is_volume:
            return True
        
        return False
    
    def _handle_stop(self) -> StepResult:
        """Handle STOP action."""
        tm = self.current_mesh.trimesh
        
        # Reward based on final state
        if tm.is_watertight and tm.is_volume:
            reward = self.printable_bonus + self._compute_quality() * self.quality_reward_scale
        else:
            reward = -self.failure_penalty  # Penalize stopping without success
        
        state = self._get_state()
        
        info = {
            "action": "STOP",
            "success": tm.is_watertight and tm.is_volume,
            "step": self.current_step,
            "actions_taken": self.actions_taken.copy(),
        }
        
        return StepResult(state, reward, True, info)
    
    def get_valid_actions(self) -> List[int]:
        """Get list of valid action indices for current state."""
        # All actions are always valid (agent learns which are useful)
        return list(range(NUM_ACTIONS))
    
    def render(self) -> str:
        """Render current state as string."""
        tm = self.current_mesh.trimesh
        features = self._get_encoder().encode(self.current_mesh)
        
        return (
            f"Step {self.current_step}/{self.max_steps} | "
            f"Verts: {len(tm.vertices)} | "
            f"Faces: {len(tm.faces)} | "
            f"Components: {features.num_components} | "
            f"Watertight: {tm.is_watertight} | "
            f"Actions: {self.actions_taken}"
        )
