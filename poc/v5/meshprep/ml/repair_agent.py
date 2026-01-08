# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Repair Agent - High-level interface for ML-based mesh repair.

This is the main public interface for the ML module.
Handles model loading, inference, and training.

Usage:
    from meshprep.ml import RepairAgent
    
    agent = RepairAgent()
    result = agent.repair("model.stl")
    print(f"Success: {result.success}, Actions: {result.actions}")
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union, Callable
import logging
import time

import numpy as np

from .agent import PPOAgent
from .environment import MeshRepairEnv, IDX_TO_ACTION

logger = logging.getLogger(__name__)


@dataclass
class RepairResult:
    """Result of repair operation."""
    success: bool
    mesh: object  # Repaired mesh
    actions: List[str]
    reward: float
    steps: int
    duration_ms: float
    is_printable: bool


class RepairAgent:
    """
    High-level interface for RL-based mesh repair.
    
    Example:
        agent = RepairAgent()
        
        # Repair a mesh
        result = agent.repair("broken.stl")
        if result.is_printable:
            result.mesh.trimesh.export("fixed.stl")
        
        # Train on dataset
        agent.train("path/to/meshes/", iterations=100)
    """
    
    DEFAULT_MODEL = Path(__file__).parent / "models" / "repair_agent.pt"
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: str = "auto",
    ):
        """
        Initialize repair agent.
        
        Args:
            model_path: Path to saved model (optional)
            device: "auto", "cuda", or "cpu"
        """
        self.agent = PPOAgent(device=device)
        self.model_path = model_path or self.DEFAULT_MODEL
        
        # Try to load existing model
        if self.model_path.exists():
            try:
                self.agent.load(self.model_path)
                logger.info(f"Loaded model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
        
        logger.info(f"RepairAgent ready (device={self.agent.device})")
    
    def repair(
        self,
        mesh_or_path: Union[str, Path, object],
        output_path: Optional[Path] = None,
        deterministic: bool = True,
        verbose: bool = False,
    ) -> RepairResult:
        """
        Repair a mesh using learned policy.
        
        Args:
            mesh_or_path: Mesh object, STL path, or any trimesh-loadable file
            output_path: Save repaired mesh here (optional)
            deterministic: Use greedy action selection (recommended)
            verbose: Print actions as they happen
            
        Returns:
            RepairResult with repaired mesh and stats
        """
        from meshprep.core import Mesh
        
        start = time.time()
        
        # Load mesh
        if isinstance(mesh_or_path, (str, Path)):
            mesh = Mesh.load(mesh_or_path)
        else:
            mesh = mesh_or_path
        
        # Run episode
        env = self.agent.env
        state = env.reset(mesh)
        
        total_reward = 0.0
        actions = []
        done = False
        
        while not done:
            action_idx, _, _ = self.agent.select_action(state, deterministic)
            action_name = IDX_TO_ACTION[action_idx]
            
            result = env.step(action_idx)
            
            if verbose and action_name != "STOP":
                logger.info(f"  {action_name} -> reward={result.reward:.2f}")
            
            if action_name != "STOP":
                actions.append(action_name)
            
            total_reward += result.reward
            state = result.state
            done = result.done
        
        duration = (time.time() - start) * 1000
        
        # Get final mesh
        final_mesh = env.mesh
        is_printable = env._is_printable()
        
        # Save if requested
        if output_path and final_mesh:
            final_mesh.trimesh.export(output_path)
            logger.info(f"Saved to {output_path}")
        
        return RepairResult(
            success=is_printable,
            mesh=final_mesh,
            actions=actions,
            reward=total_reward,
            steps=len(actions),
            duration_ms=duration,
            is_printable=is_printable,
        )
    
    def train(
        self,
        mesh_source: Union[Path, Callable],
        iterations: int = 100,
        meshes_per_iter: int = 4,
        epochs: int = 4,
        batch_size: int = 32,
        eval_interval: int = 20,
        save_interval: int = 50,
        verbose: bool = True,
    ):
        """
        Train the agent on meshes.
        
        Args:
            mesh_source: Directory with mesh files OR callable that returns meshes
            iterations: Number of training iterations
            meshes_per_iter: Meshes to collect per iteration
            epochs: PPO epochs per update
            batch_size: Mini-batch size
            eval_interval: How often to evaluate
            save_interval: How often to save
            verbose: Print progress
        """
        import trimesh
        from meshprep.core import Mesh
        
        # Setup mesh generator
        if isinstance(mesh_source, Path):
            files = list(mesh_source.glob("*.stl")) + list(mesh_source.glob("*.ctm"))
            if not files:
                raise ValueError(f"No meshes in {mesh_source}")
            
            def mesh_gen():
                path = files[np.random.randint(len(files))]
                try:
                    tm = trimesh.load(path, force='mesh')
                    if hasattr(tm, 'to_geometry'):
                        tm = tm.to_geometry()
                    return Mesh(tm)
                except Exception:
                    return self._make_broken_mesh()
        else:
            mesh_gen = mesh_source
        
        logger.info(f"Training for {iterations} iterations...")
        
        for i in range(iterations):
            # Collect experience
            meshes = [mesh_gen() for _ in range(meshes_per_iter)]
            rollout = self.agent.collect_rollout(meshes)
            
            # Update
            metrics = self.agent.update(rollout, epochs, batch_size)
            
            # Log
            if verbose and i % 10 == 0:
                logger.info(
                    f"Iter {i}: loss={metrics['loss']:.4f}, "
                    f"episodes={metrics['episodes']}"
                )
            
            # Evaluate
            if eval_interval and i % eval_interval == 0:
                success_rate = self._evaluate(mesh_gen, n=5)
                if verbose:
                    logger.info(f"  Eval: {success_rate*100:.0f}% success")
            
            # Save
            if save_interval and i % save_interval == 0:
                self.save()
        
        # Final save
        self.save()
        logger.info(f"Training complete. Episodes: {self.agent.total_episodes}")
    
    def _evaluate(self, mesh_gen: Callable, n: int = 5) -> float:
        """Quick evaluation on random meshes."""
        successes = 0
        for _ in range(n):
            result = self.repair(mesh_gen(), deterministic=True)
            if result.is_printable:
                successes += 1
        return successes / n
    
    def _make_broken_mesh(self):
        """Create a simple broken mesh for training."""
        import trimesh
        from meshprep.core import Mesh
        
        tm = trimesh.creation.icosphere(subdivisions=2)
        tm.faces = tm.faces[::2]
        return Mesh(tm)
    
    def save(self, path: Optional[Path] = None):
        """Save model."""
        save_path = path or self.model_path
        self.agent.save(save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load(self, path: Path):
        """Load model."""
        self.agent.load(path)
        logger.info(f"Model loaded from {path}")
    
    @property
    def stats(self) -> dict:
        """Get training statistics."""
        return {
            "episodes": self.agent.total_episodes,
            "steps": self.agent.total_steps,
            "device": str(self.agent.device),
        }
