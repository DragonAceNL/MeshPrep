# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
RL Trainer - Training loop for the repair agent.
"""

import numpy as np
from typing import List, Dict, Optional, Callable
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)

from .agent import RepairAgent


class RLTrainer:
    """
    Trainer for the RL repair agent.
    
    Handles:
    - Training loop with mesh batches
    - Logging and metrics
    - Model checkpointing
    - Evaluation
    """
    
    def __init__(
        self,
        agent: RepairAgent,
        save_dir: Path = Path("models/rl_agent"),
        save_interval: int = 100,
        eval_interval: int = 50,
    ):
        self.agent = agent
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        
        # Metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate_history = []
    
    def train(
        self,
        mesh_generator: Callable,
        num_iterations: int = 1000,
        meshes_per_iteration: int = 4,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        eval_meshes: Optional[List] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Train the agent.
        
        Args:
            mesh_generator: Function that yields meshes for training
            num_iterations: Number of training iterations
            meshes_per_iteration: Meshes to collect per iteration
            ppo_epochs: PPO update epochs
            batch_size: Mini-batch size
            eval_meshes: Meshes for evaluation
            verbose: Print progress
            
        Returns:
            Training statistics
        """
        logger.info(f"Starting training for {num_iterations} iterations")
        
        start_time = time.time()
        
        for iteration in range(num_iterations):
            # Collect meshes
            meshes = [mesh_generator() for _ in range(meshes_per_iteration)]
            
            # Collect rollout
            buffer = self.agent.collect_rollout(meshes)
            
            # Track rewards
            for i in range(len(buffer.rewards)):
                if buffer.dones[i]:
                    # End of episode
                    pass
            
            # Update policy
            metrics = self.agent.update(buffer, epochs=ppo_epochs, batch_size=batch_size)
            
            # Logging
            if verbose and iteration % 10 == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"Iter {iteration}/{num_iterations} | "
                    f"Episodes: {metrics['total_episodes']} | "
                    f"Policy Loss: {metrics['policy_loss']:.4f} | "
                    f"Value Loss: {metrics['value_loss']:.4f} | "
                    f"Entropy: {metrics['entropy']:.4f} | "
                    f"Time: {elapsed:.1f}s"
                )
            
            # Evaluation
            if eval_meshes and iteration % self.eval_interval == 0:
                eval_results = self.evaluate(eval_meshes)
                self.success_rate_history.append(eval_results["success_rate"])
                
                if verbose:
                    logger.info(
                        f"  Eval: Success={eval_results['success_rate']*100:.1f}% | "
                        f"Avg Reward={eval_results['avg_reward']:.2f} | "
                        f"Avg Steps={eval_results['avg_steps']:.1f}"
                    )
            
            # Save checkpoint
            if iteration % self.save_interval == 0:
                self.agent.save(self.save_dir / "checkpoint.pt")
        
        # Final save
        self.agent.save(self.save_dir / "final.pt")
        
        total_time = time.time() - start_time
        
        return {
            "total_episodes": self.agent.total_episodes,
            "total_steps": self.agent.total_steps,
            "training_time": total_time,
            "success_rate_history": self.success_rate_history,
        }
    
    def evaluate(self, meshes: List, verbose: bool = False) -> Dict:
        """
        Evaluate agent on test meshes.
        
        Args:
            meshes: List of test meshes
            verbose: Print per-mesh results
            
        Returns:
            Evaluation metrics
        """
        successes = 0
        total_reward = 0.0
        total_steps = 0
        
        for mesh in meshes:
            result = self.agent.repair(mesh, deterministic=True, verbose=verbose)
            
            if result.is_printable:
                successes += 1
            
            total_reward += result.total_reward
            total_steps += result.num_steps
        
        return {
            "success_rate": successes / len(meshes),
            "avg_reward": total_reward / len(meshes),
            "avg_steps": total_steps / len(meshes),
            "num_meshes": len(meshes),
        }
    
    def train_on_thingi10k(
        self,
        thingi_dir: Path,
        num_iterations: int = 500,
        verbose: bool = True,
    ) -> Dict:
        """
        Convenience method to train on Thingi10K dataset.
        """
        import trimesh
        from meshprep.core import Mesh
        
        # Get all mesh files
        mesh_files = list(thingi_dir.glob("*.stl")) + list(thingi_dir.glob("*.ctm"))
        
        if not mesh_files:
            raise ValueError(f"No mesh files found in {thingi_dir}")
        
        logger.info(f"Found {len(mesh_files)} meshes in {thingi_dir}")
        
        def mesh_generator():
            """Generate random mesh from dataset."""
            idx = np.random.randint(len(mesh_files))
            path = mesh_files[idx]
            
            try:
                tm = trimesh.load(path, force='mesh')
                if isinstance(tm, trimesh.Scene):
                    tm = tm.to_geometry()
                return Mesh(tm)
            except Exception:
                # Fallback to simple broken mesh
                tm = trimesh.creation.icosphere()
                tm.faces = tm.faces[::2]  # Remove half faces
                return Mesh(tm)
        
        # Create eval set
        eval_meshes = []
        for _ in range(10):
            eval_meshes.append(mesh_generator())
        
        return self.train(
            mesh_generator=mesh_generator,
            num_iterations=num_iterations,
            eval_meshes=eval_meshes,
            verbose=verbose,
        )
