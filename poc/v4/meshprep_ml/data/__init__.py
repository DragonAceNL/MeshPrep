# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Dataset and data preparation for training ML models.

This module loads repair history from POC v3 and prepares it for ML training.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import trimesh

logger = logging.getLogger(__name__)


class MeshRepairDataset(Dataset):
    """
    PyTorch dataset for mesh repair prediction.
    
    Loads meshes from POC v3 repair history and their outcomes.
    """
    
    def __init__(
        self,
        data_dir: Path,
        learning_db_path: Path,
        num_points: int = 2048,
        split: str = "train",
        train_ratio: float = 0.8,
    ):
        """
        Args:
            data_dir: Directory containing original STL files
            learning_db_path: Path to meshprep_learning.db from POC v3
            num_points: Number of points to sample from each mesh
            split: "train" or "val"
            train_ratio: Fraction of data for training
        """
        self.data_dir = Path(data_dir)
        self.learning_db_path = Path(learning_db_path)
        self.num_points = num_points
        self.split = split
        
        # Load repair history from database
        self.samples = self._load_repair_history()
        
        # Train/val split
        np.random.shuffle(self.samples)
        split_idx = int(len(self.samples) * train_ratio)
        if split == "train":
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
        
        logger.info(f"Loaded {len(self.samples)} samples for {split}")
    
    def _load_repair_history(self) -> List[Dict]:
        """Load repair history from POC v3 database."""
        samples = []
        
        with sqlite3.connect(self.learning_db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get all successfully repaired models
            cursor.execute("""
                SELECT 
                    m.model_id,
                    m.fingerprint,
                    m.profile,
                    m.winning_pipeline,
                    m.total_attempts,
                    m.faces_before,
                    m.body_count
                FROM model_results m
                WHERE m.success = 1
                  AND m.winning_pipeline IS NOT NULL
            """)
            
            for row in cursor.fetchall():
                samples.append({
                    "model_id": row["model_id"],
                    "fingerprint": row["fingerprint"],
                    "profile": row["profile"],
                    "winning_pipeline": row["winning_pipeline"],
                    "total_attempts": row["total_attempts"],
                    "faces_before": row["faces_before"],
                    "body_count": row["body_count"],
                })
        
        return samples
    
    def _load_mesh(self, model_id: str) -> Optional[trimesh.Trimesh]:
        """Load mesh from file."""
        # Try common locations
        search_paths = [
            self.data_dir / f"{model_id}.stl",
            self.data_dir / f"{model_id}.ctm",
            self.data_dir / "raw_meshes" / f"{model_id}.stl",
            self.data_dir / "raw_meshes" / f"{model_id}.ctm",
        ]
        
        for path in search_paths:
            if path.exists():
                try:
                    mesh = trimesh.load(str(path), force='mesh')
                    if isinstance(mesh, trimesh.Scene):
                        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
                    return mesh
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")
        
        return None
    
    def _sample_points(self, mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample points uniformly from mesh surface.
        
        Returns:
            points: (num_points, 3) array
            normals: (num_points, 3) array
        """
        try:
            points, face_indices = trimesh.sample.sample_surface(mesh, self.num_points)
            normals = mesh.face_normals[face_indices]
            return points, normals
        except Exception:
            # Fallback: use vertices if sampling fails
            vertices = mesh.vertices
            if len(vertices) >= self.num_points:
                indices = np.random.choice(len(vertices), self.num_points, replace=False)
            else:
                indices = np.random.choice(len(vertices), self.num_points, replace=True)
            
            points = vertices[indices]
            # Approximate normals
            try:
                normals = mesh.vertex_normals[indices]
            except:
                normals = np.zeros_like(points)
            
            return points, normals
    
    def _compute_stats(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Compute mesh statistics for model input.
        
        Returns:
            stats: (10,) array of mesh statistics
        """
        try:
            is_watertight = float(mesh.is_watertight)
            is_manifold = float(mesh.is_volume)
            face_count_log = np.log10(max(len(mesh.faces), 1))
            vertex_count_log = np.log10(max(len(mesh.vertices), 1))
            
            components = mesh.split(only_watertight=False)
            body_count_log = np.log10(len(components))
            
            try:
                volume_log = np.log10(max(abs(mesh.volume), 1e-6))
            except:
                volume_log = 0.0
            
            try:
                bbox = mesh.bounds
                bbox_diag = np.linalg.norm(bbox[1] - bbox[0])
                bbox_diag_log = np.log10(max(bbox_diag, 1e-6))
            except:
                bbox_diag_log = 0.0
            
            # Degenerate face ratio
            try:
                face_areas = mesh.area_faces
                degenerate_ratio = np.sum(face_areas < 1e-10) / len(face_areas)
            except:
                degenerate_ratio = 0.0
            
            # Edge length statistics
            try:
                edges = mesh.edges_unique_length
                avg_edge_log = np.log10(max(np.mean(edges), 1e-6))
                edge_std_log = np.log10(max(np.std(edges), 1e-6))
            except:
                avg_edge_log = 0.0
                edge_std_log = 0.0
            
            stats = np.array([
                is_watertight,
                is_manifold,
                face_count_log,
                vertex_count_log,
                body_count_log,
                volume_log,
                bbox_diag_log,
                degenerate_ratio,
                avg_edge_log,
                edge_std_log,
            ], dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Failed to compute stats: {e}")
            stats = np.zeros(10, dtype=np.float32)
        
        return stats
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            dict with:
                - points: (num_points, 3) surface points
                - normals: (num_points, 3) surface normals
                - stats: (10,) mesh statistics
                - pipeline_id: scalar pipeline index
                - quality_score: scalar quality (1-5)
                - model_id: string identifier
        """
        sample = self.samples[idx]
        
        # Load mesh
        mesh = self._load_mesh(sample["model_id"])
        if mesh is None:
            # Return dummy data if mesh not found
            logger.warning(f"Mesh {sample['model_id']} not found, using dummy data")
            return {
                "points": torch.zeros(self.num_points, 3),
                "normals": torch.zeros(self.num_points, 3),
                "stats": torch.zeros(10),
                "pipeline_id": torch.tensor(0),
                "quality_score": torch.tensor(3.0),
                "model_id": sample["model_id"],
            }
        
        # Sample points and normals
        points, normals = self._sample_points(mesh)
        
        # Compute statistics
        stats = self._compute_stats(mesh)
        
        # Get pipeline ID (need to map pipeline name to index)
        # For now, use a simple hash-based mapping
        pipeline_id = hash(sample["winning_pipeline"]) % 50  # Assume 50 pipelines
        
        # Get quality score (from quality_feedback.db if available, else estimate)
        quality_score = self._get_quality_score(sample)
        
        return {
            "points": torch.from_numpy(points).float(),
            "normals": torch.from_numpy(normals).float(),
            "stats": torch.from_numpy(stats).float(),
            "pipeline_id": torch.tensor(pipeline_id, dtype=torch.long),
            "quality_score": torch.tensor(quality_score, dtype=torch.float),
            "model_id": sample["model_id"],
        }
    
    def _get_quality_score(self, sample: Dict) -> float:
        """Get quality score for a sample."""
        # Try to load from quality_feedback.db
        quality_db = self.learning_db_path.parent / "quality_feedback.db"
        if quality_db.exists():
            try:
                with sqlite3.connect(quality_db) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT rating_value FROM quality_ratings
                        WHERE model_fingerprint = ?
                        ORDER BY rated_at DESC LIMIT 1
                    """, (sample["fingerprint"],))
                    row = cursor.fetchone()
                    if row:
                        return float(row[0])
            except Exception as e:
                logger.debug(f"Could not load quality score: {e}")
        
        # Estimate from repair attempts (fewer attempts = better quality)
        attempts = sample.get("total_attempts", 1)
        if attempts == 1:
            return 5.0  # Perfect - worked first try
        elif attempts <= 3:
            return 4.0  # Good
        elif attempts <= 5:
            return 3.0  # Acceptable
        else:
            return 2.0  # Poor - needed many attempts


def create_dataloaders(
    data_dir: Path,
    learning_db_path: Path,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir: Directory with mesh files
        learning_db_path: Path to meshprep_learning.db
        batch_size: Batch size
        num_workers: Number of data loading workers
        
    Returns:
        train_loader, val_loader
    """
    train_dataset = MeshRepairDataset(
        data_dir=data_dir,
        learning_db_path=learning_db_path,
        split="train",
    )
    
    val_dataset = MeshRepairDataset(
        data_dir=data_dir,
        learning_db_path=learning_db_path,
        split="val",
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader
