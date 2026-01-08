# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""
Train ML model on Thingi10K data.

This script:
1. Loads meshes from Thingi10K
2. Tries different repair pipelines on each
3. Records which pipeline worked best
4. Trains the PipelinePredictor on this data
"""

import sys
sys.path.insert(0, '.')

import os
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed. Run: pip install torch")
    sys.exit(1)

from meshprep.core import Mesh, Pipeline
from meshprep.actions import trimesh, pymeshfix, blender, open3d, core


# Define pipelines to test
PIPELINES = {
    "cleanup": Pipeline(
        name="cleanup",
        actions=[
            {"name": "remove_duplicates"},
            {"name": "fix_normals"},
            {"name": "fill_holes"},
        ]
    ),
    "standard": Pipeline(
        name="standard",
        actions=[
            {"name": "remove_duplicates"},
            {"name": "fix_normals"},
            {"name": "fill_holes"},
            {"name": "pymeshfix_repair"},
        ]
    ),
    "aggressive": Pipeline(
        name="aggressive",
        actions=[
            {"name": "pymeshfix_clean"},
            {"name": "fill_holes"},
            {"name": "pymeshfix_repair"},
            {"name": "make_watertight"},
        ]
    ),
    "fragments": Pipeline(
        name="fragments",
        actions=[
            {"name": "keep_largest"},
            {"name": "fix_normals"},
            {"name": "fill_holes"},
            {"name": "pymeshfix_repair"},
        ]
    ),
    "reconstruction": Pipeline(
        name="reconstruction",
        actions=[
            {"name": "poisson_reconstruction", "params": {"depth": 8}},
            {"name": "smooth"},
        ]
    ),
}


def evaluate_repair(original: Mesh, repaired: Mesh) -> float:
    """Score how well the repair worked (0-1)."""
    if repaired is None:
        return 0.0
    
    score = 0.0
    
    # Watertight is most important
    if repaired.metadata.is_watertight:
        score += 0.5
    
    # Single body is good
    if repaired.metadata.body_count == 1:
        score += 0.3
    
    # Preserved faces (not too much loss)
    if repaired.metadata.face_count > 0:
        face_ratio = repaired.metadata.face_count / max(original.metadata.face_count, 1)
        if 0.5 < face_ratio < 2.0:  # Reasonable change
            score += 0.2
    
    return score


def collect_training_data(
    mesh_dir: Path,
    max_meshes: int = 100,
    output_file: Path = None,
) -> List[Dict]:
    """
    Collect training data by testing pipelines on real meshes.
    
    Returns list of {mesh_path, features, best_pipeline, scores}
    """
    mesh_files = list(mesh_dir.glob("*.stl"))[:max_meshes]
    logger.info(f"Found {len(mesh_files)} mesh files")
    
    training_data = []
    
    for i, mesh_path in enumerate(mesh_files):
        logger.info(f"[{i+1}/{len(mesh_files)}] Processing {mesh_path.name}")
        
        try:
            mesh = Mesh.load(mesh_path)
            
            # Extract features
            features = {
                "vertex_count": mesh.metadata.vertex_count,
                "face_count": mesh.metadata.face_count,
                "body_count": mesh.metadata.body_count,
                "is_watertight": mesh.metadata.is_watertight,
                "is_manifold": mesh.metadata.is_manifold,
            }
            
            # Test each pipeline
            scores = {}
            for name, pipeline in PIPELINES.items():
                try:
                    result = pipeline.execute(mesh.copy())
                    if result.success and result.mesh:
                        result.mesh._update_metadata_from_mesh()
                        scores[name] = evaluate_repair(mesh, result.mesh)
                    else:
                        scores[name] = 0.0
                except Exception as e:
                    logger.warning(f"  Pipeline {name} failed: {e}")
                    scores[name] = 0.0
            
            # Find best pipeline
            best_pipeline = max(scores, key=scores.get)
            best_score = scores[best_pipeline]
            
            logger.info(f"  Best: {best_pipeline} (score={best_score:.2f})")
            
            training_data.append({
                "mesh_path": str(mesh_path),
                "features": features,
                "best_pipeline": best_pipeline,
                "scores": scores,
            })
            
        except Exception as e:
            logger.error(f"  Error loading mesh: {e}")
    
    # Save training data
    if output_file:
        with open(output_file, "w") as f:
            json.dump(training_data, f, indent=2)
        logger.info(f"Saved training data to {output_file}")
    
    return training_data


class MeshFeatureDataset(Dataset):
    """Dataset for training from pre-computed features."""
    
    def __init__(self, training_data: List[Dict], pipeline_names: List[str]):
        self.data = training_data
        self.pipeline_names = pipeline_names
        self.pipeline_to_idx = {name: i for i, name in enumerate(pipeline_names)}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Simple feature vector (could be improved with actual point cloud)
        features = torch.tensor([
            item["features"]["vertex_count"] / 10000,  # Normalize
            item["features"]["face_count"] / 10000,
            item["features"]["body_count"] / 10,
            float(item["features"]["is_watertight"]),
            float(item["features"]["is_manifold"]),
        ], dtype=torch.float32)
        
        label = self.pipeline_to_idx[item["best_pipeline"]]
        
        return features, label


class SimpleClassifier(nn.Module):
    """Simple classifier for testing (uses features, not point cloud)."""
    
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_simple_model(
    training_data: List[Dict],
    epochs: int = 50,
    output_path: Path = None,
):
    """Train a simple feature-based classifier."""
    
    pipeline_names = list(PIPELINES.keys())
    dataset = MeshFeatureDataset(training_data, pipeline_names)
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Create model
    model = SimpleClassifier(input_dim=5, num_classes=len(pipeline_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    logger.info(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)}")
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total if total > 0 else 0
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}, Val Acc: {accuracy:.1f}%")
    
    # Save model
    if output_path:
        torch.save({
            "model_state": model.state_dict(),
            "pipeline_names": pipeline_names,
        }, output_path)
        logger.info(f"Saved model to {output_path}")
    
    return model, pipeline_names


def main():
    """Main training workflow."""
    
    # Paths
    thingi_dir = Path(r"C:\Users\Dragon Ace\Source\repos\Thingi10K\raw_meshes")
    output_dir = Path(r"C:\Users\Dragon Ace\Source\repos\MeshPrep\poc\v5\models")
    output_dir.mkdir(exist_ok=True)
    
    training_data_file = output_dir / "training_data.json"
    model_file = output_dir / "simple_classifier.pt"
    
    # Step 1: Collect training data (or load if exists)
    if training_data_file.exists():
        logger.info("Loading existing training data...")
        with open(training_data_file) as f:
            training_data = json.load(f)
    else:
        logger.info("Collecting training data from Thingi10K...")
        training_data = collect_training_data(
            thingi_dir,
            max_meshes=20,  # Start small
            output_file=training_data_file,
        )
    
    if len(training_data) < 10:
        logger.error("Not enough training data!")
        return
    
    # Step 2: Train model
    logger.info("Training model...")
    model, pipeline_names = train_simple_model(
        training_data,
        epochs=50,
        output_path=model_file,
    )
    
    logger.info("Training complete!")
    logger.info(f"Model saved to: {model_file}")
    
    # Step 3: Test prediction
    logger.info("\nTesting prediction on a new mesh...")
    test_mesh = Mesh.load(thingi_dir / "100026.stl")
    
    model.eval()
    with torch.no_grad():
        features = torch.tensor([
            test_mesh.metadata.vertex_count / 10000,
            test_mesh.metadata.face_count / 10000,
            test_mesh.metadata.body_count / 10,
            float(test_mesh.metadata.is_watertight),
            float(test_mesh.metadata.is_manifold),
        ], dtype=torch.float32).unsqueeze(0)
        
        outputs = model(features)
        probs = torch.softmax(outputs, dim=1).squeeze()
        
        # Top 3 predictions
        top_probs, top_indices = torch.topk(probs, 3)
        
        print("\nPredictions for 100026.stl:")
        for prob, idx in zip(top_probs, top_indices):
            print(f"  {pipeline_names[idx]}: {prob.item():.1%}")


if __name__ == "__main__":
    main()

