# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""
Use trained ML model to predict and repair meshes.
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from pathlib import Path

from meshprep.core import Mesh, Pipeline
from meshprep.actions import trimesh, pymeshfix, blender, open3d, core


# Simple classifier (must match training)
class SimpleClassifier(nn.Module):
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


# Define pipelines
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


def load_model(model_path: Path):
    """Load trained model."""
    checkpoint = torch.load(model_path, map_location="cpu")
    pipeline_names = checkpoint["pipeline_names"]
    
    model = SimpleClassifier(input_dim=5, num_classes=len(pipeline_names))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    return model, pipeline_names


def predict_pipeline(model, pipeline_names, mesh: Mesh) -> str:
    """Predict best pipeline for a mesh."""
    with torch.no_grad():
        features = torch.tensor([
            mesh.metadata.vertex_count / 10000,
            mesh.metadata.face_count / 10000,
            mesh.metadata.body_count / 10,
            float(mesh.metadata.is_watertight),
            float(mesh.metadata.is_manifold),
        ], dtype=torch.float32).unsqueeze(0)
        
        outputs = model(features)
        probs = torch.softmax(outputs, dim=1).squeeze()
        
        # Top 3 predictions
        top_probs, top_indices = torch.topk(probs, 3)
        
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            predictions.append((pipeline_names[idx], prob.item()))
        
        return predictions


def ml_repair(mesh_path: Path, model_path: Path = None, output_path: Path = None, output_dir: Path = None):
    """
    Repair a mesh using ML-predicted pipeline.
    
    Args:
        mesh_path: Path to input mesh
        model_path: Path to trained model (default: models/simple_classifier.pt)
        output_path: Path for output mesh (overrides output_dir)
        output_dir: Directory for output (default: ./repaired/)
    """
    if model_path is None:
        model_path = Path("models/simple_classifier.pt")
    
    if output_path is None:
        if output_dir is None:
            output_dir = Path("repaired")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{mesh_path.stem}_repaired.stl"
    
    # Load model
    print(f"Loading model from {model_path}")
    model, pipeline_names = load_model(model_path)
    
    # Load mesh
    print(f"Loading mesh from {mesh_path}")
    mesh = Mesh.load(mesh_path)
    
    print(f"\n=== INPUT MESH ===")
    print(f"Vertices: {mesh.metadata.vertex_count:,}")
    print(f"Faces: {mesh.metadata.face_count:,}")
    print(f"Bodies: {mesh.metadata.body_count}")
    print(f"Watertight: {mesh.metadata.is_watertight}")
    
    # Predict pipeline
    print(f"\n=== ML PREDICTION ===")
    predictions = predict_pipeline(model, pipeline_names, mesh)
    for name, prob in predictions:
        print(f"  {name}: {prob:.1%}")
    
    best_pipeline_name = predictions[0][0]
    print(f"\nUsing pipeline: {best_pipeline_name}")
    
    # Run pipeline
    pipeline = PIPELINES[best_pipeline_name]
    result = pipeline.execute(mesh)
    
    if result.success and result.mesh:
        result.mesh._update_metadata_from_mesh()
        
        print(f"\n=== OUTPUT MESH ===")
        print(f"Vertices: {result.mesh.metadata.vertex_count:,}")
        print(f"Faces: {result.mesh.metadata.face_count:,}")
        print(f"Bodies: {result.mesh.metadata.body_count}")
        print(f"Watertight: {result.mesh.metadata.is_watertight}")
        print(f"Duration: {result.duration_ms:.1f}ms")
        
        # Save
        result.mesh.trimesh.export(output_path)
        print(f"\nSaved to: {output_path}")
        
        return result.mesh
    else:
        print(f"\nRepair failed: {result.error}")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ML-powered mesh repair")
    parser.add_argument("mesh", type=Path, help="Path to input mesh")
    parser.add_argument("--model", type=Path, default=Path("models/simple_classifier.pt"))
    parser.add_argument("--output", type=Path, default=None, help="Output file path")
    parser.add_argument("--output-dir", type=Path, default=Path("repaired"), help="Output directory (default: ./repaired/)")
    
    args = parser.parse_args()
    
    ml_repair(args.mesh, args.model, args.output, args.output_dir)
