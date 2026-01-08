# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Test the Smart ML Repair Engine.

This script demonstrates the learning ML system:
1. Repairs meshes using neural network predictions
2. Records outcomes for learning
3. Trains the model
4. Shows improvement over time
"""

import sys
sys.path.insert(0, '.')

import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_smart_engine():
    """Test the SmartRepairEngine."""
    
    print("=" * 60)
    print("SMART ML REPAIR ENGINE TEST")
    print("=" * 60)
    
    # Import after path setup
    from meshprep.ml.learning_engine import SmartRepairEngine, TrainingConfig
    from meshprep.core import Mesh
    
    # Configuration
    config = TrainingConfig(
        latent_dim=128,
        learning_rate=1e-3,
        min_samples_to_train=5,  # Low for testing
        model_dir=Path("models/smart_engine"),
    )
    
    # Create engine
    print("\n1. Creating SmartRepairEngine...")
    engine = SmartRepairEngine(
        config=config,
        device="auto",
        auto_train=True,
        train_interval=5,
    )
    
    print(f"   Device: {engine.learning_loop.device}")
    print(f"   Initial stats: {engine.get_statistics()}")
    
    # Test meshes from Thingi10K
    thingi_dir = Path(r"C:\Users\Dragon Ace\Source\repos\Thingi10K\raw_meshes")
    output_dir = Path("repaired/smart")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get some test meshes
    test_meshes = list(thingi_dir.glob("*.stl"))[:10]
    
    if not test_meshes:
        print("\nNo test meshes found in Thingi10K directory")
        print("Testing with a generated mesh instead...")
        
        import trimesh
        mesh = Mesh(trimesh.creation.icosphere())
        # Damage it
        mesh.trimesh.faces = mesh.trimesh.faces[::2]  # Remove half the faces
        
        print("\n2. Testing repair on damaged icosphere...")
        result = engine.repair(mesh)
        
        print(f"   Success: {result.success}")
        print(f"   Quality: {result.quality_score}/5")
        print(f"   Printable: {result.is_printable}")
        print(f"   Actions: {result.actions}")
        print(f"   Confidence: {result.prediction_confidence:.2f}")
        print(f"   Duration: {result.duration_ms:.1f}ms")
        
        return
    
    print(f"\n2. Testing repair on {len(test_meshes)} meshes from Thingi10K...")
    print("-" * 60)
    
    results = []
    for i, mesh_path in enumerate(test_meshes):
        print(f"\n[{i+1}/{len(test_meshes)}] {mesh_path.name}")
        
        try:
            output_path = output_dir / f"{mesh_path.stem}_repaired.stl"
            result = engine.repair(mesh_path, output_path)
            
            results.append(result)
            
            status = "✓" if result.is_printable else "✗"
            print(f"   {status} Quality: {result.quality_score}/5 | "
                  f"Printable: {result.is_printable} | "
                  f"Actions: {len(result.actions)} | "
                  f"Confidence: {result.prediction_confidence:.2f}")
            
            if result.fidelity:
                print(f"      Volume: {result.fidelity.volume_change_pct:+.1f}% | "
                      f"Hausdorff: {result.fidelity.hausdorff_relative*100:.2f}%")
            
        except Exception as e:
            print(f"   ERROR: {e}")
            continue
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if results:
        successful = sum(1 for r in results if r.is_printable)
        avg_quality = sum(r.quality_score for r in results) / len(results)
        avg_confidence = sum(r.prediction_confidence for r in results) / len(results)
        
        print(f"Total repairs: {len(results)}")
        print(f"Successful (printable): {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
        print(f"Average quality: {avg_quality:.2f}/5")
        print(f"Average confidence: {avg_confidence:.2f}")
    
    # Engine statistics
    print("\n" + "-" * 60)
    print("ENGINE STATISTICS")
    print("-" * 60)
    stats = engine.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Force training
    print("\n3. Forcing training step...")
    metrics = engine.train(force=True)
    if metrics:
        print(f"   Training complete: loss={metrics['loss']:.4f}, samples={metrics['samples_used']}")
    else:
        print("   Not enough samples to train")
    
    # Save model
    print("\n4. Saving model...")
    engine.save()
    print("   Model saved!")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


def test_prediction_only():
    """Test prediction without execution."""
    print("\n" + "=" * 60)
    print("PREDICTION ONLY TEST")
    print("=" * 60)
    
    from meshprep.ml.learning_engine import SmartRepairEngine
    from meshprep.core import Mesh
    import trimesh
    
    engine = SmartRepairEngine(auto_train=False)
    
    # Create test mesh
    mesh = Mesh(trimesh.creation.icosphere())
    mesh.trimesh.faces = mesh.trimesh.faces[::3]  # Remove 2/3 of faces
    
    print("\nPredicting strategy for damaged mesh...")
    actions, params, confidence = engine.predict_only(mesh)
    
    print(f"Predicted actions: {actions}")
    print(f"Parameters: {params}")
    print(f"Confidence: {confidence:.2f}")


def test_learning_improvement():
    """Test that the model improves with more data."""
    print("\n" + "=" * 60)
    print("LEARNING IMPROVEMENT TEST")
    print("=" * 60)
    
    from meshprep.ml.learning_engine import SmartRepairEngine, TrainingConfig
    from meshprep.core import Mesh
    import trimesh
    
    config = TrainingConfig(
        min_samples_to_train=3,
        epochs_per_update=20,
        model_dir=Path("models/learning_test"),
    )
    
    engine = SmartRepairEngine(config=config, auto_train=False)
    
    # Clear previous data
    engine.learning_loop.tracker.clear()
    
    print("\nPhase 1: Initial predictions (untrained)...")
    mesh1 = Mesh(trimesh.creation.icosphere())
    actions1, _, conf1 = engine.predict_only(mesh1)
    print(f"  Actions: {actions1[:3]}... Confidence: {conf1:.2f}")
    
    print("\nPhase 2: Recording some repair outcomes...")
    for i in range(5):
        mesh = Mesh(trimesh.creation.icosphere())
        
        # Record a "good" outcome
        engine.learning_loop.record_outcome(
            mesh=mesh,
            actions=["fix_normals", "pymeshfix_repair", "make_watertight"],
            parameters={},
            is_printable=True,
            quality_score=4.5,
            volume_change_pct=2.0,
            hausdorff_relative=0.01,
        )
    
    print(f"  Recorded {engine.learning_loop.total_samples_seen} outcomes")
    
    print("\nPhase 3: Training on recorded outcomes...")
    metrics = engine.train(force=True)
    if metrics:
        print(f"  Loss: {metrics['loss']:.4f}")
    
    print("\nPhase 4: Predictions after training...")
    actions2, _, conf2 = engine.predict_only(mesh1)
    print(f"  Actions: {actions2[:3]}... Confidence: {conf2:.2f}")
    
    print("\nLearning effect:")
    print(f"  Confidence change: {conf1:.2f} -> {conf2:.2f}")


if __name__ == "__main__":
    test_smart_engine()
    test_prediction_only()
    test_learning_improvement()
