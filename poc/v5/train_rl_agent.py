# Train RL Agent on Thingi10K
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

from pathlib import Path
import numpy as np
import trimesh
from meshprep.core import Mesh
from meshprep.ml.rl_agent import RepairAgent, RLTrainer

print("=" * 60)
print("RL AGENT TRAINING ON THINGI10K")
print("=" * 60)

# Create agent
agent = RepairAgent(
    hidden_dim=256,
    lr=3e-4,
    device="auto",
)
print(f"Device: {agent.device}")

# Thingi10K path
thingi_dir = Path(r"C:\Users\Dragon Ace\Source\repos\Thingi10K\raw_meshes")
mesh_files = list(thingi_dir.glob("*.stl"))[:100]  # First 100 for speed
print(f"Found {len(mesh_files)} meshes")

def load_mesh(idx):
    """Load mesh by index."""
    path = mesh_files[idx % len(mesh_files)]
    try:
        tm = trimesh.load(path, force='mesh')
        if isinstance(tm, trimesh.Scene):
            tm = tm.to_geometry()
        return Mesh(tm)
    except Exception:
        # Fallback
        tm = trimesh.creation.icosphere()
        tm.faces = tm.faces[::2]
        return Mesh(tm)

def mesh_generator():
    idx = np.random.randint(len(mesh_files))
    return load_mesh(idx)

# Create eval set
print("\nCreating evaluation set...")
eval_meshes = [load_mesh(i * 10) for i in range(5)]

# Train
print("\nTraining (50 iterations)...")
trainer = RLTrainer(agent, save_dir=Path("models/rl_thingi"))

for iteration in range(50):
    # Collect rollout from 4 meshes
    meshes = [mesh_generator() for _ in range(4)]
    buffer = agent.collect_rollout(meshes)
    
    # Update
    metrics = agent.update(buffer, epochs=4, batch_size=32)
    
    if iteration % 10 == 0:
        # Evaluate
        successes = 0
        for m in eval_meshes:
            result = agent.repair(m.copy(), deterministic=True)
            if result.is_printable:
                successes += 1
        
        print(f"Iter {iteration}: episodes={metrics['total_episodes']}, "
              f"policy_loss={metrics['policy_loss']:.4f}, "
              f"eval_success={successes}/{len(eval_meshes)}")

# Final evaluation
print("\n" + "=" * 60)
print("FINAL EVALUATION")
print("=" * 60)

successes = 0
for i, m in enumerate(eval_meshes):
    print(f"\nMesh {i+1}:")
    result = agent.repair(m.copy(), deterministic=True, verbose=True)
    print(f"  Success: {result.success}, Steps: {result.num_steps}")
    if result.is_printable:
        successes += 1

print(f"\nFinal success rate: {successes}/{len(eval_meshes)} ({successes/len(eval_meshes)*100:.0f}%)")

# Save
agent.save(Path("models/rl_thingi/final.pt"))
print("\nAgent saved!")
