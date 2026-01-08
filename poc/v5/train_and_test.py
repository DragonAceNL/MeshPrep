# Train RL agent on Thingi10K and test on 400i
import sys
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from pathlib import Path
import numpy as np
import trimesh
from meshprep.core import Mesh
from meshprep.ml import RepairAgent

print("=" * 60)
print("TRAINING RL AGENT")
print("=" * 60)

# Create agent
agent = RepairAgent()
print(f"Device: {agent.agent.device}")

# Mesh generator from Thingi10K
thingi_dir = Path(r"C:\Users\Dragon Ace\Source\repos\Thingi10K\raw_meshes")
mesh_files = list(thingi_dir.glob("*.stl"))[:50]  # First 50 for speed
print(f"Training meshes: {len(mesh_files)}")

def mesh_gen():
    path = mesh_files[np.random.randint(len(mesh_files))]
    try:
        tm = trimesh.load(path, force='mesh')
        if hasattr(tm, 'to_geometry'):
            tm = tm.to_geometry()
        return Mesh(tm)
    except:
        tm = trimesh.creation.icosphere()
        tm.faces = tm.faces[::2]
        return Mesh(tm)

# Train
print("\nTraining (30 iterations)...")
for i in range(30):
    meshes = [mesh_gen() for _ in range(4)]
    rollout = agent.agent.collect_rollout(meshes)
    metrics = agent.agent.update(rollout, epochs=4)
    
    if i % 10 == 0:
        print(f"  Iter {i}: loss={metrics['loss']:.3f}, episodes={metrics['episodes']}")

print(f"\nTrained {agent.agent.total_episodes} episodes")

# Save
agent.save(Path("models/rl_trained.pt"))

# Test on 400i
print("\n" + "=" * 60)
print("TESTING ON 400i.ctm")
print("=" * 60)

mesh_path = Path(r"C:\Users\Dragon Ace\Source\repos\Thingi10K\raw_meshes\400i.ctm")
scene = trimesh.load(mesh_path)
tm = scene.to_geometry() if isinstance(scene, trimesh.Scene) else scene
mesh = Mesh(tm)

print(f"Input: {len(tm.vertices)} verts, {len(tm.faces)} faces")

result = agent.repair(mesh, verbose=True)

print(f"\nResult:")
print(f"  Success: {result.success}")
print(f"  Printable: {result.is_printable}")
print(f"  Actions: {result.actions}")
print(f"  Duration: {result.duration_ms/1000:.1f}s")

if result.mesh:
    out = result.mesh.trimesh
    print(f"  Output: {len(out.vertices)} verts, {len(out.faces)} faces")
    print(f"  Watertight: {out.is_watertight}")
