# Test clean RL on challenging 400i.ctm
import sys
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from pathlib import Path
import trimesh
from meshprep.core import Mesh
from meshprep.ml import RepairAgent

print("=" * 60)
print("TESTING CLEAN RL ON 400i.ctm (14,375 components)")
print("=" * 60)

# Load challenging mesh
mesh_path = Path(r"C:\Users\Dragon Ace\Source\repos\Thingi10K\raw_meshes\400i.ctm")
scene = trimesh.load(mesh_path)
if isinstance(scene, trimesh.Scene):
    tm = scene.to_geometry()
else:
    tm = scene

mesh = Mesh(tm)

print(f"\nInput mesh:")
print(f"  Vertices: {len(tm.vertices)}")
print(f"  Faces: {len(tm.faces)}")
print(f"  Watertight: {tm.is_watertight}")

# Check features
from meshprep.ml.encoder import MeshEncoder
encoder = MeshEncoder()
features = encoder.encode(mesh)
print(f"  Components: {features.num_components}")
print(f"  Fragmented: {features.is_fragmented}")
print(f"  Very fragmented: {features.is_very_fragmented}")

# Create agent and repair
print("\nRepairing with RL agent...")
agent = RepairAgent()

result = agent.repair(mesh, verbose=True)

print(f"\nResult:")
print(f"  Success: {result.success}")
print(f"  Printable: {result.is_printable}")
print(f"  Actions: {result.actions}")
print(f"  Steps: {result.steps}")
print(f"  Duration: {result.duration_ms/1000:.1f}s")

if result.mesh:
    out = result.mesh.trimesh
    print(f"\nOutput mesh:")
    print(f"  Vertices: {len(out.vertices)}")
    print(f"  Faces: {len(out.faces)}")
    print(f"  Watertight: {out.is_watertight}")
    
    # Save
    output_path = Path("repaired/400i_rl_clean.stl")
    output_path.parent.mkdir(exist_ok=True)
    out.export(output_path)
    print(f"  Saved to: {output_path}")

print("\n" + "=" * 60)
