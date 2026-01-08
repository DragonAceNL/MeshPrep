# Test the RL Repair Agent
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

from pathlib import Path
import trimesh
from meshprep.core import Mesh
from meshprep.ml.rl_agent import RepairAgent, RLTrainer

print("=" * 60)
print("REINFORCEMENT LEARNING REPAIR AGENT TEST")
print("=" * 60)

# Create agent
print("\n1. Creating RL Agent...")
agent = RepairAgent(
    hidden_dim=256,
    lr=3e-4,
    device="auto",
)
print(f"   Device: {agent.device}")
print(f"   Actions: {len(agent.env.get_valid_actions())}")

# Test on a simple broken mesh
print("\n2. Testing on damaged icosphere...")
tm = trimesh.creation.icosphere(subdivisions=2)
tm.faces = tm.faces[::2]  # Remove half the faces
mesh = Mesh(tm)

print(f"   Input: {len(mesh.trimesh.vertices)} verts, {len(mesh.trimesh.faces)} faces")
print(f"   Watertight: {mesh.trimesh.is_watertight}")

result = agent.repair(mesh, deterministic=False, verbose=True)

print(f"\n   Result:")
print(f"   Success: {result.success}")
print(f"   Printable: {result.is_printable}")
print(f"   Actions: {result.actions_taken}")
print(f"   Total reward: {result.total_reward:.2f}")
print(f"   Steps: {result.num_steps}")

if result.mesh:
    out = result.mesh.trimesh
    print(f"   Output: {len(out.vertices)} verts, {len(out.faces)} faces")
    print(f"   Watertight: {out.is_watertight}")

# Quick training test
print("\n3. Quick training test (10 iterations)...")

def mesh_generator():
    """Generate random broken mesh."""
    import numpy as np
    shape = np.random.choice(['sphere', 'box', 'cylinder'])
    
    if shape == 'sphere':
        tm = trimesh.creation.icosphere(subdivisions=2)
    elif shape == 'box':
        tm = trimesh.creation.box()
    else:
        tm = trimesh.creation.cylinder(radius=1, height=2)
    
    # Damage it randomly
    damage = np.random.choice(['holes', 'fragments', 'degenerate'])
    
    if damage == 'holes':
        tm.faces = tm.faces[::2]
    elif damage == 'fragments':
        # Create multiple components
        tm2 = tm.copy()
        tm2.vertices += [2, 0, 0]
        tm = trimesh.util.concatenate([tm, tm2])
    else:
        # Remove some faces
        keep = np.random.rand(len(tm.faces)) > 0.3
        tm.faces = tm.faces[keep]
    
    return Mesh(tm)

trainer = RLTrainer(agent, save_dir=Path("models/rl_test"))

# Just a few iterations for testing
print("   Training...")
for i in range(10):
    meshes = [mesh_generator() for _ in range(2)]
    buffer = agent.collect_rollout(meshes)
    metrics = agent.update(buffer, epochs=2, batch_size=32)
    
    if i % 5 == 0:
        print(f"   Iter {i}: policy_loss={metrics['policy_loss']:.4f}, episodes={metrics['total_episodes']}")

print(f"\n   Total episodes trained: {agent.total_episodes}")
print(f"   Total steps: {agent.total_steps}")

# Test after training
print("\n4. Testing after training...")
result2 = agent.repair(mesh_generator(), deterministic=True, verbose=True)
print(f"   Success: {result2.success}")
print(f"   Actions: {result2.actions_taken}")

# Save
agent.save(Path("models/rl_test/agent.pt"))
print("\n5. Agent saved!")

print("\n" + "=" * 60)
print("RL AGENT TEST COMPLETE")
print("=" * 60)
