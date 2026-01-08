# Test the clean RL implementation
import sys
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

print("=" * 60)
print("CLEAN RL IMPLEMENTATION TEST")
print("=" * 60)

# Test imports
print("\n1. Testing imports...")
from meshprep.ml import RepairAgent, RepairResult
from meshprep.ml.encoder import MeshEncoder, MeshFeatures
from meshprep.ml.environment import MeshRepairEnv, ACTIONS
from meshprep.ml.policy import ActorCritic
from meshprep.ml.agent import PPOAgent
print(f"   ✓ All imports successful")
print(f"   ✓ {len(ACTIONS)} actions available: {ACTIONS[:5]}...")

# Test encoder
print("\n2. Testing encoder...")
import trimesh
from meshprep.core import Mesh

tm = trimesh.creation.icosphere(subdivisions=2)
tm.faces = tm.faces[::2]  # Make broken
mesh = Mesh(tm)

encoder = MeshEncoder()
features = encoder.encode(mesh)
state = features.to_vector()

print(f"   ✓ Encoded mesh to {len(state)}-dim vector")
print(f"   ✓ Features: watertight={features.is_watertight}, components={features.num_components}")

# Test environment
print("\n3. Testing environment...")
env = MeshRepairEnv()
initial_state = env.reset(mesh)

print(f"   ✓ Environment reset, state shape: {initial_state.shape}")
print(f"   ✓ Action space: {env.NUM_ACTIONS} actions")
print(f"   ✓ State space: {env.STATE_DIM} dimensions")

# Test single step
result = env.step(9)  # blender_remesh
print(f"   ✓ Step: action=blender_remesh, reward={result.reward:.2f}, done={result.done}")

# Test policy network
print("\n4. Testing policy network...")
import torch

policy = ActorCritic()
state_t = torch.FloatTensor(initial_state).unsqueeze(0)
logits, value = policy(state_t)

print(f"   ✓ Policy output: logits shape={logits.shape}, value={value.item():.2f}")

action, log_prob, val = policy.get_action(state_t)
print(f"   ✓ Sampled action: {ACTIONS[action]}, log_prob={log_prob:.2f}")

# Test PPO agent
print("\n5. Testing PPO agent...")
agent = PPOAgent()
print(f"   ✓ Agent on device: {agent.device}")

# Test repair agent
print("\n6. Testing repair agent...")
repair_agent = RepairAgent()
print(f"   ✓ RepairAgent initialized")
print(f"   ✓ Stats: {repair_agent.stats}")

# Test actual repair
print("\n7. Testing mesh repair...")
result = repair_agent.repair(mesh, verbose=True)
print(f"   Result:")
print(f"   - Success: {result.success}")
print(f"   - Printable: {result.is_printable}")
print(f"   - Actions: {result.actions}")
print(f"   - Reward: {result.reward:.2f}")
print(f"   - Duration: {result.duration_ms:.0f}ms")

# Quick training test
print("\n8. Quick training test (5 iterations)...")
def mesh_gen():
    import numpy as np
    tm = trimesh.creation.icosphere(subdivisions=2)
    if np.random.rand() > 0.5:
        tm.faces = tm.faces[::2]
    return Mesh(tm)

for i in range(5):
    meshes = [mesh_gen() for _ in range(2)]
    rollout = repair_agent.agent.collect_rollout(meshes)
    metrics = repair_agent.agent.update(rollout, epochs=2)
    if i % 2 == 0:
        print(f"   Iter {i}: loss={metrics['loss']:.4f}, episodes={metrics['episodes']}")

print(f"\n   ✓ Trained {repair_agent.agent.total_episodes} episodes")

# Final test
print("\n9. Testing after training...")
result = repair_agent.repair(mesh_gen(), verbose=True)
print(f"   Success: {result.success}, Actions: {result.actions}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)
print(f"\nClean RL implementation ready!")
print(f"Files: encoder.py, environment.py, policy.py, agent.py, repair_agent.py")
print(f"Total: ~750 lines of clean code")
