# MeshPrep ML Module

Reinforcement Learning for 3D mesh repair.

## Architecture

```
meshprep/ml/
├── __init__.py        # Public API: RepairAgent
├── encoder.py         # Mesh → 16-dim feature vector
├── environment.py     # RL environment (state, action, reward)
├── policy.py          # Actor-Critic neural network
├── agent.py           # PPO implementation
└── repair_agent.py    # High-level interface
```

## Quick Start

```python
from meshprep.ml import RepairAgent

# Create agent
agent = RepairAgent()

# Repair a mesh
result = agent.repair("broken_model.stl")
print(f"Success: {result.success}")
print(f"Actions: {result.actions}")

if result.is_printable:
    result.mesh.trimesh.export("fixed.stl")
```

## Training

```python
from pathlib import Path
from meshprep.ml import RepairAgent

agent = RepairAgent()

# Train on Thingi10K dataset
agent.train(
    mesh_source=Path("/path/to/meshes"),
    iterations=500,
    verbose=True,
)

# Or with custom mesh generator
def mesh_generator():
    # Return a Mesh object
    pass

agent.train(mesh_source=mesh_generator, iterations=500)
```

## How It Works

### Reinforcement Learning

The agent learns through trial and error:

1. **State**: 16-dimensional feature vector describing the mesh
   - Geometry: vertex count, face count, volume, area
   - Topology: components, holes, manifoldness
   - Problems: degeneracy, fragmentation

2. **Actions**: 13 discrete repair operations
   - Basic: fix_normals, fill_holes, make_watertight
   - Advanced: blender_remesh, pymeshfix_repair, poisson_reconstruction
   - Special: STOP (end episode)

3. **Reward**:
   - +10 for making mesh printable (watertight + manifold)
   - -0.1 per step (encourage efficiency)
   - -1.0 for failed actions
   - Partial rewards for progress

4. **Policy**: PPO (Proximal Policy Optimization)
   - Actor-Critic architecture
   - Stable training with clipped objectives

### Why RL?

Mesh repair is sequential decision-making:
- Order matters (can't smooth before fixing holes)
- Different meshes need different strategies
- No "correct" labels, just outcomes

RL learns optimal action sequences from experience.

## Training Time Estimates

| Level | Iterations | Time | Quality |
|-------|------------|------|---------|
| Quick test | 50 | ~15 min | Demo only |
| Basic | 500 | ~2 hours | Functional |
| Good | 2000 | ~8 hours | Reliable |
| Production | 10000+ | ~2 days | Robust |

Bottleneck: Blender calls (~2-5s each)

## Files

| File | Lines | Purpose |
|------|-------|---------|
| encoder.py | ~100 | Feature extraction |
| environment.py | ~150 | RL environment |
| policy.py | ~100 | Neural network |
| agent.py | ~200 | PPO algorithm |
| repair_agent.py | ~200 | Public interface |

**Total: ~750 lines** - Clean, focused implementation.
