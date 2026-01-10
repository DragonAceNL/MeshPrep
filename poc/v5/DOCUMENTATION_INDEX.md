# MeshPrep v5 - Documentation Index

## 📚 Master Document for Development

This document provides a complete index of POC v5.

---

## 🎯 Project Status

| Metric | Value |
|--------|-------|
| **Tests** | 56/56 passing (100%) |
| **Python** | 3.11 or 3.12 |
| **Status** | Production-ready |

---

## 📁 File Structure

```
poc/v5/
├── README.md              # Overview
├── INSTALL.md             # Installation guide
├── DOCUMENTATION_INDEX.md # This file
├── requirements.txt       # Dependencies
├── setup.py               # Package config
├── setup.bat              # Windows setup
├── setup.sh               # Linux/Mac setup
├── pytest.ini             # Test config
│
├── meshprep/              # Main package
│   ├── __init__.py
│   ├── core/              # Core components
│   │   ├── mesh.py        # Mesh wrapper
│   │   ├── action.py      # Action registry
│   │   ├── pipeline.py    # Pipeline execution
│   │   ├── validator.py   # Validation
│   │   ├── repair_engine.py
│   │   └── bootstrap.py   # Auto-install deps
│   │
│   ├── actions/           # 20 repair actions
│   │   ├── trimesh/       # 10 basic actions
│   │   ├── pymeshfix/     # 3 repair actions
│   │   ├── blender/       # 3 Blender actions
│   │   ├── open3d/        # 3 reconstruction
│   │   └── core/          # Validation action
│   │
│   ├── ml/                # RL-based repair (~990 lines)
│   │   ├── encoder.py     # Mesh → features
│   │   ├── environment.py # RL environment
│   │   ├── policy.py      # Actor-Critic network
│   │   ├── agent.py       # PPO algorithm
│   │   └── repair_agent.py# Public API
│   │
│   ├── learning/          # Statistics tracking
│   │   ├── history_tracker.py
│   │   └── strategy_learner.py
│   │
│   └── cli/               # CLI interface
│       └── main.py
│
├── tests/                 # Test suite (56 tests)
│   ├── conftest.py
│   ├── test_bootstrap_setup.py
│   ├── test_core_real.py
│   ├── test_actions_real.py
│   ├── test_pipelines_real.py
│   ├── test_learning_real.py
│   └── test_integration_full.py
│
└── venv/                  # Virtual environment
```

---

## 🚀 Quick Start

### Setup
```cmd
# Windows
setup.bat

# Or manual
py -3.12 -m venv venv
venv\Scripts\activate
pip install -e ".[all]"
```

### Run Tests
```cmd
pytest tests/ -v
```

### Use ML Repair Agent
```python
from meshprep.ml import RepairAgent

agent = RepairAgent()
result = agent.repair("broken.stl")

if result.is_printable:
    result.mesh.trimesh.export("fixed.stl")
```

### Train Agent
```python
from pathlib import Path
agent.train(Path("meshes/"), iterations=500)
```

---

## 🔧 Key Components

### Core (meshprep/core/)
- **Mesh**: Wrapper with metadata
- **ActionRegistry**: Plugin system for repairs
- **Pipeline**: Sequential action execution
- **Validator**: Geometric + fidelity checks

### Actions (meshprep/actions/)
| Category | Count | Examples |
|----------|-------|----------|
| trimesh | 10 | fix_normals, fill_holes, decimate |
| pymeshfix | 3 | repair, clean, remove_small |
| blender | 3 | remesh, boolean_union, solidify |
| open3d | 3 | poisson, ball_pivot, simplify |

### ML (meshprep/ml/)
Clean RL implementation using PPO:
- **State**: 16-dim mesh features
- **Actions**: 13 discrete repair operations
- **Reward**: +10 printable, -0.1/step, -1 failure

---

## 📊 Dependencies

| Package | Required | Purpose |
|---------|----------|---------|
| numpy | Yes | Core |
| trimesh | Yes | Mesh processing |
| scipy | Yes | Spatial operations |
| torch | Optional | ML |
| pymeshfix | Optional | Repair |
| open3d | Optional | Reconstruction |
| click | Yes | CLI |

---

## 📞 Context for Future Sessions

1. **Environment**: Python 3.12 venv
2. **Setup**: `setup.bat` or manual
3. **Tests**: `pytest tests/ -v` (56 passing)
4. **ML**: Clean RL in `meshprep/ml/` (~990 lines)
5. **Blender**: v5.0 works

---

**Last Updated**: After cleanup session  
**Structure**: Clean, minimal, production-ready
