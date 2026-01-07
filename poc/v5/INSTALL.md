# MeshPrep v5 - Installation & Setup Guide

## ⚠️ IMPORTANT: Python Version Requirement

**MeshPrep v5 requires Python 3.11 or 3.12**

- ✅ Python 3.11 (recommended)
- ✅ Python 3.12 (supported)
- ❌ Python 3.13+ (Open3D not yet available)
- ❌ Python 3.10 or older (too old)

---

## 🚀 Installation

### Step 1: Create Virtual Environment

**Why?** Isolates dependencies and ensures correct Python version.

```bash
# Using Python 3.12 (recommended)
py -3.12 -m venv venv

# Or using Python 3.11
py -3.11 -m venv venv

# On Linux/Mac
python3.12 -m venv venv
```

### Step 2: Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

You should see `(venv)` in your prompt.

### Step 3: Install MeshPrep

```bash
# Development install with all features
pip install -e ".[all]"

# Or just core features
pip install -e .
```

**This installs:**
- ✅ numpy, trimesh, click (core)
- ✅ torch, torchvision (ML)
- ✅ pymeshfix (repair)
- ✅ open3d (reconstruction)

---

## 🧪 Verify Installation

### Quick Test

```bash
python test_runner_simple.py
```

**Expected output:**
```
[1/5] Generating test meshes...
[2/5] Loading actions...
  Loaded 20 actions
[3/5] Test: Fill holes in broken mesh
  Before: watertight=False, faces=9
  Action: success=True, duration=2.0ms
  After: faces=12

SUCCESS: All tests passed!
```

### Full Test Suite

```bash
# Install test tools
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=meshprep --cov-report=html
```

---

## 🔧 Troubleshooting

### Issue: "Python 3.14 detected"

**Problem:** Open3D doesn't support Python 3.14 yet.

**Solution:**
```bash
# Check your Python version
python --version

# If 3.14, install Python 3.12:
# Download from: https://www.python.org/downloads/
# Then create venv with correct version:
py -3.12 -m venv venv
```

### Issue: "Could not find a version that satisfies the requirement open3d"

**Problem:** Wrong Python version or platform not supported.

**Solution:**
1. Check Python version: `python --version` (must be 3.11 or 3.12)
2. Ensure virtual environment is activated: `(venv)` in prompt
3. Try: `pip install open3d==0.18.0` (specific version)

### Issue: "Import Error: cannot import name 'Mesh'"

**Problem:** Package not installed or wrong Python environment.

**Solution:**
```bash
# Ensure venv is activated
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Reinstall
pip install -e ".[all]"
```

---

## 📦 Installation Modes

### Minimal (Core Only)
```bash
pip install -e .
```
**Includes:** numpy, trimesh, click  
**Actions available:** 10 trimesh actions

### With PyMeshFix
```bash
pip install -e ".[pymeshfix]"
```
**Adds:** pymeshfix  
**Additional actions:** 3 PyMeshFix actions

### With ML
```bash
pip install -e ".[ml]"
```
**Adds:** torch, torchvision  
**Enables:** Pipeline prediction, quality scoring

### Complete (Recommended)
```bash
pip install -e ".[all]"
```
**Includes everything:** torch, pymeshfix, open3d  
**All 20 actions available**

---

## 🎯 Post-Installation

### Test Basic Functionality

```bash
# Activate venv
venv\Scripts\activate

# Test import
python -c "from meshprep import Mesh, RepairEngine; print('✓ Import successful')"

# Test CLI
meshprep --version
# Output: MeshPrep v5.0.0

# List actions
meshprep list-actions
```

### Run Example

```python
from meshprep import Mesh, RepairEngine
import trimesh

# Create test mesh
cube = trimesh.primitives.Box()
mesh = Mesh(cube)

# Create engine
engine = RepairEngine()

# Test action
from meshprep.core import ActionRegistry
result = ActionRegistry.execute("fix_normals", mesh)

print(f"✓ Action executed: success={result.success}")
```

---

## 🌍 Environment Setup Summary

```bash
# Complete setup from scratch:

# 1. Create virtual environment
py -3.12 -m venv venv

# 2. Activate
venv\Scripts\activate

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install MeshPrep with all features
pip install -e ".[all]"

# 5. Verify
python test_runner_simple.py

# 6. Run tests (optional)
pip install pytest pytest-cov
pytest tests/ -v

# ✓ Ready to use!
```

---

## 📝 Development Workflow

```bash
# Daily workflow:

# 1. Activate venv
venv\Scripts\activate

# 2. Make changes to code
# ... edit files ...

# 3. Test changes
python test_runner_simple.py

# 4. Run full tests (before commit)
pytest tests/ -v

# 5. Deactivate when done
deactivate
```

---

## ⚡ Quick Reference

| Command | Purpose |
|---------|---------|
| `py -3.12 -m venv venv` | Create venv |
| `venv\Scripts\activate` | Activate (Windows) |
| `source venv/bin/activate` | Activate (Linux/Mac) |
| `pip install -e ".[all]"` | Install everything |
| `python test_runner_simple.py` | Quick test |
| `pytest tests/ -v` | Full test suite |
| `deactivate` | Exit venv |

---

## 🎉 You're Ready!

After completing these steps, you have:
- ✅ Correct Python version (3.11/3.12)
- ✅ Isolated virtual environment
- ✅ All dependencies installed
- ✅ Tests passing
- ✅ Ready for development or use

**Next:** See main README.md for usage examples and API documentation.
