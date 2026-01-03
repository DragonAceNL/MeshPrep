# MeshPrep

**Automatically fix broken 3D models so they actually print.**

Downloaded an STL that won't slice? Getting "non-manifold" or "not watertight" errors? MeshPrep automatically repairs your models and verifies they'll print correctly — no Blender skills required.

## How It Works

1. 📂 **Drop in your STL file**
2. 🔧 **MeshPrep automatically fixes issues** (holes, bad geometry, errors)
3. ✅ **Your slicer verifies it works** (PrusaSlicer, OrcaSlicer, SuperSlicer, Cura)
4. 🎉 **Get a printable file back**

## Why MeshPrep?

- **No 3D modeling skills needed** — it's fully automatic
- **Guaranteed to print** — verified with real slicers, not just geometry checks
- **Share what works** — save your repair settings and share with others
- **Handles tough cases** — automatic Blender escalation for difficult meshes

## Features

- **Slicer-verified output** — Every model is tested with your actual slicer for ~99% print success
- **Iterative repair loop** — Automatically tries different fixes until the slicer is happy
- **Automatic profile detection** — Analyzes models and suggests appropriate repair workflows
- **Extensive filter library** — 60+ repair actions from trimesh, pymeshfix, and Blender
- **Visual filter editor** — Drag-and-drop interface for building repair workflows
- **Shareable filter scripts** — Export proven repair workflows for others to use
- **GUI + CLI** — Desktop app for Windows, command-line for automation

## Requirements

- **Python 3.11 or 3.12** (pymeshfix doesn't have wheels for 3.13+)
- **A slicer** (PrusaSlicer, OrcaSlicer, SuperSlicer, or Cura)
- **Blender** (optional, for tough cases)

## Installation

```bash
# Clone the repository
git clone https://github.com/DragonAceNL/MeshPrep.git
cd MeshPrep

# Create virtual environment (use Python 3.11 or 3.12)
python -m venv .venv
.venv\Scripts\activate  # Windows
# or: source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Verify installation
python scripts/checkenv.py
```

## Quick Start

### GUI

```bash
python -m meshprep.gui
```

### CLI

```bash
# Auto-detect profile and repair (includes slicer validation)
python scripts/auto_fix_stl.py --input model.stl --output ./clean/

# Use a specific filter script
python scripts/auto_fix_stl.py --input model.stl --filter my_filter.json

# Use a community preset
python scripts/auto_fix_stl.py --input model.stl --preset holes-only

# Skip slicer validation with a trusted filter script (faster)
python scripts/auto_fix_stl.py --input model.stl --preset proven-preset --trust-filter-script

# Export run package for sharing
python scripts/auto_fix_stl.py --input model.stl --export-run ./share/run1/

# Verbose output
python scripts/auto_fix_stl.py --input model.stl --verbose
```

## Documentation

- [Functional Specification](docs/functional_spec.md) — Requirements and design
- [GUI Specification](docs/gui_spec.md) — Interface design and mockups
- [Model Profiles](docs/model_profiles.md) — Profile detection system
- [Filter Actions](docs/filter_actions.md) — Available repair actions
- [Repair Strategy Guide](docs/repair_strategy_guide.md) — Best practices for mesh repair
- [Code Style](docs/CODE_STYLE.md) — Coding standards

## Design Philosophy

> A model is not "fixed" until it passes slicer validation.

MeshPrep prioritizes **quality over speed**. While processing takes longer, you're guaranteed a printable result. Share your validated filter scripts with the community so others can skip the validation wait.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

Copyright 2025 Dragon Ace
