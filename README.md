# MeshPrep

**Automated STL cleanup pipeline for 3D printing**

MeshPrep converts difficult or messy STL files into 3D-printable models with minimal manual intervention. It provides both a graphical interface for non-technical users and a powerful CLI for automation.

## Features

- **Automatic profile detection** — Analyzes models and suggests appropriate repair workflows
- **Extensive filter library** — 60+ repair actions from trimesh, pymeshfix, and Blender
- **Visual filter editor** — Drag-and-drop interface for building repair workflows
- **Dry-run preview** — Simulate repairs before committing changes
- **Reproducible runs** — Export run packages for sharing and collaboration
- **Blender escalation** — Automatic fallback to Blender for difficult meshes

## Installation

```bash
# Clone the repository
git clone https://github.com/DragonAceNL/MeshPrep.git
cd MeshPrep

# Create virtual environment
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
# Auto-detect profile and repair
python scripts/auto_fix_stl.py --input model.stl --output ./clean/

# Use a specific filter script
python scripts/auto_fix_stl.py --input model.stl --filter my_filter.json

# Dry-run with verbose output
python scripts/auto_fix_stl.py --input model.stl --dry-run --verbose
```

## Documentation

- [Functional Specification](docs/functional_spec.md) — Requirements and design
- [GUI Specification](docs/gui_spec.md) — Interface design and mockups
- [Model Profiles](docs/model_profiles.md) — Profile detection system
- [Code Style](docs/CODE_STYLE.md) — Coding standards

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

Copyright 2025 Dragon Ace
