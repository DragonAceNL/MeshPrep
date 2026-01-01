# MeshPrep POC v1

A proof-of-concept implementation of the MeshPrep automated STL cleanup pipeline.

## Overview

This POC demonstrates the core functionality of MeshPrep:
- Profile detection from mesh diagnostics
- Filter script creation, editing, and execution
- GUI with step-by-step wizard interface
- CLI for automation

**Note:** This is a POC with **mocked mesh libraries**. The actual trimesh, pymeshfix, meshio, and Blender integrations are simulated to demonstrate the workflow without requiring those dependencies.

## Installation

```bash
cd poc/v1
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Running the GUI

```bash
python run_gui.py
```

## Running the CLI

```bash
# Auto-detect profile and repair
python run_cli.py --input model.stl --output ./clean/

# Use a specific filter script
python run_cli.py --input model.stl --filter filters/holes-only.json

# Dry-run with verbose output
python run_cli.py --input model.stl --dry-run --verbose

# Show help
python run_cli.py --help
```

## Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
poc/v1/
├── meshprep/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── mock_mesh.py       # Mock implementations
│   │   ├── diagnostics.py     # Mesh diagnostics
│   │   ├── profiles.py        # Profile detection
│   │   ├── actions.py         # Action registry
│   │   └── filter_script.py   # Filter script handling
│   ├── gui/
│   │   ├── __init__.py
│   │   ├── main_window.py     # Main GUI window
│   │   ├── widgets.py         # Custom widgets
│   │   ├── filter_editor.py   # Filter script editor
│   │   └── styles.py          # Theme styles
│   └── cli/
│       ├── __init__.py
│       └── main.py            # CLI implementation
├── config/
│   └── filter_library.json    # Action catalog
├── filters/
│   ├── basic-cleanup.json     # Preset: basic cleanup
│   ├── holes-only.json        # Preset: hole filling
│   ├── full-repair.json       # Preset: comprehensive repair
│   └── aggressive-blender.json # Preset: Blender escalation
├── tests/
│   ├── fixtures/              # Test STL files
│   ├── test_mock_mesh.py
│   ├── test_diagnostics.py
│   └── test_filter_script.py
├── output/                    # Default output directory
├── requirements.txt
├── run_gui.py                 # GUI entry point
├── run_cli.py                 # CLI entry point
└── README.md
```

## Features Demonstrated

### GUI Features
- Step-by-step wizard interface
- Environment check simulation
- File selection with drag-and-drop support
- Profile detection with confidence scores
- Filter script preview and editing
- Filter script editor with drag-and-drop
- Dry-run preview
- Execution with progress tracking
- Results summary and export

### CLI Features
- Auto-detection or manual filter script selection
- Dry-run mode
- JSON/CSV reporting
- Run package export for reproducibility
- Verbose logging

### Core Features
- Mock mesh operations (trimesh, pymeshfix, Blender)
- Diagnostics computation
- Profile detection with rule engine
- Filter script validation and execution
- Action registry with parameterized actions

## Mock Implementations

The POC includes mock implementations for:

| Component | Mock Class | Simulates |
|-----------|------------|-----------|
| Mesh loading/saving | `load_mock_stl`, `save_mock_stl` | trimesh STL I/O |
| Basic cleanup | `MockTrimesh` | trimesh operations |
| Mesh repair | `MockPyMeshFix` | pymeshfix repair |
| Advanced repair | `MockBlender` | Blender operations |

Mock operations modify mesh properties (vertex count, watertight status, etc.) to simulate the effects of real operations.

## Presets Included

| Preset | Description |
|--------|-------------|
| `basic-cleanup` | Merge vertices, remove degenerates, validate |
| `holes-only` | Fill holes and fix normals |
| `full-repair` | Comprehensive repair with pymeshfix |
| `aggressive-blender` | Blender remesh for difficult meshes |

## Next Steps for Real Implementation

1. Replace mock mesh operations with actual trimesh calls
2. Integrate pymeshfix for manifold repair
3. Add Blender subprocess integration
4. Implement real STL loading with issue detection
5. Add 3D preview rendering
6. Expand profile detection rules
7. Add batch processing mode
