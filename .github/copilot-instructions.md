# GitHub Copilot Instructions for MeshPrep

This file provides context and instructions for GitHub Copilot when working on the MeshPrep project.

## Project Overview

MeshPrep is an automated STL cleanup pipeline that converts difficult or messy STL files into 3D-printable models. It provides both a GUI (PySide6) and CLI interface.

**Repository**: https://github.com/DragonAceNL/MeshPrep  
**License**: Apache License 2.0  
**Primary Language**: Python 3.11+

## Architecture

```
MeshPrep/
├── scripts/           # CLI tools (auto_fix_stl.py, checkenv.py)
├── src/meshprep/      # Main package
│   ├── core/          # Core logic (actions, profiles, validation)
│   ├── gui/           # PySide6 GUI components
│   └── cli/           # CLI implementation
├── config/            # Configuration files (filter_library.json)
├── filters/           # Filter script presets (JSON/YAML)
├── tests/             # Unit and integration tests
│   └── fixtures/      # Test STL files
└── docs/              # Documentation
```

## Code Style Requirements

### License Header (REQUIRED)

Every new Python file MUST start with this header:

```python
# Copyright 2025 Dragon Ace
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep
```

### Python Style

- Follow PEP 8
- Use type hints for all function signatures
- Write docstrings for public functions and classes (Google style)
- Maximum line length: 100 characters
- Use `pathlib.Path` for file paths, not string concatenation

### Example Function

```python
# Copyright 2025 Dragon Ace
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""Module for mesh validation checks."""

from pathlib import Path
from typing import Optional

import trimesh


def check_watertight(mesh: trimesh.Trimesh) -> bool:
    """Check if a mesh is watertight (closed, no holes).

    Args:
        mesh: The trimesh mesh object to validate.

    Returns:
        True if the mesh is watertight, False otherwise.
    """
    return mesh.is_watertight
```

## Key Technologies

| Component | Library | Notes |
|-----------|---------|-------|
| Mesh processing | `trimesh` | Primary mesh library |
| Mesh repair | `pymeshfix` | Hole filling, manifold fixes |
| File I/O | `meshio` | Additional format support |
| GUI | `PySide6` | Qt for Python |
| Testing | `pytest` | Test framework |
| Optional | Blender (external) | Escalation for difficult meshes |

## Filter Scripts

Filter scripts are JSON/YAML documents that define repair workflows:

```json
{
  "name": "basic-cleanup",
  "version": "1.0.0",
  "actions": [
    { "name": "trimesh_basic", "params": {} },
    { "name": "fill_holes", "params": { "max_hole_size": 1000 } },
    { "name": "validate", "params": {} }
  ]
}
```

Actions are registered in the action registry and map to implementations.

## GUI Components (PySide6)

- Main window uses a stacked widget for step navigation
- Filter Script Editor uses QSplitter with three panels
- Long operations run in QThread with signals for progress updates
- Theme colors defined in `docs/gui_spec.md`

### Dark Theme Colors

| Element | Color |
|---------|-------|
| Background | `#0f1720` |
| Panel | `#111822` |
| Accent | `#4fe8c4` |
| Text | `#dff6fb` |
| Button | `#1b2b33` |

## Testing

- Unit tests in `tests/test_*.py`
- Test fixtures (STL files) in `tests/fixtures/`
- Run tests: `pytest tests/`
- Each model profile should have at least one test fixture

## Common Patterns

### Action Implementation

```python
def action_fill_holes(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """Fill holes in a mesh.

    Args:
        mesh: Input mesh.
        params: Action parameters (max_hole_size, method).

    Returns:
        Modified mesh with holes filled.
    """
    max_size = params.get("max_hole_size", 1000)
    trimesh.repair.fill_holes(mesh)
    return mesh
```

### GUI Signal Pattern

```python
class WorkerThread(QThread):
    progress = Signal(int, str)  # (percentage, message)
    finished = Signal(bool, str)  # (success, result_or_error)

    def run(self):
        try:
            # Long operation
            self.progress.emit(50, "Processing...")
            self.finished.emit(True, "Complete")
        except Exception as e:
            self.finished.emit(False, str(e))
```

## Documentation

- `docs/functional_spec.md` — Requirements and high-level design
- `docs/gui_spec.md` — GUI specification with mockups
- `docs/model_profiles.md` — Model profile definitions
- `docs/CODE_STYLE.md` — Code style and header requirements

### Single Source of Truth

Maintain ONE authoritative document for each topic. Other documents should **reference** the source document rather than duplicating information.

| Topic | Source of Truth |
|-------|----------------|
| Filter actions catalog | `docs/functional_spec.md` |
| GUI layout and theming | `docs/gui_spec.md` |
| Model profiles | `docs/model_profiles.md` |
| Code style and headers | `docs/CODE_STYLE.md` |
| Third-party licenses | `NOTICE` |
| Contribution process | `CONTRIBUTING.md` |

**Example — correct:**
```markdown
See `docs/functional_spec.md` for the complete action catalog with parameters.
```

**Example — incorrect:**
```markdown
The available actions are: trimesh_basic, fill_holes, validate...  <!-- duplicating info -->
```

### Keep Documentation Up to Date

Before generating or modifying code:

1. **Check** that referenced documentation is current and accurate
2. **Update** docs when adding features, changing APIs, or modifying behavior
3. **Verify** cross-references still point to valid sections
4. **Flag** any outdated information found during development

When making changes that affect documented behavior:
- Update the source-of-truth document first
- Ensure all referencing documents still make sense
- Add changelog entries for significant changes

## Do NOT

- Do not use `os.path` — use `pathlib.Path` instead
- Do not hardcode file paths — use configuration
- Do not forget the license header on new files
- Do not bundle or link to Blender — invoke as subprocess only
- Do not modify third-party library code
- Do not duplicate information — reference the source-of-truth document instead
- Do not leave documentation outdated after making changes

## Helpful Context

When generating code for MeshPrep:

1. **Check documentation first** — verify specs are current before implementing
2. Check `docs/functional_spec.md` for the action catalog
3. Check `docs/gui_spec.md` for UI component specifications
4. Check `docs/model_profiles.md` for profile detection logic
5. Follow the patterns in existing code
6. Always add the license header to new files
7. **Update docs** if your changes affect documented behavior
