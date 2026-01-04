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
├── docs/              # Documentation (single source of truth per topic)
└── learning_data/     # Self-learning databases (SQLite)
```

---

## ⚠️ CRITICAL: Single Source of Truth Principle

**This is the most important documentation rule for MeshPrep.**

### The Rule

Each topic has ONE authoritative document. All other documents must **reference** it, never duplicate the information.

### Why This Matters

1. **Prevents token limit errors** — Duplicated content causes prompt overflow
2. **Ensures consistency** — One place to update, no conflicting information
3. **Easier maintenance** — Changes only need to be made once
4. **Clear ownership** — Everyone knows where to find/update information

### Source of Truth Map

| Topic | Authoritative Document |
|-------|------------------------|
| **Overview & high-level flow** | `docs/functional_spec.md` |
| **CLI commands & options** | `docs/cli_reference.md` |
| **Filter actions catalog** | `docs/filter_actions.md` |
| **Model profiles** | `docs/model_profiles.md` |
| **GUI layout & theming** | `docs/gui_spec.md` |
| **Validation criteria** | `docs/validation.md` |
| **Repair tool behavior** | `docs/repair_strategy_guide.md` |
| **Pipeline stages** | `docs/repair_pipeline.md` |
| **Learning systems** | `docs/learning_systems.md` |
| **Benchmark testing** | `docs/thingi10k_testing.md` |
| **Code style & headers** | `docs/CODE_STYLE.md` |
| **Third-party licenses** | `NOTICE` |
| **Contribution process** | `CONTRIBUTING.md` |

### ✅ Correct Pattern

```markdown
See [Filter Actions](filter_actions.md) for the complete action catalog.
```

### ❌ Incorrect Pattern

```markdown
The available actions are: trimesh_basic, fill_holes, validate...
<!-- WRONG: This duplicates content from filter_actions.md -->
```

### When Writing Documentation

1. **Before adding content** — Check if it belongs in an existing source-of-truth document
2. **If it does** — Add it there, then reference it from other docs
3. **If it doesn't** — Consider if a new focused document is needed
4. **Keep documents focused** — Each doc should cover ONE topic well
5. **Use tables over prose** — More compact, easier to scan
6. **Remove verbose code examples** — Implementation belongs in source files

### When Updating Documentation

1. **Update the source-of-truth document first**
2. **Check all cross-references** — Ensure they still point to valid sections
3. **Don't duplicate** — If you find yourself copying content, stop and reference instead

---

## Code Style Requirements

### License Header (REQUIRED)

Every new Python file MUST start with this header:

```python
# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
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
def check_watertight(mesh: trimesh.Trimesh) -> bool:
    """Check if a mesh is watertight (closed, no holes).

    Args:
        mesh: The trimesh mesh object to validate.

    Returns:
        True if the mesh is watertight, False otherwise.
    """
    return mesh.is_watertight
```

---

## Key Technologies

| Component | Library | Notes |
|-----------|---------|-------|
| Mesh processing | `trimesh` | Primary mesh library |
| Mesh repair | `pymeshfix` | Hole filling, manifold fixes |
| Surface reconstruction | `open3d` | Screened Poisson, ball pivoting |
| Morphological reconstruction | `scikit-image` | Voxel-based reconstruction |
| Spatial operations | `scipy` | KD-trees, Hausdorff distance |
| File I/O | `meshio` | Additional format support |
| GUI | `PySide6` | Qt for Python |
| Testing | `pytest` | Test framework |
| Optional | Blender (external) | Escalation for difficult meshes |

---

## Do NOT

- Do not use `os.path` — use `pathlib.Path` instead
- Do not hardcode file paths — use configuration
- Do not forget the license header on new files
- Do not bundle or link to Blender — invoke as subprocess only
- Do not modify third-party library code
- **Do not duplicate information** — reference the source-of-truth document instead
- **Do not leave documentation outdated** after making changes
- Do not make assumptions — gather context when needed
- **Do not delete existing test results** — POC v3 batch tests take hours to run

---

## When Adding or Removing Dependencies

**ALWAYS update ALL of the following files:**

1. **`poc/v2/requirements.txt`** — Package with version constraint
2. **`NOTICE`** — Third-party attribution
3. **`docs/functional_spec.md`** — If it's a core tool
4. **`.github/copilot-instructions.md`** — Key Technologies table

---

## Helpful Context

When generating code for MeshPrep:

1. **Check the source-of-truth document first** for the topic you're working on
2. Follow patterns in existing code
3. Always add the license header to new files
4. **Update docs** if your changes affect documented behavior
5. Reference docs, don't duplicate content

---

## Repair Strategy Guide Maintenance

The `docs/repair_strategy_guide.md` captures critical lessons learned. **Update it when:**

- Discovering new tool behavior or edge cases
- Finding bugs or limitations (document workarounds)
- Adding new repair actions
- Observing unexpected results
- Testing new model categories

---

## Quick Reference

| Need to know about... | Check this document |
|----------------------|---------------------|
| What actions exist | `docs/filter_actions.md` |
| CLI options | `docs/cli_reference.md` |
| Validation thresholds | `docs/validation.md` |
| Which tool to use when | `docs/repair_strategy_guide.md` |
| Learning system details | `docs/learning_systems.md` |
| GUI components | `docs/gui_spec.md` |
