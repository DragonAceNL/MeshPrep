# Functional Specification: MeshPrep

## Overview

**Goal:** Automated pipeline converting messy STL files into 3D-printable models via GUI and CLI.

**Repository:** https://github.com/DragonAceNL/MeshPrep  
**License:** Apache 2.0  
**Python:** 3.11-3.12 (pymeshfix constraint)

---

## Scope

| In Scope | Out of Scope |
|----------|--------------|
| STL repair pipeline | Cloud service |
| GUI (PySide6) + CLI | Manufacturing simulation |
| Filter script editor | Mobile platforms |
| Profile detection | Guarantee all meshes fixable |
| Slicer validation | |

### Supported Input Formats

Primary: STL (ASCII/binary). Also: OBJ, PLY, 3MF, GLTF, CTM, STEP*, IGES* via trimesh/meshio.

*CAD formats require `pip install trimesh[easy]`

---

## High-Level Flow

1. **Environment Check** - Detect tools, auto-install if needed
2. **Model Selection** - Select file, choose auto-detect or existing filter
3. **Profile Detection** - Match diagnostics to profile, generate filter script
4. **Review** - Show suggested filter, allow editing
5. **Execute** - Run filter actions in order
6. **Validate** - Geometric + fidelity checks
7. **Escalate** - Blender if validation fails
8. **Slicer Validate** - Run through slicer (recommended)
9. **Output** - Export STL + reports

---

## Filter Scripts

JSON/YAML documents defining ordered repair actions:

```json
{
  "name": "holes-only-suggested",
  "version": "1.0.0",
  "meta": {
    "model_fingerprint": "MP:42f3729aa758",
    "profile": "holes-only"
  },
  "actions": [
    { "name": "trimesh_basic", "params": {} },
    { "name": "fill_holes", "params": { "max_hole_size": 1000 } },
    { "name": "validate", "params": {} }
  ]
}
```

See [Filter Actions](filter_actions.md) for complete action catalog.

---

## Model Fingerprints

Every model gets a searchable fingerprint: `MP:xxxxxxxxxxxx` (12 hex chars from SHA256 of original file bytes).

Enables community sharing: search fingerprint on Reddit to find working filter scripts.

---

## Validation Levels

| Level | Checks | Confidence | Default |
|-------|--------|------------|---------|
| 1. Basic | Watertight, volume | ~80% | Internal |
| 2. Full Geometry | + normals, self-intersections | ~90% | Internal |
| **3. Slicer Validated** | + successful slicer pass | ~95% | **Output** |
| 4. Quality Verified | + user rating ≥ 3 | ~99% | Recommended |

See [Validation Guide](validation.md) for criteria details.

---

## Slicer Validation

Run repaired model through slicer (PrusaSlicer, OrcaSlicer, etc.) to verify printability.

| Slicer Issue | Repair Actions |
|--------------|----------------|
| Thin walls | `offset_surface`, `blender_solidify` |
| Non-manifold | `pymeshfix_repair`, `blender_remesh` |
| Holes | `fill_holes`, `pymeshfix_repair` |
| Self-intersections | `fix_self_intersections`, `blender_boolean_union` |

**Important:** Use `--info` mode (STRICT), not `--export-gcode` (auto-repairs internally).

---

## External Tools

Auto-detected and auto-installed to `%LOCALAPPDATA%\MeshPrep\tools\`:

| Tool | Purpose | Size |
|------|---------|------|
| Blender 4.2 | Escalation repairs | ~400 MB |
| PrusaSlicer 2.8 | Slicer validation | ~200 MB |

**Detection order:** Explicit config → MeshPrep tools dir → System PATH → Standard locations

---

## Learning Systems

MeshPrep includes self-learning capabilities that improve over time:

| System | Purpose |
|--------|---------|
| **Learning Engine** | Track pipeline success rates, optimal ordering |
| **Pipeline Evolution** | Create new action combinations via genetic algorithm |
| **Profile Discovery** | Cluster similar meshes, discover new profiles |
| **Adaptive Thresholds** | Learn optimal parameter values from outcomes |
| **Quality Feedback** | Learn from user ratings (1-5 scale) |

Data stored in `learning_data/` at repo root. See [Learning Systems](learning_systems.md) for details.

---

## Reports

`report.json` includes:
- Input fingerprint and metadata
- Profile detected with confidence
- Filter script used
- Per-step diagnostics and timing
- Validation results (geometric + fidelity)
- Slicer validation results
- Tool versions and platform
- Reproducibility info (for exact reproduction)

---

## See Also

- [CLI Reference](cli_reference.md) - Command-line options
- [Filter Actions](filter_actions.md) - Complete action catalog
- [Model Profiles](model_profiles.md) - Profile definitions
- [GUI Spec](gui_spec.md) - GUI specification
- [Validation Guide](validation.md) - Validation criteria
- [Repair Strategy Guide](repair_strategy_guide.md) - Tool behavior and best practices
- [Repair Pipeline](repair_pipeline.md) - Pipeline stages
- [Learning Systems](learning_systems.md) - Self-learning components
- [Thingi10K Testing](thingi10k_testing.md) - Benchmark testing
