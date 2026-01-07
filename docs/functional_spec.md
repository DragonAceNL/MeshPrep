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
6. **Validate** - Geometric + fidelity + auto-quality checks
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
| 4. Quality Verified | + auto/user rating ≥ 3 | ~99% | Recommended |

See [Validation Guide](validation.md) for criteria details.

---

## Auto-Quality Scoring

MeshPrep automatically computes a quality score (1-5) from geometric fidelity metrics, enabling fully automated training without manual ratings.

### How It Works

After a successful repair, the system computes:

| Metric | What It Measures | Impact on Score |
|--------|------------------|-----------------|
| **Volume Change %** | Shape preservation | High (±30% = -2 points) |
| **Hausdorff Distance** | Surface deviation | High (>5% = -1.5 points) |
| **Bounding Box Change** | Overall size | Medium (>5% = -1 point) |
| **Surface Area Change** | Detail changes | Low (>30% = -0.5 points) |
| **Printability** | Geometric validity | Bonus/penalty (±0.5 points) |

### Score Interpretation

| Score | Meaning | Automated Action |
|-------|---------|------------------|
| 5 | Perfect - indistinguishable from original | Trust for training |
| 4 | Good - minor smoothing, fully usable | Trust for training |
| 3 | Acceptable - noticeable but recognizable | Flag for review |
| 2 | Poor - significant detail loss | Penalize pipeline |
| 1 | Rejected - unrecognizable | Exclude pipeline |

### Training Flow

```
Repair Completes
    ↓
Compute Fidelity Metrics (Hausdorff, volume, bbox)
    ↓
Calculate Auto-Quality Score (1-5)
    ↓
Record to Quality Feedback Database
    ↓
Learning Engine Uses Score to:
  • Rank pipelines by quality
  • Penalize low-quality pipelines
  • Learn profile-specific thresholds
```

### CLI Integration

Auto-scoring is enabled by default during batch processing:

```bash
# Batch process with auto-scoring (default)
python run_full_test.py --input-dir ./models/

# Disable auto-scoring
python run_full_test.py --input-dir ./models/ --no-auto-quality

# View quality statistics
python run_full_test.py --quality-stats
```

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
| **Quality Feedback** | Learn from auto-scores and user ratings (1-5 scale) |
| **Error Learning** | Track failures to avoid repeated mistakes |

Data stored in `learning_data/` at repo root. See [Learning Systems](learning_systems.md) for details.

---

## Error Handling & Stability

**Stability is MeshPrep's #1 priority.** The system is designed to:

- **Never crash** - Gracefully handle all errors
- **Always produce output** - Return original mesh if repair fails  
- **Learn from failures** - Track errors to improve over time
- **Provide visibility** - Log everything for debugging

### Error Flow

```
Action Fails
    │
    ├──► Log to console/file
    │
    ├──► Log to SQLite (for learning)
    │
    └──► Return original mesh (never crash)
```

### Error Data Locations

| Data | Location |
|------|----------|
| Daily error logs | `learning_data/error_logs/errors_YYYY-MM-DD.log` |
| Failure patterns | `learning_data/action_crashes.db` |
| Crash tracking | `learning_data/action_crashes.db` |

See [Error Handling](error_handling.md) for complete details.

---

## Reports

`report.json` includes:
- Input fingerprint and metadata
- Profile detected with confidence
- Filter script used
- Per-step diagnostics and timing
- Validation results (geometric + fidelity)
- **Auto-quality score and breakdown**
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
- [Error Handling](error_handling.md) - Error handling and stability
- [Thingi10K Testing](thingi10k_testing.md) - Benchmark testing
