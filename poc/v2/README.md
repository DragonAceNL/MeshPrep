# MeshPrep POC v2 - Real Filter Script Validation

A proof-of-concept that validates filter scripts work with **real mesh libraries** (not mocked).

## Purpose

This POC proves that our filter scripts actually repair meshes by:
1. Loading real STL files from Thingi10K test fixtures
2. Running actual trimesh/pymeshfix operations
3. Validating results are printable and visually unchanged
4. **NEW: Validating with real slicers (PrusaSlicer, OrcaSlicer)**
5. Reporting success rates per defect category

## Quick Start

```bash
cd poc/v2

# IMPORTANT: Python 3.11 or 3.12 required for pre-built pymeshfix wheels
# Python 3.13+ does not have pre-built wheels yet

# Create virtual environment with Python 3.12
py -3.12 -m venv .venv          # Windows
python3.12 -m venv .venv        # Linux/Mac

# Activate
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run validation (tests 3 categories by default)
python validate_filters.py

# Test all categories
python validate_filters.py --all --limit 5

# Test specific category
python validate_filters.py --category holes --limit 10

# Test with slicer validation
python validate_filters.py --filter slicer-validated --category clean --limit 5

# Save results to JSON
python validate_filters.py --all --output results.json
```

## Slicer Validation (NEW)

POC v2 now supports **real slicer validation** to ensure repaired models are truly 3D printable.

### Important: STRICT vs SLICE Mode

Modern slicers (PrusaSlicer, OrcaSlicer) have **built-in auto-repair** that silently fixes mesh issues. This means a model that "passes" slicing may still have issues when used with other slicers.

MeshPrep uses **STRICT mode** by default:

| Mode | How It Works | Result |
|------|--------------|--------|
| **STRICT** (`--info`) | Analyzes mesh WITHOUT auto-repair | True mesh quality |
| **SLICE** (`--export-gcode`) | Tests if slicer can produce G-code | May hide issues |

Example difference:

```
Model: 100036.stl (has 46 open edges)

STRICT MODE: FAIL
  manifold: False
  open_edges: 46
  is_clean: False

SLICE MODE: PASS
  (slicer auto-fixed issues internally)
```

With STRICT mode, MeshPrep ensures the mesh itself is clean, not just that one particular slicer can work around issues.

### Supported Slicers

| Slicer | Detected Automatically | Installation |
|--------|------------------------|--------------|
| PrusaSlicer | ✅ Yes | `winget install Prusa3D.PrusaSlicer` |
| OrcaSlicer | ✅ Yes | `winget install SoftFever.OrcaSlicer` |
| SuperSlicer | ✅ Yes | Manual install |

### Test Slicer Integration

```bash
python test_slicer.py
```

Expected output:
```
============================================================
   MESHPREP POC v2 - SLICER INTEGRATION TEST
============================================================
  [OK] Found 1 slicer(s):
    - prusa: C:\Program Files\Prusa3D\PrusaSlicer\prusa-slicer-console.exe
  ...
  All tests passed! Slicer integration is working.
```

### Slicer-Validated Presets

New filter presets that include slicer validation:

| Preset | Description |
|--------|-------------|
| `slicer-validated` | Full repair + slicer validation |
| `conservative-slicer-validated` | Conservative repair + slicer validation |

### Iterative Slicer Repair Loop

When slicer validation fails, the system **automatically tries different repair strategies**:

```bash
python test_slicer_repair_loop.py
```

The repair loop:
1. Validates mesh with slicer
2. If it fails, parses errors to identify issues
3. Maps issues to repair strategies
4. Tries repairs in priority order
5. Repeats until success or max attempts reached

Example output:
```
RUNNING SLICER REPAIR LOOP
  Attempt 1: Running slicer validation...
  Slicer validation FAILED. Issues: ['unknown']
  Trying repair: trimesh_basic {}
    Repair broke geometry...
  Attempt 2: Running slicer validation...
  Trying repair: pymeshfix_repair {}
    Repair applied, geometry valid
  Attempt 3: Running slicer validation...
  Slicer validation PASSED on attempt 3

REPAIR RESULT
  Success: True
  Total attempts: 3
  Final mesh: Watertight=True, Is Volume=True
  Final slicer validation: PASS
```

## What It Tests

### Categories Tested

| Category | Filter Used | What It Tests |
|----------|-------------|---------------|
| clean | basic-cleanup | Models should pass through unchanged |
| holes | fill-holes | trimesh.repair.fill_holes works |
| many_small_holes | full-repair | pymeshfix handles complex holes |
| non_manifold | manifold-repair | Non-manifold edges are fixed |
| self_intersecting | full-repair | Self-intersections are resolved |
| fragmented | full-repair | Multiple components are joined |
| multiple_components | full-repair | Components are merged |
| complex | full-repair | Multiple issues are fixed |

### Validation Criteria

**Geometric Validation (must pass for 3D printing):**
- ✅ Watertight (no holes)
- ✅ Manifold (valid topology)
- ✅ Positive volume
- ✅ Consistent winding

**Fidelity Validation (must pass for visual accuracy):**
- ✅ Volume change < 1%
- ✅ Bounding box unchanged
- ✅ Hausdorff distance < 0.1% of bbox diagonal

**Slicer Validation (ultimate test):**
- ✅ Model produces valid G-code
- ✅ No fatal slicer errors
- ✅ Model has extrusions in first layer

## Available Actions

```bash
python validate_filters.py --list-actions
```

| Action | Description | Risk |
|--------|-------------|------|
| `trimesh_basic` | Merge vertices, remove degenerates | Low |
| `fill_holes` | Fill holes using trimesh | Medium |
| `fix_normals` | Fix face normals | Low |
| `pymeshfix_repair` | Full repair with PyMeshFix | Medium |
| `make_manifold` | Make mesh manifold | Medium |
| `keep_largest_component` | Remove small components | High |
| `simplify` | Reduce face count | Medium |
| `place_on_bed` | Move mesh to build plate (Z=0) | Low |
| `center_mesh` | Center mesh at origin | Low |
| `slicer_validate` | Validate with slicer (fatal on fail) | Low |
| `slicer_check` | Validate with slicer (warning only) | Low |

## Filter Presets

```bash
python validate_filters.py --list-presets
```

| Preset | Description |
|--------|-------------|
| `basic-cleanup` | Merge vertices, remove degenerates, fix normals |
| `fill-holes` | Fill holes using pymeshfix |
| `full-repair` | Complete repair using PyMeshFix |
| `conservative-repair` | Preserves multi-component models |
| `manifold-repair` | Make mesh manifold |
| `blender-remesh` | Aggressive Blender voxel remesh |
| `slicer-validated` | Full repair + slicer validation |
| `conservative-slicer-validated` | Conservative + slicer validation |

## Project Structure

```
poc/v2/
├── meshprep_poc/
│   ├── __init__.py
│   ├── mesh_ops.py              # Mesh loading, diagnostics
│   ├── validation.py            # Two-stage validation
│   ├── filter_script.py         # Filter script runner
│   ├── slicer_repair_loop.py    # Iterative slicer repair (NEW)
│   └── actions/
│       ├── __init__.py
│       ├── registry.py          # Action registration
│       ├── trimesh_actions.py   # trimesh-based actions
│       ├── pymeshfix_actions.py # pymeshfix-based actions
│       ├── blender_actions.py   # Blender-based actions
│       └── slicer_actions.py    # Slicer validation (NEW)
├── filters/
│   ├── basic-cleanup.json
│   ├── fill-holes.json
│   ├── full-repair.json
│   └── manifold-repair.json
├── output/                      # Repaired models (optional)
├── requirements.txt
├── validate_filters.py          # Main validation script
├── test_slicer.py              # Slicer integration test (NEW)
├── test_slicer_repair_loop.py  # Iterative repair test (NEW)
└── README.md
```

## Example Output

```
============================================================
VALIDATION SUMMARY
============================================================

Total models tested: 60

Success Rates:
  Repair completed:       95.0% (57/60)
  Geometrically valid:    78.3% (47/60)
  Visually unchanged:     71.7% (43/60)
  Overall success:        78.3% (47/60)

Performance:
  Avg duration: 234.5ms
  Avg volume change: 0.42%

By Category:
  clean               : 100.0% (20/20)
  holes               :  85.0% (17/20)
  non_manifold        :  50.0% (10/20)
============================================================
```

## Dependencies

- **trimesh** - Core mesh operations
- **pymeshfix** - Manifold repair (optional but recommended)
- **numpy** - Numerical operations
- **scipy** - Hausdorff distance computation
- **PrusaSlicer** (optional) - Slicer validation

## What This Proves

If this POC shows good success rates, it means:

1. ✅ Our filter script concept works
2. ✅ trimesh + pymeshfix can actually repair meshes
3. ✅ Our validation criteria are reasonable
4. ✅ **Slicer validation provides definitive proof of printability**
5. ✅ **Iterative repair loop can fix slicer failures automatically**
6. ✅ We can proceed to build MeshPrep v1 with confidence

## Next Steps After POC v2

1. **If success rates are good (>80%)**:
   - Proceed to MeshPrep v1 implementation
   - Use these actions and presets as the foundation
   - Add GUI and profile detection

2. **If success rates are low**:
   - Investigate which actions fail and why
   - Add Blender escalation for difficult cases
   - Tune parameters and add new repair strategies
