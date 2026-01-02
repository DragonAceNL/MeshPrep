# MeshPrep POC v2 - Real Filter Script Validation

A proof-of-concept that validates filter scripts work with **real mesh libraries** (not mocked).

## Purpose

This POC proves that our filter scripts actually repair meshes by:
1. Loading real STL files from Thingi10K test fixtures
2. Running actual trimesh/pymeshfix operations
3. Validating results are printable and visually unchanged
4. Reporting success rates per defect category

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

# Save results to JSON
python validate_filters.py --all --output results.json
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

## Filter Presets

```bash
python validate_filters.py --list-presets
```

| Preset | Description |
|--------|-------------|
| `basic-cleanup` | Merge vertices, remove degenerates, fix normals |
| `fill-holes` | Fill holes using trimesh |
| `full-repair` | Complete repair using PyMeshFix |
| `manifold-repair` | Make mesh manifold |

## Project Structure

```
poc/v2/
├── meshprep_poc/
│   ├── __init__.py
│   ├── mesh_ops.py          # Mesh loading, diagnostics
│   ├── validation.py        # Two-stage validation
│   ├── filter_script.py     # Filter script runner
│   └── actions/
│       ├── __init__.py
│       ├── registry.py      # Action registration
│       ├── trimesh_actions.py   # trimesh-based actions
│       └── pymeshfix_actions.py # pymeshfix-based actions
├── filters/
│   ├── basic-cleanup.json
│   ├── fill-holes.json
│   ├── full-repair.json
│   └── manifold-repair.json
├── output/                  # Repaired models (optional)
├── requirements.txt
├── validate_filters.py      # Main validation script
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

## What This Proves

If this POC shows good success rates, it means:

1. ✅ Our filter script concept works
2. ✅ trimesh + pymeshfix can actually repair meshes
3. ✅ Our validation criteria are reasonable
4. ✅ We can proceed to build MeshPrep v1 with confidence

## Next Steps After POC v2

1. **If success rates are good (>80%)**:
   - Proceed to MeshPrep v1 implementation
   - Use these actions and presets as the foundation
   - Add GUI and profile detection

2. **If success rates are low**:
   - Investigate which actions fail and why
   - Add Blender escalation for difficult cases
   - Tune parameters and add new repair strategies
