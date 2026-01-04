# Validation Guide

## Overview

MeshPrep uses a **two-stage validation** approach to ensure repaired models are both **geometrically valid** (printable) and **visually unchanged** (faithful to the original).

```
success = is_printable AND is_visually_unchanged
```

---

## Two-Stage Validation Pipeline

| Stage | Purpose | Result |
|-------|---------|--------|
| **Stage 1: Geometric** | Ensure mesh is valid for 3D printing | `is_printable` |
| **Stage 2: Fidelity** | Ensure model appearance is preserved | `is_visually_unchanged` |

---

## Stage 1: Geometric Validation

### Checks

| Check | Criteria | Common Failures |
|-------|----------|-----------------|
| **Watertight** | No boundary edges, all edges shared by exactly 2 faces | Open holes, gaps, missing faces |
| **Manifold** | Every edge shared by exactly 2 faces, vertices form single fan | T-junctions, bowtie vertices, self-touching faces |
| **Positive Volume** | Volume > 0 | Inverted normals (inside-out), 2D/degenerate geometry |
| **No Self-Intersections** | Faces don't intersect (except at shared edges) | Overlapping geometry |
| **No Degenerate Faces** | No zero-area faces, no duplicate vertices per face | Collapsed triangles |

### Non-Manifold Types

| Type | Description |
|------|-------------|
| T-junction | Edge shared by 3+ faces |
| Bowtie vertex | Vertex shared by disconnected face groups |
| Self-touching | Faces that touch but don't share topology |

### Performance Notes

- Self-intersection detection is computationally expensive
- For large meshes: use sampling-based approximation or bounding box pre-filtering

---

## Stage 2: Fidelity Validation

### Checks

| Check | Default Threshold | Notes |
|-------|-------------------|-------|
| **Volume Change** | < 1% | Strict: 0.1%, Permissive: 5% |
| **Bounding Box** | Unchanged (tolerance: 1e-6) | Relative dimension check |
| **Hausdorff Distance** | < 0.1% of bbox diagonal | 10,000 sample points default |
| **Surface Area** | < 2% change | May increase slightly when filling holes |

### Hausdorff Distance

Measures maximum surface deviation between original and repaired meshes.

| Value | Meaning |
|-------|---------|
| 0 | Surfaces identical |
| Small | Minor local deviations (acceptable) |
| Large | Significant surface changes (problem) |

**Calculation:** Uses bidirectional KD-tree nearest-neighbor lookup with surface sampling.

---

## Validation Modes

| Mode | Volume Tolerance | Hausdorff Relative | Use Case |
|------|------------------|-------------------|----------|
| **Strict** | < 0.1% | < 0.01% | High-precision, mechanical parts |
| **Normal** (default) | < 1.0% | < 0.1% | General 3D printing |
| **Permissive** | < 5.0% | < 1.0% | Artistic models, topology repair |

---

## Default Configuration

```yaml
# config/validation_thresholds.yaml

geometric:
  max_self_intersections: 0
  max_degenerate_faces: 0

fidelity:
  max_volume_change_pct: 1.0
  bbox_tolerance: 1e-6
  max_hausdorff_relative: 0.001
  max_surface_area_change_pct: 2.0
  hausdorff_sample_count: 10000

mode: normal  # Options: strict, normal, permissive
```

---

## Validation Report Structure

The validation report contains:

| Field | Type | Description |
|-------|------|-------------|
| `is_successful` | bool | Overall pass/fail |
| `geometric.is_printable` | bool | All geometric checks pass |
| `geometric.issues` | list | List of geometric problems |
| `fidelity.is_visually_unchanged` | bool | All fidelity checks pass |
| `fidelity.changes` | list | List of detected changes |
| `warnings` | list | Borderline cases (e.g., volume change 0.5-1.0%) |

---

## Implementation Reference

The validation functions are implemented in:
- `src/meshprep/core/validation.py` - Core validation logic
- `poc/v3/batch_processor.py` - Batch validation integration

Key classes:
- `GeometricValidation` - Stage 1 results
- `FidelityValidation` - Stage 2 results  
- `ValidationReport` - Combined report with `to_dict()` export

---

## See Also

- [Repair Strategy Guide](repair_strategy_guide.md) - Tool behavior and best practices
- [Functional Spec](functional_spec.md) - Action catalog and parameters
