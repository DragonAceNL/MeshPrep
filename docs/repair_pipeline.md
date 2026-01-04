# Mesh Repair Pipeline

## Overview

The repair pipeline ensures models are **geometrically valid** (printable) and **visually unchanged** (faithful to original).

---

## Pipeline Stages

```
Input STL
    ↓
1. Load & Baseline Capture (fingerprint, volume, bbox, surface samples)
    ↓
2. Diagnostics Computation (watertight, manifold, holes, components, normals)
    ↓
3. Profile Detection (match diagnostics → profile with confidence)
    ↓
4. Filter Script Generation (suggested actions for profile)
    ↓
5. Filter Script Execution (run actions, capture per-step diagnostics)
    ↓
6. Geometric Validation (watertight, manifold, volume, self-intersections)
    ↓
7. Fidelity Validation (volume change, bbox, Hausdorff distance)
    ↓
8. Escalation (if needed → Blender)
    ↓
9. Output & Reporting (STL, report.json, report.csv)
    ↓
Output STL
```

---

## Key Data Structures

### Baseline Metrics (captured before repair)

| Field | Purpose |
|-------|---------|
| `fingerprint` | SHA256 of file |
| `volume`, `surface_area` | Geometry metrics |
| `bbox_*` | Bounding box |
| `surface_samples` | 10K points for Hausdorff |

### Diagnostics

| Field | Purpose |
|-------|---------|
| `is_watertight`, `is_manifold` | Topology status |
| `hole_count`, `component_count` | Issue counts |
| `non_manifold_edge_count` | Topology errors |
| `degenerate_face_count` | Quality issues |
| `has_self_intersections` | Intersection status |

---

## Success Criteria

### Geometric Validity (Required)

- ✅ Watertight
- ✅ Manifold
- ✅ Positive volume
- ✅ No self-intersections
- ✅ No degenerate faces

### Visual Fidelity (Required)

- ✅ Volume change < 1%
- ✅ Bounding box unchanged
- ✅ Hausdorff distance < 0.1% of bbox diagonal

---

## Action Risk Classification

| Risk | Actions | Visual Impact |
|------|---------|---------------|
| 🟢 Safe | `merge_vertices`, `fix_normals`, `remove_degenerates` | None |
| 🟡 Low | `fill_holes` (small), `pymeshfix_repair` | Minimal |
| 🟠 Medium | `fill_holes` (large), `remove_small_components`, `smooth` | May affect detail |
| 🔴 High | `boolean_union`, `blender_remesh` | Changes topology |

---

## Escalation Levels

| Level | Actions | When Used |
|-------|---------|-----------|
| 0 | trimesh + pymeshfix | Default |
| 1 | Aggressive pymeshfix | Level 0 fails |
| 2 | Blender boolean | Level 1 fails |
| 3 | Blender remesh | Last resort (may alter geometry) |

Level 3 requires user consent for destructive changes.

---

## Configuration

```yaml
# config/fidelity_thresholds.yaml
fidelity:
  max_volume_change_pct: 1.0
  max_hausdorff_relative: 0.001
  bbox_tolerance: 1e-6

escalation:
  enabled: true
  max_level: 2
  require_consent_level: 2
```

---

## See Also

- [Validation Guide](validation.md) - Validation criteria details
- [Filter Actions](filter_actions.md) - Action catalog
- [Repair Strategy Guide](repair_strategy_guide.md) - Tool behavior
