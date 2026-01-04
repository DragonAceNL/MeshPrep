# Model Profiles

## Overview

MeshPrep automatically detects model profiles from diagnostics and suggests appropriate filter scripts. The system includes **110+ profiles** organized into categories.

---

## Profile Detection

On model load, the scanner computes a diagnostics vector:

| Diagnostic | Type | Description |
|------------|------|-------------|
| `is_watertight` | bool | Closed mesh |
| `hole_count` | int | Boundary loops |
| `component_count` | int | Disconnected parts |
| `non_manifold_edge_count` | int | Edges shared by 3+ faces |
| `degenerate_face_count` | int | Zero-area triangles |
| `normal_consistency` | float | 0..1 consistency |
| `self_intersections` | bool | Self-intersecting |
| `triangle_count` | int | Total faces |

A rule engine matches diagnostics to profiles with confidence scores.

---

## Profile Categories

| Category | Example Profiles | Primary Signal |
|----------|------------------|----------------|
| **Clean** | `clean`, `clean-minor-issues` | Watertight, manifold |
| **Holes** | `holes-only`, `open-bottom`, `many-small-holes` | `hole_count > 0` |
| **Fragmented** | `fragmented`, `debris-particles`, `floating-components` | `component_count` high |
| **Topology** | `non-manifold`, `degenerate-heavy`, `t-junction-heavy` | Non-manifold edges/vertices |
| **Normals** | `normals-inconsistent`, `inverted-normals` | `normal_consistency` low |
| **Self-Intersection** | `self-intersecting`, `overlapping-shells` | Intersection detected |
| **Internal** | `hollow-porous`, `nested-shells`, `internal-geometry` | Nested components |
| **Thin Features** | `thin-shell`, `thin-walls-localized`, `knife-edge` | `estimated_min_thickness` low |
| **Scan/Noisy** | `noisy-scan`, `spike-artifacts`, `high-triangle-density` | High density, many defects |
| **Scale** | `small-part`, `oversized`, `high-aspect-ratio` | Bbox dimensions |
| **Printability** | `overhang-heavy`, `trapped-volume`, `requires-supports` | Geometry analysis |

---

## Core Profiles Quick Reference

| Profile | Detection | Suggested Actions |
|---------|-----------|-------------------|
| `clean` | Watertight, manifold, single component | `trimesh_basic` → `validate` |
| `holes-only` | Has holes, single component, manifold | `trimesh_basic` → `fill_holes` → `validate` |
| `non-manifold` | Non-manifold edges > 0 | `trimesh_basic` → `pymeshfix_repair` → `validate` |
| `fragmented` | Many components, largest < 80% | `remove_small_components` → `fill_holes` → `validate` |
| `self-intersecting` | Self-intersections detected | `pymeshfix_repair` → escalate to Blender if needed |
| `extreme-fragmented` | 1000+ components | `fragment_aware_reconstruct` (surface reconstruction) |

---

## Filter Script Format

```json
{
  "name": "holes-only-suggested",
  "version": "1.0.0",
  "meta": {
    "generated_by": "model_scan",
    "profile": "holes-only",
    "model_fingerprint": "<hash>"
  },
  "actions": [
    { "name": "trimesh_basic", "params": {} },
    { "name": "fill_holes", "params": { "max_hole_size": 1000 } },
    { "name": "validate", "params": {} }
  ]
}
```

---

## Extensibility

- **Configurable thresholds** in `config/`
- **Confidence scores** shown in UI (0.0–1.0)
- **Multi-profile detection** - top 2-3 shown with scores
- **Profile combinations** for co-occurring issues

---

## Testing

- Fixtures in `tests/fixtures/` named `profile_<name>.stl`
- Each profile needs at least one test fixture

---

## See Also

- [Filter Actions](filter_actions.md) - Available repair actions
- [Repair Strategy Guide](repair_strategy_guide.md) - When to use each approach
