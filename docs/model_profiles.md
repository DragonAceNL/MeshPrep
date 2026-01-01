# Model Profiles

## Purpose

This document describes the model profiles the system can detect automatically and the heuristics used to assign a profile to a model. Each profile is associated with a suggested filter script (ordered actions) that can be reviewed and edited by the user.

## How Profiles Are Selected

On model load the scanner computes a diagnostics vector using `trimesh` and other helpers. Typical diagnostics include:

| Diagnostic | Type | Description |
|------------|------|-------------|
| `is_watertight` | bool | Whether the mesh is closed |
| `hole_count` | int | Number of boundary loops (holes) |
| `component_count` | int | Number of disconnected components |
| `largest_component_pct` | float | Percentage of mesh in largest component |
| `non_manifold_edge_count` | int | Edges shared by more than 2 faces |
| `non_manifold_vertex_count` | int | Vertices with non-manifold topology |
| `degenerate_face_count` | int | Zero-area or invalid triangles |
| `normal_consistency` | float | 0..1 consistency of face normals |
| `bbox` | tuple | Bounding box dimensions (x, y, z) |
| `bbox_volume` | float | Volume of bounding box |
| `avg_edge_length` | float | Average triangle edge length |
| `triangle_count` | int | Total number of triangles |
| `triangle_density` | float | Triangles per unit volume |
| `self_intersections` | bool | Whether mesh self-intersects |
| `self_intersection_count` | int | Number of self-intersecting face pairs |
| `estimated_min_thickness` | float | Thinnest wall/feature detected |
| `genus` | int | Topological genus (handles) |
| `euler_characteristic` | int | V - E + F |
| `duplicate_vertex_ratio` | float | Ratio of duplicate vertices |
| `aspect_ratio` | float | Ratio of longest to shortest bbox dimension |
| `nested_shell_count` | int | Number of nested internal shells |
| `overhang_face_ratio` | float | Ratio of faces with steep overhangs |

A rule engine evaluates these diagnostics against configurable thresholds and selects the best matching profile(s). The GUI shows the diagnostics and a short explanation of why a profile was suggested. Users can accept, tweak, or replace the suggested filter script.

---

## Profile Catalog

The system includes **40+ profiles** organized into categories. Each profile has:
- **Summary**: What the profile represents
- **Detection**: Heuristics that trigger this profile
- **Suggested Actions**: Default filter script actions

### Category: Clean / Minimal Repair

#### `clean`
- **Summary**: Model is already printable or nearly printable.
- **Detection**: `is_watertight == true`, `non_manifold_edge_count == 0`, `degenerate_face_count` low, `component_count == 1`.
- **Suggested Actions**: `trimesh_basic`, `validate`, `export_stl`.

#### `clean-minor-issues`
- **Summary**: Nearly clean model with minor fixable issues.
- **Detection**: `is_watertight == true`, few degenerate faces, minor normal inconsistencies.
- **Suggested Actions**: `trimesh_basic`, `remove_degenerate_faces`, `recalculate_normals`, `validate`.

---

### Category: Holes and Boundaries

#### `holes-only`
- **Summary**: Single-component model with open holes.
- **Detection**: `is_watertight == false`, `hole_count > 0`, `component_count == 1`, few degenerate faces.
- **Suggested Actions**: `trimesh_basic`, `fill_holes`, `recalculate_normals`, `validate`.

#### `open-bottom`
- **Summary**: Large opening on one face (flat base missing).
- **Detection**: Single large boundary loop on a planar face, rest of mesh closed.
- **Suggested Actions**: `fill_holes(method=planar)`, `recalculate_normals`, `validate`.

#### `mesh-with-holes-and-non-manifold`
- **Summary**: Holes combined with non-manifold edges.
- **Detection**: `hole_count > 0` AND `non_manifold_edge_count > 0`.
- **Suggested Actions**: `trimesh_basic`, `pymeshfix_repair`, `fill_holes`, `recalculate_normals`, `validate`.

---

### Category: Fragmented / Multi-Component

#### `fragmented`
- **Summary**: Model contains many small disconnected components.
- **Detection**: `component_count` high, `largest_component_pct` below threshold (e.g., < 80%).
- **Suggested Actions**: `remove_small_components(threshold)`, `merge_vertices(eps)`, `fill_holes`, `validate`.

#### `multiple-disconnected-large`
- **Summary**: More than one large component present.
- **Detection**: Multiple components each with > 10% of total mesh.
- **Suggested Actions**: `identify_components`, prompt user to `keep_all`, `boolean_union`, or `split_to_files`.

#### `floating-components`
- **Summary**: Disconnected components positioned away from main part.
- **Detection**: Small components with centroid distance > threshold from largest component.
- **Suggested Actions**: `remove_floating_components(distance_threshold)`, `validate`.

---

### Category: Topology Errors

#### `non-manifold`
- **Summary**: Topology errors (non-manifold edges/vertices) present.
- **Detection**: `non_manifold_edge_count > 0` or `non_manifold_vertex_count > 0`.
- **Suggested Actions**: `trimesh_basic`, `remove_degenerate_faces`, `pymeshfix_repair`, `recalculate_normals`, `validate`.

#### `non-manifold-shells`
- **Summary**: Multiple shells sharing problematic topology.
- **Detection**: Multiple shells with shared non-manifold edges between them.
- **Suggested Actions**: `separate_shells`, `pymeshfix_repair` per shell, `merge_shells`, `validate`.

#### `degenerate-heavy`
- **Summary**: Large number of degenerate (zero-area) faces.
- **Detection**: `degenerate_face_count` > threshold (e.g., > 1% of faces).
- **Suggested Actions**: `remove_degenerate_faces`, `merge_vertices(eps)`, `fill_holes`, `validate`.

#### `duplicate-vertices-heavy`
- **Summary**: High ratio of duplicate/near-duplicate vertices.
- **Detection**: `duplicate_vertex_ratio` > threshold (e.g., > 5%).
- **Suggested Actions**: `merge_vertices(eps)`, `remove_degenerate_faces`, `validate`.

#### `zero-volume`
- **Summary**: Closed shell with near-zero volume.
- **Detection**: `is_watertight == true` but `volume` ≈ 0.
- **Suggested Actions**: `check_winding_order`, `fix_normals`, `validate`; flag for user review.

---

### Category: Normal Issues

#### `normals-inconsistent`
- **Summary**: Face normals inconsistent or inverted.
- **Detection**: `normal_consistency < 0.8` or many flipped faces detected.
- **Suggested Actions**: `reorient_normals`, `unify_normals`, `remove_degenerate_faces`, `validate`.

#### `inverted-normals`
- **Summary**: All or most normals pointing inward.
- **Detection**: Computed volume is negative, or normal consistency check shows global inversion.
- **Suggested Actions**: `flip_normals`, `validate`.

#### `inverted-scale`
- **Summary**: Negative scale or global inverted normals from bad transform.
- **Detection**: Negative determinant in transform matrix, or all normals inverted.
- **Suggested Actions**: `fix_winding_order`, `flip_normals`, `validate`.

---

### Category: Self-Intersection

#### `self-intersecting`
- **Summary**: Mesh contains self-intersections or overlapping geometry.
- **Detection**: `self_intersections == true` or intersection test positive.
- **Suggested Actions**: `separate_shells`, `boolean_union`, `pymeshfix_repair`; escalate to `blender_remesh_boolean` if unresolved.

#### `self-touching`
- **Summary**: Parts touch but do not cleanly intersect.
- **Detection**: Near-zero distance between components without actual intersection.
- **Suggested Actions**: `boolean_union`, `merge_vertices(eps)`, `validate`.

#### `boolean-artifacts`
- **Summary**: Signs of prior bad boolean operations.
- **Detection**: Zero-area faces, T-junctions, duplicate edges at intersection seams.
- **Suggested Actions**: `remove_degenerate_faces`, `merge_vertices(eps)`, `pymeshfix_repair`, `validate`.

---

### Category: Internal Geometry / Hollow

#### `hollow-porous`
- **Summary**: Contains internal cavities, nested shells, or porous regions.
- **Detection**: Multiple nested components, volume anomalies, internal component detection.
- **Suggested Actions**: `identify_interior_components`, `remove_internal_components` (prompt), `fill_holes`, `validate`.

#### `nested-shells`
- **Summary**: Shells nested inside each other (cavities).
- **Detection**: `nested_shell_count > 0`, shell containment analysis.
- **Suggested Actions**: `identify_nested_shells`, prompt user to `keep`, `remove_inner`, or `boolean_subtract`.

#### `internal-geometry`
- **Summary**: Internal components fully enclosed by outer shell.
- **Detection**: Components with all vertices inside another component's bounding volume.
- **Suggested Actions**: `remove_internal_components`, `validate`.

#### `likely-intentional-hollow`
- **Summary**: Thin-walled hollow model likely intentional.
- **Detection**: Uniform thin wall thickness, single outer shell, clean topology.
- **Suggested Actions**: `validate`; prompt user before any fill/removal operations.

---

### Category: Thin Features / Wall Thickness

#### `thin-shell`
- **Summary**: Thin walls or features that may not print reliably.
- **Detection**: `estimated_min_thickness < min_thickness_threshold` (e.g., < 0.8mm).
- **Suggested Actions**: `identify_thin_regions`, `thicken_regions(thickness)`, `smooth`, `validate`.

#### `thin-walls-localized`
- **Summary**: Localized regions with thickness below threshold.
- **Detection**: Thin regions detected but majority of mesh is adequate thickness.
- **Suggested Actions**: `identify_thin_regions`, prompt user, optionally `thicken_regions`, `validate`.

#### `thin-pin-features`
- **Summary**: Many narrow elongated pin-like features.
- **Detection**: High aspect ratio features with small cross-section.
- **Suggested Actions**: `identify_thin_features`, prompt user, optionally `thicken_features`, `validate`.

---

### Category: Scan / Noisy Mesh

#### `noisy-scan`
- **Summary**: High-detail noisy mesh (3D scans) with many tiny defects.
- **Detection**: High `triangle_density`, many `degenerate_face_count`, many tiny components.
- **Suggested Actions**: `decimate(target_reduction)`, `remove_degenerate_faces`, `pymeshfix_repair`, `laplacian_smooth`, `validate`.

#### `repeated-pattern-artifact`
- **Summary**: Repetitive tiny artifacts typical of scan noise.
- **Detection**: Frequency analysis shows repetitive small-scale geometry.
- **Suggested Actions**: `smooth_taubin`, `decimate`, `remove_small_components`, `validate`.

#### `high-triangle-density`
- **Summary**: Extremely high triangle count relative to model size.
- **Detection**: `triangle_density` > threshold, or `triangle_count` > practical limit.
- **Suggested Actions**: `decimate(target_count)`, `validate`.

#### `low-triangle-density`
- **Summary**: Undersampled faceted surfaces.
- **Detection**: `triangle_density` very low, visible faceting, low `triangle_count` for bbox size.
- **Suggested Actions**: `subdivide`, `smooth`, `validate`.

#### `anisotropic-triangulation`
- **Summary**: Highly non-uniform triangle sizes across surface.
- **Detection**: High variance in triangle areas.
- **Suggested Actions**: `remesh_isotropic`, `validate`.

---

### Category: Complex Topology

#### `complex-high-genus`
- **Summary**: High genus or complex topology requiring remeshing.
- **Detection**: `genus` estimate high, uneven triangle sizing, repeated repair failures.
- **Suggested Actions**: `trimesh_basic`, `pymeshfix_repair`, then `blender_remesh` (aggressive) if needed.

#### `high-genus-localized`
- **Summary**: Small region with many handles or holes.
- **Detection**: Local genus analysis shows concentrated complexity.
- **Suggested Actions**: `identify_complex_regions`, `remesh_region`, `validate`.

---

### Category: Scale / Dimension Issues

#### `small-part`
- **Summary**: Model is very small (may be in wrong units).
- **Detection**: `bbox_volume` below small threshold (e.g., < 1 mm³).
- **Suggested Actions**: Prompt user for unit conversion, `scale(factor)`, `validate`.

#### `oversized`
- **Summary**: Model exceeds target printer build volume.
- **Detection**: Any `bbox` dimension exceeds configured build volume.
- **Suggested Actions**: Prompt user to `scale_to_fit`, `split_for_print`, or adjust settings.

#### `uniform-scale-error`
- **Summary**: Extreme non-uniform scale across axes.
- **Detection**: Aspect ratio between axes is extreme (e.g., > 100:1) unexpectedly.
- **Suggested Actions**: Prompt user, optionally `rescale_axis`, `validate`.

#### `mixed-units-suspect`
- **Summary**: Dimensions inconsistent with expected units.
- **Detection**: Bbox dimensions suggest mixed mm/inches or other unit mismatch.
- **Suggested Actions**: Prompt user for unit clarification, `convert_units`, `validate`.

#### `high-aspect-ratio`
- **Summary**: One dimension is much larger than others.
- **Detection**: `aspect_ratio` > threshold (e.g., > 20:1).
- **Suggested Actions**: `validate`; flag for user (may be intentional, e.g., sword blade).

---

### Category: Printability Hints

#### `overhang-heavy`
- **Summary**: Many faces with steep overhang angles.
- **Detection**: `overhang_face_ratio` > threshold (e.g., > 30% faces at > 45°).
- **Suggested Actions**: `validate`; suggest reorientation or support generation in slicer.

#### `bridge-heavy`
- **Summary**: Long unsupported spans detected.
- **Detection**: Bridge detection algorithm finds spans > threshold length.
- **Suggested Actions**: `validate`; suggest splitting or adding internal supports.

#### `requires-supports-by-default`
- **Summary**: Automatic slicing would generate heavy supports.
- **Detection**: Combined overhang and bridge analysis exceeds threshold.
- **Suggested Actions**: `validate`; suggest reorientation, splitting, or accept supports.

#### `requires-splitting`
- **Summary**: Model too large or complex; suggests splitting for build.
- **Detection**: Exceeds build volume, or complexity suggests multi-part print.
- **Suggested Actions**: `split_model`, `validate`.

---

### Category: Fine Detail / Precision

#### `text-labels-or-fine-engraving`
- **Summary**: Very small high-frequency geometry (text, engravings).
- **Detection**: Small-scale high-detail regions detected.
- **Suggested Actions**: `validate`; warn about minimum feature size for printer resolution.

#### `precision-model`
- **Summary**: Extremely fine features requiring high-resolution printing.
- **Detection**: Many features near or below typical printer resolution.
- **Suggested Actions**: `validate`; suggest high-resolution print settings.

---

## Profile Summary Table

| # | Profile | Category | Primary Detection Signal |
|---|---------|----------|--------------------------|
| 1 | `clean` | Clean | Watertight, manifold, single component |
| 2 | `clean-minor-issues` | Clean | Nearly clean with few degenerate faces |
| 3 | `holes-only` | Holes | Holes present, single component |
| 4 | `open-bottom` | Holes | Single large planar hole |
| 5 | `mesh-with-holes-and-non-manifold` | Holes | Holes + non-manifold edges |
| 6 | `fragmented` | Fragmented | Many small components |
| 7 | `multiple-disconnected-large` | Fragmented | Multiple large components |
| 8 | `floating-components` | Fragmented | Distant small components |
| 9 | `non-manifold` | Topology | Non-manifold edges/vertices |
| 10 | `non-manifold-shells` | Topology | Shells with shared bad topology |
| 11 | `degenerate-heavy` | Topology | Many degenerate faces |
| 12 | `duplicate-vertices-heavy` | Topology | High duplicate vertex ratio |
| 13 | `zero-volume` | Topology | Closed shell, zero volume |
| 14 | `normals-inconsistent` | Normals | Low normal consistency |
| 15 | `inverted-normals` | Normals | All normals pointing inward |
| 16 | `inverted-scale` | Normals | Negative scale transform |
| 17 | `self-intersecting` | Self-Intersection | Self-intersections detected |
| 18 | `self-touching` | Self-Intersection | Parts touch without intersection |
| 19 | `boolean-artifacts` | Self-Intersection | Bad boolean operation remnants |
| 20 | `hollow-porous` | Internal | Internal cavities, porous |
| 21 | `nested-shells` | Internal | Shells inside shells |
| 22 | `internal-geometry` | Internal | Enclosed internal components |
| 23 | `likely-intentional-hollow` | Internal | Clean thin-walled hollow |
| 24 | `thin-shell` | Thin Features | Global thin walls |
| 25 | `thin-walls-localized` | Thin Features | Localized thin regions |
| 26 | `thin-pin-features` | Thin Features | Narrow pin-like features |
| 27 | `noisy-scan` | Scan/Noisy | High density, many defects |
| 28 | `repeated-pattern-artifact` | Scan/Noisy | Repetitive noise patterns |
| 29 | `high-triangle-density` | Scan/Noisy | Excessive triangle count |
| 30 | `low-triangle-density` | Scan/Noisy | Undersampled, faceted |
| 31 | `anisotropic-triangulation` | Scan/Noisy | Non-uniform triangle sizes |
| 32 | `complex-high-genus` | Complex | High genus topology |
| 33 | `high-genus-localized` | Complex | Local complex region |
| 34 | `small-part` | Scale | Very small dimensions |
| 35 | `oversized` | Scale | Exceeds build volume |
| 36 | `uniform-scale-error` | Scale | Extreme non-uniform scale |
| 37 | `mixed-units-suspect` | Scale | Unit mismatch suspected |
| 38 | `high-aspect-ratio` | Scale | Extreme dimension ratio |
| 39 | `overhang-heavy` | Printability | Many steep overhangs |
| 40 | `bridge-heavy` | Printability | Long unsupported spans |
| 41 | `requires-supports-by-default` | Printability | Heavy support needed |
| 42 | `requires-splitting` | Printability | Too large/complex for single print |
| 43 | `text-labels-or-fine-engraving` | Fine Detail | Small high-frequency geometry |
| 44 | `precision-model` | Fine Detail | Features near resolution limit |

---

## Suggested Filter Script Format

A filter script is JSON or YAML describing ordered actions and optional parameters.

### JSON Example

```json
{
  "name": "holes-only-suggested",
  "version": "1.0.0",
  "meta": {
    "generated_by": "model_scan",
    "profile": "holes-only",
    "model_fingerprint": "<hash>",
    "timestamp": "<iso>"
  },
  "actions": [
    { "name": "trimesh_basic", "params": {} },
    { "name": "fill_holes", "params": { "max_hole_size": 1000 } },
    { "name": "recalculate_normals", "params": {} },
    { "name": "validate", "params": {} }
  ]
}
```

### YAML Example

```yaml
name: holes-only-suggested
version: "1.0.0"
meta:
  generated_by: model_scan
  profile: holes-only
  model_fingerprint: "<hash>"
  timestamp: "<iso>"
actions:
  - name: trimesh_basic
  - name: fill_holes
    params:
      max_hole_size: 1000
  - name: recalculate_normals
  - name: validate
```

---

## Extensibility and Tuning

- **Configurable thresholds**: All detection thresholds are stored in `config/` and adjustable via the GUI.
- **Pluggable rules**: New rules and composite heuristics can be added as detection modules.
- **Confidence scores**: Each profile includes a confidence score (0.0–1.0) shown in the UI.
- **Explanations**: Short explanation strings describe why a profile was selected.
- **Multi-profile detection**: Models may match multiple profiles; top 2–3 are shown with scores.

---

## Testing and Validation

- Add fixtures for each profile in `tests/fixtures/` following the naming convention: `profile_<profile_name>.stl`
- Unit tests verify detection accuracy and suggested filter script behavior in dry-run mode
- Each profile should have at least one representative test fixture

---

## Implementation Notes

- Profile detection implemented in `src/meshprep/core/profiles.py`
- Diagnostics computation in `src/meshprep/core/diagnostics.py`
- Filter script generation in `src/meshprep/core/filter_generator.py`
- CLI scanning via `scripts/model_scan.py`

---

## References

- See `docs/functional_spec.md` for the complete action catalog with parameters.
- See `docs/gui_spec.md` for how profiles are presented in the GUI.
