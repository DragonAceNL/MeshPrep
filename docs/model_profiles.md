Model Profiles
===============

Purpose
-------
This document describes the initial set of model profiles the system will detect automatically and the high-level heuristics used to assign a profile to a model. Each profile is associated with a suggested filter script (ordered actions) that can be reviewed and edited by the user.

How profiles are selected
------------------------
On model load the scanner computes a diagnostics vector using `trimesh` and other helpers. Typical diagnostics include:
- `is_watertight` (bool)
- `hole_count` (int)
- `component_count` (int)
- `largest_component_pct` (float)
- `non_manifold_edge_count` (int)
- `degenerate_face_count` (int)
- `normal_consistency` (0..1)
- `bbox` (dimensions)
- `avg_edge_length` (float)
- `triangle_density` (triangles / bbox_volume)
- `self_intersections` (bool)
- `estimated_min_thickness` (float)

A rule engine evaluates these diagnostics against configurable thresholds and selects the best matching profile. The GUI shows the diagnostics and a short explanation of why a profile was suggested. Users can accept, tweak, or replace the suggested filter script.

Initial model profiles (names only)
-----------------------------------
The initial release includes these 10 profiles. A suggested filter script will be generated for each.

1. `clean`
2. `holes-only`
3. `fragmented`
4. `non-manifold`
5. `normals-inconsistent`
6. `thin-shell`
7. `noisy-scan`
8. `self-intersecting`
9. `hollow-porous`
10. `complex-high-genus`

Profile summaries and detection heuristics
-----------------------------------------
(Brief summaries and typical detection signals; thresholds are configurable.)

- `clean`
  - Summary: Model is already printable or nearly printable.
  - Detection: `is_watertight == true`, `non_manifold_edge_count == 0`, `degenerate_face_count` low, `component_count == 1`.
  - Suggested actions: `trimesh_basic`, `recalculate_normals`, `export`.

- `holes-only`
  - Summary: Single-component model with open holes.
  - Detection: `is_watertight == false`, `hole_count > 0`, `component_count == 1`, few degenerate faces.
  - Suggested actions: `trimesh_basic`, `fill_holes`, `recalculate_normals`, `validate`.

- `fragmented`
  - Summary: Model contains many small disconnected components.
  - Detection: `component_count` high, `largest_component_pct` below threshold.
  - Suggested actions: `remove_small_components(threshold)`, `merge_vertices(eps)`, `fill_holes`, `validate`.

- `non-manifold`
  - Summary: Topology errors (non-manifold edges/vertices) present.
  - Detection: `non_manifold_edge_count > 0` or `non_manifold_vertex_count > 0`.
  - Suggested actions: `trimesh_basic`, `remove_degenerate_faces`, `pymeshfix_repair`, `recalculate_normals`, `validate`.

- `normals-inconsistent`
  - Summary: Face normals inconsistent or inverted.
  - Detection: `normal_consistency < 0.8` or many flipped faces.
  - Suggested actions: `reorient_normals`, `unify_normals`, `remove_degenerate_faces`, `validate`.

- `thin-shell`
  - Summary: Thin walls or features that may not print reliably.
  - Detection: `estimated_min_thickness < min_thickness_threshold`.
  - Suggested actions: `identify_thin_regions`, `thicken_regions(thickness)`, `smooth`, `validate` (prompt user before irreversible changes).

- `noisy-scan`
  - Summary: High-detail noisy mesh (scans) with many tiny defects.
  - Detection: high `triangle_density`, many `degenerate_face_count`, many tiny components.
  - Suggested actions: `decimate(target_reduction)`, `remove_degenerate_faces`, `pymeshfix_repair`, `laplacian_smooth`, `validate`.

- `self-intersecting`
  - Summary: Mesh contains self-intersections or overlapping geometry.
  - Detection: `self_intersections == true` or intersection tests positive.
  - Suggested actions: `separate_shells`, `boolean_union`, `pymeshfix_repair`; escalate to `blender_remesh_boolean` if unresolved.

- `hollow-porous`
  - Summary: Contains internal cavities, nested shells, or porous regions.
  - Detection: multiple nested components, volume anomalies, internal component detection.
  - Suggested actions: `identify_interior_components`, `remove_internal_components` (prompt), `fill_holes`, `validate`.

- `complex-high-genus`
  - Summary: High genus or complex topology requiring remeshing.
  - Detection: genus estimate high, uneven triangle sizing, repeated repair failures.
  - Suggested actions: `trimesh_basic`, `pymeshfix_repair`, then `blender_remesh` (aggressive) if needed.

Suggested filter script format
-----------------------------
A filter script is JSON or YAML describing ordered actions and optional parameters. Minimal example (JSON):

{
  "name": "holes-only-suggested",
  "meta": { "generated_by": "model_scan", "model_fingerprint": "<hash>", "timestamp": "<iso>" },
  "actions": [
    { "name": "trimesh_basic" },
    { "name": "fill_holes", "params": { "max_hole_size": 1000 } },
    { "name": "recalculate_normals" },
    { "name": "validate" }
  ]
}

Extensibility and tuning
------------------------
- Thresholds used for detection are configuration values stored in `config/` and adjustable via the GUI.
- The rule set is pluggable: new rules and composite heuristics can be added as detection modules.
- Each profile includes a confidence score and a short explanation string shown in the UI so users know why a profile was selected.

Testing and validation
----------------------
- Add fixtures for each profile in `tests/fixtures/` and unit tests that verify detection and the suggested filter script behavior in dry-run mode.

Next actions
------------
- Create example filter script presets for each profile in `filters/` (or `pipelines/`) as starting points.
- Implement `scripts/model_scan.py` to compute diagnostics and write suggested filter scripts.

Additional detectable profiles (expanded list)
---------------------------------------------
The system can detect many more profiles using the diagnostics vector. The following list includes additional candidate profiles with short detection hints.

- `duplicate-vertices-heavy` — high duplicate-vertex ratio
- `degenerate-heavy` — large `degenerate_face_count`
- `small-part` — bbox volume below a small threshold
- `oversized` — bbox exceeds target printer build volume
- `thin-walls-localized` — localized regions with thickness < threshold
- `uniform-scale-error` — extreme non-uniform scale across axes
- `mixed-units-suspect` — bbox dimensions inconsistent with expected units
- `multiple-disconnected-large` — more than one large component present
- `nested-shells` — shells nested inside each other (cavities)
- `internal-geometry` — internal components fully enclosed by outer shell
- `open-bottom` — large opening on one face (flat base missing)
- `high-aspect-ratio` — one dimension is much larger than others
- `overhang-heavy` — many faces with steep overhang angles (slicer hint)
- `bridge-heavy` — long unsupported spans detected (slicer hint)
- `thin-pin-features` — many narrow elongated pin-like features
- `text-labels-or-fine-engraving` — very small high-frequency geometry
- `mesh-with-holes-and-no-manifold` — holes combined with non-manifold edges
- `self-touching` — parts touch but do not cleanly intersect
- `floating-components` — disconnected components positioned away from main part
- `high-genus-localized` — small region with many handles or holes
- `high-triangle-density` — local regions with extreme triangle density
- `low-triangle-density` — undersampled faceted surfaces
- `anisotropic-triangulation` — highly non-uniform triangle sizes across surface
- `repeated-pattern-artifact` — repetitive tiny artifacts typical of scan noise
- `boolean-artifacts` — signs of prior bad boolean ops (zero-area faces / edges)
- `non-manifold-shells` — multiple shells sharing problematic topology
- `zero-volume` — closed shell with near-zero volume
- `inverted-scale` — negative scale or global inverted normals
- `precision-model` — extremely fine features that need high-resolution printing
- `requires-splitting` — model too large or complex; suggests splitting for build
- `requires-supports-by-default` — automatic slicing would generate heavy supports
- `likely-intentional-hollow` — thin-walled hollow model likely intentional; prompt user before removing

Integration notes
-----------------
- Add these profiles incrementally and provide sample fixtures to tune detection thresholds.
- The GUI should present the top 2–3 matching profiles with confidence scores and an explanation of the main diagnostics that triggered them.
- Users should be able to save a tuned preset for a model and optionally contribute it to shared presets.

