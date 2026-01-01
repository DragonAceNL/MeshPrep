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
| `ngon_count` | int | Number of n-gon faces (> 4 vertices) |
| `quad_count` | int | Number of quad faces |
| `concave_face_count` | int | Number of concave (non-convex) faces |
| `coplanar_face_count` | int | Number of coplanar adjacent faces |
| `t_junction_count` | int | Number of T-junction vertices |
| `boundary_edge_count` | int | Number of naked/boundary edges |
| `symmetry_score` | float | 0..1 bilateral symmetry measure |
| `up_axis` | str | Detected up axis (Y, Z, or unknown) |
| `face_orientation_variance` | float | Variance in face orientations |

A rule engine evaluates these diagnostics against configurable thresholds and selects the best matching profile(s). The GUI shows the diagnostics and a short explanation of why a profile was suggested. Users can accept, tweak, or replace the suggested filter script.

---

## Profile Catalog

The system includes **75+ profiles** organized into categories. Each profile has:
- **Summary**: What the profile represents
- **Detection**: Heuristics that trigger this profile
- **Suggested Actions**: Default filter script actions

---

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

#### `partial-boundary-loop`
- **Summary**: Incomplete boundary edges that don't form closed loops.
- **Detection**: Boundary edges present but don't form complete loops.
- **Suggested Actions**: `stitch_boundaries`, `fill_holes`, `validate`.

#### `many-small-holes`
- **Summary**: Numerous small holes scattered across the surface.
- **Detection**: `hole_count` high but individual hole sizes small.
- **Suggested Actions**: `fill_holes(max_hole_size=small)`, `smooth`, `validate`.

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

#### `debris-particles`
- **Summary**: Tiny isolated triangles or micro-components.
- **Detection**: Many components with < 10 triangles each.
- **Suggested Actions**: `remove_small_components(min_faces=10)`, `validate`.

#### `split-along-seam`
- **Summary**: Model split into parts along what should be a continuous surface.
- **Detection**: Multiple components with matching boundary edges.
- **Suggested Actions**: `stitch_boundaries(tolerance)`, `merge_vertices`, `validate`.

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

#### `t-junction-heavy`
- **Summary**: Many T-junction vertices causing topology issues.
- **Detection**: `t_junction_count` > threshold.
- **Suggested Actions**: `fix_t_junctions`, `merge_vertices(eps)`, `validate`.

#### `duplicate-faces`
- **Summary**: Overlapping or duplicate triangles present.
- **Detection**: Face duplication detection finds overlapping triangles.
- **Suggested Actions**: `remove_duplicate_faces`, `validate`.

#### `inconsistent-winding`
- **Summary**: Face winding order is inconsistent across the mesh.
- **Detection**: Adjacent faces have inconsistent vertex ordering.
- **Suggested Actions**: `fix_winding_order`, `recalculate_normals`, `validate`.

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

#### `mixed-flipped-faces`
- **Summary**: Random faces flipped throughout the mesh.
- **Detection**: Scattered flipped faces detected (not consistent pattern).
- **Suggested Actions**: `unify_normals`, `recalculate_normals`, `validate`.

#### `smoothing-group-artifacts`
- **Summary**: Normal artifacts from incorrect smoothing groups.
- **Detection**: Sharp normal discontinuities at unexpected locations.
- **Suggested Actions**: `recalculate_normals(angle_threshold)`, `validate`.

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

#### `overlapping-shells`
- **Summary**: Multiple shells occupying the same space.
- **Detection**: Shells with significant volume overlap.
- **Suggested Actions**: `boolean_union`, `validate`.

#### `interpenetrating-parts`
- **Summary**: Distinct parts of the model penetrating each other.
- **Detection**: Separate logical parts with intersection detected.
- **Suggested Actions**: `boolean_union` or `separate_parts`, prompt user, `validate`.

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

#### `internal-faces`
- **Summary**: Faces inside the mesh that serve no purpose.
- **Detection**: Faces completely enclosed within mesh volume.
- **Suggested Actions**: `remove_internal_faces`, `validate`.

#### `double-walled`
- **Summary**: Model has double walls (two surfaces close together).
- **Detection**: Parallel surfaces at near-constant small distance.
- **Suggested Actions**: Prompt user to `keep` or `merge_walls`, `validate`.

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

#### `paper-thin-faces`
- **Summary**: Single-layer faces with no thickness (not a solid).
- **Detection**: Open surface with no volume, planar or near-planar regions.
- **Suggested Actions**: `solidify(thickness)`, `validate`.

#### `knife-edge`
- **Summary**: Extremely sharp edges that won't print.
- **Detection**: Edges with near-zero dihedral angle forming sharp ridges.
- **Suggested Actions**: `chamfer_edges` or `fillet_edges`, `validate`.

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

#### `spike-artifacts`
- **Summary**: Spike or needle-like protrusions from scan errors.
- **Detection**: Vertices with extreme displacement from local surface.
- **Suggested Actions**: `remove_spikes`, `smooth`, `validate`.

#### `scan-alignment-seam`
- **Summary**: Visible seams from multi-scan alignment.
- **Detection**: Linear artifacts or steps from scan registration.
- **Suggested Actions**: `smooth_seams`, `blend_regions`, `validate`.

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

#### `topological-noise`
- **Summary**: Small topological features that add unnecessary complexity.
- **Detection**: Many small handles or tunnels that could be simplified.
- **Suggested Actions**: `simplify_topology`, `fill_small_holes`, `validate`.

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

#### `microscale`
- **Summary**: Model has features at microscale level.
- **Detection**: Features detected at sub-millimeter scale.
- **Suggested Actions**: `validate`; suggest scaling up or specialized micro-printing.

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

#### `island-regions`
- **Summary**: Regions that would print as unsupported islands per layer.
- **Detection**: Layer simulation shows disconnected regions.
- **Suggested Actions**: `validate`; suggest supports or reorientation.

#### `trapped-volume`
- **Summary**: Enclosed volumes that could trap resin or powder.
- **Detection**: Internal cavities with no drainage path.
- **Suggested Actions**: `add_drain_holes` or prompt user, `validate`.

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

#### `mechanical-tolerances`
- **Summary**: Model appears to have mechanical fit requirements.
- **Detection**: Cylindrical features, gear teeth, or assembly interfaces detected.
- **Suggested Actions**: `validate`; warn about shrinkage and tolerance compensation.

---

### Category: Polygon / Face Issues

#### `ngon-heavy`
- **Summary**: Many n-gon faces (polygons with more than 4 vertices).
- **Detection**: `ngon_count` > threshold.
- **Suggested Actions**: `triangulate`, `validate`.

#### `quad-dominant`
- **Summary**: Model is primarily quads, needs triangulation.
- **Detection**: `quad_count` > 50% of faces.
- **Suggested Actions**: `triangulate`, `validate`.

#### `concave-faces`
- **Summary**: Concave (non-convex) polygons present.
- **Detection**: `concave_face_count` > 0.
- **Suggested Actions**: `triangulate`, `validate`.

#### `coplanar-faces`
- **Summary**: Many coplanar adjacent faces that could be merged.
- **Detection**: `coplanar_face_count` > threshold.
- **Suggested Actions**: `merge_coplanar_faces` (optional), `validate`.

#### `sliver-triangles`
- **Summary**: Very thin elongated triangles with poor aspect ratio.
- **Detection**: Triangles with aspect ratio > threshold.
- **Suggested Actions**: `remesh_isotropic`, `validate`.

#### `zero-area-faces`
- **Summary**: Faces with zero or near-zero area.
- **Detection**: Face area calculation finds zero-area triangles.
- **Suggested Actions**: `remove_degenerate_faces`, `validate`.

---

### Category: Edge Issues

#### `boundary-edges-open`
- **Summary**: Open boundary edges (naked edges).
- **Detection**: `boundary_edge_count` > 0 on what should be closed mesh.
- **Suggested Actions**: `fill_holes`, `stitch_boundaries`, `validate`.

#### `crease-artifacts`
- **Summary**: Unwanted hard edges from crease/sharp edge data.
- **Detection**: Sharp edges detected at unexpected locations.
- **Suggested Actions**: `remove_creases`, `smooth`, `validate`.

#### `edge-collapse-needed`
- **Summary**: Very short edges that should be collapsed.
- **Detection**: Edges with length below threshold.
- **Suggested Actions**: `collapse_short_edges(threshold)`, `validate`.

#### `long-thin-edges`
- **Summary**: Abnormally long edges creating poor triangulation.
- **Detection**: Edge length variance is very high.
- **Suggested Actions**: `remesh`, `validate`.

---

### Category: Orientation Issues

#### `wrong-up-axis`
- **Summary**: Model oriented with wrong axis as up.
- **Detection**: `up_axis` detection doesn't match expected (Z-up vs Y-up).
- **Suggested Actions**: `rotate_to_z_up` or `rotate_to_y_up`, `validate`.

#### `upside-down`
- **Summary**: Model appears to be inverted/upside down.
- **Detection**: Base detection suggests model is flipped.
- **Suggested Actions**: `rotate(180, axis)`, `validate`.

#### `tilted`
- **Summary**: Model is tilted and not aligned to axes.
- **Detection**: Principal axes don't align with world axes.
- **Suggested Actions**: `align_to_axes`, `validate`.

#### `off-origin`
- **Summary**: Model is positioned far from origin.
- **Detection**: Centroid or bbox is far from world origin.
- **Suggested Actions**: `center_to_origin`, `validate`.

---

### Category: Symmetry Issues

#### `asymmetric-unexpected`
- **Summary**: Model should be symmetric but isn't.
- **Detection**: `symmetry_score` low for model type that typically is symmetric.
- **Suggested Actions**: `detect_symmetry_plane`, prompt user, optionally `mirror_repair`, `validate`.

#### `mirror-seam-artifacts`
- **Summary**: Problems at the mirror/symmetry plane.
- **Detection**: Artifacts detected along detected symmetry plane.
- **Suggested Actions**: `merge_vertices_at_seam`, `validate`.

#### `radial-symmetry-broken`
- **Summary**: Radial symmetry is inconsistent.
- **Detection**: Radial pattern detected but not consistent.
- **Suggested Actions**: Prompt user, optionally `repair_radial_symmetry`, `validate`.

---

### Category: CAD Conversion Artifacts

#### `tessellation-artifacts`
- **Summary**: Artifacts from NURBS to mesh conversion.
- **Detection**: Regular grid-like patterns, faceting on curved surfaces.
- **Suggested Actions**: `smooth`, `subdivide`, `validate`; suggest re-export with finer tessellation.

#### `cad-tolerance-gaps`
- **Summary**: Small gaps from CAD model tolerance issues.
- **Detection**: Small gaps between what should be connected surfaces.
- **Suggested Actions**: `stitch_boundaries(tolerance)`, `fill_holes`, `validate`.

#### `parametric-seams`
- **Summary**: Visible seams from parametric surface boundaries.
- **Detection**: Linear seams on curved surfaces from UV boundaries.
- **Suggested Actions**: `merge_vertices`, `smooth_seams`, `validate`.

#### `fillet-faceting`
- **Summary**: Fillets and rounds are heavily faceted.
- **Detection**: Curved features have low polygon count.
- **Suggested Actions**: `subdivide`, `smooth`, `validate`.

#### `chamfer-artifacts`
- **Summary**: Artifacts on chamfered edges.
- **Detection**: T-junctions or degenerate faces at chamfer regions.
- **Suggested Actions**: `fix_t_junctions`, `merge_vertices`, `validate`.

---

### Category: Sculpt / Artistic Mesh

#### `sculpt-base-mesh`
- **Summary**: Low-poly base mesh for sculpting (not final).
- **Detection**: Very low polygon count with quad topology.
- **Suggested Actions**: `validate`; warn this appears to be a base mesh.

#### `dynmesh-artifacts`
- **Summary**: Artifacts from dynamic remeshing (ZBrush Dynamesh, etc).
- **Detection**: Uniform triangle size with small holes or intersections.
- **Suggested Actions**: `fill_holes`, `pymeshfix_repair`, `validate`.

#### `remesh-boundary-artifacts`
- **Summary**: Artifacts at boundaries of remeshed regions.
- **Detection**: Topology changes at region boundaries.
- **Suggested Actions**: `smooth_boundaries`, `merge_vertices`, `validate`.

#### `multires-mismatch`
- **Summary**: Multi-resolution sculpting levels not properly collapsed.
- **Detection**: Inconsistent detail levels across surface.
- **Suggested Actions**: `remesh_uniform`, `validate`.

#### `zbrush-polygroup-seams`
- **Summary**: Visible seams at polygroup boundaries.
- **Detection**: Vertex splits or discontinuities at group boundaries.
- **Suggested Actions**: `merge_vertices`, `smooth_seams`, `validate`.

---

### Category: Photogrammetry / Reconstruction

#### `photogrammetry-holes`
- **Summary**: Holes from incomplete photogrammetry reconstruction.
- **Detection**: Multiple holes in otherwise dense mesh.
- **Suggested Actions**: `fill_holes`, `smooth`, `validate`.

#### `photogrammetry-noise`
- **Summary**: Noisy surface from photogrammetry reconstruction.
- **Detection**: High-frequency noise on surface, color data present.
- **Suggested Actions**: `smooth_taubin`, `decimate`, `validate`.

#### `point-cloud-incomplete`
- **Summary**: Incomplete surface from point cloud reconstruction.
- **Detection**: Many boundary edges, sparse regions.
- **Suggested Actions**: `fill_holes`, `reconstruct_surface`, `validate`.

#### `texture-projection-artifacts`
- **Summary**: Mesh artifacts from texture-based reconstruction.
- **Detection**: Geometry follows texture boundaries unnaturally.
- **Suggested Actions**: `smooth`, `remesh`, `validate`.

#### `floating-reconstruction-debris`
- **Summary**: Floating debris from reconstruction errors.
- **Detection**: Many small disconnected components.
- **Suggested Actions**: `remove_small_components`, `validate`.

---

### Category: Import / Data Issues

#### `corrupted-data`
- **Summary**: File appears to have corrupted data.
- **Detection**: Invalid vertex coordinates (NaN, Inf), malformed faces.
- **Suggested Actions**: `remove_invalid_data`, `validate`; suggest re-export from source.

#### `incomplete-import`
- **Summary**: Import appears incomplete or truncated.
- **Detection**: Unexpected end of file, missing data.
- **Suggested Actions**: `validate`; suggest checking source file.

#### `format-conversion-artifacts`
- **Summary**: Artifacts from file format conversion.
- **Detection**: Data loss patterns typical of format conversion.
- **Suggested Actions**: `repair`, `validate`; suggest native format if available.

#### `empty-mesh`
- **Summary**: File contains no geometry.
- **Detection**: Zero vertices or zero faces.
- **Suggested Actions**: Flag error; cannot process.

#### `single-triangle`
- **Summary**: Mesh contains only one or very few triangles.
- **Detection**: `triangle_count` < threshold (e.g., < 4).
- **Suggested Actions**: Flag warning; likely not a valid 3D model.

#### `ascii-precision-loss`
- **Summary**: Precision loss from ASCII format export.
- **Detection**: Vertex coordinates show truncation artifacts.
- **Suggested Actions**: `validate`; suggest binary format re-export.

---

### Category: Subdivision / LOD

#### `over-subdivided`
- **Summary**: Model has been over-subdivided unnecessarily.
- **Detection**: Very high triangle count with little geometric variation.
- **Suggested Actions**: `decimate`, `validate`.

#### `under-subdivided`
- **Summary**: Model needs more subdivision for smooth appearance.
- **Detection**: Low polygon count on curved surfaces.
- **Suggested Actions**: `subdivide`, `smooth`, `validate`.

#### `lod-mismatch`
- **Summary**: Inconsistent level of detail across the model.
- **Detection**: Large variance in triangle density across surface.
- **Suggested Actions**: `remesh_adaptive`, `validate`.

#### `subdivision-crease-loss`
- **Summary**: Lost crease data from subdivision surface.
- **Detection**: Rounded edges where sharp edges expected.
- **Suggested Actions**: `validate`; suggest re-export with creases preserved.

---

### Category: Material / Color Data

#### `vertex-color-only`
- **Summary**: Model has vertex colors but no UVs/textures.
- **Detection**: Vertex color data present, no UV coordinates.
- **Suggested Actions**: `validate`; note color data will be lost in STL export.

#### `multi-material-seams`
- **Summary**: Seams or artifacts at material boundaries.
- **Detection**: Discontinuities at material assignment boundaries.
- **Suggested Actions**: `merge_vertices_at_seams`, `validate`.

#### `uv-seam-splits`
- **Summary**: Vertex splits from UV seams affecting geometry.
- **Detection**: Duplicate vertices at UV boundaries.
- **Suggested Actions**: `merge_vertices`, `validate`.

---

## Profile Summary Table

| # | Profile | Category | Primary Detection Signal |
|---|---------|----------|--------------------------|
| 1 | `clean` | Clean | Watertight, manifold, single component |
| 2 | `clean-minor-issues` | Clean | Nearly clean with few degenerate faces |
| 3 | `holes-only` | Holes | Holes present, single component |
| 4 | `open-bottom` | Holes | Single large planar hole |
| 5 | `mesh-with-holes-and-non-manifold` | Holes | Holes + non-manifold edges |
| 6 | `partial-boundary-loop` | Holes | Incomplete boundary edges |
| 7 | `many-small-holes` | Holes | Numerous small holes scattered |
| 8 | `fragmented` | Fragmented | Many small components |
| 9 | `multiple-disconnected-large` | Fragmented | Multiple large components |
| 10 | `floating-components` | Fragmented | Distant small components |
| 11 | `debris-particles` | Fragmented | Tiny isolated triangles |
| 12 | `split-along-seam` | Fragmented | Split parts with matching boundaries |
| 13 | `non-manifold` | Topology | Non-manifold edges/vertices |
| 14 | `non-manifold-shells` | Topology | Shells with shared bad topology |
| 15 | `degenerate-heavy` | Topology | Many degenerate faces |
| 16 | `duplicate-vertices-heavy` | Topology | High duplicate vertex ratio |
| 17 | `zero-volume` | Topology | Closed shell, zero volume |
| 18 | `t-junction-heavy` | Topology | Many T-junction vertices |
| 19 | `duplicate-faces` | Topology | Overlapping triangles |
| 20 | `inconsistent-winding` | Topology | Inconsistent vertex ordering |
| 21 | `normals-inconsistent` | Normals | Low normal consistency |
| 22 | `inverted-normals` | Normals | All normals pointing inward |
| 23 | `inverted-scale` | Normals | Negative scale transform |
| 24 | `mixed-flipped-faces` | Normals | Random flipped faces |
| 25 | `smoothing-group-artifacts` | Normals | Normal discontinuities |
| 26 | `self-intersecting` | Self-Intersection | Self-intersections detected |
| 27 | `self-touching` | Self-Intersection | Parts touch without intersection |
| 28 | `boolean-artifacts` | Self-Intersection | Bad boolean operation remnants |
| 29 | `overlapping-shells` | Self-Intersection | Shells in same space |
| 30 | `interpenetrating-parts` | Self-Intersection | Parts penetrating each other |
| 31 | `hollow-porous` | Internal | Internal cavities, porous |
| 32 | `nested-shells` | Internal | Shells inside shells |
| 33 | `internal-geometry` | Internal | Enclosed internal components |
| 34 | `likely-intentional-hollow` | Internal | Clean thin-walled hollow |
| 35 | `internal-faces` | Internal | Faces inside mesh |
| 36 | `double-walled` | Internal | Parallel double surfaces |
| 37 | `thin-shell` | Thin Features | Global thin walls |
| 38 | `thin-walls-localized` | Thin Features | Localized thin regions |
| 39 | `thin-pin-features` | Thin Features | Narrow pin-like features |
| 40 | `paper-thin-faces` | Thin Features | Single-layer, no thickness |
| 41 | `knife-edge` | Thin Features | Extremely sharp edges |
| 42 | `noisy-scan` | Scan/Noisy | High density, many defects |
| 43 | `repeated-pattern-artifact` | Scan/Noisy | Repetitive noise patterns |
| 44 | `high-triangle-density` | Scan/Noisy | Excessive triangle count |
| 45 | `low-triangle-density` | Scan/Noisy | Undersampled, faceted |
| 46 | `anisotropic-triangulation` | Scan/Noisy | Non-uniform triangle sizes |
| 47 | `spike-artifacts` | Scan/Noisy | Spike protrusions |
| 48 | `scan-alignment-seam` | Scan/Noisy | Multi-scan seams |
| 49 | `complex-high-genus` | Complex | High genus topology |
| 50 | `high-genus-localized` | Complex | Local complex region |
| 51 | `topological-noise` | Complex | Small unnecessary features |
| 52 | `small-part` | Scale | Very small dimensions |
| 53 | `oversized` | Scale | Exceeds build volume |
| 54 | `uniform-scale-error` | Scale | Extreme non-uniform scale |
| 55 | `mixed-units-suspect` | Scale | Unit mismatch suspected |
| 56 | `high-aspect-ratio` | Scale | Extreme dimension ratio |
| 57 | `microscale` | Scale | Microscale features |
| 58 | `overhang-heavy` | Printability | Many steep overhangs |
| 59 | `bridge-heavy` | Printability | Long unsupported spans |
| 60 | `requires-supports-by-default` | Printability | Heavy support needed |
| 61 | `requires-splitting` | Printability | Too large/complex for single print |
| 62 | `island-regions` | Printability | Unsupported islands per layer |
| 63 | `trapped-volume` | Printability | Enclosed volumes, no drainage |
| 64 | `text-labels-or-fine-engraving` | Fine Detail | Small high-frequency geometry |
| 65 | `precision-model` | Fine Detail | Features near resolution limit |
| 66 | `mechanical-tolerances` | Fine Detail | Mechanical fit requirements |
| 67 | `ngon-heavy` | Polygon | Many n-gon faces |
| 68 | `quad-dominant` | Polygon | Primarily quad faces |
| 69 | `concave-faces` | Polygon | Non-convex polygons |
| 70 | `coplanar-faces` | Polygon | Coplanar adjacent faces |
| 71 | `sliver-triangles` | Polygon | Thin elongated triangles |
| 72 | `zero-area-faces` | Polygon | Zero-area faces |
| 73 | `boundary-edges-open` | Edge | Open boundary edges |
| 74 | `crease-artifacts` | Edge | Unwanted hard edges |
| 75 | `edge-collapse-needed` | Edge | Very short edges |
| 76 | `long-thin-edges` | Edge | Abnormally long edges |
| 77 | `wrong-up-axis` | Orientation | Wrong axis as up |
| 78 | `upside-down` | Orientation | Model inverted |
| 79 | `tilted` | Orientation | Not aligned to axes |
| 80 | `off-origin` | Orientation | Far from origin |
| 81 | `asymmetric-unexpected` | Symmetry | Should be symmetric but isn't |
| 82 | `mirror-seam-artifacts` | Symmetry | Problems at symmetry plane |
| 83 | `radial-symmetry-broken` | Symmetry | Inconsistent radial symmetry |
| 84 | `tessellation-artifacts` | CAD | NURBS conversion artifacts |
| 85 | `cad-tolerance-gaps` | CAD | Gaps from tolerance issues |
| 86 | `parametric-seams` | CAD | Seams from UV boundaries |
| 87 | `fillet-faceting` | CAD | Faceted fillets/rounds |
| 88 | `chamfer-artifacts` | CAD | Artifacts on chamfers |
| 89 | `sculpt-base-mesh` | Sculpt | Low-poly base mesh |
| 90 | `dynmesh-artifacts` | Sculpt | Dynamic remesh artifacts |
| 91 | `remesh-boundary-artifacts` | Sculpt | Boundary region artifacts |
| 92 | `multires-mismatch` | Sculpt | Multi-res level issues |
| 93 | `zbrush-polygroup-seams` | Sculpt | Polygroup boundary seams |
| 94 | `photogrammetry-holes` | Photogrammetry | Incomplete reconstruction holes |
| 95 | `photogrammetry-noise` | Photogrammetry | Noisy reconstruction surface |
| 96 | `point-cloud-incomplete` | Photogrammetry | Incomplete point cloud surface |
| 97 | `texture-projection-artifacts` | Photogrammetry | Texture-based artifacts |
| 98 | `floating-reconstruction-debris` | Photogrammetry | Floating debris |
| 99 | `corrupted-data` | Import | Invalid data (NaN, Inf) |
| 100 | `incomplete-import` | Import | Truncated file |
| 101 | `format-conversion-artifacts` | Import | Format conversion issues |
| 102 | `empty-mesh` | Import | No geometry |
| 103 | `single-triangle` | Import | Very few triangles |
| 104 | `ascii-precision-loss` | Import | ASCII truncation |
| 105 | `over-subdivided` | Subdivision | Too many subdivisions |
| 106 | `under-subdivided` | Subdivision | Needs more subdivision |
| 107 | `lod-mismatch` | Subdivision | Inconsistent detail levels |
| 108 | `subdivision-crease-loss` | Subdivision | Lost crease data |
| 109 | `vertex-color-only` | Material | Vertex colors, no UVs |
| 110 | `multi-material-seams` | Material | Material boundary seams |
| 111 | `uv-seam-splits` | Material | UV seam vertex splits |

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
- **Profile combinations**: Some issues co-occur; the system can suggest combined filter scripts.

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
