# Filter Actions Reference

## Overview

This document provides a comprehensive reference for all filter actions available in MeshPrep. Actions are organized by category and tool source. Each action includes:

- **Description**: What the action does
- **Tool**: Which library provides this action
- **Parameters**: Available parameters with types and defaults
- **Risk Level**: Impact on visual fidelity (🟢 Safe, 🟡 Low, 🟠 Medium, 🔴 High)
- **Use Cases**: When to use this action

---

## Action Categories

| Category | Description | Actions |
|----------|-------------|---------|
| Loading & Cleanup | Basic mesh loading and cleanup | 7 |
| Hole Filling | Close holes and boundaries | 3 |
| Normal Correction | Fix face normals | 5 |
| Component Management | Handle disconnected parts | 5 |
| Repair & Manifold | Fix topology issues | 5 |
| Simplification | Reduce complexity | 6 |
| Geometry Analysis | Analyze features | 4 |
| Boolean Operations | Combine/split meshes | 4 |
| Validation | Check mesh quality | 7 |
| Export | Save mesh to file | 5 |
| Blender (Escalation) | Advanced Blender operations | 6 |

---

## Tool Sources

| Tool | Type | Description |
|------|------|-------------|
| `trimesh` | Python library | Primary mesh processing library |
| `pymeshfix` | Python library | Robust mesh repair |
| `meshio` | Python library | File format I/O |
| `blender` | External tool | Advanced operations via Blender CLI |
| `internal` | MeshPrep | Custom implementations |

---

## Category: Loading & Basic Cleanup

### `load_stl`

Load an STL file into memory.

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟢 Safe |
| **Modifies Mesh** | No (creates new) |

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `path` | string | required | Path to STL file |

---

### `trimesh_basic`

Apply basic cleanup: merge duplicate vertices, remove degenerate faces, remove infinite values.

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟢 Safe |
| **Modifies Mesh** | Yes |

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `merge_tex` | bool | true | Merge texture coordinates |
| `merge_norm` | bool | true | Merge normals |

**Implementation:**
```python
def trimesh_basic(mesh, params):
    mesh.merge_vertices()
    mesh.remove_degenerate_faces()
    mesh.remove_infinite_values()
    return mesh
```

---

### `merge_vertices`

Weld duplicate vertices within a tolerance. Reduces vertex count and fixes small gaps.

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟢 Safe |
| **Modifies Mesh** | Yes |

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `eps` | float | 1e-8 | Distance threshold for merging |

**Use Cases:**
- Fix duplicate vertices from bad exports
- Close micro-gaps between faces
- Reduce vertex count without losing detail

---

### `remove_degenerate_faces`

Remove faces with zero area or invalid topology.

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟢 Safe |
| **Modifies Mesh** | Yes |

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `height` | float | 1e-8 | Height threshold for degenerate detection |

---

### `remove_duplicate_faces`

Remove faces that are exact duplicates.

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟢 Safe |
| **Modifies Mesh** | Yes |

**Parameters:** None

---

### `remove_infinite_values`

Remove vertices or faces containing NaN or Inf values.

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟢 Safe |
| **Modifies Mesh** | Yes |

**Parameters:** None

---

### `remove_unreferenced_vertices`

Remove vertices not referenced by any face.

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟢 Safe |
| **Modifies Mesh** | Yes |

**Parameters:** None

---

## Category: Hole Filling

### `fill_holes`

Fill holes in the mesh up to a maximum size.

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟡 Low to 🟠 Medium |
| **Modifies Mesh** | Yes |

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `max_hole_size` | int | 1000 | Maximum hole size in edges |
| `method` | enum | "fan" | Fill method: "fan" or "ear" |

**Risk Notes:**
- Small holes (< 100 edges): Low risk, minimal geometry added
- Large holes (> 1000 edges): Medium risk, significant geometry added

**Use Cases:**
- Close small gaps and holes
- Make mesh watertight
- Repair incomplete models

---

### `fill_holes_pymeshfix`

Use pymeshfix's hole-filling algorithm for more robust repairs.

| Property | Value |
|----------|-------|
| **Tool** | pymeshfix |
| **Risk** | 🟡 Low |
| **Modifies Mesh** | Yes |

**Parameters:** None

**Use Cases:**
- When trimesh `fill_holes` fails
- Complex hole geometries
- Non-planar holes

---

### `cap_holes`

Cap open boundaries with flat faces. Best for planar openings.

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟡 Low |
| **Modifies Mesh** | Yes |

**Parameters:** None

**Use Cases:**
- Close flat bottom openings
- Cap simple boundary loops

---

## Category: Normal Correction

### `recalculate_normals`

Recompute face normals from vertex winding order.

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟢 Safe |
| **Modifies Mesh** | Yes (normals only) |

**Parameters:** None

---

### `reorient_normals`

Attempt to make all face normals point outward consistently.

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟢 Safe |
| **Modifies Mesh** | Yes (winding order) |

**Parameters:** None

---

### `unify_normals`

Unify normals so adjacent faces have consistent orientation.

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟢 Safe |
| **Modifies Mesh** | Yes (winding order) |

**Parameters:** None

---

### `fix_normals`

Combine recalculate and reorient for a single-step fix.

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟢 Safe |
| **Modifies Mesh** | Yes |

**Parameters:** None

**Implementation:**
```python
def fix_normals(mesh, params):
    mesh.fix_normals()
    return mesh
```

---

### `flip_normals`

Invert all face normals (useful if model is inside-out).

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟢 Safe |
| **Modifies Mesh** | Yes |

**Parameters:** None

---

## Category: Component Management

### `remove_small_components`

Remove disconnected components below a threshold.

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟠 Medium |
| **Modifies Mesh** | Yes |

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `min_volume` | float | 0.0 | Minimum volume to keep |
| `min_faces` | int | 100 | Minimum face count to keep |

**Risk Notes:**
- May remove intentional small parts
- Always preview components before removal

---

### `keep_largest_component`

Keep only the largest connected component.

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟠 Medium |
| **Modifies Mesh** | Yes |

**Parameters:** None

**Risk Notes:**
- Removes ALL other components
- Use with caution on multi-part models

---

### `separate_shells`

Split mesh into separate shell components.

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟢 Safe |
| **Modifies Mesh** | No (returns list) |

**Parameters:** None

---

### `boolean_union`

Merge overlapping shells into a single watertight mesh.

| Property | Value |
|----------|-------|
| **Tool** | trimesh/blender |
| **Risk** | 🔴 High |
| **Modifies Mesh** | Yes |

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `engine` | enum | "manifold" | Engine: "blender" or "manifold" |

**Risk Notes:**
- Can significantly alter topology
- May fail on complex intersections
- Verify result with fidelity check

---

### `remove_internal_geometry`

Remove components fully enclosed by the outer shell.

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟠 Medium |
| **Modifies Mesh** | Yes |

**Parameters:** None

---

## Category: Repair & Manifold Fixes

### `pymeshfix_repair`

Run pymeshfix's automatic repair pass.

| Property | Value |
|----------|-------|
| **Tool** | pymeshfix |
| **Risk** | 🟡 Low |
| **Modifies Mesh** | Yes |

**Parameters:** None

**Capabilities:**
- Fix non-manifold edges
- Close small holes
- Remove degenerate faces
- Unify normals

---

### `fix_non_manifold_edges`

Attempt to fix non-manifold edges by splitting or removing.

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟡 Low |
| **Modifies Mesh** | Yes |

**Parameters:** None

---

### `fix_non_manifold_vertices`

Fix vertices shared by non-adjacent faces.

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟡 Low |
| **Modifies Mesh** | Yes |

**Parameters:** None

---

### `stitch_boundaries`

Stitch open boundaries where edges nearly match.

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟡 Low |
| **Modifies Mesh** | Yes |

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `tolerance` | float | 1e-5 | Distance tolerance for stitching |

---

### `close_cracks`

Close small cracks by merging nearby vertices along boundaries.

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟡 Low |
| **Modifies Mesh** | Yes |

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `tolerance` | float | 1e-5 | Distance tolerance for closing |

---

## Category: Simplification & Remeshing

### `decimate`

Reduce face count while preserving shape.

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟠 Medium |
| **Modifies Mesh** | Yes |

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `target_faces` | int | null | Target face count |
| `target_ratio` | float | 0.5 | Target ratio (0.0-1.0) |

**Risk Notes:**
- Reduces geometric detail
- May affect fine features
- Use fidelity check after decimation

---

### `simplify_quadric`

Quadric error decimation for high-quality simplification.

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟠 Medium |
| **Modifies Mesh** | Yes |

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `target_faces` | int | required | Target face count |
| `agg` | float | 5.0 | Aggressiveness factor |

---

### `subdivide`

Subdivide faces to increase mesh resolution.

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟡 Low |
| **Modifies Mesh** | Yes |

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `iterations` | int | 1 | Number of subdivision iterations |

---

### `smooth_laplacian`

Apply Laplacian smoothing to reduce noise.

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟠 Medium |
| **Modifies Mesh** | Yes |

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `iterations` | int | 1 | Smoothing iterations |
| `lamb` | float | 0.5 | Smoothing factor (0.0-1.0) |

**Risk Notes:**
- Can shrink the mesh
- May lose sharp edges
- Use `smooth_taubin` for less shrinkage

---

### `smooth_taubin`

Taubin smoothing (reduces shrinkage compared to Laplacian).

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟠 Medium |
| **Modifies Mesh** | Yes |

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `iterations` | int | 1 | Smoothing iterations |
| `lamb` | float | 0.5 | Smoothing factor |
| `mu` | float | -0.53 | Inflation factor |

---

### `remesh_isotropic`

Remesh to uniform triangle sizes.

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🔴 High |
| **Modifies Mesh** | Yes |

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `target_edge_length` | float | required | Target edge length |

---

## Category: Geometry Analysis

### `identify_thin_regions`

Detect regions thinner than a threshold. Report only, no modification.

| Property | Value |
|----------|-------|
| **Tool** | internal |
| **Risk** | 🟢 Safe |
| **Modifies Mesh** | No |

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `min_thickness` | float | 0.8 | Minimum thickness (mm) |

**Returns:**
- List of thin regions with locations and thicknesses

---

### `thicken_regions`

Thicken thin walls to meet minimum printable thickness.

| Property | Value |
|----------|-------|
| **Tool** | blender |
| **Risk** | 🔴 High |
| **Modifies Mesh** | Yes |

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `target_thickness` | float | 1.0 | Target thickness (mm) |

---

### `offset_surface`

Offset the mesh surface inward or outward.

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟠 Medium |
| **Modifies Mesh** | Yes |

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `distance` | float | required | Offset distance (+ outward, - inward) |

---

### `hollow`

Create a hollow shell with specified wall thickness.

| Property | Value |
|----------|-------|
| **Tool** | blender |
| **Risk** | 🔴 High |
| **Modifies Mesh** | Yes |

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `wall_thickness` | float | required | Wall thickness (mm) |

---

## Category: Boolean & Intersection Fixes

### `detect_self_intersections`

Check for self-intersecting faces. Report only, no modification.

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟢 Safe |
| **Modifies Mesh** | No |

**Parameters:** None

**Returns:**
- Boolean indicating presence of self-intersections
- List of intersecting face pairs

---

### `fix_self_intersections`

Attempt to resolve self-intersections using boolean operations.

| Property | Value |
|----------|-------|
| **Tool** | blender |
| **Risk** | 🔴 High |
| **Modifies Mesh** | Yes |

**Parameters:** None

---

### `boolean_difference`

Subtract one mesh from another.

| Property | Value |
|----------|-------|
| **Tool** | blender |
| **Risk** | 🔴 High |
| **Modifies Mesh** | Yes |

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `tool_mesh` | path | required | Path to mesh to subtract |

---

### `boolean_intersect`

Keep only the intersection of two meshes.

| Property | Value |
|----------|-------|
| **Tool** | blender |
| **Risk** | 🔴 High |
| **Modifies Mesh** | Yes |

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `tool_mesh` | path | required | Path to second mesh |

---

## Category: Validation & Diagnostics

### `validate`

Run all validation checks and produce a diagnostics report.

| Property | Value |
|----------|-------|
| **Tool** | internal |
| **Risk** | 🟢 Safe |
| **Modifies Mesh** | No |

**Parameters:** None

**Returns:**
- Complete `Diagnostics` object

---

### `check_watertight`

Check if mesh is watertight (closed, no holes).

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟢 Safe |
| **Modifies Mesh** | No |

**Returns:**
- Boolean: is watertight

---

### `check_manifold`

Check for non-manifold edges and vertices.

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟢 Safe |
| **Modifies Mesh** | No |

**Returns:**
- Boolean: is manifold
- Counts of non-manifold edges/vertices

---

### `check_normals`

Check for inconsistent or inverted normals.

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟢 Safe |
| **Modifies Mesh** | No |

**Returns:**
- Normal consistency score (0.0-1.0)
- Count of inverted normals

---

### `check_volume`

Compute and report mesh volume.

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟢 Safe |
| **Modifies Mesh** | No |

**Returns:**
- Volume value
- Is positive (valid solid)

---

### `check_bounding_box`

Report bounding box dimensions.

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟢 Safe |
| **Modifies Mesh** | No |

**Returns:**
- Bounding box min/max
- Dimensions (x, y, z)

---

### `compute_diagnostics`

Compute full diagnostics vector for profile detection.

| Property | Value |
|----------|-------|
| **Tool** | internal |
| **Risk** | 🟢 Safe |
| **Modifies Mesh** | No |

**Returns:**
- Complete `Diagnostics` object with all metrics

---

## Category: Export

### `export_stl`

Export mesh to STL file (binary by default).

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟢 Safe |
| **Modifies Mesh** | No |

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `path` | string | required | Output path |
| `ascii` | bool | false | Use ASCII format |

---

### `export_stl_ascii`

Export mesh to ASCII STL file.

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟢 Safe |
| **Modifies Mesh** | No |

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `path` | string | required | Output path |

---

### `export_obj`

Export mesh to OBJ format.

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟢 Safe |
| **Modifies Mesh** | No |

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `path` | string | required | Output path |

---

### `export_ply`

Export mesh to PLY format.

| Property | Value |
|----------|-------|
| **Tool** | trimesh |
| **Risk** | 🟢 Safe |
| **Modifies Mesh** | No |

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `path` | string | required | Output path |

---

### `export_3mf`

Export mesh to 3MF format (preferred for 3D printing).

| Property | Value |
|----------|-------|
| **Tool** | meshio |
| **Risk** | 🟢 Safe |
| **Modifies Mesh** | No |

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `path` | string | required | Output path |

---

## Category: Blender (Escalation)

> ⚠️ **Note**: Blender actions require Blender to be installed and accessible via CLI.
> These are "escalation" actions used when standard repairs fail.

### `blender_remesh`

Apply Blender's voxel remesh for aggressive topology repair.

| Property | Value |
|----------|-------|
| **Tool** | blender |
| **Risk** | 🔴 High |
| **Modifies Mesh** | Yes (significantly) |

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `voxel_size` | float | 0.1 | Voxel size (smaller = more detail) |
| `adaptivity` | float | 0.0 | Adaptivity (0.0 = uniform) |

**Risk Notes:**
- Completely reconstructs surface topology
- May lose fine details
- May change mesh appearance
- **Always verify with fidelity check**

---

### `blender_decimate`

Blender's decimate modifier.

| Property | Value |
|----------|-------|
| **Tool** | blender |
| **Risk** | 🟠 Medium |
| **Modifies Mesh** | Yes |

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `ratio` | float | 0.5 | Reduction ratio (0.0-1.0) |
| `use_collapse` | bool | true | Use collapse decimation |

---

### `blender_boolean_union`

Merge all mesh parts using Blender's boolean solver.

| Property | Value |
|----------|-------|
| **Tool** | blender |
| **Risk** | 🔴 High |
| **Modifies Mesh** | Yes |

**Parameters:** None

**Use Cases:**
- Merge self-intersecting shells
- Create single watertight mesh from multiple parts

---

### `blender_solidify`

Add thickness to thin surfaces using the solidify modifier.

| Property | Value |
|----------|-------|
| **Tool** | blender |
| **Risk** | 🔴 High |
| **Modifies Mesh** | Yes |

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `thickness` | float | 1.0 | Wall thickness to add |

---

### `blender_smooth`

Apply Blender's smooth modifier.

| Property | Value |
|----------|-------|
| **Tool** | blender |
| **Risk** | 🟠 Medium |
| **Modifies Mesh** | Yes |

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `iterations` | int | 1 | Smoothing iterations |
| `factor` | float | 0.5 | Smoothing factor |

---

### `blender_triangulate`

Triangulate all faces (ensure STL compatibility).

| Property | Value |
|----------|-------|
| **Tool** | blender |
| **Risk** | 🟢 Safe |
| **Modifies Mesh** | Yes |

**Parameters:** None

---

## See Also

- [Repair Pipeline](repair_pipeline.md) - How actions fit into the repair process
- [Model Profiles](model_profiles.md) - Profile-specific action recommendations
- [Filter Script Format](functional_spec.md#filter-script-representation) - How to define filter scripts
