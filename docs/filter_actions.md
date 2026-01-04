# Filter Actions Reference

## Overview

Filter actions are the building blocks of repair pipelines. Each action has a **risk level** indicating impact on visual fidelity.

| Risk | Meaning | Examples |
|------|---------|----------|
| 🟢 Safe | No visual change | `merge_vertices`, `fix_normals`, `validate` |
| 🟡 Low | Minimal geometry added | `fill_holes` (small), `pymeshfix_repair` |
| 🟠 Medium | May affect detail | `remove_small_components`, `decimate`, `smooth` |
| 🔴 High | Significant topology change | `boolean_union`, `blender_remesh` |

---

## Actions by Category

### Loading & Basic Cleanup (🟢 Safe)

| Action | Tool | Description |
|--------|------|-------------|
| `load_stl` | trimesh | Load STL file |
| `trimesh_basic` | trimesh | Merge vertices, remove degenerates, remove infinites |
| `merge_vertices` | trimesh | Weld duplicates within tolerance (`eps`: float) |
| `remove_degenerate_faces` | trimesh | Remove zero-area faces |
| `remove_duplicate_faces` | trimesh | Remove exact duplicates |
| `remove_unreferenced_vertices` | trimesh | Remove orphan vertices |

### Hole Filling (🟡 Low to 🟠 Medium)

| Action | Tool | Risk | Description |
|--------|------|------|-------------|
| `fill_holes` | trimesh | 🟡/🟠 | Fill holes (`max_hole_size`: int, `method`: "fan"/"ear") |
| `fill_holes_pymeshfix` | pymeshfix | 🟡 | Robust hole filling |
| `cap_holes` | trimesh | 🟡 | Cap planar openings |

### Normal Correction (🟢 Safe)

| Action | Tool | Description |
|--------|------|-------------|
| `recalculate_normals` | trimesh | Recompute from winding |
| `reorient_normals` | trimesh | Point outward |
| `unify_normals` | trimesh | Consistent orientation |
| `fix_normals` | trimesh | Combined fix |
| `flip_normals` | trimesh | Invert all normals |

### Component Management (🟠 Medium)

| Action | Tool | Description |
|--------|------|-------------|
| `remove_small_components` | trimesh | Remove below threshold (`min_volume`, `min_faces`) |
| `keep_largest_component` | trimesh | Keep only largest |
| `separate_shells` | trimesh | Split into components |
| `boolean_union` | trimesh/blender | 🔴 Merge shells |
| `remove_internal_geometry` | trimesh | Remove enclosed parts |

### Repair & Manifold (🟡 Low)

| Action | Tool | Description |
|--------|------|-------------|
| `pymeshfix_repair` | pymeshfix | Automatic repair |
| `fix_non_manifold_edges` | trimesh | Split/remove bad edges |
| `fix_non_manifold_vertices` | trimesh | Fix shared vertices |
| `stitch_boundaries` | trimesh | Stitch matching edges (`tolerance`: float) |
| `close_cracks` | trimesh | Merge along boundaries |

### Simplification (🟠 Medium)

| Action | Tool | Description |
|--------|------|-------------|
| `decimate` | trimesh | Reduce faces (`target_faces`/`target_ratio`) |
| `simplify_quadric` | trimesh | Quadric decimation |
| `subdivide` | trimesh | Increase resolution (`iterations`: int) |
| `smooth_laplacian` | trimesh | Laplacian smoothing |
| `smooth_taubin` | trimesh | Taubin smoothing (less shrinkage) |
| `remesh_isotropic` | trimesh | 🔴 Uniform triangles |

### Surface Reconstruction (🔴 High - for extreme fragmentation)

| Action | Tool | Description |
|--------|------|-------------|
| `fragment_aware_reconstruct` | open3d | Auto-select best method |
| `open3d_screened_poisson` | open3d | Gold standard for point cloud (`depth`: 8-12) |
| `open3d_ball_pivoting` | open3d | Good for uniform density |
| `morphological_voxel_reconstruct` | scipy | Voxelize + morphological ops |
| `shrinkwrap_reconstruct` | trimesh | Project envelope onto fragments |

### Geometry Analysis (🟢 Safe - read only)

| Action | Tool | Description |
|--------|------|-------------|
| `identify_thin_regions` | internal | Find thin walls (`min_thickness`: float) |
| `detect_self_intersections` | trimesh | Check for intersections |

### Validation (🟢 Safe - read only)

| Action | Tool | Description |
|--------|------|-------------|
| `validate` | internal | Full diagnostics report |
| `check_watertight` | trimesh | Is watertight? |
| `check_manifold` | trimesh | Is manifold? |
| `check_normals` | trimesh | Normal consistency |
| `check_volume` | trimesh | Volume value |
| `compute_diagnostics` | internal | Full diagnostics vector |

### Export (🟢 Safe)

| Action | Tool | Description |
|--------|------|-------------|
| `export_stl` | trimesh | Binary STL (`path`, `ascii`: bool) |
| `export_obj` | trimesh | OBJ format |
| `export_ply` | trimesh | PLY format |
| `export_3mf` | meshio | 3MF format (preferred) |

### Blender Escalation (🔴 High)

| Action | Tool | Description |
|--------|------|-------------|
| `blender_remesh` | blender | Voxel remesh (`voxel_size`: float) |
| `blender_boolean_union` | blender | Boolean merge |
| `blender_solidify` | blender | Add thickness |
| `blender_decimate` | blender | Decimate modifier |
| `blender_smooth` | blender | Smooth modifier |
| `blender_triangulate` | blender | Ensure STL compatibility |

---

## See Also

- [Repair Strategy Guide](repair_strategy_guide.md) - When to use each tool
- [Model Profiles](model_profiles.md) - Profile-specific recommendations
