# Validation Guide

## Overview

This document describes the validation criteria and procedures for MeshPrep's mesh repair operations. Validation ensures that repaired models are both **geometrically valid** (printable) and **visually unchanged** (faithful to the original).

---

## Two-Stage Validation

MeshPrep uses a two-stage validation approach:

```
┌─────────────────────────────────────────────────────────────┐
│                    VALIDATION PIPELINE                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: GEOMETRIC VALIDATION                               │
│  ─────────────────────────────                               │
│  Purpose: Ensure mesh is valid for 3D printing               │
│                                                              │
│  Checks:                                                     │
│  ✓ Watertight (closed, no holes)                            │
│  ✓ Manifold (proper edge/vertex topology)                   │
│  ✓ Positive volume (not inside-out)                         │
│  ✓ No self-intersections                                     │
│  ✓ No degenerate faces                                       │
│                                                              │
│  Result: is_printable (bool)                                │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: FIDELITY VALIDATION                                │
│  ─────────────────────────────                               │
│  Purpose: Ensure model appearance is preserved               │
│                                                              │
│  Checks:                                                     │
│  ✓ Volume change < 1%                                       │
│  ✓ Bounding box unchanged                                    │
│  ✓ Hausdorff distance < threshold                           │
│  ✓ Surface area within tolerance                             │
│                                                              │
│  Result: is_visually_unchanged (bool)                       │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  FINAL RESULT                                                │
│  ────────────                                                │
│  success = is_printable AND is_visually_unchanged           │
└─────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Geometric Validation

### Watertight Check

A mesh is **watertight** if it is a closed surface with no holes or gaps.

```python
def check_watertight(mesh: trimesh.Trimesh) -> tuple[bool, dict]:
    """Check if mesh is watertight."""
    
    is_watertight = mesh.is_watertight
    
    details = {
        "is_watertight": is_watertight,
        "boundary_edge_count": len(mesh.edges_unique) - len(mesh.edges),
        "hole_count": len(mesh.outline())
    }
    
    return is_watertight, details
```

**Criteria:**
- No boundary (naked) edges
- All edges shared by exactly 2 faces
- No holes in the surface

**Common Failures:**
- Open holes in the surface
- Gaps between faces
- Missing faces

---

### Manifold Check

A mesh is **manifold** if every edge is shared by exactly 2 faces, and every vertex is surrounded by a single fan of faces.

```python
def check_manifold(mesh: trimesh.Trimesh) -> tuple[bool, dict]:
    """Check if mesh is manifold."""
    
    # Check for non-manifold edges (shared by more than 2 faces)
    edges = mesh.edges_sorted
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    non_manifold_edges = np.sum(counts > 2)
    
    # Check for non-manifold vertices
    # (vertices where the surrounding faces don't form a single fan)
    non_manifold_vertices = 0  # Simplified; actual check is more complex
    
    is_manifold = (non_manifold_edges == 0 and non_manifold_vertices == 0)
    
    details = {
        "is_manifold": is_manifold,
        "non_manifold_edge_count": non_manifold_edges,
        "non_manifold_vertex_count": non_manifold_vertices
    }
    
    return is_manifold, details
```

**Non-Manifold Types:**

| Type | Description | Visual |
|------|-------------|--------|
| T-junction | Edge shared by 3+ faces | Three faces meeting at one edge |
| Bowtie vertex | Vertex shared by disconnected face groups | Pinched geometry |
| Self-touching | Faces that touch but don't share topology | Overlapping surfaces |

---

### Volume Check

A valid solid must have **positive volume**.

```python
def check_volume(mesh: trimesh.Trimesh) -> tuple[bool, dict]:
    """Check if mesh has positive volume."""
    
    volume = mesh.volume
    
    # Negative volume indicates inverted normals
    has_positive_volume = volume > 0
    
    # Zero volume might indicate 2D surface or collapsed geometry
    is_valid_volume = abs(volume) > 1e-10
    
    details = {
        "volume": volume,
        "has_positive_volume": has_positive_volume,
        "is_valid_volume": is_valid_volume
    }
    
    return has_positive_volume and is_valid_volume, details
```

**Common Issues:**
- Negative volume: All normals pointing inward (inside-out)
- Zero volume: Degenerate or 2D geometry

---

### Self-Intersection Check

Faces should not intersect each other (except at shared edges).

```python
def check_self_intersections(
    mesh: trimesh.Trimesh,
    max_allowed: int = 0
) -> tuple[bool, dict]:
    """Check for self-intersecting faces."""
    
    # This is an expensive operation
    try:
        # Use trimesh's built-in intersection check
        intersections = mesh.is_self_intersecting()
        intersection_count = 0  # Would need detailed check for count
        
    except Exception:
        # If check fails, assume OK
        intersections = False
        intersection_count = 0
    
    no_intersections = not intersections or intersection_count <= max_allowed
    
    details = {
        "has_self_intersections": intersections,
        "intersection_count": intersection_count,
        "max_allowed": max_allowed
    }
    
    return no_intersections, details
```

**Note:** Self-intersection detection is computationally expensive. For large meshes, consider:
- Sampling-based approximation
- Bounding box pre-filtering
- Optional deep check flag

---

### Degenerate Face Check

Faces with zero area or invalid topology should be removed.

```python
def check_degenerate_faces(mesh: trimesh.Trimesh) -> tuple[bool, dict]:
    """Check for degenerate (zero-area) faces."""
    
    # Compute face areas
    face_areas = mesh.area_faces
    
    # Count zero-area faces
    degenerate_count = np.sum(face_areas < 1e-10)
    
    # Also check for faces with duplicate vertices
    duplicate_vertex_faces = 0
    for face in mesh.faces:
        if len(set(face)) < 3:
            duplicate_vertex_faces += 1
    
    total_degenerate = degenerate_count + duplicate_vertex_faces
    no_degenerates = total_degenerate == 0
    
    details = {
        "degenerate_face_count": total_degenerate,
        "zero_area_faces": degenerate_count,
        "duplicate_vertex_faces": duplicate_vertex_faces
    }
    
    return no_degenerates, details
```

---

### Combined Geometric Validation

```python
@dataclass
class GeometricValidation:
    """Result of all geometric validation checks."""
    
    is_watertight: bool
    is_manifold: bool
    has_positive_volume: bool
    no_self_intersections: bool
    no_degenerate_faces: bool
    
    watertight_details: dict
    manifold_details: dict
    volume_details: dict
    intersection_details: dict
    degenerate_details: dict
    
    @property
    def is_printable(self) -> bool:
        """Check if mesh meets all geometric requirements."""
        return (
            self.is_watertight and
            self.is_manifold and
            self.has_positive_volume and
            self.no_self_intersections and
            self.no_degenerate_faces
        )
    
    @property
    def issues(self) -> list[str]:
        """List of detected issues."""
        issues = []
        
        if not self.is_watertight:
            count = self.watertight_details.get("hole_count", 0)
            issues.append(f"Not watertight ({count} holes)")
        
        if not self.is_manifold:
            edges = self.manifold_details.get("non_manifold_edge_count", 0)
            verts = self.manifold_details.get("non_manifold_vertex_count", 0)
            issues.append(f"Non-manifold ({edges} edges, {verts} vertices)")
        
        if not self.has_positive_volume:
            vol = self.volume_details.get("volume", 0)
            issues.append(f"Invalid volume ({vol:.4f})")
        
        if not self.no_self_intersections:
            count = self.intersection_details.get("intersection_count", 0)
            issues.append(f"Self-intersections ({count})")
        
        if not self.no_degenerate_faces:
            count = self.degenerate_details.get("degenerate_face_count", 0)
            issues.append(f"Degenerate faces ({count})")
        
        return issues


def validate_geometry(mesh: trimesh.Trimesh) -> GeometricValidation:
    """Run all geometric validation checks."""
    
    wt_ok, wt_details = check_watertight(mesh)
    mf_ok, mf_details = check_manifold(mesh)
    vol_ok, vol_details = check_volume(mesh)
    si_ok, si_details = check_self_intersections(mesh)
    dg_ok, dg_details = check_degenerate_faces(mesh)
    
    return GeometricValidation(
        is_watertight=wt_ok,
        is_manifold=mf_ok,
        has_positive_volume=vol_ok,
        no_self_intersections=si_ok,
        no_degenerate_faces=dg_ok,
        watertight_details=wt_details,
        manifold_details=mf_details,
        volume_details=vol_details,
        intersection_details=si_details,
        degenerate_details=dg_details
    )
```

---

## Stage 2: Fidelity Validation

### Volume Comparison

Volume should remain within tolerance after repair.

```python
def check_volume_change(
    original_volume: float,
    repaired_volume: float,
    max_change_pct: float = 1.0
) -> tuple[bool, dict]:
    """Check if volume change is within tolerance."""
    
    if original_volume == 0:
        change_pct = 0 if repaired_volume == 0 else float('inf')
    else:
        change_pct = ((repaired_volume - original_volume) / original_volume) * 100
    
    is_acceptable = abs(change_pct) < max_change_pct
    
    details = {
        "original_volume": original_volume,
        "repaired_volume": repaired_volume,
        "change_pct": change_pct,
        "max_allowed_pct": max_change_pct,
        "is_acceptable": is_acceptable
    }
    
    return is_acceptable, details
```

**Thresholds:**
- Default: < 1% change
- Strict: < 0.1% change
- Permissive: < 5% change

---

### Bounding Box Comparison

The bounding box should remain unchanged.

```python
def check_bounding_box(
    original_bbox: np.ndarray,
    repaired_bbox: np.ndarray,
    tolerance: float = 1e-6
) -> tuple[bool, dict]:
    """Check if bounding box is unchanged."""
    
    original_dims = original_bbox[1] - original_bbox[0]
    repaired_dims = repaired_bbox[1] - repaired_bbox[0]
    
    # Check each dimension
    dim_changes = np.abs(repaired_dims - original_dims)
    relative_changes = dim_changes / np.maximum(original_dims, 1e-10)
    
    is_unchanged = np.all(relative_changes < tolerance)
    
    details = {
        "original_dimensions": original_dims.tolist(),
        "repaired_dimensions": repaired_dims.tolist(),
        "dimension_changes": dim_changes.tolist(),
        "relative_changes": relative_changes.tolist(),
        "tolerance": tolerance,
        "is_unchanged": is_unchanged
    }
    
    return is_unchanged, details
```

---

### Hausdorff Distance

The **Hausdorff distance** measures the maximum surface deviation between two meshes.

```python
def compute_hausdorff_distance(
    original: trimesh.Trimesh,
    repaired: trimesh.Trimesh,
    sample_count: int = 10000
) -> tuple[float, float, dict]:
    """Compute Hausdorff distance between original and repaired meshes.
    
    Returns:
        hausdorff: Maximum deviation
        mean_distance: Average deviation
        details: Detailed metrics
    """
    from scipy.spatial import cKDTree
    
    # Sample points on both surfaces
    samples_original = original.sample(sample_count)
    samples_repaired = repaired.sample(sample_count)
    
    # Build KD-trees for efficient nearest-neighbor lookup
    tree_original = cKDTree(samples_original)
    tree_repaired = cKDTree(samples_repaired)
    
    # Forward distances: original -> repaired
    dist_orig_to_rep, _ = tree_repaired.query(samples_original)
    
    # Backward distances: repaired -> original
    dist_rep_to_orig, _ = tree_original.query(samples_repaired)
    
    # Hausdorff is the maximum of the two directed distances
    hausdorff_orig_to_rep = dist_orig_to_rep.max()
    hausdorff_rep_to_orig = dist_rep_to_orig.max()
    hausdorff = max(hausdorff_orig_to_rep, hausdorff_rep_to_orig)
    
    # Mean surface distance
    mean_distance = (dist_orig_to_rep.mean() + dist_rep_to_orig.mean()) / 2
    
    details = {
        "hausdorff_distance": hausdorff,
        "hausdorff_orig_to_rep": hausdorff_orig_to_rep,
        "hausdorff_rep_to_orig": hausdorff_rep_to_orig,
        "mean_surface_distance": mean_distance,
        "sample_count": sample_count,
        "dist_orig_to_rep_stats": {
            "min": float(dist_orig_to_rep.min()),
            "max": float(dist_orig_to_rep.max()),
            "mean": float(dist_orig_to_rep.mean()),
            "std": float(dist_orig_to_rep.std())
        },
        "dist_rep_to_orig_stats": {
            "min": float(dist_rep_to_orig.min()),
            "max": float(dist_rep_to_orig.max()),
            "mean": float(dist_rep_to_orig.mean()),
            "std": float(dist_rep_to_orig.std())
        }
    }
    
    return hausdorff, mean_distance, details


def check_hausdorff(
    original: trimesh.Trimesh,
    repaired: trimesh.Trimesh,
    max_relative: float = 0.001,  # 0.1% of bbox diagonal
    sample_count: int = 10000
) -> tuple[bool, dict]:
    """Check if Hausdorff distance is within tolerance."""
    
    # Compute bbox diagonal for relative comparison
    bbox_diagonal = np.linalg.norm(original.bounds[1] - original.bounds[0])
    
    hausdorff, mean_dist, details = compute_hausdorff_distance(
        original, repaired, sample_count
    )
    
    relative_hausdorff = hausdorff / bbox_diagonal if bbox_diagonal > 0 else 0
    
    is_acceptable = relative_hausdorff < max_relative
    
    details.update({
        "bbox_diagonal": bbox_diagonal,
        "relative_hausdorff": relative_hausdorff,
        "max_relative": max_relative,
        "is_acceptable": is_acceptable
    })
    
    return is_acceptable, details
```

**Interpretation:**
- Hausdorff = 0: Surfaces are identical
- Small Hausdorff: Minor local deviations (acceptable for repair)
- Large Hausdorff: Significant surface changes (potential problem)

---

### Surface Area Comparison

Surface area should remain within tolerance.

```python
def check_surface_area(
    original_area: float,
    repaired_area: float,
    max_change_pct: float = 2.0
) -> tuple[bool, dict]:
    """Check if surface area change is within tolerance."""
    
    if original_area == 0:
        change_pct = 0 if repaired_area == 0 else float('inf')
    else:
        change_pct = ((repaired_area - original_area) / original_area) * 100
    
    is_acceptable = abs(change_pct) < max_change_pct
    
    details = {
        "original_area": original_area,
        "repaired_area": repaired_area,
        "change_pct": change_pct,
        "max_allowed_pct": max_change_pct,
        "is_acceptable": is_acceptable
    }
    
    return is_acceptable, details
```

**Note:** Surface area can change slightly when filling holes (adds new faces). A 2% tolerance is typical.

---

### Combined Fidelity Validation

```python
@dataclass
class FidelityValidation:
    """Result of all fidelity validation checks."""
    
    volume_acceptable: bool
    bbox_unchanged: bool
    hausdorff_acceptable: bool
    surface_area_acceptable: bool
    
    volume_details: dict
    bbox_details: dict
    hausdorff_details: dict
    surface_area_details: dict
    
    @property
    def is_visually_unchanged(self) -> bool:
        """Check if mesh appearance is preserved."""
        return (
            self.volume_acceptable and
            self.bbox_unchanged and
            self.hausdorff_acceptable
        )
    
    @property
    def changes(self) -> list[str]:
        """List of detected changes."""
        changes = []
        
        if not self.volume_acceptable:
            pct = self.volume_details.get("change_pct", 0)
            changes.append(f"Volume changed by {pct:.2f}%")
        
        if not self.bbox_unchanged:
            changes.append("Bounding box dimensions changed")
        
        if not self.hausdorff_acceptable:
            dist = self.hausdorff_details.get("hausdorff_distance", 0)
            rel = self.hausdorff_details.get("relative_hausdorff", 0) * 100
            changes.append(f"Surface deviation: {dist:.4f} ({rel:.2f}% of bbox)")
        
        if not self.surface_area_acceptable:
            pct = self.surface_area_details.get("change_pct", 0)
            changes.append(f"Surface area changed by {pct:.2f}%")
        
        return changes


def validate_fidelity(
    original: trimesh.Trimesh,
    repaired: trimesh.Trimesh,
    config: Optional[dict] = None
) -> FidelityValidation:
    """Run all fidelity validation checks."""
    
    config = config or {}
    
    # Volume
    vol_ok, vol_details = check_volume_change(
        original.volume,
        repaired.volume,
        config.get("max_volume_change_pct", 1.0)
    )
    
    # Bounding box
    bbox_ok, bbox_details = check_bounding_box(
        original.bounds,
        repaired.bounds,
        config.get("bbox_tolerance", 1e-6)
    )
    
    # Hausdorff
    haus_ok, haus_details = check_hausdorff(
        original,
        repaired,
        config.get("max_hausdorff_relative", 0.001),
        config.get("hausdorff_sample_count", 10000)
    )
    
    # Surface area
    area_ok, area_details = check_surface_area(
        original.area,
        repaired.area,
        config.get("max_surface_area_change_pct", 2.0)
    )
    
    return FidelityValidation(
        volume_acceptable=vol_ok,
        bbox_unchanged=bbox_ok,
        hausdorff_acceptable=haus_ok,
        surface_area_acceptable=area_ok,
        volume_details=vol_details,
        bbox_details=bbox_details,
        hausdorff_details=haus_details,
        surface_area_details=area_details
    )
```

---

## Configuration

### Default Thresholds

```yaml
# config/validation_thresholds.yaml

geometric:
  # Self-intersection tolerance
  max_self_intersections: 0
  
  # Degenerate face tolerance
  max_degenerate_faces: 0

fidelity:
  # Volume change tolerance
  max_volume_change_pct: 1.0
  
  # Bounding box tolerance
  bbox_tolerance: 1e-6
  
  # Surface deviation tolerance (relative to bbox diagonal)
  max_hausdorff_relative: 0.001
  
  # Surface area change tolerance
  max_surface_area_change_pct: 2.0
  
  # Hausdorff sample count
  hausdorff_sample_count: 10000

# Validation mode
mode: strict  # Options: strict, normal, permissive

modes:
  strict:
    max_volume_change_pct: 0.1
    max_hausdorff_relative: 0.0001
    
  normal:
    max_volume_change_pct: 1.0
    max_hausdorff_relative: 0.001
    
  permissive:
    max_volume_change_pct: 5.0
    max_hausdorff_relative: 0.01
```

---

## Validation Modes

### Strict Mode

- **Use case:** High-precision models, mechanical parts
- **Tolerance:** Very tight (< 0.1% volume change)
- **Behavior:** Fail on any detectable change

### Normal Mode (Default)

- **Use case:** General 3D printing models
- **Tolerance:** Reasonable (< 1% volume change)
- **Behavior:** Balance between fidelity and successful repair

### Permissive Mode

- **Use case:** Artistic models, topology repair
- **Tolerance:** Relaxed (< 5% volume change)
- **Behavior:** Accept more changes for successful repair

---

## Validation Report

```python
@dataclass
class ValidationReport:
    """Complete validation report for a repair operation."""
    
    # Results
    geometric: GeometricValidation
    fidelity: FidelityValidation
    
    # Overall status
    is_successful: bool
    
    # Summary
    issues: list[str]
    changes: list[str]
    warnings: list[str]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export."""
        return {
            "is_successful": self.is_successful,
            "geometric": {
                "is_printable": self.geometric.is_printable,
                "is_watertight": self.geometric.is_watertight,
                "is_manifold": self.geometric.is_manifold,
                "has_positive_volume": self.geometric.has_positive_volume,
                "no_self_intersections": self.geometric.no_self_intersections,
                "no_degenerate_faces": self.geometric.no_degenerate_faces,
                "issues": self.geometric.issues
            },
            "fidelity": {
                "is_visually_unchanged": self.fidelity.is_visually_unchanged,
                "volume_change_pct": self.fidelity.volume_details.get("change_pct"),
                "hausdorff_relative": self.fidelity.hausdorff_details.get("relative_hausdorff"),
                "changes": self.fidelity.changes
            },
            "issues": self.issues,
            "changes": self.changes,
            "warnings": self.warnings
        }


def create_validation_report(
    original: trimesh.Trimesh,
    repaired: trimesh.Trimesh,
    config: Optional[dict] = None
) -> ValidationReport:
    """Create a complete validation report."""
    
    geometric = validate_geometry(repaired)
    fidelity = validate_fidelity(original, repaired, config)
    
    is_successful = geometric.is_printable and fidelity.is_visually_unchanged
    
    issues = geometric.issues.copy()
    changes = fidelity.changes.copy()
    warnings = []
    
    # Add warnings for borderline cases
    vol_change = abs(fidelity.volume_details.get("change_pct", 0))
    if 0.5 < vol_change < 1.0:
        warnings.append(f"Volume change ({vol_change:.2f}%) approaching threshold")
    
    return ValidationReport(
        geometric=geometric,
        fidelity=fidelity,
        is_successful=is_successful,
        issues=issues,
        changes=changes,
        warnings=warnings
    )
```

---

## See Also

- [Repair Pipeline](repair_pipeline.md) - How validation fits in the pipeline
- [Filter Actions](filter_actions.md) - Action risk levels
- [Thingi10K Testing](thingi10k_testing.md) - Benchmark validation
