# Mesh Repair Pipeline

## Overview

This document describes the complete mesh repair pipeline in MeshPrep, from loading a model through validation of the repaired output. The pipeline is designed to:

1. **Preserve visual fidelity** - Repair without altering the model's appearance
2. **Ensure geometric validity** - Produce watertight, manifold meshes
3. **Be reproducible** - Identical inputs produce identical outputs
4. **Support validation** - Verify repairs meet quality criteria

---

## Pipeline Stages

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MESH REPAIR PIPELINE                               │
└─────────────────────────────────────────────────────────────────────────────┘

     ┌──────────────┐
     │  Input STL   │
     └──────┬───────┘
            │
            ▼
┌───────────────────────────────────────┐
│  STAGE 1: LOAD & BASELINE CAPTURE     │
│  ─────────────────────────────────    │
│  • Load mesh with trimesh             │
│  • Compute baseline metrics           │
│  • Generate model fingerprint (SHA256)│
│  • Capture: volume, bbox, vertex/face │
│    count, surface samples             │
└───────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────┐
│  STAGE 2: DIAGNOSTICS COMPUTATION     │
│  ─────────────────────────────────    │
│  • Watertight check                   │
│  • Manifold edge/vertex analysis      │
│  • Hole detection and counting        │
│  • Component analysis                 │
│  • Normal consistency check           │
│  • Self-intersection detection        │
│  • Degenerate face detection          │
│  • Thin feature analysis              │
└───────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────┐
│  STAGE 3: PROFILE DETECTION           │
│  ─────────────────────────────────    │
│  • Match diagnostics to profiles      │
│  • Compute confidence scores          │
│  • Select best-matching profile(s)    │
│  • Generate explanation text          │
└───────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────┐
│  STAGE 4: FILTER SCRIPT GENERATION    │
│  ─────────────────────────────────    │
│  • Generate suggested filter script   │
│  • Include metadata and provenance    │
│  • Present for user review            │
│  • Allow editing before execution     │
└───────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────┐
│  STAGE 5: FILTER SCRIPT EXECUTION     │
│  ─────────────────────────────────    │
│  • Execute actions in order           │
│  • Capture per-step diagnostics       │
│  • Handle errors per action policy    │
│  • Log all operations                 │
└───────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────┐
│  STAGE 6: GEOMETRIC VALIDATION        │
│  ─────────────────────────────────    │
│  • Recompute all diagnostics          │
│  • Check watertight                   │
│  • Check manifold                     │
│  • Check self-intersections           │
│  • Determine if printable             │
└───────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────┐
│  STAGE 7: FIDELITY VALIDATION         │
│  ─────────────────────────────────    │
│  • Compare to baseline metrics        │
│  • Check volume change                │
│  • Check bounding box unchanged       │
│  • Compute Hausdorff distance         │
│  • Verify visual preservation         │
└───────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────┐
│  STAGE 8: ESCALATION (if needed)      │
│  ─────────────────────────────────    │
│  • If validation fails, escalate      │
│  • Run Blender-based repairs          │
│  • Re-validate after escalation       │
│  • Mark as failed if still invalid    │
└───────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────┐
│  STAGE 9: OUTPUT & REPORTING          │
│  ─────────────────────────────────    │
│  • Export repaired STL                │
│  • Generate report.json               │
│  • Generate report.csv summary        │
│  • Include fidelity metrics           │
└───────────────────────────────────────┘
            │
            ▼
     ┌──────────────┐
     │  Output STL  │
     └──────────────┘
```

---

## Stage Details

### Stage 1: Load & Baseline Capture

Before any modifications, capture the model's baseline state for later comparison.

```python
@dataclass
class BaselineMetrics:
    """Baseline metrics captured before repair."""
    
    # Identification
    source_path: Path
    fingerprint: str  # SHA256 of file contents
    
    # Geometry
    vertex_count: int
    face_count: int
    volume: float
    surface_area: float
    
    # Bounding box
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    bbox_dimensions: tuple[float, float, float]
    bbox_diagonal: float
    
    # Surface sampling (for Hausdorff comparison)
    surface_samples: np.ndarray  # N x 3 array of surface points
    sample_count: int = 10000
    
    # Centroid
    centroid: tuple[float, float, float]
```

**Why baseline capture matters:**
- Enables fidelity validation after repair
- Provides reference for volume/bbox change detection
- Surface samples enable Hausdorff distance computation

---

### Stage 2: Diagnostics Computation

Compute comprehensive diagnostics to understand the model's issues.

```python
@dataclass
class Diagnostics:
    """Complete diagnostics for a mesh."""
    
    # Basic topology
    is_watertight: bool
    is_volume: bool  # has positive volume
    vertex_count: int
    face_count: int
    edge_count: int
    
    # Holes and boundaries
    hole_count: int
    boundary_edge_count: int
    largest_hole_size: int  # edges in largest hole
    
    # Components
    component_count: int
    largest_component_pct: float
    
    # Manifold status
    non_manifold_edge_count: int
    non_manifold_vertex_count: int
    is_manifold: bool
    
    # Normals
    normal_consistency: float  # 0.0 - 1.0
    inverted_normal_count: int
    
    # Degenerates
    degenerate_face_count: int
    duplicate_face_count: int
    zero_area_face_count: int
    
    # Self-intersection
    has_self_intersections: bool
    self_intersection_count: int
    
    # Geometry
    volume: float
    surface_area: float
    bbox: tuple[float, float, float]
    genus: int
    euler_characteristic: int
    
    # Thin features
    estimated_min_thickness: float
    thin_region_count: int
    
    # Quality metrics
    avg_edge_length: float
    edge_length_variance: float
    triangle_aspect_ratio_avg: float
    sliver_triangle_count: int
```

---

### Stage 3: Profile Detection

Match diagnostics to model profiles using a rule engine.

```python
class ProfileMatcher:
    """Match diagnostics to model profiles."""
    
    def match(self, diagnostics: Diagnostics) -> list[ProfileMatch]:
        """Return profiles matching the diagnostics, sorted by confidence."""
        matches = []
        
        for profile in self.profiles:
            confidence = profile.compute_confidence(diagnostics)
            if confidence > 0.0:
                matches.append(ProfileMatch(
                    profile=profile,
                    confidence=confidence,
                    explanation=profile.explain(diagnostics)
                ))
        
        return sorted(matches, key=lambda m: m.confidence, reverse=True)
```

**Profile matching criteria examples:**

| Profile | Detection Logic |
|---------|-----------------|
| `clean` | `is_watertight AND is_manifold AND component_count == 1` |
| `holes-only` | `NOT is_watertight AND hole_count > 0 AND is_manifold` |
| `non-manifold` | `non_manifold_edge_count > 0 OR non_manifold_vertex_count > 0` |
| `fragmented` | `component_count > 3 AND largest_component_pct < 0.8` |
| `self-intersecting` | `has_self_intersections == true` |

---

### Stage 4: Filter Script Generation

Generate a suggested filter script based on the detected profile.

```python
def generate_filter_script(
    profile: ModelProfile,
    fingerprint: str,
    diagnostics: Diagnostics
) -> FilterScript:
    """Generate a filter script for the detected profile."""
    
    actions = []
    
    # Always start with basic cleanup
    actions.append(FilterAction(name="trimesh_basic"))
    
    # Add profile-specific actions
    for action_name in profile.suggested_actions:
        action_def = get_action_definition(action_name)
        params = compute_optimal_params(action_def, diagnostics)
        actions.append(FilterAction(name=action_name, params=params))
    
    # Always end with validation
    actions.append(FilterAction(name="validate"))
    
    return FilterScript(
        name=f"{profile.name}-suggested",
        version="1.0.0",
        meta=FilterScriptMeta(
            generated_by="model_scan",
            profile=profile.name,
            model_fingerprint=fingerprint,
            timestamp=datetime.now().isoformat()
        ),
        actions=actions
    )
```

---

### Stage 5: Filter Script Execution

Execute actions in order with per-step tracking.

```python
@dataclass
class StepResult:
    """Result of executing one filter action."""
    
    action_name: str
    status: str  # "success", "warning", "error", "skipped"
    runtime_ms: float
    
    # Diagnostics after this step
    diagnostics_after: Diagnostics
    
    # Changes
    vertex_count_delta: int
    face_count_delta: int
    
    # Messages
    message: str = ""
    error: str = ""
    warnings: list[str] = field(default_factory=list)

class FilterScriptRunner:
    """Execute a filter script on a mesh."""
    
    def run(self, script: FilterScript, mesh: Trimesh) -> RunResult:
        steps = []
        current_mesh = mesh
        
        for action in script.actions:
            start_time = time.perf_counter()
            
            try:
                current_mesh = self.execute_action(action, current_mesh)
                diagnostics = compute_diagnostics(current_mesh)
                
                steps.append(StepResult(
                    action_name=action.name,
                    status="success",
                    runtime_ms=(time.perf_counter() - start_time) * 1000,
                    diagnostics_after=diagnostics
                ))
                
            except Exception as e:
                if action.on_error == "abort":
                    raise
                elif action.on_error == "skip":
                    steps.append(StepResult(
                        action_name=action.name,
                        status="skipped",
                        error=str(e)
                    ))
        
        return RunResult(steps=steps, final_mesh=current_mesh)
```

---

### Stage 6: Geometric Validation

Verify the repaired mesh meets printability criteria.

```python
@dataclass
class GeometricValidation:
    """Result of geometric validation checks."""
    
    is_watertight: bool
    is_manifold: bool
    is_single_component: bool  # or intentional multi
    has_positive_volume: bool
    no_self_intersections: bool
    no_degenerate_faces: bool
    
    @property
    def is_printable(self) -> bool:
        """Check if mesh meets minimum printability requirements."""
        return (
            self.is_watertight and
            self.is_manifold and
            self.has_positive_volume and
            self.no_self_intersections
        )
    
    @property
    def issues(self) -> list[str]:
        """List remaining issues."""
        issues = []
        if not self.is_watertight:
            issues.append("Not watertight (has holes)")
        if not self.is_manifold:
            issues.append("Non-manifold geometry")
        if not self.has_positive_volume:
            issues.append("Zero or negative volume")
        if not self.no_self_intersections:
            issues.append("Self-intersections present")
        return issues
```

---

### Stage 7: Fidelity Validation

**Critical**: Ensure the repair didn't alter the model's visual appearance.

```python
@dataclass
class FidelityValidation:
    """Result of fidelity (visual preservation) validation."""
    
    # Volume comparison
    volume_before: float
    volume_after: float
    volume_change_pct: float
    volume_change_acceptable: bool
    
    # Bounding box comparison
    bbox_before: tuple[float, float, float]
    bbox_after: tuple[float, float, float]
    bbox_changed: bool
    
    # Surface deviation
    hausdorff_distance: float  # maximum surface deviation
    mean_surface_distance: float  # average surface deviation
    hausdorff_relative: float  # relative to bbox diagonal
    
    # Topology changes
    vertex_count_before: int
    vertex_count_after: int
    vertex_count_change_pct: float
    
    face_count_before: int
    face_count_after: int
    face_count_change_pct: float
    
    # Thresholds (configurable)
    max_volume_change_pct: float = 1.0  # 1%
    max_hausdorff_relative: float = 0.001  # 0.1% of bbox diagonal
    max_vertex_loss_pct: float = 5.0  # 5%
    
    @property
    def is_visually_unchanged(self) -> bool:
        """Check if model is within acceptable visual change limits."""
        return (
            abs(self.volume_change_pct) < self.max_volume_change_pct and
            self.hausdorff_relative < self.max_hausdorff_relative and
            not self.bbox_changed
        )
    
    @property
    def changes(self) -> list[str]:
        """List detected changes."""
        changes = []
        if abs(self.volume_change_pct) >= self.max_volume_change_pct:
            changes.append(f"Volume changed by {self.volume_change_pct:.2f}%")
        if self.bbox_changed:
            changes.append("Bounding box dimensions changed")
        if self.hausdorff_relative >= self.max_hausdorff_relative:
            changes.append(f"Surface deviation: {self.hausdorff_distance:.4f} units")
        return changes


def compute_fidelity(
    baseline: BaselineMetrics,
    repaired_mesh: Trimesh
) -> FidelityValidation:
    """Compare repaired mesh to baseline for fidelity."""
    
    # Volume comparison
    volume_after = repaired_mesh.volume
    volume_change_pct = ((volume_after - baseline.volume) / baseline.volume * 100
                         if baseline.volume != 0 else 0)
    
    # Bounding box comparison
    bbox_after = repaired_mesh.bounds
    bbox_dims_after = bbox_after[1] - bbox_after[0]
    bbox_changed = not np.allclose(
        baseline.bbox_dimensions, 
        bbox_dims_after, 
        rtol=1e-6
    )
    
    # Surface deviation (Hausdorff distance)
    samples_after = repaired_mesh.sample(baseline.sample_count)
    
    # Compute directed Hausdorff distances
    from scipy.spatial import cKDTree
    tree_before = cKDTree(baseline.surface_samples)
    tree_after = cKDTree(samples_after)
    
    dist_before_to_after, _ = tree_after.query(baseline.surface_samples)
    dist_after_to_before, _ = tree_before.query(samples_after)
    
    hausdorff_distance = max(dist_before_to_after.max(), dist_after_to_before.max())
    mean_surface_distance = (dist_before_to_after.mean() + dist_after_to_before.mean()) / 2
    hausdorff_relative = hausdorff_distance / baseline.bbox_diagonal
    
    return FidelityValidation(
        volume_before=baseline.volume,
        volume_after=volume_after,
        volume_change_pct=volume_change_pct,
        volume_change_acceptable=abs(volume_change_pct) < 1.0,
        bbox_before=baseline.bbox_dimensions,
        bbox_after=tuple(bbox_dims_after),
        bbox_changed=bbox_changed,
        hausdorff_distance=hausdorff_distance,
        mean_surface_distance=mean_surface_distance,
        hausdorff_relative=hausdorff_relative,
        vertex_count_before=baseline.vertex_count,
        vertex_count_after=len(repaired_mesh.vertices),
        vertex_count_change_pct=((len(repaired_mesh.vertices) - baseline.vertex_count) 
                                  / baseline.vertex_count * 100),
        face_count_before=baseline.face_count,
        face_count_after=len(repaired_mesh.faces),
        face_count_change_pct=((len(repaired_mesh.faces) - baseline.face_count) 
                                / baseline.face_count * 100)
    )
```

---

### Stage 8: Escalation

If validation fails, escalate to more aggressive repairs.

```python
class EscalationManager:
    """Manage escalation to advanced repair methods."""
    
    ESCALATION_LEVELS = [
        # Level 0: Standard trimesh + pymeshfix
        ["trimesh_basic", "pymeshfix_repair", "fill_holes", "fix_normals"],
        
        # Level 1: More aggressive pymeshfix
        ["pymeshfix_repair_aggressive", "fill_holes", "fix_normals"],
        
        # Level 2: Blender boolean cleanup
        ["blender_boolean_union", "blender_triangulate"],
        
        # Level 3: Blender remesh (last resort, may alter geometry)
        ["blender_remesh"],
    ]
    
    def escalate(
        self,
        mesh: Trimesh,
        current_level: int,
        diagnostics: Diagnostics
    ) -> tuple[Trimesh, int]:
        """Attempt next escalation level."""
        
        if current_level >= len(self.ESCALATION_LEVELS):
            raise EscalationExhausted("All escalation levels tried")
        
        actions = self.ESCALATION_LEVELS[current_level]
        
        for action_name in actions:
            mesh = execute_action(action_name, mesh)
        
        return mesh, current_level + 1
```

**Escalation warnings:**
- Level 3 (remesh) may alter model geometry
- User consent required for destructive escalation
- Fidelity validation is critical after escalation

---

### Stage 9: Output & Reporting

Generate comprehensive output and reports.

```python
@dataclass
class RepairReport:
    """Complete report of a repair operation."""
    
    # Input
    input_path: str
    input_fingerprint: str
    
    # Profile
    detected_profile: str
    profile_confidence: float
    profile_explanation: str
    
    # Execution
    filter_script: FilterScript
    steps: list[StepResult]
    total_runtime_ms: float
    escalation_level: int
    
    # Validation
    geometric_validation: GeometricValidation
    fidelity_validation: FidelityValidation
    
    # Result
    success: bool
    output_path: str
    
    # Diagnostics
    diagnostics_before: Diagnostics
    diagnostics_after: Diagnostics
    
    # Environment
    tool_versions: dict[str, str]
    platform: str
    timestamp: str
    
    def to_json(self) -> str:
        """Export report as JSON."""
        return json.dumps(asdict(self), indent=2, default=str)
    
    def to_csv_row(self) -> dict:
        """Export summary as CSV row."""
        return {
            "filename": Path(self.input_path).name,
            "status": "success" if self.success else "failed",
            "profile": self.detected_profile,
            "confidence": self.profile_confidence,
            "runtime_ms": self.total_runtime_ms,
            "watertight_before": self.diagnostics_before.is_watertight,
            "watertight_after": self.diagnostics_after.is_watertight,
            "manifold_after": self.geometric_validation.is_manifold,
            "volume_change_pct": self.fidelity_validation.volume_change_pct,
            "visually_unchanged": self.fidelity_validation.is_visually_unchanged,
            "escalation_level": self.escalation_level,
            "output_path": self.output_path
        }
```

---

## Success Criteria

A repair is considered **successful** if:

1. **Geometric Validity**
   - ✅ Mesh is watertight
   - ✅ Mesh is manifold (no non-manifold edges/vertices)
   - ✅ Mesh has positive volume
   - ✅ No self-intersections (or below threshold)
   - ✅ No degenerate faces

2. **Visual Fidelity**
   - ✅ Volume change < 1%
   - ✅ Bounding box unchanged
   - ✅ Hausdorff distance < 0.1% of bbox diagonal
   - ✅ No significant vertex/face loss (unless intentional cleanup)

3. **Practical Printability**
   - ✅ Single component (or intentional multi-part)
   - ✅ No trapped volumes without drainage
   - ✅ Minimum wall thickness met (if specified)

---

## Action Risk Classification

Actions are classified by their impact on visual fidelity:

| Risk Level | Actions | Visual Impact | Use Case |
|------------|---------|---------------|----------|
| 🟢 **Safe** | `merge_vertices`, `remove_degenerate_faces`, `fix_normals`, `recalculate_normals` | None | Always safe to use |
| 🟢 **Safe** | `remove_duplicate_faces`, `remove_unreferenced_vertices` | None | Cleanup only |
| 🟡 **Low** | `fill_holes` (small), `stitch_boundaries` | Adds minimal geometry | Close small gaps |
| 🟡 **Low** | `pymeshfix_repair` | May modify topology | Standard repair |
| 🟠 **Medium** | `fill_holes` (large), `remove_small_components` | Removes/adds geometry | May affect detail |
| 🟠 **Medium** | `smooth_laplacian`, `smooth_taubin` | Alters surface | Noise reduction |
| 🟠 **Medium** | `decimate` | Reduces detail | Optimization |
| 🔴 **High** | `boolean_union` | Changes topology | Merge shells |
| 🔴 **High** | `blender_remesh` | Reconstructs surface | Last resort |
| 🔴 **High** | `blender_solidify` | Adds thickness | Thin wall fix |

---

## Configuration

### Fidelity Thresholds

```yaml
# config/fidelity_thresholds.yaml
fidelity:
  max_volume_change_pct: 1.0      # Maximum acceptable volume change
  max_hausdorff_relative: 0.001   # Max surface deviation (relative to bbox)
  max_vertex_loss_pct: 5.0        # Max vertex count reduction
  max_face_loss_pct: 5.0          # Max face count reduction
  bbox_tolerance: 1e-6            # Bounding box comparison tolerance

validation:
  strict_mode: true               # Fail on any fidelity violation
  warn_on_topology_change: true   # Warn if topology significantly changes
```

### Escalation Policy

```yaml
# config/escalation_policy.yaml
escalation:
  enabled: true
  max_level: 2                    # Don't auto-escalate to remesh
  require_consent_level: 2        # Require user consent at this level
  blender_required_level: 2       # Blender needed at this level
```

---

## See Also

- [Filter Actions Reference](filter_actions.md) - Complete action catalog
- [Model Profiles](model_profiles.md) - Profile detection and matching
- [Validation Guide](validation.md) - Detailed validation criteria
- [Thingi10K Testing](thingi10k_testing.md) - Benchmark testing procedures
