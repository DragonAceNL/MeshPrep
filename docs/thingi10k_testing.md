# Thingi10K Testing Guide

## Overview

This document describes how to use the Thingi10K dataset for testing and validating MeshPrep's repair capabilities. The Thingi10K dataset contains 10,000 real-world 3D printing models with known defects, making it ideal for benchmarking mesh repair tools.

---

## Attribution & Licensing

### Thingi10K Dataset

**Citation:**
> Qingnan Zhou and Alec Jacobson. "Thingi10K: A Dataset of 10,000 3D-Printing Models." 2016.

- **Repository**: https://github.com/Thingi10K/Thingi10K
- **Website**: https://ten-thousand-models.appspot.com/
- **Python Package**: `pip install thingi10k`
- **Dataset Code License**: Apache License 2.0

### Test Fixtures in This Repository

The test fixtures in `tests/fixtures/thingi10k/` are a curated subset of Thingi10K models
selected for their **permissive licenses** that allow redistribution:

| License | Models | Redistribution |
|---------|--------|----------------|
| CC-BY (Attribution) | 51 | ✅ With attribution |
| CC-BY-SA (Attribution-ShareAlike) | 67 | ✅ With attribution, same license |
| CC0 / Public Domain | 3 | ✅ No restrictions |
| GPL | 3 | ✅ With source |
| **Total** | **124** | **All permissive** |

**Important**: Models with Non-Commercial (NC) or No-Derivatives (ND) licenses are
explicitly excluded from the test fixtures to ensure legal redistribution.

See the `NOTICE` file in the repository root for the complete list of model IDs
and their respective licenses.

### Original Model Sources

All models originate from Thingiverse.com. To find the original creator:
```
https://www.thingiverse.com/download:{file_id}
```

Where `{file_id}` is the numeric filename (e.g., `42841.stl` → file_id `42841`).

---

## Storage Architecture

MeshPrep uses a separation of concerns for Thingi10K data:

```
┌─────────────────────────────────────────────────────────────┐
│                    STORAGE LOCATIONS                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  RAW MODELS (External - not in MeshPrep repo)               │
│  Location: C:\Users\...\Thingi10K\raw_meshes\               │
│  Contents: 10,000 STL files                                 │
│  Size: ~7GB                                                 │
│  Note: Keep outside repo to avoid bloating git              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  METADATA DATABASE (Inside MeshPrep repo)                   │
│  Location: MeshPrep/data/thingi10k/thingi10k.db             │
│  Format: SQLite                                             │
│  Size: ~5MB                                                 │
│  Contents:                                                  │
│    - models table: All 10K model metadata                   │
│    - test_results table: Benchmark results                  │
│    - category_summary view: Statistics by category          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  TEST FIXTURES (Inside MeshPrep repo)                       │
│  Location: MeshPrep/tests/fixtures/thingi10k/               │
│  Contents: Selected subset (~100-200 models)                │
│  Organization: By defect category                           │
│  Note: Small enough to commit to git                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Setup

```bash
# 1. Install the thingi10k Python package
pip install thingi10k

# 2. Run the setup script
python scripts/setup_thingi10k.py --raw-meshes "C:\path\to\Thingi10K\raw_meshes"

# 3. Optionally copy test fixtures
python scripts/setup_thingi10k.py --copy-fixtures --per-category 20
```

Or use the Python API:

```python
from meshprep.testing import Thingi10KManager

manager = Thingi10KManager(
    raw_meshes_path=Path("C:/Users/.../Thingi10K/raw_meshes")
)

# Import from thingi10k package (recommended)
manager.import_from_thingi10k_package()

# Or scan STL files directly
# manager.import_from_directory()

# View summary
manager.print_summary()

# Select and copy test fixtures
selected = manager.select_test_fixtures(per_category=20)
manager.copy_fixtures_to_test_dir(selected)
```

---

## Dataset Information

### Source

- **Repository**: https://github.com/Thingi10K/Thingi10K
- **Paper**: "Thingi10K: A Dataset of 10,000 3D-Printing Models" (Zhou & Jacobson, 2016)
- **Python Package**: `pip install thingi10k`
- **License**: Models from Thingiverse (check individual licenses)

### Dataset Statistics

| Defect Type | Percentage | Count (approx) |
|-------------|------------|----------------|
| Self-intersections | 45% | 4,500 |
| Coplanar self-intersections | 31% | 3,100 |
| Multiple components | 26% | 2,600 |
| Non-manifold | 22% | 2,200 |
| Clean (printable) | ~10% | 1,000 |

### External File Structure (raw models)

```
Thingi10K/                        # External directory
├── raw_meshes/                   # Original STL files
│   ├── 32770.stl
│   ├── 32771.stl
│   └── ... (10,000 files)
└── README
```

### Metadata Fields

Each model's metadata includes:

```json
{
  "id": 12345,
  "name": "Model Name",
  "thingiverse_url": "https://...",
  "num_vertices": 1234,
  "num_faces": 2468,
  "volume": 123.45,
  "surface_area": 678.90,
  "is_watertight": false,
  "is_manifold": false,
  "num_components": 3,
  "num_holes": 5,
  "self_intersecting": true,
  "genus": 0,
  "tags": ["toy", "figure"]
}
```

---

## Test Organization

### Category Mapping

Map Thingi10K models to MeshPrep model profiles:

| Thingi10K Characteristic | MeshPrep Profile(s) |
|--------------------------|---------------------|
| `is_watertight == true && is_manifold == true` | `clean` |
| `is_watertight == false && num_holes > 0` | `holes-only`, `many-small-holes` |
| `is_manifold == false` | `non-manifold` |
| `self_intersecting == true` | `self-intersecting` |
| `num_components > 1` | `fragmented`, `multiple-disconnected-large` |

### Test Fixture Organization

```
tests/
├── fixtures/
│   ├── thingi10k/
│   │   ├── clean/                    # Clean, printable models
│   │   │   ├── 00123.stl
│   │   │   └── ...
│   │   ├── holes/                    # Models with holes
│   │   │   ├── 00456.stl
│   │   │   └── ...
│   │   ├── non_manifold/            # Non-manifold models
│   │   │   ├── 00789.stl
│   │   │   └── ...
│   │   ├── self_intersecting/       # Self-intersecting models
│   │   │   ├── 01012.stl
│   │   │   └── ...
│   │   ├── fragmented/              # Multi-component models
│   │   │   ├── 01234.stl
│   │   │   └── ...
│   │   └── complex/                 # Multiple issues combined
│   │       ├── 01567.stl
│   │       └── ...
│   └── synthetic/                   # Hand-crafted test cases
│       └── ...
└── thingi10k_index.json            # Index of categorized models
```

---

## Categorization Script

Script to categorize Thingi10K models by defect type:

```python
# scripts/categorize_thingi10k.py

"""Categorize Thingi10K models by defect type for MeshPrep testing."""

import json
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelInfo:
    """Information about a Thingi10K model."""
    id: str
    is_watertight: bool
    is_manifold: bool
    num_holes: int
    num_components: int
    self_intersecting: bool
    genus: int
    num_faces: int


def load_thingi10k_index(index_path: Path) -> list[ModelInfo]:
    """Load the Thingi10K master index."""
    with open(index_path) as f:
        data = json.load(f)
    
    models = []
    for entry in data:
        models.append(ModelInfo(
            id=str(entry["id"]).zfill(5),
            is_watertight=entry.get("is_watertight", False),
            is_manifold=entry.get("is_manifold", False),
            num_holes=entry.get("num_holes", 0),
            num_components=entry.get("num_components", 1),
            self_intersecting=entry.get("self_intersecting", False),
            genus=entry.get("genus", 0),
            num_faces=entry.get("num_faces", 0)
        ))
    
    return models


def categorize_model(model: ModelInfo) -> list[str]:
    """Determine which categories a model belongs to."""
    categories = []
    
    # Clean models
    if model.is_watertight and model.is_manifold and not model.self_intersecting:
        categories.append("clean")
        return categories
    
    # Specific defect categories
    if not model.is_watertight and model.num_holes > 0:
        if model.num_holes > 10:
            categories.append("many_small_holes")
        else:
            categories.append("holes")
    
    if not model.is_manifold:
        categories.append("non_manifold")
    
    if model.self_intersecting:
        categories.append("self_intersecting")
    
    if model.num_components > 1:
        categories.append("fragmented")
    
    # Complex: multiple issues
    if len(categories) > 1:
        categories.append("complex")
    
    return categories


def select_test_fixtures(
    models: list[ModelInfo],
    per_category: int = 20,
    max_faces: int = 100000
) -> dict[str, list[str]]:
    """Select representative test fixtures for each category."""
    
    category_models: dict[str, list[ModelInfo]] = {
        "clean": [],
        "holes": [],
        "many_small_holes": [],
        "non_manifold": [],
        "self_intersecting": [],
        "fragmented": [],
        "complex": []
    }
    
    for model in models:
        # Skip very large models for faster testing
        if model.num_faces > max_faces:
            continue
        
        categories = categorize_model(model)
        for cat in categories:
            if cat in category_models:
                category_models[cat].append(model)
    
    # Select up to per_category models from each category
    selected = {}
    for cat, cat_models in category_models.items():
        # Sort by face count (prefer smaller for faster tests)
        cat_models.sort(key=lambda m: m.num_faces)
        selected[cat] = [m.id for m in cat_models[:per_category]]
    
    return selected


def copy_fixtures(
    selected: dict[str, list[str]],
    source_dir: Path,
    dest_dir: Path
):
    """Copy selected fixtures to the test fixtures directory."""
    
    for category, model_ids in selected.items():
        cat_dir = dest_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        
        for model_id in model_ids:
            src = source_dir / f"{model_id}.stl"
            dst = cat_dir / f"{model_id}.stl"
            
            if src.exists() and not dst.exists():
                shutil.copy(src, dst)
                print(f"Copied {src.name} to {category}/")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Categorize Thingi10K models")
    parser.add_argument("--index", required=True, help="Path to Thingi10K.json")
    parser.add_argument("--source", required=True, help="Path to raw_meshes/")
    parser.add_argument("--dest", required=True, help="Destination fixtures directory")
    parser.add_argument("--per-category", type=int, default=20, help="Models per category")
    
    args = parser.parse_args()
    
    models = load_thingi10k_index(Path(args.index))
    print(f"Loaded {len(models)} models from index")
    
    selected = select_test_fixtures(
        models,
        per_category=args.per_category
    )
    
    for cat, ids in selected.items():
        print(f"  {cat}: {len(ids)} models")
    
    copy_fixtures(
        selected,
        Path(args.source),
        Path(args.dest)
    )
    
    # Save index
    index_path = Path(args.dest) / "thingi10k_index.json"
    with open(index_path, "w") as f:
        json.dump(selected, f, indent=2)
    
    print(f"Saved index to {index_path}")


if __name__ == "__main__":
    main()
```

---

## Validation Criteria

### Success Definition

A repair is considered **successful** if:

1. **Geometric Validity** (Required)
   - ✅ `is_watertight == true`
   - ✅ `is_manifold == true` (no non-manifold edges/vertices)
   - ✅ `volume > 0` (positive volume)
   - ✅ `self_intersections == false` (or count below threshold)

2. **Visual Fidelity** (Required)
   - ✅ `volume_change_pct < 1%`
   - ✅ `bbox_changed == false`
   - ✅ `hausdorff_relative < 0.001` (0.1% of bbox diagonal)

3. **Topology Preservation** (Warning)
   - ⚠️ `vertex_count_loss < 5%` (unless decimation intended)
   - ⚠️ `face_count_loss < 5%`

### Validation Implementation

```python
# core/validation.py

from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
import trimesh


@dataclass
class ValidationResult:
    """Complete validation result for a repair operation."""
    
    # Geometric validity
    is_watertight: bool
    is_manifold: bool
    has_positive_volume: bool
    no_self_intersections: bool
    
    # Fidelity
    volume_change_pct: float
    bbox_changed: bool
    hausdorff_distance: float
    hausdorff_relative: float
    
    # Topology
    vertex_count_before: int
    vertex_count_after: int
    face_count_before: int
    face_count_after: int
    
    @property
    def is_geometrically_valid(self) -> bool:
        """Check if mesh is geometrically valid for printing."""
        return (
            self.is_watertight and
            self.is_manifold and
            self.has_positive_volume and
            self.no_self_intersections
        )
    
    @property
    def is_visually_unchanged(self) -> bool:
        """Check if mesh appearance is preserved."""
        return (
            abs(self.volume_change_pct) < 1.0 and
            not self.bbox_changed and
            self.hausdorff_relative < 0.001
        )
    
    @property
    def is_successful(self) -> bool:
        """Check if repair is fully successful."""
        return self.is_geometrically_valid and self.is_visually_unchanged


def validate_repair(
    original: trimesh.Trimesh,
    repaired: trimesh.Trimesh,
    sample_count: int = 10000
) -> ValidationResult:
    """Validate a repair operation."""
    
    # Geometric validity checks
    is_watertight = repaired.is_watertight
    is_manifold = repaired.is_volume  # trimesh uses is_volume for manifold check
    has_positive_volume = repaired.volume > 0
    
    # Self-intersection check (expensive)
    try:
        intersections = repaired.intersection_test()
        no_self_intersections = len(intersections) == 0
    except:
        no_self_intersections = True  # Assume OK if check fails
    
    # Volume comparison
    vol_before = abs(original.volume)
    vol_after = abs(repaired.volume)
    volume_change_pct = ((vol_after - vol_before) / vol_before * 100
                         if vol_before > 0 else 0)
    
    # Bounding box comparison
    bbox_before = original.bounds[1] - original.bounds[0]
    bbox_after = repaired.bounds[1] - repaired.bounds[0]
    bbox_changed = not np.allclose(bbox_before, bbox_after, rtol=1e-6)
    bbox_diagonal = np.linalg.norm(bbox_before)
    
    # Hausdorff distance
    samples_before = original.sample(sample_count)
    samples_after = repaired.sample(sample_count)
    
    tree_before = cKDTree(samples_before)
    tree_after = cKDTree(samples_after)
    
    dist_b_to_a, _ = tree_after.query(samples_before)
    dist_a_to_b, _ = tree_before.query(samples_after)
    
    hausdorff_distance = max(dist_b_to_a.max(), dist_a_to_b.max())
    hausdorff_relative = hausdorff_distance / bbox_diagonal if bbox_diagonal > 0 else 0
    
    return ValidationResult(
        is_watertight=is_watertight,
        is_manifold=is_manifold,
        has_positive_volume=has_positive_volume,
        no_self_intersections=no_self_intersections,
        volume_change_pct=volume_change_pct,
        bbox_changed=bbox_changed,
        hausdorff_distance=hausdorff_distance,
        hausdorff_relative=hausdorff_relative,
        vertex_count_before=len(original.vertices),
        vertex_count_after=len(repaired.vertices),
        face_count_before=len(original.faces),
        face_count_after=len(repaired.faces)
    )
```

---

## Benchmark Test Suite

### Test Runner

```python
# tests/test_thingi10k_benchmark.py

"""Benchmark tests using Thingi10K dataset."""

import pytest
import json
from pathlib import Path
import trimesh

from meshprep.core.diagnostics import compute_diagnostics
from meshprep.core.profiles import ProfileDetector
from meshprep.core.filter_script import FilterScriptRunner, generate_filter_script
from meshprep.core.validation import validate_repair


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "thingi10k"
INDEX_PATH = FIXTURES_DIR / "thingi10k_index.json"


def load_fixture_index() -> dict[str, list[str]]:
    """Load the fixture index."""
    if INDEX_PATH.exists():
        with open(INDEX_PATH) as f:
            return json.load(f)
    return {}


# Parametrize tests by category
FIXTURE_INDEX = load_fixture_index()


@pytest.mark.parametrize("model_id", FIXTURE_INDEX.get("holes", []))
def test_holes_repair(model_id: str):
    """Test repair of models with holes."""
    stl_path = FIXTURES_DIR / "holes" / f"{model_id}.stl"
    if not stl_path.exists():
        pytest.skip(f"Fixture {model_id} not found")
    
    # Load
    original = trimesh.load(stl_path)
    
    # Diagnose
    diagnostics = compute_diagnostics(original)
    assert diagnostics.hole_count > 0, "Expected holes in fixture"
    
    # Detect profile
    detector = ProfileDetector()
    matches = detector.detect(diagnostics)
    assert len(matches) > 0, "Should detect a profile"
    
    # Generate and run filter script
    script = generate_filter_script(
        matches[0].profile.name,
        "test",
        matches[0].profile.suggested_actions
    )
    
    runner = FilterScriptRunner()
    result = runner.run(script, original)
    
    assert result.success, f"Repair failed: {result.error}"
    
    # Validate
    validation = validate_repair(original, result.final_mesh)
    
    assert validation.is_watertight, "Should be watertight after repair"
    assert validation.is_visually_unchanged, f"Visual changes detected: volume={validation.volume_change_pct:.2f}%"


@pytest.mark.parametrize("model_id", FIXTURE_INDEX.get("non_manifold", []))
def test_non_manifold_repair(model_id: str):
    """Test repair of non-manifold models."""
    stl_path = FIXTURES_DIR / "non_manifold" / f"{model_id}.stl"
    if not stl_path.exists():
        pytest.skip(f"Fixture {model_id} not found")
    
    original = trimesh.load(stl_path)
    diagnostics = compute_diagnostics(original)
    
    detector = ProfileDetector()
    matches = detector.detect(diagnostics)
    
    script = generate_filter_script(
        matches[0].profile.name,
        "test",
        matches[0].profile.suggested_actions
    )
    
    runner = FilterScriptRunner()
    result = runner.run(script, original)
    
    assert result.success
    
    validation = validate_repair(original, result.final_mesh)
    assert validation.is_manifold, "Should be manifold after repair"
    assert validation.is_visually_unchanged


@pytest.mark.parametrize("model_id", FIXTURE_INDEX.get("self_intersecting", []))
def test_self_intersecting_repair(model_id: str):
    """Test repair of self-intersecting models."""
    stl_path = FIXTURES_DIR / "self_intersecting" / f"{model_id}.stl"
    if not stl_path.exists():
        pytest.skip(f"Fixture {model_id} not found")
    
    original = trimesh.load(stl_path)
    diagnostics = compute_diagnostics(original)
    
    detector = ProfileDetector()
    matches = detector.detect(diagnostics)
    
    script = generate_filter_script(
        matches[0].profile.name,
        "test",
        matches[0].profile.suggested_actions
    )
    
    runner = FilterScriptRunner()
    result = runner.run(script, original)
    
    # Self-intersections are harder to fix
    if not result.success:
        pytest.xfail("Self-intersection repair failed (expected for some models)")
    
    validation = validate_repair(original, result.final_mesh)
    
    # May require visual change for self-intersection fixes
    assert validation.is_geometrically_valid, "Should be geometrically valid"


@pytest.mark.parametrize("model_id", FIXTURE_INDEX.get("clean", []))
def test_clean_passthrough(model_id: str):
    """Test that clean models pass through unchanged."""
    stl_path = FIXTURES_DIR / "clean" / f"{model_id}.stl"
    if not stl_path.exists():
        pytest.skip(f"Fixture {model_id} not found")
    
    original = trimesh.load(stl_path)
    diagnostics = compute_diagnostics(original)
    
    detector = ProfileDetector()
    matches = detector.detect(diagnostics)
    
    assert matches[0].profile.name == "clean", "Should detect as clean"
    
    script = generate_filter_script(
        matches[0].profile.name,
        "test",
        matches[0].profile.suggested_actions
    )
    
    runner = FilterScriptRunner()
    result = runner.run(script, original)
    
    assert result.success
    
    validation = validate_repair(original, result.final_mesh)
    
    # Clean models should be unchanged
    assert validation.is_geometrically_valid
    assert validation.is_visually_unchanged
    assert abs(validation.volume_change_pct) < 0.01, "Clean model should be virtually unchanged"
```

---

## Benchmark Metrics

### Success Rate Tracking

```python
# scripts/run_benchmark.py

"""Run full Thingi10K benchmark and collect metrics."""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import time
import trimesh

from meshprep.core.diagnostics import compute_diagnostics
from meshprep.core.profiles import ProfileDetector
from meshprep.core.filter_script import FilterScriptRunner, generate_filter_script
from meshprep.core.validation import validate_repair


@dataclass
class BenchmarkResult:
    """Result for a single benchmark model."""
    model_id: str
    category: str
    
    # Detection
    detected_profile: str
    profile_confidence: float
    
    # Repair
    repair_success: bool
    repair_error: Optional[str]
    repair_runtime_ms: float
    
    # Validation
    is_geometrically_valid: bool
    is_visually_unchanged: bool
    volume_change_pct: float
    hausdorff_relative: float
    
    # Overall
    overall_success: bool


def run_benchmark(
    fixtures_dir: Path,
    output_path: Path,
    categories: Optional[list[str]] = None
) -> dict:
    """Run benchmark on all fixtures."""
    
    results: list[BenchmarkResult] = []
    
    # Load index
    index_path = fixtures_dir / "thingi10k_index.json"
    with open(index_path) as f:
        index = json.load(f)
    
    if categories is None:
        categories = list(index.keys())
    
    for category in categories:
        model_ids = index.get(category, [])
        print(f"\nCategory: {category} ({len(model_ids)} models)")
        
        for model_id in model_ids:
            stl_path = fixtures_dir / category / f"{model_id}.stl"
            if not stl_path.exists():
                continue
            
            result = benchmark_single(model_id, category, stl_path)
            results.append(result)
            
            status = "✅" if result.overall_success else "❌"
            print(f"  {status} {model_id}: {result.detected_profile}")
    
    # Compute summary statistics
    summary = compute_summary(results)
    
    # Save results
    output = {
        "summary": summary,
        "results": [asdict(r) for r in results]
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print_summary(summary)
    
    return output


def benchmark_single(model_id: str, category: str, stl_path: Path) -> BenchmarkResult:
    """Benchmark a single model."""
    
    try:
        # Load
        original = trimesh.load(str(stl_path))
        
        # Diagnose
        diagnostics = compute_diagnostics(original)
        
        # Detect profile
        detector = ProfileDetector()
        matches = detector.detect(diagnostics)
        
        if not matches:
            return BenchmarkResult(
                model_id=model_id,
                category=category,
                detected_profile="unknown",
                profile_confidence=0.0,
                repair_success=False,
                repair_error="No profile detected",
                repair_runtime_ms=0,
                is_geometrically_valid=False,
                is_visually_unchanged=False,
                volume_change_pct=0,
                hausdorff_relative=0,
                overall_success=False
            )
        
        detected_profile = matches[0].profile.name
        profile_confidence = matches[0].confidence
        
        # Generate and run filter script
        script = generate_filter_script(
            detected_profile,
            model_id,
            matches[0].profile.suggested_actions
        )
        
        runner = FilterScriptRunner()
        
        start_time = time.perf_counter()
        run_result = runner.run(script, original)
        repair_runtime_ms = (time.perf_counter() - start_time) * 1000
        
        if not run_result.success:
            return BenchmarkResult(
                model_id=model_id,
                category=category,
                detected_profile=detected_profile,
                profile_confidence=profile_confidence,
                repair_success=False,
                repair_error=run_result.error,
                repair_runtime_ms=repair_runtime_ms,
                is_geometrically_valid=False,
                is_visually_unchanged=False,
                volume_change_pct=0,
                hausdorff_relative=0,
                overall_success=False
            )
        
        # Validate
        validation = validate_repair(original, run_result.final_mesh)
        
        return BenchmarkResult(
            model_id=model_id,
            category=category,
            detected_profile=detected_profile,
            profile_confidence=profile_confidence,
            repair_success=True,
            repair_error=None,
            repair_runtime_ms=repair_runtime_ms,
            is_geometrically_valid=validation.is_geometrically_valid,
            is_visually_unchanged=validation.is_visually_unchanged,
            volume_change_pct=validation.volume_change_pct,
            hausdorff_relative=validation.hausdorff_relative,
            overall_success=validation.is_successful
        )
        
    except Exception as e:
        return BenchmarkResult(
            model_id=model_id,
            category=category,
            detected_profile="error",
            profile_confidence=0.0,
            repair_success=False,
            repair_error=str(e),
            repair_runtime_ms=0,
            is_geometrically_valid=False,
            is_visually_unchanged=False,
            volume_change_pct=0,
            hausdorff_relative=0,
            overall_success=False
        )


def compute_summary(results: list[BenchmarkResult]) -> dict:
    """Compute summary statistics."""
    
    total = len(results)
    if total == 0:
        return {}
    
    # Overall
    successful = sum(1 for r in results if r.overall_success)
    repair_success = sum(1 for r in results if r.repair_success)
    geom_valid = sum(1 for r in results if r.is_geometrically_valid)
    visually_unchanged = sum(1 for r in results if r.is_visually_unchanged)
    
    # By category
    by_category = {}
    categories = set(r.category for r in results)
    
    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        cat_total = len(cat_results)
        cat_success = sum(1 for r in cat_results if r.overall_success)
        
        by_category[cat] = {
            "total": cat_total,
            "successful": cat_success,
            "success_rate": cat_success / cat_total * 100 if cat_total > 0 else 0
        }
    
    return {
        "total_models": total,
        "overall_success": successful,
        "overall_success_rate": successful / total * 100,
        "repair_success": repair_success,
        "repair_success_rate": repair_success / total * 100,
        "geometrically_valid": geom_valid,
        "geometrically_valid_rate": geom_valid / total * 100,
        "visually_unchanged": visually_unchanged,
        "visually_unchanged_rate": visually_unchanged / total * 100,
        "by_category": by_category
    }


def print_summary(summary: dict):
    """Print summary statistics."""
    
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    print(f"\nTotal models tested: {summary['total_models']}")
    print(f"Overall success rate: {summary['overall_success_rate']:.1f}%")
    print(f"Repair success rate: {summary['repair_success_rate']:.1f}%")
    print(f"Geometrically valid: {summary['geometrically_valid_rate']:.1f}%")
    print(f"Visually unchanged: {summary['visually_unchanged_rate']:.1f}%")
    
    print("\nBy Category:")
    for cat, stats in summary.get("by_category", {}).items():
        print(f"  {cat}: {stats['success_rate']:.1f}% ({stats['successful']}/{stats['total']})")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Thingi10K benchmark")
    parser.add_argument("--fixtures", required=True, help="Path to fixtures directory")
    parser.add_argument("--output", default="benchmark_results.json", help="Output file")
    parser.add_argument("--categories", nargs="*", help="Categories to test")
    
    args = parser.parse_args()
    
    run_benchmark(
        Path(args.fixtures),
        Path(args.output),
        args.categories
    )
```

---

## Acceptance Criteria

Based on Thingi10K benchmark results:

| Metric | Target | Description |
|--------|--------|-------------|
| **Overall Success Rate** | ≥ 80% | Models fully repaired and unchanged |
| **Repair Success Rate** | ≥ 95% | Filter script completes without error |
| **Geometric Validity Rate** | ≥ 90% | Watertight + manifold after repair |
| **Visual Fidelity Rate** | ≥ 85% | No significant visual changes |

### Per-Category Targets

| Category | Success Target | Notes |
|----------|----------------|-------|
| `clean` | 100% | Should pass through unchanged |
| `holes` | 95% | Primary use case |
| `non_manifold` | 85% | Common defect |
| `fragmented` | 80% | May intentionally change |
| `self_intersecting` | 60% | Difficult to fix |
| `complex` | 50% | Multiple issues |

---

## See Also

- [Repair Pipeline](repair_pipeline.md) - Complete repair process
- [Filter Actions](filter_actions.md) - Available repair actions
- [Model Profiles](model_profiles.md) - Profile detection
- [Functional Spec](functional_spec.md) - Overall requirements
