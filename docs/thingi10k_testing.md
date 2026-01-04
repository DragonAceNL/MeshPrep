# Thingi10K Testing Guide

## Overview

The Thingi10K dataset (10,000 real-world 3D printing models) is used to benchmark MeshPrep's repair capabilities.

---

## Attribution

**Citation:** Zhou & Jacobson, "Thingi10K: A Dataset of 10,000 3D-Printing Models" (2016)

- Repository: https://github.com/Thingi10K/Thingi10K
- Python: `pip install thingi10k`
- License: Apache 2.0 (code), individual model licenses vary

Test fixtures in `tests/fixtures/thingi10k/` are curated for **permissive licenses** (CC-BY, CC0, GPL).

---

## Storage Architecture

| Location | Contents | In Repo? |
|----------|----------|----------|
| External `raw_meshes/` | 10,000 STL files (~7GB) | ❌ No |
| `data/thingi10k/thingi10k.db` | SQLite metadata (~5MB) | ✅ Yes |
| `tests/fixtures/thingi10k/` | Curated subset (~100-200) | ✅ Yes |

---

## Quick Setup

```bash
pip install thingi10k
python scripts/setup_thingi10k.py --raw-meshes "/path/to/Thingi10K/raw_meshes"
python scripts/setup_thingi10k.py --copy-fixtures --per-category 20
```

---

## Dataset Statistics

| Defect Type | Percentage |
|-------------|------------|
| Self-intersections | 45% |
| Coplanar self-intersections | 31% |
| Multiple components | 26% |
| Non-manifold | 22% |
| Clean (printable) | ~10% |

---

## Category Mapping

| Thingi10K Characteristic | MeshPrep Profile |
|--------------------------|------------------|
| Watertight + manifold | `clean` |
| Has holes | `holes-only` |
| Non-manifold | `non-manifold` |
| Self-intersecting | `self-intersecting` |
| Multiple components | `fragmented` |

---

## Test Organization

```
tests/fixtures/thingi10k/
├── clean/              # Clean, printable
├── holes/              # Models with holes
├── non_manifold/       # Non-manifold
├── self_intersecting/  # Self-intersecting
├── fragmented/         # Multi-component
└── complex/            # Multiple issues
```

---

## Success Criteria

### Overall Targets

| Metric | Target |
|--------|--------|
| Overall Success | ≥ 80% |
| Repair Success | ≥ 95% |
| Geometric Validity | ≥ 90% |
| Visual Fidelity | ≥ 85% |

### Per-Category Targets

| Category | Target |
|----------|--------|
| `clean` | 100% |
| `holes` | 95% |
| `non_manifold` | 85% |
| `fragmented` | 80% |
| `self_intersecting` | 60% |
| `complex` | 50% |

---

## Running Benchmarks

```bash
# Run full benchmark
python scripts/run_benchmark.py \
  --fixtures tests/fixtures/thingi10k \
  --output benchmark_results.json

# Run specific categories
python scripts/run_benchmark.py \
  --fixtures tests/fixtures/thingi10k \
  --categories holes non_manifold
```

---

## Validation Definition

**Success = Geometric Validity + Visual Fidelity**

### Geometric Validity
- ✅ Watertight
- ✅ Manifold
- ✅ Positive volume
- ✅ No self-intersections

### Visual Fidelity
- ✅ Volume change < 1%
- ✅ Bbox unchanged
- ✅ Hausdorff < 0.1% of bbox diagonal

---

## See Also

- [Validation Guide](validation.md) - Validation criteria
- [Repair Pipeline](repair_pipeline.md) - Pipeline stages
- [Model Profiles](model_profiles.md) - Profile detection
