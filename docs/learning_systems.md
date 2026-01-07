# Learning Systems

## Overview

MeshPrep includes self-learning capabilities that improve repair success over time. All learning data is stored in `learning_data/` at the repository root.

---

## Components

### 1. Learning Engine (`learning_engine.py`)

Tracks pipeline success rates and optimizes ordering.

**What it tracks:**
- Pipeline success/failure counts
- Pipeline success by issue type (holes, non-manifold, etc.)
- Pipeline success by mesh characteristics (face count, body count)
- Issue pattern statistics (which issue combinations map to which pipelines)
- Profile statistics (success rates per model profile)

**How it helps:**
- Returns optimal pipeline order based on mesh characteristics
- Predicts success probability for a pipeline + issue combination
- Recommends pipelines that worked for similar models

---

### 2. Adaptive Thresholds (`adaptive_thresholds.py`)

Learns optimal parameter values from outcomes.

**Tracked thresholds:**

| Threshold | Default | Purpose |
|-----------|---------|---------|
| `volume_loss_limit_pct` | 30.0 | Max volume loss before flagging |
| `face_loss_limit_pct` | 40.0 | Max face loss before flagging |
| `decimation_trigger_faces` | 100,000 | Face count to trigger decimation |
| `body_count_extreme_fragmented` | 1,000 | Bodies for extreme fragmentation |
| `body_count_fragmented` | 10 | Bodies for fragmented profile |
| `max_repair_attempts` | 20 | Max attempts before giving up |
| `repair_timeout_seconds` | 120 | Timeout for repairs |

**Optimization:** Analyzes success rates above/below thresholds and adjusts by 10% if significant difference found.

---

### 3. Pipeline Evolution (`pipeline_evolution.py`)

Creates new pipeline combinations using genetic algorithm concepts.

**Operations:**
- **Selection** - Choose parent pipelines based on success rate
- **Crossover** - Combine actions from successful pipelines
- **Mutation** - Add/remove/swap/reorder actions

**Constraints:**
- Max 5 actions per pipeline
- Prep actions first (trimesh_basic, fix_normals)
- Expensive actions last (blender_remesh)

---

### 4. Profile Discovery (`profile_discovery.py`)

Automatically identifies new mesh profile categories.

**Clustering by:**
- Face count bucket (tiny/small/medium/large/huge)
- Body count bucket (1, 2-5, 6-20, 20+)
- Is watertight
- Has degenerate faces
- Issue signature

**Discovery process:**
1. Cluster meshes by characteristic key
2. Analyze clusters meeting size threshold (50+ models)
3. Create discovered profile if success rate is analyzable
4. Track performance over time
5. Promote to standard profile if consistently successful

---

### 5. Quality Feedback (`quality_feedback.py`)

Learns from both **automatic scoring** and **user ratings** to understand visual quality.

#### Auto-Quality Scoring (NEW)

Automatically computes a 1-5 quality score from geometric fidelity metrics:

| Metric | What It Measures | Score Impact |
|--------|------------------|--------------|
| **Volume Change** | Shape preservation | ±50% = -3, ±30% = -2, ±15% = -1 |
| **Hausdorff Distance** | Surface deviation | >10% = -2, >5% = -1.5, >2% = -1 |
| **Bounding Box Change** | Overall size | >10% = -1.5, >5% = -1 |
| **Surface Area Change** | Detail preservation | >50% = -1, >30% = -0.5 |
| **Printability** | Geometric validity | Bonus +0.5 or penalty -0.5 |

**Functions:**
- `compute_auto_quality_score()` - Convert metrics to 1-5 score
- `record_auto_quality_rating()` - Compute and record for learning
- `compute_quality_score_from_metrics()` - Score from pre-computed values

#### Manual Rating Scale

| Score | Meaning |
|-------|---------|
| 5 | Perfect - indistinguishable from original |
| 4 | Good - minor smoothing, fully usable |
| 3 | Acceptable - noticeable changes but recognizable |
| 2 | Poor - significant detail loss |
| 1 | Rejected - unrecognizable or destroyed |

#### What it learns:
- Pipeline quality scores per profile
- Threshold values that correlate with good ratings
- Quality prediction model
- **Penalizes pipelines that produce low-quality results**

---

## Auto-Quality Integration

The repair pipeline automatically computes and records quality scores:

```python
# In slicer_repair_loop.py
result = run_slicer_repair_loop(
    mesh,
    auto_quality_score=True,  # Default: enabled
    model_fingerprint="MP:abc123",
    model_filename="model.stl"
)

# After successful repair:
# - Computes fidelity metrics (volume, Hausdorff, bbox)
# - Calculates quality score (1-5)
# - Records to quality_feedback.db
# - Available in result.auto_quality_score
```

### Score Calculation Example

```
Original mesh: 10,000 faces, 50 cm³ volume
Repaired mesh: 9,500 faces, 49 cm³ volume

Volume change: -2% → No penalty
Face change: -5% → Minor (tracked)
Hausdorff: 0.3% of bbox → -0.25 penalty
Bbox change: 0% → No penalty
Printable: Yes → +0.5 bonus

Raw score: 5.0 - 0.25 + 0.5 = 5.25 → Final: 5
Interpretation: "Perfect - indistinguishable from original"
```

---

## Data Storage

| Database | Contents |
|----------|----------|
| `meshprep_learning.db` | Pipeline stats, profiles, model results |
| `pipeline_evolution.db` | Evolved pipelines, action stats |
| `profile_discovery.db` | Discovered profiles, clusters |
| `adaptive_thresholds.db` | Threshold values, observations |
| `quality_feedback.db` | **Auto + manual ratings**, quality predictions |

---

## CLI Commands

```bash
# Show learning statistics
python run_full_test.py --learning-stats

# Show adaptive thresholds
python run_full_test.py --threshold-stats

# Optimize thresholds
python run_full_test.py --optimize-thresholds

# Reset thresholds to defaults
python run_full_test.py --reset-thresholds

# Run profile discovery
python run_full_test.py --discover-profiles --min-samples 50

# Show quality feedback statistics
python run_full_test.py --quality-stats

# Manually rate a model (supplements auto-scoring)
python run_full_test.py --rate MP:abc123 --rating 5 --comment "Perfect"

# Disable auto-scoring for a batch run
python run_full_test.py --input-dir ./models/ --no-auto-quality
```

---

## Integration Flow

```
Process Model
    ↓
Run Repair Pipeline
    ↓
On Success:
    ├─► Compute Fidelity Metrics
    │       • Volume change
    │       • Hausdorff distance
    │       • Bbox change
    │       • Surface area change
    │
    ├─► Calculate Auto-Quality Score (1-5)
    │
    └─► Record to Quality Feedback DB
    ↓
Record Outcome → Learning Engine
                 Pipeline Evolution  
                 Profile Discovery
                 Adaptive Thresholds
                 Quality Feedback (with auto-score)
    ↓
Next Model Processing uses:
  • Optimal pipeline order (Learning Engine)
  • **Quality-weighted pipeline ranking** (Feedback)
  • Evolved pipelines for failures (Evolution)
  • Profile-specific strategies (Discovery)
  • Optimized thresholds (Adaptive)
  • Quality predictions (Feedback)
```

---

## Training on Thingi10K

With 10,000+ models available for training, the system can:

1. **Process all models** with auto-scoring enabled (default)
2. **Accumulate quality data** without manual intervention
3. **Learn which pipelines produce high-quality results**
4. **Penalize pipelines** that consistently score low
5. **Spot-check** by manually rating a sample (~1-2%)

```bash
# Full batch run with auto-scoring
python run_full_test.py --input-dir "C:\Thingi10K\raw_meshes"

# After run, view what was learned
python run_full_test.py --quality-stats
python run_full_test.py --learning-stats
```

---

## See Also

- [Repair Strategy Guide](repair_strategy_guide.md) - When to use each approach
- [Functional Spec](functional_spec.md) - Overview
- [Validation Guide](validation.md) - Validation criteria
