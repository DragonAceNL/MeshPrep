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

### 5. Quality Feedback (Visual)

Learns from user ratings to understand visual quality.

**Rating scale:**
| Score | Meaning |
|-------|---------|
| 5 | Perfect - indistinguishable from original |
| 4 | Good - minor smoothing, fully usable |
| 3 | Acceptable - noticeable changes but recognizable |
| 2 | Poor - significant detail loss |
| 1 | Rejected - unrecognizable or destroyed |

**What it learns:**
- Pipeline quality scores per profile
- Threshold values that correlate with good ratings
- Quality prediction model

---

## Data Storage

| Database | Contents |
|----------|----------|
| `meshprep_learning.db` | Pipeline stats, profiles, model results |
| `pipeline_evolution.db` | Evolved pipelines, action stats |
| `profile_discovery.db` | Discovered profiles, clusters |
| `adaptive_thresholds.db` | Threshold values, observations |
| `quality_feedback.db` | User ratings, quality predictions |

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
```

---

## Integration Flow

```
Process Model
    ↓
Record Outcome → Learning Engine
                 Pipeline Evolution  
                 Profile Discovery
                 Adaptive Thresholds
                 Quality Feedback
    ↓
Next Model Processing uses:
  • Optimal pipeline order (Learning Engine)
  • Evolved pipelines for failures (Evolution)
  • Profile-specific strategies (Discovery)
  • Optimized thresholds (Adaptive)
  • Quality predictions (Feedback)
```

---

## See Also

- [Repair Strategy Guide](repair_strategy_guide.md) - When to use each approach
- [Functional Spec](functional_spec.md) - Overview
