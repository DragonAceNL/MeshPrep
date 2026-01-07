# MeshPrep POC v4: ML-Based Mesh Repair

## Overview

POC v4 explores using **deep learning** (PyTorch + PyTorch3D) to learn optimal repair strategies from the 10,000+ repair outcomes collected in POC v3.

### Key Innovation

Instead of manually designed rules, the system **learns from data**:

```
Traditional (POC v2/v3):
  Mesh ? Hand-coded rules ? Try pipelines in fixed order

ML-Based (POC v4):
  Mesh ? Neural network ? Predict best pipeline + quality
```

---

## Architecture

### Model Components

| Component | Input | Output | Purpose |
|-----------|-------|--------|---------|
| **Mesh Encoder** | Point cloud (2048 points + normals) | Latent vector (256D) | Embed geometry into learned space |
| **Pipeline Selector** | Latent vector + statistics | Pipeline probabilities | Predict which pipeline to try first |
| **Quality Predictor** | Latent vector + pipeline ID | Quality score (1-5) + confidence | Predict repair outcome before execution |

### Network Architecture

```
Input Mesh (broken STL)
    ?
Sample 2048 surface points + normals
    ?
???????????????????????????????????????
? MeshEncoder (PointNet++ style)      ?
?  • Conv1d layers: 6?64?128?256      ?
?  • BatchNorm + ReLU + Dropout       ?
?  • Global max pooling               ?
?  • Output: 256D latent vector       ?
???????????????????????????????????????
    ?
  Latent Vector (geometry features)
    ?
    ???????????????????????????????????
    ?                                 ?
    ?                                 ?
????????????????????      ???????????????????????
? Pipeline Selector?      ? Quality Predictor   ?
?  • MLP: 266?128?50?      ?  • Pipeline embed   ?
?  • Softmax output ?      ?  • MLP: 288?128?2   ?
?  • Top-3 pipelines?      ?  • Output: Q + conf ?
????????????????????      ???????????????????????
```

---

## Training Data

Uses repair history from POC v3:

```python
# From learning_data/meshprep_learning.db
SELECT 
    model_id,           # Which mesh
    winning_pipeline,   # Which pipeline succeeded
    total_attempts,     # How many tries
    faces_before,       # Mesh complexity
    body_count          # Fragmentation

# From learning_data/quality_feedback.db  
SELECT
    model_fingerprint,
    rating_value        # Quality score (1-5)
    
# Combine ? Training dataset
# Input: Mesh geometry
# Output: Best pipeline + expected quality
```

**Expected training set:** ~8,000 meshes (80% of 10K)  
**Expected validation set:** ~2,000 meshes (20% of 10K)

---

## Installation

### Prerequisites

```bash
# Python 3.11-3.12
python --version

# CUDA toolkit (for GPU acceleration, optional but recommended)
nvidia-smi  # Check GPU availability
```

### Install Dependencies

```bash
cd poc/v4

# Option 1: Conda (recommended for PyTorch3D)
conda create -n meshprep-ml python=3.11
conda activate meshprep-ml
conda install pytorch torchvision pytorch3d -c pytorch -c pytorch3d
pip install -r requirements.txt

# Option 2: Pip (PyTorch3D build may take time)
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

---

## Usage

### 1. Prepare Training Data

Assumes you've already run POC v3 on Thingi10K:

```bash
# Check if training data exists
ls -lh ../v3/learning_data/meshprep_learning.db
ls -lh ../v3/learning_data/quality_feedback.db

# If not, run POC v3 first:
cd ../v3
python run_full_test.py --input-dir "C:\Thingi10K\raw_meshes" --limit 1000
```

### 2. Train the Model

```bash
cd poc/v4

python -m meshprep_ml.training.train \
    --data-dir "C:\Thingi10K\raw_meshes" \
    --learning-db "../v3/learning_data/meshprep_learning.db" \
    --epochs 50 \
    --batch-size 32 \
    --device cuda

# Training will save:
#  - checkpoints/best_model.pth
#  - logs/tensorboard/
#  - outputs/training_curve.png
```

### 3. Evaluate Performance

```bash
python -m meshprep_ml.training.evaluate \
    --model-path checkpoints/best_model.pth \
    --data-dir "C:\Thingi10K\raw_meshes" \
    --learning-db "../v3/learning_data/meshprep_learning.db"

# Outputs:
#  - Pipeline prediction accuracy (top-1, top-3)
#  - Quality prediction MAE (mean absolute error)
#  - Confusion matrix
```

### 4. Use for Inference

```python
from meshprep_ml import MeshRepairNet
import torch
import trimesh

# Load trained model
model = MeshRepairNet(num_pipelines=50)
model.load_state_dict(torch.load("checkpoints/best_model.pth"))
model.eval()

# Load a broken mesh
mesh = trimesh.load("broken_model.stl")

# Predict best repair strategy
from meshprep_ml.inference import predict_repair_strategy

result = predict_repair_strategy(model, mesh)
print(f"Best pipeline: {result['pipeline_name']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Expected quality: {result['expected_quality']:.1f}/5")
```

---

## Comparison with POC v2/v3

| Feature | POC v2 | POC v3 | POC v4 (ML) |
|---------|--------|--------|-------------|
| **Decision Making** | Hard-coded rules | Statistical learning | Deep learning |
| **Pipeline Selection** | Fixed order | Learned order from history | Predicted from geometry |
| **Quality Prediction** | None | Heuristic (attempts) | Neural network |
| **Training Data** | None | 10K repair outcomes | Same + mesh geometry |
| **Inference Speed** | Instant | Instant | ~50ms (GPU) |
| **Generalization** | Limited | Good | Best (learns features) |
| **Interpretability** | High | Medium | Low |

### When to Use Each

- **POC v2**: Production use, no training data available
- **POC v3**: After collecting repair history, want statistical optimization
- **POC v4**: After POC v3, want best possible predictions, have GPU

---

## Expected Results

### Pipeline Selection Accuracy

| Metric | Target | Notes |
|--------|--------|-------|
| **Top-1 Accuracy** | >40% | Predicts exact winning pipeline |
| **Top-3 Accuracy** | >70% | Winning pipeline in top 3 |
| **Top-5 Accuracy** | >85% | Winning pipeline in top 5 |

### Quality Prediction Error

| Metric | Target | Notes |
|--------|--------|-------|
| **MAE (Mean Absolute Error)** | <0.5 | Average error in predicted quality score |
| **RMSE** | <0.7 | Root mean squared error |
| **Correlation** | >0.8 | Predicted vs actual quality |

---

## Limitations

1. **Requires Training Data**: Needs POC v3 to run first (10K+ meshes)
2. **GPU Recommended**: CPU inference is slow (~500ms vs 50ms)
3. **Black Box**: Less interpretable than rule-based approaches
4. **Overfitting Risk**: May not generalize beyond Thingi10K characteristics

---

## Future Work

- **Transfer Learning**: Pre-train on ShapeNet, fine-tune on repair data
- **Graph Neural Networks**: Use mesh connectivity, not just point clouds
- **Multi-Modal**: Combine geometry + repair history for better predictions
- **Active Learning**: Prioritize uncertain predictions for manual review
- **Explainability**: Visualize which mesh features drive predictions

---

## Files

```
poc/v4/
??? meshprep_ml/
?   ??? __init__.py              # Package exports
?   ??? models/
?   ?   ??? __init__.py          # MeshEncoder, PipelineSelector, QualityPredictor
?   ??? data/
?   ?   ??? __init__.py          # MeshRepairDataset, dataloaders
?   ??? training/
?   ?   ??? train.py             # Training script
?   ?   ??? evaluate.py          # Evaluation script
?   ?   ??? utils.py             # Training utilities
?   ??? inference/
?       ??? __init__.py          # Inference functions
??? requirements.txt             # Dependencies
??? README.md                    # This file
??? docs/
    ??? architecture.md          # Detailed architecture
```

---

## References

- **PyTorch3D**: https://pytorch3d.org/
- **PointNet**: https://arxiv.org/abs/1612.00593
- **MeshCNN**: https://arxiv.org/abs/1809.05910
- **POC v3 Learning**: `../v3/docs/learning_systems.md`
