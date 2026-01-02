# Thingi10K Data Directory

This directory contains the SQLite database for Thingi10K model metadata.

## Files

- `thingi10k.db` - SQLite database with model metadata and test results

## Setup

To populate the database, run:

```bash
# Option 1: Import from thingi10k Python package (recommended)
pip install thingi10k
python -m meshprep.testing.thingi10k_manager import --source package

# Option 2: Import by scanning STL files with trimesh
python -m meshprep.testing.thingi10k_manager import --source directory --raw-meshes "C:\path\to\Thingi10K\raw_meshes"
```

## Database Schema

### Tables

- `models` - Model metadata (10,000 rows)
- `test_results` - Benchmark test results

### Views

- `category_summary` - Summary statistics by category

## Querying

```python
from meshprep.testing import Thingi10KDatabase

db = Thingi10KDatabase()

# Get models by category
holes_models = db.get_models_by_category("holes", limit=20)

# Query by defect flags
non_manifold = db.get_models_by_defect(has_non_manifold=True, limit=50)

# Get summary
summary = db.get_category_summary()
```
