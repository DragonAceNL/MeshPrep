# MeshPrep POC v3 - Thingi10K Full Test Guide

## Overview

This POC runs automatic mesh repair against all ~10,000 models in the Thingi10K dataset, generating detailed reports with before/after comparisons.

## NEW: STRICT Slicer Pre-Check

POC v3 now includes a **STRICT pre-check** using PrusaSlicer's `--info` command **before** attempting any repair. This:

1. **Skips already-clean models** - No unnecessary repair that might break good models
2. **Detects exact issues** - manifold status, open edges, reversed facets
3. **Prevents repair damage** - Models that were good before won't be damaged by repair

```
Pre-check results on first 10 models:
  100028.stl: CLEAN (skip repair)
  100034.stl: CLEAN (skip repair)
  100026.stl: NEEDS REPAIR - open_edges (16)
  100030.stl: NEEDS REPAIR - open_edges (958)
  ...
```

## Quick Start

### 1. Start the Live Dashboard (Optional but Recommended)

Open a PowerShell terminal:
```powershell
cd "C:\Users\Dragon Ace\Source\repos\MeshPrep\poc\v3"
python -m http.server 8080
```

Then open in your browser: **http://localhost:8080/live_dashboard.html**

### 2. Start the Full Test

Open **another** PowerShell terminal:
```powershell
cd "C:\Users\Dragon Ace\Source\repos\MeshPrep\poc\v3"
..\v2\.venv312\Scripts\python.exe run_full_test.py
```

That's it! The test will run automatically.

## Commands

| Command | Description |
|---------|-------------|
| `run_full_test.py` | Run all models (auto-resumes from where it left off) |
| `run_full_test.py --status` | Check current progress without processing |
| `run_full_test.py --fresh` | Start over from scratch (clears all reports) |
| `run_full_test.py --limit 100` | Test only the first 100 models |

## Features

- ✅ **Auto-resume**: Automatically skips models that already have reports. If interrupted (power outage, crash, Ctrl+C), just run the same command again.
- ✅ **Progress tracking**: Real-time progress with ETA displayed in dashboard
- ✅ **Reports**: Markdown reports saved in `reports` subfolder with:
  - Before/after images
  - Mesh metrics (vertices, faces, volume)
  - Watertight/manifold status
  - Links to download original and fixed models
- ✅ **Filter scripts**: JSON files recording exactly which filter actions were applied to each model
- ✅ **Fixed models**: Successfully repaired models saved to `Thingi10K\raw_meshes\fixed\`
- ✅ **Blender escalation**: Difficult models automatically escalated to Blender remesh
- ✅ **Decimation**: Large meshes from Blender are decimated to ~100k faces (if it doesn't break manifold status)

## Output Locations

| Output | Location |
|--------|----------|
| Reports (`.md`) | `C:\Users\Dragon Ace\Source\repos\Thingi10K\raw_meshes\reports\` |
| Before/After Images | `C:\Users\Dragon Ace\Source\repos\Thingi10K\raw_meshes\reports\images\` |
| Filter Scripts | `C:\Users\Dragon Ace\Source\repos\Thingi10K\raw_meshes\reports\filters\` |
| Fixed Models | `C:\Users\Dragon Ace\Source\repos\Thingi10K\raw_meshes\fixed\` |
| **Results CSV** | `poc\v3\results.csv` - All results in one file for analysis |
| Progress File | `poc\v3\progress.json` |
| Dashboard | `poc\v3\dashboard.html` (static) or `live_dashboard.html` (live) |
| Logs | `poc\v3\logs\` |

## Expected Duration

Based on test results:

| Scenario | Avg per Model | Total (~10,000 models) |
|----------|---------------|------------------------|
| Simple repairs only | ~1-5 seconds | ~8-14 hours |
| With Blender escalations (~25%) | ~16+ seconds | ~45-100+ hours |

**Note**: Blender escalations can take 30 seconds to 8+ minutes each depending on model complexity.

## Repair Pipeline

1. **STRICT Pre-check** (NEW) - Run `prusa-slicer --info` to check if model is already clean
   - If CLEAN (manifold, no open edges, no reversed facets) → **Skip repair**, mark success
   - If HAS ISSUES → Continue with repair
2. **Load mesh** and compute diagnostics
3. **Run conservative-repair** filter (trimesh + pymeshfix)
4. **Check results**:
   - If watertight & manifold → Success
   - If geometry loss > 30% or not printable → Escalate to Blender
5. **Blender remesh** (if needed) - voxel remesh at 0.05 size
6. **Decimate** if > 100k faces (keeps original if decimation breaks manifold)
7. **Save** fixed model and generate report

## Troubleshooting

### Test is taking too long
- Blender escalations are slow. This is expected for complex/broken meshes.
- You can check progress with `--status` or the live dashboard.

### Dashboard not updating
- The static `dashboard.html` only updates every 10 files processed.
- Use the live dashboard (`python -m http.server 8080`) for real-time updates.

### Unicode/emoji issues
- Make sure your terminal supports UTF-8.
- The live dashboard includes `<meta charset="UTF-8">` for proper display.

### Out of disk space
- Fixed models can be large (especially Blender output that couldn't be decimated).
- Check `Thingi10K\raw_meshes\fixed\` for large files.

### Want to reprocess a specific model
- Delete its `.md` report file from the `reports` folder and run the test again.
- Example: Delete `reports\100027.md` to reprocess model 100027.

## File Structure

```
poc/v3/
├── run_full_test.py      # Main test script
├── progress.json         # Current progress state
├── summary.json          # Final summary with results
├── dashboard.html        # Static dashboard (updated during run)
├── live_dashboard.html   # Live dashboard (reads progress.json)
├── start_dashboard.bat   # Quick start for dashboard server
├── README.md             # This guide
└── logs/                 # Log files for each run
    └── run_YYYYMMDD_HHMMSS.log
```

## Example Report

Each processed model gets a report like this:

```markdown
# 100027

**Status:** SUCCESS (Blender)
**Filter:** `blender-remesh`
**Duration:** 51253ms

## Download Models
| Model | Link |
|-------|------|
| **Original** | [100027.stl](../100027.stl) |
| **Fixed** | [100027.stl](../fixed/100027.stl) |
| **Filter Script** | [100027.json](./filters/100027.json) |

## Metrics
| Metric | Before | After |
|--------|--------|-------|
| Watertight | No | Yes |
| Manifold | No | Yes |
| Faces | 508 | 4,979,000 |
```

## Example Filter Script

Each model also gets a filter script JSON:

```json
{
  "model_id": "100027",
  "filter_name": "blender-remesh",
  "filter_version": "1.0.0",
  "escalated_to_blender": true,
  "actions": [
    { "name": "trimesh_basic", "params": {} },
    { "name": "blender_remesh", "params": { "voxel_size": 0.05, "mode": "VOXEL" } },
    { "name": "fix_normals", "params": {} },
    { "name": "validate", "params": {} }
  ],
  "timestamp": "2026-01-02T20:18:01.568852"
}
