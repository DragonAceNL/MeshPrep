# MeshPrep POC v3 - Thingi10K Full Test Guide

## Overview

POC v3 is a **batch testing wrapper** around the POC v2 repair pipeline. It runs automatic repair against all ~10,000 models in the Thingi10K dataset, generating detailed reports with before/after comparisons.

**Supported formats:** STL, OBJ, PLY, OFF, 3MF, CTM

**Note:** All repair logic lives in POC v2. POC v3 only handles:
- Batch processing and progress tracking
- Report generation (markdown + images)
- Dashboard updates
- CSV export

## Architecture

```
POC v3 (run_full_test.py)
    │
    ├── Loads STL files from Thingi10K
    │
    └── Calls POC v2 for repair:
        │
        └── slicer_repair_loop.run_slicer_repair_loop()
            │
            ├── STRICT pre-check (skip if clean)
            ├── Iterative repair attempts
            └── Slicer validation after each repair
```

## NEW: STRICT Slicer Pre-Check

The POC v2 repair loop now includes a **STRICT pre-check** using PrusaSlicer's `--info` command **before** attempting any repair. This:

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

### 1. Start the Reports Server (Required)

The reports server serves everything from a single port - reports, dashboard, and MeshLab integration:

```powershell
cd "C:\Users\Dragon Ace\Source\repos\MeshPrep\poc\v3"
..\v2\.venv312\Scripts\python.exe reports_server.py
```

This starts a server with the following URLs:

| URL | Description |
|-----|-------------|
| http://localhost:8000/reports/ | Reports index (browse all results) |
| http://localhost:8000/live_dashboard.html | Live dashboard (auto-updates) |
| http://localhost:8000/dashboard | Static dashboard |
| http://localhost:8000/progress.json | Progress data (JSON) |

Features:
- Browse the reports index
- Download STL files
- **Open STL files directly in MeshLab** (click the purple "MeshLab" button)
- View live progress during test runs
- View before/after images

> **Note:** The server automatically finds MeshLab in common installation paths. If MeshLab is not found, the buttons will show a warning.

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
| `run_full_test.py --fresh` | Reprocess all files (doesn't delete existing results) |
| `run_full_test.py --limit 100` | Test only the first 100 models |
| `run_full_test.py --ctm-priority` | Process CTM meshes FIRST before other files |

## Features

- ✅ **Auto-resume**: Automatically skips models that already have reports. If interrupted (power outage, crash, Ctrl+C), just run the same command again.
- ✅ **Progress tracking**: Real-time progress with ETA displayed in dashboard
- ✅ **Reports**: HTML reports saved in `reports` subfolder with:
  - Before/after images
  - Mesh metrics (vertices, faces, volume)
  - Watertight/manifold status
  - **Direct links to open models in 3D viewer** (MeshLab, etc.)
- ✅ **Filter scripts**: JSON files recording exactly which filter actions were applied to each model
- ✅ **Fixed models**: Successfully repaired models saved to `Thingi10K\raw_meshes\fixed\`
- ✅ **Blender escalation**: Difficult models automatically escalated to Blender remesh
- ✅ **Decimation**: Large meshes from Blender are decimated to ~100k faces (if it doesn't break manifold status)

## Output Locations

All outputs are saved in a combined location under `Thingi10K/` (shared for both STL and CTM models):

| Output | Location |
|--------|----------|
| Reports (`.html`) | `C:\Users\Dragon Ace\Source\repos\Thingi10K\reports\` |
| Reports Index | `C:\Users\Dragon Ace\Source\repos\Thingi10K\reports\index.html` |
| Before/After Images | `C:\Users\Dragon Ace\Source\repos\Thingi10K\reports\images\` |
| Filter Scripts | `C:\Users\Dragon Ace\Source\repos\Thingi10K\reports\filters\` |
| Fixed Models | `C:\Users\Dragon Ace\Source\repos\Thingi10K\fixed\` |
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

## Repair Pipeline (via POC v2)

POC v3 calls `run_slicer_repair_loop()` from POC v2 which handles:

1. **STRICT Pre-check** - Run `prusa-slicer --info` to check if model is already clean
   - If CLEAN (manifold, no open edges, no reversed facets) → **Skip repair**, mark success
   - If HAS ISSUES → Continue with repair loop
2. **Iterative repair loop** - Try repair actions based on detected issues
   - Maps slicer issues to repair strategies (holes → fill_holes, non-manifold → pymeshfix, etc.)
   - Validates with slicer after each attempt
3. **Blender escalation** - After 5 attempts, escalate to Blender remesh
4. **Success validation** - Final STRICT slicer check confirms mesh is truly clean

POC v3 additionally handles:
- **Decimation** - Reduce large meshes from Blender to ~100k faces
- **Reporting** - Generate markdown reports with before/after images
- **Fixed model export** - Save successful repairs to `fixed/` directory

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
- Delete its `.html` report file from the `reports` folder and run the test again.
- Example: Delete `reports\100027.html` to reprocess model 100027.

## File Structure

```
poc/v3/
├── run_full_test.py      # Main test script
├── reports_server.py     # HTTP server with MeshLab integration
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

Each processed model gets an HTML report. View them at **http://localhost:8000/reports/** after starting the server.

Reports include:
- Status badge (Fixed, Failed, Already Clean, Blender escalation)
- Filter used and duration
- **Download buttons** for original and fixed STL files
- **MeshLab buttons** to open files directly in MeshLab (purple buttons)
- Side-by-side before/after images
- Metrics table (vertices, faces, volume, watertight/manifold status)
- MeshLab status indicator in the nav bar

**Note:** The MeshLab buttons require the `reports_server.py` to be running. The server will show the MeshLab path on startup.

## Example Filter Script

Each model gets a detailed filter script JSON with comprehensive data for analysis:

```json
{
  "model_id": "100027",
  "model_fingerprint": "MP:abc123def456",
  "original_filename": "100027.stl",
  "original_format": "stl",
  
  "filter_name": "blender-remesh",
  "success": true,
  "escalated_to_blender": true,
  
  "precheck": {
    "passed": false,
    "skipped": false,
    "mesh_info": {
      "manifold": false,
      "open_edges": 16,
      "is_clean": false,
      "issues": ["open_edges", "non_manifold"]
    }
  },
  
  "repair_attempts": {
    "total_attempts": 3,
    "total_duration_ms": 2500,
    "issues_found": ["open_edges", "non_manifold"],
    "issues_resolved": ["open_edges", "non_manifold"],
    "attempts": [
      {
        "attempt_number": 1,
        "pipeline_name": "targeted-holes",
        "actions": ["fill_holes", "fix_normals"],
        "success": false,
        "duration_ms": 150,
        "geometry_valid": true
      },
      {
        "attempt_number": 2,
        "pipeline_name": "pymeshfix",
        "actions": ["pymeshfix_repair"],
        "success": false,
        "duration_ms": 350
      },
      {
        "attempt_number": 3,
        "pipeline_name": "blender-remesh",
        "actions": ["blender_remesh", "fix_normals"],
        "success": true,
        "duration_ms": 2000
      }
    ]
  },
  
  "diagnostics": {
    "before": {
      "vertices": 5432,
      "faces": 10864,
      "is_watertight": false,
      "body_count": 3
    },
    "after": {
      "vertices": 12500,
      "faces": 25000,
      "is_watertight": true,
      "body_count": 1
    }
  },
  
  "timestamp": "2026-01-02T20:18:01.568852"
}
```

This detailed data enables:
- **Filter script optimization**: See which pipelines work for which issues
- **Model profile refinement**: Correlate mesh characteristics with repair outcomes
- **Performance analysis**: Identify slow pipelines and optimize order
- **Failure analysis**: Understand why certain repairs fail
