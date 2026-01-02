# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
POC v3: Full Thingi10K Batch Processing

Runs automatic repair against all 10,000 models in Thingi10K dataset.

Features:
- Progress tracking with ETA and percentage
- Reports saved alongside original STL files
- Easy navigation between results (prev/next links)
- Before/after images inline in reports
- Summary dashboard with statistics
- Auto-resume: automatically skips files that already have reports

Usage:
    python run_full_test.py                    # Run all models (auto-resumes)
    python run_full_test.py --limit 100        # Test first 100 models
    python run_full_test.py --status           # Show current progress
    python run_full_test.py --fresh            # Start fresh (ignore existing reports)
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
import traceback

# Setup logging with file output
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ],
    force=True
)
logger = logging.getLogger(__name__)

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Add POC v2 to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "v2"))

from meshprep_poc.mesh_ops import load_mesh, save_mesh, compute_diagnostics
from meshprep_poc.validation import validate_repair
from meshprep_poc.filter_script import (
    FilterScriptRunner,
    FilterScript,
    get_preset,
)
from meshprep_poc.actions.blender_actions import is_blender_available

# Paths
THINGI10K_PATH = Path(r"C:\Users\Dragon Ace\Source\repos\Thingi10K\raw_meshes")
REPORTS_PATH = Path(r"C:\Users\Dragon Ace\Source\repos\Thingi10K\raw_meshes\reports")
FILTERS_PATH = Path(r"C:\Users\Dragon Ace\Source\repos\Thingi10K\raw_meshes\reports\filters")
FIXED_OUTPUT_PATH = Path(r"C:\Users\Dragon Ace\Source\repos\Thingi10K\raw_meshes\fixed")
PROGRESS_FILE = Path(__file__).parent / "progress.json"
SUMMARY_FILE = Path(__file__).parent / "summary.json"
DASHBOARD_FILE = Path(__file__).parent / "dashboard.html"

# Ensure output directories exist
REPORTS_PATH.mkdir(parents=True, exist_ok=True)
FILTERS_PATH.mkdir(parents=True, exist_ok=True)
FIXED_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


@dataclass
class TestResult:
    """Result of a single model test."""
    file_id: str
    file_path: str
    success: bool = False
    error: Optional[str] = None
    filter_used: str = ""
    escalation_used: bool = False
    duration_ms: float = 0
    
    # Original metrics
    original_vertices: int = 0
    original_faces: int = 0
    original_volume: float = 0
    original_watertight: bool = False
    original_manifold: bool = False
    
    # Result metrics
    result_vertices: int = 0
    result_faces: int = 0
    result_volume: float = 0
    result_watertight: bool = False
    result_manifold: bool = False
    
    # Geometry change
    volume_change_pct: float = 0
    face_change_pct: float = 0
    
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Progress:
    """Track overall progress."""
    total_files: int = 0
    processed: int = 0
    successful: int = 0
    failed: int = 0
    escalations: int = 0
    skipped: int = 0
    
    start_time: str = ""
    last_update: str = ""
    current_file: str = ""
    
    # Timing
    total_duration_ms: float = 0
    avg_duration_ms: float = 0
    
    # ETA
    eta_seconds: float = 0
    
    @property
    def percent_complete(self) -> float:
        if self.total_files == 0:
            return 0
        return (self.processed / self.total_files) * 100
    
    @property
    def success_rate(self) -> float:
        if self.processed == 0:
            return 0
        return (self.successful / self.processed) * 100


def load_progress() -> Progress:
    """Load progress from file."""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE) as f:
                data = json.load(f)
                return Progress(**data)
        except Exception:
            pass
    return Progress()


def save_progress(progress: Progress):
    """Save progress to file."""
    progress.last_update = datetime.now().isoformat()
    with open(PROGRESS_FILE, "w") as f:
        json.dump(asdict(progress), f, indent=2)


def get_all_stl_files(limit: Optional[int] = None) -> List[Path]:
    """Get all STL files from Thingi10K."""
    if not THINGI10K_PATH.exists():
        raise FileNotFoundError(f"Thingi10K path not found: {THINGI10K_PATH}")
    
    stl_files = sorted(THINGI10K_PATH.glob("*.stl"))
    
    if limit:
        stl_files = stl_files[:limit]
    
    return stl_files


def get_best_filter(mesh) -> FilterScript:
    """Select the best filter based on mesh analysis."""
    try:
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            logger.debug(f"Multi-component model ({len(components)} parts) - using conservative")
            return get_preset("conservative-repair")
    except Exception:
        pass
    
    return get_preset("conservative-repair")  # Safe default


def check_geometry_loss(original_diag, result_mesh) -> tuple[bool, float, float]:
    """Check if repair caused significant geometry loss."""
    import numpy as np
    
    original_volume = original_diag.volume if original_diag.volume > 0 else 0
    result_volume = result_mesh.volume if result_mesh.is_volume else 0
    
    volume_loss_pct = 0
    if original_volume > 0:
        volume_loss_pct = abs(original_volume - result_volume) / original_volume * 100
    
    original_faces = original_diag.face_count
    result_faces = len(result_mesh.faces)
    face_loss_pct = 0
    if original_faces > 0:
        face_loss_pct = (original_faces - result_faces) / original_faces * 100
    
    significant_loss = volume_loss_pct > 30 or face_loss_pct > 40
    
    return significant_loss, volume_loss_pct, face_loss_pct


def decimate_mesh(mesh, target_faces: int = 100000):
    """Decimate mesh to reduce face count while preserving shape."""
    if len(mesh.faces) <= target_faces:
        return mesh
    
    try:
        # Try fast_simplification first (best quality)
        import fast_simplification
        
        # Get vertices and faces as numpy arrays
        vertices = mesh.vertices.copy()
        faces = mesh.faces.copy()
        
        # Calculate target ratio
        ratio = target_faces / len(faces)
        
        # Simplify
        new_verts, new_faces = fast_simplification.simplify(
            vertices, faces, 
            target_reduction=1.0 - ratio,
            agg=5  # Aggression level (1-10)
        )
        
        # Create new mesh
        import trimesh
        decimated = trimesh.Trimesh(vertices=new_verts, faces=new_faces)
        
        if len(decimated.faces) > 0:
            logger.info(f"  Decimated: {len(mesh.faces):,} -> {len(decimated.faces):,} faces")
            return decimated
            
    except ImportError:
        logger.warning("  fast_simplification not installed, trying trimesh method")
    except Exception as e:
        logger.warning(f"  fast_simplification failed: {e}")
    
    # Fallback to trimesh's built-in method
    try:
        decimated = mesh.simplify_quadric_decimation(target_faces)
        if decimated is not None and len(decimated.faces) > 0:
            logger.info(f"  Decimated (trimesh): {len(mesh.faces):,} -> {len(decimated.faces):,} faces")
            return decimated
    except Exception as e:
        logger.warning(f"  Trimesh decimation failed: {e}")
    
    return mesh


def save_filter_script(file_id: str, filter_script: FilterScript, escalated: bool = False):
    """Save the filter script used for a model."""
    filter_path = FILTERS_PATH / f"{file_id}.json"
    
    # Build filter data
    filter_data = {
        "model_id": file_id,
        "filter_name": filter_script.name,
        "filter_version": getattr(filter_script, 'version', '1.0.0'),
        "escalated_to_blender": escalated,
        "actions": [
            {
                "name": action.name,
                "params": action.params
            }
            for action in filter_script.actions
        ],
        "timestamp": datetime.now().isoformat()
    }
    
    with open(filter_path, "w", encoding="utf-8") as f:
        json.dump(filter_data, f, indent=2)


def process_single_model(stl_path: Path) -> TestResult:
    """Process a single model and return result."""
    file_id = stl_path.stem
    start_time = time.time()
    
    result = TestResult(
        file_id=file_id,
        file_path=str(stl_path),
    )
    
    try:
        # Load mesh
        original = load_mesh(stl_path)
        original_diag = compute_diagnostics(original)
        
        result.original_vertices = original_diag.vertex_count
        result.original_faces = original_diag.face_count
        result.original_volume = original_diag.volume
        result.original_watertight = original_diag.is_watertight
        result.original_manifold = original_diag.is_volume
        
        # Select filter
        filter_script = get_best_filter(original)
        result.filter_used = filter_script.name
        
        # Run repair
        runner = FilterScriptRunner(stop_on_error=False)
        repair_result = runner.run(filter_script, original)
        
        if not repair_result.success or repair_result.final_mesh is None:
            result.success = False
            result.error = repair_result.error or "Repair failed"
            result.duration_ms = (time.time() - start_time) * 1000
            return result
        
        repaired = repair_result.final_mesh
        
        # Check for geometry loss
        significant_loss, vol_loss, face_loss = check_geometry_loss(original_diag, repaired)
        result.volume_change_pct = vol_loss
        result.face_change_pct = face_loss
        
        # Validate
        validation = validate_repair(original, repaired)
        
        # Check if escalation needed
        needs_escalation = (
            not validation.geometric.is_printable or
            significant_loss
        )
        
        if needs_escalation and is_blender_available():
            logger.info(f"  Escalating to Blender...")
            escalation_script = get_preset("blender-remesh")
            escalation_result = runner.run(escalation_script, original)
            
            if escalation_result.success and escalation_result.final_mesh is not None:
                repaired = escalation_result.final_mesh
                filter_script = escalation_script  # Update to use escalation script
                result.filter_used = "blender-remesh"
                result.escalation_used = True
                validation = validate_repair(original, repaired)
        
        # Capture printability status BEFORE decimation
        is_printable_before_decimate = validation.geometric.is_printable
        
        # Decimate if mesh is too large
        if len(repaired.faces) > 100000:
            original_repaired = repaired.copy()
            repaired = decimate_mesh(repaired, target_faces=100000)
            
            # Check if decimation broke manifold - if so, keep original large mesh
            if is_printable_before_decimate and not repaired.is_watertight:
                logger.warning("  Decimation broke manifold status - keeping original large mesh")
                repaired = original_repaired
        
        # Final metrics (after potential decimation)
        result.result_vertices = len(repaired.vertices)
        result.result_faces = len(repaired.faces)
        result.result_volume = repaired.volume if repaired.is_volume else 0
        result.result_watertight = repaired.is_watertight
        result.result_manifold = repaired.is_volume
        
        # Success based on FINAL mesh state (not pre-decimation)
        result.success = repaired.is_watertight and repaired.is_volume
        result.duration_ms = (time.time() - start_time) * 1000
        
        # Save repaired model to fixed directory
        fixed_path = FIXED_OUTPUT_PATH / f"{file_id}.stl"
        if result.success:
            save_mesh(repaired, fixed_path)
            logger.info(f"  Saved fixed model to {fixed_path}")
        
        # Save filter script used
        save_filter_script(file_id, filter_script, result.escalation_used)
        
        # Generate report
        generate_report(stl_path, original, repaired, result, fixed_path if result.success else None)
        
        return result
        
    except Exception as e:
        result.success = False
        result.error = f"{type(e).__name__}: {str(e)}"
        result.duration_ms = (time.time() - start_time) * 1000
        logger.error(f"  Error: {result.error}")
        return result


def render_mesh_image(mesh, output_path: Path, title: str = ""):
    """Render mesh to image using matplotlib."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import numpy as np
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get mesh data
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Limit faces for performance
        max_faces = 50000
        if len(faces) > max_faces:
            indices = np.random.choice(len(faces), max_faces, replace=False)
            faces = faces[indices]
        
        # Create polygon collection
        mesh_faces = vertices[faces]
        collection = Poly3DCollection(mesh_faces, alpha=0.8, linewidth=0.1, edgecolor='gray')
        collection.set_facecolor([0.3, 0.6, 0.9])
        ax.add_collection3d(collection)
        
        # Set axis limits
        scale = vertices.max() - vertices.min()
        center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
        ax.set_xlim(center[0] - scale/2, center[0] + scale/2)
        ax.set_ylim(center[1] - scale/2, center[1] + scale/2)
        ax.set_zlim(center[2] - scale/2, center[2] + scale/2)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        if title:
            ax.set_title(title)
        
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        return True
    except Exception as e:
        logger.warning(f"Failed to render image: {e}")
        return False


def generate_report(stl_path: Path, original, repaired, result: TestResult, fixed_path: Optional[Path] = None):
    """Generate markdown report in the reports subfolder."""
    report_path = REPORTS_PATH / f"{stl_path.stem}.md"
    images_dir = REPORTS_PATH / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Render images
    before_img = images_dir / f"{stl_path.stem}_before.png"
    after_img = images_dir / f"{stl_path.stem}_after.png"
    
    render_mesh_image(original, before_img, "Before")
    render_mesh_image(repaired, after_img, "After")
    
    # Get adjacent files for navigation
    all_files = sorted(THINGI10K_PATH.glob("*.stl"))
    current_idx = next((i for i, f in enumerate(all_files) if f.name == stl_path.name), -1)
    
    prev_file = all_files[current_idx - 1] if current_idx > 0 else None
    next_file = all_files[current_idx + 1] if current_idx < len(all_files) - 1 else None
    
    # Status
    status = "SUCCESS" if result.success else "FAILED"
    status_icon = "v" if result.success else "x"
    if result.escalation_used:
        status += " (Blender)"
    
    # Generate markdown
    md_content = f"""# {stl_path.stem}

**Status:** {status}  
**Filter:** `{result.filter_used}`  
**Duration:** {result.duration_ms:.0f}ms  
{"**Escalation:** Yes (Blender)  " if result.escalation_used else ""}

---

## Navigation

"""
    
    if prev_file:
        md_content += f"[< Previous: {prev_file.stem}](./{prev_file.stem}.md) | "
    else:
        md_content += "< Previous | "
    
    md_content += "[Dashboard](../../../MeshPrep/poc/v3/dashboard.html) | "
    
    if next_file:
        md_content += f"[Next: {next_file.stem} >](./{next_file.stem}.md)"
    else:
        md_content += "Next >"
    
    md_content += f"""

---

## Download Models

| Model | Link |
|-------|------|
| **Original** | [{stl_path.name}](../{stl_path.name}) |
"""
    
    if fixed_path and fixed_path.exists():
        md_content += f"""| **Fixed** | [{fixed_path.name}](../fixed/{fixed_path.name}) |
"""
    else:
        md_content += """| **Fixed** | Not saved (repair failed) |
"""
    
    md_content += f"""| **Filter Script** | [{stl_path.stem}.json](./filters/{stl_path.stem}.json) |

---

## Visual Comparison

| Before | After |
|--------|-------|
| ![Before](./images/{stl_path.stem}_before.png) | ![After](./images/{stl_path.stem}_after.png) |

---

## Metrics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Vertices | {result.original_vertices:,} | {result.result_vertices:,} | {result.result_vertices - result.original_vertices:+,} |
| Faces | {result.original_faces:,} | {result.result_faces:,} | {result.result_faces - result.original_faces:+,} |
| Volume | {result.original_volume:.2f} | {result.result_volume:.2f} | {result.volume_change_pct:+.1f}% |
| Watertight | {"Yes" if result.original_watertight else "No"} | {"Yes" if result.result_watertight else "No"} | - |
| Manifold | {"Yes" if result.original_manifold else "No"} | {"Yes" if result.result_manifold else "No"} | - |

"""

    if result.error:
        md_content += f"""
## Error

```
{result.error}
```
"""

    md_content += f"""
---

*Generated: {result.timestamp}*
"""
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)


def generate_dashboard(progress: Progress, results: List[TestResult]):
    """Generate HTML dashboard for easy overview."""
    
    # Calculate statistics
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    escalated = [r for r in results if r.escalation_used]
    
    avg_duration = sum(r.duration_ms for r in results) / len(results) if results else 0
    
    # Recent results (last 20)
    recent = results[-20:] if len(results) > 20 else results
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>MeshPrep Thingi10K Test Dashboard</title>
    <meta http-equiv="refresh" content="30">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f1720;
            color: #dff6fb;
            margin: 0;
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #4fe8c4; margin-bottom: 10px; }}
        .subtitle {{ color: #888; margin-bottom: 30px; }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: #1b2b33;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 36px;
            font-weight: bold;
            color: #4fe8c4;
        }}
        .stat-label {{ color: #888; margin-top: 5px; }}
        
        .progress-bar {{
            background: #1b2b33;
            border-radius: 10px;
            height: 30px;
            overflow: hidden;
            margin-bottom: 30px;
        }}
        .progress-fill {{
            background: linear-gradient(90deg, #4fe8c4, #2ecc71);
            height: 100%;
            transition: width 0.5s;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }}
        
        .current-file {{
            background: #1b2b33;
            border-radius: 12px;
            padding: 15px 20px;
            margin-bottom: 30px;
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        .spinner {{
            width: 20px;
            height: 20px;
            border: 3px solid #333;
            border-top-color: #4fe8c4;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            background: #1b2b33;
            border-radius: 12px;
            overflow: hidden;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #2a3a43;
        }}
        th {{ background: #0f1720; color: #4fe8c4; }}
        tr:hover {{ background: #2a3a43; }}
        
        .success {{ color: #2ecc71; }}
        .failed {{ color: #e74c3c; }}
        .escalated {{ color: #f39c12; }}
        
        a {{ color: #4fe8c4; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        
        .eta {{ font-size: 14px; color: #888; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔧 MeshPrep Thingi10K Test Dashboard</h1>
        <p class="subtitle">
            Started: {progress.start_time[:19] if progress.start_time else 'Not started'} | 
            Last Update: {progress.last_update[:19] if progress.last_update else 'Never'}
            <span class="eta">| ETA: {str(timedelta(seconds=int(progress.eta_seconds))) if progress.eta_seconds > 0 else 'Calculating...'}</span>
        </p>
        
        <div class="progress-bar">
            <div class="progress-fill" style="width: {progress.percent_complete:.1f}%">
                {progress.percent_complete:.1f}% ({progress.processed:,} / {progress.total_files:,})
            </div>
        </div>
        
        <div class="current-file">
            <div class="spinner"></div>
            <span>Currently processing: <strong>{progress.current_file or 'Waiting...'}</strong></span>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{progress.total_files:,}</div>
                <div class="stat-label">Total Models</div>
            </div>
            <div class="stat-card">
                <div class="stat-value success">{progress.successful:,}</div>
                <div class="stat-label">Successful</div>
            </div>
            <div class="stat-card">
                <div class="stat-value failed">{progress.failed:,}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value escalated">{progress.escalations:,}</div>
                <div class="stat-label">Escalations</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{progress.success_rate:.1f}%</div>
                <div class="stat-label">Success Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg_duration/1000:.1f}s</div>
                <div class="stat-label">Avg Duration</div>
            </div>
        </div>
        
        <h2>Recent Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Status</th>
                    <th>Filter</th>
                    <th>Duration</th>
                    <th>Faces</th>
                    <th>Report</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for r in reversed(recent):
        status_class = "success" if r.success else "failed"
        status_text = "✅" if r.success else "❌"
        if r.escalation_used:
            status_text += " 🚀"
        
        report_link = f"{REPORTS_PATH}/{r.file_id}.md"
        
        html += f"""
                <tr>
                    <td>{r.file_id}</td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{r.filter_used}</td>
                    <td>{r.duration_ms/1000:.1f}s</td>
                    <td>{r.original_faces:,} → {r.result_faces:,}</td>
                    <td><a href="file:///{report_link}">View</a></td>
                </tr>
"""
    
    html += """
            </tbody>
        </table>
        
        <p style="margin-top: 30px; color: #666; font-size: 12px;">
            Dashboard auto-refreshes every 30 seconds. 
            <a href="javascript:location.reload()">Refresh now</a>
        </p>
    </div>
</body>
</html>
"""
    
    with open(DASHBOARD_FILE, "w", encoding="utf-8") as f:
        f.write(html)


def get_processed_files() -> set:
    """Get set of already processed file IDs."""
    processed = set()
    for md_file in REPORTS_PATH.glob("*.md"):
        processed.add(md_file.stem)
    return processed


def run_batch_test(limit: Optional[int] = None, resume: bool = True):
    """Run the full batch test.
    
    Args:
        limit: Optional limit on number of files to process
        resume: If True (default), skip files that already have reports
    """
    print("="* 60, flush=True)
    print("MeshPrep Thingi10K Full Test - POC v3", flush=True)
    print("=" * 60, flush=True)
    
    logger.info("=" * 60)
    logger.info("MeshPrep Thingi10K Full Test - POC v3")
    logger.info("=" * 60)
    
    # Check Blender
    if is_blender_available():
        print("[OK] Blender available for escalation", flush=True)
        logger.info("[OK] Blender available for escalation")
    else:
        print("[WARN] Blender not available - some models may fail", flush=True)
        logger.warning("[WARN] Blender not available - some models may fail")
    
    # Get all files
    all_files = get_all_stl_files(limit)
    print(f"Found {len(all_files):,} STL files", flush=True)
    logger.info(f"Found {len(all_files):,} STL files")
    
    # Always check for already processed files (auto-resume)
    processed_ids = get_processed_files()
    if processed_ids:
        print(f"Found {len(processed_ids):,} existing reports - will skip those", flush=True)
        logger.info(f"Found {len(processed_ids):,} existing reports - will skip those")
    
    # Count how many we'll actually process
    to_process = [f for f in all_files if f.stem not in processed_ids]
    print(f"Will process {len(to_process):,} new files", flush=True)
    logger.info(f"Will process {len(to_process):,} new files")
    
    if len(to_process) == 0:
        print("\nAll files already processed! Use --fresh to reprocess.", flush=True)
        logger.info("All files already processed! Use --fresh to reprocess.")
        return
    
    # Initialize progress
    progress = Progress(
        total_files=len(all_files),
        start_time=datetime.now().isoformat(),
    )
    
    results: List[TestResult] = []
    
    # Process each file
    for i, stl_path in enumerate(all_files):
        file_id = stl_path.stem
        
        # Skip if already processed (resume mode)
        if file_id in processed_ids:
            progress.skipped += 1
            progress.processed += 1
            continue
        
        # Update progress
        progress.current_file = file_id
        progress.processed = i + 1
        
        # Calculate ETA
        if len(results) > 0 and progress.avg_duration_ms > 0:
            remaining = len(to_process) - len(results)
            progress.eta_seconds = (remaining * progress.avg_duration_ms) / 1000
        
        # Log progress
        eta_str = str(timedelta(seconds=int(progress.eta_seconds))) if progress.eta_seconds > 0 else "calculating..."
        print(f"[{len(results)+1}/{len(to_process)}] Processing {file_id}... (ETA: {eta_str})", flush=True)
        logger.info(f"[{i+1}/{len(all_files)}] Processing {file_id}...")
        
        # Process
        result = process_single_model(stl_path)
        results.append(result)
        
        # Update stats
        if result.success:
            progress.successful += 1
        else:
            progress.failed += 1
        
        if result.escalation_used:
            progress.escalations += 1
        
        progress.total_duration_ms += result.duration_ms
        progress.avg_duration_ms = progress.total_duration_ms / len(results)
        
        # Save progress and dashboard every 10 files
        if i % 10 == 0:
            save_progress(progress)
            generate_dashboard(progress, results)
        
        # Log result
        status = "[OK]" if result.success else "[FAIL]"
        escalation = " [BLENDER]" if result.escalation_used else ""
        print(f"  {status}{escalation} {result.filter_used} ({result.duration_ms:.0f}ms)", flush=True)
        logger.info(f"  {status}{escalation} {result.filter_used} ({result.duration_ms:.0f}ms)")
    
    # Final save
    save_progress(progress)
    generate_dashboard(progress, results)
    
    # Summary
    print("", flush=True)
    print("=" * 60, flush=True)
    print("FINAL SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Total processed: {progress.processed:,}", flush=True)
    print(f"Successful: {progress.successful:,} ({progress.success_rate:.1f}%)", flush=True)
    print(f"Failed: {progress.failed:,}", flush=True)
    print(f"Escalations: {progress.escalations:,}", flush=True)
    print(f"Total time: {progress.total_duration_ms/1000/60:.1f} minutes", flush=True)
    print(f"Avg per model: {progress.avg_duration_ms/1000:.1f}s", flush=True)
    print("", flush=True)
    print(f"Dashboard: {DASHBOARD_FILE}", flush=True)
    print("=" * 60, flush=True)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total processed: {progress.processed:,}")
    logger.info(f"Successful: {progress.successful:,} ({progress.success_rate:.1f}%)")
    logger.info(f"Failed: {progress.failed:,}")
    logger.info(f"Escalations: {progress.escalations:,}")
    logger.info(f"Total time: {progress.total_duration_ms/1000/60:.1f} minutes")
    logger.info(f"Avg per model: {progress.avg_duration_ms/1000:.1f}s")
    logger.info("")
    logger.info(f"Dashboard: {DASHBOARD_FILE}")
    logger.info("=" * 60)
    
    # Save final summary
    with open(SUMMARY_FILE, "w") as f:
        json.dump({
            "progress": asdict(progress),
            "results": [asdict(r) for r in results[-100:]],  # Last 100 results
        }, f, indent=2)


def show_status():
    """Show current progress status."""
    progress = load_progress()
    
    print("\n" + "=" * 50)
    print("MeshPrep Thingi10K Test Status")
    print("=" * 50)
    print(f"Progress: {progress.processed:,} / {progress.total_files:,} ({progress.percent_complete:.1f}%)")
    print(f"Successful: {progress.successful:,} ({progress.success_rate:.1f}%)")
    print(f"Failed: {progress.failed:,}")
    print(f"Escalations: {progress.escalations:,}")
    print(f"Currently: {progress.current_file}")
    print(f"ETA: {str(timedelta(seconds=int(progress.eta_seconds)))}")
    print(f"Dashboard: {DASHBOARD_FILE}")
    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run MeshPrep repair against all Thingi10K models (auto-resumes by default)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        help="Limit number of models to process"
    )
    parser.add_argument(
        "--fresh", "-f",
        action="store_true",
        help="Start fresh - ignore existing reports and reprocess all"
    )
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Show current progress status"
    )
    
    args = parser.parse_args()
    
    if args.status:
        show_status()
        return
    
    # If --fresh is passed, clear existing reports first
    if args.fresh:
        logger.info("Fresh mode: clearing existing reports...")
        # Clear reports folder
        if REPORTS_PATH.exists():
            import shutil
            shutil.rmtree(REPORTS_PATH)
            REPORTS_PATH.mkdir(parents=True, exist_ok=True)
        # Clear fixed files
        if FIXED_OUTPUT_PATH.exists():
            import shutil
            shutil.rmtree(FIXED_OUTPUT_PATH)
            FIXED_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        logger.info("Cleared existing reports and fixed files")
    
    run_batch_test(limit=args.limit)


if __name__ == "__main__":
    main()