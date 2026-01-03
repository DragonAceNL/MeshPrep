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
from meshprep_poc.slicer_repair_loop import run_slicer_repair_loop, SlicerRepairResult
from meshprep_poc.actions.blender_actions import is_blender_available
from meshprep_poc.actions.slicer_actions import get_mesh_info_prusa, is_slicer_available

# Paths
THINGI10K_PATH = Path(r"C:\Users\Dragon Ace\Source\repos\Thingi10K\raw_meshes")
REPORTS_PATH = Path(r"C:\Users\Dragon Ace\Source\repos\Thingi10K\raw_meshes\reports")
FILTERS_PATH = Path(r"C:\Users\Dragon Ace\Source\repos\Thingi10K\raw_meshes\reports\filters")
FIXED_OUTPUT_PATH = Path(r"C:\Users\Dragon Ace\Source\repos\Thingi10K\raw_meshes\fixed")
PROGRESS_FILE = Path(__file__).parent / "progress.json"
SUMMARY_FILE = Path(__file__).parent / "summary.json"
RESULTS_CSV = Path(__file__).parent / "results.csv"
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
    
    # Pre-check result (slicer --info before repair)
    precheck_passed: bool = False  # True if model was already clean
    precheck_skipped: bool = False  # True if we skipped repair due to precheck
    
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
    
    # Additional diagnostics
    original_components: int = 0
    original_holes: int = 0
    result_components: int = 0
    result_holes: int = 0
    
    # File sizes (bytes)
    original_file_size: int = 0
    fixed_file_size: int = 0
    
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
    precheck_skipped: int = 0  # NEW: Models skipped because already clean
    
    start_time: str = ""
    last_update: str = ""
    current_file: str = ""
    current_action: str = ""  # NEW: Current action being executed
    current_step: int = 0  # NEW: Current step number (e.g., 1 of 4)
    total_steps: int = 0  # NEW: Total steps in current filter
    
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
    """Save the filter script used for a model (legacy)."""
    # Ensure filters directory exists
    FILTERS_PATH.mkdir(parents=True, exist_ok=True)
    
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


def save_filter_info(file_id: str, filter_used: str, escalated: bool, repair_result: Optional[SlicerRepairResult] = None):
    """Save filter/repair info for a model."""
    # Ensure filters directory exists
    FILTERS_PATH.mkdir(parents=True, exist_ok=True)
    
    filter_path = FILTERS_PATH / f"{file_id}.json"
    
    # Build filter data
    filter_data = {
        "model_id": file_id,
        "filter_name": filter_used,
        "escalated_to_blender": escalated,
        "repair_loop_used": repair_result is not None,
        "precheck_passed": repair_result.precheck_passed if repair_result else False,
        "precheck_skipped": repair_result.precheck_skipped if repair_result else False,
        "total_attempts": repair_result.total_attempts if repair_result else 0,
        "attempts": [
            {
                "action": a.strategy.action,
                "params": a.strategy.params,
                "success": a.success,
            }
            for a in (repair_result.attempts if repair_result else [])
        ],
        "timestamp": datetime.now().isoformat()
    }
    
    with open(filter_path, "w", encoding="utf-8") as f:
        json.dump(filter_data, f, indent=2)


# Global progress reference for callback
_current_progress: Optional[Progress] = None


def update_action_progress(action_index: int, action_name: str, total_actions: int):
    """Callback to update progress with current action."""
    global _current_progress
    if _current_progress:
        _current_progress.current_action = action_name
        _current_progress.current_step = action_index + 1
        _current_progress.total_steps = total_actions
        save_progress(_current_progress)


def process_single_model(stl_path: Path, skip_if_clean: bool = True, progress: Optional[Progress] = None) -> TestResult:
    """Process a single model using the POC v2 slicer repair loop.
    
    This is a thin wrapper around the v2 repair loop that:
    1. Loads the mesh
    2. Runs the slicer repair loop (which includes STRICT pre-check)
    3. Captures metrics and generates reports
    
    Args:
        stl_path: Path to the STL file
        skip_if_clean: If True, skip repair if model already passes STRICT slicer check
        progress: Optional Progress object for live updates
    """
    global _current_progress
    _current_progress = progress  # Set for callback
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
        result.original_file_size = stl_path.stat().st_size
        
        # Additional diagnostics
        try:
            result.original_components = len(original.split(only_watertight=False))
        except:
            result.original_components = 1
        
        # =====================================================================
        # Use POC v2 slicer repair loop (includes STRICT pre-check)
        # =====================================================================
        repair_result = run_slicer_repair_loop(
            mesh=original,
            slicer="auto",
            max_attempts=10,
            escalate_to_blender_after=5,
            timeout=120,
            skip_if_clean=skip_if_clean,
            progress_callback=update_action_progress,
        )
        
        # Capture pre-check results from repair loop
        result.precheck_passed = repair_result.precheck_passed
        result.precheck_skipped = repair_result.precheck_skipped
        
        if repair_result.precheck_skipped:
            # Model was already clean - no repair needed
            logger.info(f"  PRE-CHECK PASSED: Model already clean (from v2 repair loop)")
            result.success = True
            result.filter_used = "none (already clean)"
            result.duration_ms = repair_result.total_duration_ms
            
            # Result = Original (no changes)
            result.result_vertices = result.original_vertices
            result.result_faces = result.original_faces
            result.result_volume = result.original_volume
            result.result_watertight = result.original_watertight
            result.result_manifold = result.original_manifold
            
            # Generate report showing it was skipped
            generate_report(stl_path, original, original, result, None)
            return result
        
        # Repair was attempted
        if repair_result.success and repair_result.final_mesh is not None:
            repaired = repair_result.final_mesh
            result.filter_used = "slicer-repair-loop"
            
            # Check if Blender escalation was used
            for attempt in repair_result.attempts:
                if "blender" in attempt.strategy.action:
                    result.escalation_used = True
                    result.filter_used = f"slicer-repair-loop (blender)"
                    break
        else:
            # Repair failed, fall back to conservative filter script approach
            logger.info(f"  Slicer repair loop failed, trying filter script approach...")
            filter_script = get_best_filter(original)
            result.filter_used = filter_script.name
            
            runner = FilterScriptRunner(stop_on_error=False)
            filter_result = runner.run(filter_script, original, progress_callback=update_action_progress)
            
            if not filter_result.success or filter_result.final_mesh is None:
                result.success = False
                result.error = filter_result.error or repair_result.error or "Repair failed"
                result.duration_ms = (time.time() - start_time) * 1000
                
                # Generate failed report
                generate_report(stl_path, original, original, result, None)
                return result
            
            repaired = filter_result.final_mesh
            
            # Check for geometry loss and escalate if needed
            validation = validate_repair(original, repaired)
            significant_loss, vol_loss, face_loss = check_geometry_loss(original_diag, repaired)
            
            needs_escalation = (
                not validation.geometric.is_printable or
                significant_loss
            )
            
            if needs_escalation and is_blender_available():
                logger.info(f"  Escalating to Blender...")
                escalation_script = get_preset("blender-remesh")
                escalation_result = runner.run(escalation_script, original, progress_callback=update_action_progress)
                
                if escalation_result.success and escalation_result.final_mesh is not None:
                    repaired = escalation_result.final_mesh
                    result.filter_used = "blender-remesh"
                    result.escalation_used = True
        
        # Calculate geometry changes
        significant_loss, vol_loss, face_loss = check_geometry_loss(original_diag, repaired)
        result.volume_change_pct = vol_loss
        result.face_change_pct = face_loss
        
        # Capture printability status BEFORE decimation
        validation = validate_repair(original, repaired)
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
        
        # Result diagnostics
        try:
            result.result_components = len(repaired.split(only_watertight=False))
        except:
            result.result_components = 1
        
        # Success based on FINAL mesh state
        result.success = repaired.is_watertight and repaired.is_volume
        result.duration_ms = (time.time() - start_time) * 1000
        
        # Save repaired model to fixed directory
        fixed_path = FIXED_OUTPUT_PATH / f"{file_id}.stl"
        if result.success:
            save_mesh(repaired, fixed_path)
            result.fixed_file_size = fixed_path.stat().st_size
            logger.info(f"  Saved fixed model to {fixed_path}")
        
        # Save filter info (simplified since we use repair loop now)
        save_filter_info(file_id, result.filter_used, result.escalation_used, repair_result)
        
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
    """Generate HTML report in the reports subfolder."""
    report_path = REPORTS_PATH / f"{stl_path.stem}.html"
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
    if result.precheck_skipped:
        status_class = "skipped"
        status_text = "&#10003; Already Clean"
    elif result.success:
        if result.escalation_used:
            status_class = "escalated"
            status_text = "&#10003; Fixed (Blender)"
        else:
            status_class = "success"
            status_text = "&#10003; Fixed"
    else:
        status_class = "failed"
        status_text = "&#10007; Failed"
    
    # Calculate changes
    vertex_change = result.result_vertices - result.original_vertices
    face_change = result.result_faces - result.original_faces
    face_change_pct = (face_change / result.original_faces * 100) if result.original_faces > 0 else 0
    
    # Navigation links
    prev_link = f'<a href="{prev_file.stem}.html">&lt; {prev_file.stem}</a>' if prev_file else '<span class="disabled">&lt; Previous</span>'
    next_link = f'<a href="{next_file.stem}.html">{next_file.stem} &gt;</a>' if next_file else '<span class="disabled">Next &gt;</span>'
    
    # Fixed model link - use relative path (works when served via http server)
    if fixed_path and fixed_path.exists():
        fixed_rel_path = f"../fixed/{fixed_path.name}"
        fixed_link = f'''<a href="{fixed_rel_path}" class="download-btn" download>&#11015; Download Fixed</a>
            <button class="download-btn meshlab-btn" onclick="openInMeshLab('{fixed_rel_path}')">&#128065; MeshLab</button>'''
    elif result.precheck_skipped:
        # Model was already clean, no need to save a fixed version
        fixed_link = '<span class="no-file">Original is already clean</span>'
    else:
        fixed_link = '<span class="no-file">Repair failed - no fixed model</span>'
    
    # Original model link - use relative path
    original_rel_path = f'../{stl_path.name}'
    
    # Duration formatting
    duration_sec = result.duration_ms / 1000
    if duration_sec >= 60:
        duration_text = f"{duration_sec/60:.1f} minutes"
    else:
        duration_text = f"{duration_sec:.1f} seconds"
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{stl_path.stem} - MeshPrep Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f1720;
            color: #dff6fb;
            margin: 0;
            padding: 20px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #4fe8c4; margin-bottom: 5px; }}
        
        .nav-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #1b2b33;
            padding: 10px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .nav-bar a {{ color: #4fe8c4; text-decoration: none; padding: 5px 10px; }}
        .nav-bar a:hover {{ background: #2a3a43; border-radius: 4px; }}
        .nav-bar .disabled {{ color: #555; }}
        
        .status-badge {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            margin-bottom: 20px;
        }}
        .status-badge.success {{ background: #2ecc71; color: #0f1720; }}
        .status-badge.failed {{ background: #e74c3c; color: white; }}
        .status-badge.skipped {{ background: #3498db; color: white; }}
        .status-badge.escalated {{ background: #f39c12; color: #0f1720; }}
        
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .info-card {{
            background: #1b2b33;
            padding: 15px;
            border-radius: 8px;
        }}
        .info-card .label {{ color: #888; font-size: 12px; margin-bottom: 5px; }}
        .info-card .value {{ font-size: 18px; font-weight: bold; }}
        
        .comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }}
        .comparison-panel {{
            background: #1b2b33;
            border-radius: 12px;
            overflow: hidden;
        }}
        .comparison-panel h3 {{
            margin: 0;
            padding: 15px;
            background: #0f1720;
            color: #4fe8c4;
        }}
        .comparison-panel img {{
            width: 100%;
            height: 400px;
            object-fit: contain;
            background: #0a0f14;
        }}
        
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            background: #1b2b33;
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 30px;
        }}
        .metrics-table th, .metrics-table td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #2a3a43;
        }}
        .metrics-table th {{ background: #0f1720; color: #4fe8c4; }}
        
        .change-positive {{ color: #e74c3c; }}
        .change-negative {{ color: #2ecc71; }}
        .change-neutral {{ color: #888; }}
        
        .downloads {{
            display: flex;
            gap: 15px;
            margin-bottom: 30px;
        }}
        .download-btn {{
            display: inline-block;
            background: #4fe8c4;
            color: #0f1720;
            padding: 12px 24px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: bold;
        }}
        .download-btn:hover {{ background: #3dd4b0; }}
        .download-btn.secondary {{ background: #1b2b33; color: #4fe8c4; }}
        .no-file {{ color: #888; padding: 12px 24px; }}
        
        .error-box {{
            background: #2a1a1a;
            border: 1px solid #e74c3c;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 30px;
        }}
        .error-box h3 {{ color: #e74c3c; margin-top: 0; }}
        .error-box pre {{ margin: 0; white-space: pre-wrap; }}
        
        .footer {{ color: #555; font-size: 12px; margin-top: 30px; }}
        
        a {{ color: #4fe8c4; }}
        
        .meshlab-btn {{
            background: #9b59b6 !important;
            color: white !important;
            border: none;
            cursor: pointer;
        }}
        .meshlab-btn:hover {{ background: #8e44ad !important; }}
        
        .meshlab-status {{
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 12px;
            margin-left: 10px;
        }}
        .meshlab-available {{ background: #27ae60; color: white; }}
        .meshlab-unavailable {{ background: #e74c3c; color: white; }}
    </style>
    <script>
        async function openInMeshLab(filePath) {{
            const btn = event.target;
            const originalText = btn.innerHTML;
            btn.innerHTML = '&#8987; Opening...';
            btn.disabled = true;
            
            try {{
                const url = '/api/open-meshlab?file=' + encodeURIComponent(filePath);
                console.log('Requesting:', url);
                
                const response = await fetch(url);
                const data = await response.json();
                console.log('Response:', data);
                
                if (!response.ok || !data.success) {{
                    alert('Failed to open in MeshLab:\n\n' + (data.error || 'Unknown error'));
                }} else {{
                    btn.innerHTML = '&#10003; Opened!';
                    setTimeout(() => {{ btn.innerHTML = originalText; btn.disabled = false; }}, 2000);
                    return;
                }}
            }} catch (e) {{
                console.error('Error:', e);
                alert('Error: ' + e.message + '\n\nMake sure you are using the MeshPrep reports server:\n\ncd poc/v3\npython reports_server.py');
            }}
            
            btn.innerHTML = originalText;
            btn.disabled = false;
        }}
        
        // Check MeshLab availability on page load
        async function checkMeshLab() {{
            try {{
                const response = await fetch('/api/meshlab-status');
                const data = await response.json();
                const indicator = document.getElementById('meshlab-indicator');
                if (indicator) {{
                    if (data.available) {{
                        indicator.className = 'meshlab-status meshlab-available';
                        indicator.textContent = '\u2713 MeshLab Ready';
                        indicator.title = 'MeshLab: ' + data.path;
                    }} else {{
                        indicator.className = 'meshlab-status meshlab-unavailable';
                        indicator.textContent = '\u2717 MeshLab Not Found';
                    }}
                }}
            }} catch (e) {{
                // Server doesn't support MeshLab API
                const indicator = document.getElementById('meshlab-indicator');
                if (indicator) {{
                    indicator.className = 'meshlab-status meshlab-unavailable';
                    indicator.textContent = 'Use reports_server.py';
                    indicator.title = 'Run: python reports_server.py';
                }}
            }}
        }}
        window.onload = checkMeshLab;
    </script>
</head>
<body>
    <div class="container">
        <div class="nav-bar">
            <div>{prev_link}</div>
            <div>
                <a href="index.html">&#128209; Index</a>
                <a href="/dashboard">&#128202; Dashboard</a>
                <span id="meshlab-indicator" class="meshlab-status">Checking MeshLab...</span>
            </div>
            <div>{next_link}</div>
        </div>
        
        <h1>{stl_path.stem}</h1>
        <span class="status-badge {status_class}">{status_text}</span>
        
        <div class="info-grid">
            <div class="info-card">
                <div class="label">Filter Used</div>
                <div class="value">{result.filter_used}</div>
            </div>
            <div class="info-card">
                <div class="label">Duration</div>
                <div class="value">{duration_text}</div>
            </div>
            <div class="info-card">
                <div class="label">Original Faces</div>
                <div class="value">{result.original_faces:,}</div>
            </div>
            <div class="info-card">
                <div class="label">Result Faces</div>
                <div class="value">{result.result_faces:,}</div>
            </div>
        </div>
        
        <div class="downloads">
            <a href="{original_rel_path}" class="download-btn secondary" download>&#11015; Download Original</a>
            <button class="download-btn meshlab-btn" onclick="openInMeshLab('{original_rel_path}')">&#128065; MeshLab</button>
            {fixed_link}
            <a href="filters/{stl_path.stem}.json" class="download-btn secondary">&#128196; Filter Script</a>
        </div>
        
        <h2>Visual Comparison</h2>
        <div class="comparison">
            <div class="comparison-panel">
                <h3>Before</h3>
                <img src="images/{stl_path.stem}_before.png" alt="Before">
            </div>
            <div class="comparison-panel">
                <h3>After</h3>
                <img src="images/{stl_path.stem}_after.png" alt="After">
            </div>
        </div>
        
        <h2>Metrics</h2>
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Before</th>
                    <th>After</th>
                    <th>Change</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Vertices</td>
                    <td>{result.original_vertices:,}</td>
                    <td>{result.result_vertices:,}</td>
                    <td class="{'change-positive' if vertex_change > 0 else 'change-negative' if vertex_change < 0 else 'change-neutral'}">{vertex_change:+,}</td>
                </tr>
                <tr>
                    <td>Faces</td>
                    <td>{result.original_faces:,}</td>
                    <td>{result.result_faces:,}</td>
                    <td class="{'change-positive' if face_change > 0 else 'change-negative' if face_change < 0 else 'change-neutral'}">{face_change:+,} ({face_change_pct:+.1f}%)</td>
                </tr>
                <tr>
                    <td>Volume</td>
                    <td>{result.original_volume:.2f}</td>
                    <td>{result.result_volume:.2f}</td>
                    <td class="{'change-positive' if result.volume_change_pct > 5 else 'change-negative' if result.volume_change_pct < -5 else 'change-neutral'}">{result.volume_change_pct:+.1f}%</td>
                </tr>
                <tr>
                    <td>Watertight</td>
                    <td>{'&#10003; Yes' if result.original_watertight else '&#10007; No'}</td>
                    <td>{'&#10003; Yes' if result.result_watertight else '&#10007; No'}</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Manifold</td>
                    <td>{'&#10003; Yes' if result.original_manifold else '&#10007; No'}</td>
                    <td>{'&#10003; Yes' if result.result_manifold else '&#10007; No'}</td>
                    <td>-</td>
                </tr>
            </tbody>
        </table>
"""
    
    if result.error:
        html_content += f"""        <div class="error-box">
            <h3>&#9888; Error</h3>
            <pre>{result.error}</pre>
        </div>
"""
    
    html_content += f"""        <div class="footer">
            Generated: {result.timestamp}
        </div>
    </div>
</body>
</html>
"""
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)


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
            <div style="flex: 1;">
                <div>Currently processing: <strong>{progress.current_file or 'Waiting...'}</strong></div>
                <div style="margin-top: 5px; font-size: 14px; color: #888;">
                    Action: <span style="color: #4fe8c4;">{progress.current_action or '-'}</span>
                    {f'(step {progress.current_step} of {progress.total_steps})' if progress.current_step else ''}
                </div>
            </div>
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
                <div class="stat-value" style="color: #3498db;">{progress.precheck_skipped:,}</div>
                <div class="stat-label">Already Clean</div>
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


def generate_reports_index(results: List[TestResult]):
    """Generate an index.html in the reports folder for easy navigation."""
    index_path = REPORTS_PATH / "index.html"
    
    # Sort results by file_id
    sorted_results = sorted(results, key=lambda r: r.file_id)
    
    # Calculate stats
    total = len(results)
    successful = sum(1 for r in results if r.success)
    failed = total - successful
    precheck_skipped = sum(1 for r in results if r.precheck_skipped)
    escalations = sum(1 for r in results if r.escalation_used)
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>MeshPrep Reports Index</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f1720;
            color: #dff6fb;
            margin: 0;
            padding: 20px;
        }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        h1 {{ color: #4fe8c4; margin-bottom: 10px; }}
        .subtitle {{ color: #888; margin-bottom: 20px; }}
        
        .stats-row {{
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .stat {{
            background: #1b2b33;
            padding: 10px 20px;
            border-radius: 8px;
        }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #4fe8c4; }}
        .stat-label {{ font-size: 12px; color: #888; }}
        
        .filters {{
            background: #1b2b33;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
        }}
        .filters label {{ color: #888; }}
        .filters select, .filters input {{
            background: #0f1720;
            color: #dff6fb;
            border: 1px solid #333;
            padding: 8px 12px;
            border-radius: 4px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            background: #1b2b33;
            border-radius: 12px;
            overflow: hidden;
        }}
        th, td {{
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid #2a3a43;
        }}
        th {{
            background: #0f1720;
            color: #4fe8c4;
            cursor: pointer;
            user-select: none;
            position: sticky;
            top: 0;
        }}
        th:hover {{ background: #1a2a35; }}
        tr:hover {{ background: #2a3a43; }}
        
        .success {{ color: #2ecc71; }}
        .failed {{ color: #e74c3c; }}
        .skipped {{ color: #3498db; }}
        .escalated {{ color: #f39c12; }}
        
        .change-positive {{ color: #e74c3c; }}  /* Red for increase */
        .change-negative {{ color: #2ecc71; }}  /* Green for decrease */
        .change-neutral {{ color: #888; }}
        
        a {{ color: #4fe8c4; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        
        .thumbnail {{
            width: 60px;
            height: 60px;
            object-fit: contain;
            background: #0f1720;
            border-radius: 4px;
        }}
        
        .search-box {{
            flex: 1;
            min-width: 200px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📋 MeshPrep Reports Index</h1>
        <p class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Total: {total} models</p>
        
        <div class="stats-row">
            <div class="stat">
                <div class="stat-value">{total}</div>
                <div class="stat-label">Total</div>
            </div>
            <div class="stat">
                <div class="stat-value success">{successful}</div>
                <div class="stat-label">Successful</div>
            </div>
            <div class="stat">
                <div class="stat-value failed">{failed}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat">
                <div class="stat-value skipped">{precheck_skipped}</div>
                <div class="stat-label">Already Clean</div>
            </div>
            <div class="stat">
                <div class="stat-value escalated">{escalations}</div>
                <div class="stat-label">Blender</div>
            </div>
        </div>
        
        <div class="filters">
            <label>Filter:</label>
            <select id="statusFilter" onchange="filterTable()">
                <option value="all">All</option>
                <option value="success">Successful</option>
                <option value="failed">Failed</option>
                <option value="skipped">Already Clean</option>
                <option value="escalated">Blender Escalation</option>
            </select>
            
            <label>Search:</label>
            <input type="text" id="searchBox" class="search-box" placeholder="Search by model ID..." onkeyup="filterTable()">
        </div>
        
        <table id="resultsTable">
            <thead>
                <tr>
                    <th onclick="sortTable(0)">Model ID</th>
                    <th onclick="sortTable(1)">Status</th>
                    <th onclick="sortTable(2)">Filter</th>
                    <th onclick="sortTable(3)">Faces Before</th>
                    <th onclick="sortTable(4)">Faces After</th>
                    <th onclick="sortTable(5)">Face Change</th>
                    <th onclick="sortTable(6)">Volume Change</th>
                    <th onclick="sortTable(7)">Duration</th>
                    <th>Report</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for r in sorted_results:
        # Status
        if r.precheck_skipped:
            status_class = "skipped"
            status_text = "&#10003; Clean"  # checkmark
            status_data = "skipped"
        elif r.success:
            if r.escalation_used:
                status_class = "escalated"
                status_text = "&#10003; Blender"  # checkmark
                status_data = "escalated"
            else:
                status_class = "success"
                status_text = "&#10003; Fixed"  # checkmark
                status_data = "success"
        else:
            status_class = "failed"
            status_text = "&#10007; Failed"  # X mark
            status_data = "failed"
        
        # Face change
        face_change = r.result_faces - r.original_faces
        face_change_pct = (face_change / r.original_faces * 100) if r.original_faces > 0 else 0
        if face_change > 0:
            face_change_class = "change-positive"
            face_change_text = f"+{face_change:,} ({face_change_pct:+.1f}%)"
        elif face_change < 0:
            face_change_class = "change-negative"
            face_change_text = f"{face_change:,} ({face_change_pct:+.1f}%)"
        else:
            face_change_class = "change-neutral"
            face_change_text = "No change"
        
        # Volume change
        if r.volume_change_pct > 5:
            vol_change_class = "change-positive"
        elif r.volume_change_pct < -5:
            vol_change_class = "change-negative"
        else:
            vol_change_class = "change-neutral"
        vol_change_text = f"{r.volume_change_pct:+.1f}%" if r.volume_change_pct != 0 else "~0%"
        
        # Duration
        duration_sec = r.duration_ms / 1000
        if duration_sec >= 60:
            duration_text = f"{duration_sec/60:.1f}m"
        else:
            duration_text = f"{duration_sec:.1f}s"
        
        html += f"""                <tr data-status="{status_data}">
                    <td><strong>{r.file_id}</strong></td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{r.filter_used}</td>
                    <td>{r.original_faces:,}</td>
                    <td>{r.result_faces:,}</td>
                    <td class="{face_change_class}">{face_change_text}</td>
                    <td class="{vol_change_class}">{vol_change_text}</td>
                    <td>{duration_text}</td>
                    <td><a href="{r.file_id}.html">View</a></td>
                </tr>
"""
    
    html += """            </tbody>
        </table>
    </div>
    
    <script>
        let sortDirection = {};
        
        function sortTable(columnIndex) {
            const table = document.getElementById('resultsTable');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            
            // Toggle sort direction
            sortDirection[columnIndex] = !sortDirection[columnIndex];
            const dir = sortDirection[columnIndex] ? 1 : -1;
            
            rows.sort((a, b) => {
                let aVal = a.cells[columnIndex].textContent.trim();
                let bVal = b.cells[columnIndex].textContent.trim();
                
                // Try numeric sort first
                const aNum = parseFloat(aVal.replace(/[^\d.-]/g, ''));
                const bNum = parseFloat(bVal.replace(/[^\d.-]/g, ''));
                
                if (!isNaN(aNum) && !isNaN(bNum)) {
                    return (aNum - bNum) * dir;
                }
                
                return aVal.localeCompare(bVal) * dir;
            });
            
            rows.forEach(row => tbody.appendChild(row));
        }
        
        function filterTable() {
            const statusFilter = document.getElementById('statusFilter').value;
            const searchText = document.getElementById('searchBox').value.toLowerCase();
            const rows = document.querySelectorAll('#resultsTable tbody tr');
            
            rows.forEach(row => {
                const status = row.getAttribute('data-status');
                const modelId = row.cells[0].textContent.toLowerCase();
                
                let showByStatus = statusFilter === 'all' || status === statusFilter;
                let showBySearch = modelId.includes(searchText);
                
                row.style.display = (showByStatus && showBySearch) ? '' : 'none';
            });
        }
    </script>
</body>
</html>
"""
    
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    logger.info(f"Generated reports index: {index_path}")


def get_processed_files() -> set:
    """Get set of already processed file IDs."""
    processed = set()
    for html_file in REPORTS_PATH.glob("*.html"):
        # Skip the index file
        if html_file.stem != "index":
            processed.add(html_file.stem)
    return processed


def append_to_csv(result: TestResult):
    """Append a single result to the CSV file."""
    import csv
    
    # Define CSV columns
    columns = [
        'file_id', 'success', 'filter_used', 'escalation_used', 'duration_ms',
        'precheck_passed', 'precheck_skipped',  # NEW: Pre-check fields
        'original_vertices', 'original_faces', 'original_volume', 
        'original_watertight', 'original_manifold',
        'result_vertices', 'result_faces', 'result_volume',
        'result_watertight', 'result_manifold',
        'volume_change_pct', 'face_change_pct',
        'original_components', 'original_holes',
        'result_components', 'result_holes',
        'original_file_size', 'fixed_file_size',
        'error', 'timestamp'
    ]
    
    # Check if file exists (write header if not)
    write_header = not RESULTS_CSV.exists()
    
    with open(RESULTS_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        
        if write_header:
            writer.writeheader()
        
        # Convert result to dict and write
        row = {
            'file_id': result.file_id,
            'success': result.success,
            'filter_used': result.filter_used,
            'escalation_used': result.escalation_used,
            'duration_ms': result.duration_ms,
            'precheck_passed': result.precheck_passed,
            'precheck_skipped': result.precheck_skipped,
            'original_vertices': result.original_vertices,
            'original_faces': result.original_faces,
            'original_volume': result.original_volume,
            'original_watertight': result.original_watertight,
            'original_manifold': result.original_manifold,
            'result_vertices': result.result_vertices,
            'result_faces': result.result_faces,
            'result_volume': result.result_volume,
            'result_watertight': result.result_watertight,
            'result_manifold': result.result_manifold,
            'volume_change_pct': result.volume_change_pct,
            'face_change_pct': result.face_change_pct,
            'original_components': result.original_components,
            'original_holes': result.original_holes,
            'result_components': result.result_components,
            'result_holes': result.result_holes,
            'original_file_size': result.original_file_size,
            'fixed_file_size': result.fixed_file_size,
            'error': result.error or '',
            'timestamp': result.timestamp
        }
        writer.writerow(row)


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
    
    # Get ALL STL files (for total count)
    all_stl_files = get_all_stl_files(limit=None)
    total_stl_count = len(all_stl_files)
    
    # Get files to consider (with optional limit)
    files_to_consider = get_all_stl_files(limit)
    print(f"Found {total_stl_count:,} total STL files", flush=True)
    if limit:
        print(f"Processing first {len(files_to_consider):,} files (--limit {limit})", flush=True)
    logger.info(f"Found {total_stl_count:,} STL files")
    
    # Always check for already processed files (auto-resume)
    processed_ids = get_processed_files()
    if processed_ids:
        print(f"Found {len(processed_ids):,} existing reports - will skip those", flush=True)
        logger.info(f"Found {len(processed_ids):,} existing reports - will skip those")
    
    # Count how many we'll actually process
    to_process = [f for f in files_to_consider if f.stem not in processed_ids]
    print(f"Will process {len(to_process):,} new files", flush=True)
    logger.info(f"Will process {len(to_process):,} new files")
    
    if len(to_process) == 0:
        print("\nAll files already processed! Use --fresh to reprocess.", flush=True)
        logger.info("All files already processed! Use --fresh to reprocess.")
        return
    
    # Initialize progress - use total count, not limited count
    progress = Progress(
        total_files=total_stl_count,
        processed=len(processed_ids),  # Start from existing report count
        successful=len(processed_ids),  # Assume existing are successful
        start_time=datetime.now().isoformat(),
    )
    
    # Save initial progress immediately so dashboard shows correct state
    save_progress(progress)
    generate_dashboard(progress, [])
    
    results: List[TestResult] = []
    
    # Process each file (only files that need processing)
    for i, stl_path in enumerate(to_process):
        file_id = stl_path.stem
        
        # Update progress
        progress.current_file = file_id
        progress.processed += 1
        
        # Calculate ETA
        if len(results) > 0 and progress.avg_duration_ms > 0:
            remaining = len(to_process) - len(results)
            progress.eta_seconds = (remaining * progress.avg_duration_ms) / 1000
        
        # Log progress
        eta_str = str(timedelta(seconds=int(progress.eta_seconds))) if progress.eta_seconds > 0 else "calculating..."
        print(f"[{len(results)+1}/{len(to_process)}] Processing {file_id}... (ETA: {eta_str})", flush=True)
        logger.info(f"[{i+1}/{len(to_process)}] Processing {file_id}...")
        
        # Process
        result = process_single_model(stl_path, progress=progress)
        results.append(result)
        
        # Append to CSV for incremental export
        append_to_csv(result)
        
        # Update stats
        if result.success:
            progress.successful += 1
        else:
            progress.failed += 1
        
        if result.escalation_used:
            progress.escalations += 1
        
        if result.precheck_skipped:
            progress.precheck_skipped += 1
        
        progress.total_duration_ms += result.duration_ms
        progress.avg_duration_ms = progress.total_duration_ms / len(results)
        
        # Save progress after every file (for live dashboard)
        save_progress(progress)
        
        # Update dashboard every 10 files (slower operation)
        if i % 10 == 0:
            generate_dashboard(progress, results)
            generate_reports_index(results)
        
        # Log result
        status = "[OK]" if result.success else "[FAIL]"
        escalation = " [BLENDER]" if result.escalation_used else ""
        print(f"  {status}{escalation} {result.filter_used} ({result.duration_ms:.0f}ms)", flush=True)
        logger.info(f"  {status}{escalation} {result.filter_used} ({result.duration_ms:.0f}ms)")
    
    # Final save
    save_progress(progress)
    generate_dashboard(progress, results)
    generate_reports_index(results)  # Generate navigable index in reports folder
    
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