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
- Adaptive thresholds: learns optimal values from repair outcomes
- Auto-optimization: thresholds are optimized every 100 models and at batch end
- Quality feedback: learn from user ratings to improve repair quality

Usage:
    python run_full_test.py                    # Run all models (auto-resumes)
    python run_full_test.py --limit 100        # Test first 100 models
    python run_full_test.py --status           # Show current progress
    python run_full_test.py --fresh            # Start fresh (ignore existing reports)
    python run_full_test.py --learning-stats   # Show learning engine statistics
    python run_full_test.py --threshold-stats  # Show adaptive thresholds status
    python run_full_test.py --optimize-thresholds  # Manually optimize thresholds
    python run_full_test.py --reset-thresholds # Reset thresholds to defaults
    python run_full_test.py --no-auto-optimize # Disable automatic optimization
    
    # Quality feedback commands:
    python run_full_test.py --quality-stats    # Show quality feedback statistics
    python run_full_test.py --rate MP:abc123 --rating 4  # Rate a model (1-5)
    python run_full_test.py --rate MP:abc123 --rating 3 --comment "Minor issues"
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


# =============================================================================
# Colored Logging Formatter
# =============================================================================
class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels in terminal output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def __init__(self, fmt=None, datefmt=None, use_colors=True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors
    
    def format(self, record):
        # Save original levelname
        original_levelname = record.levelname
        
        if self.use_colors and record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            # Color the entire line for ERROR and WARNING, just levelname for others
            if record.levelname in ('ERROR', 'CRITICAL'):
                record.levelname = f"{self.BOLD}{color}{record.levelname}{self.RESET}"
                record.msg = f"{self.BOLD}{color}{record.msg}{self.RESET}"
            elif record.levelname == 'WARNING':
                record.levelname = f"{color}{record.levelname}{self.RESET}"
                record.msg = f"{color}{record.msg}{self.RESET}"
            else:
                record.levelname = f"{color}{record.levelname}{self.RESET}"
        
        result = super().format(record)
        
        # Restore original levelname for other handlers (like file handler)
        record.levelname = original_levelname
        
        return result


# Setup logging with file output
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Create handlers
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(ColoredFormatter(
    fmt="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    use_colors=True
))

file_handler = logging.FileHandler(LOG_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
file_handler.setFormatter(logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
))

logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler, file_handler],
    force=True
)
logger = logging.getLogger(__name__)

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Add POC v2 to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "v2"))

import trimesh

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
from meshprep_poc.fingerprint import compute_file_fingerprint, compute_full_file_hash
from meshprep_poc.learning_engine import get_learning_engine, LearningEngine
from meshprep_poc.quality_feedback import (
    get_quality_engine,
    QualityRating,
    QualityFeedbackEngine,
)

# Try to import adaptive thresholds
try:
    from meshprep_poc.adaptive_thresholds import get_adaptive_thresholds, DEFAULT_THRESHOLDS
    ADAPTIVE_THRESHOLDS_AVAILABLE = True
except ImportError:
    ADAPTIVE_THRESHOLDS_AVAILABLE = False
    get_adaptive_thresholds = None
    DEFAULT_THRESHOLDS = {}

# Paths
THINGI10K_PATH = Path(r"C:\Users\Dragon Ace\Source\repos\Thingi10K\raw_meshes")
CTM_MESHES_PATH = Path(r"C:\Users\Dragon Ace\Source\repos\Thingi10K\ctm_meshes")

# Combined output paths (shared for STL and CTM)
OUTPUT_BASE_PATH = Path(r"C:\Users\Dragon Ace\Source\repos\Thingi10K")
REPORTS_PATH = OUTPUT_BASE_PATH / "reports"
FILTERS_PATH = OUTPUT_BASE_PATH / "reports" / "filters"
FIXED_OUTPUT_PATH = OUTPUT_BASE_PATH / "fixed"

PROGRESS_FILE = Path(__file__).parent / "progress.json"
SUMMARY_FILE = Path(__file__).parent / "summary.json"
RESULTS_CSV = Path(__file__).parent / "results.csv"
DASHBOARD_FILE = Path(__file__).parent / "dashboard.html"

# Supported mesh formats
SUPPORTED_FORMATS = {".stl", ".ctm", ".obj", ".ply", ".3mf", ".off"}

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
    
    # Model fingerprint for filter script discovery
    model_fingerprint: str = ""  # Searchable fingerprint: MP:xxxxxxxxxxxx
    original_file_hash: str = ""  # Full SHA256 for exact matching
    
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


def get_all_mesh_files(limit: Optional[int] = None, ctm_priority: bool = False) -> List[Path]:
    """Get all mesh files from Thingi10K and CTM meshes.
    
    Args:
        limit: Optional limit on number of files to return
        ctm_priority: If True, CTM meshes are listed FIRST (before other files)
    """
    thingi_files = []
    ctm_files = []
    
    # Get all supported mesh files from Thingi10K (STL, OBJ, PLY, OFF, 3MF)
    if THINGI10K_PATH.exists():
        for ext in SUPPORTED_FORMATS:
            if ext != '.ctm':  # CTM files are in separate directory
                found = list(THINGI10K_PATH.glob(f"*{ext}"))
                thingi_files.extend(found)
        logger.info(f"Found {len(thingi_files):,} mesh files in Thingi10K")
        
        # Log breakdown by type
        by_ext = {}
        for f in thingi_files:
            ext = f.suffix.lower()
            by_ext[ext] = by_ext.get(ext, 0) + 1
        for ext, count in sorted(by_ext.items()):
            logger.info(f"  {ext}: {count:,}")
    else:
        logger.warning(f"Thingi10K path not found: {THINGI10K_PATH}")
    
    # Always get CTM files
    if CTM_MESHES_PATH.exists():
        ctm_files = list(CTM_MESHES_PATH.glob("*.ctm"))
        logger.info(f"Found {len(ctm_files):,} CTM files in ctm_meshes")
    
    # Combine files - CTM first if priority requested
    if ctm_priority and ctm_files:
        logger.info("CTM PRIORITY MODE: CTM files will be processed first")
        # Sort CTM files first, then Thingi files
        ctm_files = sorted(ctm_files, key=lambda p: p.stem)
        thingi_files = sorted(thingi_files, key=lambda p: p.stem)
        mesh_files = ctm_files + thingi_files
    else:
        # Default: sort all files together by stem
        mesh_files = sorted(thingi_files + ctm_files, key=lambda p: p.stem)
    
    if limit:
        mesh_files = mesh_files[:limit]
    
    return mesh_files


def get_all_stl_files(limit: Optional[int] = None) -> List[Path]:
    """Get all STL files from Thingi10K (legacy function, use get_all_mesh_files)."""
    return get_all_mesh_files(limit=limit)


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


def check_geometry_loss(original_diag, result_mesh, profile: str = "unknown") -> tuple[bool, float, float]:
    """Check if repair caused significant geometry loss.
    
    Uses adaptive thresholds if available.
    """
    import numpy as np
    
    # Get thresholds (adaptive or defaults)
    if ADAPTIVE_THRESHOLDS_AVAILABLE:
        thresholds = get_adaptive_thresholds()
        volume_limit = thresholds.get("volume_loss_limit_pct", profile)
        face_limit = thresholds.get("face_loss_limit_pct", profile)
    else:
        volume_limit = 30.0
        face_limit = 40.0
    
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
    
    significant_loss = volume_loss_pct > volume_limit or face_loss_pct > face_limit
    
    return significant_loss, volume_loss_pct, face_loss_pct


def decimate_mesh(mesh, target_faces: Optional[int] = None, profile: str = "unknown"):
    """Decimate mesh to reduce face count while preserving shape.
    
    Uses adaptive thresholds for target if not specified.
    """
    # Get target from adaptive thresholds if not specified
    if target_faces is None:
        if ADAPTIVE_THRESHOLDS_AVAILABLE:
            thresholds = get_adaptive_thresholds()
            target_faces = int(thresholds.get("decimation_target_faces", profile))
        else:
            target_faces = 100000
    
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


def save_filter_info(
    file_id: str,
    filter_used: str,
    escalated: bool,
    repair_result: Optional[SlicerRepairResult] = None,
    model_fingerprint: str = "",
    original_filename: str = "",
    original_format: str = "",
    before_diagnostics: Optional[Dict[str, Any]] = None,
    after_diagnostics: Optional[Dict[str, Any]] = None,
):
    """Save detailed filter/repair info for a model.
    
    This saves comprehensive data for later analysis to:
    - Refine filter script selection
    - Improve model profile detection
    - Optimize repair pipeline performance
    
    The saved filter script includes the model fingerprint to enable
    community sharing and discovery. Search "MP:xxxxxxxxxxxx" on Reddit
    to find filter scripts for a specific model.
    """
    # Ensure filters directory exists
    FILTERS_PATH.mkdir(parents=True, exist_ok=True)
    
    filter_path = FILTERS_PATH / f"{file_id}.json"
    
    # Extract detailed attempt info for analysis
    attempts_detail = []
    if repair_result and repair_result.attempts:
        for attempt in repair_result.attempts:
            attempt_info = {
                "attempt_number": attempt.attempt_number,
                "pipeline_name": attempt.pipeline_name,
                "actions": attempt.pipeline_actions,
                "success": attempt.success,
                "duration_ms": attempt.duration_ms,
                "error": attempt.error,
                "geometry_valid": attempt.geometry_valid,
            }
            # Include slicer validation results if available
            if attempt.slicer_result:
                attempt_info["slicer_validation"] = {
                    "valid": attempt.slicer_result.valid,
                    "issues": attempt.slicer_result.issues,
                    "warnings": attempt.slicer_result.warnings[:5] if attempt.slicer_result.warnings else [],  # Limit
                    "errors": attempt.slicer_result.errors[:5] if attempt.slicer_result.errors else [],  # Limit
                }
            attempts_detail.append(attempt_info)
    
    # Extract precheck info for analysis
    precheck_info = None
    if repair_result and repair_result.precheck_mesh_info:
        precheck_info = {
            "manifold": repair_result.precheck_mesh_info.manifold,
            "open_edges": repair_result.precheck_mesh_info.open_edges,
            "reversed_facets": getattr(repair_result.precheck_mesh_info, 'reversed_facets', 0),
            "is_clean": repair_result.precheck_mesh_info.is_clean,
            "issues": repair_result.precheck_mesh_info.issues,
        }
    
    # Build comprehensive filter data for analysis
    filter_data = {
        # Model identification
        "model_id": file_id,
        "model_fingerprint": model_fingerprint,  # Searchable: MP:xxxxxxxxxxxx
        "original_filename": original_filename,
        "original_format": original_format,
        
        # Repair outcome
        "filter_name": filter_used,
        "success": repair_result.success if repair_result else False,
        "escalated_to_blender": escalated,
        
        # Precheck results (for analyzing which models need repair)
        "precheck": {
            "passed": repair_result.precheck_passed if repair_result else False,
            "skipped": repair_result.precheck_skipped if repair_result else False,
            "mesh_info": precheck_info,
        },
        
        # Detailed attempt history (for pipeline optimization)
        "repair_attempts": {
            "total_attempts": repair_result.total_attempts if repair_result else 0,
            "total_duration_ms": repair_result.total_duration_ms if repair_result else 0,
            "issues_found": repair_result.issues_found if repair_result else [],
            "issues_resolved": repair_result.issues_resolved if repair_result else [],
            "attempts": attempts_detail,
        },
        
        # Mesh diagnostics (for model profile analysis)
        "diagnostics": {
            "before": before_diagnostics,
            "after": after_diagnostics,
        },
        
        # Metadata
        "timestamp": datetime.now().isoformat(),
        "meshprep_version": "0.1.0",
        "meshprep_url": "https://github.com/DragonAceNL/MeshPrep",
        "sharing_note": f"Search '{model_fingerprint}' on Reddit to find/share filter scripts for this model",
    }
    
    with open(filter_path, "w", encoding="utf-8") as f:
        json.dump(filter_data, f, indent=2)


def extract_mesh_diagnostics(mesh: trimesh.Trimesh, label: str = "") -> Dict[str, Any]:
    """Extract comprehensive mesh diagnostics for analysis.
    
    This captures mesh characteristics useful for:
    - Model profile detection
    - Filter script selection
    - Understanding repair outcomes
    """
    if mesh is None:
        return None
    
    try:
        diagnostics = {
            # Basic geometry
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "edges": len(mesh.edges_unique) if hasattr(mesh, 'edges_unique') else 0,
            
            # Volume and bounds
            "volume": float(mesh.volume) if mesh.is_watertight else None,
            "bounding_box": {
                "min": mesh.bounds[0].tolist() if mesh.bounds is not None else None,
                "max": mesh.bounds[1].tolist() if mesh.bounds is not None else None,
            },
            "extents": mesh.extents.tolist() if hasattr(mesh, 'extents') else None,
            
            # Topology
            "is_watertight": bool(mesh.is_watertight),
            "is_winding_consistent": bool(mesh.is_winding_consistent) if hasattr(mesh, 'is_winding_consistent') else None,
            "euler_number": int(mesh.euler_number) if hasattr(mesh, 'euler_number') else None,
            
            # Body count (fragmentation)
            "body_count": len(mesh.split(only_watertight=False)) if hasattr(mesh, 'split') else 1,
            
            # Face analysis
            "degenerate_faces": int(mesh.degenerate_faces.sum()) if hasattr(mesh, 'degenerate_faces') else 0,
            
            # Area
            "surface_area": float(mesh.area) if hasattr(mesh, 'area') else None,
        }
        
        # Try to get additional diagnostics
        try:
            # Check for non-manifold edges
            if hasattr(mesh, 'edges_unique') and hasattr(mesh, 'faces'):
                # Simple heuristic: non-manifold if we have unusual edge-face relationships
                edges = mesh.edges_sorted.reshape(-1, 2)
                edge_counts = {}
                for e in map(tuple, edges):
                    edge_counts[e] = edge_counts.get(e, 0) + 1
                non_manifold_edges = sum(1 for c in edge_counts.values() if c > 2)
                diagnostics["non_manifold_edges"] = non_manifold_edges
        except:
            pass
        
        return diagnostics
        
    except Exception as e:
        return {
            "error": f"Failed to extract diagnostics: {str(e)}",
            "vertices": len(mesh.vertices) if hasattr(mesh, 'vertices') else 0,
            "faces": len(mesh.faces) if hasattr(mesh, 'faces') else 0,
        }


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
        # Compute fingerprint FIRST (from original file bytes)
        # This ensures CTM files are fingerprinted as CTM, not as decompressed mesh
        result.model_fingerprint = compute_file_fingerprint(stl_path)
        result.original_file_hash = compute_full_file_hash(stl_path)
        logger.info(f"  Fingerprint: {result.model_fingerprint}")
        
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
            max_attempts=20,  # More attempts for pipeline combinations
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
        
        # Decimate if mesh is too large (use adaptive threshold)
        if ADAPTIVE_THRESHOLDS_AVAILABLE:
            thresholds = get_adaptive_thresholds()
            decimation_trigger = int(thresholds.get("decimation_trigger_faces", "unknown"))
        else:
            decimation_trigger = 100000
        
        if len(repaired.faces) > decimation_trigger:
            original_repaired = repaired.copy()
            repaired = decimate_mesh(repaired, profile="unknown")  # Uses adaptive target
            
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
        
        # Extract diagnostics for analysis
        before_diagnostics = extract_mesh_diagnostics(original, "before")
        after_diagnostics = extract_mesh_diagnostics(repaired, "after") if repaired else None
        
        # Save detailed filter info for later analysis
        save_filter_info(
            file_id=file_id,
            filter_used=result.filter_used,
            escalated=result.escalation_used,
            repair_result=repair_result,
            model_fingerprint=result.model_fingerprint,
            original_filename=stl_path.name,
            original_format=stl_path.suffix.lstrip('.').lower(),
            before_diagnostics=before_diagnostics,
            after_diagnostics=after_diagnostics,
        )
        
        # Feed result to learning engine for continuous improvement
        try:
            learning_engine = get_learning_engine()
            filter_data = {
                "success": result.success,
                "escalated_to_blender": result.escalation_used,
                "precheck": {
                    "passed": repair_result.precheck_passed if repair_result else False,
                    "skipped": repair_result.precheck_skipped if repair_result else False,
                    "mesh_info": {
                        "issues": repair_result.issues_found if repair_result else [],
                    } if repair_result else None,
                },
                "repair_attempts": {
                    "total_attempts": repair_result.total_attempts if repair_result else 0,
                    "attempts": [
                        {
                            "pipeline_name": a.pipeline_name,
                            "success": a.success,
                            "duration_ms": a.duration_ms,
                        }
                        for a in (repair_result.attempts if repair_result else [])
                    ],
                },
                "diagnostics": {
                    "before": before_diagnostics,
                    "after": after_diagnostics,
                },
            }
            learning_engine.record_result(filter_data)
        except Exception as le_error:
            logger.debug(f"Learning engine update failed: {le_error}")
        
        # Feed observations to adaptive thresholds engine
        try:
            if ADAPTIVE_THRESHOLDS_AVAILABLE:
                adaptive = get_adaptive_thresholds()
                profile = "unknown"  # TODO: detect profile from diagnostics
                
                # Calculate quality score based on geometry preservation
                quality = 1.0
                if vol_loss > 0:
                    quality -= min(vol_loss / 100, 0.5)  # Up to 0.5 penalty for volume loss
                if face_loss > 0:
                    quality -= min(face_loss / 100, 0.3)  # Up to 0.3 penalty for face loss
                quality = max(0, quality)
                
                # Record geometry loss observations
                adaptive.record_geometry_loss(
                    volume_loss_pct=vol_loss,
                    face_loss_pct=face_loss,
                    success=result.success,
                    quality=quality,
                    profile=profile,
                    escalated=result.escalation_used,
                )
                
                # Record decimation observations if decimation was performed
                if result.original_faces > decimation_trigger:
                    decimation_success = result.success and result.result_watertight
                    adaptive.record_decimation(
                        original_faces=result.original_faces,
                        target_faces=int(adaptive.get("decimation_target_faces", profile)),
                        result_faces=result.result_faces,
                        success=decimation_success,
                        quality=quality,
                        profile=profile,
                    )
                
                # Record repair attempt observations
                if repair_result:
                    adaptive.record_repair_attempts(
                        attempts_used=repair_result.total_attempts,
                        duration_ms=repair_result.total_duration_ms,
                        success=result.success,
                        profile=profile,
                    )
                
                # Record body count threshold observations for learning
                # This helps the system learn optimal fragmentation thresholds
                if before_diagnostics and "body_count" in before_diagnostics:
                    body_count = before_diagnostics["body_count"]
                    
                    # Record observation for extreme fragmentation threshold
                    extreme_threshold = adaptive.get("body_count_extreme_fragmented")
                    adaptive.record_observation(
                        threshold_name="body_count_extreme_fragmented",
                        threshold_value=extreme_threshold,
                        actual_value=body_count,
                        success=result.success,
                        quality=quality,
                        profile=profile,
                    )
                    
                    # Record observation for fragmented threshold
                    frag_threshold = adaptive.get("body_count_fragmented")
                    adaptive.record_observation(
                        threshold_name="body_count_fragmented",
                        threshold_value=frag_threshold,
                        actual_value=body_count,
                        success=result.success,
                        quality=quality,
                        profile=profile,
                    )
        except Exception as at_error:
            logger.debug(f"Adaptive thresholds update failed: {at_error}")
        
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
        
        /* Fingerprint box for filter script discovery */
        .fingerprint-box {{
            background: linear-gradient(135deg, #1b2b33 0%, #2a3a43 100%);
            border: 2px solid #4fe8c4;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 25px;
            text-align: center;
        }}
        .fingerprint-label {{
            color: #888;
            font-size: 14px;
            margin-bottom: 10px;
        }}
        .fingerprint-value {{
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 28px;
            font-weight: bold;
            color: #4fe8c4;
            background: #0f1720;
            padding: 15px 25px;
            border-radius: 8px;
            display: inline-block;
            cursor: pointer;
            user-select: all;
            transition: all 0.2s;
        }}
        .fingerprint-value:hover {{
            background: #1a2530;
            transform: scale(1.02);
        }}
        .fingerprint-help {{
            margin-top: 12px;
            font-size: 13px;
            color: #666;
        }}
        .fingerprint-help a {{
            color: #4fe8c4;
            text-decoration: none;
            padding: 4px 8px;
            border-radius: 4px;
            background: #1b2b33;
        }}
        .fingerprint-help a:hover {{
            background: #2a3a43;
        }}
        .copy-hint {{
            color: #555;
            font-style: italic;
        }}
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
        
        // Copy fingerprint to clipboard
        function copyFingerprint() {{
            const fingerprint = '{result.model_fingerprint}';
            navigator.clipboard.writeText(fingerprint).then(() => {{
                const el = document.querySelector('.fingerprint-value');
                const original = el.textContent;
                el.textContent = '\u2713 Copied!';
                el.style.background = '#27ae60';
                setTimeout(() => {{
                    el.textContent = original;
                    el.style.background = '#0f1720';
                }}, 1500);
            }}).catch(err => {{
                // Fallback for older browsers
                const el = document.querySelector('.fingerprint-value');
                const range = document.createRange();
                range.selectNodeContents(el);
                window.getSelection().removeAllRanges();
                window.getSelection().addRange(range);
                document.execCommand('copy');
                window.getSelection().removeAllRanges();
            }});
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
        
        <!-- Fingerprint Box - for searching/sharing filter scripts -->
        <div class="fingerprint-box">
            <div class="fingerprint-label">&#128269; Model Fingerprint (search this on Reddit to find filter scripts)</div>
            <div class="fingerprint-value" onclick="copyFingerprint()" title="Click to copy">{result.model_fingerprint}</div>
            <div class="fingerprint-help">
                <a href="https://www.reddit.com/search/?q={result.model_fingerprint}" target="_blank">Search Reddit</a> | 
                <a href="https://www.google.com/search?q={result.model_fingerprint}" target="_blank">Search Google</a> |
                <a href="https://github.com/DragonAceNL/MeshPrep" target="_blank">MeshPrep GitHub</a> |
                <span class="copy-hint">Click fingerprint to copy</span>
            </div>
        </div>
        
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
        
        .nav-links {{
            margin-bottom: 20px;
        }}
        .nav-links a {{
            color: #4fe8c4;
            text-decoration: none;
            padding: 8px 16px;
            background: #1b2b33;
            border-radius: 6px;
            margin-right: 10px;
        }}
        .nav-links a:hover {{
            background: #2a3a43;
        }}
        
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
        
        <div class="nav-links">
            <a href="/reports/index.html">📋 Reports Index</a>
            <a href="/learning">🧠 Learning Status</a>
            <a href="/live">📺 Live Dashboard</a>
        </div>
        
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
                    <td>{r.original_faces:,} -> {r.result_faces:,}</td>
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


def load_results_from_reports() -> List[TestResult]:
    """Load test results from existing HTML reports and filter JSON files."""
    results = []
    
    # Get all HTML reports (excluding index.html)
    for html_file in REPORTS_PATH.glob("*.html"):
        if html_file.stem == "index":
            continue
        
        file_id = html_file.stem
        filter_json = FILTERS_PATH / f"{file_id}.json"
        
        # Try to load from filter JSON which has more accurate data
        if filter_json.exists():
            try:
                with open(filter_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                result = TestResult(
                    file_id=file_id,
                    file_path=str(THINGI10K_PATH / f"{file_id}.stl"),
                    success=True,  # If report exists, assume success
                    filter_used=data.get("filter_name", "unknown"),
                    escalation_used=data.get("escalated_to_blender", False),
                    precheck_skipped=data.get("precheck_skipped", False),
                    precheck_passed=data.get("precheck_passed", False),
                    timestamp=data.get("timestamp", ""),
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to load filter JSON for {file_id}: {e}")
                # Create minimal result
                results.append(TestResult(
                    file_id=file_id,
                    file_path=str(THINGI10K_PATH / f"{file_id}.stl"),
                    success=True,
                    filter_used="unknown",
                ))
        else:
            # No filter JSON - try to parse HTML for status
            try:
                with open(html_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Check if it's an "Already Clean" report
                is_clean = 'Already Clean' in html_content or 'already clean' in html_content.lower()
                is_skipped = 'skipped' in html_content.lower() or 'none (already clean)' in html_content.lower()
                
                results.append(TestResult(
                    file_id=file_id,
                    file_path=str(THINGI10K_PATH / f"{file_id}.stl"),
                    success=True,
                    filter_used="none (already clean)" if (is_clean or is_skipped) else "unknown",
                    precheck_skipped=(is_clean or is_skipped),
                ))
            except Exception as e:
                logger.warning(f"Failed to parse HTML for {file_id}: {e}")
                results.append(TestResult(
                    file_id=file_id,
                    file_path=str(THINGI10K_PATH / f"{file_id}.stl"),
                    success=True,
                    filter_used="unknown",
                ))
    
    return results


def generate_reports_index(results: List[TestResult] = None):
    """Generate an index.html in the reports folder for easy navigation.
    
    If results is None or empty, loads results from existing report files.
    """
    # If no results passed, load from existing reports
    if not results:
        results = load_results_from_reports()
    
    if not results:
        logger.warning("No results to generate index from")
        return
    
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
                const aNum = parseFloat(aVal.replace(/[^\\d.-]/g, ''));
                const bNum = parseFloat(bVal.replace(/[^\\d.-]/g, ''));
                
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
    
    # Define CSV columns - fingerprint first for easy searching
    columns = [
        'model_fingerprint', 'file_id', 'success', 'filter_used', 'escalation_used', 'duration_ms',
        'precheck_passed', 'precheck_skipped',
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
            'model_fingerprint': result.model_fingerprint,  # Searchable fingerprint
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


def run_batch_test(limit: Optional[int] = None, skip_existing: bool = True, ctm_priority: bool = False, auto_optimize: bool = True):
    """Run the full batch test.
    
    Args:
        limit: Optional limit on number of files to process
        skip_existing: If True (default), skip files that already have reports
        ctm_priority: If True, process CTM meshes FIRST before other files
        auto_optimize: If True (default), automatically optimize thresholds during processing
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
    
    # Check PyMeshLab for CTM support
    try:
        import pymeshlab
        print("[OK] PyMeshLab available for CTM support", flush=True)
        logger.info("[OK] PyMeshLab available for CTM support")
    except ImportError:
        print("[WARN] PyMeshLab not installed - CTM files will fail", flush=True)
        print("        Install with: pip install pymeshlab", flush=True)
        logger.warning("PyMeshLab not installed - CTM files may fail")
    
    # Initialize learning engine and show stats
    try:
        learning_engine = get_learning_engine()
        stats = learning_engine.get_stats_summary()
        if stats["total_models_processed"] > 0:
            print(f"[OK] Learning engine: {stats['total_models_processed']} models learned", flush=True)
            logger.info(f"Learning engine loaded: {stats['total_models_processed']} models")
            if stats["top_pipelines"]:
                top = stats["top_pipelines"][0]
                print(f"     Top pipeline: {top['name']} ({top['success_rate']*100:.0f}% success)", flush=True)
        else:
            print("[OK] Learning engine: Starting fresh (no prior data)", flush=True)
            logger.info("Learning engine: Starting fresh")
    except Exception as e:
        print(f"[WARN] Learning engine unavailable: {e}", flush=True)
        logger.warning(f"Learning engine error: {e}")
    
    # Initialize adaptive thresholds and show stats
    if ADAPTIVE_THRESHOLDS_AVAILABLE:
        try:
            adaptive = get_adaptive_thresholds()
            stats = adaptive.get_stats_summary()
            if stats["total_observations"] > 0:
                print(f"[OK] Adaptive thresholds: {stats['total_observations']:,} observations, {stats['thresholds_adjusted']} adjusted", flush=True)
                logger.info(f"Adaptive thresholds: {stats['total_observations']} observations")
            else:
                print("[OK] Adaptive thresholds: Starting fresh (collecting observations)", flush=True)
                logger.info("Adaptive thresholds: Starting fresh")
        except Exception as e:
            print(f"[WARN] Adaptive thresholds unavailable: {e}", flush=True)
            logger.warning(f"Adaptive thresholds error: {e}")
    else:
        print("[INFO] Adaptive thresholds: Not available (using defaults)", flush=True)
    
    # Get ALL mesh files (for total count)
    all_mesh_files = get_all_mesh_files(limit=None, ctm_priority=ctm_priority)
    total_mesh_count = len(all_mesh_files)
    
    # Get files to consider (with optional limit)
    files_to_consider = get_all_mesh_files(limit, ctm_priority=ctm_priority)
    print(f"Found {total_mesh_count:,} total mesh files", flush=True)
    
    # Show breakdown by type
    by_ext = {}
    for f in all_mesh_files:
        ext = f.suffix.lower()
        by_ext[ext] = by_ext.get(ext, 0) + 1
    for ext, count in sorted(by_ext.items()):
        print(f"  - {ext}: {count:,}", flush=True)
    
    if limit:
        print(f"Processing first {len(files_to_consider):,} files (--limit {limit})", flush=True)
    logger.info(f"Found {total_mesh_count:,} mesh files")
    
    # Check for already processed files
    processed_ids = get_processed_files()
    if processed_ids:
        print(f"Found {len(processed_ids):,} existing reports", flush=True)
        logger.info(f"Found {len(processed_ids):,} existing reports")
    
    # Determine which files to process
    if skip_existing:
        to_process = [f for f in files_to_consider if f.stem not in processed_ids]
        print(f"Will process {len(to_process):,} new files (skipping existing)", flush=True)
        logger.info(f"Will process {len(to_process):,} new files (skipping existing)")
    else:
        to_process = files_to_consider
        print(f"Will process {len(to_process):,} files (--fresh mode, may overwrite existing)", flush=True)
        logger.info(f"Will process {len(to_process):,} files (--fresh mode)")
    
    if len(to_process) == 0:
        print("\nAll files already processed! Use --fresh to reprocess.", flush=True)
        logger.info("All files already processed! Use --fresh to reprocess.")
        return
    
    # Load existing results from reports (for index regeneration)
    existing_results = load_results_from_reports()
    existing_ids = {r.file_id for r in existing_results}
    logger.info(f"Loaded {len(existing_results)} existing results from reports")
    
    # Initialize progress - use total count, not limited count
    progress = Progress(
        total_files=total_mesh_count,
        processed=len(processed_ids),  # Start from existing report count
        successful=len(processed_ids),  # Assume existing are successful
        start_time=datetime.now().isoformat(),
    )
    
    # Save initial progress immediately so dashboard shows correct state
    save_progress(progress)
    generate_dashboard(progress, existing_results)  # Use existing results for initial dashboard
    
    # Start with existing results, new results will be added
    results: List[TestResult] = list(existing_results)
    new_results_count = 0  # Track only new results for progress display
    
    # Process each file (only files that need processing)
    for i, stl_path in enumerate(to_process):
        file_id = stl_path.stem
        
        # Update progress
        progress.current_file = file_id
        progress.processed += 1
        
        # Calculate ETA based on new results only
        if new_results_count > 0 and progress.avg_duration_ms > 0:
            remaining = len(to_process) - new_results_count
            progress.eta_seconds = (remaining * progress.avg_duration_ms) / 1000
        
        # Log progress
        eta_str = str(timedelta(seconds=int(progress.eta_seconds))) if progress.eta_seconds > 0 else "calculating..."
        print(f"[{new_results_count+1}/{len(to_process)}] Processing {file_id}... (ETA: {eta_str})", flush=True)
        logger.info(f"[{i+1}/{len(to_process)}] Processing {file_id}...")
        
        # Process
        result = process_single_model(stl_path, progress=progress)
        results.append(result)
        new_results_count += 1
        
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
        if new_results_count > 0:
            progress.avg_duration_ms = progress.total_duration_ms / new_results_count
        
        # Save progress after every file (for live dashboard)
        save_progress(progress)
        
        # Update dashboard every 10 files (slower operation)
        if i % 10 == 0:
            generate_dashboard(progress, results)
            generate_reports_index(results)
            # Also update learning status page
            try:
                from generate_learning_status import generate_learning_status_page
                generate_learning_status_page()
            except Exception as e:
                logger.debug(f"Failed to generate learning status page: {e}")
        
        # Auto-optimize thresholds every 100 models
        if auto_optimize and ADAPTIVE_THRESHOLDS_AVAILABLE and new_results_count > 0 and new_results_count % 100 == 0:
            try:
                adaptive = get_adaptive_thresholds()
                adjustments = adaptive.optimize_thresholds(min_samples=20)
                if adjustments:
                    logger.info(f"Auto-optimized {len(adjustments)} thresholds after {new_results_count} models")
                    for adj in adjustments:
                        logger.info(f"  {adj['threshold']}: {adj['old_value']:.2f} -> {adj['new_value']:.2f}")
            except Exception as e:
                logger.debug(f"Auto-optimization failed: {e}")
        
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
    
    # Final threshold optimization
    if auto_optimize and ADAPTIVE_THRESHOLDS_AVAILABLE and new_results_count >= 20:
        try:
            adaptive = get_adaptive_thresholds()
            stats = adaptive.get_stats_summary()
            print(f"\nAdaptive thresholds: {stats['total_observations']:,} observations", flush=True)
            
            adjustments = adaptive.optimize_thresholds(min_samples=20)
            if adjustments:
                print(f"Auto-optimized {len(adjustments)} thresholds:", flush=True)
                for adj in adjustments:
                    print(f"  {adj['threshold']}: {adj['old_value']:.2f} -> {adj['new_value']:.2f}", flush=True)
            else:
                print(f"Thresholds are optimal ({stats['thresholds_adjusted']} adjusted from defaults)", flush=True)
        except Exception as e:
            logger.debug(f"Final optimization failed: {e}")
    
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


def show_learning_stats():
    """Display detailed learning engine statistics."""
    print("=" * 60)
    
    # Show adaptive thresholds stats
    print("\n" + "=" * 60)
    print("Adaptive Thresholds Statistics")
    print("=" * 60)
    
    if ADAPTIVE_THRESHOLDS_AVAILABLE:
        try:
            adaptive = get_adaptive_thresholds()
            stats = adaptive.get_stats_summary()
            
            print(f"\nTotal observations: {stats['total_observations']:,}")
            print(f"Thresholds adjusted: {stats['thresholds_adjusted']} / {stats['total_thresholds']}")
            
            if stats['threshold_status']:
                print(f"\nCurrent thresholds (changed from default marked with *):")
                print(f"  {'Threshold':<35} {'Current':>12} {'Default':>12} {'Change':>10}")
                print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*10}")
                for t in stats['threshold_status']:
                    marker = "*" if t['changed'] else " "
                    change_str = f"{t['change_pct']:+.1f}%" if t['changed'] else "-"
                    print(f"{marker} {t['name']:<35} {t['current']:>12.2f} {t['default']:>12.2f} {change_str:>10}")
            
            if stats['recent_adjustments']:
                print(f"\nRecent adjustments:")
                for adj in stats['recent_adjustments'][:5]:
                    print(f"  {adj['threshold']}: {adj['old']:.2f} -> {adj['new']:.2f}")
                    print(f"    Reason: {adj['reason']}")
            
        except Exception as e:
            print(f"Error loading adaptive thresholds: {e}")
    else:
        print("\nAdaptive thresholds not available.")
    
    print("=" * 60)
    print("MeshPrep Learning Engine Statistics")
    print("=" * 60)
    
    try:
        engine = get_learning_engine()
        stats = engine.get_stats_summary()
        
        print(f"\nModels processed: {stats['total_models_processed']:,}")
        print(f"Last updated: {stats['last_updated'] or 'Never'}")
        print(f"Pipelines tracked: {stats['pipelines_tracked']}")
        print(f"Issue patterns tracked: {stats['issue_patterns_tracked']}")
        print(f"Profiles tracked: {stats['profiles_tracked']}")
        
        if stats['optimal_pipeline_order']:
            print(f"\nOptimal pipeline order (top 5):")
            for i, name in enumerate(stats['optimal_pipeline_order'], 1):
                print(f"  {i}. {name}")
        
        if stats['top_pipelines']:
            print(f"\nTop pipelines by success rate:")
            print(f"  {'Pipeline':<30} {'Success':>8} {'Avg Time':>10} {'Attempts':>10}")
            print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*10}")
            for p in stats['top_pipelines']:
                print(f"  {p['name']:<30} {p['success_rate']*100:>7.1f}% {p['avg_duration_ms']:>8.0f}ms {p['attempts']:>10}")
        
        if stats['profile_summary']:
            print(f"\nProfile summary:")
            print(f"  {'Profile':<20} {'Total':>8} {'Fix Rate':>10}")
            print(f"  {'-'*20} {'-'*8} {'-'*10}")
            for name, data in stats['profile_summary'].items():
                print(f"  {name:<20} {data['total']:>8} {data['fix_rate']*100:>9.1f}%")
        
        # Show where data is saved
        print(f"\nLearning data location: {engine.data_path}")
        
    except Exception as e:
        print(f"Error loading learning stats: {e}")
    
    # Show evolution stats if available
    print("\n" + "=" * 60)
    print("Pipeline Evolution Statistics")
    print("=" * 60)
    
    try:
        from meshprep_poc.pipeline_evolution import get_evolution_engine
        evolution = get_evolution_engine()
        evo_stats = evolution.get_stats_summary()
        
        print(f"\nEvolved pipelines: {evo_stats['total_evolved_pipelines']}")
        print(f"Successful evolved: {evo_stats['successful_evolved_pipelines']}")
        print(f"Actions tracked: {evo_stats['tracked_actions']}")
        print(f"Current generation: {evo_stats['current_generation']}")
        
        if evo_stats['top_evolved_pipelines']:
            print(f"\nTop evolved pipelines:")
            print(f"  {'Pipeline':<40} {'Success':>8} {'Attempts':>10}")
            print(f"  {'-'*40} {'-'*8} {'-'*10}")
            for p in evo_stats['top_evolved_pipelines']:
                print(f"  {p['name']:<40} {p['success_rate']*100:>7.1f}% {p['attempts']:>10}")
        else:
            print("\nNo evolved pipelines yet. They will be generated when standard pipelines fail.")
        
    except ImportError:
        print("\nEvolution engine not available.")
    except Exception as e:
        print(f"Error loading evolution stats: {e}")
    
    # Show profile discovery stats
    print("\n" + "=" * 60)
    print("Profile Discovery Statistics")
    print("=" * 60)
    
    try:
        from meshprep_poc.profile_discovery import get_discovery_engine
        discovery = get_discovery_engine()
        stats = discovery.get_stats_summary()
        
        print(f"\nClusters: {stats['total_clusters']}")
        print(f"Models clustered: {stats['total_models_clustered']:,}")
        print(f"Active discovered profiles: {stats['active_profiles']}")
        print(f"Models with profiles: {stats.get('models_with_profiles', 0):,}")
        print(f"Unassigned clusters: {stats['unassigned_clusters']}")
        
        if stats.get('top_profiles'):
            print(f"\nTop discovered profiles:")
            print(f"  {'Name':<40} {'Models':>8} {'Success':>8}")
            print(f"  {'-'*40} {'-'*8} {'-'*8}")
            for p in stats['top_profiles'][:5]:
                print(f"  {p['name']:<40} {p['total_models']:>8} {p['success_rate']*100:>7.1f}%")
        else:
            print("\nNo discovered profiles yet. Run --discover-profiles after processing enough models.")
        
    except ImportError:
        print("\nProfile discovery not available.")
    except Exception as e:
        print(f"Error loading profile discovery stats: {e}")
    
    print("=" * 60)


def run_profile_discovery(min_samples: int = 50):
    """Run profile discovery to create new profiles from clustered data."""
    print("=" * 60)
    print("MeshPrep Profile Discovery")
    print("=" * 60)
    
    try:
        from meshprep_poc.profile_discovery import get_discovery_engine
        
        discovery = get_discovery_engine()
        stats = discovery.get_stats_summary()
        
        print(f"\nCurrent state:")
        print(f"  Total clusters: {stats['total_clusters']}")
        print(f"  Models clustered: {stats['total_models_clustered']:,}")
        print(f"  Active profiles: {stats['active_profiles']}")
        print(f"  Unassigned clusters: {stats['unassigned_clusters']}")
        print(f"  Unassigned models: {stats.get('unassigned_models', 0):,}")
        
        if stats['total_models_clustered'] < min_samples:
            print(f"\nNot enough samples for discovery ({stats['total_models_clustered']} < {min_samples})")
            print("Process more models first.")
            return
        
        print(f"\nRunning profile discovery (min_samples={min_samples})...")
        
        profiles = discovery.run_discovery(min_samples=min_samples)
        
        if profiles:
            print(f"\n[OK] Discovered {len(profiles)} new profiles:")
            for p in profiles:
                print(f"  - {p.name}")
                print(f"    {p.description}")
                print(f"    Models: {p.total_models}, Success rate: {p.success_rate*100:.1f}%")
                if p.best_pipeline:
                    print(f"    Best pipeline: {p.best_pipeline}")
        else:
            print("\nNo new profiles discovered.")
            print("This could mean:")
            print("  - All clusters are already assigned to profiles")
            print("  - No clusters meet the minimum size requirements")
        
        # Show updated stats
        stats = discovery.get_stats_summary()
        print(f"\nUpdated state:")
        print(f"  Active profiles: {stats['active_profiles']}")
        print(f"  Models with profiles: {stats.get('models_with_profiles', 0):,}")
        print(f"  Avg profile success rate: {stats.get('avg_profile_success_rate', 0)*100:.0f}%")
        
        if stats.get('top_profiles'):
            print(f"\nTop profiles:")
            print(f"  {'Name':<40} {'Models':>8} {'Success':>8}")
            print(f"  {'-'*40} {'-'*8} {'-'*8}")
            for p in stats['top_profiles'][:5]:
                print(f"  {p['name']:<40} {p['total_models']:>8} {p['success_rate']*100:>7.1f}%")
        
    except ImportError:
        print("\nProfile discovery engine not available.")
        print("Make sure meshprep_poc.profile_discovery is importable.")
    except Exception as e:
        print(f"\nError during profile discovery: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)


def optimize_adaptive_thresholds(min_samples: int = 20):
    """Optimize adaptive thresholds based on collected observations."""
    print("=" * 60)
    print("MeshPrep Adaptive Thresholds Optimization")
    print("=" * 60)
    
    if not ADAPTIVE_THRESHOLDS_AVAILABLE:
        print("\nAdaptive thresholds not available.")
        return
    
    try:
        adaptive = get_adaptive_thresholds()
        stats = adaptive.get_stats_summary()
        
        print(f"\nCurrent state:")
        print(f"  Total observations: {stats['total_observations']:,}")
        print(f"  Thresholds adjusted: {stats['thresholds_adjusted']} / {stats['total_thresholds']}")
        
        if stats['total_observations'] < min_samples:
            print(f"\nNot enough observations for optimization ({stats['total_observations']} < {min_samples})")
            print("Process more models first.")
            return
        
        print(f"\nRunning optimization (min_samples={min_samples})...")
        
        adjustments = adaptive.optimize_thresholds(min_samples=min_samples)
        
        if adjustments:
            print(f"\n[OK] Made {len(adjustments)} adjustments:")
            for adj in adjustments:
                print(f"  {adj['threshold']}: {adj['old_value']:.2f} -> {adj['new_value']:.2f}")
                print(f"    Reason: {adj['reason']}")
                print(f"    Based on {adj['observations']} observations")
        else:
            print("\nNo adjustments needed.")
            print("Current thresholds are optimal based on observations.")
        
        # Show updated stats
        stats = adaptive.get_stats_summary()
        print(f"\nUpdated state:")
        print(f"  Thresholds adjusted: {stats['thresholds_adjusted']} / {stats['total_thresholds']}")
        
        if stats['threshold_status']:
            print(f"\nCurrent thresholds:")
            print(f"  {'Threshold':<35} {'Current':>12} {'Default':>12}")
            print(f"  {'-'*35} {'-'*12} {'-'*12}")
            for t in stats['threshold_status']:
                if t['changed']:
                    print(f"* {t['name']:<35} {t['current']:>12.2f} {t['default']:>12.2f}")
        
    except Exception as e:
        print(f"\nError during optimization: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)


def reset_adaptive_thresholds():
    """Reset all adaptive thresholds to defaults."""
    print("=" * 60)
    print("MeshPrep Adaptive Thresholds Reset")
    print("=" * 60)
    
    if not ADAPTIVE_THRESHOLDS_AVAILABLE:
        print("\nAdaptive thresholds not available.")
        return
    
    try:
        adaptive = get_adaptive_thresholds()
        stats = adaptive.get_stats_summary()
        
        print(f"\nCurrent state:")
        print(f"  Thresholds adjusted: {stats['thresholds_adjusted']} / {stats['total_thresholds']}")
        
        if stats['thresholds_adjusted'] == 0:
            print("\nNo thresholds have been adjusted from defaults.")
            return
        
        print("\nResetting all thresholds to defaults...")
        adaptive.reset_to_defaults()
        
        print("[OK] All thresholds reset to default values.")
        print("\nNote: Observation history is preserved for future optimization.")
        
    except Exception as e:
        print(f"\nError during reset: {e}")
    
    print("\n" + "=" * 60)


def show_quality_stats():
    """Show quality feedback statistics."""
    print("=" * 60)
    print("MeshPrep Quality Feedback Statistics")
    print("=" * 60)
    
    try:
        quality_engine = get_quality_engine()
        stats = quality_engine.get_summary_stats()
        
        print(f"\nTotal ratings: {stats['total_ratings']:,}")
        print(f"Average rating: {stats['avg_rating']:.2f}/5")
        print(f"Overall acceptance rate: {stats['overall_acceptance_rate']:.1f}%")
        print(f"Pipeline/profile combinations tracked: {stats['pipeline_profile_combinations']}")
        print(f"Ready for prediction: {'Yes' if stats['ready_for_prediction'] else 'No (need more ratings)'}")
        
        if stats['rating_distribution']:
            print("\nRating distribution:")
            for score in range(5, 0, -1):
                count = stats['rating_distribution'].get(score, 0)
                bar = "#" * min(count, 50)
                print(f"  {score} stars: {count:>5} {bar}")
        
        # Show pipeline quality stats
        all_stats = quality_engine.get_all_pipeline_quality_stats()
        if all_stats:
            print("\nPipeline quality by profile:")
            print(f"  {'Pipeline':<25} {'Profile':<15} {'Ratings':>8} {'Avg':>6} {'Accept%':>8}")
            print(f"  {'-'*25} {'-'*15} {'-'*8} {'-'*6} {'-'*8}")
            
            for pipeline, profiles in all_stats.items():
                for profile, pstats in profiles.items():
                    if pstats.total_ratings > 0:
                        print(f"  {pipeline:<25} {profile:<15} {pstats.total_ratings:>8} {pstats.avg_rating:>6.2f} {pstats.acceptance_rate*100:>7.1f}%")
        
    except Exception as e:
        print(f"\nError getting quality stats: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)


def rate_model_by_fingerprint(fingerprint: str, rating: int, comment: Optional[str] = None):
    """Rate a model by its fingerprint."""
    print("=" * 60)
    print("MeshPrep Quality Rating")
    print("=" * 60)
    
    # Normalize fingerprint format
    if not fingerprint.startswith("MP:"):
        fingerprint = f"MP:{fingerprint}"
    
    print(f"\nFingerprint: {fingerprint}")
    print(f"Rating: {rating}/5")
    if comment:
        print(f"Comment: {comment}")
    
    try:
        quality_engine = get_quality_engine()
        
        # Check if this fingerprint already has a rating
        existing = quality_engine.get_rating_by_fingerprint(fingerprint)
        if existing:
            print(f"\nNote: This model was previously rated {existing.rating_value}/5")
            print(f"  by {existing.rated_by or 'anonymous'} on {existing.rated_at}")
            print("  This new rating will be added to the history.")
        
        # Find matching report to get context
        # Search in reports directory for matching fingerprint
        model_filename = "unknown"
        pipeline_used = "unknown"
        profile = "standard"
        volume_change_pct = 0.0
        face_count_change_pct = 0.0
        escalated = False
        
        # Try to find matching report
        for report_file in REPORTS_PATH.rglob("report.json"):
            try:
                with open(report_file) as f:
                    data = json.load(f)
                if data.get("model_fingerprint") == fingerprint:
                    model_filename = Path(data.get("input_file", "unknown")).name
                    pipeline_used = data.get("filter_script", "unknown")
                    escalated = data.get("escalation_used", False)
                    
                    # Get metrics
                    orig = data.get("original_diagnostics", {})
                    rep = data.get("repaired_diagnostics", {})
                    if orig and rep:
                        if orig.get("volume", 0) != 0:
                            volume_change_pct = ((rep.get("volume", 0) - orig.get("volume", 0)) / abs(orig.get("volume", 1))) * 100
                        if orig.get("face_count", 0) > 0:
                            face_count_change_pct = ((rep.get("face_count", 0) - orig.get("face_count", 0)) / orig.get("face_count", 1)) * 100
                    
                    print(f"\nFound matching report:")
                    print(f"  File: {model_filename}")
                    print(f"  Pipeline: {pipeline_used}")
                    break
            except:
                continue
        
        # Create and record rating
        quality_rating = QualityRating(
            model_fingerprint=fingerprint,
            model_filename=model_filename,
            rating_type="gradational",
            rating_value=rating,
            user_comment=comment,
            rated_by="cli_user",
            pipeline_used=pipeline_used,
            profile=profile,
            escalated=escalated,
            volume_change_pct=volume_change_pct,
            face_count_change_pct=face_count_change_pct,
        )
        
        quality_engine.record_rating(quality_rating)
        
        print(f"\n[OK] Rating recorded successfully!")
        
        # Show updated prediction for this pipeline/profile
        prediction = quality_engine.predict_quality(
            pipeline=pipeline_used,
            profile=profile,
            volume_change_pct=volume_change_pct,
            face_count_change_pct=face_count_change_pct,
            escalated=escalated,
        )
        print(f"\nUpdated prediction for {pipeline_used}:")
        print(f"  Predicted score: {prediction.score:.2f}/5")
        print(f"  Confidence: {prediction.confidence:.1%}")
        print(f"  Based on {prediction.based_on_samples} samples")
        
    except Exception as e:
        print(f"\nError recording rating: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)


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
        help="Reprocess all files (doesn't delete existing results, just overwrites)"
    )
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Show current progress status"
    )
    parser.add_argument(
        "--ctm-priority",
        action="store_true",
        help="Process CTM meshes FIRST before other files"
    )
    parser.add_argument(
        "--learning-stats",
        action="store_true",
        help="Show learning engine statistics"
    )
    parser.add_argument(
        "--generate-status-page",
        action="store_true",
        help="Generate HTML learning status page"
    )
    parser.add_argument(
        "--discover-profiles",
        action="store_true",
        help="Run profile discovery to create new profiles from clustered data"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=50,
        help="Minimum samples required before running profile discovery (default: 50)"
    )
    parser.add_argument(
        "--optimize-thresholds",
        action="store_true",
        help="Optimize adaptive thresholds based on collected observations"
    )
    parser.add_argument(
        "--reset-thresholds",
        action="store_true",
        help="Reset all adaptive thresholds to defaults"
    )
    parser.add_argument(
        "--threshold-stats",
        action="store_true",
        help="Show adaptive thresholds statistics only"
    )
    parser.add_argument(
        "--no-auto-optimize",
        action="store_true",
        help="Disable automatic threshold optimization during batch processing"
    )
    
    # Quality feedback options
    parser.add_argument(
        "--quality-stats",
        action="store_true",
        help="Show quality feedback statistics"
    )
    parser.add_argument(
        "--rate",
        type=str,
        metavar="FINGERPRINT",
        help="Rate a model by fingerprint (e.g., MP:42f3729aa758)"
    )
    parser.add_argument(
        "--rating",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Quality rating (1-5) to assign when using --rate"
    )
    parser.add_argument(
        "--comment",
        type=str,
        help="Optional comment when rating a model"
    )
    
    args = parser.parse_args()
    
    if args.status:
        show_status()
        return
    
    if args.learning_stats:
        show_learning_stats()
        return
    
    if args.generate_status_page:
        from generate_learning_status import generate_learning_status_page
        page_path = generate_learning_status_page()
        print(f"\nStatus page generated: {page_path}")
        print(f"Open in browser or serve with: python reports_server.py")
        return
    
    if args.discover_profiles:
        run_profile_discovery(min_samples=args.min_samples)
        return
    
    if args.optimize_thresholds:
        optimize_adaptive_thresholds(min_samples=args.min_samples)
        return
    
    if args.reset_thresholds:
        reset_adaptive_thresholds()
        return
    
    if args.threshold_stats:
        if ADAPTIVE_THRESHOLDS_AVAILABLE:
            adaptive = get_adaptive_thresholds()
            stats = adaptive.get_stats_summary()
            
            print("=" * 60)
            print("Adaptive Thresholds Statistics")
            print("=" * 60)
            print(f"\nTotal observations: {stats['total_observations']:,}")
            print(f"Thresholds adjusted: {stats['thresholds_adjusted']} / {stats['total_thresholds']}")
            
            if stats['threshold_status']:
                print(f"\nCurrent thresholds:")
                print(f"  {'Threshold':<35} {'Current':>12} {'Default':>12} {'Change':>10}")
                print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*10}")
                for t in stats['threshold_status']:
                    marker = "*" if t['changed'] else " "
                    change_str = f"{t['change_pct']:+.1f}%" if t['changed'] else "-"
                    print(f"{marker} {t['name']:<35} {t['current']:>12.2f} {t['default']:>12.2f} {change_str:>10}")
            
            print("=" * 60)
        else:
            print("Adaptive thresholds not available.")
        return
    
    if args.quality_stats:
        show_quality_stats()
        return
    
    if args.rate:
        if not args.rating:
            print("Error: --rating is required when using --rate")
            return
        rate_model_by_fingerprint(args.rate, args.rating, args.comment)
        return
    
    # If --fresh is passed, reprocess all files (but don't delete existing results)
    if args.fresh:
        logger.info("Fresh mode: will reprocess all files (existing results preserved until overwritten)")
    
    run_batch_test(
        limit=args.limit,
        skip_existing=not args.fresh,
        ctm_priority=args.ctm_priority,
        auto_optimize=not args.no_auto_optimize,
    )


if __name__ == "__main__":
    main()