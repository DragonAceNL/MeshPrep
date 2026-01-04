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

# Import HTML generators (separated for cleaner code)
from report_generator import generate_model_report
from dashboard_generator import generate_dashboard as generate_dashboard_html
from index_generator import generate_reports_index as generate_index_html, load_results_from_reports

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
    
    # Reconstruction mode (for extreme-fragmented meshes)
    is_reconstruction: bool = False  # True if mesh was reconstructed (not repaired)
    reconstruction_method: str = ""  # Pipeline that performed reconstruction
    geometry_loss_pct: float = 0  # Face loss percentage
    
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
    precheck_skipped: int = 0  # Models skipped because already clean
    reconstructed: int = 0  # NEW: Models reconstructed (significant geometry change)
    
    start_time: str = ""
    last_update: str = ""
    current_file: str = ""
    current_action: str = ""  # Current action being executed
    current_step: int = 0  # Current step number (e.g., 1 of 4)
    total_steps: int = 0  # Total steps in current filter
    
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
                    "success": attempt.slicer_result.success,
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
        
        # Capture reconstruction info from repair loop
        result.is_reconstruction = repair_result.is_reconstruction
        result.reconstruction_method = repair_result.reconstruction_method or ""
        result.geometry_loss_pct = repair_result.geometry_loss_pct
        
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
    images_dir = REPORTS_PATH / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Render images
    before_img = images_dir / f"{stl_path.stem}_before.png"
    after_img = images_dir / f"{stl_path.stem}_after.png"
    
    render_mesh_image(original, before_img, "Before")
    render_mesh_image(repaired, after_img, "After")
    
    # Use the dedicated report generator module
    generate_model_report(
        stl_path=stl_path,
        result=result,
        reports_path=REPORTS_PATH,
        thingi10k_path=THINGI10K_PATH,
        fixed_path=fixed_path,
    )


def generate_dashboard_wrapper(progress: Progress, results: List[TestResult]):
    """Generate HTML dashboard - wrapper for the module function."""
    generate_dashboard_html(progress, results, DASHBOARD_FILE)


def generate_reports_index_wrapper(results: Optional[List[TestResult]] = None):
    """Generate reports index - wrapper for the module function."""
    generate_index_html(results, REPORTS_PATH, FILTERS_PATH)


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
    
    # Initialize quality feedback engine (creates DB if not exists)
    try:
        quality_engine = get_quality_engine()
        quality_stats = quality_engine.get_summary_stats()
        if quality_stats["total_ratings"] > 0:
            print(f"[OK] Quality feedback: {quality_stats['total_ratings']} user ratings", flush=True)
            print(f"     Avg rating: {quality_stats['avg_rating']:.1f}/5, Acceptance: {quality_stats['overall_acceptance_rate']:.0f}%", flush=True)
            logger.info(f"Quality feedback loaded: {quality_stats['total_ratings']} ratings")
        else:
            print("[OK] Quality feedback: No user ratings yet (use --rate to add)", flush=True)
            logger.info("Quality feedback: No ratings yet")
    except Exception as e:
        print(f"[WARN] Quality feedback unavailable: {e}", flush=True)
        logger.warning(f"Quality feedback error: {e}")
    
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
    existing_results = load_results_from_reports(REPORTS_PATH, FILTERS_PATH)
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
    generate_dashboard_wrapper(progress, existing_results)  # Use wrapper function
    
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
            generate_dashboard_wrapper(progress, results)
            generate_reports_index_wrapper(results)
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
        status = "SUCCESS" if result.success else "FAIL"
        escalation = "BLENDER" if result.escalation_used else " "
        print(f"  [{status}] [{escalation}] {result.filter_used} ({result.duration_ms:.0f}ms)", flush=True)
        logger.info(f"  [{status}] [{escalation}] {result.filter_used} ({result.duration_ms:.0f}ms)")
    
    # Final save
    save_progress(progress)
    generate_dashboard_wrapper(progress, results)
    generate_reports_index_wrapper(results)  # Generate navigable index in reports folder
    
    # Summary
    print("", flush=True)
    print("=" * 60, flush=True)
    print("FINAL SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Total processed: {progress.processed:,} models", flush=True)
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
                    print(f"  {adj['threshold']}: {adj['old_value']:.2f} -> {adj['new_value']:.2f}")
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
        
    except ImportError:
        print("\nProfile discovery not available.")
    except Exception as e:
        print(f"Error loading profile discovery stats: {e}")
    
    print("=" * 60)


def show_quality_stats():
    """Display quality feedback statistics."""
    print("=" * 60)
    print("MeshPrep Quality Feedback Statistics")
    print("=" * 60)
    
    try:
        quality_engine = get_quality_engine()
        summary = quality_engine.get_summary_stats()
        
        print(f"\nTotal user ratings: {summary['total_ratings']}")
        
        if summary['total_ratings'] == 0:
            print("\nNo ratings yet. Use --rate to add quality ratings:")
            print("  python run_full_test.py --rate MP:abc123 --rating 4")
            print("  python run_full_test.py --rate MP:abc123 --rating 3 --comment 'Minor issues'")
            print("\nRating scale:")
            print("  5 = Perfect - Indistinguishable from original")
            print("  4 = Good - Minor smoothing/simplification, fully usable")
            print("  3 = Acceptable - Noticeable changes but recognizable")
            print("  2 = Poor - Significant detail loss")
            print("  1 = Rejected - Unrecognizable or destroyed")
            return
        
        print(f"Average rating: {summary['avg_rating']:.2f}/5")
        print(f"Acceptance rate: {summary['overall_acceptance_rate']:.1f}% (ratings >= 3)")
        print(f"Pipeline/profile combinations: {summary['pipeline_profile_combinations']}")
        print(f"Ready for prediction: {'Yes' if summary['ready_for_prediction'] else 'No (need more ratings)'}")
        
        # Rating distribution
        if summary.get('rating_distribution'):
            print("\nRating distribution:")
            for rating in range(5, 0, -1):
                count = summary['rating_distribution'].get(rating, 0)
                bar = "█" * (count // 2 + 1) if count > 0 else ""
                print(f"  {rating} stars: {count:3d} {bar}")
        
        # Get all pipeline stats
        all_stats = quality_engine.get_all_pipeline_quality_stats()
        if all_stats:
            print(f"\nPipeline quality by profile:")
            print(f"  {'Pipeline':<30} {'Profile':<15} {'Avg':>6} {'Count':>6} {'Accept':>7}")
            print(f"  {'-'*30} {'-'*15} {'-'*6} {'-'*6} {'-'*7}")
            for pipeline, profiles in all_stats.items():
                for profile, stats in profiles.items():
                    accept_pct = stats.acceptance_rate * 100
                    print(f"  {pipeline:<30} {profile:<15} {stats.avg_rating:>5.1f} {stats.total_ratings:>6} {accept_pct:>6.0f}%")
        
        print("\n" + "-" * 60)
        print("Quality feedback helps the system learn:")
        print("  1. Which pipelines produce best results for each profile")
        print("  2. When to warn about potential quality issues")
        print("  3. How to reorder pipelines for better results")
        print("\nAdd more ratings to improve pipeline selection!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error loading quality stats: {e}")
        import traceback
        traceback.print_exc()


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
        pipeline_used = "unknown";
        profile = "standard";
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
                    pipeline_used = data.get("filter_script", "unknown");
                    escalated = data.get("escalation_used", False);
                    
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


def reprocess_single_model(model_id: str):
    """Reprocess a specific model by ID.
    
    Finds the model file, deletes existing report/filter/fixed files, and reprocesses.
    
    Args:
        model_id: Model ID (filename without extension, e.g., "100i", "100027")
    """
    print(f"Reprocessing model: {model_id}")
    
    # Find model file in CTM or raw_meshes
    model_path = None
    for ext in SUPPORTED_FORMATS:
        for search_path in [CTM_MESHES_PATH, THINGI10K_PATH]:
            candidate = search_path / f"{model_id}{ext}"
            if candidate.exists():
                model_path = candidate
                break
        if model_path:
            break
    
    if not model_path:
        print(f"[ERROR] Model not found: {model_id}")
        print(f"Searched: {CTM_MESHES_PATH}, {THINGI10K_PATH}")
        return
    
    print(f"Found: {model_path}")
    
    # Delete existing files
    for path, label in [
        (REPORTS_PATH / f"{model_id}.html", "Report"),
        (FILTERS_PATH / f"{model_id}.json", "Filter"),
        (FIXED_OUTPUT_PATH / f"{model_id}.stl", "Fixed"),
    ]:
        if path.exists():
            path.unlink()
            print(f"Deleted: {label}")
    
    # Process
    print("-" * 40)
    result = process_single_model(model_path)
    print("-" * 40)
    
    # Summary
    status = "SUCCESS" if result.success else "FAILED"
    escalation = "BLENDER" if result.escalation_used else " "
    print(f"Result: {status}")
    print(f"Filter: {result.filter_used}")
    print(f"Duration: {result.duration_ms/1000:.1f}s")
    print(f"Fingerprint: {result.model_fingerprint}")
    if result.error:
        print(f"Error: {result.error}")


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
        help="Show current progress"
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
    
    # Reprocess specific model
    parser.add_argument(
        "--reprocess",
        type=str,
        metavar="MODEL_ID",
        help="Reprocess a specific model by ID (e.g., 100i, 100027). Deletes existing report and reprocesses."
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
            try:
                adaptive = get_adaptive_thresholds()
                stats = adaptive.get_stats_summary()
                
                print("=" * 60)
                print("Adaptive Thresholds Statistics")
                print("=" * 60)
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
                        print(f"  {adj['threshold']}: {adj['old_value']:.2f} -> {adj['new_value']:.2f}")
            except Exception as e:
                print(f"Error loading adaptive thresholds: {e}")
        else:
            print("Adaptive thresholds not available.")
        return
    
    if args.quality_stats:
        show_quality_stats()
        return
    
    if args.rate:
        rate_model_by_fingerprint(args.rate, args.rating, args.comment)
        return
    
    if args.reprocess:
        reprocess_single_model(args.reprocess)
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