# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Batch processing runner for POC v3.

Contains the main batch processing loop and helper functions for
running repairs against multiple mesh files.
"""

import csv
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List

# Add POC v2 to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "v2"))

from meshprep_poc.actions.blender_actions import is_blender_available
from meshprep_poc.learning_engine import get_learning_engine
from meshprep_poc.quality_feedback import get_quality_engine

from config import (
    THINGI10K_PATH, CTM_MESHES_PATH,
    REPORTS_PATH, FILTERS_PATH,
    PROGRESS_FILE, SUMMARY_FILE, RESULTS_CSV, DASHBOARD_FILE,
    SUPPORTED_FORMATS,
    DASHBOARD_UPDATE_INTERVAL, THRESHOLD_OPTIMIZE_INTERVAL, PROFILE_DISCOVERY_INTERVAL,
)
from test_result import TestResult
from progress_tracker import Progress, save_progress
from mesh_utils import ADAPTIVE_THRESHOLDS_AVAILABLE
from model_processor import process_single_model, set_progress_file
from report_generator import generate_model_report
from dashboard_generator import generate_dashboard as generate_dashboard_html
from index_generator import generate_reports_index as generate_index_html, load_results_from_reports

if ADAPTIVE_THRESHOLDS_AVAILABLE:
    from meshprep_poc.adaptive_thresholds import get_adaptive_thresholds

# Try to import profile discovery
try:
    from meshprep_poc.profile_discovery import get_discovery_engine
    PROFILE_DISCOVERY_AVAILABLE = True
except ImportError:
    PROFILE_DISCOVERY_AVAILABLE = False
    get_discovery_engine = None

logger = logging.getLogger(__name__)


def get_all_mesh_files(limit: Optional[int] = None, ctm_priority: bool = False) -> List[Path]:
    """Get all mesh files from Thingi10K and CTM meshes.
    
    Args:
        limit: Optional limit on number of files to return
        ctm_priority: If True, CTM meshes are listed FIRST (before other files)
        
    Returns:
        List of paths to mesh files
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
        ctm_files = sorted(ctm_files, key=lambda p: p.stem)
        thingi_files = sorted(thingi_files, key=lambda p: p.stem)
        mesh_files = ctm_files + thingi_files
    else:
        mesh_files = sorted(thingi_files + ctm_files, key=lambda p: p.stem)
    
    if limit:
        mesh_files = mesh_files[:limit]
    
    return mesh_files


def get_processed_files() -> set:
    """Get set of already processed file IDs."""
    processed = set()
    for html_file in REPORTS_PATH.glob("*.html"):
        if html_file.stem != "index":
            processed.add(html_file.stem)
    return processed


def append_to_csv(result: TestResult) -> None:
    """Append a single result to the CSV file."""
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
    
    write_header = not RESULTS_CSV.exists()
    
    with open(RESULTS_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        
        if write_header:
            writer.writeheader()
        
        row = {
            'model_fingerprint': result.model_fingerprint,
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


def run_batch_test(
    limit: Optional[int] = None, 
    skip_existing: bool = True, 
    ctm_priority: bool = False, 
    auto_optimize: bool = True
) -> None:
    """Run the full batch test.
    
    Args:
        limit: Optional limit on number of files to process
        skip_existing: If True (default), skip files that already have reports
        ctm_priority: If True, process CTM meshes FIRST before other files
        auto_optimize: If True (default), automatically optimize thresholds during processing
    """
    # Set the progress file for the model processor
    set_progress_file(PROGRESS_FILE)
    
    print("=" * 60, flush=True)
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
    
    # Initialize quality feedback engine
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
    
    # Initialize adaptive thresholds
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
    
    # Load existing results from reports
    existing_results = load_results_from_reports(REPORTS_PATH, FILTERS_PATH)
    logger.info(f"Loaded {len(existing_results)} existing results from reports")
    
    # Initialize progress
    progress = Progress(
        total_files=total_mesh_count,
        processed=len(processed_ids),
        successful=len(processed_ids),
        start_time=datetime.now().isoformat(),
    )
    
    # Save initial progress
    save_progress(progress, PROGRESS_FILE)
    generate_dashboard_html(progress, existing_results, DASHBOARD_FILE)
    
    # Start with existing results
    results: List[TestResult] = list(existing_results)
    new_results_count = 0
    
    # Process each file
    for i, stl_path in enumerate(to_process):
        file_id = stl_path.stem
        
        # Update progress
        progress.current_file = file_id
        progress.processed += 1
        
        # Calculate ETA
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
        
        # Append to CSV
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
        
        # Save progress after every file
        save_progress(progress, PROGRESS_FILE)
        
        # Update dashboard periodically
        if i % DASHBOARD_UPDATE_INTERVAL == 0:
            generate_dashboard_html(progress, results, DASHBOARD_FILE)
            generate_index_html(results, REPORTS_PATH, FILTERS_PATH)
            try:
                from generate_learning_status import generate_learning_status_page
                generate_learning_status_page()
            except Exception as e:
                logger.debug(f"Failed to generate learning status page: {e}")
        
        # Auto-optimize thresholds periodically
        if auto_optimize and ADAPTIVE_THRESHOLDS_AVAILABLE and new_results_count > 0 and new_results_count % THRESHOLD_OPTIMIZE_INTERVAL == 0:
            try:
                adaptive = get_adaptive_thresholds()
                adjustments = adaptive.optimize_thresholds(min_samples=20)
                if adjustments:
                    logger.info(f"Auto-optimized {len(adjustments)} thresholds after {new_results_count} models")
                    for adj in adjustments:
                        logger.info(f"  {adj['threshold']}: {adj['old_value']:.2f} -> {adj['new_value']:.2f}")
            except Exception as e:
                logger.debug(f"Auto-optimization failed: {e}")
        
        # Auto-run profile discovery periodically
        if auto_optimize and PROFILE_DISCOVERY_AVAILABLE and new_results_count > 0 and new_results_count % PROFILE_DISCOVERY_INTERVAL == 0:
            try:
                discovery = get_discovery_engine()
                discovered = discovery.run_discovery(min_samples=50)
                if discovered:
                    logger.info(f"Auto-discovered {len(discovered)} new profiles after {new_results_count} models")
                    for profile in discovered:
                        logger.info(f"  {profile.name}: {profile.total_models} models, {profile.success_rate*100:.0f}% success")
            except Exception as e:
                logger.debug(f"Auto-discovery failed: {e}")
        
        # Log result
        status = "SUCCESS" if result.success else "FAIL"
        escalation = "BLENDER" if result.escalation_used else " "
        print(f"  [{status}] [{escalation}] {result.filter_used} ({result.duration_ms:.0f}ms)", flush=True)
        logger.info(f"  [{status}] [{escalation}] {result.filter_used} ({result.duration_ms:.0f}ms)")
    
    # Final save
    save_progress(progress, PROGRESS_FILE)
    generate_dashboard_html(progress, results, DASHBOARD_FILE)
    generate_index_html(results, REPORTS_PATH, FILTERS_PATH)
    
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
    
    # Final profile discovery
    if auto_optimize and PROFILE_DISCOVERY_AVAILABLE and new_results_count >= 50:
        try:
            discovery = get_discovery_engine()
            stats = discovery.get_stats_summary()
            print(f"\nProfile discovery: {stats['total_clusters']} clusters, {stats['active_profiles']} profiles", flush=True)
            
            # Run discovery on any remaining unassigned clusters
            if stats['unassigned_clusters'] > 0:
                discovered = discovery.run_discovery(min_samples=50)
                if discovered:
                    print(f"Auto-discovered {len(discovered)} new profiles:", flush=True)
                    for profile in discovered:
                        print(f"  {profile.name}: {profile.total_models} models, {profile.success_rate*100:.0f}% success", flush=True)
                else:
                    print(f"No new profiles discovered ({stats['unassigned_models']} models in {stats['unassigned_clusters']} clusters awaiting more data)", flush=True)
            else:
                print(f"All clusters assigned to profiles", flush=True)
        except Exception as e:
            logger.debug(f"Final profile discovery failed: {e}")
    
    # Save final summary
    with open(SUMMARY_FILE, "w") as f:
        json.dump({
            "progress": asdict(progress),
            "results": [asdict(r) for r in results[-100:]],
        }, f, indent=2)
