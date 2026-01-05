# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep � https://github.com/DragonAceNL/MeshPrep

"""
CLI display and interaction commands for POC v3.

Contains functions for displaying statistics, status information,
and handling user interactions like rating models.
"""

import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Optional

# Add POC v2 to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "v2"))

from meshprep_poc.learning_engine import get_learning_engine
from meshprep_poc.quality_feedback import get_quality_engine, QualityRating

from config import (
    REPORTS_PATH, FILTERS_PATH, FIXED_OUTPUT_PATH,
    THINGI10K_PATH, SUPPORTED_FORMATS,
)
from progress_tracker import load_progress
from mesh_utils import ADAPTIVE_THRESHOLDS_AVAILABLE

if ADAPTIVE_THRESHOLDS_AVAILABLE:
    from meshprep_poc.adaptive_thresholds import get_adaptive_thresholds


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
    print(f"Live Dashboard: http://localhost:8000/live")
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
                bar = "?" * (count // 2 + 1) if count > 0 else ""
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


def show_threshold_stats():
    """Show adaptive thresholds statistics only."""
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


def rate_model_by_fingerprint(fingerprint: str, rating: int, comment: Optional[str] = None):
    """Rate a model by its fingerprint.
    
    Args:
        fingerprint: Model fingerprint (MP:xxxx format)
        rating: Quality rating 1-5
        comment: Optional comment
    """
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


def reprocess_single_model(model_id: str):
    """Reprocess a specific model by ID.
    
    Finds the model file, deletes existing report/filter/fixed files, and reprocesses.
    
    Args:
        model_id: Model ID (filename without extension, e.g., "100i", "100027")
    """
    # Import here to avoid circular import
    from model_processor import process_single_model
    
    print(f"Reprocessing model: {model_id}")
    
    # Find model file in CTM or raw_meshes
    model_path = None
    for ext in SUPPORTED_FORMATS:
        for search_path in [THINGI10K_PATH]:
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


def show_error_stats():
    """Show error/crash statistics from the error log."""
    try:
        from meshprep_poc.error_logger import (
            get_error_summary,
            get_error_log_path,
            get_all_error_logs,
        )
        from meshprep_poc.subprocess_executor import (
            get_failure_tracker,
            get_crash_tracker,
        )
    except ImportError as e:
        print(f"[ERROR] Error logging not available: {e}")
        return
    
    print("\n" + "=" * 60)
    print("MeshPrep Error/Crash Statistics")
    print("=" * 60)
    
    # Get error log summary for today
    log_path = get_error_log_path()
    if log_path.exists():
        print(f"\n[LOG] Today's Error Log: {log_path.name}")
        summary = get_error_summary(log_path)
        print(f"   Total errors: {summary['total']}")
        
        if summary['by_category']:
            print("\n   By Category:")
            for cat, count in summary['by_category'].items():
                print(f"      {cat}: {count}")
        
        if summary['by_action']:
            print("\n   By Action:")
            for action, count in list(summary['by_action'].items())[:10]:
                print(f"      {action}: {count}")
        
        if summary['by_type']:
            print("\n   By Type:")
            for ftype, count in summary['by_type'].items():
                print(f"      {ftype}: {count}")
    else:
        print("\n[LOG] No error log for today")
    
    # Show all available log files
    all_logs = get_all_error_logs()
    if len(all_logs) > 1:
        print(f"\n[FILES] Available Logs ({len(all_logs)} total):")
        for log in all_logs[:5]:
            print(f"   {log.name}")
        if len(all_logs) > 5:
            print(f"   ... and {len(all_logs) - 5} more")
    
    # Get failure tracker stats (from SQLite DB)
    print("\n" + "-" * 60)
    print("Failure Pattern Learning (SQLite DB)")
    print("-" * 60)
    
    tracker = get_failure_tracker()
    failure_stats = tracker.get_failure_stats()
    
    print(f"\n[STATS] Total failures tracked: {failure_stats['total_failures']}")
    
    if failure_stats['by_category']:
        print("\n   Failures by Category:")
        for item in failure_stats['by_category'][:5]:
            print(f"      {item['category']}: {item['count']}")
    
    if failure_stats['patterns']:
        print("\n   Learned Patterns (skip recommendations):")
        for pattern in failure_stats['patterns'][:10]:
            skip_marker = "[SKIP]" if pattern['skip'] else "[WARN]"
            rate = pattern['failures'] / (pattern['failures'] + pattern['successes']) * 100 if pattern['failures'] + pattern['successes'] > 0 else 0
            print(f"      {skip_marker} {pattern['action']} + {pattern['category']} on {pattern['size_bin']}: {rate:.0f}% fail")
            if pattern['reason']:
                print(f"         Reason: {pattern['reason']}")
    
    # Get crash tracker stats
    crash_tracker = get_crash_tracker()
    crash_stats = crash_tracker.get_crash_stats()
    
    print("\n" + "-" * 60)
    print("Crash Tracking (Process Crashes)")
    print("-" * 60)
    print(f"\n[CRASH] Total crashes: {crash_stats['total_crashes']}")
    
    if crash_stats['patterns']:
        print("\n   Crash Patterns:")
        for pattern in crash_stats['patterns'][:10]:
            skip_marker = "[SKIP]" if pattern['skip'] else "[WARN]"
            print(f"      {skip_marker} {pattern['action']} on {pattern['size_bin']}: {pattern['crashes']} crashes, {pattern['successes']} successes")
    
    print("\n" + "=" * 60 + "\n")
