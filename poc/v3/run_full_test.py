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
    python run_full_test.py --error-stats      # Show error/crash statistics
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
import logging
import sys
from datetime import datetime
from pathlib import Path

# Import local modules
from config import LOG_DIR
from colored_logging import ColoredFormatter

# Setup logging with file output
LOG_DIR.mkdir(exist_ok=True)

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
        "--error-stats",
        action="store_true",
        help="Show error/crash statistics from the error log"
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
    
    # Version tracking options
    parser.add_argument(
        "--version-info",
        action="store_true",
        help="Show version tracking information"
    )
    parser.add_argument(
        "--reset-skips",
        action="store_true",
        help="Reset skip recommendations for current version (allows retrying failed actions)"
    )
    parser.add_argument(
        "--reset-skips-confirm",
        action="store_true",
        help="Reset skip recommendations without confirmation prompt"
    )
    parser.add_argument(
        "--check-version",
        action="store_true",
        help="Check if software version has changed"
    )
    
    args = parser.parse_args()
    
    # Import CLI commands (deferred to avoid circular imports)
    from cli_commands import (
        show_status, show_learning_stats, show_quality_stats,
        show_threshold_stats, rate_model_by_fingerprint, reprocess_single_model,
        show_error_stats, show_version_info, reset_skips, check_version_change,
    )
    from batch_runner import run_batch_test
    
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
    
    if args.error_stats:
        show_error_stats()
        return
    
    if args.version_info:
        show_version_info()
        return
    
    if args.reset_skips or args.reset_skips_confirm:
        reset_skips(confirm=args.reset_skips_confirm)
        return
    
    if args.check_version:
        check_version_change()
        return
    
    if args.discover_profiles:
        # TODO: implement run_profile_discovery
        print("Profile discovery not yet implemented in modular version")
        return
    
    if args.optimize_thresholds:
        # TODO: implement optimize_adaptive_thresholds
        print("Threshold optimization not yet implemented in modular version")
        return
    
    if args.reset_thresholds:
        # TODO: implement reset_adaptive_thresholds
        print("Threshold reset not yet implemented in modular version")
        return
    
    if args.threshold_stats:
        show_threshold_stats()
        return
    
    if args.quality_stats:
        show_quality_stats()
        return
    
    if args.rate:
        if args.rating is None:
            print("Error: --rating is required when using --rate")
            return
        rate_model_by_fingerprint(args.rate, args.rating, args.comment)
        return
    
    if args.reprocess:
        reprocess_single_model(args.reprocess)
        return
    
    # If --fresh is passed, reprocess all files
    if args.fresh:
        logger.info("Fresh mode: will reprocess all files (existing results preserved until overwritten)")
    
    run_batch_test(
        limit=args.limit,
        skip_existing=not args.fresh,
        auto_optimize=not args.no_auto_optimize,
    )


if __name__ == "__main__":
    main()