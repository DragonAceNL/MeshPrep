# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
CLI utility for managing MeshPrep learning data and error tracking.

Usage:
    python manage_learning.py --error-stats       Show error statistics
    python manage_learning.py --version-info      Show version tracking info
    python manage_learning.py --reset-skips       Reset skip recommendations for current version
    python manage_learning.py --learning-stats    Show learning engine statistics
    python manage_learning.py --check-version     Check for version changes
"""

import argparse
import sys
from pathlib import Path

# Add meshprep_poc to path
sys.path.insert(0, str(Path(__file__).parent))

from meshprep_poc import __version__
from meshprep_poc.subprocess_executor import (
    get_failure_tracker,
    get_pymeshlab_version,
    get_meshprep_version,
)


def show_error_stats():
    """Show error/failure statistics."""
    print("=" * 60)
    print("MeshPrep Error Statistics")
    print("=" * 60)
    print()
    
    tracker = get_failure_tracker()
    stats = tracker.get_failure_stats()
    
    print(f"Total failures recorded: {stats['total_failures']}")
    print()
    
    if stats['by_category']:
        print("Failures by category:")
        for item in stats['by_category'][:10]:
            print(f"  {item['category']}: {item['count']}")
        print()
    
    if stats['patterns']:
        print("Failure patterns (actions being skipped):")
        skip_count = 0
        for p in stats['patterns'][:15]:
            skip_marker = " [SKIP]" if p['skip'] else ""
            if p['skip']:
                skip_count += 1
            print(f"  {p['action']} ({p['category']}, {p['size_bin']}): "
                  f"{p['failures']} fails, {p['successes']} ok{skip_marker}")
        print()
        print(f"Total patterns marked as skip: {skip_count}")


def show_version_info():
    """Show version tracking information."""
    print("=" * 60)
    print("MeshPrep Version Tracking")
    print("=" * 60)
    print()
    
    tracker = get_failure_tracker()
    info = tracker.get_version_info()
    
    print("Current version:")
    print(f"  MeshPrep:  {info['current_version']['meshprep']}")
    print(f"  PyMeshLab: {info['current_version']['pymeshlab']}")
    print(f"  Python:    {info['current_version']['python']}")
    print()
    
    if info['version_history']:
        print("Version history:")
        for vh in info['version_history'][:5]:
            print(f"  MeshPrep {vh['meshprep']}, PyMeshLab {vh['pymeshlab']}, "
                  f"Python {vh['python']} - first seen: {vh['first_seen']}")
        print()
    
    if info['skips_by_version']:
        print("Skip recommendations by version:")
        for sv in info['skips_by_version']:
            print(f"  MeshPrep {sv['meshprep']}, PyMeshLab {sv['pymeshlab']}: "
                  f"{sv['skip_count']} skips")
        print()
    else:
        print("No skip recommendations recorded yet.")
        print()
    
    print("Note: Skip recommendations are VERSION-SPECIFIC.")
    print("When you upgrade MeshPrep or PyMeshLab, old skip recommendations")
    print("won't apply - giving the new version a fresh chance to fix bugs.")


def reset_skips():
    """Reset skip recommendations for current version."""
    print("=" * 60)
    print("Reset Skip Recommendations")
    print("=" * 60)
    print()
    
    tracker = get_failure_tracker()
    info = tracker.get_version_info()
    
    print(f"Current version: MeshPrep {info['current_version']['meshprep']}, "
          f"PyMeshLab {info['current_version']['pymeshlab']}")
    print()
    
    # Show current skips for this version
    current_skips = [
        sv for sv in info['skips_by_version']
        if sv['meshprep'] == info['current_version']['meshprep'] 
        and sv['pymeshlab'] == info['current_version']['pymeshlab']
    ]
    
    if not current_skips or current_skips[0]['skip_count'] == 0:
        print("No skip recommendations for current version. Nothing to reset.")
        return
    
    skip_count = current_skips[0]['skip_count']
    print(f"Found {skip_count} skip recommendations for current version.")
    print()
    
    # Confirm
    response = input("Reset all skip recommendations? This will retry previously failing actions. [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Reset
    count = tracker.reset_skips_for_current_version()
    print()
    print(f"Reset {count} skip recommendations.")
    print("Previously skipped actions will now be retried on new models.")


def check_version_change():
    """Check if version has changed and report."""
    print("=" * 60)
    print("Version Change Check")
    print("=" * 60)
    print()
    
    tracker = get_failure_tracker()
    
    print(f"Current version: MeshPrep {get_meshprep_version()}, "
          f"PyMeshLab {get_pymeshlab_version()}")
    print()
    
    changed = tracker.check_version_change()
    
    if changed:
        print("✓ Version change detected!")
        print("  Skip recommendations from older versions won't apply.")
        print("  Actions that failed before will get a fresh chance.")
    else:
        print("No version change detected.")
        print("Using existing skip recommendations for this version.")


def show_learning_stats():
    """Show learning engine statistics."""
    print("=" * 60)
    print("MeshPrep Learning Engine Statistics")
    print("=" * 60)
    print()
    
    try:
        from meshprep_poc.learning_engine import get_learning_engine
        engine = get_learning_engine()
        stats = engine.get_statistics()
        
        print(f"Models processed: {stats.get('total_models', 0)}")
        print(f"Pipelines tracked: {stats.get('total_pipelines', 0)}")
        print(f"Issue patterns: {stats.get('total_issue_patterns', 0)}")
        print()
        
        if stats.get('top_pipelines'):
            print("Top performing pipelines:")
            for p in stats['top_pipelines'][:5]:
                success_rate = p.get('success_rate', 0) * 100
                avg_time = p.get('avg_duration_ms', 0)
                print(f"  {p['name']}: {success_rate:.1f}% success, {avg_time:.0f}ms avg")
            print()
        
        if stats.get('profiles'):
            print("Profile statistics:")
            for profile, data in list(stats['profiles'].items())[:5]:
                rate = data.get('success_rate', 0) * 100
                count = data.get('count', 0)
                print(f"  {profile}: {rate:.1f}% success ({count} models)")
            print()
            
    except Exception as e:
        print(f"Could not load learning engine: {e}")
        print("Learning data may not exist yet. Run some repairs first.")


def main():
    parser = argparse.ArgumentParser(
        description="MeshPrep Learning Data Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manage_learning.py --error-stats       Show error statistics
  python manage_learning.py --version-info      Show version tracking info  
  python manage_learning.py --reset-skips       Reset skip recommendations
  python manage_learning.py --learning-stats    Show learning engine stats
  python manage_learning.py --check-version     Check for version changes
        """
    )
    
    parser.add_argument(
        "--error-stats",
        action="store_true",
        help="Show error/failure statistics"
    )
    parser.add_argument(
        "--version-info",
        action="store_true", 
        help="Show version tracking information"
    )
    parser.add_argument(
        "--reset-skips",
        action="store_true",
        help="Reset skip recommendations for current version"
    )
    parser.add_argument(
        "--check-version",
        action="store_true",
        help="Check if software version has changed"
    )
    parser.add_argument(
        "--learning-stats",
        action="store_true",
        help="Show learning engine statistics"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"MeshPrep POC v2 {__version__}"
    )
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if not any([args.error_stats, args.version_info, args.reset_skips, 
                args.check_version, args.learning_stats]):
        parser.print_help()
        return
    
    # Execute requested commands
    if args.error_stats:
        show_error_stats()
    
    if args.version_info:
        show_version_info()
    
    if args.reset_skips:
        reset_skips()
    
    if args.check_version:
        check_version_change()
    
    if args.learning_stats:
        show_learning_stats()


if __name__ == "__main__":
    main()
