#!/usr/bin/env python
# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Manage Thingi10K test fixtures.

This script manages the test fixtures directory by creating symlinks to the
raw meshes directory. This avoids duplicating large STL files while keeping
the test structure organized.

If symlinks are not available (requires admin/developer mode on Windows),
it falls back to copying files.

Usage:
    python scripts/manage_fixtures.py build      # Build test fixtures directory
    python scripts/manage_fixtures.py clean      # Remove test fixtures directory
    python scripts/manage_fixtures.py rebuild    # Clean and rebuild
    python scripts/manage_fixtures.py status     # Show current status
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# Default paths
DEFAULT_RAW_MESHES = Path(r"C:\Users\Dragon Ace\Source\repos\Thingi10K\raw_meshes")
DEFAULT_FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "thingi10k"
DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "thingi10k" / "thingi10k.db"


def can_create_symlinks() -> bool:
    """Check if we can create symlinks on this system."""
    test_dir = Path(__file__).parent / ".symlink_test"
    test_link = test_dir / "test_link"
    test_target = test_dir / "test_target"
    
    try:
        test_dir.mkdir(exist_ok=True)
        test_target.touch()
        test_link.symlink_to(test_target)
        
        # Cleanup
        test_link.unlink()
        test_target.unlink()
        test_dir.rmdir()
        return True
    except (OSError, NotImplementedError):
        # Cleanup on failure
        if test_target.exists():
            test_target.unlink()
        if test_dir.exists():
            test_dir.rmdir()
        return False


def build_fixtures(
    raw_meshes_path: Path,
    fixtures_path: Path,
    per_category: int = 20,
    max_faces: int = 100000,
    use_symlinks: bool = True
) -> dict[str, list[int]]:
    """
    Build the test fixtures directory.
    
    Args:
        raw_meshes_path: Path to Thingi10K raw_meshes directory
        fixtures_path: Path to test fixtures directory
        per_category: Number of fixtures per category
        max_faces: Maximum face count for fixtures
        use_symlinks: Use symlinks instead of copying (if available)
    
    Returns:
        Dictionary mapping category -> list of file_ids
    """
    # Force reload of the module to get latest code
    import importlib
    import meshprep.testing.thingi10k_manager as mgr
    importlib.reload(mgr)
    
    db = mgr.Thingi10KDatabase()
    
    if db.count_models() == 0:
        logger.error("Database is empty. Run 'python scripts/setup_thingi10k.py' first.")
        sys.exit(1)
    
    # Check if symlinks are available
    if use_symlinks:
        if can_create_symlinks():
            logger.info("Using symlinks (no file duplication)")
            link_mode = "symlink"
        else:
            logger.warning("Symlinks not available. Run as admin or enable Developer Mode on Windows.")
            logger.info("Falling back to copying files...")
            link_mode = "copy"
    else:
        link_mode = "copy"
    
    # Use the manager's select method which has proper queries
    manager = mgr.Thingi10KManager(raw_meshes_path=raw_meshes_path)
    
    logger.info(f"\nBuilding fixtures ({per_category} per category, max {max_faces} faces)")
    logger.info(f"Source: {raw_meshes_path}")
    logger.info(f"Destination: {fixtures_path}")
    logger.info(f"Mode: {link_mode}\n")
    
    # Select fixtures using the manager's improved query logic
    selected = manager.select_test_fixtures(
        per_category=per_category,
        max_faces=max_faces
    )
    
    total_files = 0
    total_size = 0
    
    logger.info("\nCopying files:")
    for category, file_ids in selected.items():
        if not file_ids:
            continue
        
        cat_dir = fixtures_path / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        
        cat_size = 0
        copied = 0
        
        for file_id in file_ids:
            src = raw_meshes_path / f"{file_id}.stl"
            dst = cat_dir / f"{file_id}.stl"
            
            if not src.exists():
                continue
            
            if dst.exists():
                dst.unlink()  # Remove existing
            
            if link_mode == "symlink":
                dst.symlink_to(src)
            else:
                shutil.copy2(src, dst)
            
            copied += 1
            cat_size += src.stat().st_size
        
        total_files += copied
        total_size += cat_size
        
        size_mb = cat_size / (1024 * 1024)
        logger.info(f"  {category}: {copied} models ({size_mb:.1f} MB)")
    
    # Save index file
    index_path = fixtures_path / "index.json"
    index_data = {
        "source": str(raw_meshes_path),
        "mode": link_mode,
        "per_category": per_category,
        "max_faces": max_faces,
        "categories": selected,
        "total_files": total_files
    }
    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=2)
    
    total_mb = total_size / (1024 * 1024)
    logger.info(f"\nTotal: {total_files} fixtures ({total_mb:.1f} MB)")
    logger.info(f"Index saved to: {index_path}")
    
    if link_mode == "symlink":
        logger.info("\nNote: Fixtures are symlinks. The actual files remain in the raw_meshes directory.")
    
    return selected


def clean_fixtures(fixtures_path: Path):
    """Remove the test fixtures directory."""
    if fixtures_path.exists():
        logger.info(f"Removing: {fixtures_path}")
        shutil.rmtree(fixtures_path)
        logger.info("Done.")
    else:
        logger.info("Fixtures directory does not exist.")


def show_status(fixtures_path: Path, raw_meshes_path: Path):
    """Show current status of fixtures."""
    logger.info("Thingi10K Fixtures Status")
    logger.info("=" * 50)
    
    # Check raw meshes
    if raw_meshes_path.exists():
        stl_count = len(list(raw_meshes_path.glob("*.stl")))
        logger.info(f"Raw meshes: {raw_meshes_path}")
        logger.info(f"  STL files: {stl_count}")
    else:
        logger.info(f"Raw meshes: NOT FOUND at {raw_meshes_path}")
    
    # Check database
    db_path = DEFAULT_DB_PATH
    if db_path.exists():
        from meshprep.testing import Thingi10KDatabase
        db = Thingi10KDatabase()
        logger.info(f"\nDatabase: {db_path}")
        logger.info(f"  Models: {db.count_models()}")
        logger.info(f"  Size: {db_path.stat().st_size / 1024:.1f} KB")
    else:
        logger.info(f"\nDatabase: NOT FOUND at {db_path}")
    
    # Check fixtures
    logger.info(f"\nFixtures: {fixtures_path}")
    if fixtures_path.exists():
        index_path = fixtures_path / "index.json"
        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)
            
            logger.info(f"  Mode: {index.get('mode', 'unknown')}")
            logger.info(f"  Total files: {index.get('total_files', 0)}")
            logger.info(f"  Categories:")
            for cat, ids in index.get("categories", {}).items():
                logger.info(f"    {cat}: {len(ids)}")
        else:
            # Count manually
            total = 0
            for cat_dir in fixtures_path.iterdir():
                if cat_dir.is_dir():
                    count = len(list(cat_dir.glob("*.stl")))
                    logger.info(f"    {cat_dir.name}: {count}")
                    total += count
            logger.info(f"  Total: {total}")
    else:
        logger.info("  NOT FOUND (run 'build' to create)")


def main():
    parser = argparse.ArgumentParser(
        description="Manage Thingi10K test fixtures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  build     Build test fixtures directory from database
  clean     Remove test fixtures directory
  rebuild   Clean and rebuild fixtures
  status    Show current status

Examples:
  python scripts/manage_fixtures.py build
  python scripts/manage_fixtures.py build --per-category 50 --max-faces 50000
  python scripts/manage_fixtures.py rebuild --no-symlinks
  python scripts/manage_fixtures.py clean
"""
    )
    
    parser.add_argument(
        "command",
        choices=["build", "clean", "rebuild", "status"],
        help="Command to execute"
    )
    parser.add_argument(
        "--raw-meshes",
        type=Path,
        default=DEFAULT_RAW_MESHES,
        help="Path to Thingi10K raw_meshes directory"
    )
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=DEFAULT_FIXTURES_DIR,
        help="Path to test fixtures directory"
    )
    parser.add_argument(
        "--per-category",
        type=int,
        default=20,
        help="Number of fixtures per category (default: 20)"
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=100000,
        help="Maximum face count for fixtures (default: 100000)"
    )
    parser.add_argument(
        "--no-symlinks",
        action="store_true",
        help="Copy files instead of creating symlinks"
    )
    
    args = parser.parse_args()
    
    if args.command == "status":
        show_status(args.fixtures_dir, args.raw_meshes)
    
    elif args.command == "clean":
        clean_fixtures(args.fixtures_dir)
    
    elif args.command == "build":
        build_fixtures(
            args.raw_meshes,
            args.fixtures_dir,
            args.per_category,
            args.max_faces,
            use_symlinks=not args.no_symlinks
        )
    
    elif args.command == "rebuild":
        clean_fixtures(args.fixtures_dir)
        build_fixtures(
            args.raw_meshes,
            args.fixtures_dir,
            args.per_category,
            args.max_faces,
            use_symlinks=not args.no_symlinks
        )


if __name__ == "__main__":
    main()
