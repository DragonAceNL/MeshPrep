#!/usr/bin/env python
# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Setup script for Thingi10K test fixtures.

This script:
1. Installs the thingi10k package if needed
2. Imports metadata into the SQLite database
3. Selects test fixtures for each category
4. Optionally copies fixtures to the test directory

Usage:
    python scripts/setup_thingi10k.py
    python scripts/setup_thingi10k.py --copy-fixtures
    python scripts/setup_thingi10k.py --raw-meshes "D:\Thingi10K\raw_meshes"
"""

import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def install_thingi10k_package():
    """Install the thingi10k package if not already installed."""
    try:
        import thingi10k
        logger.info("✓ thingi10k package already installed")
        return True
    except ImportError:
        logger.info("Installing thingi10k package...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "thingi10k"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            logger.info("✓ thingi10k package installed successfully")
            return True
        else:
            logger.error(f"✗ Failed to install thingi10k: {result.stderr}")
            return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup Thingi10K test fixtures")
    parser.add_argument(
        "--raw-meshes",
        type=Path,
        default=Path(r"C:\Users\Dragon Ace\Source\repos\Thingi10K\raw_meshes"),
        help="Path to Thingi10K raw_meshes directory"
    )
    parser.add_argument(
        "--copy-fixtures",
        action="store_true",
        help="Copy selected fixtures to test directory"
    )
    parser.add_argument(
        "--per-category",
        type=int,
        default=20,
        help="Number of test fixtures per category"
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=100000,
        help="Maximum face count for test fixtures"
    )
    parser.add_argument(
        "--use-directory-scan",
        action="store_true",
        help="Scan STL files instead of using thingi10k package"
    )
    
    args = parser.parse_args()
    
    # Step 1: Install thingi10k package
    if not args.use_directory_scan:
        if not install_thingi10k_package():
            logger.warning("Falling back to directory scan method")
            args.use_directory_scan = True
    
    # Step 2: Import metadata
    # Import here to allow for package installation
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from meshprep.testing import Thingi10KManager
    
    logger.info(f"\nRaw meshes path: {args.raw_meshes}")
    
    manager = Thingi10KManager(raw_meshes_path=args.raw_meshes)
    
    # Check if database already has data
    existing_count = manager.db.count_models()
    if existing_count > 0:
        logger.info(f"Database already contains {existing_count} models")
        response = input("Re-import metadata? [y/N]: ").strip().lower()
        if response != "y":
            logger.info("Skipping import")
        else:
            if args.use_directory_scan:
                count = manager.import_from_directory()
            else:
                count = manager.import_from_thingi10k_package()
            logger.info(f"✓ Imported {count} models")
    else:
        logger.info("\nImporting metadata...")
        if args.use_directory_scan:
            count = manager.import_from_directory()
        else:
            count = manager.import_from_thingi10k_package()
        logger.info(f"✓ Imported {count} models")
    
    # Step 3: Print summary
    logger.info("\n")
    manager.print_summary()
    
    # Step 4: Select test fixtures
    logger.info(f"\nSelecting test fixtures ({args.per_category} per category, max {args.max_faces} faces)...")
    selected = manager.select_test_fixtures(
        per_category=args.per_category,
        max_faces=args.max_faces
    )
    
    total_selected = sum(len(ids) for ids in selected.values())
    logger.info(f"✓ Selected {total_selected} fixtures across {len(selected)} categories")
    
    # Step 5: Copy fixtures if requested
    if args.copy_fixtures:
        logger.info("\nCopying fixtures to test directory...")
        manager.copy_fixtures_to_test_dir(selected, overwrite=False)
        logger.info(f"✓ Fixtures copied to {manager.fixtures_path}")
    else:
        logger.info("\nTo copy fixtures, run with --copy-fixtures flag")
    
    # Step 6: Export to JSON
    json_path = manager.db.db_path.parent / "thingi10k_metadata.json"
    response = input(f"\nExport metadata to JSON ({json_path})? [y/N]: ").strip().lower()
    if response == "y":
        manager.db.export_to_json(json_path)
        logger.info(f"✓ Exported to {json_path}")
    
    logger.info("\n✓ Setup complete!")
    logger.info(f"\nDatabase location: {manager.db.db_path}")
    logger.info(f"Fixtures location: {manager.fixtures_path}")


if __name__ == "__main__":
    main()
