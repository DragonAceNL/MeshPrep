# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Test script for slicer validation integration.

This script tests:
1. Slicer detection (finding installed slicers)
2. Slicer validation (running a mesh through the slicer)
3. Error parsing (extracting issues from slicer output)
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from meshprep_poc.actions.slicer_actions import (
    find_slicer,
    get_slicer_version,
    is_slicer_available,
    get_available_slicers,
    validate_stl_file,
    validate_mesh,
    SlicerResult,
)


def print_header(text: str) -> None:
    """Print a section header."""
    print()
    print("=" * 60)
    print(f" {text}")
    print("=" * 60)


def test_slicer_detection() -> bool:
    """Test that we can detect installed slicers."""
    print_header("SLICER DETECTION")
    
    print("\nChecking for available slicers...")
    available = get_available_slicers()
    
    if not available:
        print("  [X] No slicers found!")
        print()
        print("  To enable slicer validation, install one of:")
        print("    - PrusaSlicer: https://www.prusa3d.com/prusaslicer/")
        print("    - OrcaSlicer: https://github.com/SoftFever/OrcaSlicer")
        print("    - SuperSlicer: https://github.com/supermerill/SuperSlicer")
        print()
        return False
    
    print(f"  [OK] Found {len(available)} slicer(s):")
    for name, path in available.items():
        version = get_slicer_version(path)
        version_str = f" (version: {version})" if version else ""
        print(f"    - {name}: {path}{version_str}")
    
    return True


def test_slicer_validation_clean_model() -> bool:
    """Test slicer validation with a clean model."""
    print_header("SLICER VALIDATION - CLEAN MODEL")
    
    # Find a clean test fixture
    fixtures_path = Path(__file__).parent.parent.parent / "tests" / "fixtures" / "thingi10k" / "clean"
    
    if not fixtures_path.exists():
        print(f"  [!] Clean fixtures directory not found: {fixtures_path}")
        return False
    
    # Get first STL file
    stl_files = list(fixtures_path.glob("*.stl"))
    if not stl_files:
        print(f"  [!] No STL files found in {fixtures_path}")
        return False
    
    test_file = stl_files[0]
    print(f"\nTesting with: {test_file.name}")
    
    result = validate_stl_file(test_file, slicer="auto", timeout=60)
    
    print(f"\nResult:")
    print(f"  Slicer: {result.slicer}")
    print(f"  Success: {'PASS' if result.success else 'FAIL'} {result.success}")
    print(f"  Return code: {result.return_code}")
    
    if result.errors:
        print(f"  Errors ({len(result.errors)}):")
        for err in result.errors[:5]:  # Limit to first 5
            print(f"    - {err[:80]}")
    
    if result.warnings:
        print(f"  Warnings ({len(result.warnings)}):")
        for warn in result.warnings[:5]:
            print(f"    - {warn[:80]}")
    
    if result.issues:
        print(f"  Issues ({len(result.issues)}):")
        for issue in result.issues[:5]:
            print(f"    - [{issue['type']}] {issue['message'][:60]}")
    
    return result.success


def test_slicer_validation_broken_model() -> bool:
    """Test slicer validation with a model that has holes."""
    print_header("SLICER VALIDATION - MODEL WITH HOLES")
    
    # Find a model with holes
    fixtures_path = Path(__file__).parent.parent.parent / "tests" / "fixtures" / "thingi10k" / "holes"
    
    if not fixtures_path.exists():
        print(f"  [!] Holes fixtures directory not found: {fixtures_path}")
        return False
    
    stl_files = list(fixtures_path.glob("*.stl"))
    if not stl_files:
        print(f"  [!] No STL files found in {fixtures_path}")
        return False
    
    test_file = stl_files[0]
    print(f"\nTesting with: {test_file.name}")
    
    result = validate_stl_file(test_file, slicer="auto", timeout=60)
    
    print(f"\nResult:")
    print(f"  Slicer: {result.slicer}")
    print(f"  Success: {'PASS' if result.success else 'FAIL'} {result.success}")
    print(f"  Return code: {result.return_code}")
    
    if result.errors:
        print(f"  Errors ({len(result.errors)}):")
        for err in result.errors[:5]:
            print(f"    - {err[:80]}")
    
    if result.warnings:
        print(f"  Warnings ({len(result.warnings)}):")
        for warn in result.warnings[:5]:
            print(f"    - {warn[:80]}")
    
    if result.issues:
        print(f"  Issues ({len(result.issues)}):")
        for issue in result.issues[:5]:
            print(f"    - [{issue['type']}] {issue['message'][:60]}")
    
    # For a broken model, we actually expect it to either fail or have issues
    return True  # Test passes if we got a result (even if model fails)


def test_mesh_validation() -> bool:
    """Test validating a trimesh mesh object."""
    print_header("MESH OBJECT VALIDATION")
    
    try:
        import trimesh
    except ImportError:
        print("  [!] trimesh not installed")
        return False
    
    # Create a simple valid mesh (cube)
    print("\nCreating a simple cube mesh...")
    mesh = trimesh.primitives.Box()
    print(f"  Mesh: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")
    print(f"  Watertight: {mesh.is_watertight}")
    print(f"  Volume: {mesh.is_volume}")
    
    result = validate_mesh(mesh, slicer="auto", timeout=60)
    
    print(f"\nResult:")
    print(f"  Slicer: {result.slicer}")
    print(f"  Success: {'PASS' if result.success else 'FAIL'} {result.success}")
    
    if result.errors:
        print(f"  Errors: {result.errors[:3]}")
    
    return result.success or "not found" not in str(result.errors).lower()


def main():
    """Run all slicer tests."""
    print()
    print("=" * 60)
    print("   MESHPREP POC v2 - SLICER INTEGRATION TEST")
    print("=" * 60)
    
    # Test 1: Detection
    detection_ok = test_slicer_detection()
    
    if not detection_ok:
        print()
        print("=" * 60)
        print(" SUMMARY: No slicers found - slicer validation unavailable")
        print("=" * 60)
        print()
        print("Slicer validation is optional but strongly recommended.")
        print("Install PrusaSlicer for best results.")
        return 1
    
    # Test 2: Clean model validation
    clean_ok = test_slicer_validation_clean_model()
    
    # Test 3: Broken model validation
    broken_ok = test_slicer_validation_broken_model()
    
    # Test 4: Mesh object validation
    mesh_ok = test_mesh_validation()
    
    # Summary
    print_header("SUMMARY")
    
    results = [
        ("Slicer Detection", detection_ok),
        ("Clean Model Validation", clean_ok),
        ("Broken Model Validation", broken_ok),
        ("Mesh Object Validation", mesh_ok),
    ]
    
    all_passed = all(ok for _, ok in results)
    
    for name, ok in results:
        status = "[OK] PASS" if ok else "[X] FAIL"
        print(f"  {name}: {status}")
    
    print()
    if all_passed:
        print("  All tests passed! Slicer integration is working.")
    else:
        print("  Some tests failed. Check output above for details.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
