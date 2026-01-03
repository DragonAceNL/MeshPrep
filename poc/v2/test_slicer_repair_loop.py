# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Test the iterative slicer repair loop.
"""

import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from meshprep_poc.mesh_ops import load_mesh, compute_diagnostics, print_diagnostics
from meshprep_poc.actions.slicer_actions import validate_stl_file, is_slicer_available
from meshprep_poc.slicer_repair_loop import run_slicer_repair_loop


def print_header(text: str) -> None:
    """Print a section header."""
    print()
    print("=" * 60)
    print(f" {text}")
    print("=" * 60)


def test_slicer_repair_loop(stl_path: Path):
    """Test the slicer repair loop on a single model."""
    print_header(f"TESTING: {stl_path.name}")
    
    # Load and diagnose original mesh
    mesh = load_mesh(stl_path)
    diag = compute_diagnostics(mesh)
    
    print("\nOriginal mesh:")
    print(f"  Vertices: {diag.vertex_count}")
    print(f"  Faces: {diag.face_count}")
    print(f"  Watertight: {diag.is_watertight}")
    print(f"  Is Volume: {diag.is_volume}")
    
    # First, validate with slicer to see if it fails
    print("\nInitial slicer validation:")
    slicer_result = validate_stl_file(stl_path, slicer="auto", timeout=60)
    print(f"  Success: {slicer_result.success}")
    if slicer_result.errors:
        print(f"  Errors: {slicer_result.errors[:3]}")
    if slicer_result.stderr:
        print(f"  Stderr: {slicer_result.stderr[:200]}")
    
    if slicer_result.success:
        print("\n  Model already passes slicer validation - no repair needed!")
        return True
    
    # Run the repair loop
    print_header("RUNNING SLICER REPAIR LOOP")
    
    result = run_slicer_repair_loop(
        mesh,
        slicer="auto",
        max_attempts=5,
        escalate_to_blender_after=3,
        timeout=60
    )
    
    print_header("REPAIR RESULT")
    print(f"  Success: {result.success}")
    print(f"  Total attempts: {result.total_attempts}")
    print(f"  Total duration: {result.total_duration_ms:.0f}ms")
    print(f"  Issues found: {result.issues_found}")
    print(f"  Issues resolved: {result.issues_resolved}")
    
    if result.error:
        print(f"  Error: {result.error}")
    
    print("\nAttempt history:")
    for attempt in result.attempts:
        status = "OK" if attempt.success else "FAIL"
        print(f"  {attempt.attempt_number}. {attempt.strategy.action}: {status} ({attempt.duration_ms:.0f}ms)")
        if attempt.error:
            print(f"     Error: {attempt.error[:80]}")
    
    if result.success and result.final_mesh:
        final_diag = compute_diagnostics(result.final_mesh)
        print("\nFinal mesh:")
        print(f"  Vertices: {final_diag.vertex_count}")
        print(f"  Faces: {final_diag.face_count}")
        print(f"  Watertight: {final_diag.is_watertight}")
        print(f"  Is Volume: {final_diag.is_volume}")
    
    return result.success


def main():
    print_header("SLICER REPAIR LOOP TEST")
    
    # Check slicer availability
    if not is_slicer_available():
        print("ERROR: No slicer available!")
        return 1
    
    print("Slicer: Available")
    
    # Test models that are known to fail slicer validation
    test_models = [
        # These fail slicer validation
        Path("C:/Users/Dragon Ace/Source/repos/MeshPrep/tests/fixtures/thingi10k/self_intersecting/100827.stl"),
        Path("C:/Users/Dragon Ace/Source/repos/MeshPrep/tests/fixtures/thingi10k/fragmented/1004825.stl"),
    ]
    
    results = []
    for model_path in test_models:
        if model_path.exists():
            success = test_slicer_repair_loop(model_path)
            results.append((model_path.name, success))
        else:
            print(f"WARNING: Model not found: {model_path}")
    
    # Summary
    print_header("SUMMARY")
    for name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  {name}: {status}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\nOverall: {passed}/{total} models repaired successfully")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
