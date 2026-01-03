# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Test the slicer repair loop with a synthetically broken mesh.
"""

import logging
import sys
import tempfile
from pathlib import Path

import trimesh
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from meshprep_poc.mesh_ops import load_mesh, compute_diagnostics
from meshprep_poc.actions.slicer_actions import validate_mesh, is_slicer_available
from meshprep_poc.slicer_repair_loop import run_slicer_repair_loop


def create_broken_mesh():
    """Create a mesh with holes that should fail slicer validation."""
    # Create a box and remove some faces to create holes
    mesh = trimesh.primitives.Box(extents=[20, 20, 20])
    mesh = mesh.copy()

    # Remove a few faces to create holes
    faces = mesh.faces.copy()
    faces_to_keep = np.ones(len(faces), dtype=bool)
    faces_to_keep[0] = False
    faces_to_keep[1] = False
    faces_to_keep[2] = False
    faces_to_keep[3] = False

    mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=faces[faces_to_keep])
    return mesh


def print_header(text: str) -> None:
    print()
    print("=" * 60)
    print(f" {text}")
    print("=" * 60)


def main():
    print_header("SLICER REPAIR LOOP TEST - SYNTHETIC MESH")
    
    if not is_slicer_available():
        print("ERROR: No slicer available!")
        return 1
    
    print("Slicer: Available")
    
    # Create broken mesh
    print("\nCreating broken mesh (box with holes)...")
    mesh = create_broken_mesh()
    
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Faces: {len(mesh.faces)}")
    print(f"  Watertight: {mesh.is_watertight}")
    print(f"  Is Volume: {mesh.is_volume}")
    
    # Verify it fails slicer
    print("\nInitial slicer validation:")
    result = validate_mesh(mesh, slicer="auto", timeout=60)
    status = "PASS" if result.success else "FAIL"
    print(f"  Result: {status}")
    if result.stderr:
        print(f"  Error: {result.stderr[:200]}")
    
    if result.success:
        print("\n  Mesh already passes slicer - creating more broken mesh...")
        # Make it worse
        faces = mesh.faces.copy()
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=faces[:4])
        print(f"  New face count: {len(mesh.faces)}")
    
    # Run repair loop
    print_header("RUNNING SLICER REPAIR LOOP")
    
    repair_result = run_slicer_repair_loop(
        mesh,
        slicer="auto",
        max_attempts=5,
        escalate_to_blender_after=3,
        timeout=60
    )
    
    print_header("REPAIR RESULT")
    print(f"  Success: {repair_result.success}")
    print(f"  Total attempts: {repair_result.total_attempts}")
    print(f"  Total duration: {repair_result.total_duration_ms:.0f}ms")
    print(f"  Issues found: {repair_result.issues_found}")
    print(f"  Issues resolved: {repair_result.issues_resolved}")
    
    if repair_result.error:
        print(f"  Error: {repair_result.error}")
    
    print("\nAttempt history:")
    for attempt in repair_result.attempts:
        status = "OK" if attempt.success else "FAIL"
        print(f"  {attempt.attempt_number}. {attempt.strategy.action}: {status} ({attempt.duration_ms:.0f}ms)")
        if attempt.error:
            err_short = attempt.error[:60] + "..." if len(attempt.error) > 60 else attempt.error
            print(f"     Error: {err_short}")
    
    if repair_result.success and repair_result.final_mesh:
        final = repair_result.final_mesh
        print("\nFinal mesh:")
        print(f"  Vertices: {len(final.vertices)}")
        print(f"  Faces: {len(final.faces)}")
        print(f"  Watertight: {final.is_watertight}")
        print(f"  Is Volume: {final.is_volume}")
        
        # Verify with slicer
        final_result = validate_mesh(final, slicer="auto", timeout=60)
        final_status = "PASS" if final_result.success else "FAIL"
        print(f"\n  Final slicer validation: {final_status}")
    
    return 0 if repair_result.success else 1


if __name__ == "__main__":
    sys.exit(main())
