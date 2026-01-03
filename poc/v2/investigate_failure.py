# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Investigate a failing model.
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')

sys.path.insert(0, str(Path(__file__).parent))

from meshprep_poc.mesh_ops import load_mesh, compute_diagnostics
from meshprep_poc.actions.slicer_actions import validate_stl_file


def main():
    # Test the failing model
    stl_file = Path("C:/Users/Dragon Ace/Source/repos/MeshPrep/tests/fixtures/thingi10k/self_intersecting/100827.stl")

    print(f"Testing: {stl_file.name}")
    print()

    mesh = load_mesh(stl_file)
    diag = compute_diagnostics(mesh)

    print("=== MESH DIAGNOSTICS ===")
    print(f"  Vertices: {diag.vertex_count}")
    print(f"  Faces: {diag.face_count}")
    print(f"  Watertight: {diag.is_watertight}")
    print(f"  Is Volume: {diag.is_volume}")
    print()

    print("=== SLICER VALIDATION ===")
    result = validate_stl_file(stl_file, slicer="auto", timeout=120)

    print(f"Success: {result.success}")
    print(f"Return code: {result.return_code}")
    print(f"Errors: {result.errors}")
    print(f"Warnings: {result.warnings}")
    print(f"Issues: {result.issues}")
    print()
    print("=== STDOUT (first 1000 chars) ===")
    print(result.stdout[:1000] if result.stdout else "(empty)")
    print()
    print("=== STDERR (first 1000 chars) ===")
    print(result.stderr[:1000] if result.stderr else "(empty)")


if __name__ == "__main__":
    main()
