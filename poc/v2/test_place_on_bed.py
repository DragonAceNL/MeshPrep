# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Test if placing a model on the bed fixes slicer validation.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from meshprep_poc.mesh_ops import load_mesh, compute_diagnostics
from meshprep_poc.actions.slicer_actions import validate_stl_file
from meshprep_poc.actions.trimesh_actions import action_place_on_bed


def main():
    stl_file = Path("C:/Users/Dragon Ace/Source/repos/MeshPrep/tests/fixtures/thingi10k/fragmented/65585.stl")
    mesh = load_mesh(stl_file)
    diag = compute_diagnostics(mesh)

    print(f"Model: {stl_file.name}")
    print(f"  Vertices: {diag.vertex_count}")
    print(f"  Faces: {diag.face_count}")
    print(f"  Watertight: {diag.is_watertight}")
    print(f"  Is Volume: {diag.is_volume}")
    print(f"  Bounds: {mesh.bounds}")
    print(f"  Min Z: {mesh.bounds[0][2]}")
    print(f"  Max Z: {mesh.bounds[1][2]}")
    vol = mesh.volume if mesh.is_volume else "N/A"
    print(f"  Volume: {vol}")

    # This model is good geometry but probably floating
    # Lets place it on bed and try slicer again
    placed_mesh = action_place_on_bed(mesh, {})
    print(f"\nAfter place_on_bed:")
    print(f"  Min Z: {placed_mesh.bounds[0][2]}")
    print(f"  Max Z: {placed_mesh.bounds[1][2]}")

    # Save to temp and validate
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        placed_mesh.export(f.name)
        temp_path = Path(f.name)

    result = validate_stl_file(temp_path, slicer="auto", timeout=60)
    status = "PASS" if result.success else "FAIL"
    print(f"\nSlicer after place_on_bed: {status}")
    if not result.success:
        stderr_snippet = result.stderr[:200] if result.stderr else "(none)"
        print(f"  Error: {stderr_snippet}")
    else:
        print("  Model now passes slicer validation!")

    temp_path.unlink()  # Clean up


if __name__ == "__main__":
    main()
