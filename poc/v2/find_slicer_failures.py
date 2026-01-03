# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Find models that fail slicer validation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from meshprep_poc.mesh_ops import load_mesh
from meshprep_poc.actions.slicer_actions import validate_stl_file


def main():
    fixtures_base = Path("C:/Users/Dragon Ace/Source/repos/MeshPrep/tests/fixtures/thingi10k")
    categories = ["holes", "non_manifold", "self_intersecting", "complex", "fragmented"]
    
    print("Looking for models that fail slicer but are repairable...")
    print()
    
    for cat in categories:
        fixtures = fixtures_base / cat
        if not fixtures.exists():
            continue
            
        for stl_file in fixtures.glob("*.stl"):
            mesh = load_mesh(stl_file)
            result = validate_stl_file(stl_file, slicer="auto", timeout=60)
            
            # We want models that FAIL slicer
            if not result.success:
                stderr_snippet = result.stderr[:100] if result.stderr else "(none)"
                print(f"{cat}/{stl_file.name}:")
                print(f"  Watertight: {mesh.is_watertight}, Volume: {mesh.is_volume}")
                print(f"  Slicer: FAIL")
                print(f"  Error: {stderr_snippet}")
                print()


if __name__ == "__main__":
    main()
