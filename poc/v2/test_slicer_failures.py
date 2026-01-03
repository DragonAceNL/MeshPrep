# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Test script to find models that fail slicer validation.
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))

from meshprep_poc.mesh_ops import load_mesh
from meshprep_poc.actions.slicer_actions import validate_stl_file


def main():
    # Test more broken models - self-intersecting might fail
    categories = ["self_intersecting", "complex", "fragmented", "non_manifold", "holes"]
    fixtures_base = Path("C:/Users/Dragon Ace/Source/repos/MeshPrep/tests/fixtures/thingi10k")

    for cat in categories:
        fixtures = fixtures_base / cat
        if not fixtures.exists():
            continue
        stl_files = list(fixtures.glob("*.stl"))[:3]
        
        print(f"\n=== {cat.upper()} ===")
        for stl_file in stl_files:
            mesh = load_mesh(stl_file)
            result = validate_stl_file(stl_file, slicer="auto", timeout=60)
            
            status = "PASS" if result.success else "FAIL"
            issues = len(result.issues)
            warnings = len(result.warnings)
            print(f"{stl_file.name}: {status} (issues={issues}, warnings={warnings})")
            
            # Show slicer stdout/stderr if there are issues
            if result.issues:
                for i in result.issues[:2]:
                    issue_type = i["type"]
                    issue_msg = i["message"][:60]
                    print(f"  Issue: [{issue_type}] {issue_msg}")


if __name__ == "__main__":
    main()
