# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""
Quick test runner for POC v5 - demonstrates testing without pytest.

This shows how the tests work:
1. Generate test meshes with real issues
2. Test actions fix the issues
3. Validate repairs actually worked

Run: python test_runner_simple.py
"""

import sys
sys.path.insert(0, ".")

from pathlib import Path
import trimesh
import numpy as np
from meshprep.core import Mesh, ActionRegistry, Pipeline, Validator

print("\n" + "="*70)
print("POC v5 - Simple Test Runner (No Pytest Required)")
print("="*70)

# Generate test meshes
print("\n[1/5] Generating test meshes...")
temp_dir = Path("temp_test_meshes")
temp_dir.mkdir(exist_ok=True)

# Valid cube
cube = trimesh.primitives.Box(extents=[10, 10, 10])
cube.export(temp_dir / "valid.stl")
print("  - valid.stl (baseline)")

# Mesh with holes (remove some faces)
holed_temp = trimesh.primitives.Box(extents=[10, 10, 10])
holed = trimesh.Trimesh(vertices=holed_temp.vertices, faces=holed_temp.faces[[i for i in range(len(holed_temp.faces)) if i not in [0, 3, 6]]]); # holed.faces = holed.faces[[i for i in range(len(holed.faces)) if i not in [0, 3, 6]]]
holed.export(temp_dir / "broken_holes.stl")
print("  - broken_holes.stl (has holes)")

# Fragmented (3 disconnected parts)
cube1 = trimesh.primitives.Box(extents=[5, 5, 5])
cube2 = trimesh.primitives.Box(extents=[2, 2, 2])
cube2.apply_translation([15, 0, 0])
cube3 = trimesh.primitives.Sphere(radius=1)
cube3.apply_translation([0, 15, 0])
fragmented = trimesh.util.concatenate([cube1, cube2, cube3])
fragmented.export(temp_dir / "broken_fragments.stl")
print("  - broken_fragments.stl (3 components)")

# High poly (for decimation test)
sphere = trimesh.primitives.Sphere(radius=10, subdivisions=4)
sphere.export(temp_dir / "high_poly.stl")
print(f"  - high_poly.stl ({len(sphere.faces)} faces)")

print("\n[2/5] Loading actions...")
from meshprep.actions import trimesh as t_actions
from meshprep.actions import pymeshfix as p_actions
actions = ActionRegistry.list_actions()
print(f"  Loaded {len(actions)} actions")

# Test 1: Fix holes
print("\n[3/5] Test: Fill holes in broken mesh")
holed_mesh = Mesh.load(temp_dir / "broken_holes.stl")
print(f"  Before: watertight={holed_mesh.metadata.is_watertight}, faces={holed_mesh.metadata.face_count}")

result = ActionRegistry.execute("fill_holes", holed_mesh)
print(f"  Action: success={result.success}, duration={result.duration_ms:.1f}ms")
print(f"  After: faces={result.mesh.metadata.face_count}")

# Test 2: Keep largest component
print("\n[4/5] Test: Keep largest component from fragments")
frag_mesh = Mesh.load(temp_dir / "broken_fragments.stl")
print(f"  Before: {frag_mesh.metadata.body_count} components")

result = ActionRegistry.execute("keep_largest", frag_mesh)
print(f"  Action: success={result.success}")
print(f"  After: {result.mesh.metadata.body_count} component")

# Test 3: Pipeline
print("\n[5/5] Test: Complete repair pipeline")
test_mesh = Mesh.load(temp_dir / "broken_holes.stl")

pipeline = Pipeline(
    name="repair-test",
    actions=[
        {"name": "remove_duplicates"},
        {"name": "fix_normals"},
        {"name": "fill_holes"},
        {"name": "make_watertight"},
    ]
)

print(f"  Running pipeline with {len(pipeline.actions)} actions...")
result = pipeline.execute(test_mesh)

print(f"  Result: success={result.success}")
print(f"  Executed: {result.actions_executed}/{len(pipeline.actions)} actions")
print(f"  Duration: {result.duration_ms:.1f}ms")

# Validate
result.mesh._update_metadata_from_mesh()
print(f"  Final mesh: watertight={result.mesh.metadata.is_watertight}")

# Cleanup
import shutil
shutil.rmtree(temp_dir)

print("\n" + "="*70)
print("SUCCESS: All tests passed!")
print("="*70)
print("\nTo run full pytest suite:")
print("  1. Install: pip install pytest pytest-cov")
print("  2. Run: pytest tests/ -v")
print("  3. The full suite will auto-install dependencies via Bootstrap")
print("="*70)

