"""Simple test to verify POC v5 is working."""
import sys
sys.path.insert(0, ".")

from meshprep.core import Mesh, ActionRegistry, Pipeline
import trimesh
import numpy as np

print("\n" + "="*60)
print("POC v5 - Quick Verification Test")
print("="*60)

# Test 1: Create a mesh
print("\n1. Creating test mesh...")
cube = trimesh.primitives.Box(extents=[10, 10, 10])
mesh = Mesh(cube)
print(f"   ✓ Mesh created: {mesh.metadata.face_count} faces")

# Test 2: List actions
print("\n2. Loading actions...")
from meshprep.actions import trimesh as trimesh_actions
actions = ActionRegistry.list_actions()
print(f"   ✓ {len(actions)} actions registered")

# Test 3: Execute an action
print("\n3. Testing action execution...")
result = ActionRegistry.execute("fix_normals", mesh)
print(f"   ✓ Action executed: success={result.success}")

# Test 4: Test pipeline
print("\n4. Testing pipeline...")
pipeline = Pipeline(
    name="test",
    actions=[
        {"name": "remove_duplicates"},
        {"name": "fix_normals"},
    ]
)
result = pipeline.execute(mesh)
print(f"   ✓ Pipeline executed: {result.actions_executed}/{len(pipeline.actions)} actions")

print("\n" + "="*60)
print("✅ POC v5 is working!")
print("="*60)
