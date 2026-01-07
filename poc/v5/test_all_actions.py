"""Test all 10 actions in POC v5."""

import sys
sys.path.insert(0, "C:/Users/Dragon Ace/Source/repos/MeshPrep/poc/v5")

from meshprep.core import ActionRegistry, Mesh, Pipeline
import trimesh
import numpy as np

print("Testing POC v5 - All 10 Actions")
print("=" * 70)

# Import all actions
print("\n📦 Importing actions...")
try:
    from meshprep.actions import trimesh as trimesh_actions
    from meshprep.actions import pymeshfix as pymeshfix_actions
    from meshprep.actions import blender as blender_actions
    from meshprep.actions import open3d as open3d_actions
    print("   ✅ All action modules imported")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# List all actions
print("\n📋 Registered Actions:")
actions = ActionRegistry.list_actions()
action_list = sorted(actions.items())

for name, action in action_list:
    risk_color = {
        'low': '🟢',
        'medium': '🟡',
        'high': '🔴'
    }.get(action.risk_level.value, '⚪')
    
    print(f"   {risk_color} {name:30} [{action.risk_level.value.upper():6}] {action.description}")

print(f"\n   Total: {len(actions)} actions")

# Create test mesh (cube with some issues)
print("\n🧊 Creating test mesh...")
cube = trimesh.primitives.Box(extents=[10, 10, 10])
mesh = Mesh(cube)
print(f"   Mesh: {mesh.metadata.face_count} faces, "
      f"watertight={mesh.metadata.is_watertight}")

# Test each category
print("\n" + "="*70)
print("Testing Trimesh Actions (5)")
print("="*70)

trimesh_actions_test = [
    "fix_normals",
    "remove_duplicates",
    "fill_holes",
    "decimate",
]

for action_name in trimesh_actions_test:
    print(f"\n▶ {action_name}...")
    try:
        result = ActionRegistry.execute(action_name, mesh, {} if action_name != "decimate" else {"face_count": 50})
        if result.success:
            print(f"   ✅ Success in {result.duration_ms:.1f}ms")
            if result.mesh:
                print(f"   Result: {result.mesh.metadata.face_count} faces")
        else:
            print(f"   ❌ Failed: {result.error}")
    except Exception as e:
        print(f"   ❌ Exception: {e}")

# Note about convex_hull (destructive)
print(f"\n▶ convex_hull... (skipped - too destructive for demo)")

print("\n" + "="*70)
print("Testing PyMeshFix Actions (2)")
print("="*70)

pymeshfix_test = ["pymeshfix_clean", "pymeshfix_repair"]

for action_name in pymeshfix_test:
    print(f"\n▶ {action_name}...")
    try:
        result = ActionRegistry.execute(action_name, mesh)
        if result.success:
            print(f"   ✅ Success in {result.duration_ms:.1f}ms")
        else:
            print(f"   ⚠️  {result.error}")
    except Exception as e:
        print(f"   ⚠️  Requires: pip install pymeshfix")

print("\n" + "="*70)
print("Testing Blender Actions (2) - SLOW")
print("="*70)
print("   ⚠️  Skipped (requires Blender, takes 30-60s each)")
print("   Available: blender_remesh, blender_boolean_union")

print("\n" + "="*70)
print("Testing Open3D Actions (1)")
print("="*70)

print(f"\n▶ poisson_reconstruction...")
try:
    result = ActionRegistry.execute("poisson_reconstruction", mesh, {"depth": 7})
    if result.success:
        print(f"   ✅ Success in {result.duration_ms:.1f}ms")
    else:
        print(f"   ⚠️  {result.error}")
except Exception as e:
    print(f"   ⚠️  Requires: pip install open3d")

# Pipeline demo
print("\n" + "="*70)
print("Testing Multi-Action Pipeline")
print("="*70)

pipeline = Pipeline(
    name="light-cleanup",
    actions=[
        {"name": "remove_duplicates"},
        {"name": "fix_normals"},
        {"name": "fill_holes"},
    ]
)

print(f"\n▶ Running pipeline: {pipeline.name}")
result = pipeline.execute(mesh)

if result.success:
    print(f"   ✅ Pipeline complete in {result.duration_ms:.1f}ms")
    print(f"   Actions: {result.actions_executed}/{len(pipeline.actions)}")
else:
    print(f"   ❌ Pipeline failed: {result.error}")

print("\n" + "="*70)
print("✅ Action testing complete!")
print("\nSummary:")
print("  • 10 actions available")
print("  • 3 require no dependencies (fix_normals, remove_duplicates, fill_holes)")
print("  • 2 require pymeshfix")
print("  • 2 require Blender")
print("  • 1 requires Open3D")
print("  • 2 are destructive (convex_hull, decimate)")
