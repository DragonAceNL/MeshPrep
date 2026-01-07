"""Test new actions in POC v5."""

import sys
sys.path.insert(0, "C:/Users/Dragon Ace/Source/repos/MeshPrep/poc/v5")

from meshprep.core import ActionRegistry, Mesh
import trimesh

print("Testing POC v5 Actions...")
print("=" * 60)

# Import actions (triggers @register_action)
print("\n📦 Importing actions...")
try:
    from meshprep.actions import trimesh as trimesh_actions
    from meshprep.actions import pymeshfix as pymeshfix_actions
    from meshprep.actions import blender as blender_actions
    print("   ✅ Actions imported")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# List registered actions
print("\n📋 Registered Actions:")
actions = ActionRegistry.list_actions()
for name, action in actions.items():
    print(f"   ✅ {name:20} - {action.description} (risk: {action.risk_level.value})")

print(f"\n   Total: {len(actions)} actions registered")

# Create test mesh
print("\n🧊 Creating test mesh (cube)...")
cube = trimesh.primitives.Box(extents=[10, 10, 10])
mesh = Mesh(cube)
print(f"   Original: {mesh}")

# Test fix_normals (should always work)
print("\n1️⃣ Testing fix_normals...")
try:
    result = ActionRegistry.execute("fix_normals", mesh)
    if result.success:
        print(f"   ✅ Success in {result.duration_ms:.1f}ms")
        print(f"   Result: {result.mesh}")
    else:
        print(f"   ❌ Failed: {result.error}")
except Exception as e:
    print(f"   ❌ Exception: {e}")

# Test fill_holes (should work on cube)
print("\n2️⃣ Testing fill_holes...")
try:
    result = ActionRegistry.execute("fill_holes", mesh)
    if result.success:
        print(f"   ✅ Success in {result.duration_ms:.1f}ms")
    else:
        print(f"   ❌ Failed: {result.error}")
except Exception as e:
    print(f"   ❌ Exception: {e}")

# Test pymeshfix_repair (requires pymeshfix installed)
print("\n3️⃣ Testing pymeshfix_repair...")
try:
    result = ActionRegistry.execute("pymeshfix_repair", mesh)
    if result.success:
        print(f"   ✅ Success in {result.duration_ms:.1f}ms")
        print(f"   Result: {result.mesh}")
    else:
        print(f"   ⚠️  Failed (expected if pymeshfix not installed): {result.error}")
except Exception as e:
    print(f"   ⚠️  Exception (expected if pymeshfix not installed): {e}")

# Test blender_remesh (requires Blender installed)
print("\n4️⃣ Testing blender_remesh...")
try:
    result = ActionRegistry.execute("blender_remesh", mesh, {"voxel_size": 0.5})
    if result.success:
        print(f"   ✅ Success in {result.duration_ms:.1f}ms")
        print(f"   Result: {result.mesh}")
    else:
        print(f"   ⚠️  Failed (expected if Blender not installed): {result.error}")
except Exception as e:
    print(f"   ⚠️  Exception (expected if Blender not installed): {e}")

print("\n" + "=" * 60)
print("✅ Action testing complete!")
print("\nNote: pymeshfix and blender actions may fail if dependencies not installed.")
print("Core actions (fix_normals, fill_holes) should always work.")
