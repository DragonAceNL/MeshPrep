import sys
sys.path.insert(0, ".")

# Test imports
print("1. Testing imports...")
from meshprep.core import Mesh, ActionRegistry
print("   ✓ Core imports OK")

# Test action imports
print("2. Testing action imports...")
from meshprep.actions import trimesh, pymeshfix, blender, open3d, core
print("   ✓ Action imports OK")

# Test action registry
print("3. Testing action registry...")
actions = ActionRegistry.list_actions()
print(f"   ✓ {len(actions)} actions registered")

# Test mesh creation
print("4. Testing mesh creation...")
import trimesh as tm
cube = tm.primitives.Box()
mesh = Mesh(cube)
print(f"   ✓ Mesh created: {mesh.metadata.face_count} faces")

# Test action execution
print("5. Testing action execution...")
result = ActionRegistry.execute("fix_normals", mesh)
print(f"   ✓ Action executed: success={result.success}")

print("\n✅ All diagnostics passed!")
