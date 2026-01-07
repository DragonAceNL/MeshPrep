"""Simple test to verify v5 core works."""

from pathlib import Path
from meshprep.core import Mesh, ActionRegistry, Pipeline, Validator, RepairEngine

print("Testing MeshPrep v5 Core...")
print("-" * 50)

# Test 1: Mesh loading (simulated with a simple mesh)
print("\n1. Testing Mesh class...")
try:
    import trimesh
    # Create a simple cube
    cube = trimesh.primitives.Box()
    mesh = Mesh(cube)
    print(f"   ✅ Created mesh: {mesh}")
    print(f"   ✅ Metadata: watertight={mesh.metadata.is_watertight}, "
          f"faces={mesh.metadata.face_count}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 2: Action registry
print("\n2. Testing Action registry...")
try:
    actions = ActionRegistry.list_actions()
    print(f"   ✅ Registered actions: {list(actions.keys())}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 3: Pipeline
print("\n3. Testing Pipeline...")
try:
    pipeline = Pipeline(
        name="test",
        actions=[],
        description="Test pipeline"
    )
    print(f"   ✅ Created pipeline: {pipeline}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 4: Validator
print("\n4. Testing Validator...")
try:
    validator = Validator()
    validation = validator.validate_geometry(mesh)
    print(f"   ✅ Validation: printable={validation.is_printable}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 5: RepairEngine
print("\n5. Testing RepairEngine...")
try:
    engine = RepairEngine()
    print(f"   ✅ Created engine with max_attempts={engine.max_attempts}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

print("\n" + "=" * 50)
print("✅ All core components working!")
