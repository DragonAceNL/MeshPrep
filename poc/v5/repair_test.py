import sys
sys.path.insert(0, '.')

from meshprep.core import Mesh, Pipeline, ActionRegistry
from meshprep.actions import trimesh, pymeshfix, blender, open3d, core

# Load the model
model_path = r'C:\Users\Dragon Ace\Source\repos\Thingi10K\raw_meshes\100026.stl'
mesh = Mesh.load(model_path)

print('=== BEFORE REPAIR ===')
print(f'Vertices: {mesh.metadata.vertex_count:,}')
print(f'Faces: {mesh.metadata.face_count:,}')
print(f'Bodies: {mesh.metadata.body_count}')
print(f'Is Watertight: {mesh.metadata.is_watertight}')
print(f'Is Manifold: {mesh.metadata.is_manifold}')
print(f'Volume: {mesh.metadata.volume:.2f}')

# Try a repair pipeline
print('\n=== RUNNING REPAIR PIPELINE ===')

pipeline = Pipeline(
    name='auto-repair',
    actions=[
        {'name': 'remove_duplicates'},
        {'name': 'fix_normals'},
        {'name': 'fill_holes'},
        {'name': 'pymeshfix_repair'},
        {'name': 'make_watertight'},
    ]
)

result = pipeline.execute(mesh)

if result.success:
    print('\n=== AFTER REPAIR ===')
    result.mesh._update_metadata_from_mesh()
    print(f'Vertices: {result.mesh.metadata.vertex_count:,}')
    print(f'Faces: {result.mesh.metadata.face_count:,}')
    print(f'Bodies: {result.mesh.metadata.body_count}')
    print(f'Is Watertight: {result.mesh.metadata.is_watertight}')
    print(f'Is Manifold: {result.mesh.metadata.is_manifold}')
    print(f'Volume: {result.mesh.metadata.volume:.2f}')
    print(f'\nPipeline duration: {result.duration_ms:.1f}ms')
    
    # Save the repaired mesh
    output_path = r'C:\Users\Dragon Ace\Source\repos\MeshPrep\poc\v5\repaired_100026.stl'
    result.mesh.trimesh.export(output_path)
    print(f'\nRepaired mesh saved to: {output_path}')
else:
    print(f'\nRepair failed: {result.error}')
    for ar in result.action_results:
        status = 'OK' if ar.success else 'FAIL'
        action_name = ar.metadata.get('action', 'unknown')
        err = ar.error if ar.error else 'success'
        print(f'  [{status}] {action_name}: {err}')
