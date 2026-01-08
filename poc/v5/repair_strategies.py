import sys
sys.path.insert(0, '.')

from meshprep.core import Mesh, Pipeline, ActionRegistry
from meshprep.actions import trimesh, pymeshfix, blender, open3d, core

model_path = r'C:\Users\Dragon Ace\Source\repos\Thingi10K\raw_meshes\100027.stl'
mesh = Mesh.load(model_path)

print(f'Model: 100027.stl')
print(f'BEFORE: {mesh.metadata.vertex_count} verts, {mesh.metadata.face_count} faces, '
      f'{mesh.metadata.body_count} bodies')
print()

# Strategy 1: Keep largest component first
print('Strategy 1: Keep largest + repair')
mesh1 = mesh.copy()
pipeline1 = Pipeline(
    name='keep-largest-first',
    actions=[
        {'name': 'keep_largest'},
        {'name': 'fix_normals'},
        {'name': 'fill_holes'},
        {'name': 'pymeshfix_repair'},
    ]
)
result1 = pipeline1.execute(mesh1)
if result1.success and result1.mesh:
    result1.mesh._update_metadata_from_mesh()
    print(f'  AFTER: {result1.mesh.metadata.vertex_count} verts, {result1.mesh.metadata.face_count} faces, '
          f'{result1.mesh.metadata.body_count} bodies, watertight={result1.mesh.metadata.is_watertight}')
else:
    print(f'  FAILED: {result1.error}')
print()

# Strategy 2: Use Blender boolean union
print('Strategy 2: Blender boolean union + remesh')
mesh2 = mesh.copy()
pipeline2 = Pipeline(
    name='blender-union',
    actions=[
        {'name': 'blender_boolean_union'},
        {'name': 'blender_remesh', 'params': {'voxel_size': 0.5}},
    ]
)
result2 = pipeline2.execute(mesh2)
if result2.success and result2.mesh:
    result2.mesh._update_metadata_from_mesh()
    print(f'  AFTER: {result2.mesh.metadata.vertex_count} verts, {result2.mesh.metadata.face_count} faces, '
          f'{result2.mesh.metadata.body_count} bodies, watertight={result2.mesh.metadata.is_watertight}')
else:
    print(f'  FAILED: {result2.error}')
print()

# Strategy 3: Convex hull (extreme - loses detail)
print('Strategy 3: Convex hull (loses detail)')
mesh3 = mesh.copy()
result3 = ActionRegistry.execute('convex_hull', mesh3)
if result3.success and result3.mesh:
    result3.mesh._update_metadata_from_mesh()
    print(f'  AFTER: {result3.mesh.metadata.vertex_count} verts, {result3.mesh.metadata.face_count} faces, '
          f'{result3.mesh.metadata.body_count} bodies, watertight={result3.mesh.metadata.is_watertight}')
else:
    print(f'  FAILED: {result3.error}')
