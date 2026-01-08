import sys
sys.path.insert(0, '.')

from meshprep.core import Mesh, Pipeline
from meshprep.actions import trimesh, pymeshfix, blender, open3d, core
import os

# Models to test
models = [
    r'C:\Users\Dragon Ace\Source\repos\Thingi10K\raw_meshes\100026.stl',
    r'C:\Users\Dragon Ace\Source\repos\Thingi10K\raw_meshes\100027.stl',
    r'C:\Users\Dragon Ace\Source\repos\Thingi10K\raw_meshes\100032.stl',
    r'C:\Users\Dragon Ace\Source\repos\Thingi10K\raw_meshes\100070.stl',
]

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

print('=' * 80)
print('MESHPREP v5 - REAL MODEL REPAIR TEST')
print('=' * 80)

for model_path in models:
    filename = os.path.basename(model_path)
    print(f'\n--- {filename} ---')
    
    try:
        mesh = Mesh.load(model_path)
        
        # Before
        print(f'BEFORE: {mesh.metadata.vertex_count:,} verts, {mesh.metadata.face_count:,} faces, '
              f'{mesh.metadata.body_count} bodies, watertight={mesh.metadata.is_watertight}')
        
        # Repair
        result = pipeline.execute(mesh)
        
        if result.success:
            result.mesh._update_metadata_from_mesh()
            print(f'AFTER:  {result.mesh.metadata.vertex_count:,} verts, {result.mesh.metadata.face_count:,} faces, '
                  f'{result.mesh.metadata.body_count} bodies, watertight={result.mesh.metadata.is_watertight}')
            print(f'TIME:   {result.duration_ms:.1f}ms')
            
            # Check if actually fixed
            if result.mesh.metadata.is_watertight and result.mesh.metadata.body_count == 1:
                print('STATUS: [OK] FIXED')
            elif result.mesh.metadata.is_watertight:
                print('STATUS: [~] PARTIALLY FIXED (watertight but multiple bodies)')
            else:
                print('STATUS: [X] NOT FULLY FIXED')
        else:
            print(f'STATUS: [X] REPAIR FAILED - {result.error}')
            
    except Exception as e:
        print(f'STATUS: [X] ERROR - {e}')

print('\n' + '=' * 80)
