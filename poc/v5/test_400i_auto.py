# Test improved smart engine on 400i.ctm
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import trimesh
from meshprep.core import Mesh
from meshprep.ml.learning_engine import SmartRepairEngine

# Load the challenging mesh
mesh_path = Path(r'C:\Users\Dragon Ace\Source\repos\Thingi10K\raw_meshes\400i.ctm')
scene = trimesh.load(mesh_path)
if isinstance(scene, trimesh.Scene):
    combined = scene.to_geometry()
else:
    combined = scene

mesh = Mesh(combined)
print(f'Input: {len(combined.vertices)} verts, {len(combined.faces)} faces')

# Check features
engine = SmartRepairEngine(auto_train=True)
features = engine.learning_loop.feature_encoder.encode(mesh)
print(f'Components: {features.num_components}')
print(f'Extremely fragmented: {features.is_extremely_fragmented}')
print(f'BBox diagonal: {features.bbox_diagonal:.1f}')

# Get prediction BEFORE repair
actions, params, conf = engine.predict_only(mesh)
print(f'\nPrediction:')
print(f'  Actions: {actions}')
print(f'  Params: {params}')
print(f'  Confidence: {conf:.2f}')

# Now repair
print(f'\nRepairing...')
output = Path('repaired/400i_auto_v2.stl')
output.parent.mkdir(exist_ok=True)

result = engine.repair(mesh, output, max_attempts=3)

print(f'\nResult:')
print(f'  Success: {result.success}')
print(f'  Printable: {result.is_printable}')
print(f'  Quality: {result.quality_score}/5')
print(f'  Actions used: {result.actions}')
print(f'  Attempts: {result.attempts}')
print(f'  Duration: {result.duration_ms/1000:.1f}s')

if result.mesh:
    out = result.mesh.trimesh
    print(f'  Output: {len(out.vertices)} verts, {len(out.faces)} faces')
    print(f'  Watertight: {out.is_watertight}')

# Stats
stats = engine.get_statistics()
print(f'\nLearning stats:')
total = stats["total_outcomes"]
rate = stats["success_rate"] * 100
print(f'  Total outcomes: {total}')
print(f'  Success rate: {rate:.1f}%')
