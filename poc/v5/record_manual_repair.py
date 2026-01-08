# Record manual repair for learning
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import trimesh
from meshprep.core import Mesh
from meshprep.ml.learning_engine import SmartRepairEngine

# Load original mesh
mesh_path = Path(r'C:\Users\Dragon Ace\Source\repos\Thingi10K\raw_meshes\400i.ctm')
scene = trimesh.load(mesh_path)
if isinstance(scene, trimesh.Scene):
    original = scene.to_geometry()
else:
    original = scene

mesh = Mesh(original)

# Record the MANUAL successful repair for learning
engine = SmartRepairEngine(auto_train=False)

engine.learning_loop.record_outcome(
    mesh=mesh,
    actions=['blender_remesh'],
    parameters={'blender_remesh': {'voxel_size': 34.0}},
    is_printable=True,
    quality_score=4.0,
    volume_change_pct=0.0,
    hausdorff_relative=0.042,
    duration_ms=15000,
    mesh_id='400i_manual',
)

print('Recorded successful manual repair!')
stats = engine.get_statistics()
print(f"Total outcomes: {stats['total_outcomes']}")

engine.train(force=True)
engine.save()
print('Model updated!')
