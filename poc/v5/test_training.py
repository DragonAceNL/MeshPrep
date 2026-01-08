# Training test with 20 meshes
import sys
sys.path.insert(0, '.')

from meshprep.ml.learning_engine import SmartRepairEngine, TrainingConfig
from pathlib import Path

config = TrainingConfig(
    min_samples_to_train=10,
    epochs_per_update=20,
    model_dir=Path('models/trained'),
)

engine = SmartRepairEngine(config=config, auto_train=False)
print(f'Device: {engine.learning_loop.device}')

# Get more test meshes
thingi_dir = Path(r'C:\Users\Dragon Ace\Source\repos\Thingi10K\raw_meshes')
test_meshes = sorted(thingi_dir.glob('*.stl'))[10:30]  # Skip repaired ones
print(f'Processing {len(test_meshes)} meshes...')

success_count = 0
total_quality = 0

for i, mesh_path in enumerate(test_meshes):
    try:
        result = engine.repair(mesh_path)
        if result.is_printable:
            success_count += 1
        total_quality += result.quality_score
        print(f'[{i+1}] {mesh_path.stem}: Q={result.quality_score}, Print={result.is_printable}')
    except Exception as e:
        print(f'[{i+1}] {mesh_path.stem}: ERROR')

print(f'\nSuccess: {success_count}/{len(test_meshes)} ({success_count/len(test_meshes)*100:.0f}%)')
print(f'Avg Quality: {total_quality/len(test_meshes):.1f}/5')

# Train
print('\nTraining...')
metrics = engine.train(force=True)
if metrics:
    print(f'Loss: {metrics["loss"]:.4f}')
    print(f'Samples: {metrics["samples_used"]}')

engine.save()

print('\nFinal Statistics:')
stats = engine.get_statistics()
print(f'  Total outcomes: {stats["total_outcomes"]}')
print(f'  Success rate: {stats["success_rate"]*100:.1f}%')
print(f'  Model updates: {stats["model_updates"]}')
