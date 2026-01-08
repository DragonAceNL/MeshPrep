# Quick learning test
import sys
sys.path.insert(0, '.')

from meshprep.ml.learning_engine import SmartRepairEngine, TrainingConfig
from pathlib import Path

# Configure
config = TrainingConfig(
    min_samples_to_train=3,
    epochs_per_update=10,
    model_dir=Path('models/learning_test'),
)

# Create engine
engine = SmartRepairEngine(config=config, auto_train=False)
print(f'Device: {engine.learning_loop.device}')

# Clear old data  
engine.learning_loop.tracker.clear()

# Get test meshes
thingi_dir = Path(r'C:\Users\Dragon Ace\Source\repos\Thingi10K\raw_meshes')
output_dir = Path('repaired/learning_test')
output_dir.mkdir(parents=True, exist_ok=True)

test_meshes = list(thingi_dir.glob('*.stl'))[:5]
print(f'Testing on {len(test_meshes)} meshes...')

results = []
for i, mesh_path in enumerate(test_meshes):
    print(f'[{i+1}/{len(test_meshes)}] {mesh_path.name}')
    
    try:
        result = engine.repair(mesh_path, output_dir / f'{mesh_path.stem}_repaired.stl')
        results.append(result)
        
        status = 'OK' if result.is_printable else 'FAIL'
        print(f'  {status} | Quality: {result.quality_score}/5 | Actions: {len(result.actions)} | {result.duration_ms:.0f}ms')
        
    except Exception as e:
        print(f'  ERROR: {e}')

# Summary
print('--- SUMMARY ---')
if results:
    printable = sum(1 for r in results if r.is_printable)
    avg_quality = sum(r.quality_score for r in results) / len(results)
    print(f'Printable: {printable}/{len(results)} ({printable/len(results)*100:.0f}%)')
    print(f'Avg quality: {avg_quality:.1f}/5')

# Training
print('--- TRAINING ---')
print(f'Samples recorded: {engine.learning_loop.total_samples_seen}')

metrics = engine.train(force=True)
if metrics:
    loss = metrics["loss"]
    samples = metrics["samples_used"]
    print(f'Training loss: {loss:.4f}')
    print(f'Samples used: {samples}')
else:
    print('Not enough samples to train')

print('--- STATISTICS ---')
stats = engine.get_statistics()
print(f'Total outcomes: {stats["total_outcomes"]}')
print(f'Success rate: {stats["success_rate"]*100:.1f}%')
print(f'Model updates: {stats["model_updates"]}')

engine.save()
print('Model saved!')
