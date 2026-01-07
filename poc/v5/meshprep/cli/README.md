# MeshPrep CLI

Command-line interface for MeshPrep v5.

---

## Installation

```bash
cd poc/v5
pip install -e .
```

This installs the `meshprep` command system-wide.

---

## Commands

### 1. **repair** - Repair a mesh

```bash
meshprep repair input.stl

# With options
meshprep repair input.stl -o output.stl
meshprep repair input.stl --pipeline cleanup
meshprep repair input.stl --no-ml --no-learning
```

**Options:**
- `-o, --output PATH`: Output file (default: input_fixed.stl)
- `-p, --pipeline NAME`: Specific pipeline to use
- `--no-ml`: Disable ML prediction
- `--no-learning`: Disable learning system
- `-v, --verbose`: Enable verbose logging

**Example:**
```bash
meshprep repair broken_model.stl -o fixed.stl --verbose
```

---

### 2. **stats** - View statistics

```bash
# View overall statistics
meshprep stats

# View specific pipeline stats
meshprep stats --pipeline cleanup

# Show top 20 pipelines
meshprep stats --limit 20
```

**Options:**
- `-p, --pipeline NAME`: Filter by pipeline
- `-n, --limit N`: Number of results (default: 10)

**Output:**
```
📊 Repair Statistics
==================================================
Total repairs: 1234
Successes: 1050
Success rate: 85.1%

🏆 Top 10 Pipelines:
  1. cleanup: 0.820
  2. standard: 0.753
  3. aggressive: 0.682
  ...

💡 Suggestions:
  • Pipeline 'old-method' has low success rate (35%). Consider removing.
```

---

### 3. **list-actions** - List available actions

```bash
meshprep list-actions
```

**Output:**
```
📋 Available Actions
==================================================
🟢 fix_normals              [low   ] Fix face winding
🟢 remove_duplicates        [low   ] Remove duplicate vertices
🟡 fill_holes               [medium] Fill mesh holes
...

Total: 20 actions
```

---

## Complete Workflow

### Setup
```bash
# Install
cd poc/v5
pip install -e .

# Verify installation
meshprep --version
meshprep --help
```

### Basic Repair
```bash
# Repair a single file
meshprep repair broken_model.stl

# Output: broken_model_fixed.stl
```

### Batch Repair
```bash
# Repair multiple files
for file in *.stl; do
    meshprep repair "$file" -o "fixed/$file"
done
```

### View Progress
```bash
# Check statistics
meshprep stats

# See what's working best
meshprep stats --limit 5
```

### Debug
```bash
# Verbose output
meshprep repair broken.stl --verbose

# Disable ML/learning for testing
meshprep repair broken.stl --no-ml --no-learning
```

---

## Integration with Python

You can also use the CLI from Python:

```python
from meshprep.cli import cli

# Run CLI programmatically
cli(["repair", "broken_model.stl"])
```

---

## Configuration

CLI respects environment variables:

```bash
# Set default output directory
export MESHPREP_OUTPUT_DIR="./fixed"

# Set learning database location
export MESHPREP_LEARNING_DB="./learning_data/history.db"
```

---

## Examples

### Example 1: Quick Repair
```bash
meshprep repair model.stl
# Uses ML prediction + learning
# Output: model_fixed.stl
```

### Example 2: Custom Pipeline
```bash
meshprep repair model.stl --pipeline aggressive
# Forces specific pipeline
```

### Example 3: No Intelligence (Fast)
```bash
meshprep repair model.stl --no-ml --no-learning
# Uses default pipeline only
# Faster but less intelligent
```

### Example 4: Batch Processing
```bash
# Repair all STL files in directory
find . -name "*.stl" -exec meshprep repair {} \;
```

### Example 5: Monitor Statistics
```bash
# After batch processing
meshprep stats

# Check specific pipeline
meshprep stats --pipeline cleanup

# See all actions
meshprep list-actions
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Repair failed |
| 2 | Invalid arguments |

---

## Performance

Typical performance on laptop (i7, 16GB RAM):

| Operation | Time |
|-----------|------|
| Load mesh | ~50ms |
| ML prediction | ~50ms (CPU) / ~5ms (GPU) |
| Repair (cleanup) | ~1-2s |
| Repair (aggressive) | ~5-10s |
| Repair (reconstruction) | ~10-30s |

---

## Troubleshooting

### Command Not Found

```bash
# Make sure you installed with -e flag
pip install -e .

# Or reinstall
pip uninstall meshprep
pip install -e .
```

### Import Errors

```bash
# Install dependencies
pip install -r requirements.txt

# For ML features
pip install torch torchvision

# For all features
pip install -e ".[all]"
```

### Permission Errors

```bash
# On Linux/Mac, may need sudo
sudo pip install -e .

# Or install in user directory
pip install --user -e .
```

---

## Development

### Running from Source

```bash
# Without installation
python -m meshprep.cli.main repair model.stl

# Or
cd poc/v5
python -c "from meshprep.cli import cli; cli()" repair model.stl
```

### Adding New Commands

Edit `meshprep/cli/main.py`:

```python
@cli.command()
def my_command():
    """My custom command."""
    click.echo("Hello from my command!")
```

---

## See Also

- [MeshPrep Documentation](../../README.md)
- [Action Catalog](../../docs/filter_actions.md)
- [Learning System](../learning/README.md)
- [ML Components](../ml/README.md)
