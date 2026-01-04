# CLI Reference

## Commands

```bash
meshprep repair      # Repair a mesh
meshprep diagnose    # Show diagnostics
meshprep validate    # Check printability
meshprep list-presets # Show available presets
meshprep checkenv    # Verify environment
```

---

## Repair Command Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--input`, `-i` | path | required | Input mesh file |
| `--output`, `-o` | path | `./output/` | Output directory |
| `--filter` | path | — | Filter script (JSON/YAML) |
| `--preset` | string | — | Named preset from `filters/` |
| `--report` | path | `./report.json` | JSON report output |
| `--csv` | path | `./report.csv` | CSV report output |
| `--export-run` | path | — | Export reproducible run package |
| `--overwrite` | flag | false | Overwrite existing output |
| `--verbose`, `-v` | flag | false | Verbose logging |

### Blender Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--use-blender` | choice | `on-failure` | `always`, `on-failure`, `never` |
| `--blender-path` | path | auto-detect | Path to Blender executable |

### Slicer Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--slicer` | choice | `auto` | `prusa`, `orca`, `cura`, `auto`, `none` |
| `--slicer-config` | path | — | Slicer printer profile |
| `--slicer-repair` | choice | `auto` | `auto`, `prompt`, `never` |
| `--max-repair-attempts` | int | 10 | Max attempts in slicer loop |
| `--skip-slicer-validation` | flag | false | Skip slicer validation |
| `--trust-filter-script` | flag | false | Alias for skip-slicer-validation |

### CAD Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--cad-resolution` | float | 0.01 | Tessellation resolution for STEP/IGES |

---

## Examples

```bash
# Basic repair with auto-detection
meshprep repair -i model.stl -o ./clean/

# Use specific filter script
meshprep repair -i model.stl --filter my_filter.json

# Use named preset
meshprep repair -i model.stl --preset full-repair

# Use PrusaSlicer for validation
meshprep repair -i model.stl --slicer prusa

# Skip slicer validation (trusted filter)
meshprep repair -i model.stl --preset community-verified --trust-filter-script

# Export reproducible run package
meshprep repair -i model.stl --export-run ./share/run1/

# Convert and repair CAD file
meshprep repair -i part.step -o ./clean/ --cad-resolution 0.005
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `MESHPREP_TOOLS_DIR` | Override tools directory |
| `MESHPREP_BLENDER_PATH` | Path to Blender |
| `MESHPREP_SLICER_PATH` | Path to preferred slicer |
| `MESHPREP_AUTO_INSTALL` | Set to `0` to disable auto-install |

---

## See Also

- [Functional Spec](functional_spec.md) - Overview
- [Filter Actions](filter_actions.md) - Action catalog
