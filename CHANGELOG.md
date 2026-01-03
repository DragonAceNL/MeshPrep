# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- (none)

### Changed
- (none)

## [0.2.0] - 2025-01-03

### Added

#### Reproducibility & Versioning System
- New `reproducibility.py` module with comprehensive version tracking
- `config/compatibility.json` - Tool compatibility matrix defining supported versions
- `config/action_registry.json` - Action registry with version tracking per action
- Three reproducibility levels: Loose, Standard, Strict
- `scripts/checkenv.py` - Environment validation CLI tool
- Environment snapshots capture all tool versions for exact reproduction
- Filter scripts now include `meta` block with:
  - `meshprep_version` - MeshPrep version used
  - `action_registry_version` - Action registry version
  - `tool_versions` - Dictionary of all tool versions (trimesh, pymeshfix, etc.)
  - `created_with` - Python version and platform info
  - `model_fingerprint` - Input file fingerprint for sharing
- Reports now include full `reproducibility` block with all version info
- `FilterScript.populate_version_metadata()` - Auto-populate version info
- `FilterScript.compute_hash()` - Compute stable hash of filter script content
- Compatibility checking against version matrix with warnings/errors
- `check_filter_script_compatibility()` - Validate script against current environment

#### Documentation
- Updated functional spec with comprehensive "Reproducibility & Versioning" section
- Added PyMeshLab stability notes (NumPy 2.x now fully supported)
- Documented reproducibility levels and their use cases
- Added CLI options for reproducibility control

### Changed
- `VERSION` file updated to 0.2.0
- Filter scripts use `meta` block instead of `metadata` (backward compatible)
- `generate_json_report()` now includes reproducibility block by default
- `RepairReport` dataclass now includes reproducibility fields

### Notes
- **PyMeshLab + NumPy 2.x**: PyMeshLab 2023.12.post2 and later (including 2025.7) fully support NumPy 2.x. The previous guidance to pin NumPy ≤1.26.x is outdated.

## [0.1.0] - 2025-01-01

Initial development release.

### Added
- Project scaffolding and documentation
- Functional specification (`docs/functional_spec.md`)
- GUI specification (`docs/gui_spec.md`)
- Model profiles documentation (`docs/model_profiles.md`)
- Code style guidelines (`docs/CODE_STYLE.md`)
- Contributing guidelines (`CONTRIBUTING.md`)
- GitHub Copilot instructions (`.github/copilot-instructions.md`)
- Filter script specification
- GUI mockups and theming
- POC v2 with real mesh operations
- POC v3 batch testing framework
- Learning engine for pipeline optimization
- Adaptive thresholds system
- Pipeline evolution engine
- Profile discovery system
