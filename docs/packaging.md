# Packaging Guide

This document describes how resources and assets are organized for packaging MeshPrep.

## Resource Files

Resources (images, config files) are stored within the package at `poc/v1/meshprep/data/`:

```
poc/v1/meshprep/
├── data/
│   ├── images/
│   │   └── MeshPrepLogo.svg
│   └── config/
│       └── filter_library.json
├── resources.py
└── ...
```

### Accessing Resources in Code

Use the `meshprep.resources` module to access resource files:

```python
from meshprep.resources import get_logo_path, get_config_path, get_resource_path

# Get path to the logo
logo_path = get_logo_path()  # Returns Path to data/images/MeshPrepLogo.svg

# Get path to a config file
config_path = get_config_path("filter_library.json")  # Returns Path to data/config/filter_library.json

# Get path to any resource
resource_path = get_resource_path("images/some_image.png")
```

### Why Package Resources?

Resources are stored inside the package (`meshprep/data/`) rather than at the project root for several reasons:

1. **Portability**: Works correctly when installed via pip
2. **PyInstaller compatibility**: The `resources.py` module handles `_MEIPASS` for frozen executables
3. **Clean separation**: Resources travel with the package, not the repository

## Packaging Configuration

### pyproject.toml

When creating a `pyproject.toml` for distribution, include the data directory:

```toml
[tool.setuptools.package-data]
meshprep = ["data/**/*"]
```

### setup.py (alternative)

If using `setup.py`:

```python
from setuptools import setup, find_packages

setup(
    name="meshprep",
    packages=find_packages(),
    package_data={
        "meshprep": ["data/**/*"],
    },
    include_package_data=True,
)
```

### MANIFEST.in (for source distributions)

```
recursive-include meshprep/data *
```

## PyInstaller

When bundling with PyInstaller, the `resources.py` module automatically handles the `_MEIPASS` path. Include the data directory in your `.spec` file:

```python
# In your .spec file
datas=[
    ('meshprep/data', 'meshprep/data'),
],
```

Or use the `--add-data` flag:

```bash
pyinstaller --add-data "meshprep/data:meshprep/data" main.py
```

## Adding New Resources

1. Place the resource file in the appropriate subdirectory under `poc/v1/meshprep/data/`
2. Access it using `get_resource_path("subdirectory/filename.ext")`
3. For common resources, consider adding a convenience function in `resources.py`

### Example: Adding a New Config File

```python
# In resources.py, add:
def get_profiles_config() -> Path:
    """Get path to the profiles configuration file."""
    return get_resource_path("config/profiles.json")
```

## Directory Structure Reference

| Path | Description |
|------|-------------|
| `data/images/` | Image assets (logos, icons) |
| `data/config/` | Configuration files (JSON, YAML) |
| `data/templates/` | Template files (future) |
| `data/presets/` | Built-in filter script presets (future) |
