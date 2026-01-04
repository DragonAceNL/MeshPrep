# Packaging Guide

## Resource Organization

Resources are stored inside the package for portability:

```
src/meshprep/
├── data/
│   ├── images/
│   │   └── MeshPrepLogo.svg
│   └── config/
│       └── filter_library.json
└── resources.py
```

---

## Accessing Resources

```python
from meshprep.resources import get_logo_path, get_config_path, get_resource_path

logo_path = get_logo_path()
config_path = get_config_path("filter_library.json")
resource_path = get_resource_path("images/icon.png")
```

The `resources.py` module handles `_MEIPASS` for PyInstaller compatibility.

---

## Package Configuration

### pyproject.toml

```toml
[tool.setuptools.package-data]
meshprep = ["data/**/*"]
```

### MANIFEST.in

```
recursive-include meshprep/data *
```

---

## PyInstaller

```python
# In .spec file
datas=[('meshprep/data', 'meshprep/data')]
```

Or CLI:
```bash
pyinstaller --add-data "meshprep/data:meshprep/data" main.py
```

---

## Resource Directories

| Path | Description |
|------|-------------|
| `data/images/` | Logos, icons |
| `data/config/` | Configuration files |
| `data/presets/` | Built-in filter presets (future) |
