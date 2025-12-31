MeshPrep GUI prototype (PySide6)

Quickstart

1. Open a command prompt or PowerShell and change directory to the repository root where this project lives.

   Example (Windows CMD / PowerShell):

```powershell
cd C:\Users\Dragon Ace\Source\repos\MeshPrep
```

2. Create a virtualenv and install dependencies (this will create `.venv` in the project root).

   Windows (CMD):

```bat
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install PySide6 trimesh pymeshfix meshio
```

   PowerShell:

```powershell
python -m venv .venv
# If PowerShell blocks scripts you may need to allow the session:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install PySide6 trimesh pymeshfix meshio
```

   If a command fails (especially `pymeshfix` on Windows), see the "Conda alternative" below.

3. Run the prototype from the repository root:

```powershell
python poc\gui\v1\app.py
```

Conda alternative (recommended if `pip install pymeshfix` fails)

```bash
conda create -n meshprep python=3.10
conda activate meshprep
conda install -c conda-forge pymeshfix trimesh meshio
python -m pip install PySide6
python poc\gui\v1\app.py
```

Notes
- The commands above assume you run them from the repository root (`C:\Users\Dragon Ace\Source\repos\MeshPrep`).
- You may also create the virtualenv inside `gui/` if you prefer a per-folder venv; if so `cd gui` first and run the same venv commands there.
- This is a UI skeleton for the core screens: environment check, model selection, suggested filter, dry-run, execution, and results.
- Wire-up to real model scanning, filter script generation, and action implementations is required.
