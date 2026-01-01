# Code Style and Standards

This document defines coding standards and requirements for all source code in the MeshPrep project.

## Source Code Headers

All source code files must include the following license header at the top of the file.

### Python Files (.py)

```python
# Copyright 2025 Dragon Ace
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep
```

### JSON Files (.json) — where comments are supported via preprocessing

For JSON files that support comments (e.g., JSON5 or preprocessed):
```json5
// Copyright 2025 Dragon Ace
// Licensed under the Apache License, Version 2.0 (see LICENSE).
// This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep
```

Note: Standard JSON does not support comments. For `.json` files, include attribution in a top-level `_copyright` field if needed:
```json
{
  "_copyright": "Copyright 2025 Dragon Ace. Apache License 2.0. MeshPrep.",
  ...
}
```

### YAML Files (.yaml, .yml)

```yaml
# Copyright 2025 Dragon Ace
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep
```

### Shell Scripts (.sh, .bash)

```bash
#!/bin/bash
# Copyright 2025 Dragon Ace
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep
```

### PowerShell Scripts (.ps1)

```powershell
# Copyright 2025 Dragon Ace
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep
```

### Batch Files (.bat, .cmd)

```batch
@REM Copyright 2025 Dragon Ace
@REM Licensed under the Apache License, Version 2.0 (see LICENSE).
@REM This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep
```

### QML Files (.qml)

```qml
// Copyright 2025 Dragon Ace
// Licensed under the Apache License, Version 2.0 (see LICENSE).
// This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep
```

### CSS/QSS Files (.css, .qss)

```css
/* Copyright 2025 Dragon Ace
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 * This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep
 */
```

### HTML Files (.html)

```html
<!--
  Copyright 2025 Dragon Ace
  Licensed under the Apache License, Version 2.0 (see LICENSE).
  This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep
-->
```

## Exceptions

The following files do NOT require license headers:

- `LICENSE` — contains the full license text
- `NOTICE` — contains attribution notices
- `README.md` — project readme
- `.gitignore`, `.gitattributes` — git configuration
- `requirements.txt`, `pyproject.toml`, `setup.py` — package configuration
- Data files (`.stl`, `.csv`, `.json` test fixtures)
- Generated files (build outputs, caches)
- Third-party files (must retain original license headers)

## Updating the Year

When making significant changes to a file in a new calendar year, update the copyright year:

- Single year: `Copyright 2025 Dragon Ace`
- Year range: `Copyright 2025-2026 Dragon Ace`

## Enforcement

- Code reviewers should verify headers are present on new files.
- A pre-commit hook or CI check may be added to automate verification.
- See `CONTRIBUTING.md` for the PR checklist.

## Template

Copy-paste template for Python files:

```python
# Copyright 2025 Dragon Ace
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Module description here.
"""
```
