# Code Style and Standards

## License Headers

All source code files must include a license header. Files that do NOT require headers: `LICENSE`, `NOTICE`, `README.md`, `.gitignore`, `requirements.txt`, `pyproject.toml`, data files, generated files, third-party files.

### Header Templates

| File Type | Header |
|-----------|--------|
| Python (.py) | `# Copyright 2025 Dragon Ace`<br>`# Licensed under the Apache License, Version 2.0 (see LICENSE).`<br>`# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep` |
| YAML (.yaml) | Same as Python (use `#` comments) |
| Shell (.sh) | Add `#!/bin/bash` first, then same header |
| PowerShell (.ps1) | Same as Python (use `#` comments) |
| Batch (.bat) | Use `@REM` instead of `#` |
| JSON5/QML/CSS | Use `//` or `/* */` comment syntax |
| HTML | Use `<!-- -->` comment syntax |
| Standard JSON | Use `"_copyright"` field in object |

### Year Updates

- Single year: `Copyright 2025 Dragon Ace`
- Year range: `Copyright 2025-2026 Dragon Ace`

Update when making significant changes in a new calendar year.

---

## Enforcement

- Code reviewers verify headers on new files
- Pre-commit hook or CI check may automate verification
- See `CONTRIBUTING.md` for PR checklist
