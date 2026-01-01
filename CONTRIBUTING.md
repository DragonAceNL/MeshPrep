# Contributing to MeshPrep

Thank you for your interest in contributing to MeshPrep! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributors of all skill levels.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment (see `docs/INSTALL.md`)
4. Create a new branch for your feature or fix

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/MeshPrep.git
cd MeshPrep

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# or: source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Verify environment
python scripts/checkenv.py
```

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/filter-editor-undo` — new features
- `fix/hole-filling-crash` — bug fixes
- `docs/update-install-guide` — documentation
- `refactor/action-registry` — code refactoring

### Code Style

- Follow PEP 8 for Python code
- **All source files must include the license header** — see `docs/CODE_STYLE.md`
- Use type hints for function signatures
- Write docstrings for public functions and classes
- Keep functions focused and reasonably sized

### License Header Reminder

Every new source file must start with:

```python
# Copyright 2025 Dragon Ace
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep
```

See `docs/CODE_STYLE.md` for headers in other file formats.

### Testing

- Add tests for new features in `tests/`
- Run existing tests before submitting: `pytest tests/`
- Include test fixtures in `tests/fixtures/` if needed

### Documentation

- Update relevant docs in `docs/` for feature changes
- Add docstrings to new public APIs
- Update `README.md` if adding major features

## Pull Request Process

### Before Submitting

Use this checklist:

- [ ] Code follows the project style guidelines
- [ ] All new files have the license header
- [ ] Tests pass locally (`pytest tests/`)
- [ ] Documentation updated (if applicable)
- [ ] `requirements.txt` updated (if dependencies changed)
- [ ] `VERSION` bumped (if applicable)
- [ ] `CHANGELOG.md` updated (if applicable)

### Submitting

1. Push your branch to your fork
2. Open a Pull Request against `main`
3. Fill out the PR template
4. Wait for review

### PR Title Format

Use conventional commit style:
- `feat: add undo/redo to filter editor`
- `fix: handle empty STL files gracefully`
- `docs: update installation guide for Windows`
- `refactor: simplify action registry lookup`
- `test: add fixtures for multi-shell models`

## Contributing Filter Presets

To contribute a community filter preset:

1. Create your filter script and test it
2. Add metadata (author, description, tags)
3. Include a test fixture STL (or reference a public model)
4. Add a minimal test case
5. Submit a PR with:
   - Filter script in `filters/community/`
   - Test fixture in `tests/fixtures/`
   - Test case in `tests/`

### Preset Requirements

- Must include complete metadata
- Must work with the current driver version
- Should include a brief explanation of use case
- Should reference a reproducible test model

## Reporting Issues

When reporting bugs:

1. Check existing issues first
2. Include MeshPrep version (`cat VERSION`)
3. Include `checkenv.py` output
4. Attach the `report.json` if available
5. Provide a minimal STL file that reproduces the issue (if possible)

## Questions?

- Open a GitHub Discussion for questions
- Tag issues appropriately (`bug`, `enhancement`, `question`)

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
