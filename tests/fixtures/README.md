# Test Fixtures

This directory contains STL files used for testing MeshPrep functionality.

## Naming Convention

- `profile_<profile_name>.stl` — Test fixture for profile detection
- `broken_<issue_type>.stl` — Models with specific issues for repair testing
- `valid_<description>.stl` — Valid models for validation testing

## Adding Fixtures

When adding new fixtures:
1. Use small file sizes where possible
2. Document the purpose and expected characteristics
3. Add corresponding test cases in `tests/`
