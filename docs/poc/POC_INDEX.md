# POC Index

This document provides an overview of all Proof of Concept (POC) implementations for MeshPrep.

---

## Purpose

POCs validate critical technical decisions before full implementation. Each POC answers specific technical questions and identifies risks early.

---

## POC Summary

| POC ID | Name | Priority | Est. Effort | Status | Document |
|--------|------|----------|-------------|--------|----------|
| POC-01 | Format Import | High | 2-3 days | ⬜ Not Started | [Link](POC_01_FORMAT_IMPORT.md) |
| POC-02 | Fingerprinting | High | 1 day | ⬜ Not Started | [Link](POC_02_FINGERPRINT.md) |
| POC-03 | Slicer Integration | High | 1-2 days | ⬜ Not Started | [Link](POC_03_SLICER_INTEGRATION.md) |
| POC-04 | 3D Preview | Medium | 2-3 days | ⬜ Not Started | [Link](POC_04_3D_PREVIEW.md) |
| POC-05 | Mesh Repair | High | 3-5 days | ⬜ Not Started | [Link](POC_05_MESH_REPAIR.md) |
| POC-06 | Geometry Comparison | Medium | 2-3 days | ⬜ Not Started | [Link](POC_06_GEOMETRY_COMPARISON.md) |
| POC-07 | RL Pipeline | High | 5-7 days | ⬜ Not Started | [Link](POC_07_RL_PIPELINE.md) |

**Total Estimated Effort: 17-24 days**

---

## Recommended Execution Order

```
POC-01 (Format Import)
    │
    ▼
POC-03 (Slicer Integration) ────────────────┐
    │                                        │
    ▼                                        │
POC-05 (Mesh Repair) ───────────────────────┤
    │                                        │
    ▼                                        │
POC-02 (Fingerprinting)                      │
    │                                        │
    ▼                                        │
POC-04 (3D Preview)                          │
    │                                        │
    ▼                                        │
POC-06 (Geometry Comparison) ───────────────┤
    │                                        │
    ▼                                        │
POC-07 (RL Pipeline) ◄──────────────────────┘
        (depends on POC-03, POC-05, POC-06)
```

### Rationale

1. **POC-01** first - foundation for loading any models
2. **POC-03** early - slicer validation needed for RL reward
3. **POC-05** early - core repair functionality needed for everything
4. **POC-02** can run in parallel - simple, low risk
5. **POC-04** can run in parallel - UX focused
6. **POC-06** depends on POC-05 - needs mesh comparison from MeshLib
7. **POC-07** last - depends on POC-03, POC-05, POC-06 for complete RL loop

---

## Key Technologies Being Validated

| Technology | POC | Purpose | Risk Level |
|------------|-----|---------|------------|
| Assimp.NET | POC-01 | 40+ format import | Low |
| OpenCascade | POC-01 | STEP/IGES import | Medium |
| SHA-256 | POC-02 | File fingerprinting | Low |
| PrusaSlicer CLI | POC-03 | Slicer validation | Low |
| Helix Toolkit | POC-04 | 3D visualization | Low |
| MeshLib | POC-05, POC-06 | Mesh repair + Hausdorff | Medium |
| TorchSharp | POC-07 | Reinforcement Learning | High |
| CUDA | POC-05, POC-07 | GPU acceleration | Medium |

---

## Risk Assessment

### High Risk POCs

| POC | Risk | Mitigation |
|-----|------|------------|
| POC-07 (RL) | RL training may not converge | Start with simple scenarios, have fallback to rule-based |
| POC-05 (Mesh Repair) | Complex models may fail | Test with Thingi10K dataset, have manual override |

### Medium Risk POCs

| POC | Risk | Mitigation |
|-----|------|------------|
| POC-01 (OpenCascade) | STEP tessellation quality | Test various STEP files, allow tessellation settings |
| POC-06 (Hausdorff) | Performance on large meshes | Use GPU acceleration, sampling if needed |

### Low Risk POCs

| POC | Risk | Mitigation |
|-----|------|------------|
| POC-02 | None - standard crypto | N/A |
| POC-03 | Slicer CLI changes | Support multiple slicers |
| POC-04 | Performance on huge meshes | LOD, virtualization |

---

## POC Code Structure

```
/poc/
├── _POC_TEMPLATE.md
├── POC_INDEX.md
├── POC_01_FORMAT_IMPORT.md
├── POC_02_FINGERPRINT.md
├── POC_03_SLICER_INTEGRATION.md
├── POC_04_3D_PREVIEW.md
├── POC_05_MESH_REPAIR.md
├── POC_06_GEOMETRY_COMPARISON.md
├── POC_07_RL_PIPELINE.md
└── test-data/
    ├── cube.stl
    ├── spaceship.stl
    └── ...
```

**POC Project Structure:**
```
/src/
├── MeshPrep.POC.FormatImport/
├── MeshPrep.POC.Fingerprint/
├── MeshPrep.POC.SlicerIntegration/
├── MeshPrep.POC.Preview/
├── MeshPrep.POC.MeshRepair/
├── MeshPrep.POC.GeometryComparison/
└── MeshPrep.POC.RLPipeline/
```

---

## Success Criteria Summary

| POC | Must Have | Nice to Have |
|-----|-----------|--------------|
| POC-01 | STL, OBJ, STEP import | All 40+ formats |
| POC-02 | Fast SHA-256 hashing | Async with progress |
| POC-03 | PrusaSlicer integration | All 3 slicers |
| POC-04 | 60 FPS at 500K triangles | 60 FPS at 1M+ |
| POC-05 | Hole fill, non-manifold fix | All repair operations |
| POC-06 | Hausdorff calculation | GPU acceleration |
| POC-07 | DQN training works | Agent learns effectively |

---

## Document History

| Date | Update | Author |
|------|--------|--------|
| 2026-01-10 | POC index and all POC documents created | |
