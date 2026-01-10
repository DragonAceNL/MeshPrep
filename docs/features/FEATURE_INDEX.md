# Feature Index

This document provides an overview of all features for MeshPrep.

---

## How to Use This Index

1. Each feature has its own document in this folder
2. Features are numbered sequentially (F-001, F-002, etc.)
3. Use the template `_FEATURE_TEMPLATE.md` when creating new feature documents
4. Update this index when adding new features

---

## Feature Summary

| ID | Feature Name | Priority | Status | Effort | POC Required | Document |
|----|--------------|----------|--------|--------|--------------|----------|
| F-001 | Multi-format Import | High | ⬜ Not Started | Large | ✅ POC-01 | [Link](FEATURE_01_FORMAT_IMPORT.md) |
| F-002 | Model Fingerprinting | High | ⬜ Not Started | Small | ✅ POC-02 | [Link](FEATURE_02_FINGERPRINT.md) |
| F-003 | Mesh Analysis | High | ⬜ Not Started | Medium | ⚠️ POC-05 | [Link](FEATURE_03_MESH_ANALYSIS.md) |
| F-004 | ML Filter Generation (RL) | High | ⬜ Not Started | XL | ✅ POC-07 | [Link](FEATURE_04_ML_LEARNING.md) |
| F-005 | Filter Script Application | High | ⬜ Not Started | Medium | ❌ No | [Link](FEATURE_05_FILTER_APPLICATION.md) |
| F-006 | Slicer Validation | High | ⬜ Not Started | Medium | ✅ POC-03 | [Link](FEATURE_06_SLICER_VALIDATION.md) |
| F-007 | Visual Fidelity Check | Medium | ⬜ Not Started | Large | ✅ POC-06 | [Link](FEATURE_07_VISUAL_COMPARISON.md) |
| F-008 | STL Export | High | ⬜ Not Started | Small | ❌ No | [Link](FEATURE_08_STL_EXPORT.md) |
| F-009 | Script Import/Export | High | ⬜ Not Started | Small | ❌ No | [Link](FEATURE_09_SCRIPT_SHARING.md) |
| F-010 | Batch Processing | Medium | ⬜ Not Started | Medium | ❌ No | [Link](FEATURE_10_BATCH_PROCESSING.md) |
| F-011 | 3D Preview | Medium | ⬜ Not Started | Large | ✅ POC-04 | [Link](FEATURE_11_3D_PREVIEW.md) |
| F-012 | User Feedback System | Medium | ⬜ Not Started | Medium | ❌ No | [Link](FEATURE_12_USER_FEEDBACK.md) |
| F-013 | Scaling & Unit Conversion | Medium | ⬜ Not Started | Small | ❌ No | [Link](FEATURE_13_SCALING_UNITS.md) |
| F-014 | Build Plate Orientation | Medium | ⬜ Not Started | Medium | ❌ No | [Link](FEATURE_14_ORIENTATION.md) |
| F-015 | Multi-Part Model Handling | High | ⬜ Not Started | Large | ⚠️ POC-05 | [Link](FEATURE_15_MULTI_PART.md) |
| F-016 | Undo/Redo System | Medium | ⬜ Not Started | Medium | ❌ No | [Link](FEATURE_16_UNDO_REDO.md) |

---

## POC Requirements

| POC ID | Feature Area | Purpose | Priority | Est. Effort | Status |
|--------|--------------|---------|----------|-------------|--------|
| POC-01 | Format Import | Test Assimp.NET + OpenCascade for all target formats | High | 2-3 days | ⬜ Not Started |
| POC-02 | Fingerprinting | Design SHA-256 file hash, test with various formats | High | 1 day | ⬜ Not Started |
| POC-03 | Slicer Integration | CLI calls to PrusaSlicer/Cura/OrcaSlicer, parse validation output | High | 1-2 days | ⬜ Not Started |
| POC-04 | 3D Preview | Helix Toolkit with complex meshes (1M+ triangles), before/after view | Medium | 2-3 days | ⬜ Not Started |
| POC-05 | Mesh Repair | Test geometry3Sharp + alternatives for complex spaceship models | High | 3-5 days | ⬜ Not Started |
| POC-06 | Visual Comparison | Algorithm to score mesh similarity (geometry + appearance) | Medium | 2-3 days | ⬜ Not Started |
| POC-07 | RL Pipeline | Reinforcement learning setup with ML.NET/ONNX, reward system | High | 5-7 days | ⬜ Not Started |

**Recommended POC Order:**
1. POC-01 (Format Import) - Foundation for everything
2. POC-03 (Slicer Integration) - Needed for validation/reward
3. POC-05 (Mesh Repair) - Core functionality
4. POC-02 (Fingerprinting) - Needed for script binding
5. POC-04 (3D Preview) - UX foundation
6. POC-06 (Visual Comparison) - Needed for RL reward
7. POC-07 (RL Pipeline) - Depends on all above

---

## Features by Priority

### High Priority (Core Functionality)
| ID | Feature | Rationale |
|----|---------|-----------|
| F-001 | Multi-format Import | Must read various 3D formats |
| F-002 | Model Fingerprinting | Required for script-model binding |
| F-003 | Mesh Analysis | Needed to identify issues |
| F-004 | ML Filter Generation | Core differentiator - auto-learning with RL |
| F-005 | Filter Script Application | Core functionality of Runner |
| F-006 | Slicer Validation | Ensures output is printable, provides RL reward |
| F-008 | STL Export | Required output format |
| F-009 | Script Import/Export | Required for sharing workflow |
| F-015 | Multi-Part Model Handling | Essential for complex models (spaceships) |

### Medium Priority (Enhanced Experience)
| ID | Feature | Rationale |
|----|---------|-----------|
| F-007 | Visual Fidelity Check | Ensures model looks same after repair, RL reward |
| F-010 | Batch Processing | Power user feature |
| F-011 | 3D Preview | Essential for Studio UX |
| F-012 | User Feedback System | Improves RL over time |
| F-013 | Scaling & Unit Conversion | Common need for downloaded models |
| F-014 | Build Plate Orientation | Print preparation convenience |
| F-016 | Undo/Redo System | Essential for Studio workflow |

---

## Features by Application

### MeshPrep Studio
- F-001: Multi-format Import
- F-002: Model Fingerprinting
- F-003: Mesh Analysis
- F-004: ML Filter Generation (RL)
- F-006: Slicer Validation
- F-007: Visual Fidelity Check
- F-009: Script Import/Export
- F-011: 3D Preview
- F-012: User Feedback System
- F-013: Scaling & Unit Conversion
- F-014: Build Plate Orientation
- F-015: Multi-Part Model Handling
- F-016: Undo/Redo System

### MeshPrep Runner
- F-001: Multi-format Import
- F-002: Model Fingerprinting (verification)
- F-005: Filter Script Application
- F-008: STL Export
- F-009: Script Import/Export
- F-010: Batch Processing (CLI)
- F-011: 3D Preview (GUI only)
- F-013: Scaling & Unit Conversion
- F-015: Multi-Part Model Handling

### Shared Core Library
- F-001: Multi-format Import
- F-002: Model Fingerprinting
- F-003: Mesh Analysis
- F-005: Filter Script Application
- F-008: STL Export
- F-013: Scaling & Unit Conversion
- F-015: Multi-Part Model Handling

---

## Implementation Order (Recommended)

### Phase 0: POC Validation
1. POC-01 - Format Import
2. POC-03 - Slicer Integration
3. POC-05 - Mesh Repair
4. POC-02 - Fingerprinting
5. POC-04 - 3D Preview
6. POC-06 - Visual Comparison
7. POC-07 - RL Pipeline

### Phase 1: Foundation
1. F-001 - Multi-format Import
2. F-008 - STL Export
3. F-003 - Mesh Analysis

### Phase 2: Core Pipeline
4. F-002 - Model Fingerprinting (two-tier)
5. F-005 - Filter Script Application
6. F-009 - Script Import/Export
7. F-015 - Multi-Part Model Handling

### Phase 3: Validation & Comparison
8. F-006 - Slicer Validation
9. F-007 - Visual Fidelity Check

### Phase 4: Intelligence (RL)
10. F-004 - ML Filter Generation
11. F-012 - User Feedback System

### Phase 5: User Experience
12. F-011 - 3D Preview
13. F-016 - Undo/Redo System
14. F-013 - Scaling & Unit Conversion
15. F-014 - Build Plate Orientation
16. F-010 - Batch Processing

---

## Dependencies

```
POC-01 (Import) ──────────────────────────────────────┐
    │                                                 │
    ▼                                                 │
F-001 (Import) ◄──────────────────────────────────────┤
    │                                                 │
    ├──► F-003 (Analysis) ──────────────┐             │
    │         │                         │             │
    │         ▼                         ▼             │
    │    F-015 (Multi-Part)        F-004 (ML/RL) ◄── F-012 (Feedback)
    │         │                         │
    ▼         │                         │
F-002 (Fingerprint) ◄───────────────────┤
    │                                   │
    ▼                                   │
F-005 (Apply Script) ◄──────────────────┘
    │
    ▼
F-008 (Export) ──► F-006 (Slicer Validation)
    │                       │
    │                       ▼
    │              F-007 (Visual Check)
    │
    ▼
F-009 (Script Sharing)
    │
    ▼
F-010 (Batch Processing)

Parallel Development:
├── F-011 (3D Preview) - Start early for UX
├── F-013 (Scaling) - Can develop independently
├── F-014 (Orientation) - Can develop independently
└── F-016 (Undo/Redo) - Can develop independently
```

---

## Notes

- POC phase is critical - validates technology choices before full implementation
- Features F-001 through F-009 + F-015 are required for MVP
- F-011 (3D Preview) should be started early as it affects UX design
- F-004 (RL) is complex and depends on F-006 and F-007 for reward signals
- Multi-Part Handling (F-015) is essential for complex spaceship models
- Two-tier fingerprinting allows both exact match and cross-format matching
