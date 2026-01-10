# MeshPrep Development Tracker

## Project Overview
**Project Name:** MeshPrep  
**Start Date:** 2026-01-10  
**Current Phase:** Planning & Documentation  
**Technology Stack:** C# / .NET 10 / WPF / Helix Toolkit  
**ML Approach:** Reinforcement Learning

---

## Architecture Overview

MeshPrep consists of two applications sharing a common core library:

```
┌─────────────────────────────────────────────────────────────┐
│                    MeshPrep Solution                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │ MeshPrep.Studio │  │ MeshPrep.Runner │                  │
│  │     (GUI)       │  │   (GUI + CLI)   │                  │
│  └────────┬────────┘  └────────┬────────┘                  │
│           │                    │                            │
│           └──────────┬─────────┘                            │
│                      ▼                                      │
│           ┌─────────────────────┐                          │
│           │   MeshPrep.Core     │                          │
│           │  (Shared Library)   │                          │
│           └─────────────────────┘                          │
│                      │                                      │
│    ┌─────────────────┼─────────────────┐                   │
│    ▼                 ▼                 ▼                   │
│ [Assimp.NET]  [geometry3Sharp]  [ML.NET/ONNX]              │
│ [OpenCascade] [CGAL wrapper]    [SQLite]                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Development Phases

### Phase 0: POC Validation (NEW)
| POC ID | Feature Area | Status | Est. Effort | Notes |
|--------|--------------|--------|-------------|-------|
| POC-01 | Format Import | ⬜ Not Started | 2-3 days | Assimp.NET + OpenCascade |
| POC-02 | Fingerprinting | ⬜ Not Started | 1 day | SHA-256 file hash |
| POC-03 | Slicer Integration | ⬜ Not Started | 1-2 days | CLI validation |
| POC-04 | 3D Preview | ⬜ Not Started | 2-3 days | Helix Toolkit, 1M+ triangles |
| POC-05 | Mesh Repair | ⬜ Not Started | 3-5 days | geometry3Sharp + complex models |
| POC-06 | Visual Comparison | ⬜ Not Started | 2-3 days | Similarity algorithm |
| POC-07 | RL Pipeline | ⬜ Not Started | 5-7 days | ML.NET/ONNX, reward system |

**Recommended POC Order:** POC-01 → POC-03 → POC-05 → POC-02 → POC-04 → POC-06 → POC-07

### Phase 1: Planning & Documentation
| Task | Status | Notes |
|------|--------|-------|
| Create project structure | ✅ Complete | Directories created |
| Functional Description Document | ✅ Complete | Full spec with RL approach |
| Feature Documents | 🔄 In Progress | Index created, 16 features defined |
| Technology Stack Decision | ✅ Complete | C# / .NET 10 |
| POC Planning | ✅ Complete | 7 POCs identified |

### Phase 2: Core Infrastructure
| Task | Status | Notes |
|------|--------|-------|
| Solution & Project Setup | ⬜ Not Started | .NET 10 solution structure |
| Core Library Architecture | ⬜ Not Started | Interfaces and base classes |
| Dependency Integration | ⬜ Not Started | NuGet packages, native wrappers |
| Unit Test Framework | ⬜ Not Started | xUnit setup |

### Phase 3: Feature Implementation

#### Phase 3.1: Foundation
| Feature | Document | Status | Notes |
|---------|----------|--------|-------|
| F-001 Multi-format Import | [Link](features/FEATURE_01_FORMAT_IMPORT.md) | ⬜ Not Started | Assimp.NET + OpenCascade |
| F-008 STL Export | [Link](features/FEATURE_08_STL_EXPORT.md) | ⬜ Not Started | Binary & ASCII support |
| F-003 Mesh Analysis | [Link](features/FEATURE_03_MESH_ANALYSIS.md) | ⬜ Not Started | Issue detection |

#### Phase 3.2: Core Pipeline
| Feature | Document | Status | Notes |
|---------|----------|--------|-------|
| F-002 Model Fingerprinting | [Link](features/FEATURE_02_FINGERPRINT.md) | ⬜ Not Started | SHA-256 file hash |
| F-005 Filter Script Application | [Link](features/FEATURE_05_FILTER_APPLICATION.md) | ⬜ Not Started | JSON script execution |
| F-009 Script Import/Export | [Link](features/FEATURE_09_SCRIPT_SHARING.md) | ⬜ Not Started | File I/O |
| F-015 Multi-Part Handling | [Link](features/FEATURE_15_MULTI_PART.md) | ⬜ Not Started | Complex models |

#### Phase 3.3: Validation
| Feature | Document | Status | Notes |
|---------|----------|--------|-------|
| F-006 Slicer Validation | [Link](features/FEATURE_06_SLICER_VALIDATION.md) | ⬜ Not Started | PrusaSlicer, Cura, OrcaSlicer |
| F-007 Visual Fidelity Check | [Link](features/FEATURE_07_VISUAL_COMPARISON.md) | ⬜ Not Started | Similarity scoring, RL reward |

#### Phase 3.4: Intelligence (Reinforcement Learning)
| Feature | Document | Status | Notes |
|---------|----------|--------|-------|
| F-004 ML Filter Generation | [Link](features/FEATURE_04_ML_LEARNING.md) | ⬜ Not Started | RL agent, trial-and-error |
| F-012 User Feedback System | [Link](features/FEATURE_12_USER_FEEDBACK.md) | ⬜ Not Started | RL reward signal |

#### Phase 3.5: User Experience
| Feature | Document | Status | Notes |
|---------|----------|--------|-------|
| F-011 3D Preview | [Link](features/FEATURE_11_3D_PREVIEW.md) | ⬜ Not Started | Helix Toolkit |
| F-010 Batch Processing | [Link](features/FEATURE_10_BATCH_PROCESSING.md) | ⬜ Not Started | CLI implementation |
| F-013 Scaling & Units | [Link](features/FEATURE_13_SCALING_UNITS.md) | ⬜ Not Started | Unit conversion |
| F-014 Build Plate Orientation | [Link](features/FEATURE_14_ORIENTATION.md) | ⬜ Not Started | Print placement |
| F-016 Undo/Redo | [Link](features/FEATURE_16_UNDO_REDO.md) | ⬜ Not Started | Studio only |

### Phase 4: Application Assembly
| Task | Status | Notes |
|------|--------|-------|
| MeshPrep.Studio GUI | ⬜ Not Started | WPF application |
| MeshPrep.Runner GUI | ⬜ Not Started | Simplified WPF app |
| MeshPrep.Runner CLI | ⬜ Not Started | Command-line interface |

### Phase 5: Testing & Refinement
| Task | Status | Notes |
|------|--------|-------|
| Unit Tests | ⬜ Not Started | |
| Integration Tests | ⬜ Not Started | |
| Performance Testing | ⬜ Not Started | Complex spaceship models |
| Thingi10K Testing | ⬜ Not Started | RL training validation |
| User Acceptance Testing | ⬜ Not Started | |

### Phase 6: Release
| Task | Status | Notes |
|------|--------|-------|
| Documentation Finalization | ⬜ Not Started | User guide, API docs |
| Installer Creation | ⬜ Not Started | MSI or MSIX |
| Release Build | ⬜ Not Started | |

---

## Status Legend
- ✅ Complete
- 🔄 In Progress
- ⬜ Not Started
- ❌ Blocked
- 🔙 Deferred

---

## Milestones

| Milestone | Target Date | Status | Description |
|-----------|-------------|--------|-------------|
| M0: POC Complete | TBD | ⬜ Not Started | All 7 POCs validated |
| M1: Documentation Complete | TBD | 🔄 In Progress | All feature docs written |
| M2: Core Library MVP | TBD | ⬜ Not Started | Import, Export, Analysis working |
| M3: Filter Pipeline | TBD | ⬜ Not Started | Scripts can be created and applied |
| M4: RL Integration | TBD | ⬜ Not Started | RL agent learning from repairs |
| M5: Studio Alpha | TBD | ⬜ Not Started | Full Studio GUI functional |
| M6: Runner Alpha | TBD | ⬜ Not Started | GUI + CLI functional |
| M7: Beta Release | TBD | ⬜ Not Started | Feature complete, testing |
| M8: v1.0 Release | TBD | ⬜ Not Started | Production ready |

---

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-01-10 | Initial project structure created | |
| 2026-01-10 | Documentation framework established | |
| 2026-01-10 | Functional description completed | |
| 2026-01-10 | Feature index created with 12 features | |
| 2026-01-10 | Added 4 new features (F-013 to F-016) | |
| 2026-01-10 | Added POC phase with 7 POCs | |
| 2026-01-10 | Defined Reinforcement Learning approach | |
| 2026-01-10 | Simplified to single-tier fingerprinting (SHA-256 file hash) | |

---

## Notes & Decisions

### Architecture Decisions
| Decision | Rationale | Date |
|----------|-----------|------|
| C# / .NET 10 | Good Windows GUI support, can call C++ libraries, current LTS | 2026-01-10 |
| Two separate applications | Different user needs: creators vs consumers | 2026-01-10 |
| Shared core library | Code reuse, consistent behavior | 2026-01-10 |
| JSON filter scripts | Human-readable, easy to share/edit | 2026-01-10 |
| Fingerprint-based binding | Legal compliance, prevents wrong script usage | 2026-01-10 |
| Reinforcement Learning | Learns from trial/error, no labeled data needed | 2026-01-10 |
| Single-tier fingerprinting | SHA-256 file hash only; geometry hash adds complexity without benefit for online search | 2026-01-10 |
| Thingi10K for training | Large dataset with various mesh issues | 2026-01-10 |

### Open Questions
- [x] Which ML approach? → **Reinforcement Learning**
- [ ] Which RL framework: ML.NET vs ONNX Runtime vs custom?
- [ ] Cross-platform: WPF (Windows only) vs AvaloniaUI (cross-platform)?
- [ ] CGAL integration: C++/CLI wrapper vs subprocess?
- [x] Fingerprint algorithm? → **SHA-256 file hash (single-tier)**
- [ ] Visual similarity algorithm: Hausdorff distance? Image-based? Both?

### Blockers
*None currently*

---

## Next Steps

1. ⬜ Create individual feature documents (F-001 through F-016)
2. ⬜ Create POC document templates
3. ⬜ Start POC-01: Format Import
4. ⬜ Set up .NET 10 solution structure
5. ⬜ Download Thingi10K sample models for testing
