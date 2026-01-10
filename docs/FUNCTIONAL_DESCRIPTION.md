# MeshPrep - Functional Description

## 1. Introduction

### 1.1 Purpose
MeshPrep is a two-part toolset designed to convert 3D models of any format into printable STL files while respecting intellectual property rights. Instead of sharing modified STL files (which may violate licenses), MeshPrep enables users to create and share **filter scripts** - reproducible repair recipes tied to specific models via unique fingerprints.

### 1.2 Scope

**In Scope:**
- Import 3D models from various formats (STL, OBJ, 3MF, STEP, IGES, FBX, GLTF, etc.)
- Analyze mesh issues (non-manifold, holes, self-intersections, etc.)
- Auto-learn optimal repair strategies using machine learning
- Generate and apply filter scripts tied to model fingerprints
- Validate output against slicer requirements
- Ensure repaired model maintains visual fidelity to original
- Export printable STL files

**Out of Scope:**
- 3D modeling/sculpting capabilities
- Slicing (handled by external slicers)
- Direct hosting of filter scripts (uses external platforms like Reddit, Discord)

### 1.3 Target Users

| User Type | Description | Primary Tool |
|-----------|-------------|--------------|
| **Filter Creators** | Advanced users who develop repair strategies for specific models | MeshPrep.FilterScriptCreator |
| **End Users** | Users who apply existing filter scripts to make models printable | MeshPrep.ModelFixer |
| **Automation Users** | Users who batch process multiple models | MeshPrep.ModelFixer CLI |

---

## 2. System Overview

### 2.1 High-Level Description
MeshPrep consists of two complementary applications:

1. **MeshPrep.FilterScriptCreator** - A GUI application for analyzing models, developing filter scripts, and training the ML system
2. **MeshPrep.ModelFixer** - A GUI + CLI application for applying filter scripts to models

### 2.2 System Context

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MeshPrep Ecosystem                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                  MeshPrep.FilterScriptCreator (GUI)               │   │
│  │                                                                     │   │
│  │  [3D Model] ──► [Analysis] ──► [ML Learning] ──► [Filter Script]   │   │
│  │                      │              │                    │          │   │
│  │                      ▼              ▼                    ▼          │   │
│  │               [Issue Detection] [Slicer Test]    [Fingerprint ID]   │   │
│  │                                 [User Feedback]                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                        │                                    │
│                                        ▼                                    │
│                    ┌──────────────────────────────────┐                    │
│                    │   Filter Script + Fingerprint    │                    │
│                    │   (Shareable via Reddit/Discord) │                    │
│                    └──────────────────────────────────┘                    │
│                                        │                                    │
│                                        ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                   MeshPrep.ModelFixer (GUI + CLI)                  │   │
│  │                                                                     │   │
│  │  [3D Model] + [Filter Script] ──► [Apply] ──► [Printable STL]      │   │
│  │       │              │                                              │   │
│  │       ▼              ▼                                              │   │
│  │  [Fingerprint] ═══ [Match?]                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Legal Workflow Advantage

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Traditional Approach (Often Violates License)                              │
│  ─────────────────────────────────────────────                              │
│  Download Model ──► Fix Manually ──► Share Fixed STL ❌                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  MeshPrep Approach (License Compliant)                                      │
│  ─────────────────────────────────────                                      │
│  Creator: Model ──► MeshPrep.FilterScriptCreator ──► Share Filter Script ✓     │
│  User:    Model + Filter Script ──► MeshPrep.ModelFixer ──► Own STL ✓         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Functional Requirements

### 3.1 Core Functionality

| ID | Requirement | Priority | Feature Doc |
|----|-------------|----------|-------------|
| FR-001 | Import models from multiple 3D formats | High | [FEATURE_01](features/FEATURE_01_FORMAT_IMPORT.md) |
| FR-002 | Generate unique model fingerprint | High | [FEATURE_02](features/FEATURE_02_FINGERPRINT.md) |
| FR-003 | Analyze mesh for printability issues | High | [FEATURE_03](features/FEATURE_03_MESH_ANALYSIS.md) |
| FR-004 | ML-based filter script generation | High | [FEATURE_04](features/FEATURE_04_ML_LEARNING.md) |
| FR-005 | Apply filter scripts to models | High | [FEATURE_05](features/FEATURE_05_FILTER_APPLICATION.md) |
| FR-006 | Slicer validation integration | High | [FEATURE_06](features/FEATURE_06_SLICER_VALIDATION.md) |
| FR-007 | Visual fidelity comparison | Medium | [FEATURE_07](features/FEATURE_07_VISUAL_COMPARISON.md) |
| FR-008 | Export printable STL | High | [FEATURE_08](features/FEATURE_08_STL_EXPORT.md) |
| FR-009 | Filter script import/export | High | [FEATURE_09](features/FEATURE_09_SCRIPT_SHARING.md) |
| FR-010 | Batch processing (CLI) | Medium | [FEATURE_10](features/FEATURE_10_BATCH_PROCESSING.md) |
| FR-011 | 3D preview with before/after | Medium | [FEATURE_11](features/FEATURE_11_3D_PREVIEW.md) |
| FR-012 | User feedback for ML training | Medium | [FEATURE_12](features/FEATURE_12_USER_FEEDBACK.md) |
| FR-013 | Model scaling and unit conversion | Medium | [FEATURE_13](features/FEATURE_13_SCALING_UNITS.md) |
| FR-014 | Build plate orientation/placement | Medium | [FEATURE_14](features/FEATURE_14_ORIENTATION.md) |
| FR-015 | Multi-part model handling | High | [FEATURE_15](features/FEATURE_15_MULTI_PART.md) |
| FR-016 | Undo/redo system (Creator) | Medium | [FEATURE_16](features/FEATURE_16_UNDO_REDO.md) |

### 3.2 Input Requirements

| Input Type | Format | Description |
|------------|--------|-------------|
| Mesh Files | STL, OBJ, PLY, 3MF, FBX, GLTF/GLB | Standard mesh formats from 3D modeling, scanning, downloads |
| CAD Files | STEP, IGES | Engineering/CAD formats requiring tessellation |
| Filter Scripts | JSON (custom format) | MeshPrep filter script files containing repair instructions |
| User Feedback | GUI input | Accept/reject results, quality ratings for ML training |

### 3.3 Output Requirements

| Output Type | Format | Description |
|-------------|--------|-------------|
| Printable Mesh | STL (binary/ASCII) | Watertight, manifold mesh ready for slicing |
| Filter Script | JSON + Fingerprint | Shareable repair recipe tied to specific model |
| Fingerprint ID | String (hash-based) | Unique identifier for model matching and online search |
| Validation Report | JSON/HTML | Slicer compatibility and visual fidelity metrics |

### 3.4 Processing Requirements

**Mesh Repair Operations:**
- Hole filling (various algorithms)
- Non-manifold edge repair
- Self-intersection removal
- Duplicate vertex/face removal
- Normal orientation fixing
- Watertight conversion
- Mesh simplification/decimation
- Mesh smoothing (Laplacian, Taubin)
- Component merging/separation
- Remeshing (voxel-based, isotropic)
- Boolean operations (union, difference, intersection)
- Shell/wall thickness enforcement
- Degenerate face removal
- Edge collapse/split
- Vertex welding

**Target Model Complexity:**
- Simple printable models (Thingiverse, etc.)
- Complex game/render assets (spaceships, characters)
- High-poly sculpts
- CAD models with NURBS tessellation
- Thingi10K dataset models for ML training

**ML Processing (Reinforcement Learning Approach):**
- Model fingerprint extraction
- Issue classification
- Repair strategy prediction via RL agent
- Trial-and-error learning from repair attempts
- Reward signal from slicer validation
- Reward signal from visual fidelity score
- Reward signal from user feedback
- State: mesh issues + repair history
- Actions: available repair operations
- Policy: learned optimal repair sequences

---

## 4. Non-Functional Requirements

### 4.1 Performance
- Import models up to 10M triangles within 30 seconds
- Apply filter scripts within 2 minutes for typical models
- ML prediction response within 5 seconds
- GUI remains responsive during processing (async operations)

### 4.2 Usability
- Intuitive drag-and-drop model import
- One-click filter script application
- Clear before/after visual comparison
- Easy copy-paste of fingerprint IDs for sharing
- CLI with simple, scriptable commands

### 4.3 Reliability
- Graceful handling of corrupted/invalid input files
- Automatic backup before destructive operations
- Recovery from failed repair attempts
- Consistent fingerprints across runs (deterministic)

### 4.4 Compatibility
- **Platform**: Windows (primary), with potential for cross-platform
- **Framework**: .NET 10 / C#
- **Slicer Integration**: PrusaSlicer, Cura, OrcaSlicer (validation)
- **3D Libraries**: geometry3Sharp, Assimp.NET, OpenCascade.NET

---

## 5. User Interface

### 5.1 Interface Type

| Application | Interface | Description |
|-------------|-----------|-------------|
| MeshPrep.FilterScriptCreator | GUI | Full-featured interface for filter script development |
| MeshPrep.ModelFixer | GUI | Simple interface for applying scripts |
| MeshPrep.ModelFixer | CLI | Command-line for automation and batch processing |

### 5.2 User Interactions

**MeshPrep.FilterScriptCreator (GUI):**
```
┌─────────────────────────────────────────────────────────────────────────┐
│  MeshPrep.FilterScriptCreator                               [─][□][×]  │
├─────────────────────────────────────────────────────────────────────────┤
│  [File] [Edit] [Tools] [Help]                                          │
├───────────────────────┬─────────────────────────────────────────────────┤
│                       │                                                 │
│   Model Browser       │           3D Preview                            │
│   ─────────────       │   ┌─────────────────┬─────────────────┐        │
│   📁 Recent           │   │                 │                 │        │
│   📁 Import           │   │    Original     │    Repaired     │        │
│                       │   │                 │                 │        │
│   Issues Detected     │   └─────────────────┴─────────────────┘        │
│   ─────────────────   │                                                 │
│   ⚠ Non-manifold: 12  │   Fingerprint: [abc123xyz...] [📋 Copy]        │
│   ⚠ Holes: 3          │                                                 │
│   ⚠ Self-intersect: 0 ├─────────────────────────────────────────────────┤
│                       │   Filter Script Actions                         │
│   ML Suggestion       │   ────────────────────                          │
│   ─────────────────   │   1. [Fix Normals    ] [▲][▼][×]               │
│   Confidence: 87%     │   2. [Fill Holes     ] [▲][▼][×]               │
│   [Apply Suggested]   │   3. [Make Watertight] [▲][▼][×]               │
│                       │   [+ Add Action]                                │
│   User Feedback       │                                                 │
│   ─────────────────   │   [Test with Slicer] [Export Script] [Apply]   │
│   [👍 Good] [👎 Bad]  │                                                 │
│                       │                                                 │
└───────────────────────┴─────────────────────────────────────────────────┘
```

**MeshPrep.ModelFixer (GUI):**
```
┌─────────────────────────────────────────────────────────────────────────┐
│  MeshPrep.ModelFixer                                        [─][□][×]  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                                                                 │  │
│   │              Drag & Drop Model Here                             │  │
│   │                     or                                          │  │
│   │                [Browse Files]                                   │  │
│   │                                                                 │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│   Model: dragon_statue.obj                                              │
│   Fingerprint: abc123xyz789...                                          │
│                                                                         │
│   Filter Script: [None loaded           ] [Browse] [Paste ID]          │
│                                                                         │
│   Status: Ready                                                         │
│                                                                         │
│                    [Apply & Export STL]                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**MeshPrep.ModelFixer (CLI):**
```bash
# Apply filter script to model
meshprep run model.obj --script filter.json --output model_fixed.stl

# Get model fingerprint (for finding scripts online)
meshprep fingerprint model.obj

# Batch process directory
meshprep batch ./models/ --script filter.json --output ./fixed/

# Validate against slicer
meshprep validate model.stl --slicer prusaslicer
```

---

## 6. Data Requirements

### 6.1 Data Formats

**Filter Script Format (JSON):**
```json
{
  "version": "1.0",
  "fingerprint": "abc123xyz789def456...",
  "model_name": "dragon_statue",
  "created": "2024-01-15T10:30:00Z",
  "author": "username",
  "actions": [
    {
      "action": "fix_normals",
      "params": {}
    },
    {
      "action": "fill_holes",
      "params": { "max_hole_size": 100 }
    },
    {
      "action": "make_watertight",
      "params": { "method": "voxel", "resolution": 256 }
    }
  ],
  "validation": {
    "slicer": "prusaslicer",
    "passed": true,
    "visual_similarity": 0.97
  }
}
```

**Model Fingerprint:**
- SHA-256 hash of file contents
- Fast to compute
- Deterministic - same file always produces same fingerprint
- Used for: exact script matching, online search
- Format: Hex string (64 characters), easy to copy/paste and search online
- Example: `a1b2c3d4e5f6...` (truncated for display in UI)

### 6.2 Data Storage

| Data | Storage | Purpose |
|------|---------|---------|
| ML Training Data | Local SQLite database | Store successful repair strategies |
| User Preferences | Local config file | Settings, recent files, slicer paths |
| Filter Script Cache | Local folder | Quick access to recently used scripts |
| Fingerprint Index | Local SQLite | Map fingerprints to known scripts |

---

## 7. Constraints & Assumptions

### 7.1 Constraints
- **No STL distribution**: The tool does not host or distribute repaired STL files
- **Script-model binding**: Filter scripts are tied to specific model fingerprints
- **External slicer**: Slicer validation requires installed slicer software
- **Windows first**: Initial development targets Windows platform
- **Internet optional**: Core functionality works offline; sharing requires manual copy/paste

### 7.2 Assumptions
- Users have legal access to the original 3D models
- Users will share filter scripts through existing platforms (Reddit, Discord, forums)
- Model fingerprints are sufficiently unique to avoid collisions
- Slicer validation is a reliable proxy for printability
- Visual similarity can be quantified for ML training

---

## 8. Feature List Summary

| Feature ID | Feature Name | Priority | Status | Document |
|------------|--------------|----------|--------|----------|
| F-001 | Multi-format Import | High | Not Started | [FEATURE_01](features/FEATURE_01_FORMAT_IMPORT.md) |
| F-002 | Model Fingerprinting | High | Not Started | [FEATURE_02](features/FEATURE_02_FINGERPRINT.md) |
| F-003 | Mesh Analysis | High | Not Started | [FEATURE_03](features/FEATURE_03_MESH_ANALYSIS.md) |
| F-004 | ML Filter Generation | High | Not Started | [FEATURE_04](features/FEATURE_04_ML_LEARNING.md) |
| F-005 | Filter Script Application | High | Not Started | [FEATURE_05](features/FEATURE_05_FILTER_APPLICATION.md) |
| F-006 | Slicer Validation | High | Not Started | [FEATURE_06](features/FEATURE_06_SLICER_VALIDATION.md) |
| F-007 | Visual Fidelity Check | Medium | Not Started | [FEATURE_07](features/FEATURE_07_VISUAL_COMPARISON.md) |
| F-008 | STL Export | High | Not Started | [FEATURE_08](features/FEATURE_08_STL_EXPORT.md) |
| F-009 | Script Import/Export | High | Not Started | [FEATURE_09](features/FEATURE_09_SCRIPT_SHARING.md) |
| F-010 | Batch Processing | Medium | Not Started | [FEATURE_10](features/FEATURE_10_BATCH_PROCESSING.md) |
| F-011 | 3D Preview | Medium | Not Started | [FEATURE_11](features/FEATURE_11_3D_PREVIEW.md) |
| F-012 | User Feedback System | Medium | Not Started | [FEATURE_12](features/FEATURE_12_USER_FEEDBACK.md) |
| F-013 | Scaling & Unit Conversion | Medium | Not Started | [FEATURE_13](features/FEATURE_13_SCALING_UNITS.md) |
| F-014 | Build Plate Orientation | Medium | Not Started | [FEATURE_14](features/FEATURE_14_ORIENTATION.md) |
| F-015 | Multi-Part Model Handling | High | Not Started | [FEATURE_15](features/FEATURE_15_MULTI_PART.md) |
| F-016 | Undo/Redo System | Medium | Not Started | [FEATURE_16](features/FEATURE_16_UNDO_REDO.md) |

---

## 9. Technology Stack

### 9.1 Chosen Platform
- **Language**: C# / .NET 10
- **GUI Framework**: WPF or AvaloniaUI (cross-platform option)
- **3D Viewport**: Helix Toolkit

### 9.2 Core Libraries

| Purpose | Library | Notes |
|---------|---------|-------|
| Mesh I/O | Assimp.NET | 40+ format support |
| CAD Import | OpenCascade.NET | STEP/IGES support |
| Mesh Operations | geometry3Sharp | Native C# mesh processing |
| Heavy Repair | CGAL (via C++/CLI) | Complex geometry operations |
| ML Framework | ML.NET or ONNX Runtime | Machine learning |
| 3D Preview | Helix Toolkit | WPF 3D visualization |
| Database | SQLite | Local storage |

---

## 10. Glossary

| Term | Definition |
|------|------------|
| Mesh | A 3D model represented as vertices, edges, and faces (triangles) |
| Manifold | A mesh where every edge is shared by exactly two faces |
| Watertight | A mesh with no holes - completely enclosed volume |
| Filter Script | A JSON file containing ordered repair actions tied to a model fingerprint |
| Fingerprint | A unique hash-based identifier for a specific 3D model |
| Slicer | Software that converts 3D models to printer instructions (G-code) |
| Non-manifold | Geometry errors where edges have more or fewer than 2 adjacent faces |
| Self-intersection | When faces of a mesh pass through other faces of the same mesh |
| Visual Fidelity | Measure of how similar the repaired model looks to the original |

---

## 11. POC Requirements

The following features require proof-of-concept implementation before full development:

| POC ID | Feature Area | Purpose | Priority | Est. Effort |
|--------|--------------|---------|----------|-------------|
| POC-01 | Format Import | Test Assimp.NET + OpenCascade for all target formats | High | 2-3 days |
| POC-02 | Fingerprinting | Design dual-tier hash algorithm, test cross-format consistency | High | 2-3 days |
| POC-03 | Slicer Integration | CLI calls to PrusaSlicer/Cura/OrcaSlicer, parse validation output | High | 1-2 days |
| POC-04 | 3D Preview | Helix Toolkit with complex meshes (1M+ triangles), before/after view | Medium | 2-3 days |
| POC-05 | Mesh Repair | Test geometry3Sharp + alternatives for complex spaceship models | High | 3-5 days |
| POC-06 | Visual Comparison | Algorithm to score mesh similarity (geometry + appearance) | Medium | 2-3 days |
| POC-07 | RL Pipeline | Reinforcement learning setup with ML.NET/ONNX, reward system | High | 5-7 days |

---

## 12. Training Data Strategy

### 12.1 Thingi10K Dataset
- ~10,000 models with known mesh issues
- Variety of complexity levels
- Can be used for supervised pre-training
- Source: https://ten-thousand-models.appspot.com/

### 12.2 Complex Model Sources
- Game asset repositories (with appropriate licenses)
- Spaceship/vehicle models
- High-poly character sculpts
- User-submitted problem models

### 12.3 Learning Loop
```
┌─────────────────────────────────────────────────────────────────┐
│                   Reinforcement Learning Loop                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   [Model + Issues] ──► [RL Agent] ──► [Select Action]          │
│                            │                                    │
│                            ▼                                    │
│                    [Apply Repair Action]                        │
│                            │                                    │
│              ┌─────────────┼─────────────┐                     │
│              ▼             ▼             ▼                     │
│        [Slicer Test] [Visual Check] [User Feedback]            │
│              │             │             │                     │
│              └─────────────┼─────────────┘                     │
│                            ▼                                    │
│                    [Calculate Reward]                           │
│                            │                                    │
│                            ▼                                    │
│                    [Update Policy]                              │
│                            │                                    │
│                            ▼                                    │
│                    [Next Episode]                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-01-10 | | Initial draft |
| 0.2 | 2026-01-10 | | Added two-program architecture, ML learning, fingerprinting |
| 0.3 | 2026-01-10 | | Added RL approach, two-tier fingerprint, 4 new features, POC requirements |
