# Feature F-001: Multi-format Import

---

## Feature ID: F-001

## Feature Name
Multi-format 3D Model Import

## Status
- [x] Not Started
- [ ] In Analysis
- [ ] In Development
- [ ] In Testing
- [ ] Complete

## Priority
**High** - Foundation for all other features

## Estimated Effort
**Large** (3-7 days)

## Related POC
**POC-01** - Format Import validation

---

## 1. Description

### 1.1 Overview
Import 3D models from various file formats commonly used in 3D printing workflows. This feature is the entry point for all mesh processing in MeshPrep.

### 1.2 User Story

As a **3D printing enthusiast**, I want **to import models from any common 3D format** so that **I can repair and prepare them for printing regardless of their source**.

### 1.3 Acceptance Criteria

- [ ] Import STL files (binary and ASCII)
- [ ] Import OBJ files (with MTL materials optional)
- [ ] Import 3MF files
- [ ] Import STEP/STP files (via OpenCascade tessellation)
- [ ] Import IGES files (via OpenCascade tessellation)
- [ ] Import PLY files
- [ ] Import FBX files
- [ ] Import glTF/GLB files
- [ ] Import COLLADA (.dae) files
- [ ] Import 3DS files
- [ ] Import Blender files (.blend)
- [ ] Handle files up to 500MB
- [ ] Handle meshes with 5M+ triangles
- [ ] Report import errors clearly to user
- [ ] Preserve mesh topology on import

---

## 2. Functional Details

### 2.1 Inputs

| Input | Type | Required | Description |
|-------|------|----------|-------------|
| File Path | string | Yes | Path to the 3D model file |
| Import Options | ImportOptions | No | Optional settings (units, up-axis, etc.) |

### 2.2 Outputs

| Output | Type | Description |
|--------|------|-------------|
| MeshModel | MeshModel | Internal mesh representation |
| ImportResult | ImportResult | Success/failure status with details |
| Warnings | List<string> | Any non-fatal issues encountered |

### 2.3 Processing Logic

1. Detect file format from extension
2. Select appropriate importer (Assimp.NET or OpenCascade)
3. Load file into memory
4. Parse geometry data
5. Convert to internal MeshModel representation
6. Validate mesh integrity
7. Return result with any warnings

### 2.4 Business Rules

- STEP/IGES files must be tessellated to triangle mesh
- Large files (>100MB) should show progress indicator
- Corrupted files should fail gracefully with clear error message
- Multi-part models should preserve part hierarchy

---

## 3. Technical Details

### 3.1 Dependencies

- **Assimp.NET** (NuGet) - 40+ mesh formats
- **OpenCascade** - STEP, IGES CAD tessellation

### 3.2 Supported Formats

**Via Assimp.NET (40+ formats):**

| Category | Formats |
|----------|--------|
| **Common Interchange** | COLLADA (.dae), glTF 1.0/2.0 (.gltf, .glb), FBX (.fbx), OBJ (.obj), 3MF (.3mf) |
| **Mesh Formats** | STL (.stl), PLY (.ply), OFF (.off) |
| **3D Software** | Blender (.blend), 3DS Max (.3ds, .ase), LightWave (.lwo, .lws), Modo (.lxo) |
| **Game Formats** | DirectX X (.x), Quake (.mdl, .md2, .md3), Valve (.smd), Ogre (.mesh) |
| **Other** | AC3D (.ac), Milkshape (.ms3d), DXF (.dxf)*, IFC (.ifc), XGL (.xgl) |

*Limited support

**Via OpenCascade (CAD formats requiring tessellation):**

| Format | Extension | Notes |
|--------|-----------|-------|
| STEP | .step, .stp | AP203, AP214, AP242 |
| IGES | .iges, .igs | Version 5.3 |
| BREP | .brep | OpenCascade native |

### 3.3 Affected Components

- `MeshPrep.Core` - Import interfaces and implementations
- `MeshPrep.FilterScriptCreator` - File open dialog
- `MeshPrep.ModelFixer` - File open dialog, CLI argument

### 3.4 Technical Approach

```
File → Format Detection → Importer Selection
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
         Assimp.NET                      OpenCascade
    (STL, OBJ, PLY, FBX, 3MF)          (STEP, IGES)
              │                               │
              └───────────────┬───────────────┘
                              ▼
                    Internal MeshModel
```

### 3.5 API/Interface

```csharp
namespace MeshPrep.Core.Import
{
    public interface IMeshImporter
    {
        bool CanImport(string filePath);
        ImportResult Import(string filePath, ImportOptions? options = null);
        Task<ImportResult> ImportAsync(string filePath, ImportOptions? options = null, 
            IProgress<double>? progress = null, CancellationToken ct = default);
    }

    public class ImportOptions
    {
        public Units TargetUnits { get; set; } = Units.Millimeters;
        public Axis UpAxis { get; set; } = Axis.Z;
        public bool PreservePartHierarchy { get; set; } = true;
    }

    public class ImportResult
    {
        public bool Success { get; set; }
        public MeshModel? Model { get; set; }
        public string? ErrorMessage { get; set; }
        public List<string> Warnings { get; set; } = new();
        public TimeSpan ImportTime { get; set; }
    }
}
```

---

## 4. User Interface

### 4.1 UI Changes Required

**FilterScriptCreator:**
- File → Open menu item
- Drag & drop support on main window
- Recent files list

**ModelFixer GUI:**
- File → Open menu item
- Drag & drop support

**ModelFixer CLI:**
- `--input <filepath>` argument

### 4.2 User Interaction Flow

```
User drags file ──► Format detected ──► Progress shown ──► Model displayed
         │                                    │
         ▼                                    ▼
    Invalid format                      Import failed
         │                                    │
         ▼                                    ▼
    Error message                       Error details
```

---

## 5. Testing

### 5.1 Test Cases

| Test ID | Description | Input | Expected Output | Status |
|---------|-------------|-------|-----------------|--------|
| TC-001 | Import binary STL | Valid binary STL | Success, mesh loaded | ⬜ |
| TC-002 | Import ASCII STL | Valid ASCII STL | Success, mesh loaded | ⬜ |
| TC-003 | Import OBJ with MTL | OBJ + MTL files | Success, mesh loaded | ⬜ |
| TC-004 | Import 3MF | Valid 3MF | Success, mesh loaded | ⬜ |
| TC-005 | Import STEP | Valid STEP file | Success, tessellated mesh | ⬜ |
| TC-006 | Import large file | 200MB STL | Success within 30s | ⬜ |
| TC-007 | Import corrupted file | Invalid data | Graceful failure, error message | ⬜ |
| TC-008 | Import unsupported format | .xyz file | Clear error message | ⬜ |
| TC-009 | Import multi-part model | Multi-body STEP | All parts preserved | ⬜ |
| TC-010 | Cancel import | Large file, cancel | Import stops cleanly | ⬜ |

### 5.2 Edge Cases

- Zero-size file
- File with no triangles
- File with millions of triangles
- File with non-standard encoding
- Network path / slow storage
- File locked by another process

---

## 6. Notes & Open Questions

### Open Questions
- [x] Which library for CAD formats? → **OpenCascade for STEP/IGES**
- [x] Handle materials/colors? → **Optional, preserve if present**

### Notes
- Assimp.NET supports most formats out of the box
- OpenCascade required for parametric CAD formats (STEP, IGES)
- Consider memory-mapped files for very large meshes
- Import is blocking operation - use async for GUI responsiveness

---

## 7. Implementation Log

| Date | Update | Author |
|------|--------|--------|
| 2026-01-10 | Feature document created | |
