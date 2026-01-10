# POC-01: Format Import

---

## POC ID: POC-01

## POC Name
Multi-format 3D Model Import

## Status
- [x] Not Started
- [ ] In Progress
- [ ] Completed - Success
- [ ] Completed - Failed
- [ ] Blocked

## Estimated Effort
**2-3 days**

## Related Features
- F-001: Multi-format Import
- F-015: Multi-Part Model Handling

---

## 1. Objective

### 1.1 What We're Proving
Validate that Assimp.NET and OpenCascade can successfully import all target 3D file formats with acceptable performance and fidelity.

### 1.2 Success Criteria

- [ ] Import STL (binary and ASCII) successfully
- [ ] Import OBJ with materials
- [ ] Import 3MF files
- [ ] Import STEP files and tessellate to mesh
- [ ] Import IGES files and tessellate to mesh
- [ ] Import glTF/GLB files
- [ ] Import FBX files
- [ ] Handle files up to 500MB within 60 seconds
- [ ] Handle meshes with 1M+ triangles
- [ ] Preserve multi-part structure from STEP files

### 1.3 Failure Criteria

- Major format not supported (STL, OBJ, STEP)
- Performance unacceptable (>2 minutes for 100MB file)
- Frequent crashes or memory issues
- Loss of geometry data during import

---

## 2. Technical Approach

### 2.1 Technologies to Evaluate

| Technology | Version | Purpose |
|------------|---------|---------|
| Assimp.NET | Latest NuGet | STL, OBJ, FBX, glTF, 3MF, PLY |
| OpenCascade.NET | Latest | STEP, IGES tessellation |
| .NET | 10 | Runtime |

### 2.2 Test Scenarios

1. **Basic Import** - Load simple cube/sphere in each format
2. **Complex Model** - Load spaceship model with 500K+ triangles
3. **Multi-Part STEP** - Load CAD assembly with multiple bodies
4. **Large File** - Load 200MB+ STL file
5. **Format Variants** - Binary vs ASCII STL, glTF vs GLB
6. **Error Handling** - Corrupted files, missing references

### 2.3 Test Data

| File | Format | Size | Triangles | Purpose |
|------|--------|------|-----------|---------|
| cube.stl | STL Binary | 1KB | 12 | Basic test |
| cube_ascii.stl | STL ASCII | 2KB | 12 | ASCII variant |
| spaceship.obj | OBJ | 50MB | 500K | Complex mesh |
| assembly.step | STEP | 10MB | N/A | Multi-part CAD |
| large_terrain.stl | STL | 200MB | 2M | Performance test |
| character.fbx | FBX | 20MB | 100K | Animation format |
| scene.gltf | glTF | 5MB | 50K | Web format |

**Sources:**
- Thingi10K dataset
- Free spaceship models
- Custom test files

---

## 3. Implementation

### 3.1 Setup Steps

1. Create new .NET 10 console project: `MeshPrep.POC.FormatImport`
2. Install NuGet packages:
   ```
   dotnet add package AssimpNet
   ```
3. Set up OpenCascade (manual or NuGet if available)
4. Download test files to `/poc/test-data/`
5. Create import wrapper classes

### 3.2 Code Location

`/poc/POC_01_FormatImport/`

### 3.3 Key Code Snippets

**Assimp.NET Import:**
```csharp
using Assimp;

public class AssimpImporter
{
    public Mesh[] Import(string filePath)
    {
        using var context = new AssimpContext();
        var scene = context.ImportFile(filePath, 
            PostProcessSteps.Triangulate | 
            PostProcessSteps.GenerateNormals |
            PostProcessSteps.JoinIdenticalVertices);
        
        return scene.Meshes.ToArray();
    }
}
```

**OpenCascade STEP Import:**
```csharp
// Pseudocode - actual API may vary
public class StepImporter
{
    public Mesh[] ImportStep(string filePath)
    {
        var reader = new STEPControl_Reader();
        reader.ReadFile(filePath);
        reader.TransferRoots();
        
        var shape = reader.OneShape();
        return Tessellate(shape);
    }
}
```

**Performance Measurement:**
```csharp
var sw = Stopwatch.StartNew();
var result = importer.Import(filePath);
sw.Stop();

Console.WriteLine($"Import time: {sw.ElapsedMilliseconds}ms");
Console.WriteLine($"Triangles: {result.Sum(m => m.FaceCount)}");
Console.WriteLine($"Memory: {GC.GetTotalMemory(false) / 1024 / 1024}MB");
```

---

## 4. Results

### 4.1 Test Results

| Test | Format | Result | Notes |
|------|--------|--------|-------|
| Basic STL Binary | .stl | ⬜ | |
| Basic STL ASCII | .stl | ⬜ | |
| OBJ with MTL | .obj | ⬜ | |
| 3MF | .3mf | ⬜ | |
| STEP simple | .step | ⬜ | |
| STEP multi-part | .step | ⬜ | |
| IGES | .iges | ⬜ | |
| glTF | .gltf | ⬜ | |
| GLB | .glb | ⬜ | |
| FBX | .fbx | ⬜ | |
| Large file (200MB) | .stl | ⬜ | |

### 4.2 Performance Metrics

| Metric | Target | Actual | Pass? |
|--------|--------|--------|-------|
| 1MB STL import time | < 1s | | ⬜ |
| 100MB STL import time | < 30s | | ⬜ |
| 200MB STL import time | < 60s | | ⬜ |
| Memory for 1M triangles | < 500MB | | ⬜ |
| STEP tessellation time | < 10s | | ⬜ |

### 4.3 Issues Encountered

*To be filled during POC execution*

---

## 5. Conclusions

### 5.1 Recommendation
*To be filled after POC completion*

- [ ] Proceed with chosen approach
- [ ] Modify approach (details below)
- [ ] Abandon and find alternative

### 5.2 Risks Identified
*To be filled after POC completion*

### 5.3 Next Steps
*To be filled after POC completion*

---

## 6. Log

| Date | Update | Author |
|------|--------|--------|
| 2026-01-10 | POC document created | |
