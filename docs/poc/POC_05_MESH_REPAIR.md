# POC-05: Mesh Repair

---

## POC ID: POC-05

## POC Name
Mesh Repair with MeshLib

## Status
- [x] Not Started
- [ ] In Progress
- [ ] Completed - Success
- [ ] Completed - Failed
- [ ] Blocked

## Estimated Effort
**3-5 days**

## Related Features
- F-003: Mesh Analysis
- F-004: ML Filter Generation
- F-005: Filter Script Application
- F-015: Multi-Part Model Handling

---

## 1. Objective

### 1.1 What We're Proving
Validate that MeshLib can analyze and repair complex 3D meshes (including spaceship models) with all required operations: hole filling, non-manifold repair, self-intersection removal, and geometry comparison.

### 1.2 Success Criteria

- [ ] Detect non-manifold edges
- [ ] Detect and fill holes
- [ ] Detect and remove self-intersections
- [ ] Fix non-manifold vertices
- [ ] Remove degenerate triangles
- [ ] Calculate Hausdorff distance (max and mean)
- [ ] GPU acceleration working (CUDA)
- [ ] Repair complex spaceship model successfully
- [ ] Performance acceptable (< 30s for 500K triangles)

### 1.3 Failure Criteria

- Cannot repair complex models
- GPU acceleration not available or slow
- Geometry comparison not available
- Crashes or memory issues
- Performance unacceptable

---

## 2. Technical Approach

### 2.1 Technologies to Evaluate

| Technology | Version | Purpose |
|------------|---------|---------|
| MeshLib | Latest NuGet | Mesh analysis and repair |
| MR.DotNet | | C# bindings |
| CUDA | 11.x+ | GPU acceleration |

### 2.2 Test Scenarios

1. **Detection** - Analyze mesh and detect all issue types
2. **Hole Filling** - Fill holes with various strategies
3. **Non-Manifold Repair** - Fix edges and vertices
4. **Self-Intersection** - Detect and remove
5. **Hausdorff Distance** - Compare original vs repaired
6. **GPU Test** - Verify CUDA acceleration
7. **Complex Model** - Full repair workflow on spaceship

### 2.3 Test Data

| Model | Issues | Purpose |
|-------|--------|---------|
| cube_clean.stl | None | Baseline |
| cube_holes.stl | 2 holes | Hole filling |
| cube_nonmanifold.stl | Non-manifold edges | Edge repair |
| cube_selfintersect.stl | Self-intersections | Intersection repair |
| spaceship_broken.stl | Multiple issues | Complex repair |
| Thingi10K samples | Various | Variety testing |

---

## 3. Implementation

### 3.1 Setup Steps

1. Create new .NET 10 console project: `MeshPrep.POC.MeshRepair`
2. Install NuGet packages:
   ```
   dotnet add package MeshLib
   ```
3. Verify CUDA toolkit installed
4. Download test meshes with known issues
5. Implement analysis and repair wrappers

### 3.2 Code Location

`/poc/POC_05_MeshRepair/`

### 3.3 Key Code Snippets

**Load and Analyze:**
```csharp
using MR.DotNet;

public class MeshAnalyzer
{
    public AnalysisResult Analyze(string filePath)
    {
        var mesh = Mesh.FromAnySupportedFormat(filePath);
        
        return new AnalysisResult
        {
            VertexCount = mesh.VertexCount,
            FaceCount = mesh.FaceCount,
            Volume = mesh.Volume,
            BoundingBox = mesh.BoundingBox,
            // Issue detection
            NonManifoldEdges = DetectNonManifoldEdges(mesh),
            Holes = DetectHoles(mesh),
            SelfIntersections = DetectSelfIntersections(mesh)
        };
    }
}
```

**Hole Filling:**
```csharp
public void FillHoles(Mesh mesh)
{
    var holes = mesh.FindHoles();
    
    foreach (var hole in holes)
    {
        // Fill with planar or curvature-aware method
        mesh.FillHole(hole, FillStrategy.Planar);
    }
}
```

**Non-Manifold Repair:**
```csharp
public void FixNonManifold(Mesh mesh)
{
    // MeshLib auto-heals non-manifold on import
    // This validates the mesh is now manifold
    var result = mesh.IsManifold();
    Console.WriteLine($"Is manifold: {result}");
}
```

**Self-Intersection Removal:**
```csharp
public void RemoveSelfIntersections(Mesh mesh)
{
    var intersections = mesh.FindSelfIntersections();
    Console.WriteLine($"Found {intersections.Count} self-intersections");
    
    mesh.RemoveSelfIntersections();
    
    var remaining = mesh.FindSelfIntersections();
    Console.WriteLine($"Remaining: {remaining.Count}");
}
```

**Hausdorff Distance:**
```csharp
public HausdorffResult CompareGeometry(Mesh original, Mesh repaired)
{
    var result = MeshDistance.ComputeHausdorff(original, repaired);
    
    return new HausdorffResult
    {
        MaxDistance = result.Max,      // Maximum deviation
        MeanDistance = result.Mean,    // Average deviation
        RmsDistance = result.Rms       // RMS deviation
    };
}
```

**Full Repair Workflow:**
```csharp
public Mesh RepairMesh(string filePath)
{
    var mesh = Mesh.FromAnySupportedFormat(filePath);
    var original = mesh.Clone();
    
    // Step 1: Fix non-manifold (auto on import)
    Console.WriteLine("Checking manifold...");
    
    // Step 2: Fill holes
    Console.WriteLine("Filling holes...");
    FillHoles(mesh);
    
    // Step 3: Remove self-intersections
    Console.WriteLine("Removing self-intersections...");
    RemoveSelfIntersections(mesh);
    
    // Step 4: Remove degenerate triangles
    Console.WriteLine("Removing degenerate faces...");
    mesh.RemoveDegenerateFaces();
    
    // Step 5: Verify geometry fidelity
    var hausdorff = CompareGeometry(original, mesh);
    Console.WriteLine($"Max Hausdorff: {hausdorff.MaxDistance:F4}mm");
    Console.WriteLine($"Mean Hausdorff: {hausdorff.MeanDistance:F4}mm");
    
    return mesh;
}
```

**GPU Verification:**
```csharp
public void VerifyGpuAcceleration()
{
    // Check if CUDA is available
    var hasCuda = MeshLib.HasCudaSupport();
    Console.WriteLine($"CUDA available: {hasCuda}");
    
    if (hasCuda)
    {
        var deviceName = MeshLib.GetCudaDeviceName();
        Console.WriteLine($"GPU: {deviceName}");
    }
}
```

---

## 4. Results

### 4.1 Test Results

| Test | Result | Notes |
|------|--------|-------|
| Load mesh | ⬜ | |
| Detect non-manifold | ⬜ | |
| Detect holes | ⬜ | |
| Detect self-intersections | ⬜ | |
| Fill holes | ⬜ | |
| Fix non-manifold | ⬜ | |
| Remove self-intersections | ⬜ | |
| Hausdorff calculation | ⬜ | |
| GPU acceleration | ⬜ | |
| Complex spaceship repair | ⬜ | |

### 4.2 Performance Metrics

| Operation | Mesh Size | Target | Actual | Pass? |
|-----------|-----------|--------|--------|-------|
| Analysis | 100K | < 1s | | ⬜ |
| Analysis | 500K | < 5s | | ⬜ |
| Hole filling | 100K | < 2s | | ⬜ |
| Self-intersection | 100K | < 5s | | ⬜ |
| Full repair | 500K | < 30s | | ⬜ |
| Hausdorff calc | 500K | < 10s | | ⬜ |

### 4.3 Issues Encountered

*To be filled during POC execution*

---

## 5. Conclusions

### 5.1 Recommendation
*To be filled after POC completion*

### 5.2 Risks Identified
*To be filled after POC completion*

### 5.3 Next Steps
*To be filled after POC completion*

---

## 6. Log

| Date | Update | Author |
|------|--------|--------|
| 2026-01-10 | POC document created | |
