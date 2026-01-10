# Feature F-003: Mesh Analysis

---

## Feature ID: F-003

## Feature Name
Mesh Analysis and Issue Detection

## Status
- [x] Not Started
- [ ] In Analysis
- [ ] In Development
- [ ] In Testing
- [ ] Complete

## Priority
**High** - Required to identify what repairs are needed

## Estimated Effort
**Medium** (1-3 days)

## Related POC
**POC-05** - Mesh Repair (analysis is prerequisite)

---

## 1. Description

### 1.1 Overview
Analyze imported 3D meshes to detect issues that prevent successful 3D printing. This analysis guides the repair process and provides input to the RL system.

### 1.2 User Story

As a **3D printing enthusiast**, I want **to see what problems exist in my model** so that **I understand what needs to be fixed before printing**.

### 1.3 Acceptance Criteria

- [ ] Detect non-manifold edges
- [ ] Detect non-manifold vertices
- [ ] Detect holes (open boundaries)
- [ ] Detect self-intersections
- [ ] Detect degenerate triangles (zero area)
- [ ] Detect inverted normals
- [ ] Calculate mesh statistics (vertices, faces, volume, surface area)
- [ ] Analysis completes in < 5 seconds for 1M triangle mesh
- [ ] Report issues with location information

---

## 2. Functional Details

### 2.1 Inputs

| Input | Type | Required | Description |
|-------|------|----------|-------------|
| MeshModel | MeshModel | Yes | The mesh to analyze |
| AnalysisOptions | AnalysisOptions | No | What to check (default: all) |

### 2.2 Outputs

| Output | Type | Description |
|--------|------|-------------|
| AnalysisResult | MeshAnalysisResult | Complete analysis report |

### 2.3 Processing Logic

1. Calculate basic statistics (vertex/face count, bounds)
2. Check edge manifoldness
3. Check vertex manifoldness
4. Detect boundary edges (holes)
5. Detect self-intersections
6. Check triangle quality (degenerate, needle triangles)
7. Verify normal consistency
8. Calculate volume and surface area
9. Compile report

### 2.4 Business Rules

- Non-manifold meshes cannot be 3D printed
- Holes must be filled for watertight mesh
- Self-intersections cause slicer failures
- Zero-volume models are invalid

---

## 3. Technical Details

### 3.1 Dependencies

- **MeshLib** (NuGet) - Mesh analysis functions

### 3.2 Affected Components

- `MeshPrep.Core` - Analysis implementation
- Both GUI applications - Display analysis results
- RL system - Analysis provides state for RL agent

### 3.3 Technical Approach

```
MeshModel ──► MeshLib Analysis ──► MeshAnalysisResult
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
    Topology    Geometry    Statistics
     Check       Check        Calc
```

### 3.4 API/Interface

```csharp
namespace MeshPrep.Core.Analysis
{
    public interface IMeshAnalyzer
    {
        MeshAnalysisResult Analyze(MeshModel mesh, AnalysisOptions? options = null);
        Task<MeshAnalysisResult> AnalyzeAsync(MeshModel mesh, 
            AnalysisOptions? options = null, CancellationToken ct = default);
    }

    public class MeshAnalysisResult
    {
        // Statistics
        public int VertexCount { get; set; }
        public int FaceCount { get; set; }
        public int EdgeCount { get; set; }
        public double Volume { get; set; }
        public double SurfaceArea { get; set; }
        public BoundingBox Bounds { get; set; }

        // Issues
        public int NonManifoldEdgeCount { get; set; }
        public int NonManifoldVertexCount { get; set; }
        public int HoleCount { get; set; }
        public int SelfIntersectionCount { get; set; }
        public int DegenerateTriangleCount { get; set; }
        public int InvertedNormalCount { get; set; }

        // Issue locations (for visualization)
        public List<EdgeIssue> NonManifoldEdges { get; set; }
        public List<VertexIssue> NonManifoldVertices { get; set; }
        public List<HoleInfo> Holes { get; set; }
        public List<IntersectionInfo> SelfIntersections { get; set; }

        // Summary
        public bool IsWatertight => HoleCount == 0;
        public bool IsManifold => NonManifoldEdgeCount == 0 && NonManifoldVertexCount == 0;
        public bool IsPrintable => IsWatertight && IsManifold && SelfIntersectionCount == 0;
    }
}
```

---

## 4. User Interface

### 4.1 UI Changes Required

**FilterScriptCreator:**
- Analysis panel showing all detected issues
- Issue counts with icons (✅ good, ⚠️ warning, ❌ error)
- Click on issue to highlight in 3D view

**ModelFixer:**
- Summary view of analysis results
- Simple pass/fail indicators

### 4.2 User Interaction Flow

```
Model imported ──► Auto-analyze ──► Display results
                                        │
                        ┌───────────────┼───────────────┐
                        ▼               ▼               ▼
                   No issues       Minor issues     Critical issues
                        │               │               │
                        ▼               ▼               ▼
                   "Ready to      "May need        "Repair required"
                    print"         repair"
```

---

## 5. Testing

### 5.1 Test Cases

| Test ID | Description | Input | Expected Output | Status |
|---------|-------------|-------|-----------------|--------|
| TC-001 | Analyze clean mesh | Watertight STL | No issues detected | ⬜ |
| TC-002 | Detect non-manifold edges | Mesh with shared edges | Issues reported | ⬜ |
| TC-003 | Detect holes | Mesh with open boundary | Hole count > 0 | ⬜ |
| TC-004 | Detect self-intersection | Overlapping geometry | Intersections found | ⬜ |
| TC-005 | Detect degenerate triangles | Mesh with zero-area faces | Degenerate count > 0 | ⬜ |
| TC-006 | Calculate volume | Known geometry | Correct volume | ⬜ |
| TC-007 | Performance test | 1M triangle mesh | < 5 seconds | ⬜ |
| TC-008 | Handle empty mesh | 0 triangles | Appropriate result | ⬜ |

### 5.2 Edge Cases

- Empty mesh (no triangles)
- Single triangle
- Mesh with duplicate vertices
- Mesh with Inf/NaN coordinates

---

## 6. Notes & Open Questions

### Open Questions
- [x] Which library for analysis? → **MeshLib - comprehensive, GPU-accelerated**

### Notes
- Analysis results feed into RL state representation
- Issue locations needed for visualization
- Consider caching analysis results for unchanged meshes
- MeshLib auto-heals non-manifold on import (detect before healing for reporting)

---

## 7. Implementation Log

| Date | Update | Author |
|------|--------|--------|
| 2026-01-10 | Feature document created | |
