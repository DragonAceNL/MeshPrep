# Feature F-015: Multi-Part Model Handling

---

## Feature ID: F-015

## Feature Name
Multi-Part Model Handling

## Status
- [x] Not Started
- [ ] In Analysis
- [ ] In Development
- [ ] In Testing
- [ ] Complete

## Priority
**High** - Essential for complex models (spaceships)

## Estimated Effort
**Large** (3-7 days)

## Related POC
**POC-05** - Mesh Repair (includes multi-part testing)

---

## 1. Description

### 1.1 Overview
Handle 3D models consisting of multiple separate parts (shells/bodies). Each part can be analyzed and repaired independently, then exported together or separately.

### 1.2 User Story

As a **spaceship model enthusiast**, I want **to work with models that have multiple parts** so that **I can repair complex assemblies without losing their structure**.

### 1.3 Acceptance Criteria

- [ ] Detect separate parts in imported model
- [ ] Display part list with statistics
- [ ] Analyze each part independently
- [ ] Repair parts individually or all together
- [ ] Merge selected parts into one
- [ ] Split disconnected geometry into parts
- [ ] Export all parts or selected parts

---

## 2. Functional Details

### 2.1 Inputs

| Input | Type | Required | Description |
|-------|------|----------|-------------|
| MeshModel | MeshModel | Yes | Model to analyze |

### 2.2 Outputs

| Output | Type | Description |
|--------|------|-------------|
| Parts | List<MeshPart> | Individual parts |
| PartAnalysis | List<MeshAnalysisResult> | Analysis per part |

### 2.3 Part Operations

| Operation | Description |
|-----------|-------------|
| Detect Parts | Find disconnected shells |
| Select Part | Choose specific part(s) for operations |
| Merge Parts | Combine selected parts into one |
| Split Parts | Separate disconnected geometry |
| Delete Part | Remove part from model |
| Export Part | Export single part to file |
| Repair Part | Apply repairs to specific part |

### 2.4 Business Rules

- Parts detected by connectivity (shared vertices/edges)
- Each part has its own analysis
- Merged parts become single mesh
- Part hierarchy preserved from CAD imports when possible

---

## 3. Technical Details

### 3.1 Dependencies

- **MeshLib** - Connected component detection

### 3.2 Affected Components

- `MeshPrep.Core` - Part detection and management
- Both GUI applications - Part list UI

### 3.3 Part Detection Algorithm

```
1. Build adjacency graph (faces sharing edges)
2. Find connected components using BFS/DFS
3. Each component = one part
4. Calculate bounding box per part
5. Optional: Name parts by position (Top, Bottom, Left, etc.)
```

### 3.4 API/Interface

```csharp
namespace MeshPrep.Core.Parts
{
    public interface IPartManager
    {
        List<MeshPart> DetectParts(MeshModel mesh);
        MeshModel MergeParts(List<MeshPart> parts);
        List<MeshPart> SplitIntoParts(MeshModel mesh);
        MeshModel ExtractPart(MeshModel mesh, MeshPart part);
        MeshModel DeletePart(MeshModel mesh, MeshPart part);
    }

    public class MeshPart
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public int FaceCount { get; set; }
        public int VertexCount { get; set; }
        public BoundingBox Bounds { get; set; }
        public double Volume { get; set; }
        public bool IsSelected { get; set; }
        
        // Indices into parent mesh
        public List<int> FaceIndices { get; set; }
    }

    public class MultiPartModel
    {
        public MeshModel FullMesh { get; set; }
        public List<MeshPart> Parts { get; set; }
        public int TotalParts => Parts.Count;
    }
}
```

---

## 4. User Interface

### 4.1 UI Components

**Parts Panel:**
```
┌─────────────────────────────────────────┐
│            Model Parts (5)              │
├─────────────────────────────────────────┤
│  ☑ Part 1 - Main Body                  │
│     Faces: 45,230  Volume: 125.3 cm³   │
│     Issues: 2 holes                     │
│                                         │
│  ☑ Part 2 - Wing Left                  │
│     Faces: 12,450  Volume: 23.1 cm³    │
│     Issues: None ✅                     │
│                                         │
│  ☑ Part 3 - Wing Right                 │
│     Faces: 12,450  Volume: 23.1 cm³    │
│     Issues: None ✅                     │
│                                         │
│  ☐ Part 4 - Antenna (tiny)             │
│     Faces: 234  Volume: 0.1 cm³        │
│     Issues: Non-manifold               │
│                                         │
│  ☑ Part 5 - Engine                     │
│     Faces: 8,920  Volume: 15.7 cm³     │
│     Issues: 1 hole                      │
├─────────────────────────────────────────┤
│  [ Select All ]  [ Select None ]        │
│  [ Merge Selected ]  [ Delete Selected ]│
│  [ Repair Selected ]                    │
└─────────────────────────────────────────┘
```

### 4.2 3D Preview

- Different color per part
- Click part in 3D to select
- Highlight selected part
- Show/hide individual parts

### 4.3 User Interaction Flow

```
Import Model ──► Detect Parts ──► Show Parts List
                                       │
                              ┌────────┴────────┐
                              ▼                 ▼
                        Single Part        Multi-Part
                              │                 │
                              ▼                 ▼
                       Normal workflow    Part-aware workflow
                                               │
                               ┌───────────────┼───────────────┐
                               ▼               ▼               ▼
                         Repair All    Repair Selected    Merge Parts
```

---

## 5. Testing

### 5.1 Test Cases

| Test ID | Description | Input | Expected Output | Status |
|---------|-------------|-------|-----------------|--------|
| TC-001 | Detect single part | Simple cube | 1 part | ⬜ |
| TC-002 | Detect multiple parts | 3 separate objects | 3 parts | ⬜ |
| TC-003 | Part statistics | Multi-part model | Correct counts | ⬜ |
| TC-004 | Merge two parts | Select 2, merge | 1 part | ⬜ |
| TC-005 | Delete part | Select 1, delete | Part removed | ⬜ |
| TC-006 | Export single part | Select 1, export | Single part STL | ⬜ |
| TC-007 | Repair specific part | Part with holes | Part repaired | ⬜ |
| TC-008 | Complex spaceship | Many parts | All detected | ⬜ |

### 5.2 Edge Cases

- Parts touching but not connected
- Very small parts (noise)
- Nested parts (part inside part)
- Hundreds of parts

---

## 6. Notes & Open Questions

### Open Questions
- [x] Auto-detect vs manual split? → **Auto-detect on import**
- [ ] Name parts automatically? → **By size/position heuristics**

### Notes
- Essential for spaceship models with separate components
- Part colors help visualization
- Consider "ignore small parts" option for cleanup
- STEP imports often have named parts from CAD

---

## 7. Implementation Log

| Date | Update | Author |
|------|--------|--------|
| 2026-01-10 | Feature document created | |
