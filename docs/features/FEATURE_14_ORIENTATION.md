# Feature F-014: Build Plate Orientation

---

## Feature ID: F-014

## Feature Name
Build Plate Orientation

## Status
- [x] Not Started
- [ ] In Analysis
- [ ] In Development
- [ ] In Testing
- [ ] Complete

## Priority
**Medium** - Print preparation convenience

## Estimated Effort
**Medium** (1-3 days)

## Related POC
None - Straightforward implementation

---

## 1. Description

### 1.1 Overview
Position and orient models relative to the build plate for optimal printing. Includes laying flat, centering, and rotating for best print orientation.

### 1.2 User Story

As a **3D printing enthusiast**, I want **to orient my model on the build plate** so that **it prints with minimal supports and best quality**.

### 1.3 Acceptance Criteria

- [ ] Center model on build plate (X, Y)
- [ ] Drop model to build plate (Z = 0)
- [ ] Auto-detect flat surfaces for laying flat
- [ ] Manual rotation around X, Y, Z axes
- [ ] Show build plate grid in preview
- [ ] Configurable build plate size

---

## 2. Functional Details

### 2.1 Inputs

| Input | Type | Required | Description |
|-------|------|----------|-------------|
| MeshModel | MeshModel | Yes | Model to orient |
| OrientationOptions | OrientationOptions | No | Orientation settings |
| BuildPlateSize | Size3D | No | Printer build volume |

### 2.2 Outputs

| Output | Type | Description |
|--------|------|-------------|
| OrientedMesh | MeshModel | Positioned model |
| Transformation | Matrix4x4 | Applied transformation |

### 2.3 Orientation Operations

| Operation | Description |
|-----------|-------------|
| Center XY | Move to center of build plate |
| Drop to Bed | Move down until lowest point touches Z=0 |
| Lay Flat | Rotate to place largest flat surface on bed |
| Rotate X/Y/Z | Manual rotation around axis |
| Auto-Orient | ML-suggested optimal orientation |

### 2.4 Business Rules

- Model must fit within build volume
- Warn if model exceeds build plate
- Z=0 is the build plate surface
- Lay Flat finds surfaces within tolerance of flat

---

## 3. Technical Details

### 3.1 Dependencies

- MeshLib for transformations
- 3D math (Matrix4x4)

### 3.2 Affected Components

- `MeshPrep.Core` - Orientation algorithms
- Both GUI applications - Orientation UI

### 3.3 Lay Flat Algorithm

```
1. Find all face normals
2. Group faces with similar normals (cluster)
3. Calculate area of each cluster
4. Find clusters pointing "down" (could be placed on bed)
5. Select largest cluster
6. Rotate model so cluster faces -Z (down)
7. Drop to bed
```

### 3.4 API/Interface

```csharp
namespace MeshPrep.Core.Transform
{
    public interface IOrientationService
    {
        MeshModel CenterOnBuildPlate(MeshModel mesh, Size3D buildPlate);
        MeshModel DropToBed(MeshModel mesh);
        MeshModel LayFlat(MeshModel mesh);
        MeshModel Rotate(MeshModel mesh, double angleX, double angleY, double angleZ);
        OrientationSuggestion SuggestOrientation(MeshModel mesh);
        bool FitsBuildPlate(MeshModel mesh, Size3D buildPlate);
    }

    public class OrientationOptions
    {
        public bool CenterXY { get; set; } = true;
        public bool DropToZ0 { get; set; } = true;
        public bool AutoLayFlat { get; set; } = false;
    }

    public class Size3D
    {
        public double X { get; set; }  // Build plate width
        public double Y { get; set; }  // Build plate depth
        public double Z { get; set; }  // Max build height
    }

    public class OrientationSuggestion
    {
        public double RotationX { get; set; }
        public double RotationY { get; set; }
        public double RotationZ { get; set; }
        public string Reason { get; set; }
        public double SupportVolumeEstimate { get; set; }
    }
}
```

---

## 4. User Interface

### 4.1 UI Components

**Orientation Panel:**
```
┌─────────────────────────────────────────┐
│         Model Orientation               │
├─────────────────────────────────────────┤
│  Quick Actions:                         │
│  [ Center ]  [ Drop ]  [ Lay Flat ]     │
│                                         │
│  Manual Rotation:                       │
│    X: [______0°______] ◄─────►          │
│    Y: [______0°______] ◄─────►          │
│    Z: [______0°______] ◄─────►          │
│                                         │
│  Model Position:                        │
│    Center: (100, 100, 22.5) mm         │
│    Size: 45 × 80 × 45 mm               │
│                                         │
│  Build Plate: 200 × 200 × 200 mm       │
│    ✅ Model fits                        │
│                                         │
│         [ Apply ]  [ Reset ]            │
└─────────────────────────────────────────┘
```

**Build Plate Warning:**
```
┌─────────────────────────────────────────┐
│  ⚠️ Model Exceeds Build Plate           │
│                                         │
│  Model: 250 × 80 × 45 mm               │
│  Build: 200 × 200 × 200 mm             │
│                                         │
│  X dimension exceeds by 50mm            │
│                                         │
│  [ Scale to Fit ]  [ Ignore ]          │
└─────────────────────────────────────────┘
```

### 4.2 3D Preview

- Show build plate as grid
- Show build volume as wireframe box
- Model colored red if outside build volume

---

## 5. Testing

### 5.1 Test Cases

| Test ID | Description | Input | Expected Output | Status |
|---------|-------------|-------|-----------------|--------|
| TC-001 | Center model | Off-center model | Centered | ⬜ |
| TC-002 | Drop to bed | Floating model | Z min = 0 | ⬜ |
| TC-003 | Lay flat | Tilted model | Flat face on bed | ⬜ |
| TC-004 | Rotate 90° X | Any model | Correct rotation | ⬜ |
| TC-005 | Fit check pass | Small model | Returns true | ⬜ |
| TC-006 | Fit check fail | Oversized model | Returns false | ⬜ |
| TC-007 | Combined operations | Center + Drop | Both applied | ⬜ |

### 5.2 Edge Cases

- Model already centered
- Model with no clear flat surface
- Model larger than build plate
- Model at origin already

---

## 6. Notes & Open Questions

### Open Questions
- [x] Include auto-orient (ML)? → **Basic heuristics for v1.0, ML later**
- [ ] Support multiple build plate presets? → **Yes, common printers**

### Notes
- Lay Flat is heuristic - may not always find optimal orientation
- Future: ML-based optimal orientation (minimize supports)
- Build plate size stored in user preferences
- Common printer presets: Prusa MK3S (250×210×210), Ender 3 (220×220×250)

---

## 7. Implementation Log

| Date | Update | Author |
|------|--------|--------|
| 2026-01-10 | Feature document created | |
