# Feature F-013: Scaling & Unit Conversion

---

## Feature ID: F-013

## Feature Name
Scaling and Unit Conversion

## Status
- [x] Not Started
- [ ] In Analysis
- [ ] In Development
- [ ] In Testing
- [ ] Complete

## Priority
**Medium** - Common need for downloaded models

## Estimated Effort
**Small** (< 1 day)

## Related POC
None - Straightforward implementation

---

## 1. Description

### 1.1 Overview
Scale models and convert between unit systems (mm, cm, inches). Many downloaded models have incorrect scale or use different units than the user's printer expects.

### 1.2 User Story

As a **3D printing enthusiast**, I want **to scale my model and convert units** so that **it prints at the correct size**.

### 1.3 Acceptance Criteria

- [ ] Scale model uniformly (X, Y, Z together)
- [ ] Scale model non-uniformly (X, Y, Z independent)
- [ ] Convert between mm, cm, m, inches, feet
- [ ] Auto-detect likely unit mismatch
- [ ] Preview scaled dimensions before applying
- [ ] Preserve relative proportions for uniform scale

---

## 2. Functional Details

### 2.1 Inputs

| Input | Type | Required | Description |
|-------|------|----------|-------------|
| MeshModel | MeshModel | Yes | Model to scale |
| ScaleOptions | ScaleOptions | Yes | Scaling parameters |

### 2.2 Outputs

| Output | Type | Description |
|--------|------|-------------|
| ScaledMesh | MeshModel | Scaled model |
| DimensionsAfter | Dimensions | New bounding box size |

### 2.3 Supported Units

| Unit | Abbreviation | To mm |
|------|--------------|-------|
| Millimeters | mm | 1.0 |
| Centimeters | cm | 10.0 |
| Meters | m | 1000.0 |
| Inches | in | 25.4 |
| Feet | ft | 304.8 |

### 2.4 Auto-Detection Logic

```
If model bounds > 1000mm in any axis AND fits typical inch range (1-50):
    Suggest: "Model may be in inches. Convert to mm?"

If model bounds < 1mm in all axes:
    Suggest: "Model may be in meters. Convert to mm?"
```

---

## 3. Technical Details

### 3.1 Dependencies

- Built-in math operations
- MeshLib for vertex manipulation

### 3.2 Affected Components

- `MeshPrep.Core` - Scaling operations
- Both GUI applications - Scale UI

### 3.3 API/Interface

```csharp
namespace MeshPrep.Core.Transform
{
    public interface IScaleService
    {
        MeshModel Scale(MeshModel mesh, ScaleOptions options);
        MeshModel ConvertUnits(MeshModel mesh, Units from, Units to);
        UnitSuggestion DetectUnits(MeshModel mesh);
    }

    public class ScaleOptions
    {
        public double ScaleX { get; set; } = 1.0;
        public double ScaleY { get; set; } = 1.0;
        public double ScaleZ { get; set; } = 1.0;
        public bool Uniform { get; set; } = true;
        public Point3D Origin { get; set; } = Point3D.Zero;  // Scale about this point
    }

    public enum Units
    {
        Millimeters,
        Centimeters,
        Meters,
        Inches,
        Feet
    }

    public class Dimensions
    {
        public double Width { get; set; }   // X
        public double Height { get; set; }  // Y
        public double Depth { get; set; }   // Z
        public Units Unit { get; set; }
    }

    public class UnitSuggestion
    {
        public bool HasSuggestion { get; set; }
        public Units DetectedUnit { get; set; }
        public Units SuggestedUnit { get; set; }
        public string Message { get; set; }
    }
}
```

---

## 4. User Interface

### 4.1 UI Components

**Scale Panel:**
```
┌─────────────────────────────────────────┐
│            Scale Model                  │
├─────────────────────────────────────────┤
│  Current Size (mm):                     │
│    X: 150.0  Y: 80.0  Z: 45.0          │
│                                         │
│  ☑ Uniform Scale                        │
│    Scale: [____100___] %                │
│                                         │
│  ☐ Non-Uniform Scale                    │
│    X: [____100___] %                    │
│    Y: [____100___] %                    │
│    Z: [____100___] %                    │
│                                         │
│  New Size (mm):                         │
│    X: 150.0  Y: 80.0  Z: 45.0          │
│                                         │
│         [ Apply ]  [ Cancel ]           │
└─────────────────────────────────────────┘
```

**Unit Conversion:**
```
┌─────────────────────────────────────────┐
│          Convert Units                  │
├─────────────────────────────────────────┤
│  From: [Inches     ▼]                   │
│  To:   [Millimeters▼]                   │
│                                         │
│  Current: 5.9 × 3.1 × 1.8 in           │
│  After:   150.0 × 80.0 × 45.0 mm       │
│                                         │
│         [ Convert ]  [ Cancel ]         │
└─────────────────────────────────────────┘
```

### 4.2 Auto-Detection Prompt

```
┌─────────────────────────────────────────┐
│  ⚠️ Unit Mismatch Detected              │
│                                         │
│  This model appears to be in INCHES     │
│  but your printer expects MILLIMETERS.  │
│                                         │
│  Current size: 5.9 × 3.1 × 1.8         │
│  Converted:    150 × 80 × 45 mm        │
│                                         │
│  [ Convert to mm ]  [ Keep as-is ]     │
└─────────────────────────────────────────┘
```

---

## 5. Testing

### 5.1 Test Cases

| Test ID | Description | Input | Expected Output | Status |
|---------|-------------|-------|-----------------|--------|
| TC-001 | Uniform 200% scale | 100mm cube | 200mm cube | ⬜ |
| TC-002 | Uniform 50% scale | 100mm cube | 50mm cube | ⬜ |
| TC-003 | Non-uniform scale | X:2, Y:1, Z:0.5 | Stretched cube | ⬜ |
| TC-004 | Inches to mm | 1in cube | 25.4mm cube | ⬜ |
| TC-005 | mm to inches | 25.4mm cube | 1in cube | ⬜ |
| TC-006 | Auto-detect inches | 6in model | Suggests conversion | ⬜ |
| TC-007 | Auto-detect meters | 0.01m model | Suggests conversion | ⬜ |

### 5.2 Edge Cases

- Scale to zero (should prevent)
- Negative scale (mirror)
- Very large scale factors
- Scale about different origins

---

## 6. Notes & Open Questions

### Open Questions
- [x] Support non-uniform scaling? → **Yes**
- [ ] Allow negative scale (mirroring)? → **Consider separately**

### Notes
- Most models are in mm or inches
- Auto-detection helps catch common mistakes
- Preview prevents accidental wrong scaling
- Scale operation can be part of filter script

---

## 7. Implementation Log

| Date | Update | Author |
|------|--------|--------|
| 2026-01-10 | Feature document created | |
