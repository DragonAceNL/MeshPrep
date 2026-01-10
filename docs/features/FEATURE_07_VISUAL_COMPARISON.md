# Feature F-007: Geometry Fidelity Check

---

## Feature ID: F-007

## Feature Name
Geometry Fidelity Check (Hybrid Hausdorff)

## Status
- [x] Not Started
- [ ] In Analysis
- [ ] In Development
- [ ] In Testing
- [ ] Complete

## Priority
**Medium** - Required for RL reward signal

## Estimated Effort
**Large** (3-7 days)

## Related POC
**POC-06** - Geometry Comparison validation

---

## 1. Description

### 1.1 Overview
Compare the repaired mesh to the original to ensure the repair process hasn't significantly altered the model's geometry. Uses Hybrid Hausdorff distance (max + mean) to measure physical deviation in millimeters.

### 1.2 User Story

As a **3D printing enthusiast**, I want **to verify my repaired model still looks like the original** so that **the printed object matches my expectations**.

### 1.3 Acceptance Criteria

- [ ] Calculate Max Hausdorff distance between meshes
- [ ] Calculate Mean Hausdorff distance between meshes
- [ ] Report deviation in model units (mm)
- [ ] Define configurable thresholds for pass/fail
- [ ] Visualization of deviation areas (heatmap)
- [ ] Comparison completes in < 10 seconds for 1M triangle meshes

---

## 2. Functional Details

### 2.1 Inputs

| Input | Type | Required | Description |
|-------|------|----------|-------------|
| OriginalMesh | MeshModel | Yes | The original mesh before repair |
| RepairedMesh | MeshModel | Yes | The mesh after repair |
| Thresholds | ComparisonThresholds | No | Pass/fail thresholds |

### 2.2 Outputs

| Output | Type | Description |
|--------|------|-------------|
| ComparisonResult | GeometryComparisonResult | Deviation metrics and pass/fail |

### 2.3 Processing Logic

1. Align meshes if needed (should already be aligned)
2. Sample points on both mesh surfaces
3. Calculate distances from each point on original to nearest point on repaired
4. Calculate distances from each point on repaired to nearest point on original
5. Compute Max Hausdorff (maximum of all distances)
6. Compute Mean Hausdorff (average of all distances)
7. Apply thresholds to determine pass/fail

### 2.4 Business Rules

**Default Thresholds:**
- Max Hausdorff: < 0.5mm → Pass (no single point deviates too much)
- Mean Hausdorff: < 0.05mm → Pass (overall surface quality)
- Both must pass for overall pass

**RL Reward Calculation:**
```
geometryReward = 0;
if (maxHausdorff < maxThreshold && meanHausdorff < meanThreshold)
{
    // Reward inversely proportional to deviation
    geometryReward = (1.0 - meanHausdorff / meanThreshold) * 0.5;
}
else
{
    geometryReward = -0.5; // Penalty for excessive deviation
}
```

---

## 3. Technical Details

### 3.1 Dependencies

- **MeshLib** - Hausdorff distance calculation (built-in)

### 3.2 Affected Components

- `MeshPrep.Core` - Comparison implementation
- `MeshPrep.FilterScriptCreator` - Display comparison results
- RL system - Geometry reward signal

### 3.3 Technical Approach

```
Original Mesh              Repaired Mesh
     │                          │
     ▼                          ▼
  Sample Points            Sample Points
     │                          │
     └──────────┬───────────────┘
                ▼
         MeshLib Hausdorff
                │
        ┌───────┴───────┐
        ▼               ▼
  Max Hausdorff   Mean Hausdorff
        │               │
        └───────┬───────┘
                ▼
         Apply Thresholds
                │
                ▼
           Pass / Fail
```

### 3.4 API/Interface

```csharp
namespace MeshPrep.Core.Comparison
{
    public interface IGeometryComparer
    {
        GeometryComparisonResult Compare(MeshModel original, MeshModel repaired,
            ComparisonThresholds? thresholds = null);
        Task<GeometryComparisonResult> CompareAsync(MeshModel original, MeshModel repaired,
            ComparisonThresholds? thresholds = null, CancellationToken ct = default);
    }

    public class ComparisonThresholds
    {
        public double MaxHausdorffThreshold { get; set; } = 0.5;   // mm
        public double MeanHausdorffThreshold { get; set; } = 0.05; // mm
    }

    public class GeometryComparisonResult
    {
        public double MaxHausdorff { get; set; }      // Maximum deviation in mm
        public double MeanHausdorff { get; set; }     // Average deviation in mm
        public double RmsDeviation { get; set; }      // RMS deviation (informational)
        
        public bool MaxHausdorffPass { get; set; }    // Within max threshold
        public bool MeanHausdorffPass { get; set; }   // Within mean threshold
        public bool OverallPass { get; set; }         // Both pass
        
        // For visualization
        public List<DeviationPoint> DeviationMap { get; set; }
        
        public TimeSpan ComparisonTime { get; set; }
    }

    public class DeviationPoint
    {
        public Point3D Location { get; set; }
        public double Deviation { get; set; }
    }
}
```

---

## 4. User Interface

### 4.1 UI Changes Required

**FilterScriptCreator:**
- "Compare Geometry" button
- Display Max/Mean Hausdorff values
- Pass/fail indicators with color coding
- Optional: Deviation heatmap overlay on 3D view

**Settings:**
- Configurable thresholds
- Comparison quality settings (sample density)

### 4.2 User Interaction Flow

```
Apply Repair ──► Auto-compare ──► Show results
                     │
                     ▼
            ┌────────┴────────┐
            ▼                 ▼
        Pass ✅           Fail ❌
            │                 │
            ▼                 ▼
    "Geometry preserved"  "Deviation: X.XX mm"
                              │
                              ▼
                     Show deviation areas
```

---

## 5. Testing

### 5.1 Test Cases

| Test ID | Description | Input | Expected Output | Status |
|---------|-------------|-------|-----------------|--------|
| TC-001 | Identical meshes | Same mesh twice | 0.0 deviation | ⬜ |
| TC-002 | Minor deviation | Slightly modified mesh | Small values | ⬜ |
| TC-003 | Major deviation | Significantly different mesh | Large values, fail | ⬜ |
| TC-004 | Spike detection | Mesh with one spike | High max, low mean | ⬜ |
| TC-005 | Overall poor quality | Many small deviations | Low max, high mean | ⬜ |
| TC-006 | Performance test | 1M triangle meshes | < 10 seconds | ⬜ |
| TC-007 | Custom thresholds | Strict thresholds | Different pass/fail | ⬜ |

### 5.2 Edge Cases

- Meshes with different vertex counts
- Meshes with significantly different topology
- Very small models (sub-millimeter)
- Very large models (meter scale)

---

## 6. Notes & Open Questions

### Open Questions
- [x] Which algorithm? → **Hybrid Hausdorff (max + mean)**
- [x] Default thresholds? → **Max: 0.5mm, Mean: 0.05mm**

### Notes
- Hausdorff is view-independent (measures actual geometry)
- Max Hausdorff catches localized problems (spikes)
- Mean Hausdorff catches distributed problems (overall quality)
- MeshLib provides built-in Hausdorff calculation
- Consider GPU acceleration for large meshes

---

## 7. Implementation Log

| Date | Update | Author |
|------|--------|--------|
| 2026-01-10 | Feature document created | |
