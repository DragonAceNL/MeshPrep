# POC-06: Geometry Comparison

---

## POC ID: POC-06

## POC Name
Geometry Comparison (Hybrid Hausdorff)

## Status
- [x] Not Started
- [ ] In Progress
- [ ] Completed - Success
- [ ] Completed - Failed
- [ ] Blocked

## Estimated Effort
**2-3 days**

## Related Features
- F-007: Geometry Fidelity Check
- F-004: ML Filter Generation (RL reward signal)

---

## 1. Objective

### 1.1 What We're Proving
Validate that MeshLib's Hausdorff distance calculation provides accurate geometry comparison for 3D printing fidelity assessment, using both max and mean metrics.

### 1.2 Success Criteria

- [ ] Calculate Max Hausdorff distance accurately
- [ ] Calculate Mean Hausdorff distance accurately
- [ ] Results in model units (mm)
- [ ] Comparison completes in < 10s for 1M triangles
- [ ] Detect small geometry changes (< 0.1mm)
- [ ] Distinguish acceptable vs unacceptable deviations
- [ ] GPU acceleration improves performance

### 1.3 Failure Criteria

- Hausdorff not available in MeshLib
- Inaccurate distance measurements
- Performance too slow for RL training loop
- Cannot set meaningful thresholds

---

## 2. Technical Approach

### 2.1 Technologies to Evaluate

| Technology | Version | Purpose |
|------------|---------|---------|
| MeshLib | Latest NuGet | Hausdorff distance calculation |
| CUDA | 11.x+ | GPU acceleration |

### 2.2 Test Scenarios

1. **Identical Meshes** - Compare mesh to itself, expect 0.0
2. **Known Offset** - Offset mesh by 1mm, verify measurement
3. **Small Change** - Minor modification, verify detection
4. **Large Change** - Major modification, verify detection
5. **Spike Test** - Add single spike, verify Max > Mean
6. **Distributed Changes** - Many small changes, verify Mean > threshold
7. **Performance** - Large mesh comparison time

### 2.3 Test Data

| Test | Original | Modified | Expected Max | Expected Mean |
|------|----------|----------|--------------|---------------|
| Identical | cube.stl | cube.stl | 0.0 | 0.0 |
| 1mm offset | cube.stl | cube_offset1mm.stl | ~1.0 | ~1.0 |
| 0.1mm offset | cube.stl | cube_offset01mm.stl | ~0.1 | ~0.1 |
| Single spike | cube.stl | cube_spike.stl | High | Low |
| Smoothed | spaceship.stl | spaceship_smooth.stl | Varies | Varies |

---

## 3. Implementation

### 3.1 Setup Steps

1. Create new .NET 10 console project: `MeshPrep.POC.GeometryComparison`
2. Install MeshLib NuGet package
3. Create test meshes with known modifications
4. Implement comparison wrapper
5. Validate against expected values

### 3.2 Code Location

`/poc/POC_06_GeometryComparison/`

### 3.3 Key Code Snippets

**Basic Hausdorff Comparison:**
```csharp
using MR.DotNet;

public class GeometryComparer
{
    public ComparisonResult Compare(Mesh original, Mesh repaired)
    {
        var sw = Stopwatch.StartNew();
        
        // Compute Hausdorff distance
        var hausdorff = MeshDistance.ComputeHausdorff(original, repaired);
        
        sw.Stop();
        
        return new ComparisonResult
        {
            MaxHausdorff = hausdorff.Max,
            MeanHausdorff = hausdorff.Mean,
            RmsDistance = hausdorff.Rms,
            ComputeTime = sw.Elapsed
        };
    }
}
```

**Threshold Evaluation:**
```csharp
public class ComparisonThresholds
{
    public double MaxHausdorffThreshold { get; set; } = 0.5;   // mm
    public double MeanHausdorffThreshold { get; set; } = 0.05; // mm
}

public bool EvaluateResult(ComparisonResult result, ComparisonThresholds thresholds)
{
    var maxPass = result.MaxHausdorff <= thresholds.MaxHausdorffThreshold;
    var meanPass = result.MeanHausdorff <= thresholds.MeanHausdorffThreshold;
    
    Console.WriteLine($"Max Hausdorff: {result.MaxHausdorff:F4}mm " +
        $"(threshold: {thresholds.MaxHausdorffThreshold}mm) - {(maxPass ? "PASS" : "FAIL")}");
    Console.WriteLine($"Mean Hausdorff: {result.MeanHausdorff:F4}mm " +
        $"(threshold: {thresholds.MeanHausdorffThreshold}mm) - {(meanPass ? "PASS" : "FAIL")}");
    
    return maxPass && meanPass;
}
```

**RL Reward Calculation:**
```csharp
public double CalculateGeometryReward(ComparisonResult result, ComparisonThresholds thresholds)
{
    if (result.MaxHausdorff <= thresholds.MaxHausdorffThreshold &&
        result.MeanHausdorff <= thresholds.MeanHausdorffThreshold)
    {
        // Good geometry preservation - reward inversely proportional to deviation
        var meanNormalized = result.MeanHausdorff / thresholds.MeanHausdorffThreshold;
        return (1.0 - meanNormalized) * 0.5;  // 0.0 to 0.5 reward
    }
    else
    {
        // Poor geometry preservation - penalty
        return -0.5;
    }
}
```

**Visualization Data:**
```csharp
public List<DeviationPoint> GetDeviationMap(Mesh original, Mesh repaired, int sampleCount = 10000)
{
    var deviations = new List<DeviationPoint>();
    
    // Sample points on repaired mesh surface
    var samples = repaired.SampleSurface(sampleCount);
    
    foreach (var point in samples)
    {
        // Find distance to original mesh
        var distance = original.DistanceToPoint(point);
        
        deviations.Add(new DeviationPoint
        {
            Location = point,
            Deviation = distance
        });
    }
    
    return deviations;
}
```

**Performance Test:**
```csharp
public void PerformanceTest(string meshPath, int iterations = 10)
{
    var mesh1 = Mesh.FromAnySupportedFormat(meshPath);
    var mesh2 = mesh1.Clone();
    
    // Slight modification to mesh2
    mesh2.Translate(new Vector3(0.01, 0, 0));
    
    var times = new List<double>();
    
    for (int i = 0; i < iterations; i++)
    {
        var sw = Stopwatch.StartNew();
        var result = MeshDistance.ComputeHausdorff(mesh1, mesh2);
        sw.Stop();
        times.Add(sw.ElapsedMilliseconds);
    }
    
    Console.WriteLine($"Mesh triangles: {mesh1.FaceCount}");
    Console.WriteLine($"Average time: {times.Average():F1}ms");
    Console.WriteLine($"Min time: {times.Min()}ms");
    Console.WriteLine($"Max time: {times.Max()}ms");
}
```

---

## 4. Results

### 4.1 Test Results

| Test | Expected Max | Actual Max | Expected Mean | Actual Mean | Pass? |
|------|--------------|------------|---------------|-------------|-------|
| Identical meshes | 0.0 | | 0.0 | | ⬜ |
| 1mm offset | ~1.0 | | ~1.0 | | ⬜ |
| 0.1mm offset | ~0.1 | | ~0.1 | | ⬜ |
| Single spike | High | | Low | | ⬜ |
| Distributed changes | Low | | Medium | | ⬜ |

### 4.2 Performance Metrics

| Mesh Size | Target Time | Actual Time (CPU) | Actual Time (GPU) | Pass? |
|-----------|-------------|-------------------|-------------------|-------|
| 10K | < 1s | | | ⬜ |
| 100K | < 3s | | | ⬜ |
| 500K | < 5s | | | ⬜ |
| 1M | < 10s | | | ⬜ |

### 4.3 Threshold Validation

| Scenario | Max Threshold | Mean Threshold | Result |
|----------|---------------|----------------|--------|
| Good repair | 0.5mm | 0.05mm | ⬜ Pass expected |
| Minor issues | 0.5mm | 0.05mm | ⬜ Pass expected |
| Major changes | 0.5mm | 0.05mm | ⬜ Fail expected |

### 4.4 Issues Encountered

*To be filled during POC execution*

---

## 5. Conclusions

### 5.1 Recommendation
*To be filled after POC completion*

### 5.2 Recommended Thresholds
*To be filled after POC completion*

| Threshold | Value | Rationale |
|-----------|-------|-----------|
| Max Hausdorff | | |
| Mean Hausdorff | | |

### 5.3 Risks Identified
*To be filled after POC completion*

### 5.4 Next Steps
*To be filled after POC completion*

---

## 6. Log

| Date | Update | Author |
|------|--------|--------|
| 2026-01-10 | POC document created | |
