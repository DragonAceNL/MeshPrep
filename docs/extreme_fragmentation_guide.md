# Extreme Fragmentation Repair Guide

This guide covers the repair of extremely fragmented meshes - models with hundreds or thousands of disconnected bodies that need to be reconstructed into a single printable mesh.

## Profile Characteristics

Extreme fragmentation typically occurs in:
- **CAD exports gone wrong** - Boolean operations that created debris
- **Corrupted mesh files** - Data loss creating disconnected fragments
- **Complex assemblies** - Many parts that need to be unified
- **Scan data** - Noisy point clouds converted to mesh
- **Artistic/generative models** - Deliberately fragmented geometry

### Detection Criteria

| Metric | Threshold | Classification |
|--------|-----------|----------------|
| Body count | >1000 | Extreme fragmentation |
| Body count | 100-1000 | High fragmentation |
| Body count | 10-100 | Moderate fragmentation |
| Tiny bodies (<10 faces) | >70% of total | Debris-heavy |
| Large bodies (>50 faces) | Multiple | Multi-part assembly |

## Reconstruction Methods

### 1. CGAL Alpha Wrap (Recommended)

CGAL Alpha Wrap creates a guaranteed watertight envelope around the input geometry. It's the most reliable method for extreme fragmentation.

#### How It Works

Alpha Wrap "shrink-wraps" the geometry by:
1. Computing a 3D Delaunay triangulation of the input points
2. Carving out tetrahedra based on the alpha parameter
3. Extracting the outer surface as a watertight mesh

#### Parameters

| Parameter | Description | Effect |
|-----------|-------------|--------|
| `relative_alpha` | Divisor for bbox diagonal → alpha value | Higher = more detail (smaller triangles) |
| `relative_offset` | Divisor for bbox diagonal → offset value | Higher = tighter wrap |

**Key insight**: The actual alpha value = `bbox_diagonal / relative_alpha`

For a mesh with 2100mm diagonal:

| relative_alpha | Actual Alpha | Triangle Size | Detail Level |
|----------------|--------------|---------------|--------------|
| 100 | 21.0mm | ~21mm | Very coarse |
| 500 | 4.2mm | ~4mm | Coarse |
| 1000 | 2.1mm | ~2mm | Medium |
| 2000 | 1.05mm | ~1mm | High |
| 4000 | 0.53mm | ~0.5mm | Very high |
| 8000 | 0.26mm | ~0.25mm | Ultra |
| 10000+ | 0.21mm | ~0.2mm | Maximum |

#### Quality Presets

| Preset | Alpha | Offset | Time* | Use Case |
|--------|-------|--------|-------|----------|
| **Quick Preview** | 100 | 600 | ~1 min | Fast check, iteration |
| **Draft** | 500 | 1200 | ~5 min | Quick iteration, prototyping |
| **Good Quality** | 1000 | 2000 | ~10 min | General use, FDM printing |
| **High Quality** | 2000 | 4000 | ~17 min | Production, resin printing |
| **Ultra Quality** | 4000 | 8000 | 1-4 hours | Maximum detail, willing to wait |
| **Extreme Quality** | 8000 | 16000 | 4-24 hours | Absolute best, archival |
| **Maximum** | 10000+ | 20000+ | Days | No compromise on quality |

*Times are approximate for a 700K face input mesh on a modern CPU. Your results may vary based on mesh complexity and hardware.

**Note on "slow" presets**: There is no technical reason to avoid Ultra/Extreme/Maximum quality presets if you have the time. These produce genuinely better results with finer detail. Combined with smoothing afterward, you get the best of both worlds: maximum captured detail with smooth surfaces.

**Recommended workflow for best quality**:
1. Run with Extreme/Maximum settings (let it run overnight or over a weekend)
2. Apply HC Laplacian smoothing (5-10 iterations) to reduce stair-stepping
3. Optionally apply isotropic remeshing to reduce file size while preserving quality

#### Stair-Stepping Artifacts

Alpha Wrap inherently creates triangles that follow the surface in discrete steps. On angled/curved surfaces, this appears as visible "stairs" or "slabs".

**Solutions:**

1. **Increase resolution** (higher relative_alpha)
   - More triangles = smaller steps
   - Exponential time increase
   - No practical limit if you have time

2. **Post-process smoothing** (recommended)
   - HC Laplacian: Volume-preserving smoothing
   - Iterations: 3-10 depending on severity
   - Fast and effective

3. **Isotropic remeshing**
   - Redistributes triangles evenly
   - Reduces face count while improving quality
   - Best combined with smoothing

### 2. Post-Processing Options

#### HC Laplacian Smoothing

Humphrey's Classes (HC) Laplacian smoothing preserves volume better than standard Laplacian.

| Iterations | Effect | Use Case |
|------------|--------|----------|
| 1-3 | Light smoothing | Minor artifacts |
| 5-10 | Medium smoothing | Visible stair-stepping |
| 10-20 | Heavy smoothing | Severe artifacts |
| 20+ | Aggressive smoothing | May lose detail |

**Recommendation**: Start with 3 iterations, increase if needed.

#### Isotropic Remeshing

Redistributes triangles for uniform size across the surface.

| Target Edge Length | Effect |
|-------------------|--------|
| 0.1% bbox | Very fine mesh |
| 0.3% bbox | Fine mesh |
| 0.5% bbox | Medium mesh (recommended) |
| 1% bbox | Coarse mesh |

**Benefits:**
- Removes stair-step pattern
- Reduces file size (3M → 185K faces typical)
- More efficient for slicing
- Better for smoothing

### 3. Alternative Methods

For cases where CGAL Alpha Wrap isn't suitable:

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Screened Poisson** | Smooth results | May fill holes incorrectly | Clean point clouds |
| **Ball Pivoting** | Preserves detail | Can leave holes | Sparse data |
| **Voxel Remesh** | Fast, reliable | Loses sharp edges | Organic shapes |
| **Shrinkwrap** | Good for envelopes | May miss concavities | Convex-ish shapes |

## Recommended Pipelines

### Pipeline 1: Maximum Quality (Recommended for Production)

```
cgal_alpha_wrap (2000/4000) → hc_laplacian_smooth (10x) → fix_normals
```

- **Time**: ~20 minutes
- **Output**: 3M+ faces, very smooth
- **Best for**: Final production models

### Pipeline 2: Balanced Quality + Size

```
cgal_alpha_wrap (2000/4000) → isotropic_remesh (0.5%) → hc_laplacian_smooth (3x) → fix_normals
```

- **Time**: ~20 minutes
- **Output**: ~200K faces, smooth, efficient
- **Best for**: 3D printing (smaller file, same quality)

### Pipeline 3: Quick Preview

```
cgal_alpha_wrap (500/1200) → fix_normals
```

- **Time**: ~5 minutes
- **Output**: ~200K faces, some stair-stepping
- **Best for**: Quick checks, iteration

### Pipeline 4: Draft Quality

```
cgal_alpha_wrap (100/600) → fix_normals
```

- **Time**: ~1 minute
- **Output**: ~35K faces, coarse
- **Best for**: Very fast preview

### Pipeline 5: Ultra Quality (Willing to Wait Hours)

```
cgal_alpha_wrap (4000/8000) → hc_laplacian_smooth (5x) → fix_normals
```

- **Time**: 1-4 hours
- **Output**: 10M+ faces → smoothed
- **Best for**: High-end production, important models

### Pipeline 6: Extreme Quality (Overnight Processing)

```
cgal_alpha_wrap (8000/16000) → hc_laplacian_smooth (5x) → isotropic_remesh (0.5%) → fix_normals
```

- **Time**: 4-24 hours
- **Output**: Maximum detail captured, then optimized to ~500K faces
- **Best for**: Archival, museum-quality reproductions

### Pipeline 7: Maximum Quality (No Time Limit)

```
cgal_alpha_wrap (10000/20000) → hc_laplacian_smooth (10x) → isotropic_remesh (0.3%) → fix_normals
```

- **Time**: Days
- **Output**: Absolute finest detail possible
- **Best for**: When quality is the only priority

## Why High Resolution + Smoothing is Better Than Low Resolution

A common question is: "Why run at 8000/16000 and then smooth, when I could just run at 2000/4000?"

The answer lies in **information preservation**:

### Low Resolution Approach
```
Original Geometry → Coarse Alpha Wrap (loses detail) → Result
```

The coarse wrap **permanently loses** fine geometric information. Once it's gone, no amount of smoothing can bring it back.

### High Resolution + Smoothing Approach
```
Original Geometry → Fine Alpha Wrap (captures detail) → Smooth (refines surface) → Result
```

The fine wrap **captures** all the geometric detail first. Smoothing then **refines** the surface without losing the underlying accuracy. The smoothed high-res result follows the original geometry more accurately than a low-res result ever could.

### Visual Analogy

Think of it like photography:
- **Low res approach**: Taking a blurry photo and sharpening it → still blurry
- **High res + smoothing**: Taking a sharp photo and applying artistic smoothing → crisp underlying detail with refined presentation

### When This Matters

| Scenario | Low Res OK? | High Res + Smooth Better? |
|----------|-------------|---------------------------|
| Quick preview | ✅ Yes | Overkill |
| Functional prototype | ✅ Usually | Nice to have |
| Display model | ⚠️ Maybe | ✅ Yes |
| Resin printing | ⚠️ Risky | ✅ Yes |
| Professional work | ❌ No | ✅ Yes |
| Archival | ❌ No | ✅ Essential |

### File Size Consideration

High resolution produces large files (10M+ faces), but:
1. **Smoothing doesn't change face count** - still large
2. **Isotropic remeshing reduces faces** while preserving the improved accuracy
3. **Final result**: Smaller file with better quality than low-res original

Example:
- Low res (2000/4000): 3.1M faces, moderate quality
- High res (8000/16000) + smooth + remesh: ~500K faces, excellent quality

## Parameter Selection Guide

### Based on Model Size

| Bbox Diagonal | Recommended Alpha | Reasoning |
|---------------|-------------------|-----------|
| <100mm | 1000-2000 | Small models need fine detail |
| 100-500mm | 1500-2500 | Balanced |
| 500-2000mm | 2000-4000 | Large models, ~1mm resolution |
| >2000mm | 2000-4000+ | Consider overnight processing for best results |

### Based on Time Budget

| Available Time | Alpha | Smoothing | Post-Process | Expected Quality |
|----------------|-------|-----------|--------------|------------------|
| 1-2 minutes | 100 | None | None | Draft |
| 5-10 minutes | 500 | 3x HC | None | Good |
| 15-30 minutes | 2000 | 5x HC | None | High |
| 1-4 hours | 4000 | 5x HC | Optional remesh | Very high |
| Overnight | 8000 | 5x HC | Remesh | Excellent |
| Weekend | 10000+ | 10x HC | Remesh | Maximum |

**Important**: Don't dismiss the slower options. If you have a model that matters (archival, professional work, important print), the extra time produces measurably better results.

### Based on Output Requirements

| Requirement | Recommended Pipeline |
|-------------|---------------------|
| Quick visual check | Pipeline 4 (Draft) |
| Prototyping | Pipeline 3 (Quick Preview) |
| FDM printing (functional) | Pipeline 2 (Balanced) |
| FDM printing (display) | Pipeline 1 (Maximum Quality) |
| Resin printing | Pipeline 1 or 5 |
| Professional production | Pipeline 5 or 6 |
| Archival / Museum quality | Pipeline 6 or 7 |
| Absolute best possible | Pipeline 7 (Maximum) |

## Troubleshooting

### Problem: Visible Stair-Stepping

**Cause**: Alpha value too large (relative_alpha too low)

**Solutions**:
1. Increase relative_alpha (2000 → 4000 → 8000)
2. Add HC Laplacian smoothing (10+ iterations)
3. Apply isotropic remeshing before smoothing
4. Consider overnight processing with high alpha

### Problem: Process Takes Too Long

**Cause**: Alpha value too small (relative_alpha too high)

**Solutions**:
1. Decrease relative_alpha (8000 → 4000 → 2000)
2. Use Quick Preview first, then High Quality for final
3. Consider overnight/weekend processing for best results
4. Note: Time spent often pays off in quality

### Problem: Lost Detail

**Cause**: Alpha too large or too much smoothing

**Solutions**:
1. Increase relative_alpha
2. Reduce smoothing iterations
3. Use isotropic remesh instead of aggressive smoothing

### Problem: Mesh Has Holes

**Cause**: Alpha value too small for the gap size

**Solutions**:
1. Decrease relative_alpha
2. Decrease relative_offset (tighter wrap)
3. The wrap should always be watertight - check input

### Problem: Memory Error

**Cause**: Very high resolution settings

**Solutions**:
1. Reduce relative_alpha
2. Decimate input mesh first
3. Process in sections (advanced)
4. Use a machine with more RAM

## Quality Comparison Reference

Based on testing with a 2100mm diagonal, 702K face fragmented mesh:

| Setting | Faces | File Size | Stair-Step Visibility | Time |
|---------|-------|-----------|----------------------|------|
| 100/600 | 35K | 1.7 MB | Very visible | 1 min |
| 500/1200 | 191K | 9 MB | Visible | 5 min |
| 1000/2000 | 777K | 37 MB | Moderate | 10 min |
| 2000/4000 | 3.1M | 150 MB | Slight | 17 min |
| 2000/4000 + HC(10) | 3.1M | 150 MB | Minimal | 20 min |
| 2000/4000 + Remesh + HC(3) | 185K | 9 MB | Minimal | 20 min |
| 8000/16000 + Remesh + HC(5) | ~500K | ~25 MB | Negligible | 4-24 hours |

## Implementation Notes

### Action Parameters

```python
# CGAL Alpha Wrap
{
    "action": "cgal_alpha_wrap",
    "params": {
        "relative_alpha": 2000.0,  # Higher = more detail
        "relative_offset": 4000.0   # Higher = tighter wrap
    }
}

# HC Laplacian Smoothing
{
    "action": "hc_laplacian_smooth",
    "params": {
        "iterations": 10  # Number of smoothing passes
    }
}

# Combined Action
{
    "action": "cgal_alpha_wrap_smooth",
    "params": {
        "relative_alpha": 2000.0,
        "relative_offset": 4000.0,
        "smooth_iterations": 10
    }
}

# Isotropic Remeshing
{
    "action": "isotropic_remesh",
    "params": {
        "target_edge_percent": 0.5  # % of bbox diagonal
    }
}
```

### Pipeline Definition

```python
FilterPipeline(
    name="cgal-alpha-wrap-smooth",
    description="CGAL Alpha Wrap + smoothing for extreme fragmentation",
    actions=[
        {"action": "cgal_alpha_wrap_smooth", "params": {
            "relative_alpha": 2000.0,
            "relative_offset": 4000.0,
            "smooth_iterations": 10
        }},
        {"action": "fix_normals", "params": {}},
    ],
    priority=1,
)
```

## Summary

For extreme fragmentation:

1. **Default recommendation**: `cgal_alpha_wrap(2000/4000)` + `hc_laplacian_smooth(10)` (~20 min)
2. **For smaller files**: Add isotropic remeshing after alpha wrap
3. **For faster results**: Reduce alpha to 500-1000
4. **For maximum quality**: Increase alpha to 8000+ and let it run overnight

The key trade-off is always **detail vs. time**. Higher `relative_alpha` values give more detail but exponentially increase processing time. However, if you have the time (overnight, weekend), there is no reason not to use the highest quality settings - the results are genuinely better.

**Remember**: Time invested in processing is almost always worth it for important models. A few extra hours of computation can make the difference between a good print and an excellent one.
