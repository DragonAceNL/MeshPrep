# MeshPrep Repair Strategy Guide

> **Purpose:** This document captures lessons learned from testing and development to help make optimal repair decisions and achieve 100% success rates.

**Last Updated:** 2026-01-03  
**Based on:** POC v2/v3 testing with Thingi10K (10,000 models)

---

## Table of Contents

1. [Overview](#overview)
2. [Repair Tools Comparison](#repair-tools-comparison)
3. [Filter Scripts & When to Use Them](#filter-scripts--when-to-use-them)
4. [Critical Lessons Learned](#critical-lessons-learned)
5. [Detection Strategies](#detection-strategies)
6. [The Optimal Repair Algorithm](#the-optimal-repair-algorithm)
7. [Known Issues & Workarounds](#known-issues--workarounds)
8. [Performance Characteristics](#performance-characteristics)

---

## Overview

MeshPrep uses a tiered repair approach:

```
Tier 1: trimesh (fast, safe, limited capability)
    ↓
Tier 2: pymeshfix (fast, powerful, can be destructive)
    ↓
Tier 3: Blender (slow, most powerful, always works)
```

**Key Principle:** Start with the least aggressive repair and escalate only when needed.

---

## Repair Tools Comparison

### trimesh (Tier 1)

| Aspect | Details |
|--------|---------|
| **Speed** | ⚡ ~3-5ms |
| **Risk Level** | 🟢 Low |
| **Capabilities** | Merge vertices, remove degenerate/duplicate faces, fix normals |
| **Limitations** | Cannot fill holes, cannot fix non-manifold geometry |
| **Best For** | Already clean models, basic cleanup |

**Actions:**
- `trimesh_basic` - Safe cleanup operations
- `fix_normals` - Consistent face winding
- `fill_holes` - Limited hole filling (often fails on complex holes)

**When to Use:**
- ✅ First step in any repair pipeline
- ✅ Models that are already mostly clean
- ✅ Removing degenerate faces before other operations

**When NOT to Use:**
- ❌ As the only repair for models with holes
- ❌ For non-manifold geometry repair

---

### pymeshfix (Tier 2)

| Aspect | Details |
|--------|---------|
| **Speed** | ⚡ ~10-40ms |
| **Risk Level** | 🟡 Medium to High |
| **Capabilities** | Fill holes, fix non-manifold, make watertight |
| **Limitations** | Can destroy multi-component models, may remove geometry |
| **Best For** | Single-component models with holes or non-manifold issues |

**Actions:**
- `pymeshfix_repair` - Full repair with `joincomp=True` (DANGEROUS for multi-component)
- `pymeshfix_repair_conservative` - Per-component repair with geometry preservation
- `pymeshfix_clean` - Light cleanup without full repair
- `make_manifold` - Make mesh manifold

#### Critical Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `joincomp` | `True` | **DANGER:** Merges all components, keeps only largest |
| `remove_smallest_components` | `True` | Removes components < 1% of total faces |
| `verbose` | `False` | Show pymeshfix debug output |
| `max_face_loss_pct` | `50.0` | Threshold for rejecting destructive repairs |

#### The `joincomp=True` Problem

**What happens:**
1. pymeshfix merges all separate components into one
2. It then keeps only the largest resulting piece
3. Multi-component models lose 50-90% of their geometry!

**Example - Model 1004825 (fragmented):**
```
Before: 330 vertices, 620 faces, 10 components
After (joincomp=True): 33 vertices, 62 faces, 1 component
Result: 90% geometry LOST!
```

**Solution:** Use `pymeshfix_repair_conservative` which:
1. Splits mesh into components
2. Repairs each component separately
3. Recombines all components
4. Preserves multi-part models

#### Self-Intersection Handling

pymeshfix attempts to fix self-intersections by removing intersecting triangles. For severely corrupted meshes, this can destroy most of the geometry.

**Example - Model 100037:**
- Original: 180 vertices, 370 faces
- After pymeshfix: 35 vertices, 66 faces (82% loss!)
- Reason: 14+ self-intersecting triangles that couldn't be resolved

**Detection:** Check face loss percentage and bbox change after repair:
```python
face_loss_pct = (original_faces - result_faces) / original_faces * 100
bbox_change_pct = abs(original_bbox - result_bbox) / original_bbox * 100

if face_loss_pct > 50 or bbox_change_pct > 50:
    # Repair is destructive - reject and escalate
```

---

### Blender (Tier 3)

| Aspect | Details |
|--------|---------|
| **Speed** | 🐢 ~10-15 seconds |
| **Risk Level** | 🟡 Medium (topology changes) |
| **Capabilities** | Voxel remesh (guaranteed manifold), make manifold modifier |
| **Limitations** | Slow, creates dense meshes, loses sharp edges |
| **Best For** | Severely corrupted models, last resort |

**Actions:**
- `blender_remesh` - Voxel remesh (most reliable)
- `blender_make_manifold` - Make manifold modifier

#### Voxel Remesh Characteristics

| Voxel Size | Result |
|------------|--------|
| 0.01 | Very dense mesh, preserves detail, huge file |
| 0.05 | Dense mesh (default), good balance |
| 0.1 | Medium density, some detail loss |
| 0.5+ | Low density, significant detail loss |

**Example - Model 100037 with voxel_size=0.05:**
```
Before: 375 faces
After: 1,976,904 faces (5,271x increase!)
```

**When to Use:**
- ✅ After pymeshfix fails or would destroy geometry
- ✅ Models with severe self-intersections
- ✅ When guaranteed watertight output is required

**When NOT to Use:**
- ❌ As the first repair attempt (too slow)
- ❌ When sharp edges must be preserved
- ❌ When file size is critical

**Post-Processing Recommendation:**
After Blender remesh, consider decimation to reduce face count while maintaining manifold properties.

---

## Filter Scripts & When to Use Them

### Available Presets

| Preset | Actions | Best For |
|--------|---------|----------|
| `basic-cleanup` | trimesh_basic → fix_normals → validate | Clean models |
| `fill-holes` | trimesh_basic → fill_holes_pymeshfix → fix_normals → validate | Simple holes |
| `full-repair` | trimesh_basic → pymeshfix_repair → fix_normals → validate | Single-component with issues |
| `conservative-repair` | trimesh_basic → pymeshfix_repair_conservative → fix_normals → validate | Multi-component models |
| `manifold-repair` | trimesh_basic → make_manifold → fix_normals → validate | Non-manifold only |
| `blender-remesh` | trimesh_basic → blender_remesh → fix_normals → validate | Severe damage, escalation |

### Category-to-Filter Mapping

| Category | Recommended Filter | Reason |
|----------|-------------------|--------|
| clean | basic-cleanup | Already valid, minimal processing |
| holes | full-repair OR conservative-repair* | pymeshfix good at holes |
| many_small_holes | full-repair | Multiple holes need aggressive repair |
| non_manifold | manifold-repair OR conservative-repair* | Targeted fix |
| self_intersecting | full-repair | pymeshfix handles intersections |
| fragmented | conservative-repair | MUST preserve components |
| multiple_components | conservative-repair | MUST preserve components |
| complex | conservative-repair | Often has multiple components |

*Use conservative-repair if model has multiple components (auto-detected)

---

## Critical Lessons Learned

### Lesson 1: Always Detect Component Count First

**Problem:** Using aggressive repair on multi-component models destroys geometry.

**Solution:**
```python
components = mesh.split(only_watertight=False)
if len(components) > 1:
    use_conservative_repair()
else:
    use_standard_repair()
```

### Lesson 2: Basic Cleanup Can Change Component Count

**Problem:** `trimesh_basic` removes degenerate faces, which can merge or eliminate tiny components.

**Example - Model 100037:**
```
Original: 2 components
After trimesh_basic: 1 component (tiny component was degenerate faces)
```

**Implication:** Check component count BEFORE any processing if you need to detect multi-component models.

### Lesson 3: Monitor Geometry Metrics During Repair

**Key Metrics to Track:**
- Face count (loss > 40% is suspicious)
- Vertex count (loss > 40% is suspicious)
- Bounding box diagonal (change > 30% is suspicious)
- Volume (if measurable)
- Surface area

**Red Flags:**
```python
# Destructive repair indicators
face_loss_pct > 40  # Lost too many faces
bbox_change_pct > 30  # Model size changed significantly
component_drop = (before > 1 and after == 1)  # Lost components
```

### Lesson 4: Watertight ≠ Correct

**Problem:** A mesh can be "watertight" and "manifold" but completely wrong.

**Example 1 - Model 1004825:**
- After full-repair: watertight=True, manifold=True
- Reality: 90% of geometry destroyed, only 1 of 10 components remains

**Example 2 - Model 100072 (many_small_holes):**
- After full-repair: watertight=True, manifold=True
- Reality: 49% volume loss, model was cut in half!
- Original: 360 faces, 11,772 volume
- After: 214 faces, 6,003 volume

**Solution:** Always compare before/after metrics, not just quality flags:
- Volume change > 30% → escalate to Blender
- Face loss > 40% → escalate to Blender
- Bbox change > 30% → escalate to Blender

### Lesson 5: Blender is the Ultimate Fallback

**When everything else fails:**
1. pymeshfix would destroy > 50% of faces
2. Model has severe self-intersections
3. Complex non-manifold geometry

**Blender voxel remesh will always produce a valid manifold mesh** (as long as the input has some valid geometry).

### Lesson 6: Order of Operations Matters

**Correct Order:**
1. Load mesh
2. Check component count (BEFORE any modifications)
3. Select appropriate filter based on analysis
4. Run trimesh_basic (safe cleanup)
5. Run main repair (pymeshfix or Blender)
6. Fix normals
7. Validate result
8. If validation fails AND Blender available → escalate

**Wrong Order:**
- Running aggressive repair before checking components
- Skipping trimesh_basic (leaves degenerate faces)
- Not validating after repair

### Lesson 7: ALWAYS Start Fresh from Original Mesh

**Problem:** When trying multiple repair strategies in sequence, each failed repair can damage the mesh. If strategy A partially breaks the mesh, then strategy B tries to fix the already-broken mesh, not the original.

**Example - Model 100500:**
```
Original: 466 faces, 99 components
After fill_holes (failed): mesh slightly modified
After fix_normals (failed): mesh further modified  
After fill_holes_1000 (failed): mesh more damaged
After fix_winding (failed): mesh even worse
After blender_remesh: 7968 faces, 1 component - DESTROYED!
```

The Blender remesh was working on a mesh that had been damaged by 4 previous failed repairs, not the original.

**Solution:** Each repair strategy MUST start from the ORIGINAL mesh:
```python
# WRONG - stacking damage
current_mesh = mesh.copy()
for strategy in strategies:
    current_mesh = try_repair(current_mesh, strategy)  # Each builds on previous damage

# CORRECT - fresh start each time
original_mesh = mesh.copy()
for strategy in strategies:
    working_mesh = original_mesh.copy()  # Fresh start!
    result = try_repair(working_mesh, strategy)
    if result.success:
        return result
```

**Implementation:** The `run_slicer_repair_loop()` function now:
1. Keeps a pristine copy of the original mesh
2. Each repair attempt starts from the original
3. Tracks the "best" result across all attempts
4. Returns the best result if nothing fully succeeds

---

## Detection Strategies

### Detecting Multi-Component Models

```python
def is_multi_component(mesh):
    components = mesh.split(only_watertight=False)
    return len(components) > 1
```

### Detecting Destructive Repair

```python
def check_repair_quality(original, repaired, thresholds=None):
    if thresholds is None:
        thresholds = {
            'max_face_loss_pct': 50.0,
            'max_bbox_change_pct': 50.0,
        }
    
    orig_faces = len(original.faces)
    rep_faces = len(repaired.faces)
    face_loss_pct = (orig_faces - rep_faces) / orig_faces * 100
    
    orig_bbox = np.linalg.norm(original.bounds[1] - original.bounds[0])
    rep_bbox = np.linalg.norm(repaired.bounds[1] - repaired.bounds[0])
    bbox_change_pct = abs(orig_bbox - rep_bbox) / orig_bbox * 100
    
    is_destructive = (
        face_loss_pct > thresholds['max_face_loss_pct'] or
        bbox_change_pct > thresholds['max_bbox_change_pct']
    )
    
    return {
        'is_destructive': is_destructive,
        'face_loss_pct': face_loss_pct,
        'bbox_change_pct': bbox_change_pct,
    }
```

### Detecting Repair Success

```python
def is_printable(mesh):
    return (
        mesh.is_watertight and
        mesh.is_volume and  # manifold
        len(mesh.faces) > 0
    )
```

---

## The Optimal Repair Algorithm

```
INPUT: mesh, category (optional)

1. ANALYZE
   - Load mesh
   - Count components BEFORE any processing
   - Compute original diagnostics (faces, vertices, bbox, volume)

2. SELECT FILTER
   IF components > 1:
       filter = "conservative-repair"
   ELSE IF category == "clean":
       filter = "basic-cleanup"
   ELSE IF category in ["holes", "many_small_holes", "self_intersecting"]:
       filter = "full-repair"
   ELSE IF category == "non_manifold":
       filter = "manifold-repair"
   ELSE:
       filter = "conservative-repair"  # Safe default

3. EXECUTE PRIMARY REPAIR
   result = run_filter(filter, mesh)
   
4. VALIDATE
   IF result.is_printable:
       RETURN result  # Success!
   
5. CHECK FOR DESTRUCTIVE REPAIR
   quality = check_repair_quality(original, result)
   IF quality.is_destructive:
       # Repair destroyed the model, try conservative
       IF filter != "conservative-repair":
           result = run_filter("conservative-repair", mesh)
           IF result.is_printable:
               RETURN result

6. ESCALATE TO BLENDER
   IF Blender available AND NOT result.is_printable:
       result = run_filter("blender-remesh", original_mesh)
       IF result.is_printable:
           RETURN result  # Success with escalation!

7. RETURN RESULT
   # May still be failed if Blender not available or also failed
   RETURN result
```

---

## Known Issues & Workarounds

### Issue: Blender Voxel Remesh Destroys Fragmented/Thin Models

**Symptom:** Model passes watertight/manifold checks after Blender remesh, but visual inspection shows parts missing or structure destroyed.

**Example - Model 100500 (MP:c20c0f27c611):**
```
Original: 181 vertices, 466 faces, 99 components, Volume: 0
After Blender remesh: 3,986 vertices, 7,968 faces, 1 component, Volume: 7724
Result: Model is "fixed" but structure is destroyed - NOT PRINTABLE!
```

**Root Cause (Verified):** The problem IS `blender_remesh` itself, NOT stacking of previous repairs. Even when applied ONLY to the fresh original model, voxel remesh:
1. Creates a voxel grid around the model
2. Fills in all gaps between components
3. Produces a single solid blob

**Smart Detection:** The `analyze_fragmented_model()` function classifies fragmentation types:

| Type | Description | Blender Safe? |
|------|-------------|---------------|
| `debris-particles` | >70% of components are tiny (<10 faces) | ❌ NO |
| `multi-part-assembly` | Multiple large components (>=2 with >=50 faces) | ❌ NO |
| `sparse-wireframe` | Very few faces per unit bbox volume | ❌ NO |
| `split-seam` | Few components (<10), one dominant part | ✅ YES (may help reconnect) |
| `unknown` | Unclear structure with <10 components | ✅ YES (conservative) |

**Detection Code:**
```python
from meshprep_poc.filter_pipelines import analyze_fragmented_model

analysis = analyze_fragmented_model(mesh)
if analysis["is_fragmented"] and not analysis["blender_safe"]:
    # Skip blender_remesh pipelines
    logger.warning(f"Blender unsafe: {analysis['reason']}")
```

**Model 100500 Analysis:**
```
is_fragmented: True
fragmentation_type: debris-particles
blender_safe: False
reason: 97/99 components are tiny (<10 faces)
```

### Issue: pymeshfix Creates Invalid Output

**Symptom:** pymeshfix returns but mesh has 0 faces or is invalid.

**Workaround:** Catch this case and fall back to original or escalate:
```python
if len(result.faces) == 0:
    return original_mesh.copy()
```

### Issue: Decimation Breaks Manifold Status

**Symptom:** After decimating a Blender-remeshed model, it loses watertight/manifold status.

**Cause:** `fast_simplification` (and most decimation algorithms) don't guarantee manifold output. Edge collapses can create non-manifold edges.

**Workaround:** 
1. Check manifold status after decimation
2. If broken, keep the original large mesh
3. Accept larger file sizes for models that need Blender escalation

**Example - Model 100027:**
- Blender output: 4.9M faces, watertight, manifold
- After decimation: 100k faces, NOT watertight, NOT manifold
- Solution: Keep 4.9M face mesh (237 MB file)

**Future improvement:** Investigate manifold-preserving decimation algorithms or post-decimation repair.

### Issue: Blender Creates Massive Files

**Symptom:** Small input → millions of faces output.

**Workaround:** 
1. Use larger voxel_size (0.1 instead of 0.05)
2. Add decimation step after remesh
3. Accept the large file for complex models

### Issue: trimesh fill_holes Doesn't Work

**Symptom:** Holes remain after `trimesh.repair.fill_holes()`.

**Reason:** trimesh's hole filling is limited to simple cases.

**Workaround:** Use pymeshfix for hole filling instead:
```python
# Instead of: trimesh.repair.fill_holes(mesh)
# Use: pymeshfix with joincomp=False
```

### Issue: Component Detection After Cleanup is Wrong

**Symptom:** Model appears single-component after trimesh_basic but was multi-component.

**Reason:** Degenerate faces connecting components were removed.

**Workaround:** Check component count on the RAW loaded mesh, before any processing.

---

## Performance Characteristics

### Timing Benchmarks (typical)

| Operation | Time |
|-----------|------|
| Load STL | 1-5ms |
| trimesh_basic | 2-5ms |
| pymeshfix_repair | 10-40ms |
| pymeshfix_repair_conservative | 15-50ms |
| blender_remesh | 10,000-15,000ms |
| Render image (matplotlib) | 200-500ms |
| Render large mesh (>1M faces) | 60,000-120,000ms |

### Memory Considerations

| Mesh Size | Memory Impact |
|-----------|--------------|
| < 10K faces | Negligible |
| 10K-100K faces | Moderate |
| 100K-1M faces | High (watch for OOM) |
| > 1M faces | Very high (Blender output) |

### File Size Impact

| Operation | File Size Change |
|-----------|------------------|
| trimesh_basic | ~Same |
| pymeshfix | Usually smaller (removes geometry) |
| blender_remesh | Much larger (dense voxel grid) |

---

## Recommended Testing Checklist

Before deploying any repair strategy changes:

- [ ] Test on `clean` category (should not modify)
- [ ] Test on `fragmented` category (must preserve components)
- [ ] Test on `multiple_components` category (must preserve components)
- [ ] Test on `holes` category (should fill holes)
- [ ] Test on model with severe self-intersections (should escalate to Blender)
- [ ] Verify no model loses > 50% geometry without escalation
- [ ] Verify bounding box is preserved (within 30%)
- [ ] Verify all outputs are watertight and manifold

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                 MESHPREP REPAIR QUICK REFERENCE             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  STEP 1: Check component count FIRST (before any cleanup)   │
│                                                             │
│  STEP 2: Select filter:                                     │
│    • Multi-component → conservative-repair                  │
│    • Single + holes → full-repair                           │
│    • Clean model → basic-cleanup                            │
│                                                             │
│  STEP 3: Validate result:                                   │
│    • is_watertight? is_volume?                              │
│    • Face loss < 50%? Bbox change < 30%?                    │
│    • STRICT slicer validation (--info mode)                 │
│    • QUALITY RATING (does it look like original?)           │
│                                                             │
│  STEP 4: Escalate if needed:                                │
│    • Blender available? → blender-remesh                    │
│    • BUT CHECK for fragmented models first!                 │
│                                                             │
│  RED FLAGS:                                                 │
│    ⚠️ Face loss > 40%                                       │
│    ⚠️ Bbox change > 30%                                     │
│    ⚠️ Components: many → 1                                  │
│    ⚠️ Volume change > 50%                                   │
│    ⚠️ Face count INCREASE > 500% after Blender remesh       │
│    ⚠️ QUALITY RATING < 3 (visual inspection fail)           │
│                                                             │
│  NEVER:                                                     │
│    ❌ Use full-repair on multi-component without checking   │
│    ❌ Trust watertight flag alone                           │
│    ❌ Skip Blender escalation when available                │
│    ❌ Trust slicer SLICE mode (auto-repairs silently)       │
│    ❌ Use Blender remesh on highly fragmented models        │
│       (>20 components, <1000 faces)                         │
│                                                             │
│  FALSE POSITIVES (watertight but destroyed):                │
│    ⚠️ 99 components → 1 component = DESTROYED               │
│    ⚠️ 466 faces → 7968 faces (+1609%) = VOXEL FILL          │
│    ⚠️ Volume 0 → Volume 7724 = STRUCTURE LOST               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Visual Quality Feedback System

A model can pass all technical validation (watertight, manifold, slicer-validated) but be **visually destroyed**. The Visual Quality Feedback System learns from user ratings to:

1. **Predict quality scores** before completing repairs
2. **Flag suspicious repairs** that may need review
3. **Penalize pipelines** that produce visually-poor results
4. **Learn profile-specific tolerances**

### Quality Rating Scale

| Score | Label | Description |
|-------|-------|-------------|
| **5** | Perfect | Indistinguishable from original |
| **4** | Good | Minor smoothing, fully usable |
| **3** | Acceptable | Noticeable changes but recognizable |
| **2** | Poor | Significant detail loss |
| **1** | Rejected | Unrecognizable or destroyed |

### When to Flag for Review

- Predicted quality < 3.0 with high confidence
- Volume loss > profile's learned threshold
- Pipeline has historically poor quality for this profile type
- Escalation to Blender (aggressive repairs)

### Pipeline Quality Penalties

The learning engine applies quality-weighted scoring:

| Avg Rating | Effect |
|------------|--------|
| ≥ 4.0 | +0.3 bonus (prefer this pipeline) |
| 3.0-3.9 | Neutral |
| 2.0-2.9 | -0.8 penalty (avoid for quality) |
| < 2.0 | -1.5 severe penalty |

### Key Insight: Blender Remesh Quality

Blender voxel remesh is powerful but destroys details:

| Profile | Typical Quality | Recommendation |
|---------|-----------------|----------------|
| Mechanical parts | 4.0-4.5 | ✅ Use freely |
| Organic sculpts | 2.5-3.5 | ⚠️ Use cautiously |
| Fine details | 2.0-3.0 | ❌ Avoid if possible |

**Best practice:** Let the quality feedback system learn your preferences over time.

---

## Slicer Validation Modes

### Critical Discovery: Slicer Auto-Repair

Modern slicers (PrusaSlicer, OrcaSlicer, SuperSlicer) have **built-in auto-repair** that silently fixes mesh issues during slicing. This means:

1. A model that "passes" slicing may still have mesh issues
2. The slicer fixes issues internally, NOT the exported mesh
3. Users exporting to OTHER slicers may have problems

### STRICT Mode (Recommended)

Use `prusa-slicer --info model.stl` to get raw mesh statistics **without auto-repair**:

```
[100036.stl]
manifold = no          <-- Real mesh state
open_edges = 46        <-- Hole count
facets_reversed = 34   <-- Normal issues
backwards_edges = 32   <-- More normal issues
```

### SLICE Mode (Not Recommended for Validation)

Using `prusa-slicer --export-gcode` will pass models that have issues:

```
Model with 46 open edges:
  STRICT mode: FAIL (correctly detects issues)
  SLICE mode:  PASS (slicer auto-fixed internally)
```

### Validation Decision

| Use Case | Mode | Why |
|----------|------|-----|
| MeshPrep validation | STRICT | Ensures mesh is truly clean |
| Testing specific slicer | SLICE | Only if user will ONLY use that slicer |
| Sharing filter scripts | STRICT | Mesh must work with ANY slicer |

---

*Document maintained as part of MeshPrep — https://github.com/DragonAceNL/MeshPrep*
