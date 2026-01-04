# MeshPrep Repair Strategy Guide

> **Purpose:** Lessons learned from testing to achieve optimal repair decisions.

**Last Updated:** 2026-01-03  
**Based on:** POC v2/v3 testing with Thingi10K (10,000 models)

---

## Repair Tiers

```
Tier 1: trimesh (fast, safe, limited)
    ↓
Tier 2: pymeshfix (fast, powerful, can be destructive)
    ↓
Tier 3: Blender (slow, most powerful, always works)
```

**Key Principle:** Start least aggressive, escalate when needed.

---

## Tool Comparison

| Tool | Speed | Risk | Best For |
|------|-------|------|----------|
| **trimesh** | ~3-5ms | 🟢 Low | Basic cleanup, already clean models |
| **pymeshfix** | ~10-40ms | 🟡 Med-High | Single-component with holes/non-manifold |
| **Blender** | ~10-15s | 🟡 Medium | Severe damage, last resort |

---

## Critical Lessons

### 1. Check Component Count FIRST

```python
components = mesh.split(only_watertight=False)
if len(components) > 1:
    use_conservative_repair()  # Don't use joincomp=True!
```

### 2. The `joincomp=True` Problem

pymeshfix with `joincomp=True` **destroys multi-component models**:
- Merges all components
- Keeps only largest
- Can lose 50-90% of geometry!

**Solution:** Use `pymeshfix_repair_conservative` for multi-component models.

### 3. Watertight ≠ Correct

A mesh can pass watertight/manifold checks but be destroyed:

| Red Flag | Threshold |
|----------|-----------|
| Face loss | > 40% |
| Volume change | > 30% |
| Bbox change | > 30% |
| Components: many → 1 | Always suspicious |

### 4. Always Start Fresh

Each repair attempt must start from the **original mesh**, not a previously damaged attempt.

### 5. Blender Destroys Fragmented Models

Blender voxel remesh fills gaps between fragments, creating a solid blob.

**Detection:**
```python
analysis = analyze_fragmented_model(mesh)
if analysis["is_fragmented"] and not analysis["blender_safe"]:
    skip_blender_remesh()
```

| Fragmentation Type | Blender Safe? |
|--------------------|---------------|
| `debris-particles` (>70% tiny) | ❌ NO |
| `multi-part-assembly` | ❌ NO |
| `sparse-wireframe` | ❌ NO |
| `split-seam` (<10 components) | ✅ YES |

---

## Filter Selection

| Category | Recommended Filter |
|----------|-------------------|
| clean | `basic-cleanup` |
| holes | `full-repair` or `conservative-repair`* |
| non_manifold | `manifold-repair` or `conservative-repair`* |
| fragmented | `conservative-repair` (MUST) |
| self_intersecting | `full-repair` |

*Use `conservative-repair` if components > 1

---

## Quick Reference

```
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Check components BEFORE any cleanup                │
│  STEP 2: Select filter (multi-component → conservative)     │
│  STEP 3: Validate result (metrics, not just flags)          │
│  STEP 4: Escalate if needed (check fragmentation first!)    │
│                                                             │
│  RED FLAGS:                                                 │
│    ⚠️ Face loss > 40%                                       │
│    ⚠️ Bbox change > 30%                                     │
│    ⚠️ Components: many → 1                                  │
│    ⚠️ Volume 0 → large volume (voxel fill)                  │
│                                                             │
│  NEVER:                                                     │
│    ❌ full-repair on multi-component without checking       │
│    ❌ Trust watertight flag alone                           │
│    ❌ Blender remesh on fragmented models                   │
│    ❌ Stack failed repairs (always fresh start)             │
└─────────────────────────────────────────────────────────────┘
```

---

## Slicer Validation

Use **STRICT mode** (`prusa-slicer --info`) not SLICE mode:

| Mode | Behavior |
|------|----------|
| STRICT (`--info`) | Raw mesh stats, no auto-repair |
| SLICE (`--export-gcode`) | Auto-repairs internally (hides issues) |

---

## Performance

| Operation | Time |
|-----------|------|
| Load STL | 1-5ms |
| trimesh_basic | 2-5ms |
| pymeshfix_repair | 10-40ms |
| blender_remesh | 10-15s |

---

## See Also

- [Filter Actions](filter_actions.md) - Action catalog
- [Validation Guide](validation.md) - Validation criteria
- [Model Profiles](model_profiles.md) - Profile detection
