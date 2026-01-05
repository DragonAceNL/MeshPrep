# Error Handling & Stability

## Overview

**Stability is MeshPrep's #1 priority.** The system is designed to:

1. **Never crash** - Gracefully handle all errors
2. **Always produce output** - Return original mesh if repair fails
3. **Learn from failures** - Track errors to improve over time
4. **Provide visibility** - Log everything for debugging

---

## Design Principles

### 1. Fail-Safe by Default

Every action follows this pattern:

```python
def action_example(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    try:
        # Attempt repair
        result = do_repair(mesh)
        return result
    except Exception as e:
        logger.error(f"action_example failed: {e}")
        log_action_failure("action_example", str(e), mesh)
        return mesh.copy()  # ALWAYS return something usable
```

**Key guarantees:**
- Actions NEVER raise exceptions to callers
- Actions ALWAYS return a valid mesh (original if repair fails)
- Errors are logged but don't stop processing

### 2. Defense in Depth

Multiple layers of error handling:

| Layer | Responsibility | Recovery |
|-------|---------------|----------|
| **Action** | Handle library-specific errors | Return original mesh |
| **Pipeline** | Handle action failures | Try next pipeline |
| **Repair Loop** | Handle pipeline exhaustion | Escalate to Blender |
| **Batch Runner** | Handle model failures | Continue to next model |
| **Process** | Handle crashes | Log and restart |

### 3. Graceful Degradation

When repair fails, the system degrades gracefully:

```
Best: Repair succeeds with fast pipeline
  ↓
Good: Repair succeeds with slow pipeline
  ↓
OK: Blender escalation succeeds
  ↓
Fallback: Return original mesh (logged as failed)
  ↓
Worst: Process crash (auto-resume on restart)
```

---

## Error Logging Architecture

### Dual Logging System

Errors are logged to **both** text files and SQLite database:

```
Action Fails
    │
    ├──► Python Logger → Console/File
    │
    └──► Error Logging System
              │
              ├──► Text Log (daily rotation)
              │    learning_data/error_logs/errors_YYYY-MM-DD.log
              │
              └──► SQLite Database (for learning)
                   learning_data/action_crashes.db
```

### What Gets Logged

| Field | Description | Example |
|-------|-------------|---------|
| `timestamp` | When error occurred | `2026-01-05 14:32:01` |
| `action_name` | Action that failed | `pymeshfix_repair` |
| `action_type` | Category of action | `pymeshfix` |
| `error_message` | Full error text | `Memory allocation failed` |
| `error_category` | Classified category | `memory` |
| `model_id` | Model being processed | `100027` |
| `model_fingerprint` | Unique model hash | `MP:abc123def456` |
| `face_count` | Faces in mesh | `50000` |
| `vertex_count` | Vertices in mesh | `25000` |
| `body_count` | Disconnected components | `3` |
| `size_bin` | Size category | `medium` |
| `pipeline_name` | Current pipeline | `pymeshfix` |
| `attempt_number` | Which attempt | `2` |

### Error Categories

Errors are automatically classified into categories:

| Category | Pattern | Common Cause |
|----------|---------|--------------|
| `memory` | OutOfMemory, allocation | Large meshes |
| `normals_required` | normals, normal estimation | Missing vertex normals |
| `topology` | non-manifold, degenerate | Invalid geometry |
| `empty_mesh` | empty, no faces | Destroyed by action |
| `timeout` | timeout, timed out | Complex operations |
| `file_io` | file, read, write | Disk issues |
| `blender` | Blender, bpy | Blender subprocess |
| `pymeshlab` | pymeshlab, MeshLab | PyMeshLab filters |
| `unknown` | (default) | Uncategorized |

---

## Error Tracking Databases

### 1. Action Failures (`action_crashes.db`)

Tracks individual action failures for pattern learning.

**Tables:**

```sql
-- Individual failure records
CREATE TABLE action_failures (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    action_name TEXT,
    error_category TEXT,
    error_message TEXT,
    face_count INTEGER,
    vertex_count INTEGER,
    body_count INTEGER,
    size_bin TEXT,
    pymeshlab_version TEXT
);

-- Aggregated patterns with skip recommendations
CREATE TABLE failure_patterns (
    id INTEGER PRIMARY KEY,
    action_name TEXT,
    error_category TEXT,
    size_bin TEXT,
    pymeshlab_version TEXT,
    failure_count INTEGER,
    total_attempts INTEGER,
    failure_rate REAL,
    should_skip INTEGER,  -- 1 = skip this combination
    last_updated TEXT
);
```

### 2. Process Crashes (`action_crashes.db`)

Tracks subprocess crashes (Blender, slicer, etc.)

```sql
CREATE TABLE crash_patterns (
    id INTEGER PRIMARY KEY,
    action_name TEXT,
    size_bin TEXT,
    crash_count INTEGER,
    success_count INTEGER,
    last_crash TEXT
);
```

---

## Learning from Errors

### Crash-Prone Actions

Certain actions are known to potentially crash or hang. These are run in isolated subprocesses with timeout protection:

| Action | Risk | Timeout |
|--------|------|--------|
| `meshlab_reconstruct_poisson` | Access violation | 120s |
| `meshlab_reconstruct_ball_pivoting` | Access violation | 120s |
| `meshlab_alpha_wrap` | Access violation | 120s |
| `meshlab_boolean_union` | Access violation | 120s |
| `meshlab_repair` | Access violation | 120s |
| `poisson_reconstruction` (Open3D) | Memory/hang | 120s |
| `ball_pivoting` | Memory/hang | 120s |
| `pymeshfix_repair` | Hang on certain meshes | 120s |
| `pymeshfix_repair_conservative` | Hang on certain meshes | 120s |
| `pymeshfix_clean` | Hang on certain meshes | 120s |
| `make_manifold` | Hang on certain meshes | 120s |

If a crash-prone action exceeds its timeout, it is forcibly killed and recorded as a hang for future learning.

---

## Skip Recommendations

After sufficient data, the system learns which action+mesh combinations to skip:

```
Pattern Detected:
  action: meshlab_reconstruct_poisson
  category: normals_required
  size: huge
  failures: 47
  attempts: 50
  failure_rate: 94%
  
Recommendation: SKIP this action for huge meshes
```

**Skip threshold:** 70%+ failure rate with 3+ attempts

### Version-Aware Skip Recommendations

**Critical:** Skip recommendations are **version-specific**. When MeshPrep or PyMeshLab is updated, old skip recommendations **do not apply** - the new version gets a fresh chance since bugs may have been fixed.

```
Version 0.2.0: meshlab_reconstruct_poisson marked as skip (94% fail)
          ↓
Upgrade to 0.2.1 (bug fix in normal handling)
          ↓
Version 0.2.1: Fresh start - meshlab_reconstruct_poisson will be tried again
          ↓
If it works now: System learns it works in 0.2.1
If it still fails: System will re-mark as skip for 0.2.1
```

**What gets tracked per version:**

| Field | Description |
|-------|-------------|
| `pymeshlab_version` | Version of PyMeshLab library |
| `meshprep_version` | Version of MeshPrep (from `__version__`) |
| `skip_set_version` | Which version set the skip recommendation |
| `skip_set_timestamp` | When the skip was set |

**CLI to reset skips for current version:**
```bash
python run_full_test.py --reset-skips
```

### Pipeline Reordering

The learning engine uses error data to reorder pipelines:

```python
# Before learning (default order)
pipelines = ["pymeshfix", "meshlab-repair", "blender-remesh"]

# After learning (based on success rates)
pipelines = ["pymeshfix", "blender-remesh", "meshlab-repair"]
# meshlab-repair moved last because it fails more often
```

### Error Pattern Analysis

Common patterns detected by the system:

| Pattern | Learned Response |
|---------|-----------------|
| Poisson fails on meshes without normals | Add `fix_normals` before Poisson |
| PyMeshFix destroys multi-body meshes | Use `pymeshfix_repair_conservative` instead |
| Ball pivoting fails on small meshes | Skip for tiny meshes |
| Blender timeout on huge meshes | Increase timeout dynamically |

---

## Viewing Error Data

### CLI Commands

```bash
# Show error statistics
python run_full_test.py --error-stats

# Output:
# ============================================================
# MeshPrep Error/Crash Statistics
# ============================================================
# 
# [LOG] Today's Error Log: errors_2026-01-05.log
#    Total errors: 156
# 
#    By Category:
#       normals_required: 45
#       memory: 32
#       topology: 28
#       unknown: 51
# 
#    By Action:
#       meshlab_reconstruct_poisson: 45
#       pymeshfix_repair: 22
#       blender_remesh: 15
# ...
```

### Web Interface

Access error logs via the reports server:

```
http://localhost:8000/errors/
```

Features:
- Paginated error list (50 per page)
- Filter by date, category, action
- Summary statistics
- Real-time updates

### API Endpoint

```
GET /api/errors?page=1&per_page=50&category=memory&action=pymeshfix_repair

Response:
{
  "success": true,
  "page": 1,
  "total": 32,
  "total_pages": 1,
  "entries": [...],
  "summary": {
    "by_category": {"memory": 32},
    "by_action": {"pymeshfix_repair": 32}
  }
}
```

---

## Crash Protection

### Subprocess Isolation

Dangerous operations run in subprocesses:

```python
# Blender runs as subprocess - crash doesn't affect main process
result = run_blender_script(mesh, "remesh", params)

# If Blender crashes:
# - Main process continues
# - Crash is logged
# - Next pipeline is tried
```

### Timeout Protection

All external operations have timeouts:

| Operation | Default Timeout | Purpose |
|-----------|----------------|---------|
| Blender remesh | None (poll-based) | Allow long operations |
| Slicer validation | 60 seconds | Prevent hangs |
| PyMeshLab filter | 120 seconds | Prevent infinite loops |

### Memory Protection

Large mesh detection and handling:

```python
# Before processing
if mesh.faces > 1_000_000:
    logger.warning("Large mesh detected, using memory-safe mode")
    # Use streaming/chunked processing
    # Avoid memory-intensive operations
```

---

## Recovery Mechanisms

### Auto-Resume

Batch processing automatically resumes after interruption:

```bash
# First run - processes 500 models, then power outage
python run_full_test.py

# Second run - automatically skips processed models
python run_full_test.py
# "Skipping 500 already processed models..."
```

### Transaction Safety

SQLite databases use transactions:

```python
with db.transaction():
    # All or nothing
    db.record_attempt(...)
    db.update_stats(...)
# If crash here, database is consistent
```

### Checkpoint System

Progress is saved after each model:

```json
// progress.json - updated after every model
{
  "processed": 5432,
  "fixed": 4521,
  "failed": 911,
  "last_model": "100532.stl",
  "last_update": "2026-01-05T14:32:01"
}
```

---

## Action-Specific Error Handling

### PyMeshLab Actions

```python
# All PyMeshLab actions use shared error handler
from .error_logging import log_action_failure

def action_meshlab_repair(mesh, params):
    try:
        ms = trimesh_to_pymeshlab(mesh)
        ms.apply_filter(...)
        return pymeshlab_to_trimesh(ms)
    except Exception as e:
        logger.error(f"meshlab_repair failed: {e}")
        log_action_failure("meshlab_repair", str(e), mesh, "pymeshlab")
        return mesh.copy()
```

### PyMeshFix Actions

```python
def action_pymeshfix_repair(mesh, params):
    try:
        meshfix = trimesh_to_pymeshfix(mesh)
        meshfix.repair(...)
        return pymeshfix_to_trimesh(meshfix)
    except Exception as e:
        logger.error(f"pymeshfix_repair failed: {e}")
        log_action_failure("pymeshfix_repair", str(e), mesh, "pymeshfix")
        return mesh.copy()
```

### Trimesh Actions

```python
def action_fill_holes(mesh, params):
    mesh = mesh.copy()
    try:
        trimesh.repair.fill_holes(mesh)
    except Exception as e:
        logger.warning(f"fill_holes failed: {e}")
        log_action_failure("fill_holes", str(e), mesh, "trimesh")
    return mesh  # Return even if partially failed
```

### Blender Actions

```python
def run_blender_script(mesh, operation, params):
    # Blender runs in subprocess
    process = subprocess.Popen([blender, ...])
    
    try:
        # Poll-based waiting (no timeout)
        while process.poll() is None:
            time.sleep(5)
            logger.info("Blender still running...")
        
        if process.returncode != 0:
            raise RuntimeError("Blender failed")
        
        return load_result()
    except Exception:
        # Ensure process is terminated
        if process.poll() is None:
            process.terminate()
        raise  # Let caller handle
```

---

## Best Practices

### For Contributors

1. **Always wrap external calls in try/except**
2. **Always log errors with context**
3. **Always return original mesh on failure**
4. **Use the shared `log_action_failure()` function**
5. **Include mesh characteristics in error logs**

### For Users

1. **Check `--error-stats` regularly** to spot patterns
2. **Use web interface** at `/errors/` for detailed analysis
3. **Report new error patterns** as issues on GitHub
4. **Don't delete error logs** - they help improve the system

---

## See Also

- [Learning Systems](learning_systems.md) - How errors inform learning
- [Repair Strategy Guide](repair_strategy_guide.md) - When actions fail
- [Functional Spec](functional_spec.md) - System overview
