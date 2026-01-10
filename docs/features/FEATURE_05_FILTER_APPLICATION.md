# Feature F-005: Filter Script Application

---

## Feature ID: F-005

## Feature Name
Filter Script Application (Apply Repairs)

## Status
- [x] Not Started
- [ ] In Analysis
- [ ] In Development
- [ ] In Testing
- [ ] Complete

## Priority
**High** - Core functionality for ModelFixer

## Estimated Effort
**Medium** (1-3 days)

## Related POC
None - Depends on F-001, F-002, and repair operations

---

## 1. Description

### 1.1 Overview
Execute a filter script (repair sequence) on a loaded 3D model. This is the core "apply" functionality that transforms a broken mesh into a printable one using a predefined sequence of operations.

### 1.2 User Story

As a **ModelFixer user**, I want **to apply a repair script to my model** so that **I can fix it without manual intervention**.

### 1.3 Acceptance Criteria

- [ ] Load filter script from JSON file
- [ ] Verify model fingerprint matches script
- [ ] Execute repair operations in sequence
- [ ] Show progress during application
- [ ] Support cancellation mid-process
- [ ] Report success/failure for each step
- [ ] Produce final repaired mesh

---

## 2. Functional Details

### 2.1 Inputs

| Input | Type | Required | Description |
|-------|------|----------|-------------|
| MeshModel | MeshModel | Yes | The mesh to repair |
| FilterScript | FilterScript | Yes | The repair script to apply |

### 2.2 Outputs

| Output | Type | Description |
|--------|------|-------------|
| RepairedMesh | MeshModel | The fixed mesh |
| ApplicationResult | ApplicationResult | Success/failure details |

### 2.3 Filter Script Format (JSON)

```json
{
  "version": "1.0",
  "name": "Spaceship_Model_Fix",
  "description": "Repair script for spaceship model",
  "createdDate": "2026-01-10T12:00:00Z",
  "fingerprint": {
    "hash": "a1b2c3d4e5f6...",
    "fileSize": 1234567,
    "originalFileName": "spaceship.stl"
  },
  "operations": [
    {
      "type": "FillHoles",
      "parameters": {
        "strategy": "Planar"
      }
    },
    {
      "type": "RemoveSelfIntersections",
      "parameters": {}
    },
    {
      "type": "FixNonManifoldEdges",
      "parameters": {}
    }
  ]
}
```

### 2.4 Business Rules

- Fingerprint must match before script can be applied
- Operations execute in order
- If an operation fails, user can choose to continue or abort
- Final mesh should pass slicer validation

---

## 3. Technical Details

### 3.1 Dependencies

- **MeshLib** - Executes repair operations
- F-002 (Fingerprinting) - Verify model identity

### 3.2 Affected Components

- `MeshPrep.Core` - Script execution engine
- `MeshPrep.ModelFixer` - GUI and CLI interfaces

### 3.3 Technical Approach

```
FilterScript ──► Verify Fingerprint ──► Execute Operations ──► Repaired Mesh
                        │                      │
                        ▼                      ▼
                  Mismatch? ──► Error    Op Failed? ──► Handle Error
```

### 3.4 API/Interface

```csharp
namespace MeshPrep.Core.Scripts
{
    public interface IScriptExecutor
    {
        ApplicationResult Apply(MeshModel mesh, FilterScript script);
        Task<ApplicationResult> ApplyAsync(MeshModel mesh, FilterScript script,
            IProgress<ScriptProgress>? progress = null, CancellationToken ct = default);
    }

    public class FilterScript
    {
        public string Version { get; set; }
        public string Name { get; set; }
        public string Description { get; set; }
        public DateTime CreatedDate { get; set; }
        public ModelFingerprint Fingerprint { get; set; }
        public List<RepairOperation> Operations { get; set; }
    }

    public class RepairOperation
    {
        public string Type { get; set; }
        public Dictionary<string, object> Parameters { get; set; }
    }

    public class ApplicationResult
    {
        public bool Success { get; set; }
        public MeshModel RepairedMesh { get; set; }
        public List<OperationResult> OperationResults { get; set; }
        public TimeSpan TotalTime { get; set; }
    }

    public class ScriptProgress
    {
        public int CurrentOperation { get; set; }
        public int TotalOperations { get; set; }
        public string CurrentOperationName { get; set; }
        public double OperationProgress { get; set; }
    }
}
```

---

## 4. User Interface

### 4.1 UI Changes Required

**ModelFixer GUI:**
- "Apply Script" button
- Progress bar during application
- Operation-by-operation status display
- Cancel button

**ModelFixer CLI:**
- `--script <path>` argument
- Progress output to console
- Exit code for success/failure

### 4.2 User Interaction Flow

```
Load Model ──► Load Script ──► Verify Fingerprint ──► Apply
                                      │                 │
                              ┌───────┴───────┐         │
                              ▼               ▼         │
                          Match ✅      Mismatch ❌     │
                              │               │         │
                              │               ▼         │
                              │         Error message   │
                              │                         │
                              └──────────┬──────────────┘
                                         ▼
                                   Show progress
                                         │
                                         ▼
                                   Complete/Failed
```

---

## 5. Testing

### 5.1 Test Cases

| Test ID | Description | Input | Expected Output | Status |
|---------|-------------|-------|-----------------|--------|
| TC-001 | Apply valid script | Matching model + script | Repaired mesh | ⬜ |
| TC-002 | Fingerprint mismatch | Wrong model | Clear error | ⬜ |
| TC-003 | Multi-step script | Script with 5 operations | All steps execute | ⬜ |
| TC-004 | Cancel mid-process | Long script, cancel | Stops cleanly | ⬜ |
| TC-005 | Operation failure | Script with invalid op | Error reported | ⬜ |
| TC-006 | Empty script | Script with no operations | Original mesh returned | ⬜ |
| TC-007 | CLI execution | Command line args | Correct exit code | ⬜ |

### 5.2 Edge Cases

- Script with unsupported operation type
- Script version mismatch
- Malformed JSON script file
- Mesh becomes invalid mid-repair

---

## 6. Notes & Open Questions

### Open Questions
- [ ] Allow partial application (continue on error)? → **Yes, with user confirmation**
- [ ] Rollback on failure? → **Keep intermediate state, user decides**

### Notes
- Script JSON schema should be versioned
- Consider script signing for security
- Operation parameters validated before execution
- Progress reporting for long operations

---

## 7. Implementation Log

| Date | Update | Author |
|------|--------|--------|
| 2026-01-10 | Feature document created | |
