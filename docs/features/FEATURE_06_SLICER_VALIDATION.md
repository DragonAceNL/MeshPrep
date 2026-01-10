# Feature F-006: Slicer Validation

---

## Feature ID: F-006

## Feature Name
Slicer Validation (Printability Check)

## Status
- [x] Not Started
- [ ] In Analysis
- [ ] In Development
- [ ] In Testing
- [ ] Complete

## Priority
**High** - Required for RL reward signal and print readiness verification

## Estimated Effort
**Medium** (1-3 days)

## Related POC
**POC-03** - Slicer Integration validation

---

## 1. Description

### 1.1 Overview
Validate repaired meshes by sending them to actual slicer software (PrusaSlicer, Cura, OrcaSlicer) via CLI. A successful slice confirms the mesh is printable.

### 1.2 User Story

As a **3D printing enthusiast**, I want **to verify my repaired model can actually be sliced** so that **I'm confident it will print correctly**.

### 1.3 Acceptance Criteria

- [ ] Support PrusaSlicer CLI validation
- [ ] Support Cura CLI validation  
- [ ] Support OrcaSlicer CLI validation
- [ ] Auto-detect installed slicers
- [ ] Configure slicer paths in settings
- [ ] Report validation result (pass/fail)
- [ ] Parse slicer output for specific errors
- [ ] Validation completes in < 30 seconds for typical models

---

## 2. Functional Details

### 2.1 Inputs

| Input | Type | Required | Description |
|-------|------|----------|-------------|
| MeshModel | MeshModel | Yes | The mesh to validate |
| SlicerType | SlicerType | No | Which slicer to use (default: auto) |
| SlicerPath | string | No | Custom slicer executable path |

### 2.2 Outputs

| Output | Type | Description |
|--------|------|-------------|
| ValidationResult | SlicerValidationResult | Pass/fail with details |

### 2.3 Processing Logic

1. Export mesh to temporary STL file
2. Invoke slicer CLI with validation/slice command
3. Parse slicer output/exit code
4. Clean up temporary files
5. Return validation result

### 2.4 Business Rules

- At least one slicer must be available
- Validation uses minimal slice settings (speed over quality)
- Timeout after 60 seconds
- Failed validation provides reason when available

---

## 3. Technical Details

### 3.1 Dependencies

- External: PrusaSlicer, Cura, or OrcaSlicer installed
- `System.Diagnostics.Process` for CLI invocation

### 3.2 Affected Components

- `MeshPrep.Core` - Slicer integration
- Settings/configuration for slicer paths
- RL system - Uses validation as reward signal

### 3.3 Slicer CLI Commands

**PrusaSlicer:**
```bash
prusa-slicer --export-gcode --output NUL model.stl
# Exit code 0 = success, non-zero = failure
```

**Cura:**
```bash
cura-engine slice -v -j printer.json -o NUL -l model.stl
# Exit code 0 = success, non-zero = failure
```

**OrcaSlicer:**
```bash
orca-slicer --export-gcode --output NUL model.stl
# Similar to PrusaSlicer
```

### 3.4 API/Interface

```csharp
namespace MeshPrep.Core.Validation
{
    public interface ISlicerValidator
    {
        SlicerValidationResult Validate(MeshModel mesh, SlicerType? slicer = null);
        Task<SlicerValidationResult> ValidateAsync(MeshModel mesh, 
            SlicerType? slicer = null, CancellationToken ct = default);
        List<SlicerInfo> GetAvailableSlicers();
    }

    public enum SlicerType
    {
        Auto,
        PrusaSlicer,
        Cura,
        OrcaSlicer
    }

    public class SlicerValidationResult
    {
        public bool IsValid { get; set; }
        public SlicerType SlicerUsed { get; set; }
        public string SlicerVersion { get; set; }
        public string ErrorMessage { get; set; }
        public List<string> Warnings { get; set; }
        public TimeSpan ValidationTime { get; set; }
    }

    public class SlicerInfo
    {
        public SlicerType Type { get; set; }
        public string Path { get; set; }
        public string Version { get; set; }
        public bool IsAvailable { get; set; }
    }
}
```

---

## 4. User Interface

### 4.1 UI Changes Required

**FilterScriptCreator:**
- "Validate with Slicer" button
- Slicer selection dropdown
- Validation result indicator (✅/❌)

**Settings:**
- Slicer path configuration
- Default slicer selection
- Auto-detect slicers button

### 4.2 User Interaction Flow

```
Click "Validate" ──► Export temp STL ──► Invoke slicer ──► Parse result
                                              │
                                              ▼
                                      Wait for completion
                                              │
                              ┌───────────────┴───────────────┐
                              ▼                               ▼
                          Success ✅                     Failed ❌
                              │                               │
                              ▼                               ▼
                     "Ready to print"              Show error details
```

---

## 5. Testing

### 5.1 Test Cases

| Test ID | Description | Input | Expected Output | Status |
|---------|-------------|-------|-----------------|--------|
| TC-001 | Validate clean mesh | Watertight STL | Pass | ⬜ |
| TC-002 | Validate broken mesh | Non-manifold mesh | Fail | ⬜ |
| TC-003 | Auto-detect slicer | System with PrusaSlicer | Slicer found | ⬜ |
| TC-004 | Custom slicer path | Valid path | Works | ⬜ |
| TC-005 | Slicer not found | Invalid path | Clear error | ⬜ |
| TC-006 | Validation timeout | Stuck slicer | Timeout after 60s | ⬜ |
| TC-007 | Multiple slicers | All three installed | All detected | ⬜ |

### 5.2 Edge Cases

- Slicer crashes during validation
- Very large model (slow slicing)
- Slicer installed but not in PATH
- Different slicer versions/behaviors

---

## 6. Notes & Open Questions

### Open Questions
- [x] Which slicers to support? → **PrusaSlicer, Cura, OrcaSlicer**
- [ ] Include estimated print time in result? → **Nice to have, not required**

### Notes
- Validation is a key RL reward signal
- Use separate temp directory per validation (parallel execution)
- Consider caching validation results
- Slicer auto-detection searches common install paths

---

## 7. Implementation Log

| Date | Update | Author |
|------|--------|--------|
| 2026-01-10 | Feature document created | |
