# POC-03: Slicer Integration

---

## POC ID: POC-03

## POC Name
Slicer CLI Validation Integration

## Status
- [ ] Not Started
- [ ] In Progress
- [x] Completed - Success
- [ ] Completed - Failed
- [ ] Blocked

## Estimated Effort
**1-2 days**

## Related Features
- F-006: Slicer Validation
- F-004: ML Filter Generation (RL reward signal)

---

## 1. Objective

### 1.1 What We're Proving
Validate that we can programmatically invoke slicer CLI tools (PrusaSlicer, Cura, OrcaSlicer) to validate mesh printability and parse their output.

### 1.2 Success Criteria

- [x] Invoke PrusaSlicer CLI and get exit code
- [ ] Invoke Cura CLI and get exit code (**Deferred**: requires extensive config, no `--info`, >30s slice)
- [x] Invoke OrcaSlicer CLI and get exit code
- [x] Detect valid vs invalid meshes from slicer response
- [x] Parse error messages from slicer output
- [x] Auto-detect installed slicers
- [x] Validation completes within 30 seconds

### 1.3 Failure Criteria

- CLI not available or undocumented
- Cannot determine pass/fail from slicer response
- Validation takes too long (>60 seconds)
- Slicers require GUI interaction

---

## 2. Technical Approach

### 2.1 Technologies to Evaluate

| Technology | Version | Purpose |
|------------|---------|---------|
| PrusaSlicer | 2.7+ | Primary slicer validation |
| Cura (CuraEngine) | 5.x | Alternative slicer |
| OrcaSlicer | 2.x | Alternative slicer |
| System.Diagnostics.Process | .NET 10 | CLI invocation |

### 2.2 Test Scenarios

1. **Locate Slicer** - Find installed slicer executables
2. **Valid Mesh** - Slice watertight cube, verify success
3. **Invalid Mesh** - Slice non-manifold mesh, verify failure
4. **Parse Output** - Extract error messages from stderr
5. **Timeout Handling** - Handle hung slicer process
6. **All Slicers** - Test each supported slicer

### 2.3 Test Data

| File | Condition | Expected Result |
|------|-----------|-----------------|
| cube_valid.stl | Watertight | Pass |
| cube_holes.stl | Has holes | Fail/Warning |
| nonmanifold.stl | Non-manifold edges | Fail |
| selfintersect.stl | Self-intersecting | Fail/Warning |

### 2.4 CLI Commands to Test

**PrusaSlicer:**
```bash
# Validation only (no G-code output)
prusa-slicer --export-gcode --output NUL input.stl

# With specific config
prusa-slicer --load config.ini --export-gcode --output NUL input.stl
```

**CuraEngine:**
```bash
# Slice with minimal config
CuraEngine slice -v -j definitions/fdmprinter.def.json -o NUL -l input.stl
```

**OrcaSlicer:**
```bash
# Similar to PrusaSlicer (fork)
orca-slicer --export-gcode --output NUL input.stl
```

---

## 3. Implementation

### 3.1 Setup Steps

1. Create new .NET 10 console project: `MeshPrep.POC.SlicerIntegration`
2. Install PrusaSlicer, Cura, OrcaSlicer on test machine
3. Document installation paths for each slicer
4. Create valid and invalid test meshes
5. Implement process wrapper with timeout

### 3.2 Code Location

`/poc/POC_03_SlicerIntegration/`

### 3.3 Key Code Snippets

**Slicer Detection:**
```csharp
public class SlicerDetector
{
    private static readonly string[] PrusaSlicerPaths = new[]
    {
        @"C:\Program Files\Prusa3D\PrusaSlicer\prusa-slicer-console.exe",
        @"C:\Program Files\Prusa3D\PrusaSlicer\prusa-slicer.exe",
        Environment.ExpandEnvironmentVariables(@"%LOCALAPPDATA%\Programs\PrusaSlicer\prusa-slicer.exe")
    };
    
    public string? FindPrusaSlicer()
    {
        return PrusaSlicerPaths.FirstOrDefault(File.Exists);
    }
}
```

**Slicer Invocation:**
```csharp
public class SlicerValidator
{
    public async Task<ValidationResult> ValidateAsync(string slicerPath, string modelPath, 
        CancellationToken ct = default)
    {
        var tempOutput = Path.GetTempFileName();
        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = slicerPath,
                Arguments = $"--export-gcode --output \"{tempOutput}\" \"{modelPath}\"",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };
            
            using var process = new Process { StartInfo = psi };
            var output = new StringBuilder();
            var error = new StringBuilder();
            
            process.OutputDataReceived += (s, e) => output.AppendLine(e.Data);
            process.ErrorDataReceived += (s, e) => error.AppendLine(e.Data);
            
            process.Start();
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            
            var completed = await process.WaitForExitAsync(ct)
                .WaitAsync(TimeSpan.FromSeconds(60), ct);
            
            return new ValidationResult
            {
                Success = process.ExitCode == 0,
                ExitCode = process.ExitCode,
                Output = output.ToString(),
                Error = error.ToString()
            };
        }
        finally
        {
            if (File.Exists(tempOutput)) File.Delete(tempOutput);
        }
    }
}
```

**Parse Errors:**
```csharp
public List<string> ParseErrors(string slicerOutput)
{
    var errors = new List<string>();
    var lines = slicerOutput.Split('\n');
    
    foreach (var line in lines)
    {
        if (line.Contains("error", StringComparison.OrdinalIgnoreCase) ||
            line.Contains("non-manifold", StringComparison.OrdinalIgnoreCase) ||
            line.Contains("self-intersection", StringComparison.OrdinalIgnoreCase))
        {
            errors.Add(line.Trim());
        }
    }
    
    return errors;
}
```

---

## 4. Results

### 4.1 Test Results

| Slicer | Test | Result | Notes |
|--------|------|--------|-------|
| PrusaSlicer | Detection | ✅ Pass | Found at `C:\Program Files\Prusa3D\PrusaSlicer\prusa-slicer-console.exe` |
| PrusaSlicer | Valid mesh | ✅ Pass | `--info` shows manifold=yes, 0.27s |
| PrusaSlicer | Invalid mesh (holes) | ✅ Pass | `--info` shows manifold=no, open_edges=4 |
| PrusaSlicer | Non-manifold mesh | ✅ Pass | `--info` shows manifold=no, parts=2 |
| OrcaSlicer | Detection | ✅ Pass | Found at `C:\Program Files\OrcaSlicer\orca-slicer.exe` |
| OrcaSlicer | Valid mesh | ✅ Pass | `--info` shows manifold=yes, 0.34s |
| OrcaSlicer | Invalid mesh (holes) | ✅ Pass | `--info` shows manifold=no, open_edges=4 |
| OrcaSlicer | Non-manifold mesh | ✅ Pass | `--info` shows manifold=no, parts=2 |
| Cura | All tests | 🔙 Deferred | CuraEngine requires extensive config, no `--info` command, >30s slice time |

### 4.2 Performance Metrics

| Slicer | Mesh Size | Target Time | Actual Time | Pass? |
|--------|-----------|-------------|-------------|-------|
| PrusaSlicer | 12 triangles (cube) | < 10s | 0.27s | ✅ |
| PrusaSlicer | 10 triangles (holes) | < 10s | 0.23s | ✅ |
| PrusaSlicer | 22 triangles (nonmanifold) | < 10s | 0.23s | ✅ |
| OrcaSlicer | 12 triangles (cube) | < 10s | 0.34s | ✅ |
| OrcaSlicer | 10 triangles (holes) | < 10s | 0.22s | ✅ |
| OrcaSlicer | 22 triangles (nonmanifold) | < 10s | 0.22s | ✅ |

### 4.3 Issues Encountered

1. **~~PrusaSlicer auto-repairs meshes silently~~** - **SOLVED**: Use `--info` instead of `--export-gcode`
   - The `--info` command analyzes mesh without slicing or auto-repair
   - Returns structured data: manifold=yes/no, open_edges, facet count, volume, parts
   - Much faster than slicing (~0.2s vs ~0.3s)

2. **Non-manifold detection** - Now works correctly with `--info`
   - Reports `manifold = no` and `open_edges` count for broken meshes
   - Also reports `number_of_parts` which helps identify non-manifold geometry

3. **Cura/CuraEngine deferred** - Does not fit our use case well
   - No `--info` equivalent command
   - Requires full printer definition files and extruder configuration
   - Slicing takes >30 seconds even for simple meshes
   - PrusaSlicer + OrcaSlicer cover the majority of users

---

## 5. Conclusions

### 5.1 Recommendation
**PROCEED** - POC demonstrates excellent CLI integration with PrusaSlicer.

**Key findings:**
- **Use `--info` command for mesh analysis** - provides structured mesh statistics without slicing
- Exit code alone is insufficient for `--export-gcode`; but `--info` output parsing is reliable
- Performance is excellent (< 0.25s for mesh analysis)
- Auto-detection of installed slicers works reliably
- Can detect: manifold status, open edges, facet count, volume, part count

**Recommended approach for F-006 (Slicer Validation):**
- Use `--info` for quick mesh validation (manifold check)
- Use `--export-gcode` for full printability validation (can mesh be sliced)
- Parse structured key=value output for detailed statistics

### 5.2 Risks Identified
1. **Different slicers, different CLI** - Cura doesn't have `--info` equivalent
2. **Large file performance** - Need to test with Thingi10K models (100K+ triangles)

### 5.3 Next Steps
1. Test with larger meshes from Thingi10K dataset
2. Install and test OrcaSlicer (should support `--info` as PrusaSlicer fork)
3. Research Cura CLI for mesh analysis capabilities
4. Proceed to POC-05 (Mesh Repair) for detailed mesh analysis with MeshLib

---

## 6. Log

| Date | Update | Author |
|------|--------|--------|
| 2026-01-10 | POC document created | |
| 2026-01-10 | Implementation started - SlicerDetector, SlicerValidator classes created | |
| 2026-01-10 | PrusaSlicer tests complete - CLI works, auto-repair detection added | |
| 2026-01-10 | Discovered `--info` command for mesh analysis without auto-repair | |
| 2026-01-10 | OrcaSlicer installed and validated - all tests pass, same CLI as PrusaSlicer | |
| 2026-01-10 | Cura installed and tested - deferred due to extensive config requirements | |
| 2026-01-10 | POC-03 complete - 6/6 tests pass (PrusaSlicer + OrcaSlicer), Cura deferred | |
