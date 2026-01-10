# POC-03: Slicer Integration

---

## POC ID: POC-03

## POC Name
Slicer CLI Validation Integration

## Status
- [x] Not Started
- [ ] In Progress
- [ ] Completed - Success
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

- [ ] Invoke PrusaSlicer CLI and get exit code
- [ ] Invoke Cura CLI and get exit code
- [ ] Invoke OrcaSlicer CLI and get exit code
- [ ] Detect valid vs invalid meshes from slicer response
- [ ] Parse error messages from slicer output
- [ ] Auto-detect installed slicers
- [ ] Validation completes within 30 seconds

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
| PrusaSlicer | Detection | ⬜ | |
| PrusaSlicer | Valid mesh | ⬜ | |
| PrusaSlicer | Invalid mesh | ⬜ | |
| PrusaSlicer | Error parsing | ⬜ | |
| Cura | Detection | ⬜ | |
| Cura | Valid mesh | ⬜ | |
| Cura | Invalid mesh | ⬜ | |
| OrcaSlicer | Detection | ⬜ | |
| OrcaSlicer | Valid mesh | ⬜ | |
| OrcaSlicer | Invalid mesh | ⬜ | |

### 4.2 Performance Metrics

| Slicer | Mesh Size | Target Time | Actual Time | Pass? |
|--------|-----------|-------------|-------------|-------|
| PrusaSlicer | 10K triangles | < 10s | | ⬜ |
| PrusaSlicer | 100K triangles | < 30s | | ⬜ |
| Cura | 10K triangles | < 10s | | ⬜ |
| Cura | 100K triangles | < 30s | | ⬜ |

### 4.3 Issues Encountered

*To be filled during POC execution*

---

## 5. Conclusions

### 5.1 Recommendation
*To be filled after POC completion*

### 5.2 Risks Identified
*To be filled after POC completion*

### 5.3 Next Steps
*To be filled after POC completion*

---

## 6. Log

| Date | Update | Author |
|------|--------|--------|
| 2026-01-10 | POC document created | |
