# Feature F-010: Batch Processing

---

## Feature ID: F-010

## Feature Name
Batch Processing (CLI)

## Status
- [x] Not Started
- [ ] In Analysis
- [ ] In Development
- [ ] In Testing
- [ ] Complete

## Priority
**Medium** - Power user feature

## Estimated Effort
**Medium** (1-3 days)

## Related POC
None - Builds on existing functionality

---

## 1. Description

### 1.1 Overview
Process multiple model files using the same filter script via command line interface. Enables automation and integration with other workflows.

### 1.2 User Story

As a **power user**, I want **to repair multiple models at once from the command line** so that **I can automate my workflow**.

### 1.3 Acceptance Criteria

- [ ] Process multiple input files in one command
- [ ] Support wildcard patterns (*.stl)
- [ ] Apply same script to all files (fingerprint override)
- [ ] Output to specified directory
- [ ] Generate processing report
- [ ] Continue on error (don't stop batch)
- [ ] Return appropriate exit codes

---

## 2. Functional Details

### 2.1 CLI Usage

```bash
# Single file
meshprep-fixer --input model.stl --script fix.meshprep --output repaired.stl

# Multiple files
meshprep-fixer --input "*.stl" --script fix.meshprep --output-dir ./repaired/

# Batch with report
meshprep-fixer --input folder/*.stl --script fix.meshprep --output-dir ./out/ --report report.json

# Skip fingerprint check (for batch with different models)
meshprep-fixer --input "*.stl" --script fix.meshprep --output-dir ./out/ --skip-fingerprint
```

### 2.2 Inputs

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| --input | string | Yes | File path or wildcard pattern |
| --script | string | Yes | Filter script path |
| --output | string | No | Output file (single file mode) |
| --output-dir | string | No | Output directory (batch mode) |
| --skip-fingerprint | flag | No | Skip fingerprint verification |
| --report | string | No | Generate report file |
| --continue-on-error | flag | No | Don't stop on individual failures |
| --parallel | int | No | Number of parallel processes |

### 2.3 Outputs

| Output | Description |
|--------|-------------|
| Repaired files | One output per input |
| Exit code | 0=success, 1=some failed, 2=all failed |
| Console output | Progress and summary |
| Report file | JSON report if requested |

### 2.4 Business Rules

- Output filenames: `{original}_repaired.stl`
- Fingerprint check skipped in batch mode by default
- Failed files logged but don't stop batch
- Parallel processing limited by CPU cores

---

## 3. Technical Details

### 3.1 Dependencies

- `System.CommandLine` (NuGet) - CLI parsing
- Existing import/export/script features

### 3.2 Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All files processed successfully |
| 1 | Some files failed, some succeeded |
| 2 | All files failed |
| 3 | Invalid arguments |
| 4 | Script not found |
| 5 | No input files matched |

### 3.3 Report Format

```json
{
  "startTime": "2026-01-10T12:00:00Z",
  "endTime": "2026-01-10T12:05:00Z",
  "script": "fix.meshprep",
  "totalFiles": 10,
  "successful": 8,
  "failed": 2,
  "results": [
    {
      "input": "model1.stl",
      "output": "model1_repaired.stl",
      "success": true,
      "processingTime": "2.5s"
    },
    {
      "input": "model2.stl",
      "output": null,
      "success": false,
      "error": "Non-manifold edges could not be fixed"
    }
  ]
}
```

### 3.4 API/Interface

```csharp
namespace MeshPrep.ModelFixer.CLI
{
    public class BatchProcessor
    {
        public Task<BatchResult> ProcessAsync(BatchOptions options, 
            IProgress<BatchProgress>? progress = null, CancellationToken ct = default);
    }

    public class BatchOptions
    {
        public List<string> InputFiles { get; set; }
        public string ScriptPath { get; set; }
        public string OutputDirectory { get; set; }
        public bool SkipFingerprint { get; set; }
        public bool ContinueOnError { get; set; }
        public int MaxParallelism { get; set; } = 1;
    }

    public class BatchResult
    {
        public int TotalFiles { get; set; }
        public int Successful { get; set; }
        public int Failed { get; set; }
        public List<FileResult> Results { get; set; }
        public TimeSpan TotalTime { get; set; }
    }
}
```

---

## 4. User Interface

### 4.1 UI Changes Required

**ModelFixer CLI:**
- All command-line argument handling
- Progress output to console
- Summary at completion

**ModelFixer GUI (optional):**
- "Batch Process" menu item
- Folder selection dialog
- Progress window

### 4.2 Console Output

```
MeshPrep ModelFixer v1.0
Processing 10 files with script: fix.meshprep
Output directory: ./repaired/

[1/10] model1.stl... OK (2.3s)
[2/10] model2.stl... OK (1.8s)
[3/10] model3.stl... FAILED: Non-manifold repair failed
[4/10] model4.stl... OK (3.1s)
...

Summary:
  Processed: 10 files
  Successful: 8 (80%)
  Failed: 2 (20%)
  Total time: 45.2s
  
Exit code: 1 (some files failed)
```

---

## 5. Testing

### 5.1 Test Cases

| Test ID | Description | Input | Expected Output | Status |
|---------|-------------|-------|-----------------|--------|
| TC-001 | Single file | One STL | Repaired file | ⬜ |
| TC-002 | Multiple files | 5 STL files | 5 repaired files | ⬜ |
| TC-003 | Wildcard pattern | *.stl | All matching processed | ⬜ |
| TC-004 | Some failures | Mixed valid/invalid | Partial success | ⬜ |
| TC-005 | All failures | All invalid | Exit code 2 | ⬜ |
| TC-006 | Generate report | --report flag | JSON report created | ⬜ |
| TC-007 | Parallel processing | --parallel 4 | Faster completion | ⬜ |

### 5.2 Edge Cases

- No files match pattern
- Output directory doesn't exist
- Disk full during processing
- Process interrupted (Ctrl+C)

---

## 6. Notes & Open Questions

### Open Questions
- [x] Support parallel processing? → **Yes, optional --parallel flag**
- [ ] Support recursive directory scan? → **Consider for future**

### Notes
- Batch mode useful for processing entire model libraries
- Consider memory management for large batches
- Allow interruption with graceful cleanup
- Report useful for integration with other tools

---

## 7. Implementation Log

| Date | Update | Author |
|------|--------|--------|
| 2026-01-10 | Feature document created | |
