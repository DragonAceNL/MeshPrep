# Feature F-008: STL Export

---

## Feature ID: F-008

## Feature Name
STL File Export

## Status
- [x] Not Started
- [ ] In Analysis
- [ ] In Development
- [ ] In Testing
- [ ] Complete

## Priority
**High** - Required output format for 3D printing

## Estimated Effort
**Small** (< 1 day)

## Related POC
None - Straightforward implementation

---

## 1. Description

### 1.1 Overview
Export repaired meshes to STL format (binary or ASCII), the standard format accepted by all 3D printers and slicers.

### 1.2 User Story

As a **3D printing enthusiast**, I want **to export my repaired model as an STL file** so that **I can slice and print it**.

### 1.3 Acceptance Criteria

- [ ] Export to binary STL (default, smaller file size)
- [ ] Export to ASCII STL (optional, human-readable)
- [ ] Preserve mesh geometry accurately
- [ ] Handle meshes with millions of triangles
- [ ] Export completes in < 5 seconds for 1M triangles
- [ ] Generated STL validates in slicers

---

## 2. Functional Details

### 2.1 Inputs

| Input | Type | Required | Description |
|-------|------|----------|-------------|
| MeshModel | MeshModel | Yes | The mesh to export |
| FilePath | string | Yes | Output file path |
| Options | ExportOptions | No | Export settings |

### 2.2 Outputs

| Output | Type | Description |
|--------|------|-------------|
| Success | bool | Export success/failure |
| FilePath | string | Path to exported file |
| FileSize | long | Size of exported file |

### 2.3 Processing Logic

1. Validate mesh is exportable (has triangles)
2. Open output file for writing
3. Write STL header (binary) or "solid" line (ASCII)
4. Write all triangles with normals
5. Write STL footer
6. Close file

### 2.4 Business Rules

- Binary STL is default (smaller files)
- ASCII STL optional for debugging/compatibility
- Overwrite existing files only with confirmation
- Validate output file after writing

---

## 3. Technical Details

### 3.1 Dependencies

- Built-in .NET file I/O
- MeshLib for mesh data access

### 3.2 Affected Components

- `MeshPrep.Core` - Export implementation
- Both GUI applications - Save dialog
- `MeshPrep.ModelFixer` CLI - Output argument

### 3.3 STL Format

**Binary STL:**
```
UINT8[80] – Header (ignored)
UINT32 – Number of triangles
foreach triangle
    REAL32[3] – Normal vector
    REAL32[3] – Vertex 1
    REAL32[3] – Vertex 2
    REAL32[3] – Vertex 3
    UINT16 – Attribute byte count (0)
end
```

**ASCII STL:**
```
solid name
  facet normal ni nj nk
    outer loop
      vertex v1x v1y v1z
      vertex v2x v2y v2z
      vertex v3x v3y v3z
    endloop
  endfacet
endsolid name
```

### 3.4 API/Interface

```csharp
namespace MeshPrep.Core.Export
{
    public interface IMeshExporter
    {
        ExportResult Export(MeshModel mesh, string filePath, ExportOptions? options = null);
        Task<ExportResult> ExportAsync(MeshModel mesh, string filePath, 
            ExportOptions? options = null, IProgress<double>? progress = null,
            CancellationToken ct = default);
    }

    public class ExportOptions
    {
        public StlFormat Format { get; set; } = StlFormat.Binary;
        public bool OverwriteExisting { get; set; } = false;
    }

    public enum StlFormat
    {
        Binary,
        ASCII
    }

    public class ExportResult
    {
        public bool Success { get; set; }
        public string FilePath { get; set; }
        public long FileSize { get; set; }
        public int TriangleCount { get; set; }
        public TimeSpan ExportTime { get; set; }
        public string ErrorMessage { get; set; }
    }
}
```

---

## 4. User Interface

### 4.1 UI Changes Required

**FilterScriptCreator:**
- File → Export STL menu item
- Format selection (Binary/ASCII)

**ModelFixer GUI:**
- "Save Repaired" button
- Save file dialog

**ModelFixer CLI:**
- `--output <filepath>` argument
- `--format binary|ascii` option

### 4.2 User Interaction Flow

```
Click "Export" ──► Choose location ──► Select format ──► Write file
                                                            │
                                                            ▼
                                                    Show success/size
```

---

## 5. Testing

### 5.1 Test Cases

| Test ID | Description | Input | Expected Output | Status |
|---------|-------------|-------|-----------------|--------|
| TC-001 | Export binary STL | Valid mesh | Valid binary STL | ⬜ |
| TC-002 | Export ASCII STL | Valid mesh | Valid ASCII STL | ⬜ |
| TC-003 | Large mesh export | 1M triangles | Completes < 5s | ⬜ |
| TC-004 | Validate in slicer | Exported STL | Slicer accepts | ⬜ |
| TC-005 | Overwrite protection | Existing file | Prompts user | ⬜ |
| TC-006 | Invalid path | Read-only location | Clear error | ⬜ |
| TC-007 | Empty mesh | 0 triangles | Appropriate handling | ⬜ |

### 5.2 Edge Cases

- Disk full during write
- Path with special characters
- Network path (slow write)
- Very long file paths

---

## 6. Notes & Open Questions

### Open Questions
- [x] Support other export formats? → **STL only for v1.0, 3MF later**

### Notes
- Binary STL is ~5x smaller than ASCII
- Most slicers prefer binary STL
- Consider buffered writing for large files
- Validate output matches input triangle count

---

## 7. Implementation Log

| Date | Update | Author |
|------|--------|--------|
| 2026-01-10 | Feature document created | |
