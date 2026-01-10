# Feature F-009: Script Import/Export

---

## Feature ID: F-009

## Feature Name
Filter Script Import/Export (Sharing)

## Status
- [x] Not Started
- [ ] In Analysis
- [ ] In Development
- [ ] In Testing
- [ ] Complete

## Priority
**High** - Required for script sharing workflow

## Estimated Effort
**Small** (< 1 day)

## Related POC
None - Straightforward JSON I/O

---

## 1. Description

### 1.1 Overview
Save filter scripts to JSON files and load them back. This enables creators to share repair scripts with others who have the same model files.

### 1.2 User Story

As a **filter script creator**, I want **to save and share my repair scripts** so that **others with the same model can repair it automatically**.

### 1.3 Acceptance Criteria

- [ ] Save filter script to JSON file
- [ ] Load filter script from JSON file
- [ ] Validate script format on load
- [ ] Support script versioning
- [ ] Human-readable JSON format (indented)
- [ ] Include metadata (name, description, date)

---

## 2. Functional Details

### 2.1 Inputs

**Export:**
| Input | Type | Required | Description |
|-------|------|----------|-------------|
| FilterScript | FilterScript | Yes | Script to export |
| FilePath | string | Yes | Output file path |

**Import:**
| Input | Type | Required | Description |
|-------|------|----------|-------------|
| FilePath | string | Yes | Script file path |

### 2.2 Outputs

**Export:**
| Output | Type | Description |
|--------|------|-------------|
| Success | bool | Export success |

**Import:**
| Output | Type | Description |
|--------|------|-------------|
| FilterScript | FilterScript | Loaded script |
| ValidationResult | ValidationResult | Any issues found |

### 2.3 Filter Script JSON Schema

```json
{
  "$schema": "https://meshprep.io/schemas/filterscript-v1.json",
  "version": "1.0",
  "metadata": {
    "name": "Spaceship Fix",
    "description": "Repairs common issues in spaceship model",
    "author": "Creator Name",
    "createdDate": "2026-01-10T12:00:00Z",
    "modifiedDate": "2026-01-10T14:30:00Z",
    "tags": ["spaceship", "holes", "non-manifold"]
  },
  "fingerprint": {
    "hash": "a1b2c3d4e5f6789...",
    "fileSize": 1234567,
    "originalFileName": "spaceship.stl"
  },
  "operations": [
    {
      "id": 1,
      "type": "FillHoles",
      "description": "Fill small holes",
      "parameters": {
        "strategy": "Planar",
        "maxHoleSize": 100
      }
    },
    {
      "id": 2,
      "type": "FixNonManifoldEdges",
      "description": "Clean up non-manifold edges",
      "parameters": {}
    }
  ]
}
```

### 2.4 Business Rules

- Script version must be compatible
- Fingerprint is required
- Operations list can be empty
- Unknown operation types cause warning, not error

---

## 3. Technical Details

### 3.1 Dependencies

- `System.Text.Json` (built-in .NET)

### 3.2 Affected Components

- `MeshPrep.Core` - Script serialization
- Both applications - File dialogs

### 3.3 File Extension

- `.meshprep` - Custom extension for filter scripts
- Also support `.json` for compatibility

### 3.4 API/Interface

```csharp
namespace MeshPrep.Core.Scripts
{
    public interface IScriptSerializer
    {
        void Export(FilterScript script, string filePath);
        Task ExportAsync(FilterScript script, string filePath, CancellationToken ct = default);
        
        ScriptLoadResult Import(string filePath);
        Task<ScriptLoadResult> ImportAsync(string filePath, CancellationToken ct = default);
        
        bool Validate(string filePath);
    }

    public class ScriptLoadResult
    {
        public bool Success { get; set; }
        public FilterScript Script { get; set; }
        public List<string> Warnings { get; set; }
        public string ErrorMessage { get; set; }
    }

    public class ScriptMetadata
    {
        public string Name { get; set; }
        public string Description { get; set; }
        public string Author { get; set; }
        public DateTime CreatedDate { get; set; }
        public DateTime ModifiedDate { get; set; }
        public List<string> Tags { get; set; }
    }
}
```

---

## 4. User Interface

### 4.1 UI Changes Required

**FilterScriptCreator:**
- File → Save Script (Ctrl+S)
- File → Save Script As...
- File → Open Script
- Recent scripts list

**ModelFixer:**
- "Load Script" button
- Script info display (name, description)

### 4.2 User Interaction Flow

**Save Script:**
```
Click Save ──► Choose location ──► Enter name/description ──► Write JSON
                                                                  │
                                                                  ▼
                                                          Show success
```

**Load Script:**
```
Click Open ──► Select file ──► Validate JSON ──► Load script
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
               Valid ✅                       Invalid ❌
                    │                               │
                    ▼                               ▼
            Display script info              Show error message
```

---

## 5. Testing

### 5.1 Test Cases

| Test ID | Description | Input | Expected Output | Status |
|---------|-------------|-------|-----------------|--------|
| TC-001 | Export script | Valid FilterScript | Valid JSON file | ⬜ |
| TC-002 | Import script | Valid JSON file | FilterScript object | ⬜ |
| TC-003 | Round-trip | Export then import | Identical script | ⬜ |
| TC-004 | Invalid JSON | Malformed file | Clear error | ⬜ |
| TC-005 | Wrong version | Future version script | Version error | ⬜ |
| TC-006 | Missing fingerprint | Script without fingerprint | Validation error | ⬜ |
| TC-007 | Unknown operation | Script with new op type | Warning, continues | ⬜ |

### 5.2 Edge Cases

- Very large operation list
- Unicode in metadata
- File permissions issues
- Script with no operations

---

## 6. Notes & Open Questions

### Open Questions
- [x] Custom file extension? → **Yes, `.meshprep`**
- [ ] Support script compression? → **Not for v1.0**

### Notes
- JSON chosen for human readability and easy debugging
- Consider adding script signing in future versions
- Store recent scripts in user preferences
- Validate schema version before attempting to load

---

## 7. Implementation Log

| Date | Update | Author |
|------|--------|--------|
| 2026-01-10 | Feature document created | |
