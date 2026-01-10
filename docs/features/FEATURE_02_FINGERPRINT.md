# Feature F-002: Model Fingerprinting

---

## Feature ID: F-002

## Feature Name
Model Fingerprinting (SHA-256 File Hash)

## Status
- [x] Not Started
- [ ] In Analysis
- [ ] In Development
- [ ] In Testing
- [ ] Complete

## Priority
**High** - Required for script-model binding

## Estimated Effort
**Small** (< 1 day)

## Related POC
**POC-02** - Fingerprinting validation

---

## 1. Description

### 1.1 Overview
Generate a unique fingerprint for each 3D model file using SHA-256 hash. This fingerprint binds filter scripts to specific models, ensuring legal compliance and preventing misuse.

### 1.2 User Story

As a **filter script creator**, I want **my repair scripts to be bound to specific model files** so that **they cannot be used with unauthorized copies or different models**.

### 1.3 Acceptance Criteria

- [ ] Generate SHA-256 hash of entire file
- [ ] Hash calculation completes in < 1 second for files up to 100MB
- [ ] Fingerprint is deterministic (same file = same hash always)
- [ ] Store fingerprint in filter script JSON
- [ ] Verify fingerprint before applying script
- [ ] Support fingerprinting files up to 500MB

---

## 2. Functional Details

### 2.1 Inputs

| Input | Type | Required | Description |
|-------|------|----------|-------------|
| File Path | string | Yes | Path to the 3D model file |

### 2.2 Outputs

| Output | Type | Description |
|--------|------|-------------|
| Fingerprint | string | 64-character hex SHA-256 hash |
| FileSize | long | File size in bytes (for quick pre-check) |

### 2.3 Processing Logic

1. Open file as read-only stream
2. Compute SHA-256 hash of entire file contents
3. Convert hash bytes to lowercase hex string
4. Return fingerprint

### 2.4 Business Rules

- Fingerprint must match exactly for script to apply
- Even 1 byte difference results in different fingerprint
- File size stored alongside for quick mismatch detection
- Original filename stored for user reference (not for matching)

---

## 3. Technical Details

### 3.1 Dependencies

- `System.Security.Cryptography` (built-in .NET)

### 3.2 Affected Components

- `MeshPrep.Core` - Fingerprint generation
- Filter Script JSON schema
- Script verification logic

### 3.3 Technical Approach

```
File bytes ──► SHA-256 ──► 64-char hex string
                              │
                              ▼
                    "a1b2c3d4e5f6..."
```

### 3.4 API/Interface

```csharp
namespace MeshPrep.Core.Fingerprint
{
    public interface IFingerprintService
    {
        string ComputeFingerprint(string filePath);
        Task<string> ComputeFingerprintAsync(string filePath, 
            IProgress<double>? progress = null, CancellationToken ct = default);
        bool VerifyFingerprint(string filePath, string expectedFingerprint);
    }

    public class ModelFingerprint
    {
        public string Hash { get; set; }        // SHA-256 hex string
        public long FileSize { get; set; }      // For quick pre-check
        public string OriginalFileName { get; set; }  // User reference only
    }
}
```

---

## 4. User Interface

### 4.1 UI Changes Required

**FilterScriptCreator:**
- Display fingerprint in model info panel
- Show fingerprint verification status when loading script

**ModelFixer:**
- Verify fingerprint on script load
- Clear error if fingerprint mismatch

### 4.2 User Interaction Flow

```
Creating Script:
Model loaded ──► Fingerprint computed ──► Stored in script JSON

Applying Script:
Script loaded ──► Extract fingerprint ──► Compare with loaded model
                                              │
                              ┌───────────────┴───────────────┐
                              ▼                               ▼
                          Match ✅                      Mismatch ❌
                              │                               │
                              ▼                               ▼
                       Apply script                   Show error message
```

---

## 5. Testing

### 5.1 Test Cases

| Test ID | Description | Input | Expected Output | Status |
|---------|-------------|-------|-----------------|--------|
| TC-001 | Compute fingerprint | Valid STL file | 64-char hex hash | ⬜ |
| TC-002 | Same file = same hash | Same file twice | Identical hashes | ⬜ |
| TC-003 | Different files = different hash | Two different files | Different hashes | ⬜ |
| TC-004 | Verify matching fingerprint | Correct hash | Returns true | ⬜ |
| TC-005 | Verify mismatched fingerprint | Wrong hash | Returns false | ⬜ |
| TC-006 | Large file performance | 100MB file | < 1 second | ⬜ |
| TC-007 | File not found | Invalid path | Clear exception | ⬜ |
| TC-008 | File locked | Locked file | Appropriate error | ⬜ |

### 5.2 Edge Cases

- Empty file (0 bytes)
- Very large file (500MB+)
- File on network share
- File being written by another process

---

## 6. Notes & Open Questions

### Open Questions
- [x] Use geometry hash or file hash? → **File hash (SHA-256) - simpler, sufficient**
- [x] Two-tier fingerprinting needed? → **No, single-tier is sufficient**

### Notes
- SHA-256 is cryptographically secure and fast
- File hash changes if even one byte changes (strict binding)
- Store file size for quick pre-verification before computing hash
- Original filename is informational only, not used for matching

---

## 7. Implementation Log

| Date | Update | Author |
|------|--------|--------|
| 2026-01-10 | Feature document created | |
