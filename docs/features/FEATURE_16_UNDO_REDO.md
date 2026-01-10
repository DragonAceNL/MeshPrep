# Feature F-016: Undo/Redo System

---

## Feature ID: F-016

## Feature Name
Undo/Redo System

## Status
- [x] Not Started
- [ ] In Analysis
- [ ] In Development
- [ ] In Testing
- [ ] Complete

## Priority
**Medium** - Essential for FilterScriptCreator workflow

## Estimated Effort
**Medium** (1-3 days)

## Related POC
None - Standard implementation pattern

---

## 1. Description

### 1.1 Overview
Implement undo/redo functionality in FilterScriptCreator to allow users to experiment with repairs and revert changes. Track mesh state history and operation sequence.

### 1.2 User Story

As a **filter script creator**, I want **to undo and redo my repair operations** so that **I can experiment without fear of losing my work**.

### 1.3 Acceptance Criteria

- [ ] Undo last operation (Ctrl+Z)
- [ ] Redo undone operation (Ctrl+Y / Ctrl+Shift+Z)
- [ ] Multiple undo levels (configurable, default 20)
- [ ] Show undo history list
- [ ] Undo/redo updates 3D preview
- [ ] Memory-efficient state storage

---

## 2. Functional Details

### 2.1 Tracked Operations

| Operation | Undoable | Notes |
|-----------|----------|-------|
| Import model | No | Clears history |
| Apply repair | Yes | Core operation |
| Scale | Yes | Transform |
| Rotate | Yes | Transform |
| Merge parts | Yes | Part operation |
| Delete part | Yes | Part operation |
| Manual edit | Yes | Any mesh modification |

### 2.2 Inputs

| Input | Type | Required | Description |
|-------|------|----------|-------------|
| Operation | Operation | Yes | Operation to track |
| MeshBefore | MeshModel | Yes | State before operation |
| MeshAfter | MeshModel | Yes | State after operation |

### 2.3 Outputs

| Output | Type | Description |
|--------|------|-------------|
| CanUndo | bool | Whether undo is available |
| CanRedo | bool | Whether redo is available |
| History | List<HistoryEntry> | Operation history |

### 2.4 Business Rules

- New operation clears redo stack
- Import clears entire history
- History depth limited to prevent memory issues
- Large meshes may use compressed/differential storage

---

## 3. Technical Details

### 3.1 Dependencies

- None (standard pattern)

### 3.2 Affected Components

- `MeshPrep.FilterScriptCreator` - UI and commands
- State management system

### 3.3 Implementation Pattern

```
Command Pattern:

ICommand
├── Execute()
├── Undo()
└── GetDescription()

History Stack:
[Op1] [Op2] [Op3] ← Current
              ↑
         Undo here goes to [Op2]

After Undo:
[Op1] [Op2] [Op3]
         ↑
      Current (Op3 in redo stack)
```

### 3.4 Memory Optimization

For large meshes, consider:
1. **Differential storage** - Store only changed vertices/faces
2. **Compressed snapshots** - Compress mesh data
3. **Disk-based history** - Swap old states to temp files

### 3.5 API/Interface

```csharp
namespace MeshPrep.Core.History
{
    public interface IUndoRedoService
    {
        bool CanUndo { get; }
        bool CanRedo { get; }
        int UndoCount { get; }
        int RedoCount { get; }
        
        void Execute(IUndoableOperation operation);
        void Undo();
        void Redo();
        void Clear();
        
        IReadOnlyList<HistoryEntry> GetHistory();
        
        event EventHandler HistoryChanged;
    }

    public interface IUndoableOperation
    {
        string Description { get; }
        void Execute();
        void Undo();
    }

    public class HistoryEntry
    {
        public int Index { get; set; }
        public string Description { get; set; }
        public DateTime Timestamp { get; set; }
        public bool IsCurrent { get; set; }
    }

    public class UndoRedoSettings
    {
        public int MaxHistoryDepth { get; set; } = 20;
        public bool UseCompression { get; set; } = true;
        public long MaxMemoryMB { get; set; } = 500;
    }
}
```

---

## 4. User Interface

### 4.1 UI Components

**Menu:**
- Edit → Undo (Ctrl+Z)
- Edit → Redo (Ctrl+Y)

**Toolbar:**
- Undo button with dropdown showing history
- Redo button with dropdown showing redo stack

**History Panel (Optional):**
```
┌─────────────────────────────────────────┐
│            Operation History            │
├─────────────────────────────────────────┤
│  ○ Import model.stl                     │
│  ○ Fill Holes                           │
│  ○ Remove Self-Intersections            │
│  ● Fix Non-Manifold Edges  ← Current    │
│  ○ Scale 50%                            │
│  ○ Smooth Surface             (undone)  │
├─────────────────────────────────────────┤
│  Click to jump to any state             │
└─────────────────────────────────────────┘
```

### 4.2 Keyboard Shortcuts

| Action | Windows | Mac |
|--------|---------|-----|
| Undo | Ctrl+Z | Cmd+Z |
| Redo | Ctrl+Y | Cmd+Shift+Z |
| Undo multiple | Ctrl+Alt+Z | Cmd+Option+Z |

### 4.3 User Interaction Flow

```
Perform Operation ──► Add to History ──► Update UI
                                              │
                                              ▼
                                     Undo available
                                              │
                     ┌────────────────────────┘
                     ▼
              User clicks Undo
                     │
                     ▼
            Restore Previous State ──► Move to Redo Stack
                     │
                     ▼
               Update 3D View
```

---

## 5. Testing

### 5.1 Test Cases

| Test ID | Description | Input | Expected Output | Status |
|---------|-------------|-------|-----------------|--------|
| TC-001 | Basic undo | Apply op, undo | Original state | ⬜ |
| TC-002 | Basic redo | Undo, redo | Modified state | ⬜ |
| TC-003 | Multiple undo | 5 ops, undo 3 | State after op 2 | ⬜ |
| TC-004 | Redo cleared | Undo, new op | Redo unavailable | ⬜ |
| TC-005 | History limit | 25 ops (limit 20) | Oldest removed | ⬜ |
| TC-006 | Memory limit | Large mesh ops | Old states freed | ⬜ |
| TC-007 | Jump to state | Click history item | Jump to state | ⬜ |

### 5.2 Edge Cases

- Undo at beginning of history
- Redo at end of history
- Very large mesh (memory pressure)
- Rapid undo/redo clicks

---

## 6. Notes & Open Questions

### Open Questions
- [x] Include in ModelFixer? → **No, FilterScriptCreator only**
- [ ] Persist history across sessions? → **Not for v1.0**

### Notes
- Command pattern allows clean undo implementation
- Memory management critical for large meshes
- Consider lazy snapshot loading for deep history
- History could optionally persist to temp files

---

## 7. Implementation Log

| Date | Update | Author |
|------|--------|--------|
| 2026-01-10 | Feature document created | |
