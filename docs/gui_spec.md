# GUI Functional Specification

## Overview

MeshPrep provides a step-by-step wizard GUI for non-technical Windows users to repair STL files. Built with PySide6.

---

## Design Principles

1. **Progressive disclosure** - Show relevant options at each step
2. **Clear feedback** - Always show current state and next actions
3. **Non-destructive** - Dry-run available; originals never overwritten
4. **Reproducibility** - Every run can be exported as shareable package
5. **Accessibility** - Large targets, keyboard navigation, high contrast

---

## Window Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Header: MeshPrep - [step name]                        [?]  │
├──────────┬──────────────────────────────────────────────────┤
│ Step List│  Main Content Area                               │
│ 1. Env   │  (changes per step)                              │
│ 2. Select│                                                  │
│ 3. Review│                                                  │
│ 4. Execute│                                                 │
│ 5. Results│                                                 │
├──────────┴──────────────────────────────────────────────────┤
│  Global Log (collapsible)                                   │
├─────────────────────────────────────────────────────────────┤
│  Status bar                    [Previous] [Next / Run]      │
└─────────────────────────────────────────────────────────────┘
```

---

## Steps Summary

| Step | Purpose | Key Elements |
|------|---------|--------------|
| **1. Environment** | Verify tools/dependencies | Status log, Auto-setup, Manual instructions |
| **2. Model Selection** | Select STL, choose filter source | File picker, Auto-detect vs existing filter |
| **3. Review Filter** | Review/edit filter script | Diagnostics, Profile, Actions list, Editor |
| **4. Execute** | Run repair | Progress bar, Terminal log, Steps table |
| **5. Results** | Show summary, export | Artifacts, Export run package, Copy command |

---

## Filter Script Editor

Three-panel layout:

| Panel | Width | Purpose |
|-------|-------|---------|
| **Filter Library** | ~25% | Browsable action catalog by category |
| **Action List** | ~45% | Current filter script's ordered actions |
| **Parameter Editor** | ~30% | Edit selected action's parameters |

### Editor Features

- Drag-and-drop action reordering
- Inline parameter editing with type-appropriate controls
- Real-time validation
- Action search and filter
- Undo/redo support
- Quick templates (Basic Cleanup, Aggressive Repair, etc.)

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Z/Y` | Undo/Redo |
| `Delete` | Remove action |
| `Ctrl+D` | Duplicate action |
| `Ctrl+↑/↓` | Move action |
| `Ctrl+F` | Focus search |
| `Escape` | Cancel/close |

---

## Theming

### Dark Theme (default)

| Element | Color |
|---------|-------|
| Background | `#0f1720` |
| Panel | `#111822` |
| Accent | `#4fe8c4` |
| Text | `#dff6fb` |
| Button | `#1b2b33` |
| Error | `#ff6b6b` |
| Warning | `#ffd93d` |
| Success | `#6bff6b` |

### Light Theme

| Element | Color |
|---------|-------|
| Background | `#f5f5f5` |
| Panel | `#ffffff` |
| Accent | `#00a67d` |
| Text | `#1a1a1a` |
| Button | `#e0e0e0` |

All text-on-background combinations meet WCAG AA contrast requirements.

---

## Settings

| Setting | Type | Default |
|---------|------|---------|
| Output Directory | Path | `./output/` |
| Blender Path | Path | Auto-detect |
| Use Blender | Choice | `on-failure` |
| Theme | Choice | `dark` |
| Log Level | Choice | `info` |

---

## CLI Mapping

| GUI Action | CLI Equivalent |
|------------|----------------|
| Select STL | `--input <file>` |
| Use preset | `--preset <name>` |
| Load filter | `--filter <file>` |
| Set output | `--output <dir>` |
| Export run | `--export-run <dir>` |

---

## Implementation Notes

- Built with PySide6 (Qt for Python)
- Stacked widget for step navigation
- Background threads with signals for long operations
- Filter Library from `config/filter_library.json`
- Action registry validates parameters

---

## See Also

- [Functional Spec](functional_spec.md) - Requirements and action catalog
- [Filter Actions](filter_actions.md) - Complete action reference
