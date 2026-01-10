# Feature F-011: 3D Preview

---

## Feature ID: F-011

## Feature Name
3D Model Preview

## Status
- [x] Not Started
- [ ] In Analysis
- [ ] In Development
- [ ] In Testing
- [ ] Complete

## Priority
**Medium** - Essential for GUI user experience

## Estimated Effort
**Large** (3-7 days)

## Related POC
**POC-04** - 3D Preview validation (Helix Toolkit performance)

---

## 1. Description

### 1.1 Overview
Display 3D models in an interactive viewport using Helix Toolkit. Users can rotate, zoom, and pan to inspect models. Support side-by-side comparison of original and repaired meshes.

### 1.2 User Story

As a **filter script creator**, I want **to see my model in 3D** so that **I can visually verify repairs and identify problem areas**.

### 1.3 Acceptance Criteria

- [ ] Display loaded mesh in 3D viewport
- [ ] Rotate, zoom, pan with mouse
- [ ] Handle meshes with 1M+ triangles smoothly (60 FPS)
- [ ] Show wireframe mode
- [ ] Highlight problem areas (issues from analysis)
- [ ] Side-by-side before/after comparison
- [ ] Reset camera to default view

---

## 2. Functional Details

### 2.1 Inputs

| Input | Type | Required | Description |
|-------|------|----------|-------------|
| MeshModel | MeshModel | Yes | Mesh to display |
| DisplayOptions | DisplayOptions | No | Rendering options |
| HighlightData | HighlightData | No | Issues to highlight |

### 2.2 Outputs

| Output | Type | Description |
|--------|------|-------------|
| Viewport | HelixViewport3D | Interactive 3D view |

### 2.3 View Modes

- **Solid** - Standard shaded view
- **Wireframe** - Triangle edges visible
- **Solid + Wireframe** - Both combined
- **X-Ray** - Transparent for internal inspection
- **Issue Highlight** - Problem areas colored red

### 2.4 Camera Controls

| Action | Mouse | Keyboard |
|--------|-------|----------|
| Rotate | Left drag | Arrow keys |
| Zoom | Scroll wheel | +/- keys |
| Pan | Right drag | Shift + arrows |
| Reset | Middle click | Home key |
| Fit to view | Double-click | F key |

---

## 3. Technical Details

### 3.1 Dependencies

- **Helix Toolkit** (NuGet) - WPF 3D rendering library

### 3.2 Affected Components

- `MeshPrep.FilterScriptCreator` - Main viewport
- `MeshPrep.ModelFixer` (GUI) - Preview viewport

### 3.3 Performance Requirements

| Metric | Target |
|--------|--------|
| Frame rate | ≥ 60 FPS |
| Max triangles | 5M+ |
| Load time | < 2s for 1M triangles |
| Memory | < 2GB for large models |

### 3.4 Technical Approach

```
MeshModel ──► Convert to MeshGeometry3D ──► Helix Toolkit Viewport
                                                    │
                                    ┌───────────────┼───────────────┐
                                    ▼               ▼               ▼
                                Lighting       Materials       Camera
```

### 3.5 API/Interface

```csharp
namespace MeshPrep.UI.Preview
{
    public interface IMeshPreview
    {
        void LoadMesh(MeshModel mesh);
        void ClearMesh();
        void SetViewMode(ViewMode mode);
        void HighlightIssues(MeshAnalysisResult analysis);
        void ResetCamera();
        void FitToView();
        void SetComparisonMode(MeshModel original, MeshModel repaired);
    }

    public enum ViewMode
    {
        Solid,
        Wireframe,
        SolidWireframe,
        XRay,
        IssueHighlight
    }

    public class DisplayOptions
    {
        public ViewMode Mode { get; set; } = ViewMode.Solid;
        public Color MeshColor { get; set; } = Colors.LightGray;
        public Color IssueColor { get; set; } = Colors.Red;
        public bool ShowAxes { get; set; } = true;
        public bool ShowGrid { get; set; } = true;
    }
}
```

---

## 4. User Interface

### 4.1 UI Components

**Main Viewport:**
- 3D view occupies majority of window
- Toolbar with view mode buttons
- Status bar showing triangle count, FPS

**Comparison Mode:**
- Split view: Original | Repaired
- Synchronized camera movement
- Toggle to overlay mode

**View Controls:**
- View mode dropdown/buttons
- Reset camera button
- Fit to view button
- Show/hide grid toggle

### 4.2 Layout

```
┌─────────────────────────────────────────────────────┐
│ [Solid] [Wire] [X-Ray] [Issues]  │  [Reset] [Fit]  │
├─────────────────────────────────────────────────────┤
│                                                     │
│                                                     │
│                  3D Viewport                        │
│                                                     │
│                                                     │
├─────────────────────────────────────────────────────┤
│ Triangles: 125,432 │ FPS: 60 │ Zoom: 100%          │
└─────────────────────────────────────────────────────┘
```

---

## 5. Testing

### 5.1 Test Cases

| Test ID | Description | Input | Expected Output | Status |
|---------|-------------|-------|-----------------|--------|
| TC-001 | Display simple mesh | Cube STL | Renders correctly | ⬜ |
| TC-002 | Display complex mesh | 1M triangles | 60 FPS | ⬜ |
| TC-003 | Rotate view | Mouse drag | Smooth rotation | ⬜ |
| TC-004 | Zoom in/out | Scroll wheel | Smooth zoom | ⬜ |
| TC-005 | Wireframe mode | Toggle button | Shows edges | ⬜ |
| TC-006 | Highlight issues | Mesh with holes | Red highlights | ⬜ |
| TC-007 | Comparison mode | Two meshes | Side-by-side view | ⬜ |
| TC-008 | Memory large mesh | 5M triangles | < 2GB RAM | ⬜ |

### 5.2 Edge Cases

- Empty mesh (no triangles)
- Single triangle
- Very small model (sub-millimeter)
- Very large model (meters)
- Non-manifold geometry display

---

## 6. Notes & Open Questions

### Open Questions
- [x] Which 3D library? → **Helix Toolkit (best WPF support)**
- [ ] Support VR preview? → **Not for v1.0**

### Notes
- Helix Toolkit chosen for excellent WPF integration
- Consider LOD (level of detail) for very large meshes
- Optimize mesh conversion for repeated updates
- Consider GPU memory limitations

---

## 7. Implementation Log

| Date | Update | Author |
|------|--------|--------|
| 2026-01-10 | Feature document created | |
