# POC-04: 3D Preview

---

## POC ID: POC-04

## POC Name
3D Model Preview with Helix Toolkit

## Status
- [x] Not Started
- [ ] In Progress
- [ ] Completed - Success
- [ ] Completed - Failed
- [ ] Blocked

## Estimated Effort
**2-3 days**

## Related Features
- F-011: 3D Preview
- F-007: Geometry Fidelity Check (visualization)

---

## 1. Objective

### 1.1 What We're Proving
Validate that Helix Toolkit can render complex 3D meshes with acceptable performance, supporting rotation, zoom, pan, and before/after comparison views.

### 1.2 Success Criteria

- [ ] Render 1M+ triangle mesh at 60 FPS
- [ ] Smooth rotation, zoom, pan interaction
- [ ] Wireframe mode toggle
- [ ] Side-by-side comparison view
- [ ] Highlight specific faces/vertices (for issue visualization)
- [ ] Memory usage < 2GB for large models
- [ ] Load mesh into viewport in < 2 seconds

### 1.3 Failure Criteria

- FPS drops below 30 for 500K triangles
- Crashes or memory errors with large meshes
- Unacceptable visual artifacts
- Cannot highlight mesh regions

---

## 2. Technical Approach

### 2.1 Technologies to Evaluate

| Technology | Version | Purpose |
|------------|---------|---------|
| Helix Toolkit | Latest NuGet | WPF 3D rendering |
| HelixToolkit.Wpf | Latest | WPF integration |
| .NET 10 | | Runtime |
| WPF | | UI framework |

### 2.2 Test Scenarios

1. **Basic Render** - Display simple cube, verify orientation
2. **Complex Model** - Load 1M triangle spaceship model
3. **Interaction** - Rotate, zoom, pan with mouse
4. **Wireframe** - Toggle wireframe mode
5. **Comparison** - Side-by-side two meshes
6. **Highlighting** - Color specific triangles red
7. **Performance** - Measure FPS under load

### 2.3 Test Data

| Model | Triangles | Purpose |
|-------|-----------|---------|
| cube.stl | 12 | Basic test |
| sphere.stl | 10K | Medium complexity |
| spaceship.stl | 500K | Complex model |
| terrain.stl | 1M | Performance stress |
| terrain_huge.stl | 3M | Limit test |

---

## 3. Implementation

### 3.1 Setup Steps

1. Create new WPF .NET 10 project: `MeshPrep.POC.Preview`
2. Install NuGet packages:
   ```
   dotnet add package HelixToolkit.Wpf
   ```
3. Create main window with viewport
4. Implement mesh loading and display
5. Add camera controls and view modes

### 3.2 Code Location

`/poc/POC_04_Preview/`

### 3.3 Key Code Snippets

**XAML Viewport:**
```xml
<Window x:Class="MeshPrep.POC.Preview.MainWindow"
        xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        xmlns:helix="http://helix-toolkit.org/wpf"
        Title="3D Preview POC" Height="600" Width="800">
    <Grid>
        <helix:HelixViewport3D x:Name="Viewport" 
                                ShowCoordinateSystem="True"
                                ShowViewCube="True"
                                ZoomExtentsWhenLoaded="True">
            <helix:DefaultLights/>
            <ModelVisual3D x:Name="ModelContainer"/>
        </helix:HelixViewport3D>
    </Grid>
</Window>
```

**Load Mesh:**
```csharp
using HelixToolkit.Wpf;
using System.Windows.Media.Media3D;

public void LoadMesh(MeshGeometry3D mesh, Color color)
{
    var material = new DiffuseMaterial(new SolidColorBrush(color));
    var model = new GeometryModel3D(mesh, material);
    
    var visual = new ModelVisual3D { Content = model };
    ModelContainer.Children.Clear();
    ModelContainer.Children.Add(visual);
    
    Viewport.ZoomExtents();
}

public MeshGeometry3D ConvertToWpfMesh(MeshModel meshModel)
{
    var mesh = new MeshGeometry3D();
    
    foreach (var vertex in meshModel.Vertices)
    {
        mesh.Positions.Add(new Point3D(vertex.X, vertex.Y, vertex.Z));
    }
    
    foreach (var face in meshModel.Faces)
    {
        mesh.TriangleIndices.Add(face.V1);
        mesh.TriangleIndices.Add(face.V2);
        mesh.TriangleIndices.Add(face.V3);
    }
    
    return mesh;
}
```

**Wireframe Mode:**
```csharp
public void SetWireframeMode(bool enabled)
{
    if (enabled)
    {
        var wireframe = new LinesVisual3D
        {
            Points = GetWireframePoints(currentMesh),
            Color = Colors.Black,
            Thickness = 1
        };
        ModelContainer.Children.Add(wireframe);
    }
}
```

**Side-by-Side Comparison:**
```xml
<Grid>
    <Grid.ColumnDefinitions>
        <ColumnDefinition Width="*"/>
        <ColumnDefinition Width="*"/>
    </Grid.ColumnDefinitions>
    
    <helix:HelixViewport3D x:Name="LeftViewport" Grid.Column="0">
        <!-- Original mesh -->
    </helix:HelixViewport3D>
    
    <helix:HelixViewport3D x:Name="RightViewport" Grid.Column="1">
        <!-- Repaired mesh -->
    </helix:HelixViewport3D>
</Grid>
```

**Performance Monitoring:**
```csharp
private void CompositionTarget_Rendering(object sender, EventArgs e)
{
    frameCount++;
    var elapsed = stopwatch.Elapsed.TotalSeconds;
    
    if (elapsed >= 1.0)
    {
        var fps = frameCount / elapsed;
        FpsLabel.Content = $"FPS: {fps:F1}";
        frameCount = 0;
        stopwatch.Restart();
    }
}
```

---

## 4. Results

### 4.1 Test Results

| Test | Triangles | Result | Notes |
|------|-----------|--------|-------|
| Basic render | 12 | ⬜ | |
| Medium model | 10K | ⬜ | |
| Complex model | 500K | ⬜ | |
| Large model | 1M | ⬜ | |
| Huge model | 3M | ⬜ | |
| Wireframe mode | 500K | ⬜ | |
| Comparison view | 2x 500K | ⬜ | |
| Highlighting | 500K | ⬜ | |

### 4.2 Performance Metrics

| Triangles | Target FPS | Actual FPS | Memory | Pass? |
|-----------|------------|------------|--------|-------|
| 10K | 60 | | | ⬜ |
| 100K | 60 | | | ⬜ |
| 500K | 60 | | | ⬜ |
| 1M | 60 | | | ⬜ |
| 3M | 30 | | | ⬜ |

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
