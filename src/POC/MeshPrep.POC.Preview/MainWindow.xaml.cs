using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Windows;
using System.Windows.Media;
using Assimp;
using HelixToolkit.SharpDX.Core;
using HelixToolkit.Wpf.SharpDX;
using Microsoft.Win32;
using SharpDX;
using Color = System.Windows.Media.Color;
using PerspectiveCamera = HelixToolkit.Wpf.SharpDX.PerspectiveCamera;

namespace MeshPrep.POC.Preview;

/// <summary>
/// POC-04: 3D Preview with Helix Toolkit (SharpDX)
/// 
/// SUCCESS CRITERIA:
/// 1. Render 1M+ triangle mesh at 60 FPS
/// 2. Smooth rotation, zoom, pan interaction
/// 3. Wireframe mode toggle
/// 4. Side-by-side comparison view
/// 5. Highlight specific faces/vertices
/// 6. Memory usage &lt; 2GB for large models
/// 7. Load mesh into viewport in &lt; 2 seconds
/// </summary>
public partial class MainWindow : Window, INotifyPropertyChanged
{
    private MeshGeometry3D? _meshGeometry;
    private LineGeometry3D? _wireframeGeometry;
    private LineGeometry3D? _gridGeometry;
    private MeshGeometry3D? _highlightGeometry;
    private PhongMaterial? _meshMaterial;
    private PhongMaterial? _leftMaterial;
    private PhongMaterial? _rightMaterial;
    private PhongMaterial? _highlightMaterial;
    private PerspectiveCamera? _camera;
    private PerspectiveCamera? _leftCamera;
    private PerspectiveCamera? _rightCamera;
    private EffectsManager? _effectsManager;
    
    private string? _currentFilePath;
    private bool _isWireframeMode;
    private readonly Stopwatch _fpsStopwatch = new();
    private int _frameCount;
    private readonly string _testModelsPath;

    public event PropertyChangedEventHandler? PropertyChanged;

    #region Bindable Properties
    
    public EffectsManager? EffectsManager
    {
        get => _effectsManager;
        set { _effectsManager = value; OnPropertyChanged(); }
    }
    
    public MeshGeometry3D? MeshGeometry
    {
        get => _meshGeometry;
        set { _meshGeometry = value; OnPropertyChanged(); }
    }
    
    public LineGeometry3D? WireframeGeometry
    {
        get => _wireframeGeometry;
        set { _wireframeGeometry = value; OnPropertyChanged(); }
    }
    
    public LineGeometry3D? GridGeometry
    {
        get => _gridGeometry;
        set { _gridGeometry = value; OnPropertyChanged(); }
    }
    
    public MeshGeometry3D? HighlightGeometry
    {
        get => _highlightGeometry;
        set { _highlightGeometry = value; OnPropertyChanged(); }
    }
    
    public PhongMaterial? MeshMaterial
    {
        get => _meshMaterial;
        set { _meshMaterial = value; OnPropertyChanged(); }
    }
    
    public PhongMaterial? LeftMaterial
    {
        get => _leftMaterial;
        set { _leftMaterial = value; OnPropertyChanged(); }
    }
    
    public PhongMaterial? RightMaterial
    {
        get => _rightMaterial;
        set { _rightMaterial = value; OnPropertyChanged(); }
    }
    
    public PhongMaterial? HighlightMaterial
    {
        get => _highlightMaterial;
        set { _highlightMaterial = value; OnPropertyChanged(); }
    }
    
    public PerspectiveCamera? Camera
    {
        get => _camera;
        set { _camera = value; OnPropertyChanged(); }
    }
    
    public PerspectiveCamera? LeftCamera
    {
        get => _leftCamera;
        set { _leftCamera = value; OnPropertyChanged(); }
    }
    
    public PerspectiveCamera? RightCamera
    {
        get => _rightCamera;
        set { _rightCamera = value; OnPropertyChanged(); }
    }
    
    #endregion

    public MainWindow()
    {
        InitializeComponent();
        DataContext = this;
        
        // Initialize EffectsManager - required for SharpDX rendering
        EffectsManager = new DefaultEffectsManager();
        
        // Set up test models path
        _testModelsPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "..", "samples", "test-models"));
        
        // Disable ViewCube until rendering is ready to prevent NullReferenceException
        MainViewport.ShowViewCube = false;
        LeftViewport.ShowViewCube = false;
        RightViewport.ShowViewCube = false;
        
        // Enable ViewCube after first render
        MainViewport.OnRendered += Viewport_OnRendered;
        LeftViewport.OnRendered += Viewport_OnRendered;
        RightViewport.OnRendered += Viewport_OnRendered;
        
        // Initialize cameras
        Camera = new PerspectiveCamera
        {
            Position = new System.Windows.Media.Media3D.Point3D(50, 50, 50),
            LookDirection = new System.Windows.Media.Media3D.Vector3D(-50, -50, -50),
            UpDirection = new System.Windows.Media.Media3D.Vector3D(0, 1, 0),
            FarPlaneDistance = 10000,
            NearPlaneDistance = 0.1
        };
        
        LeftCamera = new PerspectiveCamera
        {
            Position = new System.Windows.Media.Media3D.Point3D(50, 50, 50),
            LookDirection = new System.Windows.Media.Media3D.Vector3D(-50, -50, -50),
            UpDirection = new System.Windows.Media.Media3D.Vector3D(0, 1, 0),
            FarPlaneDistance = 10000,
            NearPlaneDistance = 0.1
        };
        
        RightCamera = new PerspectiveCamera
        {
            Position = new System.Windows.Media.Media3D.Point3D(50, 50, 50),
            LookDirection = new System.Windows.Media.Media3D.Vector3D(-50, -50, -50),
            UpDirection = new System.Windows.Media.Media3D.Vector3D(0, 1, 0),
            FarPlaneDistance = 10000,
            NearPlaneDistance = 0.1
        };
        
        // Initialize materials
        MeshMaterial = new PhongMaterial
        {
            DiffuseColor = new Color4(0.27f, 0.51f, 0.71f, 1.0f), // Steel Blue
            SpecularColor = new Color4(1f, 1f, 1f, 1f),
            SpecularShininess = 100
        };
        
        LeftMaterial = new PhongMaterial
        {
            DiffuseColor = new Color4(0.27f, 0.51f, 0.71f, 1.0f), // Steel Blue
            SpecularColor = new Color4(1f, 1f, 1f, 1f),
            SpecularShininess = 100
        };
        
        RightMaterial = new PhongMaterial
        {
            DiffuseColor = new Color4(0.2f, 0.8f, 0.2f, 1.0f), // Lime Green
            SpecularColor = new Color4(1f, 1f, 1f, 1f),
            SpecularShininess = 100
        };
        
        HighlightMaterial = new PhongMaterial
        {
            DiffuseColor = new Color4(1f, 0f, 0f, 0.8f), // Red
            SpecularColor = new Color4(1f, 1f, 1f, 1f),
            SpecularShininess = 50
        };
        
        // Create grid
        CreateGrid();
        
        // Set up FPS counter
        CompositionTarget.Rendering += CompositionTarget_Rendering;
        _fpsStopwatch.Start();
        
        // Populate test models dropdown
        PopulateTestModels();
        
        // Update status
        UpdateStatus("Ready - Load a model or select a test model");
    }

    private void CreateGrid()
    {
        var builder = new LineBuilder();
        var size = 100f;
        var step = 10f;
        
        for (float i = -size; i <= size; i += step)
        {
            builder.AddLine(new Vector3(i, 0, -size), new Vector3(i, 0, size));
            builder.AddLine(new Vector3(-size, 0, i), new Vector3(size, 0, i));
        }
        
        GridGeometry = builder.ToLineGeometry3D();
    }

    private void PopulateTestModels()
    {
        CboTestModels.Items.Clear();
        CboTestModels.Items.Add("-- Select Test Model --");
        
        // Add test models from samples directory
        if (Directory.Exists(_testModelsPath))
        {
            var extensions = new[] { "*.stl", "*.obj", "*.ply", "*.3mf", "*.dae", "*.gltf", "*.glb", "*.fbx", "*.off" };
            foreach (var ext in extensions)
            {
                foreach (var file in Directory.GetFiles(_testModelsPath, ext))
                {
                    CboTestModels.Items.Add(Path.GetFileName(file));
                }
            }
        }
        
        // Add Thingi10K models if available
        var thingi10KPath = @"C:\Users\Dragon Ace\Source\repos\Thingi10K\raw_meshes";
        if (Directory.Exists(thingi10KPath))
        {
            CboTestModels.Items.Add("--- Thingi10K Samples ---");
            var thingiFiles = Directory.GetFiles(thingi10KPath, "*.stl").Take(20);
            foreach (var file in thingiFiles)
            {
                CboTestModels.Items.Add($"[Thingi10K] {Path.GetFileName(file)}");
            }
        }
        
        CboTestModels.SelectedIndex = 0;
    }

    private void CompositionTarget_Rendering(object? sender, EventArgs e)
    {
        _frameCount++;
        var elapsed = _fpsStopwatch.Elapsed.TotalSeconds;
        
        if (elapsed >= 1.0)
        {
            var fps = _frameCount / elapsed;
            TxtFps.Text = $"FPS: {fps:F1}";
            _frameCount = 0;
            _fpsStopwatch.Restart();
        }
        
        // Update memory usage periodically
        if (_frameCount % 60 == 0)
        {
            var memoryMb = GC.GetTotalMemory(false) / (1024.0 * 1024.0);
            TxtMemory.Text = $"Memory: {memoryMb:F1} MB";
        }
    }

    private void BtnOpen_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFileDialog
        {
            Title = "Open 3D Model",
            Filter = "All 3D Models|*.stl;*.obj;*.ply;*.3mf;*.dae;*.gltf;*.glb;*.fbx;*.off|" +
                     "STL Files|*.stl|OBJ Files|*.obj|PLY Files|*.ply|3MF Files|*.3mf|" +
                     "COLLADA|*.dae|glTF|*.gltf;*.glb|FBX|*.fbx|OFF|*.off|All Files|*.*",
            InitialDirectory = _testModelsPath
        };
        
        if (dialog.ShowDialog() == true)
        {
            LoadModel(dialog.FileName);
        }
    }

    private void CboTestModels_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
    {
        if (CboTestModels.SelectedItem is string selectedItem && 
            !selectedItem.StartsWith("--") && 
            !selectedItem.StartsWith("---"))
        {
            string filePath;
            
            if (selectedItem.StartsWith("[Thingi10K]"))
            {
                var fileName = selectedItem.Replace("[Thingi10K] ", "");
                filePath = Path.Combine(@"C:\Users\Dragon Ace\Source\repos\Thingi10K\raw_meshes", fileName);
            }
            else
            {
                filePath = Path.Combine(_testModelsPath, selectedItem);
            }
            
            if (File.Exists(filePath))
            {
                LoadModel(filePath);
            }
        }
    }

    private void LoadModel(string filePath)
    {
        try
        {
            UpdateStatus($"Loading: {Path.GetFileName(filePath)}...");
            var sw = Stopwatch.StartNew();
            
            // Load mesh using Assimp
            var mesh = LoadMeshWithAssimp(filePath);
            if (mesh == null)
            {
                UpdateStatus($"Failed to load: {Path.GetFileName(filePath)}");
                return;
            }
            
            sw.Stop();
            
            MeshGeometry = mesh;
            _currentFilePath = filePath;
            
            // Create wireframe geometry
            CreateWireframe(mesh);
            
            // Clear highlight
            HighlightGeometry = null;
            
            // Calculate bounding box and adjust camera
            if (mesh.Positions != null && mesh.Positions.Count > 0)
            {
                var bounds = mesh.Bound;
                var center = bounds.Center;
                var size = bounds.Size;
                var maxDim = Math.Max(Math.Max(size.X, size.Y), size.Z);
                
                // Adjust camera position based on model size
                var distance = maxDim * 2.0;
                Camera = new PerspectiveCamera
                {
                    Position = new System.Windows.Media.Media3D.Point3D(
                        center.X + distance,
                        center.Y + distance,
                        center.Z + distance),
                    LookDirection = new System.Windows.Media.Media3D.Vector3D(
                        -distance, -distance, -distance),
                    UpDirection = new System.Windows.Media.Media3D.Vector3D(0, 1, 0),
                    FarPlaneDistance = maxDim * 100,
                    NearPlaneDistance = maxDim * 0.001
                };
            }
            
            // Zoom to fit
            MainViewport.ZoomExtents();
            
            // Update stats
            var triangles = mesh.Indices?.Count / 3 ?? 0;
            var vertices = mesh.Positions?.Count ?? 0;
            
            TxtTriangles.Text = $"Triangles: {triangles:N0}";
            TxtVertices.Text = $"Vertices: {vertices:N0}";
            TxtLoadTime.Text = $"Load: {sw.ElapsedMilliseconds}ms";
            
            UpdateStatus($"Loaded: {Path.GetFileName(filePath)} ({triangles:N0} triangles)");
        }
        catch (Exception ex)
        {
            UpdateStatus($"Error: {ex.Message}");
            MessageBox.Show($"Failed to load model:\n{ex.Message}", "Load Error", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    private MeshGeometry3D? LoadMeshWithAssimp(string filePath)
    {
        using var importer = new AssimpContext();
        
        var scene = importer.ImportFile(filePath,
            PostProcessSteps.Triangulate |
            PostProcessSteps.GenerateNormals |
            PostProcessSteps.JoinIdenticalVertices |
            PostProcessSteps.OptimizeMeshes);
        
        if (scene == null || !scene.HasMeshes)
            return null;
        
        var builder = new MeshBuilder();
        
        foreach (var assimpMesh in scene.Meshes)
        {
            var vertexOffset = builder.Positions.Count;
            
            // Add vertices
            foreach (var vertex in assimpMesh.Vertices)
            {
                builder.Positions.Add(new Vector3(vertex.X, vertex.Y, vertex.Z));
            }
            
            // Add normals if available
            if (assimpMesh.HasNormals)
            {
                foreach (var normal in assimpMesh.Normals)
                {
                    builder.Normals.Add(new Vector3(normal.X, normal.Y, normal.Z));
                }
            }
            
            // Add faces (triangles)
            foreach (var face in assimpMesh.Faces)
            {
                if (face.IndexCount == 3)
                {
                    builder.TriangleIndices.Add(vertexOffset + face.Indices[0]);
                    builder.TriangleIndices.Add(vertexOffset + face.Indices[1]);
                    builder.TriangleIndices.Add(vertexOffset + face.Indices[2]);
                }
            }
        }
        
        return builder.ToMesh();
    }

    private void CreateWireframe(MeshGeometry3D mesh)
    {
        if (mesh.Positions == null || mesh.Indices == null) return;
        
        var builder = new LineBuilder();
        
        for (int i = 0; i < mesh.Indices.Count; i += 3)
        {
            var i0 = mesh.Indices[i];
            var i1 = mesh.Indices[i + 1];
            var i2 = mesh.Indices[i + 2];
            
            var p0 = mesh.Positions[i0];
            var p1 = mesh.Positions[i1];
            var p2 = mesh.Positions[i2];
            
            builder.AddLine(p0, p1);
            builder.AddLine(p1, p2);
            builder.AddLine(p2, p0);
        }
        
        WireframeGeometry = builder.ToLineGeometry3D();
    }

    private void BtnWireframe_Click(object sender, RoutedEventArgs e)
    {
        _isWireframeMode = BtnWireframe.IsChecked == true;
        WireframeModel.Visibility = _isWireframeMode ? Visibility.Visible : Visibility.Collapsed;
    }

    private void BtnCompare_Click(object sender, RoutedEventArgs e)
    {
        var compareMode = BtnCompare.IsChecked == true;
        
        SingleViewContainer.Visibility = compareMode ? Visibility.Collapsed : Visibility.Visible;
        CompareViewContainer.Visibility = compareMode ? Visibility.Visible : Visibility.Collapsed;
        
        if (compareMode)
        {
            LeftViewport.ZoomExtents();
            RightViewport.ZoomExtents();
        }
    }

    private void BtnZoomExtents_Click(object sender, RoutedEventArgs e)
    {
        MainViewport.ZoomExtents();
        LeftViewport?.ZoomExtents();
        RightViewport?.ZoomExtents();
    }

    private void BtnResetCamera_Click(object sender, RoutedEventArgs e)
    {
        Camera = new PerspectiveCamera
        {
            Position = new System.Windows.Media.Media3D.Point3D(50, 50, 50),
            LookDirection = new System.Windows.Media.Media3D.Vector3D(-50, -50, -50),
            UpDirection = new System.Windows.Media.Media3D.Vector3D(0, 1, 0),
            FarPlaneDistance = 10000,
            NearPlaneDistance = 0.1
        };
        MainViewport.ZoomExtents();
    }

    private void BtnTestHighlight_Click(object sender, RoutedEventArgs e)
    {
        TestHighlighting();
    }

    /// <summary>
    /// Highlight specific triangles (for showing mesh issues)
    /// </summary>
    public void HighlightTriangles(IEnumerable<int> triangleIndices)
    {
        if (MeshGeometry?.Positions == null || MeshGeometry.Indices == null) return;
        
        var builder = new MeshBuilder();
        var offset = 0.1f; // Small offset to prevent z-fighting
        
        foreach (var triIndex in triangleIndices)
        {
            var baseIndex = triIndex * 3;
            if (baseIndex + 2 >= MeshGeometry.Indices.Count) continue;
            
            var i0 = MeshGeometry.Indices[baseIndex];
            var i1 = MeshGeometry.Indices[baseIndex + 1];
            var i2 = MeshGeometry.Indices[baseIndex + 2];
            
            var p0 = MeshGeometry.Positions[i0];
            var p1 = MeshGeometry.Positions[i1];
            var p2 = MeshGeometry.Positions[i2];
            
            // Calculate normal for offset
            var v1 = p1 - p0;
            var v2 = p2 - p0;
            var normal = Vector3.Cross(v1, v2);
            normal.Normalize();
            
            // Add offset vertices
            builder.AddTriangle(
                p0 + normal * offset,
                p1 + normal * offset,
                p2 + normal * offset);
        }
        
        HighlightGeometry = builder.ToMesh();
    }

    /// <summary>
    /// Test method to highlight random triangles (for POC testing)
    /// </summary>
    public void TestHighlighting()
    {
        if (MeshGeometry?.Indices == null) return;
        
        var triangleCount = MeshGeometry.Indices.Count / 3;
        var random = new Random();
        var highlightIndices = Enumerable.Range(0, triangleCount)
            .Where(_ => random.NextDouble() < 0.1) // Highlight 10% of triangles
            .ToList();
        
        HighlightTriangles(highlightIndices);
        UpdateStatus($"Highlighted {highlightIndices.Count:N0} triangles (10% of mesh)");
    }

    private void UpdateStatus(string message)
    {
        TxtStatus.Text = message;
    }

    protected void OnPropertyChanged([CallerMemberName] string? propertyName = null)
    {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }

    private void Viewport_OnRendered(object? sender, EventArgs e)
    {
        if (sender is Viewport3DX viewport)
        {
            viewport.ShowViewCube = true;
            viewport.OnRendered -= Viewport_OnRendered; // Only need to enable once
        }
    }
}
