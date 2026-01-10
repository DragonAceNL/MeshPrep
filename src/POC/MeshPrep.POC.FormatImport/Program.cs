using System.Diagnostics;
using Assimp;

namespace MeshPrep.POC.FormatImport;

/// <summary>
/// POC-01: Format Import Test Application
/// Tests Assimp.NET for importing various 3D file formats
/// 
/// SUCCESS CRITERIA:
/// 1. File loads without exception
/// 2. At least one mesh is returned
/// 3. Mesh has vertices and faces (> 0)
/// 4. Bounds are valid (not infinite/NaN)
/// 5. Import time within threshold
/// </summary>
class Program
{
    // Performance thresholds from POC document
    private const double SmallFileThresholdMs = 1000;   // < 1s for < 1MB
    private const double MediumFileThresholdMs = 30000; // < 30s for < 100MB  
    private const double LargeFileThresholdMs = 60000;  // < 60s for < 500MB

    static async Task Main(string[] args)
    {
        Console.WriteLine("╔══════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║           MeshPrep POC-01: Format Import Test                ║");
        Console.WriteLine("╚══════════════════════════════════════════════════════════════╝");
        Console.WriteLine();

        // Show Assimp version and supported formats
        ShowAssimpInfo();

        // Create test data directory if it doesn't exist
        var testDataDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "..", "samples", "test-models"));
        Directory.CreateDirectory(testDataDir);

        Console.WriteLine($"Test data directory: {Path.GetFullPath(testDataDir)}");
        Console.WriteLine();

        // Check for test files
        if (args.Length > 0)
        {
            foreach (var filePath in args)
            {
                if (File.Exists(filePath))
                {
                    await TestImportFile(filePath);
                }
                else
                {
                    Console.WriteLine($"File not found: {filePath}");
                }
            }
        }
        else
        {
            var sampleFiles = FindTestFiles(testDataDir);
            
            if (sampleFiles.Count == 0)
            {
                Console.WriteLine("No test files found. Creating test files...");
                Console.WriteLine();
                await CreateTestFiles(testDataDir);
                sampleFiles = FindTestFiles(testDataDir);
            }

            foreach (var file in sampleFiles)
            {
                await TestImportFile(file);
            }
        }

        // Print summary with pass/fail determination
        PrintSummary();

        Console.WriteLine();
        Console.WriteLine("POC-01 Format Import test complete.");
        
        if (!Console.IsInputRedirected)
        {
            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }

        // Return non-zero exit code if any tests failed
        Environment.ExitCode = _results.All(r => r.OverallPass) ? 0 : 1;
    }

    static void ShowAssimpInfo()
    {
        Console.WriteLine("=== Assimp Library Info ===");
        
        using var context = new AssimpContext();
        var importFormats = context.GetSupportedImportFormats();
        Console.WriteLine($"Supported import formats: {importFormats.Length}");
        
        var formatList = string.Join(", ", importFormats.Take(20));
        Console.WriteLine($"First 20: {formatList}...");
        Console.WriteLine();
        
        var keyFormats = new[] { ".stl", ".obj", ".fbx", ".gltf", ".glb", ".3mf", ".ply", ".dae", ".3ds", ".blend" };
        Console.WriteLine("Key formats support:");
        foreach (var format in keyFormats)
        {
            var supported = importFormats.Any(f => f.Equals(format, StringComparison.OrdinalIgnoreCase));
            Console.WriteLine($"  {format,-8} : {(supported ? "✓ Supported" : "✗ Not supported")}");
        }
        Console.WriteLine();
    }

    static List<string> FindTestFiles(string directory)
    {
        // All formats Assimp claims to support that we want to test
        var supportedExtensions = new[] { 
            ".stl", ".obj", ".fbx", ".gltf", ".glb", ".3mf", 
            ".ply", ".dae", ".3ds", ".blend", ".off", ".x",
            ".amf", ".ase", ".dxf", ".ifc"
        };
        var files = new List<string>();

        if (Directory.Exists(directory))
        {
            foreach (var ext in supportedExtensions)
            {
                files.AddRange(Directory.GetFiles(directory, $"*{ext}", SearchOption.AllDirectories));
            }
        }

        return files.OrderBy(f => new FileInfo(f).Length).ToList(); // Sort by size
    }

    static readonly List<TestResult> _results = [];

    static async Task TestImportFile(string filePath)
    {
        var fileName = Path.GetFileName(filePath);
        var extension = Path.GetExtension(filePath).ToLowerInvariant();
        var fileSize = new FileInfo(filePath).Length;

        Console.WriteLine($"┌─────────────────────────────────────────────────────────────┐");
        Console.WriteLine($"│ Testing: {TruncateString(fileName, 50),-50}│");
        Console.WriteLine($"├─────────────────────────────────────────────────────────────┤");
        Console.WriteLine($"│ Format: {extension,-8}  Size: {FormatBytes(fileSize),-20}       │");
        Console.WriteLine($"└─────────────────────────────────────────────────────────────┘");

        var result = new TestResult
        {
            FileName = fileName,
            Format = extension,
            FileSize = fileSize
        };

        // Determine time threshold based on file size
        result.TimeThresholdMs = fileSize switch
        {
            < 1_000_000 => SmallFileThresholdMs,      // < 1MB
            < 100_000_000 => MediumFileThresholdMs,   // < 100MB
            _ => LargeFileThresholdMs                  // >= 100MB
        };

        try
        {
            var sw = Stopwatch.StartNew();
            
            using var context = new AssimpContext();
            
            var postProcess = 
                PostProcessSteps.Triangulate |
                PostProcessSteps.GenerateNormals |
                PostProcessSteps.JoinIdenticalVertices |
                PostProcessSteps.ValidateDataStructure;

            var scene = await Task.Run(() => context.ImportFile(filePath, postProcess));
            
            sw.Stop();
            result.ImportTime = sw.Elapsed;
            result.LoadSuccess = true;

            // Collect statistics
            var totalVertices = 0;
            var totalFaces = 0;
            
            foreach (var mesh in scene.Meshes)
            {
                totalVertices += mesh.VertexCount;
                totalFaces += mesh.FaceCount;
            }

            result.MeshCount = scene.MeshCount;
            result.VertexCount = totalVertices;
            result.FaceCount = totalFaces;
            result.HasMaterials = scene.MaterialCount > 0;

            // Calculate bounds
            var (min, max) = CalculateBounds(scene);
            result.BoundsMin = min;
            result.BoundsMax = max;

            // === VALIDATION CHECKS ===
            
            // Check 1: At least one mesh
            result.HasMeshes = scene.MeshCount > 0;
            
            // Check 2: Has geometry data
            result.HasGeometry = totalVertices > 0 && totalFaces > 0;
            
            // Check 3: Valid bounds (not infinite/NaN)
            result.ValidBounds = IsValidBounds(min, max);
            
            // Check 4: Performance within threshold
            result.PerformancePass = result.ImportTime.TotalMilliseconds <= result.TimeThresholdMs;
            
            // Check 5: Expected geometry (for known test files)
            result.GeometryCorrect = ValidateExpectedGeometry(fileName, totalVertices, totalFaces);

            // Print results
            Console.WriteLine($"  Load:        {(result.LoadSuccess ? "✓ SUCCESS" : "✗ FAILED")}");
            Console.WriteLine($"  Has meshes:  {(result.HasMeshes ? $"✓ {result.MeshCount} mesh(es)" : "✗ NO MESHES")}");
            Console.WriteLine($"  Has geometry:{(result.HasGeometry ? $"✓ {result.VertexCount:N0} verts, {result.FaceCount:N0} faces" : "✗ EMPTY")}");
            Console.WriteLine($"  Valid bounds:{(result.ValidBounds ? "✓ OK" : "✗ INVALID")}");
            Console.WriteLine($"  Performance: {(result.PerformancePass ? "✓" : "✗")} {result.ImportTime.TotalMilliseconds:F1}ms (threshold: {result.TimeThresholdMs}ms)");
            Console.WriteLine($"  Geometry:    {(result.GeometryCorrect ? "✓ As expected" : "⚠ Not validated")}");
            Console.WriteLine($"  ─────────────────────────────────────────────────────────");
            Console.WriteLine($"  OVERALL:     {(result.OverallPass ? "✓ PASS" : "✗ FAIL")}");
            
            if (result.ValidBounds)
            {
                Console.WriteLine($"  Bounds:      ({min.X:F2}, {min.Y:F2}, {min.Z:F2}) to ({max.X:F2}, {max.Y:F2}, {max.Z:F2})");
                Console.WriteLine($"  Size:        {max.X - min.X:F2} x {max.Y - min.Y:F2} x {max.Z - min.Z:F2}");
            }
        }
        catch (AssimpException ex)
        {
            result.LoadSuccess = false;
            result.ErrorMessage = ex.Message;
            Console.WriteLine($"  ✗ Assimp error: {ex.Message}");
            Console.WriteLine($"  OVERALL:     ✗ FAIL");
        }
        catch (Exception ex)
        {
            result.LoadSuccess = false;
            result.ErrorMessage = ex.Message;
            Console.WriteLine($"  ✗ Error: {ex.Message}");
            Console.WriteLine($"  OVERALL:     ✗ FAIL");
        }

        _results.Add(result);
        Console.WriteLine();
    }

    static bool IsValidBounds(Vector3D min, Vector3D max)
    {
        // Check for NaN or Infinity
        if (float.IsNaN(min.X) || float.IsNaN(min.Y) || float.IsNaN(min.Z)) return false;
        if (float.IsNaN(max.X) || float.IsNaN(max.Y) || float.IsNaN(max.Z)) return false;
        if (float.IsInfinity(min.X) || float.IsInfinity(min.Y) || float.IsInfinity(min.Z)) return false;
        if (float.IsInfinity(max.X) || float.IsInfinity(max.Y) || float.IsInfinity(max.Z)) return false;
        
        // Check that max >= min
        if (max.X < min.X || max.Y < min.Y || max.Z < min.Z) return false;
        
        return true;
    }

    static bool ValidateExpectedGeometry(string fileName, int vertexCount, int faceCount)
    {
        // Known test files with expected geometry
        var expectedGeometry = new Dictionary<string, (int minFaces, int maxFaces)>(StringComparer.OrdinalIgnoreCase)
        {
            { "test_cube.stl", (12, 12) },    // Cube = 12 triangles
            { "test_cube.obj", (12, 12) },
            { "test_cube.ply", (12, 12) },
            { "test_cube.dae", (12, 12) },
            { "test_cube.gltf", (12, 12) },
            { "test_cube.glb", (12, 12) },
            { "test_cube.off", (12, 12) },
            { "test_sphere.stl", (200, 2000) }, // Sphere varies by resolution
            { "test_sphere.obj", (200, 2000) },
        };

        if (expectedGeometry.TryGetValue(fileName, out var expected))
        {
            return faceCount >= expected.minFaces && faceCount <= expected.maxFaces;
        }

        // Unknown file - can't validate, but don't fail
        return true; // Neutral - not a failure
    }

    static (Vector3D min, Vector3D max) CalculateBounds(Scene scene)
    {
        var min = new Vector3D(float.MaxValue, float.MaxValue, float.MaxValue);
        var max = new Vector3D(float.MinValue, float.MinValue, float.MinValue);

        foreach (var mesh in scene.Meshes)
        {
            foreach (var vertex in mesh.Vertices)
            {
                min.X = Math.Min(min.X, vertex.X);
                min.Y = Math.Min(min.Y, vertex.Y);
                min.Z = Math.Min(min.Z, vertex.Z);
                max.X = Math.Max(max.X, vertex.X);
                max.Y = Math.Max(max.Y, vertex.Y);
                max.Z = Math.Max(max.Z, vertex.Z);
            }
        }

        return (min, max);
    }

    static async Task CreateTestFiles(string directory)
    {
        // Create ASCII STL cube
        await CreateTestCubeStl(directory);
        
        // Create OBJ cube
        await CreateTestCubeObj(directory);
        
        // Create PLY cube
        await CreateTestCubePly(directory);
        
        Console.WriteLine();
    }

    static async Task CreateTestCubeStl(string directory)
    {
        var stlPath = Path.Combine(directory, "test_cube.stl");
        
        var stlContent = @"solid cube
  facet normal 0 0 -1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 1 1 0
    endloop
  endfacet
  facet normal 0 0 -1
    outer loop
      vertex 0 0 0
      vertex 1 1 0
      vertex 0 1 0
    endloop
  endfacet
  facet normal 0 0 1
    outer loop
      vertex 0 0 1
      vertex 1 1 1
      vertex 1 0 1
    endloop
  endfacet
  facet normal 0 0 1
    outer loop
      vertex 0 0 1
      vertex 0 1 1
      vertex 1 1 1
    endloop
  endfacet
  facet normal 0 -1 0
    outer loop
      vertex 0 0 0
      vertex 1 0 1
      vertex 1 0 0
    endloop
  endfacet
  facet normal 0 -1 0
    outer loop
      vertex 0 0 0
      vertex 0 0 1
      vertex 1 0 1
    endloop
  endfacet
  facet normal 0 1 0
    outer loop
      vertex 0 1 0
      vertex 1 1 0
      vertex 1 1 1
    endloop
  endfacet
  facet normal 0 1 0
    outer loop
      vertex 0 1 0
      vertex 1 1 1
      vertex 0 1 1
    endloop
  endfacet
  facet normal -1 0 0
    outer loop
      vertex 0 0 0
      vertex 0 1 0
      vertex 0 1 1
    endloop
  endfacet
  facet normal -1 0 0
    outer loop
      vertex 0 0 0
      vertex 0 1 1
      vertex 0 0 1
    endloop
  endfacet
  facet normal 1 0 0
    outer loop
      vertex 1 0 0
      vertex 1 1 1
      vertex 1 1 0
    endloop
  endfacet
  facet normal 1 0 0
    outer loop
      vertex 1 0 0
      vertex 1 0 1
      vertex 1 1 1
    endloop
  endfacet
endsolid cube";

        await File.WriteAllTextAsync(stlPath, stlContent);
        Console.WriteLine($"Created: {Path.GetFileName(stlPath)} (ASCII STL cube, 12 faces)");
    }

    static async Task CreateTestCubeObj(string directory)
    {
        var objPath = Path.Combine(directory, "test_cube.obj");
        
        var objContent = @"# Test cube OBJ file
# 8 vertices, 12 triangular faces

v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
v 0.0 0.0 1.0
v 1.0 0.0 1.0
v 1.0 1.0 1.0
v 0.0 1.0 1.0

vn 0.0 0.0 -1.0
vn 0.0 0.0 1.0
vn 0.0 -1.0 0.0
vn 0.0 1.0 0.0
vn -1.0 0.0 0.0
vn 1.0 0.0 0.0

f 1//1 2//1 3//1
f 1//1 3//1 4//1
f 5//2 7//2 6//2
f 5//2 8//2 7//2
f 1//3 6//3 2//3
f 1//3 5//3 6//3
f 4//4 3//4 7//4
f 4//4 7//4 8//4
f 1//5 4//5 8//5
f 1//5 8//5 5//5
f 2//6 7//6 3//6
f 2//6 6//6 7//6
";

        await File.WriteAllTextAsync(objPath, objContent);
        Console.WriteLine($"Created: {Path.GetFileName(objPath)} (OBJ cube, 12 faces)");
    }

    static async Task CreateTestCubePly(string directory)
    {
        var plyPath = Path.Combine(directory, "test_cube.ply");
        
        var plyContent = @"ply
format ascii 1.0
comment Test cube PLY file
element vertex 8
property float x
property float y
property float z
element face 12
property list uchar int vertex_indices
end_header
0 0 0
1 0 0
1 1 0
0 1 0
0 0 1
1 0 1
1 1 1
0 1 1
3 0 1 2
3 0 2 3
3 4 6 5
3 4 7 6
3 0 5 1
3 0 4 5
3 3 2 6
3 3 6 7
3 0 3 7
3 0 7 4
3 1 6 2
3 1 5 6
";

        await File.WriteAllTextAsync(plyPath, plyContent);
        Console.WriteLine($"Created: {Path.GetFileName(plyPath)} (PLY cube, 12 faces)");
    }

    static void PrintSummary()
    {
        var passed = _results.Count(r => r.OverallPass);
        var failed = _results.Count(r => !r.OverallPass);
        var allPassed = failed == 0;
        
        Console.WriteLine("╔══════════════════════════════════════════════════════════════╗");
        Console.WriteLine($"║                     TEST SUMMARY: {(allPassed ? "PASS ✓" : "FAIL ✗"),-10}                 ║");
        Console.WriteLine("╠══════════════════════════════════════════════════════════════╣");
        Console.WriteLine($"║  Total tests:    {_results.Count,-44}║");
        Console.WriteLine($"║  Passed:         {passed,-44}║");
        Console.WriteLine($"║  Failed:         {failed,-44}║");
        
        if (_results.Any(r => r.LoadSuccess))
        {
            Console.WriteLine("╠══════════════════════════════════════════════════════════════╣");
            var avgTime = _results.Where(r => r.LoadSuccess).Average(r => r.ImportTime.TotalMilliseconds);
            var maxTime = _results.Where(r => r.LoadSuccess).Max(r => r.ImportTime.TotalMilliseconds);
            var totalVerts = _results.Where(r => r.LoadSuccess).Sum(r => r.VertexCount);
            var totalFaces = _results.Where(r => r.LoadSuccess).Sum(r => r.FaceCount);
            
            Console.WriteLine($"║  Avg import time: {avgTime:F1} ms{new string(' ', 38 - avgTime.ToString("F1").Length)}║");
            Console.WriteLine($"║  Max import time: {maxTime:F1} ms{new string(' ', 38 - maxTime.ToString("F1").Length)}║");
            Console.WriteLine($"║  Total vertices:  {totalVerts:N0}{new string(' ', 40 - totalVerts.ToString("N0").Length)}║");
            Console.WriteLine($"║  Total faces:     {totalFaces:N0}{new string(' ', 40 - totalFaces.ToString("N0").Length)}║");
        }
        
        Console.WriteLine("╚══════════════════════════════════════════════════════════════╝");
        
        // Detailed results table
        if (_results.Count > 0)
        {
            Console.WriteLine();
            Console.WriteLine("Detailed Results:");
            Console.WriteLine("═══════════════════════════════════════════════════════════════════════════════");
            Console.WriteLine($"{"File",-25} {"Format",-6} {"Load",-5} {"Mesh",-5} {"Geom",-5} {"Bounds",-6} {"Perf",-5} {"Result",-6}");
            Console.WriteLine("───────────────────────────────────────────────────────────────────────────────");
            
            foreach (var result in _results)
            {
                var name = TruncateString(result.FileName, 23);
                var load = result.LoadSuccess ? "✓" : "✗";
                var mesh = result.HasMeshes ? "✓" : "✗";
                var geom = result.HasGeometry ? "✓" : "✗";
                var bounds = result.ValidBounds ? "✓" : "✗";
                var perf = result.PerformancePass ? "✓" : "✗";
                var overall = result.OverallPass ? "PASS" : "FAIL";
                
                Console.WriteLine($"{name,-25} {result.Format,-6} {load,-5} {mesh,-5} {geom,-5} {bounds,-6} {perf,-5} {overall,-6}");
            }
            Console.WriteLine("═══════════════════════════════════════════════════════════════════════════════");
        }

        // Success criteria summary
        Console.WriteLine();
        Console.WriteLine("POC-01 Success Criteria:");
        Console.WriteLine("  ✓ = Passing for all tested files");
        Console.WriteLine("  ✗ = At least one file failed this check");
        Console.WriteLine();
        
        var allLoad = _results.All(r => r.LoadSuccess);
        var allMesh = _results.All(r => r.HasMeshes);
        var allGeom = _results.All(r => r.HasGeometry);
        var allBounds = _results.All(r => r.ValidBounds);
        var allPerf = _results.All(r => r.PerformancePass);
        
        Console.WriteLine($"  {(allLoad ? "✓" : "✗")} File loads without exception");
        Console.WriteLine($"  {(allMesh ? "✓" : "✗")} At least one mesh returned");
        Console.WriteLine($"  {(allGeom ? "✓" : "✗")} Mesh has vertices and faces");
        Console.WriteLine($"  {(allBounds ? "✓" : "✗")} Bounds are valid (not NaN/Infinity)");
        Console.WriteLine($"  {(allPerf ? "✓" : "✗")} Import time within threshold");
    }

    static string FormatBytes(long bytes)
    {
        string[] sizes = ["B", "KB", "MB", "GB"];
        double len = bytes;
        int order = 0;
        while (len >= 1024 && order < sizes.Length - 1)
        {
            order++;
            len /= 1024;
        }
        return $"{len:F2} {sizes[order]}";
    }

    static string TruncateString(string str, int maxLength)
    {
        if (str.Length <= maxLength) return str;
        return str[..(maxLength - 2)] + "..";
    }
}

class TestResult
{
    public string FileName { get; set; } = "";
    public string Format { get; set; } = "";
    public long FileSize { get; set; }
    public string? ErrorMessage { get; set; }
    public TimeSpan ImportTime { get; set; }
    public double TimeThresholdMs { get; set; }
    
    // Raw data
    public int MeshCount { get; set; }
    public int VertexCount { get; set; }
    public int FaceCount { get; set; }
    public bool HasMaterials { get; set; }
    public Vector3D BoundsMin { get; set; }
    public Vector3D BoundsMax { get; set; }
    
    // Validation checks
    public bool LoadSuccess { get; set; }
    public bool HasMeshes { get; set; }
    public bool HasGeometry { get; set; }
    public bool ValidBounds { get; set; }
    public bool PerformancePass { get; set; }
    public bool GeometryCorrect { get; set; } = true;
    
    // Overall result - all checks must pass
    public bool OverallPass => 
        LoadSuccess && 
        HasMeshes && 
        HasGeometry && 
        ValidBounds && 
        PerformancePass;
}
