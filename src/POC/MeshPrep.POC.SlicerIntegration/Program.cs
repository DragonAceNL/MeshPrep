using System.Diagnostics;
using System.Text;
using System.Text.RegularExpressions;

namespace MeshPrep.POC.SlicerIntegration;

/// <summary>
/// POC-03: Slicer Integration Test Application
/// Tests CLI invocation for PrusaSlicer, Cura, and OrcaSlicer
/// 
/// SUCCESS CRITERIA:
/// 1. Invoke PrusaSlicer CLI and get exit code
/// 2. Invoke Cura CLI and get exit code  
/// 3. Invoke OrcaSlicer CLI and get exit code
/// 4. Detect valid vs invalid meshes from slicer response
/// 5. Parse error messages from slicer output
/// 6. Auto-detect installed slicers
/// 7. Validation completes within 30 seconds
/// </summary>
partial class Program
{
    private const int DefaultTimeoutSeconds = 60;
    private const int ValidationTimeThresholdSeconds = 30;

    static readonly List<TestResult> _results = [];

    static async Task Main(string[] args)
    {
        Console.WriteLine("╔══════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║         MeshPrep POC-03: Slicer Integration Test             ║");
        Console.WriteLine("╚══════════════════════════════════════════════════════════════╝");
        Console.WriteLine();

        // Step 1: Detect installed slicers
        var detector = new SlicerDetector();
        var installedSlicers = detector.DetectAll();

        Console.WriteLine("=== Slicer Detection ===");
        foreach (var slicer in Enum.GetValues<SlicerType>())
        {
            var path = installedSlicers.GetValueOrDefault(slicer);
            var status = path != null ? $"✓ Found: {path}" : "✗ Not found";
            Console.WriteLine($"  {slicer,-15}: {status}");
        }
        Console.WriteLine();

        if (installedSlicers.Count == 0)
        {
            Console.WriteLine("ERROR: No slicers found. Please install at least one of:");
            Console.WriteLine("  - PrusaSlicer: https://www.prusa3d.com/prusaslicer/");
            Console.WriteLine("  - OrcaSlicer:  https://github.com/SoftFever/OrcaSlicer");
            Console.WriteLine("  - Cura:        https://ultimaker.com/software/ultimaker-cura");
            Environment.ExitCode = 1;
            return;
        }

        // Step 2: Find test models
        var testDataDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "..", "samples", "test-models"));
        Console.WriteLine($"Test data directory: {testDataDir}");
        
        // Create test models for slicer validation
        await EnsureTestModelsExist(testDataDir);
        Console.WriteLine();

        // Step 3: Run validation tests for each detected slicer
        var validator = new SlicerValidator();

        foreach (var (slicerType, slicerPath) in installedSlicers)
        {
            Console.WriteLine($"═══════════════════════════════════════════════════════════════");
            Console.WriteLine($"  Testing: {slicerType}");
            Console.WriteLine($"═══════════════════════════════════════════════════════════════");
            Console.WriteLine();

            // Test 1: Valid watertight cube
            await RunTest(validator, slicerType, slicerPath, 
                Path.Combine(testDataDir, "test_cube.stl"), 
                "Valid watertight cube", 
                expectedSuccess: true);

            // Test 2: Cube with holes (if exists)
            var holesPath = Path.Combine(testDataDir, "cube_holes.stl");
            if (File.Exists(holesPath))
            {
                await RunTest(validator, slicerType, slicerPath, 
                    holesPath, 
                    "Cube with holes", 
                    expectedSuccess: false);
            }

            // Test 3: Non-manifold mesh (if exists)
            var nonManifoldPath = Path.Combine(testDataDir, "nonmanifold.stl");
            if (File.Exists(nonManifoldPath))
            {
                await RunTest(validator, slicerType, slicerPath, 
                    nonManifoldPath, 
                    "Non-manifold edges", 
                    expectedSuccess: false);
            }

            Console.WriteLine();
        }

        // Print summary
        PrintSummary(installedSlicers);

        Console.WriteLine();
        Console.WriteLine("POC-03 Slicer Integration test complete.");

        if (!Console.IsInputRedirected)
        {
            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }

        // Return non-zero exit code if critical tests failed
        var criticalTests = _results.Where(r => r.TestName.Contains("Valid watertight"));
        Environment.ExitCode = criticalTests.All(r => r.SlicerSuccess) ? 0 : 1;
    }

    static async Task RunTest(SlicerValidator validator, SlicerType slicerType, string slicerPath, 
        string modelPath, string testName, bool expectedSuccess)
    {
        var fileName = Path.GetFileName(modelPath);
        Console.WriteLine($"┌─────────────────────────────────────────────────────────────┐");
        Console.WriteLine($"│ Test: {TruncateString(testName, 53),-53}│");
        Console.WriteLine($"│ File: {TruncateString(fileName, 53),-53}│");
        Console.WriteLine($"└─────────────────────────────────────────────────────────────┘");

        if (!File.Exists(modelPath))
        {
            Console.WriteLine($"  ⚠ SKIPPED: File not found");
            Console.WriteLine();
            return;
        }

        var result = new TestResult
        {
            SlicerType = slicerType,
            SlicerPath = slicerPath,
            ModelPath = modelPath,
            TestName = testName,
            ExpectedSuccess = expectedSuccess
        };

        try
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(DefaultTimeoutSeconds));
            
            var sw = Stopwatch.StartNew();
            var validationResult = await validator.ValidateAsync(slicerType, slicerPath, modelPath, cts.Token);
            sw.Stop();

            result.ValidationTime = sw.Elapsed;
            result.SlicerSuccess = validationResult.Success;
            result.ExitCode = validationResult.ExitCode;
            result.Output = validationResult.Output;
            result.Error = validationResult.Error;
            result.ParsedErrors = validationResult.ParsedErrors;
            result.TimeoutOccurred = false;
            result.MeshInfo = validationResult.MeshInfo;

            // Validation checks
            result.CompletedInTime = result.ValidationTime.TotalSeconds <= ValidationTimeThresholdSeconds;
            result.MatchesExpected = result.SlicerSuccess == expectedSuccess;

            // Print results
            Console.WriteLine($"  Exit code:    {result.ExitCode}");
            Console.WriteLine($"  Slicer says:  {(result.SlicerSuccess ? "✓ Valid (manifold)" : "✗ Invalid (not manifold)")}");
            
            if (result.MeshInfo != null)
            {
                Console.WriteLine($"  Mesh Info:");
                Console.WriteLine($"    - Facets:     {result.MeshInfo.FacetCount}");
                Console.WriteLine($"    - Manifold:   {(result.MeshInfo.IsManifold ? "yes" : "no")}");
                if (result.MeshInfo.OpenEdges > 0)
                    Console.WriteLine($"    - Open edges: {result.MeshInfo.OpenEdges}");
                Console.WriteLine($"    - Parts:      {result.MeshInfo.PartCount}");
                Console.WriteLine($"    - Volume:     {result.MeshInfo.Volume:F2}");
            }
            
            Console.WriteLine($"  Expected:     {(expectedSuccess ? "Valid" : "Invalid")}");
            Console.WriteLine($"  Match:        {(result.MatchesExpected ? "✓ As expected" : "⚠ Unexpected result")}");
            Console.WriteLine($"  Time:         {result.ValidationTime.TotalSeconds:F2}s {(result.CompletedInTime ? "✓" : "✗ SLOW")}");

            if (result.ParsedErrors.Count > 0)
            {
                Console.WriteLine($"  Errors found: {result.ParsedErrors.Count}");
                foreach (var error in result.ParsedErrors.Take(3))
                {
                    Console.WriteLine($"    - {TruncateString(error, 55)}");
                }
                if (result.ParsedErrors.Count > 3)
                {
                    Console.WriteLine($"    ... and {result.ParsedErrors.Count - 3} more");
                }
            }
        }
        catch (OperationCanceledException)
        {
            result.TimeoutOccurred = true;
            Console.WriteLine($"  ✗ TIMEOUT: Slicer did not respond within {DefaultTimeoutSeconds}s");
        }
        catch (Exception ex)
        {
            result.Error = ex.Message;
            Console.WriteLine($"  ✗ ERROR: {ex.Message}");
        }

        _results.Add(result);
        Console.WriteLine();
    }

    static async Task EnsureTestModelsExist(string directory)
    {
        Directory.CreateDirectory(directory);

        // Create valid cube if it doesn't exist
        var cubePath = Path.Combine(directory, "test_cube.stl");
        if (!File.Exists(cubePath))
        {
            await CreateValidCubeStl(cubePath);
            Console.WriteLine($"Created: test_cube.stl (valid watertight cube)");
        }

        // Create cube with holes (intentionally broken mesh)
        var holesPath = Path.Combine(directory, "cube_holes.stl");
        if (!File.Exists(holesPath))
        {
            await CreateCubeWithHolesStl(holesPath);
            Console.WriteLine($"Created: cube_holes.stl (cube with missing faces)");
        }

        // Create non-manifold mesh
        var nonManifoldPath = Path.Combine(directory, "nonmanifold.stl");
        if (!File.Exists(nonManifoldPath))
        {
            await CreateNonManifoldStl(nonManifoldPath);
            Console.WriteLine($"Created: nonmanifold.stl (non-manifold edges)");
        }
    }

    static async Task CreateValidCubeStl(string path)
    {
        var content = @"solid cube
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
        await File.WriteAllTextAsync(path, content);
    }

    static async Task CreateCubeWithHolesStl(string path)
    {
        // Cube missing 2 faces (top faces removed = hole)
        var content = @"solid cube_with_holes
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
endsolid cube_with_holes";
        await File.WriteAllTextAsync(path, content);
    }

    static async Task CreateNonManifoldStl(string path)
    {
        // Two cubes sharing an edge (non-manifold)
        var content = @"solid nonmanifold
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
  facet normal 0 0 -1
    outer loop
      vertex 1 0 0
      vertex 2 0 0
      vertex 2 1 0
    endloop
  endfacet
  facet normal 0 0 -1
    outer loop
      vertex 1 0 0
      vertex 2 1 0
      vertex 1 1 0
    endloop
  endfacet
  facet normal 0 0 1
    outer loop
      vertex 1 0 1
      vertex 2 1 1
      vertex 2 0 1
    endloop
  endfacet
  facet normal 0 0 1
    outer loop
      vertex 1 0 1
      vertex 1 1 1
      vertex 2 1 1
    endloop
  endfacet
  facet normal 0 -1 0
    outer loop
      vertex 1 0 0
      vertex 2 0 1
      vertex 2 0 0
    endloop
  endfacet
  facet normal 0 -1 0
    outer loop
      vertex 1 0 0
      vertex 1 0 1
      vertex 2 0 1
    endloop
  endfacet
  facet normal 0 1 0
    outer loop
      vertex 1 1 0
      vertex 2 1 0
      vertex 2 1 1
    endloop
  endfacet
  facet normal 0 1 0
    outer loop
      vertex 1 1 0
      vertex 2 1 1
      vertex 1 1 1
    endloop
  endfacet
  facet normal 1 0 0
    outer loop
      vertex 2 0 0
      vertex 2 1 1
      vertex 2 1 0
    endloop
  endfacet
  facet normal 1 0 0
    outer loop
      vertex 2 0 0
      vertex 2 0 1
      vertex 2 1 1
    endloop
  endfacet
endsolid nonmanifold";
        await File.WriteAllTextAsync(path, content);
    }

    static void PrintSummary(Dictionary<SlicerType, string> installedSlicers)
    {
        var passed = _results.Count(r => r.MatchesExpected && r.CompletedInTime && !r.TimeoutOccurred);
        var failed = _results.Count - passed;
        var allPassed = failed == 0 && _results.Count > 0;

        Console.WriteLine("╔══════════════════════════════════════════════════════════════╗");
        Console.WriteLine($"║                     TEST SUMMARY: {(allPassed ? "PASS ✓" : "FAIL ✗"),-10}                 ║");
        Console.WriteLine("╠══════════════════════════════════════════════════════════════╣");
        Console.WriteLine($"║  Slicers found:  {installedSlicers.Count,-43}║");
        Console.WriteLine($"║  Total tests:    {_results.Count,-43}║");
        Console.WriteLine($"║  Passed:         {passed,-43}║");
        Console.WriteLine($"║  Failed:         {failed,-43}║");
        Console.WriteLine("╚══════════════════════════════════════════════════════════════╝");

        // Success criteria summary
        Console.WriteLine();
        Console.WriteLine("POC-03 Success Criteria:");

        // Check each slicer
        foreach (var slicerType in Enum.GetValues<SlicerType>())
        {
            var found = installedSlicers.ContainsKey(slicerType);
            var slicerResults = _results.Where(r => r.SlicerType == slicerType).ToList();
            
            if (found && slicerResults.Count > 0)
            {
                var validTest = slicerResults.FirstOrDefault(r => r.TestName.Contains("Valid watertight"));
                var invalidTest = slicerResults.FirstOrDefault(r => r.TestName.Contains("holes"));
                var canInvoke = validTest != null && !validTest.TimeoutOccurred;
                var canDetectValid = validTest?.SlicerSuccess == true;
                var canDetectInvalid = invalidTest?.SlicerSuccess == false;
                var withinTime = slicerResults.All(r => r.CompletedInTime || r.TimeoutOccurred);
                
                Console.WriteLine($"  {slicerType}:");
                Console.WriteLine($"    {(canInvoke ? "✓" : "✗")} Can invoke CLI and get exit code");
                Console.WriteLine($"    {(canDetectValid ? "✓" : "✗")} Detects valid mesh as valid");
                Console.WriteLine($"    {(canDetectInvalid ? "✓" : "✗")} Detects invalid mesh as invalid");
                Console.WriteLine($"    {(withinTime ? "✓" : "✗")} Completes within {ValidationTimeThresholdSeconds}s");
            }
            else if (!found)
            {
                Console.WriteLine($"  {slicerType}: Not installed (skipped)");
            }
        }

        Console.WriteLine();
        var autoDetect = installedSlicers.Count > 0;
        Console.WriteLine($"  {(autoDetect ? "✓" : "✗")} Auto-detect installed slicers");

        // Detailed results
        if (_results.Count > 0)
        {
            Console.WriteLine();
            Console.WriteLine("Detailed Results:");
            Console.WriteLine("═══════════════════════════════════════════════════════════════════════════════");
            Console.WriteLine($"{"Slicer",-15} {"Test",-25} {"Manifold",-10} {"Time",-8} {"Result",-10}");
            Console.WriteLine("───────────────────────────────────────────────────────────────────────────────");

            foreach (var result in _results)
            {
                var testName = TruncateString(result.TestName, 23);
                var manifold = result.TimeoutOccurred ? "T/O" : (result.MeshInfo?.IsManifold == true ? "yes" : "no");
                var time = result.TimeoutOccurred ? ">60s" : $"{result.ValidationTime.TotalSeconds:F2}s";
                var status = result.TimeoutOccurred ? "TIMEOUT" : 
                            (result.MatchesExpected && result.CompletedInTime ? "PASS" : "FAIL");

                Console.WriteLine($"{result.SlicerType,-15} {testName,-25} {manifold,-10} {time,-8} {status,-10}");
            }
            Console.WriteLine("═══════════════════════════════════════════════════════════════════════════════");
        }
    }

    static string TruncateString(string str, int maxLength)
    {
        if (str.Length <= maxLength) return str;
        return str[..(maxLength - 2)] + "..";
    }
}

// ============================================================================
// Supporting Classes
// ============================================================================

enum SlicerType
{
    PrusaSlicer,
    OrcaSlicer,
    // Cura - Deferred: CuraEngine requires extensive configuration and has no --info equivalent
    // Would need full printer definitions, extruder settings, and takes >30s for simple meshes
}

class SlicerDetector
{
    private static readonly Dictionary<SlicerType, string[]> SlicerPaths = new()
    {
        [SlicerType.PrusaSlicer] =
        [
            @"C:\Program Files\Prusa3D\PrusaSlicer\prusa-slicer-console.exe",
            @"C:\Program Files\Prusa3D\PrusaSlicer\prusa-slicer.exe",
            Environment.ExpandEnvironmentVariables(@"%LOCALAPPDATA%\Programs\PrusaSlicer\prusa-slicer-console.exe"),
            Environment.ExpandEnvironmentVariables(@"%LOCALAPPDATA%\Programs\PrusaSlicer\prusa-slicer.exe"),
        ],
        [SlicerType.OrcaSlicer] =
        [
            @"C:\Program Files\OrcaSlicer\orca-slicer.exe",
            @"C:\Program Files\OrcaSlicer\orca-slicer-console.exe",
            Environment.ExpandEnvironmentVariables(@"%LOCALAPPDATA%\Programs\OrcaSlicer\orca-slicer.exe"),
            Environment.ExpandEnvironmentVariables(@"%LOCALAPPDATA%\OrcaSlicer\orca-slicer.exe"),
        ],
        // Note: Cura/CuraEngine deferred - requires extensive configuration, no --info command, >30s slice time
    };

    public string? Find(SlicerType slicer)
    {
        if (SlicerPaths.TryGetValue(slicer, out var paths))
        {
            return paths.FirstOrDefault(File.Exists);
        }
        return null;
    }

    public Dictionary<SlicerType, string> DetectAll()
    {
        var result = new Dictionary<SlicerType, string>();
        
        foreach (var slicer in Enum.GetValues<SlicerType>())
        {
            var path = Find(slicer);
            if (path != null)
            {
                result[slicer] = path;
            }
        }

        return result;
    }
}

/// <summary>
/// Mesh information parsed from slicer --info output
/// </summary>
class MeshInfo
{
    public double SizeX { get; set; }
    public double SizeY { get; set; }
    public double SizeZ { get; set; }
    public int FacetCount { get; set; }
    public bool IsManifold { get; set; }
    public int OpenEdges { get; set; }
    public int PartCount { get; set; }
    public double Volume { get; set; }
}

class ValidationResult
{
    public bool Success { get; set; }
    public int ExitCode { get; set; }
    public string Output { get; set; } = "";
    public string Error { get; set; } = "";
    public List<string> ParsedErrors { get; set; } = [];
    public MeshInfo? MeshInfo { get; set; }
}

partial class SlicerValidator
{
    public async Task<ValidationResult> ValidateAsync(SlicerType slicerType, string slicerPath, 
        string modelPath, CancellationToken ct = default)
    {
        return slicerType switch
        {
            // PrusaSlicer and OrcaSlicer support --info for mesh analysis
            SlicerType.PrusaSlicer => await ValidateWithInfoAsync(slicerPath, modelPath, ct),
            SlicerType.OrcaSlicer => await ValidateWithInfoAsync(slicerPath, modelPath, ct),
            _ => throw new NotSupportedException($"Slicer {slicerType} not supported")
        };
    }

    /// <summary>
    /// Use --info command to get mesh statistics without slicing.
    /// This properly detects manifold issues, open edges, etc.
    /// </summary>
    private async Task<ValidationResult> ValidateWithInfoAsync(string slicerPath, string modelPath, CancellationToken ct)
    {
        // Use --info to analyze mesh without slicing or auto-repair
        var arguments = $"--info \"{modelPath}\"";
        
        var (exitCode, output, error) = await RunProcessAsync(slicerPath, arguments, ct);
        var combinedOutput = output + "\n" + error;
        
        // Parse the --info output for mesh statistics
        var meshInfo = ParseMeshInfo(combinedOutput);
        var parsedErrors = new List<string>();
        
        if (meshInfo != null && !meshInfo.IsManifold)
        {
            parsedErrors.Add($"Mesh is not manifold (open_edges={meshInfo.OpenEdges})");
        }

        return new ValidationResult
        {
            // Success = manifold mesh with no open edges
            Success = meshInfo?.IsManifold == true,
            ExitCode = exitCode,
            Output = output,
            Error = error,
            ParsedErrors = parsedErrors,
            MeshInfo = meshInfo
        };
    }

    /// <summary>
    /// Parse PrusaSlicer/OrcaSlicer --info output
    /// Example output:
    /// [test_cube.stl]
    /// size_x = 1.000000
    /// size_y = 1.000000
    /// size_z = 1.000000
    /// number_of_facets = 12
    /// manifold = yes
    /// open_edges = 0
    /// number_of_parts = 1
    /// volume = 1.000000
    /// </summary>
    private MeshInfo? ParseMeshInfo(string output)
    {
        var info = new MeshInfo();
        var foundData = false;
        
        foreach (var line in output.Split('\n'))
        {
            var trimmed = line.Trim();
            if (string.IsNullOrEmpty(trimmed)) continue;
            
            var parts = trimmed.Split('=', 2);
            if (parts.Length != 2) continue;
            
            var key = parts[0].Trim();
            var value = parts[1].Trim();
            foundData = true;
            
            switch (key)
            {
                case "size_x":
                    if (double.TryParse(value, out var sizeX)) info.SizeX = sizeX;
                    break;
                case "size_y":
                    if (double.TryParse(value, out var sizeY)) info.SizeY = sizeY;
                    break;
                case "size_z":
                    if (double.TryParse(value, out var sizeZ)) info.SizeZ = sizeZ;
                    break;
                case "number_of_facets":
                    if (int.TryParse(value, out var facets)) info.FacetCount = facets;
                    break;
                case "manifold":
                    info.IsManifold = value.Equals("yes", StringComparison.OrdinalIgnoreCase);
                    break;
                case "open_edges":
                    if (int.TryParse(value, out var edges)) info.OpenEdges = edges;
                    break;
                case "number_of_parts":
                    if (int.TryParse(value, out var partCount)) info.PartCount = partCount;
                    break;
                case "volume":
                    if (double.TryParse(value, out var volume)) info.Volume = volume;
                    break;
            }
        }
        
        return foundData ? info : null;
    }

    private async Task<(int exitCode, string output, string error)> RunProcessAsync(
        string executable, string arguments, CancellationToken ct)
    {
        var psi = new ProcessStartInfo
        {
            FileName = executable,
            Arguments = arguments,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        using var process = new Process { StartInfo = psi };
        var output = new StringBuilder();
        var error = new StringBuilder();

        process.OutputDataReceived += (s, e) => { if (e.Data != null) output.AppendLine(e.Data); };
        process.ErrorDataReceived += (s, e) => { if (e.Data != null) error.AppendLine(e.Data); };

        process.Start();
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();

        await process.WaitForExitAsync(ct);

        return (process.ExitCode, output.ToString(), error.ToString());
    }

    private List<string> ParseSlicerErrors(string slicerOutput)
    {
        var errors = new List<string>();
        var lines = slicerOutput.Split('\n', StringSplitOptions.RemoveEmptyEntries);

        foreach (var line in lines)
        {
            var trimmed = line.Trim();
            if (string.IsNullOrWhiteSpace(trimmed)) continue;

            if (ErrorPatterns().IsMatch(trimmed))
            {
                errors.Add(trimmed);
            }
        }

        return errors;
    }

    [GeneratedRegex(@"error|non-manifold|self-intersect|invalid|degenerate|fail|cannot|could not|unable", 
        RegexOptions.IgnoreCase | RegexOptions.Compiled)]
    private static partial Regex ErrorPatterns();

    [GeneratedRegex(@"(repaired|fixed|healed|auto.?repair|mesh.?issue|hole|gap|open.?edge|flipped.?normal)", 
        RegexOptions.IgnoreCase | RegexOptions.Compiled)]
    private static partial Regex MeshIssuePatterns();

    private bool HasMeshIssueWarnings(string output)
    {
        return MeshIssuePatterns().IsMatch(output);
    }
}

class TestResult
{
    public SlicerType SlicerType { get; set; }
    public string SlicerPath { get; set; } = "";
    public string ModelPath { get; set; } = "";
    public string TestName { get; set; } = "";
    public bool ExpectedSuccess { get; set; }
    
    public TimeSpan ValidationTime { get; set; }
    public bool SlicerSuccess { get; set; }
    public int ExitCode { get; set; }
    public string Output { get; set; } = "";
    public string Error { get; set; } = "";
    public List<string> ParsedErrors { get; set; } = [];
    public MeshInfo? MeshInfo { get; set; }
    
    public bool TimeoutOccurred { get; set; }
    public bool CompletedInTime { get; set; }
    public bool MatchesExpected { get; set; }
}
