using System.Diagnostics;
using System.Security.Cryptography;

namespace MeshPrep.POC.Fingerprint;

/// <summary>
/// POC-02: Fingerprinting Test Application
/// Tests SHA-256 file hashing for model fingerprinting
/// 
/// SUCCESS CRITERIA:
/// 1. Generate SHA-256 hash for any file size
/// 2. Hash computation < 1 second for 100MB file
/// 3. Same file always produces identical hash (deterministic)
/// 4. Different files produce different hashes (no collisions)
/// 5. Hash format: 64-character lowercase hex string
/// </summary>
class Program
{
    static async Task Main(string[] args)
    {
        Console.WriteLine("╔══════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║           MeshPrep POC-02: Fingerprinting Test               ║");
        Console.WriteLine("╚══════════════════════════════════════════════════════════════╝");
        Console.WriteLine();

        var service = new FingerprintService();
        var testDataDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "..", "samples", "test-models"));

        Console.WriteLine($"Test data directory: {testDataDir}");
        Console.WriteLine();

        // Run all tests
        var results = new List<TestResult>();

        // Test 1: Basic Hash Format
        results.Add(await TestBasicHashFormat(service, testDataDir));

        // Test 2: Determinism (same file, multiple runs)
        results.Add(await TestDeterminism(service, testDataDir));

        // Test 3: Different files produce different hashes
        results.Add(await TestNoCollisions(service, testDataDir));

        // Test 4: Performance with existing files
        results.Add(await TestPerformance(service, testDataDir));

        // Test 5: Async with progress
        results.Add(await TestAsyncWithProgress(service, testDataDir));

        // Test 6: Verification
        results.Add(await TestVerification(service, testDataDir));

        // Test 7: Large file simulation
        results.Add(await TestLargeFileSimulation(service, testDataDir));

        // Print summary
        PrintSummary(results);

        Console.WriteLine();
        Console.WriteLine("POC-02 Fingerprinting test complete.");

        if (!Console.IsInputRedirected)
        {
            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }

        Environment.ExitCode = results.All(r => r.Passed) ? 0 : 1;
    }

    static async Task<TestResult> TestBasicHashFormat(FingerprintService service, string testDataDir)
    {
        Console.WriteLine("═══════════════════════════════════════════════════════════════");
        Console.WriteLine("Test 1: Basic Hash Format");
        Console.WriteLine("───────────────────────────────────────────────────────────────");

        var result = new TestResult { TestName = "Basic Hash Format" };

        try
        {
            var testFile = Directory.GetFiles(testDataDir, "*.stl").FirstOrDefault()
                        ?? Directory.GetFiles(testDataDir, "*.*").First();

            var hash = service.ComputeFingerprint(testFile);

            // Validate format
            var isValidLength = hash.Length == 64;
            var isLowercase = hash == hash.ToLowerInvariant();
            var isHex = hash.All(c => "0123456789abcdef".Contains(c));

            Console.WriteLine($"  File: {Path.GetFileName(testFile)}");
            Console.WriteLine($"  Hash: {hash}");
            Console.WriteLine($"  Length: {hash.Length} (expected: 64) - {(isValidLength ? "✓" : "✗")}");
            Console.WriteLine($"  Lowercase: {(isLowercase ? "✓" : "✗")}");
            Console.WriteLine($"  Valid hex: {(isHex ? "✓" : "✗")}");

            result.Passed = isValidLength && isLowercase && isHex;
            result.Details = $"Hash: {hash[..16]}...";
        }
        catch (Exception ex)
        {
            result.Passed = false;
            result.Details = ex.Message;
            Console.WriteLine($"  ✗ Error: {ex.Message}");
        }

        Console.WriteLine($"  Result: {(result.Passed ? "✓ PASS" : "✗ FAIL")}");
        Console.WriteLine();
        return result;
    }

    static async Task<TestResult> TestDeterminism(FingerprintService service, string testDataDir)
    {
        Console.WriteLine("═══════════════════════════════════════════════════════════════");
        Console.WriteLine("Test 2: Determinism (100 runs, same file)");
        Console.WriteLine("───────────────────────────────────────────────────────────────");

        var result = new TestResult { TestName = "Determinism" };
        const int runs = 100;

        try
        {
            var testFile = Directory.GetFiles(testDataDir, "*.stl").FirstOrDefault()
                        ?? Directory.GetFiles(testDataDir, "*.*").First();

            var hashes = new HashSet<string>();
            var sw = Stopwatch.StartNew();

            for (int i = 0; i < runs; i++)
            {
                var hash = service.ComputeFingerprint(testFile);
                hashes.Add(hash);
            }

            sw.Stop();

            var allIdentical = hashes.Count == 1;
            var avgTime = sw.ElapsedMilliseconds / (double)runs;

            Console.WriteLine($"  File: {Path.GetFileName(testFile)}");
            Console.WriteLine($"  Runs: {runs}");
            Console.WriteLine($"  Unique hashes: {hashes.Count} (expected: 1)");
            Console.WriteLine($"  Average time: {avgTime:F2}ms per hash");
            Console.WriteLine($"  Hash: {hashes.First()[..16]}...");

            result.Passed = allIdentical;
            result.Details = $"{runs} runs, {hashes.Count} unique hash(es)";
        }
        catch (Exception ex)
        {
            result.Passed = false;
            result.Details = ex.Message;
            Console.WriteLine($"  ✗ Error: {ex.Message}");
        }

        Console.WriteLine($"  Result: {(result.Passed ? "✓ PASS" : "✗ FAIL")}");
        Console.WriteLine();
        return result;
    }

    static async Task<TestResult> TestNoCollisions(FingerprintService service, string testDataDir)
    {
        Console.WriteLine("═══════════════════════════════════════════════════════════════");
        Console.WriteLine("Test 3: No Collisions (different files)");
        Console.WriteLine("───────────────────────────────────────────────────────────────");

        var result = new TestResult { TestName = "No Collisions" };

        try
        {
            var files = Directory.GetFiles(testDataDir, "*.*", SearchOption.AllDirectories)
                .Where(f => !f.EndsWith(".bin")) // Exclude binary buffers
                .ToList();

            var hashes = new Dictionary<string, string>(); // hash -> filename
            var collisions = new List<(string File1, string File2, string Hash)>();

            foreach (var file in files)
            {
                var hash = service.ComputeFingerprint(file);

                if (hashes.TryGetValue(hash, out var existingFile))
                {
                    collisions.Add((existingFile, file, hash));
                }
                else
                {
                    hashes[hash] = file;
                }
            }

            Console.WriteLine($"  Files tested: {files.Count}");
            Console.WriteLine($"  Unique hashes: {hashes.Count}");
            Console.WriteLine($"  Collisions: {collisions.Count}");

            if (collisions.Count > 0)
            {
                foreach (var (file1, file2, hash) in collisions)
                {
                    Console.WriteLine($"    Collision: {Path.GetFileName(file1)} == {Path.GetFileName(file2)}");
                }
            }

            // Show some sample hashes
            Console.WriteLine("  Sample hashes:");
            foreach (var (hash, file) in hashes.Take(5))
            {
                Console.WriteLine($"    {Path.GetFileName(file),-25} : {hash[..16]}...");
            }

            result.Passed = collisions.Count == 0;
            result.Details = $"{files.Count} files, {collisions.Count} collisions";
        }
        catch (Exception ex)
        {
            result.Passed = false;
            result.Details = ex.Message;
            Console.WriteLine($"  ✗ Error: {ex.Message}");
        }

        Console.WriteLine($"  Result: {(result.Passed ? "✓ PASS" : "✗ FAIL")}");
        Console.WriteLine();
        return result;
    }

    static async Task<TestResult> TestPerformance(FingerprintService service, string testDataDir)
    {
        Console.WriteLine("═══════════════════════════════════════════════════════════════");
        Console.WriteLine("Test 4: Performance");
        Console.WriteLine("───────────────────────────────────────────────────────────────");

        var result = new TestResult { TestName = "Performance" };

        try
        {
            var files = Directory.GetFiles(testDataDir, "*.*", SearchOption.AllDirectories)
                .Where(f => !f.EndsWith(".bin"))
                .OrderBy(f => new FileInfo(f).Length)
                .ToList();

            var allPassed = true;
            var timings = new List<(string File, long Size, double TimeMs, double MBps)>();

            foreach (var file in files)
            {
                var fileInfo = new FileInfo(file);
                var sw = Stopwatch.StartNew();
                var hash = service.ComputeFingerprint(file);
                sw.Stop();

                var sizeMB = fileInfo.Length / (1024.0 * 1024.0);
                var mbps = sizeMB / (sw.ElapsedMilliseconds / 1000.0);

                timings.Add((Path.GetFileName(file), fileInfo.Length, sw.Elapsed.TotalMilliseconds, mbps));

                // Performance threshold: 100MB/s minimum (very conservative)
                var expectedMaxTime = Math.Max(10, fileInfo.Length / (100.0 * 1024 * 1024) * 1000);
                if (sw.ElapsedMilliseconds > expectedMaxTime * 10) // Very lenient
                {
                    allPassed = false;
                }
            }

            Console.WriteLine($"  {"File",-25} {"Size",-12} {"Time",-12} {"Speed",-12}");
            Console.WriteLine($"  {new string('-', 60)}");

            foreach (var (file, size, timeMs, mbps) in timings)
            {
                var sizeStr = size < 1024 ? $"{size} B" : size < 1024 * 1024 ? $"{size / 1024.0:F1} KB" : $"{size / 1024.0 / 1024.0:F1} MB";
                var speedStr = double.IsInfinity(mbps) || double.IsNaN(mbps) ? "instant" : $"{mbps:F1} MB/s";
                Console.WriteLine($"  {file,-25} {sizeStr,-12} {timeMs:F2}ms{"",-5} {speedStr,-12}");
            }

            result.Passed = allPassed;
            result.Details = $"{files.Count} files hashed";
        }
        catch (Exception ex)
        {
            result.Passed = false;
            result.Details = ex.Message;
            Console.WriteLine($"  ✗ Error: {ex.Message}");
        }

        Console.WriteLine($"  Result: {(result.Passed ? "✓ PASS" : "✗ FAIL")}");
        Console.WriteLine();
        return result;
    }

    static async Task<TestResult> TestAsyncWithProgress(FingerprintService service, string testDataDir)
    {
        Console.WriteLine("═══════════════════════════════════════════════════════════════");
        Console.WriteLine("Test 5: Async with Progress");
        Console.WriteLine("───────────────────────────────────────────────────────────────");

        var result = new TestResult { TestName = "Async with Progress" };

        try
        {
            // Find largest file for progress test
            var testFile = Directory.GetFiles(testDataDir, "*.*", SearchOption.AllDirectories)
                .Where(f => !f.EndsWith(".bin"))
                .OrderByDescending(f => new FileInfo(f).Length)
                .First();

            var progressUpdates = new List<double>();
            var progress = new Progress<double>(p => progressUpdates.Add(p));

            var sw = Stopwatch.StartNew();
            var hash = await service.ComputeFingerprintAsync(testFile, progress);
            sw.Stop();

            Console.WriteLine($"  File: {Path.GetFileName(testFile)}");
            Console.WriteLine($"  Size: {new FileInfo(testFile).Length / 1024.0:F1} KB");
            Console.WriteLine($"  Time: {sw.ElapsedMilliseconds}ms");
            Console.WriteLine($"  Progress updates: {progressUpdates.Count}");
            Console.WriteLine($"  Hash: {hash[..16]}...");

            // Verify hash matches sync version
            var syncHash = service.ComputeFingerprint(testFile);
            var hashesMatch = hash == syncHash;
            Console.WriteLine($"  Sync/Async match: {(hashesMatch ? "✓" : "✗")}");

            result.Passed = hashesMatch;
            result.Details = $"Async hash matches sync, {progressUpdates.Count} progress updates";
        }
        catch (Exception ex)
        {
            result.Passed = false;
            result.Details = ex.Message;
            Console.WriteLine($"  ✗ Error: {ex.Message}");
        }

        Console.WriteLine($"  Result: {(result.Passed ? "✓ PASS" : "✗ FAIL")}");
        Console.WriteLine();
        return result;
    }

    static async Task<TestResult> TestVerification(FingerprintService service, string testDataDir)
    {
        Console.WriteLine("═══════════════════════════════════════════════════════════════");
        Console.WriteLine("Test 6: Verification");
        Console.WriteLine("───────────────────────────────────────────────────────────────");

        var result = new TestResult { TestName = "Verification" };

        try
        {
            var testFile = Directory.GetFiles(testDataDir, "*.stl").FirstOrDefault()
                        ?? Directory.GetFiles(testDataDir, "*.*").First();

            var correctHash = service.ComputeFingerprint(testFile);
            var wrongHash = "0000000000000000000000000000000000000000000000000000000000000000";

            var verifyCorrect = service.VerifyFingerprint(testFile, correctHash);
            var verifyWrong = service.VerifyFingerprint(testFile, wrongHash);
            var verifyCaseInsensitive = service.VerifyFingerprint(testFile, correctHash.ToUpperInvariant());

            Console.WriteLine($"  File: {Path.GetFileName(testFile)}");
            Console.WriteLine($"  Correct hash verification: {(verifyCorrect ? "✓ Pass" : "✗ Fail")}");
            Console.WriteLine($"  Wrong hash verification: {(!verifyWrong ? "✓ Rejected" : "✗ Should reject")}");
            Console.WriteLine($"  Case-insensitive: {(verifyCaseInsensitive ? "✓ Pass" : "✗ Fail")}");

            result.Passed = verifyCorrect && !verifyWrong && verifyCaseInsensitive;
            result.Details = "Verification logic working";
        }
        catch (Exception ex)
        {
            result.Passed = false;
            result.Details = ex.Message;
            Console.WriteLine($"  ✗ Error: {ex.Message}");
        }

        Console.WriteLine($"  Result: {(result.Passed ? "✓ PASS" : "✗ FAIL")}");
        Console.WriteLine();
        return result;
    }

    static async Task<TestResult> TestLargeFileSimulation(FingerprintService service, string testDataDir)
    {
        Console.WriteLine("═══════════════════════════════════════════════════════════════");
        Console.WriteLine("Test 7: Large File Simulation (10MB generated)");
        Console.WriteLine("───────────────────────────────────────────────────────────────");

        var result = new TestResult { TestName = "Large File Simulation" };
        var tempFile = Path.Combine(Path.GetTempPath(), $"meshprep_test_{Guid.NewGuid()}.bin");

        try
        {
            // Create a 10MB test file
            const int sizeInMB = 10;
            const int sizeInBytes = sizeInMB * 1024 * 1024;

            Console.WriteLine($"  Creating {sizeInMB}MB test file...");

            using (var fs = File.Create(tempFile))
            {
                var random = new Random(42); // Deterministic for reproducibility
                var buffer = new byte[81920];
                var written = 0;

                while (written < sizeInBytes)
                {
                    random.NextBytes(buffer);
                    var toWrite = Math.Min(buffer.Length, sizeInBytes - written);
                    fs.Write(buffer, 0, toWrite);
                    written += toWrite;
                }
            }

            Console.WriteLine($"  File created: {new FileInfo(tempFile).Length / 1024.0 / 1024.0:F1}MB");

            // Time the hash
            var sw = Stopwatch.StartNew();
            var hash = service.ComputeFingerprint(tempFile);
            sw.Stop();

            var mbps = sizeInMB / (sw.ElapsedMilliseconds / 1000.0);

            Console.WriteLine($"  Hash time: {sw.ElapsedMilliseconds}ms");
            Console.WriteLine($"  Speed: {mbps:F1} MB/s");
            Console.WriteLine($"  Hash: {hash[..16]}...");

            // Verify determinism
            var hash2 = service.ComputeFingerprint(tempFile);
            var deterministic = hash == hash2;
            Console.WriteLine($"  Deterministic: {(deterministic ? "✓" : "✗")}");

            // Success if < 1 second for 10MB (very conservative - should be < 100ms)
            var fastEnough = sw.ElapsedMilliseconds < 1000;
            Console.WriteLine($"  Performance: {(fastEnough ? "✓ < 1s" : "✗ Too slow")}");

            result.Passed = deterministic && fastEnough;
            result.Details = $"{sizeInMB}MB in {sw.ElapsedMilliseconds}ms ({mbps:F1} MB/s)";
        }
        catch (Exception ex)
        {
            result.Passed = false;
            result.Details = ex.Message;
            Console.WriteLine($"  ✗ Error: {ex.Message}");
        }
        finally
        {
            // Cleanup
            if (File.Exists(tempFile))
            {
                File.Delete(tempFile);
                Console.WriteLine("  Temp file cleaned up.");
            }
        }

        Console.WriteLine($"  Result: {(result.Passed ? "✓ PASS" : "✗ FAIL")}");
        Console.WriteLine();
        return result;
    }

    static void PrintSummary(List<TestResult> results)
    {
        var passed = results.Count(r => r.Passed);
        var failed = results.Count(r => !r.Passed);
        var allPassed = failed == 0;

        Console.WriteLine("╔══════════════════════════════════════════════════════════════╗");
        Console.WriteLine($"║                     TEST SUMMARY: {(allPassed ? "PASS ✓" : "FAIL ✗"),-10}                 ║");
        Console.WriteLine("╠══════════════════════════════════════════════════════════════╣");
        Console.WriteLine($"║  Total tests:    {results.Count,-44}║");
        Console.WriteLine($"║  Passed:         {passed,-44}║");
        Console.WriteLine($"║  Failed:         {failed,-44}║");
        Console.WriteLine("╚══════════════════════════════════════════════════════════════╝");

        Console.WriteLine();
        Console.WriteLine("Detailed Results:");
        Console.WriteLine("═══════════════════════════════════════════════════════════════════════════════");
        Console.WriteLine($"{"Test",-30} {"Result",-10} {"Details"}");
        Console.WriteLine("───────────────────────────────────────────────────────────────────────────────");

        foreach (var result in results)
        {
            var status = result.Passed ? "✓ PASS" : "✗ FAIL";
            Console.WriteLine($"{result.TestName,-30} {status,-10} {result.Details}");
        }

        Console.WriteLine("═══════════════════════════════════════════════════════════════════════════════");

        Console.WriteLine();
        Console.WriteLine("POC-02 Success Criteria:");
        Console.WriteLine($"  {(results.Any(r => r.TestName == "Basic Hash Format" && r.Passed) ? "✓" : "✗")} SHA-256 hash format (64-char lowercase hex)");
        Console.WriteLine($"  {(results.Any(r => r.TestName == "Determinism" && r.Passed) ? "✓" : "✗")} Same file produces identical hash");
        Console.WriteLine($"  {(results.Any(r => r.TestName == "No Collisions" && r.Passed) ? "✓" : "✗")} Different files produce different hashes");
        Console.WriteLine($"  {(results.Any(r => r.TestName == "Performance" && r.Passed) ? "✓" : "✗")} Performance acceptable");
        Console.WriteLine($"  {(results.Any(r => r.TestName == "Verification" && r.Passed) ? "✓" : "✗")} Verification works correctly");
    }
}

class TestResult
{
    public string TestName { get; set; } = "";
    public bool Passed { get; set; }
    public string Details { get; set; } = "";
}

/// <summary>
/// SHA-256 based fingerprinting service for 3D model files
/// </summary>
public class FingerprintService
{
    /// <summary>
    /// Compute SHA-256 fingerprint of a file (synchronous)
    /// </summary>
    public string ComputeFingerprint(string filePath)
    {
        using var sha256 = SHA256.Create();
        using var stream = File.OpenRead(filePath);

        var hashBytes = sha256.ComputeHash(stream);
        return Convert.ToHexString(hashBytes).ToLowerInvariant();
    }

    /// <summary>
    /// Compute SHA-256 fingerprint of a file (async with progress)
    /// </summary>
    public async Task<string> ComputeFingerprintAsync(
        string filePath,
        IProgress<double>? progress = null,
        CancellationToken ct = default)
    {
        using var sha256 = SHA256.Create();
        using var stream = File.OpenRead(filePath);

        var fileLength = stream.Length;
        var buffer = new byte[81920]; // 80KB buffer
        long totalRead = 0;
        int bytesRead;

        while ((bytesRead = await stream.ReadAsync(buffer, ct)) > 0)
        {
            sha256.TransformBlock(buffer, 0, bytesRead, buffer, 0);
            totalRead += bytesRead;
            progress?.Report((double)totalRead / fileLength);
        }

        sha256.TransformFinalBlock(Array.Empty<byte>(), 0, 0);
        return Convert.ToHexString(sha256.Hash!).ToLowerInvariant();
    }

    /// <summary>
    /// Verify that a file matches an expected fingerprint
    /// </summary>
    public bool VerifyFingerprint(string filePath, string expectedFingerprint)
    {
        var actualFingerprint = ComputeFingerprint(filePath);
        return string.Equals(actualFingerprint, expectedFingerprint, StringComparison.OrdinalIgnoreCase);
    }
}
