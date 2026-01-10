namespace MeshPrep.Core.Fingerprint;

/// <summary>
/// Interface for fingerprint generation
/// </summary>
public interface IFingerprintService
{
    /// <summary>
    /// Compute SHA-256 fingerprint of a file
    /// </summary>
    string ComputeFingerprint(string filePath);
    
    /// <summary>
    /// Compute SHA-256 fingerprint of a file asynchronously
    /// </summary>
    Task<string> ComputeFingerprintAsync(
        string filePath, 
        IProgress<double>? progress = null, 
        CancellationToken ct = default);
    
    /// <summary>
    /// Verify a file matches an expected fingerprint
    /// </summary>
    bool VerifyFingerprint(string filePath, string expectedFingerprint);
}

/// <summary>
/// Model fingerprint with metadata
/// </summary>
public class ModelFingerprint
{
    /// <summary>
    /// SHA-256 hash as lowercase hex string (64 characters)
    /// </summary>
    public required string Hash { get; set; }
    
    /// <summary>
    /// File size in bytes (for quick pre-verification)
    /// </summary>
    public long FileSize { get; set; }
    
    /// <summary>
    /// Original filename (informational only)
    /// </summary>
    public string? OriginalFileName { get; set; }
}
