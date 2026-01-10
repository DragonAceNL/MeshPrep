namespace MeshPrep.Core.Import;

using MeshPrep.Core.Models;

/// <summary>
/// Interface for mesh importers
/// </summary>
public interface IMeshImporter
{
    /// <summary>
    /// Check if this importer can handle the given file
    /// </summary>
    bool CanImport(string filePath);
    
    /// <summary>
    /// Get supported file extensions
    /// </summary>
    IReadOnlyList<string> SupportedExtensions { get; }
    
    /// <summary>
    /// Import a mesh from file
    /// </summary>
    ImportResult Import(string filePath, ImportOptions? options = null);
    
    /// <summary>
    /// Import a mesh from file asynchronously
    /// </summary>
    Task<ImportResult> ImportAsync(
        string filePath, 
        ImportOptions? options = null, 
        IProgress<double>? progress = null, 
        CancellationToken ct = default);
}

/// <summary>
/// Options for mesh import
/// </summary>
public class ImportOptions
{
    public Units TargetUnits { get; set; } = Units.Millimeters;
    public Axis UpAxis { get; set; } = Axis.Z;
    public bool PreservePartHierarchy { get; set; } = true;
}

/// <summary>
/// Result of a mesh import operation
/// </summary>
public class ImportResult
{
    public bool Success { get; set; }
    public MeshModel? Model { get; set; }
    public string? ErrorMessage { get; set; }
    public List<string> Warnings { get; set; } = [];
    public TimeSpan ImportTime { get; set; }
    public string? SourceFormat { get; set; }
}

/// <summary>
/// Unit systems for 3D models
/// </summary>
public enum Units
{
    Millimeters,
    Centimeters,
    Meters,
    Inches,
    Feet
}

/// <summary>
/// Axis orientation
/// </summary>
public enum Axis
{
    X,
    Y,
    Z
}
