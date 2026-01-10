namespace MeshPrep.Core.Comparison;

using MeshPrep.Core.Models;

/// <summary>
/// Interface for geometry comparison using Hausdorff distance
/// </summary>
public interface IGeometryComparer
{
    /// <summary>
    /// Compare two meshes and calculate Hausdorff distances
    /// </summary>
    GeometryComparisonResult Compare(
        MeshModel original, 
        MeshModel repaired,
        ComparisonThresholds? thresholds = null);
    
    /// <summary>
    /// Compare two meshes asynchronously
    /// </summary>
    Task<GeometryComparisonResult> CompareAsync(
        MeshModel original, 
        MeshModel repaired,
        ComparisonThresholds? thresholds = null, 
        CancellationToken ct = default);
}

/// <summary>
/// Thresholds for pass/fail determination
/// </summary>
public class ComparisonThresholds
{
    /// <summary>
    /// Maximum allowed Hausdorff distance in mm (catches worst-case deviation)
    /// </summary>
    public double MaxHausdorffThreshold { get; set; } = 0.5;
    
    /// <summary>
    /// Maximum allowed mean Hausdorff distance in mm (ensures overall quality)
    /// </summary>
    public double MeanHausdorffThreshold { get; set; } = 0.05;
}

/// <summary>
/// Result of geometry comparison
/// </summary>
public class GeometryComparisonResult
{
    /// <summary>
    /// Maximum deviation between surfaces in mm
    /// </summary>
    public double MaxHausdorff { get; set; }
    
    /// <summary>
    /// Average deviation between surfaces in mm
    /// </summary>
    public double MeanHausdorff { get; set; }
    
    /// <summary>
    /// Root mean square deviation (informational)
    /// </summary>
    public double RmsDeviation { get; set; }
    
    /// <summary>
    /// Whether max Hausdorff is within threshold
    /// </summary>
    public bool MaxHausdorffPass { get; set; }
    
    /// <summary>
    /// Whether mean Hausdorff is within threshold
    /// </summary>
    public bool MeanHausdorffPass { get; set; }
    
    /// <summary>
    /// Whether both thresholds pass
    /// </summary>
    public bool OverallPass => MaxHausdorffPass && MeanHausdorffPass;
    
    /// <summary>
    /// Time taken to compute comparison
    /// </summary>
    public TimeSpan ComparisonTime { get; set; }
    
    /// <summary>
    /// Per-point deviation data for visualization (optional)
    /// </summary>
    public List<DeviationPoint>? DeviationMap { get; set; }
}

/// <summary>
/// A point with its deviation value for visualization
/// </summary>
public class DeviationPoint
{
    public Vertex? Location { get; set; }
    public double Deviation { get; set; }
}
