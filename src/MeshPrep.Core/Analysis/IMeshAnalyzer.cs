namespace MeshPrep.Core.Analysis;

using MeshPrep.Core.Models;

/// <summary>
/// Interface for mesh analysis
/// </summary>
public interface IMeshAnalyzer
{
    /// <summary>
    /// Analyze a mesh for issues
    /// </summary>
    MeshAnalysisResult Analyze(MeshModel mesh, AnalysisOptions? options = null);
    
    /// <summary>
    /// Analyze a mesh for issues asynchronously
    /// </summary>
    Task<MeshAnalysisResult> AnalyzeAsync(
        MeshModel mesh, 
        AnalysisOptions? options = null, 
        CancellationToken ct = default);
}

/// <summary>
/// Options for mesh analysis
/// </summary>
public class AnalysisOptions
{
    public bool CheckNonManifold { get; set; } = true;
    public bool CheckHoles { get; set; } = true;
    public bool CheckSelfIntersections { get; set; } = true;
    public bool CheckDegenerateFaces { get; set; } = true;
    public bool CheckNormals { get; set; } = true;
}

/// <summary>
/// Result of mesh analysis
/// </summary>
public class MeshAnalysisResult
{
    // Statistics
    public int VertexCount { get; set; }
    public int FaceCount { get; set; }
    public int EdgeCount { get; set; }
    public double Volume { get; set; }
    public double SurfaceArea { get; set; }
    public BoundingBox? Bounds { get; set; }

    // Issue counts
    public int NonManifoldEdgeCount { get; set; }
    public int NonManifoldVertexCount { get; set; }
    public int HoleCount { get; set; }
    public int SelfIntersectionCount { get; set; }
    public int DegenerateTriangleCount { get; set; }
    public int InvertedNormalCount { get; set; }

    // Issue details (for visualization)
    public List<EdgeIssue> NonManifoldEdges { get; set; } = [];
    public List<VertexIssue> NonManifoldVertices { get; set; } = [];
    public List<HoleInfo> Holes { get; set; } = [];

    // Summary properties
    public bool IsWatertight => HoleCount == 0;
    public bool IsManifold => NonManifoldEdgeCount == 0 && NonManifoldVertexCount == 0;
    public bool IsPrintable => IsWatertight && IsManifold && SelfIntersectionCount == 0;
    
    public int TotalIssueCount => 
        NonManifoldEdgeCount + NonManifoldVertexCount + HoleCount + 
        SelfIntersectionCount + DegenerateTriangleCount + InvertedNormalCount;
}

public class EdgeIssue
{
    public int VertexIndex1 { get; set; }
    public int VertexIndex2 { get; set; }
    public int AdjacentFaceCount { get; set; }
}

public class VertexIssue
{
    public int VertexIndex { get; set; }
    public string? Description { get; set; }
}

public class HoleInfo
{
    public List<int> BoundaryVertices { get; set; } = [];
    public double Perimeter { get; set; }
}
