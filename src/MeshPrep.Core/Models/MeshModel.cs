namespace MeshPrep.Core.Models;

/// <summary>
/// Represents a 3D mesh model with vertices, faces, and metadata
/// </summary>
public class MeshModel
{
    public string? SourcePath { get; set; }
    public string? Name { get; set; }
    
    public List<Vertex> Vertices { get; set; } = [];
    public List<Face> Faces { get; set; } = [];
    public List<MeshPart> Parts { get; set; } = [];
    
    public BoundingBox Bounds { get; set; } = new();
    
    public int VertexCount => Vertices.Count;
    public int FaceCount => Faces.Count;
    public int PartCount => Parts.Count;
    
    public MeshModel Clone()
    {
        return new MeshModel
        {
            SourcePath = SourcePath,
            Name = Name,
            Vertices = Vertices.Select(v => v.Clone()).ToList(),
            Faces = Faces.Select(f => f.Clone()).ToList(),
            Parts = Parts.Select(p => p.Clone()).ToList(),
            Bounds = Bounds.Clone()
        };
    }
}

/// <summary>
/// Represents a 3D vertex
/// </summary>
public class Vertex
{
    public double X { get; set; }
    public double Y { get; set; }
    public double Z { get; set; }
    
    public Vertex() { }
    
    public Vertex(double x, double y, double z)
    {
        X = x;
        Y = y;
        Z = z;
    }
    
    public Vertex Clone() => new(X, Y, Z);
}

/// <summary>
/// Represents a triangular face
/// </summary>
public class Face
{
    public int V1 { get; set; }
    public int V2 { get; set; }
    public int V3 { get; set; }
    
    public Vertex? Normal { get; set; }
    
    public Face() { }
    
    public Face(int v1, int v2, int v3)
    {
        V1 = v1;
        V2 = v2;
        V3 = v3;
    }
    
    public Face Clone() => new(V1, V2, V3) { Normal = Normal?.Clone() };
}

/// <summary>
/// Represents a part/component of a multi-part model
/// </summary>
public class MeshPart
{
    public int Id { get; set; }
    public string? Name { get; set; }
    public List<int> FaceIndices { get; set; } = [];
    public BoundingBox Bounds { get; set; } = new();
    
    public MeshPart Clone() => new()
    {
        Id = Id,
        Name = Name,
        FaceIndices = [.. FaceIndices],
        Bounds = Bounds.Clone()
    };
}

/// <summary>
/// Axis-aligned bounding box
/// </summary>
public class BoundingBox
{
    public Vertex Min { get; set; } = new();
    public Vertex Max { get; set; } = new();
    
    public double Width => Max.X - Min.X;
    public double Height => Max.Y - Min.Y;
    public double Depth => Max.Z - Min.Z;
    
    public BoundingBox Clone() => new()
    {
        Min = Min.Clone(),
        Max = Max.Clone()
    };
}
