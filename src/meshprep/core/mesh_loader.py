# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Universal mesh loader with automatic format detection and conversion.

This module provides comprehensive mesh loading capabilities, supporting
a wide range of 3D file formats and automatically converting them to
trimesh's internal representation for processing by MeshPrep.

Supported formats include:
- Common 3D printing: STL, OBJ, PLY, 3MF, AMF, OFF
- Web/compressed: GLTF/GLB, CTM, Draco
- CAD: STEP, IGES, BREP (with OpenCASCADE)
- 3D modeling: FBX, DAE, 3DS, Blender
- Engineering: VTK, Gmsh, NASTRAN, Abaqus, etc.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
import logging
import tempfile
import subprocess
import shutil

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


# Format categories for documentation and UI
FORMAT_CATEGORIES = {
    "3d_printing": {
        "name": "3D Printing Formats",
        "extensions": [".stl", ".obj", ".ply", ".3mf", ".amf", ".off"],
    },
    "compressed": {
        "name": "Compressed/Optimized Formats",
        "extensions": [".ctm", ".drc"],
    },
    "cad": {
        "name": "CAD & Engineering Formats",
        "extensions": [".step", ".stp", ".iges", ".igs", ".brep", ".vtk", ".vtu", 
                       ".msh", ".nas", ".bdf", ".fem", ".inp", ".e", ".ex2", ".exo"],
    },
    "modeling": {
        "name": "3D Modeling Software Formats",
        "extensions": [".fbx", ".dae", ".3ds", ".blend", ".gltf", ".glb"],
    },
    "other": {
        "name": "Other Formats",
        "extensions": [".dxf", ".svg", ".cgns", ".xdmf", ".xmf", ".dat", ".tec",
                       ".vol", ".ele", ".node", ".ugrid", ".su2", ".mesh", ".meshb"],
    },
}

# All supported extensions (flattened)
SUPPORTED_EXTENSIONS = set()
for category in FORMAT_CATEGORIES.values():
    SUPPORTED_EXTENSIONS.update(category["extensions"])

# Extensions that require optional dependencies
OPTIONAL_DEPENDENCY_FORMATS = {
    ".ctm": "pymeshlab",
    ".step": "trimesh[easy] (OpenCASCADE)",
    ".stp": "trimesh[easy] (OpenCASCADE)",
    ".iges": "trimesh[easy] (OpenCASCADE)",
    ".igs": "trimesh[easy] (OpenCASCADE)",
    ".brep": "trimesh[easy] (OpenCASCADE)",
    ".drc": "DracoPy or trimesh[easy]",
    ".fbx": "Blender (external conversion)",
    ".blend": "Blender (external conversion)",
}

# Formats that require Blender for conversion
BLENDER_REQUIRED_FORMATS = {".blend", ".fbx"}

# Formats that PyMeshLab handles well (CTM, etc.)
PYMESHLAB_FORMATS = {".ctm"}


@dataclass
class LoadResult:
    """Result of loading a mesh file.
    
    Attributes:
        mesh: The loaded trimesh object
        original_format: Original file format (extension without dot)
        original_path: Path to the original file
        original_vertices: Vertex count before any processing
        original_faces: Face count before any processing
        conversion_method: How the file was loaded/converted
        conversion_notes: Additional notes about the conversion
        warnings: Any warnings generated during loading
    """
    mesh: trimesh.Trimesh
    original_format: str
    original_path: Path
    original_vertices: int = 0
    original_faces: int = 0
    conversion_method: str = "trimesh"
    conversion_notes: str = ""
    warnings: list = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "original_format": self.original_format,
            "original_path": str(self.original_path),
            "original_vertices": self.original_vertices,
            "original_faces": self.original_faces,
            "conversion_method": self.conversion_method,
            "conversion_notes": self.conversion_notes,
            "warnings": self.warnings,
        }


def get_supported_extensions() -> list[str]:
    """Get list of all supported file extensions."""
    return sorted(SUPPORTED_EXTENSIONS)


def get_format_category(extension: str) -> Optional[str]:
    """Get the category for a file extension."""
    ext = extension.lower() if extension.startswith(".") else f".{extension.lower()}"
    for category_id, category in FORMAT_CATEGORIES.items():
        if ext in category["extensions"]:
            return category_id
    return None


def is_format_supported(path: Union[str, Path]) -> bool:
    """Check if a file format is supported."""
    ext = Path(path).suffix.lower()
    return ext in SUPPORTED_EXTENSIONS


def get_required_dependency(extension: str) -> Optional[str]:
    """Get the optional dependency required for a format, if any."""
    ext = extension.lower() if extension.startswith(".") else f".{extension.lower()}"
    return OPTIONAL_DEPENDENCY_FORMATS.get(ext)


def _check_blender_available() -> Optional[str]:
    """Check if Blender is available and return its path."""
    # Check common locations
    blender_paths = [
        shutil.which("blender"),
        r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe",
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "/usr/bin/blender",
        "/snap/bin/blender",
    ]
    
    for path in blender_paths:
        if path and Path(path).exists():
            return path
    
    return None


def _check_pymeshlab_available() -> bool:
    """Check if PyMeshLab is available."""
    try:
        import pymeshlab  # noqa: F401
        return True
    except ImportError:
        return False


def _load_with_pymeshlab(path: Path) -> trimesh.Trimesh:
    """Load a mesh using PyMeshLab.
    
    PyMeshLab supports many formats including CTM (OpenCTM), which uses
    MG1/MG2 compression with LZMA. This is the recommended way to load
    CTM files as it's the reference implementation.
    """
    try:
        import pymeshlab
    except ImportError:
        raise ImportError(
            "CTM format requires pymeshlab package. "
            "Install with: pip install pymeshlab"
        )
    
    logger.info(f"Loading with PyMeshLab: {path}")
    
    # Create a MeshSet and load the file
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(path))
    
    # Get the current mesh
    mesh = ms.current_mesh()
    
    # Extract vertices and faces as numpy arrays
    vertices = mesh.vertex_matrix()
    faces = mesh.face_matrix()
    
    # Create trimesh object
    result = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    logger.info(f"Loaded via PyMeshLab: {len(result.vertices)} vertices, {len(result.faces)} faces")
    
    return result


def _load_with_blender(path: Path, blender_path: str) -> trimesh.Trimesh:
    """Load a file by converting it with Blender.
    
    Blender supports many formats including:
    - CTM (OpenCTM compressed meshes)
    - FBX (Autodesk)
    - BLEND (native Blender)
    - OBJ, PLY, STL, GLTF/GLB, 3DS, DAE, etc.
    """
    logger.info(f"Converting with Blender: {path}")
    
    # Create temp file for STL output
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        ext = path.suffix.lower()
        
        # Build the appropriate import command based on extension
        # Note: CTM import addon must be enabled in Blender preferences
        import_commands = {
            ".blend": f'bpy.ops.wm.open_mainfile(filepath=r"{path}")',
            ".fbx": f'bpy.ops.import_scene.fbx(filepath=r"{path}")',
            ".obj": f'bpy.ops.wm.obj_import(filepath=r"{path}")',
            ".gltf": f'bpy.ops.import_scene.gltf(filepath=r"{path}")',
            ".glb": f'bpy.ops.import_scene.gltf(filepath=r"{path}")',
            ".3ds": f'bpy.ops.import_scene.autodesk_3ds(filepath=r"{path}")',
            ".dae": f'bpy.ops.wm.collada_import(filepath=r"{path}")',
            ".stl": f'bpy.ops.import_mesh.stl(filepath=r"{path}")',
            ".ply": f'bpy.ops.wm.ply_import(filepath=r"{path}")',
            # CTM requires the CTM addon which may need to be enabled
            ".ctm": f'bpy.ops.import_mesh.ctm(filepath=r"{path}")',
        }
        
        import_cmd = import_commands.get(ext)
        if not import_cmd:
            raise ValueError(f"Unsupported format for Blender conversion: {ext}")
        
        # Blender Python script to convert to STL
        script = f'''
import bpy
import sys

# Clear default scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Enable CTM addon if needed (for .ctm files)
if "{ext}" == ".ctm":
    try:
        bpy.ops.preferences.addon_enable(module="io_mesh_ctm")
    except Exception as e:
        print(f"Warning: Could not enable CTM addon: {{e}}", file=sys.stderr)
        # Try anyway, addon might already be enabled

# Import the file
try:
    {import_cmd}
except Exception as e:
    print(f"Import failed: {{e}}", file=sys.stderr)
    sys.exit(1)

# Select all mesh objects
bpy.ops.object.select_all(action='DESELECT')
mesh_count = 0
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH':
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        mesh_count += 1

if mesh_count == 0:
    print("Error: No mesh objects found in scene", file=sys.stderr)
    sys.exit(1)

print(f"Found {{mesh_count}} mesh object(s)")

# Join if multiple objects
if mesh_count > 1:
    bpy.ops.object.join()

# Export to STL
bpy.ops.export_mesh.stl(filepath=r"{tmp_path}", use_selection=True)
print(f"Exported to: {tmp_path}")
'''
        
        # Write script to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix=".py", delete=False) as script_file:
            script_file.write(script)
            script_path = script_file.name
        
        try:
            # Run Blender
            result = subprocess.run(
                [blender_path, "--background", "--python", script_path],
                capture_output=True,
                text=True,
                timeout=120,
            )
            
            if result.returncode != 0:
                logger.error(f"Blender conversion failed: {result.stderr}")
                raise ValueError(f"Blender conversion failed: {result.stderr[:500]}")
            
            # Load the converted STL
            if not tmp_path.exists():
                raise ValueError("Blender did not produce output file")
            
            mesh = trimesh.load(str(tmp_path), force='mesh')
            logger.info(f"Converted with Blender: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            
            return mesh
            
        finally:
            # Clean up script
            Path(script_path).unlink(missing_ok=True)
    
    finally:
        # Clean up temp STL
        tmp_path.unlink(missing_ok=True)


def _load_with_trimesh(path: Path, **kwargs) -> trimesh.Trimesh:
    """Load a mesh using trimesh."""
    logger.info(f"Loading with trimesh: {path}")
    
    mesh = trimesh.load(str(path), force='mesh', **kwargs)
    
    # If it's a Scene (multiple objects), concatenate them
    if isinstance(mesh, trimesh.Scene):
        geometries = list(mesh.geometry.values())
        if len(geometries) == 0:
            raise ValueError("No geometry found in file")
        elif len(geometries) == 1:
            mesh = geometries[0]
        else:
            logger.info(f"Concatenating {len(geometries)} geometries from scene")
            mesh = trimesh.util.concatenate(geometries)
    
    logger.info(f"Loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    return mesh


def _load_with_meshio(path: Path) -> trimesh.Trimesh:
    """Load a mesh using meshio (for FEM formats)."""
    try:
        import meshio
    except ImportError:
        raise ImportError("meshio is required for this format")
    
    logger.info(f"Loading with meshio: {path}")
    
    mesh_data = meshio.read(str(path))
    
    # Extract vertices
    vertices = mesh_data.points
    
    # Find triangle cells
    faces = None
    for cell_block in mesh_data.cells:
        if cell_block.type == "triangle":
            faces = cell_block.data
            break
    
    if faces is None:
        # Try to find any polygon cells and triangulate
        for cell_block in mesh_data.cells:
            if cell_block.type in ["quad", "polygon"]:
                # Simple triangulation for quads
                if cell_block.type == "quad":
                    quads = cell_block.data
                    # Split each quad into 2 triangles
                    tri1 = quads[:, [0, 1, 2]]
                    tri2 = quads[:, [0, 2, 3]]
                    faces = np.vstack([tri1, tri2])
                    break
    
    if faces is None:
        raise ValueError("No triangular faces found in meshio data")
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    logger.info(f"Loaded via meshio: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    return mesh


def load_mesh(
    path: Union[str, Path],
    cad_resolution: float = 0.01,
    blender_path: Optional[str] = None,
) -> LoadResult:
    """
    Load a 3D mesh from any supported format.
    
    This is the primary entry point for loading meshes. It automatically
    detects the file format and uses the appropriate loader.
    
    Args:
        path: Path to the mesh file
        cad_resolution: Tessellation resolution for CAD formats (STEP, IGES).
                       Lower values = finer mesh. Default: 0.01
        blender_path: Optional path to Blender executable for formats
                     that require Blender conversion.
    
    Returns:
        LoadResult containing the mesh and metadata about the conversion.
    
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If format is not supported or loading fails
        ImportError: If required optional dependency is missing
    
    Example:
        >>> result = load_mesh("model.ctm")
        >>> mesh = result.mesh
        >>> print(f"Loaded {result.original_format} with {result.original_vertices} vertices")
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")
    
    ext = path.suffix.lower()
    warnings = []
    conversion_notes = ""
    conversion_method = "trimesh"
    
    # Check if format is supported
    if ext not in SUPPORTED_EXTENSIONS:
        # Try loading with trimesh anyway (it might support more formats)
        logger.warning(f"Unknown format '{ext}', attempting to load with trimesh")
        warnings.append(f"Unknown format '{ext}', loaded with trimesh")
    
    # Route to appropriate loader
    mesh = None
    
    try:
        # PyMeshLab formats (CTM, etc.) - best for compressed mesh formats
        if ext in PYMESHLAB_FORMATS:
            mesh = _load_with_pymeshlab(path)
            conversion_method = "pymeshlab"
            conversion_notes = "Loaded via PyMeshLab (MeshLab Python bindings)"
        
        # Formats that require Blender (FBX, BLEND)
        elif ext in BLENDER_REQUIRED_FORMATS:
            if blender_path is None:
                blender_path = _check_blender_available()
            
            if blender_path:
                mesh = _load_with_blender(path, blender_path)
                conversion_method = "blender"
                conversion_notes = f"Converted via Blender ({blender_path})"
            else:
                # Try trimesh as fallback for FBX (limited support)
                if ext == ".fbx":
                    try:
                        mesh = _load_with_trimesh(path)
                        conversion_notes = "Loaded via trimesh (limited FBX support)"
                        warnings.append("FBX loaded without Blender - some features may be missing")
                    except Exception:
                        raise ImportError(
                            f"Format {ext} requires Blender for conversion. "
                            "Please install Blender or provide --blender-path."
                        )
                else:
                    raise ImportError(
                        f"Format {ext} requires Blender for conversion. "
                        "Please install Blender or provide --blender-path."
                    )
        
        # CAD formats (may need OpenCASCADE)
        elif ext in [".step", ".stp", ".iges", ".igs", ".brep"]:
            try:
                # trimesh with OpenCASCADE should handle these
                mesh = _load_with_trimesh(path)
                conversion_notes = f"Loaded via trimesh/OpenCASCADE (resolution: {cad_resolution})"
            except Exception as e:
                if "cascade" in str(e).lower() or "occ" in str(e).lower():
                    raise ImportError(
                        f"CAD format {ext} requires OpenCASCADE. "
                        "Install with: pip install trimesh[easy]"
                    )
                raise
        
        # FEM/Engineering formats (try meshio first)
        elif ext in [".msh", ".nas", ".bdf", ".fem", ".inp", ".e", ".ex2", ".exo",
                     ".cgns", ".xdmf", ".xmf", ".dat", ".tec", ".vol", ".ele", 
                     ".node", ".ugrid", ".su2", ".mesh", ".meshb"]:
            try:
                mesh = _load_with_meshio(path)
                conversion_method = "meshio"
                conversion_notes = "Loaded via meshio"
            except Exception as meshio_error:
                logger.warning(f"meshio failed, trying trimesh: {meshio_error}")
                try:
                    mesh = _load_with_trimesh(path)
                    conversion_notes = "Loaded via trimesh (meshio fallback failed)"
                    warnings.append(f"meshio failed: {meshio_error}")
                except Exception as trimesh_error:
                    raise ValueError(
                        f"Failed to load {ext} with both meshio and trimesh. "
                        f"meshio: {meshio_error}, trimesh: {trimesh_error}"
                    )
        
        # Standard formats (trimesh handles well)
        else:
            mesh = _load_with_trimesh(path)
            conversion_notes = "Loaded via trimesh"
        
        # Ensure we got a valid mesh
        if mesh is None or len(mesh.vertices) == 0:
            raise ValueError("Loaded mesh has no vertices")
        
        if len(mesh.faces) == 0:
            raise ValueError("Loaded mesh has no faces")
        
        # Create result
        result = LoadResult(
            mesh=mesh,
            original_format=ext.lstrip("."),
            original_path=path,
            original_vertices=len(mesh.vertices),
            original_faces=len(mesh.faces),
            conversion_method=conversion_method,
            conversion_notes=conversion_notes,
            warnings=warnings,
        )
        
        logger.info(
            f"Successfully loaded {path.name}: "
            f"{result.original_vertices:,} vertices, "
            f"{result.original_faces:,} faces "
            f"(method: {conversion_method})"
        )
        
        return result
        
    except ImportError:
        # Re-raise import errors with helpful messages
        raise
    except Exception as e:
        logger.error(f"Failed to load mesh {path}: {e}")
        raise ValueError(f"Failed to load mesh: {e}") from e


def convert_to_stl(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    cad_resolution: float = 0.01,
    ascii_format: bool = False,
    blender_path: Optional[str] = None,
) -> LoadResult:
    """
    Convert any supported mesh format to STL.
    
    This is a convenience function that loads a mesh and saves it as STL.
    
    Args:
        input_path: Path to input mesh file
        output_path: Path for output STL file
        cad_resolution: Tessellation resolution for CAD formats
        ascii_format: If True, output ASCII STL instead of binary
        blender_path: Optional path to Blender executable
    
    Returns:
        LoadResult with information about the conversion
    
    Example:
        >>> result = convert_to_stl("model.ctm", "model.stl")
        >>> print(f"Converted {result.original_format} to STL")
    """
    # Load the mesh
    result = load_mesh(input_path, cad_resolution=cad_resolution, blender_path=blender_path)
    
    # Save as STL
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if ascii_format:
        result.mesh.export(str(output_path), file_type="stl_ascii")
    else:
        result.mesh.export(str(output_path), file_type="stl")
    
    logger.info(f"Converted to STL: {output_path}")
    
    return result


def list_supported_formats() -> str:
    """
    Get a formatted string listing all supported formats.
    
    Returns:
        Multi-line string suitable for display in help text.
    """
    lines = ["Supported 3D File Formats:", "=" * 40]
    
    for category_id, category in FORMAT_CATEGORIES.items():
        lines.append(f"\n{category['name']}:")
        for ext in category["extensions"]:
            dep = OPTIONAL_DEPENDENCY_FORMATS.get(ext, "")
            if dep:
                lines.append(f"  {ext:10} (requires {dep})")
            else:
                lines.append(f"  {ext}")
    
    lines.append(f"\nTotal: {len(SUPPORTED_EXTENSIONS)} formats supported")
    
    return "\n".join(lines)


# For backwards compatibility with existing code
def load_stl(path: Union[str, Path]) -> trimesh.Trimesh:
    """
    Load an STL file (legacy function for compatibility).
    
    Use load_mesh() for new code - it handles all formats.
    """
    result = load_mesh(path)
    return result.mesh
