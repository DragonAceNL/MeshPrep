# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Blender-based actions for mesh repair escalation.

These actions use Blender's powerful mesh tools for cases where
trimesh and pymeshfix fail. Blender is invoked as a subprocess
to avoid dependency on bpy in the main process.

Note: Blender must be installed and available in PATH, or the
BLENDER_PATH environment variable must be set.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

from .registry import register_action

logger = logging.getLogger(__name__)

# Blender detection
_BLENDER_PATH: Optional[str] = None


def find_blender() -> Optional[str]:
    """
    Find Blender executable.
    
    Checks:
    1. BLENDER_PATH environment variable
    2. PATH (via shutil.which)
    3. Common installation locations on Windows
    
    Returns:
        Path to blender executable, or None if not found
    """
    global _BLENDER_PATH
    
    if _BLENDER_PATH is not None:
        return _BLENDER_PATH
    
    # Check environment variable
    env_path = os.environ.get("BLENDER_PATH")
    if env_path and Path(env_path).exists():
        _BLENDER_PATH = env_path
        return _BLENDER_PATH
    
    # Check PATH
    which_result = shutil.which("blender")
    if which_result:
        _BLENDER_PATH = which_result
        return _BLENDER_PATH
    
    # Check common Windows locations
    common_paths = [
        r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.3\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe",
    ]
    
    for path in common_paths:
        if Path(path).exists():
            _BLENDER_PATH = path
            return _BLENDER_PATH
    
    return None


def is_blender_available() -> bool:
    """Check if Blender is available."""
    return find_blender() is not None


def get_blender_version() -> Optional[str]:
    """Get Blender version string."""
    blender = find_blender()
    if not blender:
        return None
    
    try:
        result = subprocess.run(
            [blender, "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        # Parse first line: "Blender 4.0.0"
        first_line = result.stdout.strip().split("\n")[0]
        return first_line
    except Exception:
        return None


# Blender Python script template for mesh repair
BLENDER_REPAIR_SCRIPT = '''
import bpy
import sys
import json

# Get arguments after "--"
argv = sys.argv
argv = argv[argv.index("--") + 1:]

input_path = argv[0]
output_path = argv[1]
operation = argv[2]
params_json = argv[3] if len(argv) > 3 else "{}"
params = json.loads(params_json)

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import STL
bpy.ops.wm.stl_import(filepath=input_path)

# Get the imported object
obj = bpy.context.selected_objects[0] if bpy.context.selected_objects else None

if obj is None:
    print("ERROR: No object imported")
    sys.exit(1)

# Make it the active object
bpy.context.view_layer.objects.active = obj
obj.select_set(True)

# Enter edit mode for mesh operations
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')

if operation == "remesh":
    # Exit edit mode for modifier
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Add remesh modifier
    modifier = obj.modifiers.new(name="Remesh", type='REMESH')
    modifier.mode = params.get("mode", "VOXEL")
    modifier.voxel_size = params.get("voxel_size", 0.1)
    modifier.adaptivity = params.get("adaptivity", 0.0)
    
    # Apply modifier
    bpy.ops.object.modifier_apply(modifier="Remesh")

elif operation == "boolean_union":
    # Exit edit mode
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Separate by loose parts
    bpy.ops.mesh.separate(type='LOOSE')
    
    # Get all mesh objects
    mesh_objects = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    
    if len(mesh_objects) > 1:
        # Use first object as base
        base = mesh_objects[0]
        bpy.context.view_layer.objects.active = base
        
        for other in mesh_objects[1:]:
            # Add boolean modifier
            modifier = base.modifiers.new(name="Boolean", type='BOOLEAN')
            modifier.operation = 'UNION'
            modifier.object = other
            modifier.solver = 'EXACT'
            
            # Apply
            bpy.ops.object.modifier_apply(modifier="Boolean")
            
            # Delete the other object
            bpy.data.objects.remove(other, do_unlink=True)
        
        obj = base

elif operation == "fill_holes":
    # In edit mode, fill holes
    bpy.ops.mesh.fill_holes(sides=params.get("max_sides", 100))
    bpy.ops.object.mode_set(mode='OBJECT')

elif operation == "make_manifold":
    # Try to make manifold
    bpy.ops.mesh.select_all(action='SELECT')
    
    # Remove doubles
    bpy.ops.mesh.remove_doubles(threshold=params.get("threshold", 0.0001))
    
    # Fill holes
    bpy.ops.mesh.select_non_manifold()
    bpy.ops.mesh.fill_holes(sides=100)
    
    # Recalculate normals
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    
    bpy.ops.object.mode_set(mode='OBJECT')

elif operation == "decimate":
    bpy.ops.object.mode_set(mode='OBJECT')
    
    modifier = obj.modifiers.new(name="Decimate", type='DECIMATE')
    modifier.ratio = params.get("ratio", 0.5)
    
    bpy.ops.object.modifier_apply(modifier="Decimate")

elif operation == "triangulate":
    bpy.ops.mesh.quads_convert_to_tris()
    bpy.ops.object.mode_set(mode='OBJECT')

else:
    # Default: just clean up
    bpy.ops.mesh.remove_doubles()
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')

# Ensure we're in object mode
if bpy.context.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')

# Export result
obj = bpy.context.view_layer.objects.active
if obj:
    obj.select_set(True)
    bpy.ops.wm.stl_export(filepath=output_path, export_selected_objects=True)
    print("SUCCESS: Exported to", output_path)
else:
    print("ERROR: No object to export")
    sys.exit(1)
'''


def run_blender_script(
    input_mesh: trimesh.Trimesh,
    operation: str,
    params: Optional[dict] = None,
    timeout: int = 120
) -> trimesh.Trimesh:
    """
    Run a Blender operation on a mesh.
    
    Args:
        input_mesh: Input trimesh mesh
        operation: Operation name (remesh, boolean_union, fill_holes, make_manifold, etc.)
        params: Operation parameters
        timeout: Timeout in seconds
        
    Returns:
        Repaired mesh
        
    Raises:
        RuntimeError: If Blender is not available or operation fails
    """
    blender = find_blender()
    if not blender:
        raise RuntimeError("Blender not found. Install Blender or set BLENDER_PATH.")
    
    params = params or {}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Save input mesh
        input_path = tmpdir / "input.stl"
        output_path = tmpdir / "output.stl"
        script_path = tmpdir / "repair_script.py"
        
        input_mesh.export(str(input_path))
        
        # Write script
        script_path.write_text(BLENDER_REPAIR_SCRIPT)
        
        # Run Blender
        cmd = [
            blender,
            "--background",
            "--python", str(script_path),
            "--",
            str(input_path),
            str(output_path),
            operation,
            json.dumps(params)
        ]
        
        logger.info(f"Running Blender {operation}...")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Blender stderr: {result.stderr}")
                raise RuntimeError(f"Blender failed: {result.stderr}")
            
            if not output_path.exists():
                raise RuntimeError("Blender did not produce output file")
            
            # Load result
            output_mesh = trimesh.load(str(output_path), force='mesh')
            
            if isinstance(output_mesh, trimesh.Scene):
                output_mesh = trimesh.util.concatenate(list(output_mesh.geometry.values()))
            
            logger.info(f"Blender {operation} complete: {len(output_mesh.vertices)} vertices")
            
            return output_mesh
            
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Blender operation timed out after {timeout}s")


@register_action(
    name="blender_remesh",
    description="Apply Blender's voxel remesh for aggressive topology repair",
    parameters={
        "voxel_size": 0.1,
        "mode": "VOXEL",
        "adaptivity": 0.0
    },
    risk_level="high"
)
def action_blender_remesh(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Apply Blender's voxel remesh.
    
    This creates a completely new topology, which can fix even
    severely broken meshes but may lose detail.
    """
    if not is_blender_available():
        logger.warning("Blender not available, returning original mesh")
        return mesh.copy()
    
    voxel_size = params.get("voxel_size", 0.1)
    
    # Auto-calculate voxel size based on mesh size if not specified
    if voxel_size == "auto":
        bbox_diagonal = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
        voxel_size = bbox_diagonal / 100  # 1% of bbox diagonal
    
    return run_blender_script(mesh, "remesh", {
        "voxel_size": voxel_size,
        "mode": params.get("mode", "VOXEL"),
        "adaptivity": params.get("adaptivity", 0.0)
    })


@register_action(
    name="blender_boolean_union",
    description="Merge all mesh parts using Blender's boolean solver",
    parameters={},
    risk_level="high"
)
def action_blender_boolean_union(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Merge overlapping mesh parts using Blender's exact boolean solver.
    
    This can fix self-intersecting geometry and merge separate components.
    """
    if not is_blender_available():
        logger.warning("Blender not available, returning original mesh")
        return mesh.copy()
    
    return run_blender_script(mesh, "boolean_union", params)


@register_action(
    name="blender_fill_holes",
    description="Fill holes using Blender's hole filling algorithm",
    parameters={"max_sides": 100},
    risk_level="medium"
)
def action_blender_fill_holes(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Fill holes using Blender.
    
    Blender's hole filling can handle complex non-planar holes
    better than trimesh in some cases.
    """
    if not is_blender_available():
        logger.warning("Blender not available, returning original mesh")
        return mesh.copy()
    
    return run_blender_script(mesh, "fill_holes", {
        "max_sides": params.get("max_sides", 100)
    })


@register_action(
    name="blender_make_manifold",
    description="Make mesh manifold using Blender's mesh tools",
    parameters={"threshold": 0.0001},
    risk_level="medium"
)
def action_blender_make_manifold(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Make mesh manifold using Blender.
    
    Combines remove doubles, hole filling, and normal recalculation.
    """
    if not is_blender_available():
        logger.warning("Blender not available, returning original mesh")
        return mesh.copy()
    
    return run_blender_script(mesh, "make_manifold", {
        "threshold": params.get("threshold", 0.0001)
    })


@register_action(
    name="blender_decimate",
    description="Reduce face count using Blender's decimate modifier",
    parameters={"ratio": 0.5},
    risk_level="medium"
)
def action_blender_decimate(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Decimate mesh using Blender.
    """
    if not is_blender_available():
        logger.warning("Blender not available, returning original mesh")
        return mesh.copy()
    
    return run_blender_script(mesh, "decimate", {
        "ratio": params.get("ratio", 0.5)
    })
