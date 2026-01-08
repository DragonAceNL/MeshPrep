# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Blender voxel remesh action (subprocess-based)."""

from typing import Dict, Any, Optional
import tempfile
import subprocess
from pathlib import Path
import shutil

from meshprep.core.action import Action, ActionRiskLevel, register_action
from meshprep.core.mesh import Mesh


@register_action
class BlenderRemeshAction(Action):
    """Voxel remesh using Blender (external process)."""
    
    name = "blender_remesh"
    description = "Voxel remesh using Blender (significant geometry changes)"
    risk_level = ActionRiskLevel.HIGH
    default_params = {
        "voxel_size": 0.01,  # Voxel size (smaller = more detail)
        "adaptivity": 0.0,   # Adaptivity (0-1, higher = adaptive)
    }
    
    def execute(self, mesh: Mesh, params: Optional[Dict[str, Any]] = None) -> Mesh:
        """Remesh using Blender."""
        params = self.get_params(params)
        
        # Find Blender
        blender = self._find_blender()
        if not blender:
            raise RuntimeError("Blender not found. Install Blender 4.2+")
        
        # Create temp files
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.stl"
            output_path = Path(tmpdir) / "output.stl"
            script_path = Path(tmpdir) / "remesh.py"
            
            # Save input mesh
            mesh.trimesh.export(str(input_path))
            
            # Create Blender Python script
            voxel_size = params["voxel_size"]
            adaptivity = params["adaptivity"]
            
            script = f'''
import bpy
import sys

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import mesh
bpy.ops.wm.stl_import(filepath=r"{input_path}")

# Get imported object
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj

# Add remesh modifier
modifier = obj.modifiers.new(name="Remesh", type='REMESH')
modifier.mode = 'VOXEL'
modifier.voxel_size = {voxel_size}
modifier.adaptivity = {adaptivity}

# Apply modifier
bpy.ops.object.modifier_apply(modifier="Remesh")

# Export
bpy.ops.wm.stl_export(filepath=r"{output_path}")

sys.exit(0)
'''
            
            script_path.write_text(script)
            
            # Run Blender
            self.logger.info(f"Running Blender remesh (voxel_size={voxel_size})...")
            
            try:
                result = subprocess.run(
                    [blender, "--background", "--python", str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                
                if result.returncode != 0:
                    raise RuntimeError(f"Blender failed: {result.stderr}")
                
                if not output_path.exists():
                    raise RuntimeError("Blender did not produce output file")
                
                # Load result
                result_mesh = Mesh.load(output_path)
                
                self.logger.info(f"Blender remesh complete: "
                               f"{result_mesh.metadata.face_count} faces "
                               f"(was {mesh.metadata.face_count})")
                
                return result_mesh
                
            except subprocess.TimeoutExpired:
                raise RuntimeError("Blender timeout (>120s)")
    
    def _find_blender(self) -> Optional[str]:
        """Find Blender executable."""
        from meshprep.core.bootstrap import get_bootstrap_manager
        manager = get_bootstrap_manager()
        
        blender_path = manager.get_blender_path()
        if blender_path:
            return str(blender_path)
        
        return None
