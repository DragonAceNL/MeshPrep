# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Add thickness to thin meshes using Blender."""

from typing import Dict, Any, Optional
import tempfile
import subprocess
from pathlib import Path
import shutil

from meshprep.core.action import Action, ActionRiskLevel, register_action
from meshprep.core.mesh import Mesh


@register_action
class BlenderSolidifyAction(Action):
    """Add thickness to thin/sheet meshes."""
    
    name = "blender_solidify"
    description = "Add thickness to thin meshes (Blender solidify modifier)"
    risk_level = ActionRiskLevel.MEDIUM
    default_params = {"thickness": 0.1}
    
    def execute(self, mesh: Mesh, params: Optional[Dict[str, Any]] = None) -> Mesh:
        """Add thickness to mesh."""
        params = self.get_params(params)
        thickness = params["thickness"]
        
        blender = self._find_blender()
        if not blender:
            raise RuntimeError("Blender not found")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.stl"
            output_path = Path(tmpdir) / "output.stl"
            script_path = Path(tmpdir) / "solidify.py"
            
            mesh.trimesh.export(str(input_path))
            
            script = f'''
import bpy
import sys

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

bpy.ops.wm.stl_import(filepath=r"{input_path}")

obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj

# Add solidify modifier
modifier = obj.modifiers.new(name="Solidify", type='SOLIDIFY')
modifier.thickness = {thickness}
modifier.offset = 0  # Centered

# Apply modifier
bpy.ops.object.modifier_apply(modifier="Solidify")

bpy.ops.wm.stl_export(filepath=r"{output_path}")
sys.exit(0)
'''
            
            script_path.write_text(script)
            
            self.logger.info(f"Adding thickness: {thickness}")
            
            try:
                result = subprocess.run(
                    [blender, "--background", "--python", str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                
                if result.returncode != 0:
                    raise RuntimeError(f"Blender failed: {result.stderr}")
                
                if not output_path.exists():
                    raise RuntimeError("Blender did not produce output")
                
                result_mesh = Mesh.load(output_path)
                
                self.logger.info(f"Solidify complete: added {thickness} thickness")
                
                return result_mesh
                
            except subprocess.TimeoutExpired:
                raise RuntimeError("Blender timeout (>60s)")
    
    def _find_blender(self) -> Optional[str]:
        """Find Blender executable."""
        from meshprep.core.bootstrap import get_bootstrap_manager
        manager = get_bootstrap_manager()
        
        blender_path = manager.get_blender_path()
        if blender_path:
            return str(blender_path)
        
        return None
