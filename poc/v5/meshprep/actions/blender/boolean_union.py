# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Boolean union to merge mesh components using Blender."""

from typing import Dict, Any, Optional
import tempfile
import subprocess
from pathlib import Path
import shutil

from meshprep.core.action import Action, ActionRiskLevel, register_action
from meshprep.core.mesh import Mesh


@register_action
class BlenderBooleanUnionAction(Action):
    """Merge all mesh components with Boolean union."""
    
    name = "blender_boolean_union"
    description = "Merge mesh components with Boolean union (Blender)"
    risk_level = ActionRiskLevel.MEDIUM
    default_params = {}
    
    def execute(self, mesh: Mesh, params: Optional[Dict[str, Any]] = None) -> Mesh:
        """Boolean union all components."""
        params = self.get_params(params)
        
        # Find Blender
        blender = self._find_blender()
        if not blender:
            raise RuntimeError("Blender not found")
        
        # Only useful if multiple components
        if mesh.metadata.body_count <= 1:
            self.logger.info("Only 1 component, skipping Boolean union")
            return mesh
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.stl"
            output_path = Path(tmpdir) / "output.stl"
            script_path = Path(tmpdir) / "boolean_union.py"
            
            mesh.trimesh.export(str(input_path))
            
            script = f'''
import bpy
import sys

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import
bpy.ops.wm.stl_import(filepath=r"{input_path}")

# Select all objects
objects = [o for o in bpy.context.scene.objects if o.type == 'MESH']

if len(objects) <= 1:
    # Nothing to union
    if objects:
        bpy.ops.wm.stl_export(filepath=r"{output_path}")
    sys.exit(0)

# Set first object as active
base = objects[0]
bpy.context.view_layer.objects.active = base

# Boolean union with all others
for obj in objects[1:]:
    modifier = base.modifiers.new(name="Boolean", type='BOOLEAN')
    modifier.operation = 'UNION'
    modifier.object = obj
    bpy.ops.object.modifier_apply(modifier="Boolean")
    bpy.data.objects.remove(obj, do_unlink=True)

# Export
bpy.ops.wm.stl_export(filepath=r"{output_path}")
sys.exit(0)
'''
            
            script_path.write_text(script)
            
            self.logger.info(f"Running Boolean union on {mesh.metadata.body_count} components...")
            
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
                    raise RuntimeError("Blender did not produce output")
                
                result_mesh = Mesh.load(output_path)
                
                self.logger.info(f"Boolean union complete: "
                               f"{result_mesh.metadata.body_count} components "
                               f"(was {mesh.metadata.body_count})")
                
                return result_mesh
                
            except subprocess.TimeoutExpired:
                raise RuntimeError("Blender timeout (>120s)")
    
    def _find_blender(self) -> Optional[str]:
        """Find Blender executable."""
        candidates = [
            "blender",
            r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender\blender.exe",
        ]
        
        for candidate in candidates:
            if shutil.which(candidate) or Path(candidate).exists():
                return candidate
        
        return None
