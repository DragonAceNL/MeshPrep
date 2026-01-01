# Copyright 2025 Dragon Ace
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""Action registry and implementations for filter scripts."""

from dataclasses import dataclass, field
from typing import Callable, Optional, Any
from enum import Enum

from .mock_mesh import MockMesh, MockTrimesh, MockPyMeshFix, MockBlender
from .diagnostics import Diagnostics, compute_diagnostics


class ActionCategory(Enum):
    """Categories of filter actions."""
    CLEANUP = "Loading & Basic Cleanup"
    HOLE_FILLING = "Hole Filling"
    NORMALS = "Normal Correction"
    COMPONENTS = "Component Management"
    REPAIR = "Repair & Manifold Fixes"
    SIMPLIFICATION = "Simplification & Remeshing"
    GEOMETRY = "Geometry Analysis"
    BOOLEAN = "Boolean & Intersection Fixes"
    VALIDATION = "Validation & Diagnostics"
    EXPORT = "Export"
    BLENDER = "Blender (Escalation)"


@dataclass
class ActionParameter:
    """Definition of an action parameter."""
    
    name: str
    param_type: str  # "int", "float", "bool", "string", "path", "enum"
    default: Any = None
    description: str = ""
    required: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    enum_values: list[str] = field(default_factory=list)


@dataclass
class Action:
    """Definition of a filter action."""
    
    name: str
    display_name: str
    category: ActionCategory
    tool: str  # "trimesh", "pymeshfix", "meshio", "blender", "internal"
    description: str
    parameters: list[ActionParameter] = field(default_factory=list)
    dry_run_supported: bool = True
    destructive: bool = True
    
    def __hash__(self):
        return hash(self.name)


@dataclass
class ActionResult:
    """Result of executing an action."""
    
    action_name: str
    success: bool
    mesh: Optional[MockMesh] = None
    diagnostics: Optional[Diagnostics] = None
    message: str = ""
    error: Optional[str] = None
    runtime_ms: float = 0.0


class ActionRegistry:
    """Registry of all available filter actions."""
    
    def __init__(self):
        """Initialize registry with all actions."""
        self._actions: dict[str, Action] = {}
        self._implementations: dict[str, Callable] = {}
        self._register_all_actions()
    
    def _register_all_actions(self):
        """Register all available actions."""
        
        # Loading & Basic Cleanup
        self._register(
            Action(
                name="trimesh_basic",
                display_name="Basic Cleanup",
                category=ActionCategory.CLEANUP,
                tool="trimesh",
                description="Load and apply basic cleanup: merge duplicate vertices, remove degenerate faces.",
                parameters=[
                    ActionParameter("merge_tex", "bool", True, "Merge texture coordinates"),
                    ActionParameter("merge_norm", "bool", True, "Merge normals"),
                ],
            ),
            self._impl_trimesh_basic,
        )
        
        self._register(
            Action(
                name="merge_vertices",
                display_name="Merge Vertices",
                category=ActionCategory.CLEANUP,
                tool="trimesh",
                description="Weld duplicate vertices within a tolerance.",
                parameters=[
                    ActionParameter("eps", "float", 1e-8, "Distance threshold", min_value=0),
                ],
            ),
            self._impl_merge_vertices,
        )
        
        self._register(
            Action(
                name="remove_degenerate_faces",
                display_name="Remove Degenerate Faces",
                category=ActionCategory.CLEANUP,
                tool="trimesh",
                description="Remove faces with zero area or invalid topology.",
            ),
            self._impl_remove_degenerate_faces,
        )
        
        # Hole Filling
        self._register(
            Action(
                name="fill_holes",
                display_name="Fill Holes",
                category=ActionCategory.HOLE_FILLING,
                tool="trimesh",
                description="Fill holes in the mesh up to a maximum size.",
                parameters=[
                    ActionParameter("max_hole_size", "int", 1000, "Maximum hole size in edges", min_value=1),
                    ActionParameter("method", "enum", "fan", "Fill method", enum_values=["fan", "ear"]),
                ],
            ),
            self._impl_fill_holes,
        )
        
        # Normal Correction
        self._register(
            Action(
                name="recalculate_normals",
                display_name="Recalculate Normals",
                category=ActionCategory.NORMALS,
                tool="trimesh",
                description="Recompute face normals from vertex winding order.",
            ),
            self._impl_recalculate_normals,
        )
        
        self._register(
            Action(
                name="fix_normals",
                display_name="Fix Normals",
                category=ActionCategory.NORMALS,
                tool="trimesh",
                description="Combine recalculate and reorient for a single-step fix.",
            ),
            self._impl_fix_normals,
        )
        
        self._register(
            Action(
                name="flip_normals",
                display_name="Flip Normals",
                category=ActionCategory.NORMALS,
                tool="trimesh",
                description="Invert all face normals (useful if model is inside-out).",
            ),
            self._impl_flip_normals,
        )
        
        # Component Management
        self._register(
            Action(
                name="remove_small_components",
                display_name="Remove Small Components",
                category=ActionCategory.COMPONENTS,
                tool="trimesh",
                description="Remove disconnected components below a threshold.",
                parameters=[
                    ActionParameter("min_faces", "int", 100, "Minimum faces to keep", min_value=1),
                ],
            ),
            self._impl_remove_small_components,
        )
        
        self._register(
            Action(
                name="boolean_union",
                display_name="Boolean Union",
                category=ActionCategory.COMPONENTS,
                tool="trimesh",
                description="Merge overlapping shells into a single watertight mesh.",
            ),
            self._impl_boolean_union,
        )
        
        self._register(
            Action(
                name="remove_internal_geometry",
                display_name="Remove Internal Geometry",
                category=ActionCategory.COMPONENTS,
                tool="trimesh",
                description="Remove components fully enclosed by the outer shell.",
            ),
            self._impl_remove_internal_geometry,
        )
        
        # Repair & Manifold Fixes
        self._register(
            Action(
                name="pymeshfix_repair",
                display_name="PyMeshFix Repair",
                category=ActionCategory.REPAIR,
                tool="pymeshfix",
                description="Run pymeshfix's automatic repair pass. Fixes many non-manifold issues.",
            ),
            self._impl_pymeshfix_repair,
        )
        
        # Simplification & Remeshing
        self._register(
            Action(
                name="decimate",
                display_name="Decimate",
                category=ActionCategory.SIMPLIFICATION,
                tool="trimesh",
                description="Reduce face count while preserving shape.",
                parameters=[
                    ActionParameter("target_ratio", "float", 0.5, "Target ratio", min_value=0.01, max_value=1.0),
                    ActionParameter("target_faces", "int", None, "Target face count"),
                ],
            ),
            self._impl_decimate,
        )
        
        self._register(
            Action(
                name="smooth_laplacian",
                display_name="Laplacian Smooth",
                category=ActionCategory.SIMPLIFICATION,
                tool="trimesh",
                description="Apply Laplacian smoothing to reduce noise.",
                parameters=[
                    ActionParameter("iterations", "int", 1, "Smoothing iterations", min_value=1, max_value=100),
                ],
            ),
            self._impl_smooth_laplacian,
        )
        
        # Geometry Analysis
        self._register(
            Action(
                name="identify_thin_regions",
                display_name="Identify Thin Regions",
                category=ActionCategory.GEOMETRY,
                tool="internal",
                description="Detect regions thinner than a threshold.",
                parameters=[
                    ActionParameter("min_thickness", "float", 0.8, "Minimum thickness (mm)", min_value=0.1),
                ],
                destructive=False,
            ),
            self._impl_identify_thin_regions,
        )
        
        self._register(
            Action(
                name="thicken_regions",
                display_name="Thicken Regions",
                category=ActionCategory.GEOMETRY,
                tool="blender",
                description="Thicken thin walls to meet minimum printable thickness.",
                parameters=[
                    ActionParameter("target_thickness", "float", 1.0, "Target thickness (mm)", min_value=0.1),
                ],
            ),
            self._impl_thicken_regions,
        )
        
        # Validation & Diagnostics
        self._register(
            Action(
                name="validate",
                display_name="Validate",
                category=ActionCategory.VALIDATION,
                tool="internal",
                description="Run all validation checks and produce a diagnostics report.",
                destructive=False,
            ),
            self._impl_validate,
        )
        
        self._register(
            Action(
                name="compute_diagnostics",
                display_name="Compute Diagnostics",
                category=ActionCategory.VALIDATION,
                tool="internal",
                description="Compute full diagnostics vector for profile detection.",
                destructive=False,
            ),
            self._impl_compute_diagnostics,
        )
        
        # Export
        self._register(
            Action(
                name="export_stl",
                display_name="Export STL",
                category=ActionCategory.EXPORT,
                tool="trimesh",
                description="Export mesh to STL file.",
                parameters=[
                    ActionParameter("path", "path", "./output/cleaned_model.stl", "Output path"),
                    ActionParameter("ascii", "bool", False, "Use ASCII format"),
                ],
            ),
            self._impl_export_stl,
        )
        
        # Blender (Escalation)
        self._register(
            Action(
                name="blender_remesh",
                display_name="Blender Remesh",
                category=ActionCategory.BLENDER,
                tool="blender",
                description="Apply Blender's voxel remesh for aggressive topology repair.",
                parameters=[
                    ActionParameter("voxel_size", "float", 0.1, "Voxel size", min_value=0.01, max_value=10.0),
                ],
            ),
            self._impl_blender_remesh,
        )
        
        self._register(
            Action(
                name="blender_boolean_union",
                display_name="Blender Boolean Union",
                category=ActionCategory.BLENDER,
                tool="blender",
                description="Merge all mesh parts using Blender's boolean solver.",
            ),
            self._impl_blender_boolean_union,
        )
        
        self._register(
            Action(
                name="blender_solidify",
                display_name="Blender Solidify",
                category=ActionCategory.BLENDER,
                tool="blender",
                description="Add thickness to thin surfaces using the solidify modifier.",
                parameters=[
                    ActionParameter("thickness", "float", 1.0, "Wall thickness", min_value=0.1),
                ],
            ),
            self._impl_blender_solidify,
        )
    
    def _register(self, action: Action, implementation: Callable):
        """Register an action with its implementation."""
        self._actions[action.name] = action
        self._implementations[action.name] = implementation
    
    def get(self, name: str) -> Optional[Action]:
        """Get an action by name."""
        return self._actions.get(name)
    
    def list_actions(self) -> list[Action]:
        """Get all registered actions."""
        return list(self._actions.values())
    
    def list_by_category(self, category: ActionCategory) -> list[Action]:
        """Get actions in a specific category."""
        return [a for a in self._actions.values() if a.category == category]
    
    def execute(self, action_name: str, mesh: MockMesh, params: dict) -> ActionResult:
        """
        Execute an action on a mesh.
        
        Args:
            action_name: Name of the action to execute.
            mesh: The mesh to process.
            params: Action parameters.
            
        Returns:
            ActionResult with the result of the execution.
        """
        import time
        
        action = self._actions.get(action_name)
        if not action:
            return ActionResult(
                action_name=action_name,
                success=False,
                error=f"Unknown action: {action_name}",
            )
        
        impl = self._implementations.get(action_name)
        if not impl:
            return ActionResult(
                action_name=action_name,
                success=False,
                error=f"No implementation for action: {action_name}",
            )
        
        start = time.perf_counter()
        
        try:
            result_mesh = impl(mesh, params)
            
            diagnostics = compute_diagnostics(result_mesh)
            elapsed = (time.perf_counter() - start) * 1000
            
            return ActionResult(
                action_name=action_name,
                success=True,
                mesh=result_mesh,
                diagnostics=diagnostics,
                message=f"Completed {action.display_name}",
                runtime_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return ActionResult(
                action_name=action_name,
                success=False,
                mesh=mesh,
                error=str(e),
                runtime_ms=elapsed,
            )
    
    # Implementation methods
    def _impl_trimesh_basic(self, mesh: MockMesh, params: dict) -> MockMesh:
        return MockTrimesh.basic_cleanup(mesh)
    
    def _impl_merge_vertices(self, mesh: MockMesh, params: dict) -> MockMesh:
        eps = params.get("eps", 1e-8)
        return MockTrimesh.merge_vertices(mesh, eps=eps)
    
    def _impl_remove_degenerate_faces(self, mesh: MockMesh, params: dict) -> MockMesh:
        return MockTrimesh.remove_degenerate_faces(mesh)
    
    def _impl_fill_holes(self, mesh: MockMesh, params: dict) -> MockMesh:
        max_hole_size = params.get("max_hole_size", 1000)
        return MockTrimesh.fill_holes(mesh, max_hole_size=max_hole_size)
    
    def _impl_recalculate_normals(self, mesh: MockMesh, params: dict) -> MockMesh:
        return MockTrimesh.recalculate_normals(mesh)
    
    def _impl_fix_normals(self, mesh: MockMesh, params: dict) -> MockMesh:
        return MockTrimesh.fix_normals(mesh)
    
    def _impl_flip_normals(self, mesh: MockMesh, params: dict) -> MockMesh:
        mesh = mesh.copy()
        mesh.normal_consistency = 1.0 - mesh.normal_consistency
        mesh.modifications.append("flip_normals")
        return mesh
    
    def _impl_remove_small_components(self, mesh: MockMesh, params: dict) -> MockMesh:
        min_faces = params.get("min_faces", 100)
        return MockTrimesh.remove_small_components(mesh, min_faces=min_faces)
    
    def _impl_boolean_union(self, mesh: MockMesh, params: dict) -> MockMesh:
        return MockBlender.boolean_union(mesh)
    
    def _impl_remove_internal_geometry(self, mesh: MockMesh, params: dict) -> MockMesh:
        mesh = mesh.copy()
        mesh.nested_shell_count = 0
        mesh.modifications.append("remove_internal_geometry")
        return mesh
    
    def _impl_pymeshfix_repair(self, mesh: MockMesh, params: dict) -> MockMesh:
        return MockPyMeshFix.repair(mesh)
    
    def _impl_decimate(self, mesh: MockMesh, params: dict) -> MockMesh:
        target_ratio = params.get("target_ratio", 0.5)
        target_faces = params.get("target_faces")
        return MockTrimesh.decimate(mesh, target_faces=target_faces, target_ratio=target_ratio)
    
    def _impl_smooth_laplacian(self, mesh: MockMesh, params: dict) -> MockMesh:
        iterations = params.get("iterations", 1)
        return MockTrimesh.smooth_laplacian(mesh, iterations=iterations)
    
    def _impl_identify_thin_regions(self, mesh: MockMesh, params: dict) -> MockMesh:
        # Non-destructive: just report
        mesh = mesh.copy()
        mesh.modifications.append(f"identify_thin_regions(min={params.get('min_thickness', 0.8)})")
        return mesh
    
    def _impl_thicken_regions(self, mesh: MockMesh, params: dict) -> MockMesh:
        thickness = params.get("target_thickness", 1.0)
        return MockBlender.solidify(mesh, thickness=thickness)
    
    def _impl_validate(self, mesh: MockMesh, params: dict) -> MockMesh:
        mesh = mesh.copy()
        mesh.modifications.append("validate")
        return mesh
    
    def _impl_compute_diagnostics(self, mesh: MockMesh, params: dict) -> MockMesh:
        mesh = mesh.copy()
        mesh.modifications.append("compute_diagnostics")
        return mesh
    
    def _impl_export_stl(self, mesh: MockMesh, params: dict) -> MockMesh:
        from pathlib import Path
        from .mock_mesh import save_mock_stl
        
        path = params.get("path")
        if path:
            save_mock_stl(mesh, Path(path), ascii_format=params.get("ascii", False))
        mesh = mesh.copy()
        mesh.modifications.append(f"export_stl({path})")
        return mesh
    
    def _impl_blender_remesh(self, mesh: MockMesh, params: dict) -> MockMesh:
        voxel_size = params.get("voxel_size", 0.1)
        return MockBlender.remesh(mesh, voxel_size=voxel_size)
    
    def _impl_blender_boolean_union(self, mesh: MockMesh, params: dict) -> MockMesh:
        return MockBlender.boolean_union(mesh)
    
    def _impl_blender_solidify(self, mesh: MockMesh, params: dict) -> MockMesh:
        thickness = params.get("thickness", 1.0)
        return MockBlender.solidify(mesh, thickness=thickness)


# Global registry instance
_registry: Optional[ActionRegistry] = None


def get_action_registry() -> ActionRegistry:
    """Get the global action registry instance."""
    global _registry
    if _registry is None:
        _registry = ActionRegistry()
    return _registry
