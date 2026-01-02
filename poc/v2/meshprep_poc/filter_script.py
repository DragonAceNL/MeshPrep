# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Filter script loading and execution.

A filter script defines a sequence of repair actions to apply to a mesh.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
import time

import trimesh

from .actions import ActionRegistry

logger = logging.getLogger(__name__)


@dataclass
class FilterAction:
    """A single action in a filter script."""
    name: str
    params: dict = field(default_factory=dict)
    enabled: bool = True


@dataclass
class FilterScript:
    """A complete filter script."""
    name: str
    version: str = "1.0.0"
    description: str = ""
    actions: list[FilterAction] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: dict) -> "FilterScript":
        """Create from dictionary."""
        actions = []
        for action_data in data.get("actions", []):
            actions.append(FilterAction(
                name=action_data.get("name", ""),
                params=action_data.get("params", {}),
                enabled=action_data.get("enabled", True)
            ))
        
        return cls(
            name=data.get("name", "unnamed"),
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            actions=actions,
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def from_json(cls, path: Path) -> "FilterScript":
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "actions": [
                {
                    "name": a.name,
                    "params": a.params,
                    "enabled": a.enabled
                }
                for a in self.actions
            ],
            "metadata": self.metadata
        }
    
    def to_json(self, path: Path) -> None:
        """Save to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class ActionResult:
    """Result of executing a single action."""
    action_name: str
    success: bool
    duration_ms: float
    error: Optional[str] = None
    
    # Mesh stats after action
    vertex_count: int = 0
    face_count: int = 0
    is_watertight: bool = False


@dataclass
class FilterScriptResult:
    """Result of executing a complete filter script."""
    success: bool
    total_duration_ms: float
    action_results: list[ActionResult] = field(default_factory=list)
    final_mesh: Optional[trimesh.Trimesh] = None
    error: Optional[str] = None
    
    @property
    def actions_completed(self) -> int:
        """Number of actions that completed successfully."""
        return sum(1 for r in self.action_results if r.success)
    
    @property
    def actions_total(self) -> int:
        """Total number of actions attempted."""
        return len(self.action_results)


class FilterScriptRunner:
    """Executes filter scripts on meshes."""
    
    def __init__(self, stop_on_error: bool = True):
        """
        Initialize runner.
        
        Args:
            stop_on_error: If True, stop execution on first error
        """
        self.stop_on_error = stop_on_error
    
    def run(
        self,
        script: FilterScript,
        mesh: trimesh.Trimesh,
        progress_callback: Optional[callable] = None
    ) -> FilterScriptResult:
        """
        Execute a filter script on a mesh.
        
        Args:
            script: The filter script to execute
            mesh: Input mesh
            progress_callback: Optional callback(action_index, action_name, total)
            
        Returns:
            FilterScriptResult with final mesh and execution details
        """
        result = FilterScriptResult(
            success=True,
            total_duration_ms=0.0,
            action_results=[],
            final_mesh=None
        )
        
        current_mesh = mesh.copy()
        total_start = time.perf_counter()
        
        enabled_actions = [a for a in script.actions if a.enabled]
        total_actions = len(enabled_actions)
        
        logger.info(f"Running filter script: {script.name} ({total_actions} actions)")
        
        for i, action in enumerate(enabled_actions):
            if progress_callback:
                progress_callback(i, action.name, total_actions)
            
            action_result = self._run_action(action, current_mesh)
            result.action_results.append(action_result)
            
            if action_result.success:
                # Get the modified mesh from the action
                current_mesh = ActionRegistry.execute(
                    action.name, current_mesh, action.params
                )
                logger.info(
                    f"  [{i+1}/{total_actions}] {action.name}: OK "
                    f"(v={action_result.vertex_count}, f={action_result.face_count})"
                )
            else:
                logger.error(f"  [{i+1}/{total_actions}] {action.name}: FAILED - {action_result.error}")
                
                if self.stop_on_error:
                    result.success = False
                    result.error = f"Action '{action.name}' failed: {action_result.error}"
                    break
        
        total_end = time.perf_counter()
        result.total_duration_ms = (total_end - total_start) * 1000
        result.final_mesh = current_mesh
        
        if result.success:
            logger.info(f"Filter script completed in {result.total_duration_ms:.1f}ms")
        
        return result
    
    def _run_action(
        self,
        action: FilterAction,
        mesh: trimesh.Trimesh
    ) -> ActionResult:
        """Run a single action and return the result."""
        start_time = time.perf_counter()
        
        try:
            # Execute the action
            result_mesh = ActionRegistry.execute(action.name, mesh, action.params)
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            return ActionResult(
                action_name=action.name,
                success=True,
                duration_ms=duration_ms,
                vertex_count=len(result_mesh.vertices),
                face_count=len(result_mesh.faces),
                is_watertight=result_mesh.is_watertight
            )
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            return ActionResult(
                action_name=action.name,
                success=False,
                duration_ms=duration_ms,
                error=str(e)
            )


def create_filter_script(
    name: str,
    actions: list[tuple[str, dict]],
    description: str = ""
) -> FilterScript:
    """
    Create a filter script from a list of actions.
    
    Args:
        name: Script name
        actions: List of (action_name, params) tuples
        description: Script description
        
    Returns:
        FilterScript object
    """
    return FilterScript(
        name=name,
        description=description,
        actions=[
            FilterAction(name=action_name, params=params)
            for action_name, params in actions
        ]
    )


# Preset filter scripts
PRESET_BASIC_CLEANUP = create_filter_script(
    name="basic-cleanup",
    description="Basic cleanup: merge vertices, remove degenerates, fix normals",
    actions=[
        ("trimesh_basic", {}),
        ("fix_normals", {}),
        ("validate", {}),
    ]
)

PRESET_FILL_HOLES = create_filter_script(
    name="fill-holes",
    description="Fill holes and fix normals",
    actions=[
        ("trimesh_basic", {}),
        ("fill_holes", {}),
        ("fix_normals", {}),
        ("validate", {}),
    ]
)

PRESET_FULL_REPAIR = create_filter_script(
    name="full-repair",
    description="Full repair using PyMeshFix",
    actions=[
        ("trimesh_basic", {}),
        ("pymeshfix_repair", {"joincomp": True}),
        ("fix_normals", {}),
        ("validate", {}),
    ]
)

PRESET_MANIFOLD_REPAIR = create_filter_script(
    name="manifold-repair",
    description="Make mesh manifold",
    actions=[
        ("trimesh_basic", {}),
        ("make_manifold", {}),
        ("fix_normals", {}),
        ("validate", {}),
    ]
)

# All presets
PRESETS = {
    "basic-cleanup": PRESET_BASIC_CLEANUP,
    "fill-holes": PRESET_FILL_HOLES,
    "full-repair": PRESET_FULL_REPAIR,
    "manifold-repair": PRESET_MANIFOLD_REPAIR,
}


def get_preset(name: str) -> Optional[FilterScript]:
    """Get a preset filter script by name."""
    return PRESETS.get(name)


def list_presets() -> list[str]:
    """List all preset names."""
    return list(PRESETS.keys())
