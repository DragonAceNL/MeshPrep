# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Action registry for mesh repair operations.

The registry maps action names to implementations, allowing filter
scripts to reference actions by name. Each action includes metadata
for documentation and the filter script editor.
"""

from typing import Callable, Any, Optional
from dataclasses import dataclass, field
import logging

import trimesh

logger = logging.getLogger(__name__)


@dataclass
class ActionDefinition:
    """
    Definition of a registered action.
    
    Attributes:
        name: Unique action identifier used in filter scripts
        func: The function that implements the action
        description: Human-readable description of what the action does
        parameters: Dictionary of parameter definitions with defaults
        risk_level: How likely the action is to modify the mesh significantly
        tool: Which tool provides this action (trimesh, pymeshfix, blender, internal)
        category: Category for organizing in the filter library
    """
    name: str
    func: Callable[[trimesh.Trimesh, dict], trimesh.Trimesh]
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    risk_level: str = "low"  # low, medium, high
    tool: str = "internal"
    category: str = "General"


class ActionRegistry:
    """
    Registry for mesh repair actions.
    
    Actions are registered at import time using the @register_action
    decorator. Filter scripts reference actions by name only.
    
    Example:
        @register_action(
            name="fill_holes",
            description="Fill holes in the mesh",
            tool="trimesh"
        )
        def action_fill_holes(mesh, params):
            trimesh.repair.fill_holes(mesh)
            return mesh
    """
    
    _actions: dict[str, ActionDefinition] = {}
    
    @classmethod
    def register(
        cls,
        name: str,
        func: Callable[[trimesh.Trimesh, dict], trimesh.Trimesh],
        description: str = "",
        parameters: Optional[dict] = None,
        risk_level: str = "low",
        tool: str = "internal",
        category: str = "General"
    ) -> None:
        """
        Register an action.
        
        Args:
            name: Unique action identifier
            func: Implementation function
            description: Human-readable description
            parameters: Parameter definitions with defaults
            risk_level: low, medium, or high
            tool: trimesh, pymeshfix, blender, or internal
            category: Category for filter library
        """
        cls._actions[name] = ActionDefinition(
            name=name,
            func=func,
            description=description,
            parameters=parameters or {},
            risk_level=risk_level,
            tool=tool,
            category=category
        )
        logger.debug(f"Registered action: {name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[ActionDefinition]:
        """Get an action by name."""
        return cls._actions.get(name)
    
    @classmethod
    def exists(cls, name: str) -> bool:
        """Check if an action exists."""
        return name in cls._actions
    
    @classmethod
    def execute(
        cls,
        name: str,
        mesh: trimesh.Trimesh,
        params: Optional[dict] = None
    ) -> trimesh.Trimesh:
        """
        Execute an action on a mesh.
        
        Args:
            name: Action name
            mesh: Input mesh
            params: Action parameters (optional)
            
        Returns:
            Modified mesh
            
        Raises:
            ValueError: If action not found
            RuntimeError: If action execution fails
        """
        action = cls.get(name)
        if action is None:
            raise ValueError(f"Unknown action: {name}")
        
        params = params or {}
        logger.info(f"Executing action: {name}")
        logger.debug(f"  Parameters: {params}")
        
        try:
            result = action.func(mesh, params)
            return result
        except Exception as e:
            logger.error(f"Action {name} failed: {e}")
            raise RuntimeError(f"Action '{name}' failed: {e}") from e
    
    @classmethod
    def list_actions(cls) -> list[str]:
        """List all registered action names."""
        return list(cls._actions.keys())
    
    @classmethod
    def get_all(cls) -> dict[str, ActionDefinition]:
        """Get all registered actions."""
        return cls._actions.copy()
    
    @classmethod
    def get_by_category(cls) -> dict[str, list[ActionDefinition]]:
        """Get actions grouped by category."""
        categories: dict[str, list[ActionDefinition]] = {}
        for action in cls._actions.values():
            if action.category not in categories:
                categories[action.category] = []
            categories[action.category].append(action)
        return categories
    
    @classmethod
    def get_by_tool(cls, tool: str) -> list[ActionDefinition]:
        """Get all actions from a specific tool."""
        return [a for a in cls._actions.values() if a.tool == tool]


def register_action(
    name: str,
    description: str = "",
    parameters: Optional[dict] = None,
    risk_level: str = "low",
    tool: str = "internal",
    category: str = "General"
):
    """
    Decorator to register an action function.
    
    Example:
        @register_action(
            name="fill_holes",
            description="Fill holes in the mesh",
            parameters={"max_hole_size": 1000},
            tool="trimesh",
            category="Hole Filling"
        )
        def action_fill_holes(mesh, params):
            ...
    """
    def decorator(func: Callable[[trimesh.Trimesh, dict], trimesh.Trimesh]):
        ActionRegistry.register(
            name=name,
            func=func,
            description=description,
            parameters=parameters,
            risk_level=risk_level,
            tool=tool,
            category=category
        )
        return func
    return decorator
