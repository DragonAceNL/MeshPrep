# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Action registry for mesh repair operations.
"""

from typing import Callable, Any, Optional
from dataclasses import dataclass, field
import trimesh
import logging

logger = logging.getLogger(__name__)


@dataclass
class ActionDefinition:
    """Definition of a registered action."""
    name: str
    func: Callable[[trimesh.Trimesh, dict], trimesh.Trimesh]
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    risk_level: str = "low"  # low, medium, high


class ActionRegistry:
    """Registry for mesh repair actions."""
    
    _actions: dict[str, ActionDefinition] = {}
    
    @classmethod
    def register(
        cls,
        name: str,
        func: Callable[[trimesh.Trimesh, dict], trimesh.Trimesh],
        description: str = "",
        parameters: Optional[dict] = None,
        risk_level: str = "low"
    ) -> None:
        """Register an action."""
        cls._actions[name] = ActionDefinition(
            name=name,
            func=func,
            description=description,
            parameters=parameters or {},
            risk_level=risk_level
        )
        logger.debug(f"Registered action: {name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[ActionDefinition]:
        """Get an action by name."""
        return cls._actions.get(name)
    
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
            params: Action parameters
            
        Returns:
            Modified mesh
            
        Raises:
            ValueError: If action not found
        """
        action = cls.get(name)
        if action is None:
            raise ValueError(f"Unknown action: {name}")
        
        params = params or {}
        logger.info(f"Executing action: {name} with params: {params}")
        
        try:
            result = action.func(mesh, params)
            return result
        except Exception as e:
            logger.error(f"Action {name} failed: {e}")
            raise
    
    @classmethod
    def list_actions(cls) -> list[str]:
        """List all registered action names."""
        return list(cls._actions.keys())
    
    @classmethod
    def get_all(cls) -> dict[str, ActionDefinition]:
        """Get all registered actions."""
        return cls._actions.copy()


def register_action(
    name: str,
    description: str = "",
    parameters: Optional[dict] = None,
    risk_level: str = "low"
):
    """Decorator to register an action function."""
    def decorator(func: Callable[[trimesh.Trimesh, dict], trimesh.Trimesh]):
        ActionRegistry.register(
            name=name,
            func=func,
            description=description,
            parameters=parameters,
            risk_level=risk_level
        )
        return func
    return decorator
