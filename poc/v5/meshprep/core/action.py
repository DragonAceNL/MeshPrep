# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Base Action class for all repair operations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum
import logging
import time

from .mesh import Mesh

logger = logging.getLogger(__name__)


class ActionRiskLevel(Enum):
    """Risk level of a repair action."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ActionResult:
    """Result of executing an action."""
    success: bool
    mesh: Optional[Mesh] = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Action(ABC):
    """Abstract base class for all repair actions."""
    
    name: str = "base_action"
    description: str = "Base action"
    risk_level: ActionRiskLevel = ActionRiskLevel.MEDIUM
    default_params: Dict[str, Any] = {}
    
    def __init__(self):
        self.logger = logging.getLogger(f"meshprep.actions.{self.name}")
    
    @abstractmethod
    def execute(self, mesh: Mesh, params: Optional[Dict[str, Any]] = None) -> Mesh:
        """Execute the repair action on a mesh."""
        pass
    
    def get_params(self, user_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Merge user parameters with defaults."""
        params = self.default_params.copy()
        if user_params:
            params.update(user_params)
        return params
    
    def run(self, mesh: Mesh, params: Optional[Dict[str, Any]] = None) -> ActionResult:
        """Run the action with error handling and timing."""
        params = self.get_params(params)
        self.logger.info(f"Executing {self.name}")
        
        start = time.perf_counter()
        
        try:
            result_mesh = self.execute(mesh, params)
            duration = (time.perf_counter() - start) * 1000
            
            return ActionResult(
                success=True,
                mesh=result_mesh,
                duration_ms=duration,
                metadata={"action": self.name, "params": params},
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            self.logger.error(f"Action failed: {e}")
            
            return ActionResult(
                success=False,
                mesh=mesh,
                error=str(e),
                duration_ms=duration,
                metadata={"action": self.name, "params": params},
            )


class ActionRegistry:
    """Registry of all available actions."""
    
    _actions: Dict[str, Action] = {}
    
    @classmethod
    def register(cls, action: Action):
        """Register an action."""
        if action.name in cls._actions:
            logger.warning(f"Action '{action.name}' already registered")
        cls._actions[action.name] = action
        logger.debug(f"Registered action: {action.name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[Action]:
        """Get an action by name."""
        return cls._actions.get(name)
    
    @classmethod
    def list_actions(cls) -> Dict[str, Action]:
        """Get all registered actions."""
        return cls._actions.copy()
    
    @classmethod
    def execute(cls, action_name: str, mesh: Mesh, params: Optional[Dict[str, Any]] = None) -> ActionResult:
        """Execute an action by name."""
        action = cls.get(action_name)
        if action is None:
            return ActionResult(success=False, error=f"Action '{action_name}' not found")
        return action.run(mesh, params)


def register_action(action_class):
    """Decorator to auto-register actions."""
    action = action_class()
    ActionRegistry.register(action)
    return action_class
