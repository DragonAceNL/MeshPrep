# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Pipeline orchestration for sequences of repair actions."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import logging
import time

from .mesh import Mesh
from .action import ActionRegistry, ActionResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of executing a pipeline."""
    success: bool
    mesh: Optional[Mesh] = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    action_results: List[ActionResult] = field(default_factory=list)
    pipeline_name: str = ""
    actions_executed: int = 0


class Pipeline:
    """Sequence of repair actions executed in order."""
    
    def __init__(self, name: str, actions: List[Dict[str, Any]], description: str = ""):
        self.name = name
        self.actions = actions
        self.description = description
        self.logger = logging.getLogger(f"meshprep.pipeline.{name}")
    
    def execute(self, mesh: Mesh, stop_on_failure: bool = True) -> PipelineResult:
        """Execute the pipeline on a mesh."""
        self.logger.info(f"Starting pipeline '{self.name}'")
        
        start = time.perf_counter()
        result = PipelineResult(success=False, pipeline_name=self.name)
        current_mesh = mesh
        
        for idx, action_def in enumerate(self.actions, 1):
            action_name = action_def.get("name")
            action_params = action_def.get("params", {})
            
            if not action_name:
                result.error = f"Action {idx} missing 'name' field"
                break
            
            self.logger.info(f"  [{idx}/{len(self.actions)}] {action_name}")
            
            action_result = ActionRegistry.execute(action_name, current_mesh, action_params)
            result.action_results.append(action_result)
            result.actions_executed = idx
            
            if not action_result.success:
                self.logger.warning(f"  [X] Failed: {action_result.error}")
                if stop_on_failure:
                    result.error = f"Action '{action_name}' failed: {action_result.error}"
                    break
            else:
                current_mesh = action_result.mesh
        
        if result.actions_executed == len(self.actions):
            if all(r.success for r in result.action_results):
                result.success = True
                result.mesh = current_mesh
                self.logger.info(f"Pipeline '{self.name}' completed successfully")
        
        result.duration_ms = (time.perf_counter() - start) * 1000
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pipeline":
        """Create pipeline from dictionary."""
        return cls(
            name=data["name"],
            actions=data["actions"],
            description=data.get("description", ""),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "actions": self.actions,
        }
