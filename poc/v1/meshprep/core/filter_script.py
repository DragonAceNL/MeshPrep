# Copyright 2025 Dragon Ace
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""Filter script representation and execution."""

import json
import yaml
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable
import hashlib

from .mock_mesh import MockMesh
from .diagnostics import Diagnostics, compute_diagnostics
from .actions import ActionRegistry, ActionResult, get_action_registry


@dataclass
class FilterAction:
    """A single action in a filter script."""
    
    name: str
    params: dict = field(default_factory=dict)
    id: Optional[str] = None
    timeout: Optional[int] = None
    on_error: str = "abort"  # "abort", "skip", "continue"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = {"name": self.name}
        if self.params:
            d["params"] = self.params
        if self.id:
            d["id"] = self.id
        if self.timeout:
            d["timeout"] = self.timeout
        if self.on_error != "abort":
            d["on_error"] = self.on_error
        return d
    
    @classmethod
    def from_dict(cls, data: dict) -> "FilterAction":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            params=data.get("params", {}),
            id=data.get("id"),
            timeout=data.get("timeout"),
            on_error=data.get("on_error", "abort"),
        )


@dataclass
class FilterScriptMeta:
    """Metadata for a filter script."""
    
    generated_by: str = "user"
    model_fingerprint: Optional[str] = None
    generator_version: str = "0.1.0"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    author: str = "user"
    description: str = ""
    source: str = "local"  # "local", "url", "community"
    tags: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "FilterScriptMeta":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class FilterScript:
    """A filter script defining a sequence of repair actions."""
    
    name: str
    version: str = "1.0.0"
    meta: FilterScriptMeta = field(default_factory=FilterScriptMeta)
    actions: list[FilterAction] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "meta": self.meta.to_dict(),
            "actions": [a.to_dict() for a in self.actions],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "FilterScript":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            version=data.get("version", "1.0.0"),
            meta=FilterScriptMeta.from_dict(data.get("meta", {})),
            actions=[FilterAction.from_dict(a) for a in data.get("actions", [])],
        )
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> "FilterScript":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def to_yaml(self) -> str:
        """Convert to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> "FilterScript":
        """Create from YAML string."""
        return cls.from_dict(yaml.safe_load(yaml_str))
    
    def save(self, path: Path):
        """Save filter script to file (JSON or YAML based on extension)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix.lower() in [".yaml", ".yml"]:
            path.write_text(self.to_yaml())
        else:
            path.write_text(self.to_json())
    
    @classmethod
    def load(cls, path: Path) -> "FilterScript":
        """Load filter script from file."""
        path = Path(path)
        content = path.read_text()
        
        if path.suffix.lower() in [".yaml", ".yml"]:
            return cls.from_yaml(content)
        else:
            return cls.from_json(content)
    
    def add_action(self, name: str, params: Optional[dict] = None, 
                   index: Optional[int] = None) -> "FilterScript":
        """Add an action to the script."""
        action = FilterAction(name=name, params=params or {})
        if index is not None:
            self.actions.insert(index, action)
        else:
            self.actions.append(action)
        return self
    
    def remove_action(self, index: int) -> "FilterScript":
        """Remove an action by index."""
        if 0 <= index < len(self.actions):
            self.actions.pop(index)
        return self
    
    def move_action(self, from_index: int, to_index: int) -> "FilterScript":
        """Move an action from one position to another."""
        if 0 <= from_index < len(self.actions) and 0 <= to_index < len(self.actions):
            action = self.actions.pop(from_index)
            self.actions.insert(to_index, action)
        return self
    
    def validate(self, registry: Optional[ActionRegistry] = None) -> list[str]:
        """
        Validate the filter script.
        
        Returns:
            List of validation errors (empty if valid).
        """
        if registry is None:
            registry = get_action_registry()
        
        errors = []
        
        if not self.name:
            errors.append("Filter script name is required")
        
        if not self.actions:
            errors.append("Filter script must have at least one action")
        
        for i, action in enumerate(self.actions):
            action_def = registry.get(action.name)
            if not action_def:
                errors.append(f"Action {i+1}: Unknown action '{action.name}'")
                continue
            
            # Validate parameters
            for param in action_def.parameters:
                if param.required and param.name not in action.params:
                    errors.append(f"Action {i+1} ({action.name}): Missing required parameter '{param.name}'")
        
        return errors


@dataclass
class StepResult:
    """Result of a single step in filter script execution."""
    
    step_number: int
    action_name: str
    status: str  # "pending", "running", "success", "warning", "error", "skipped"
    message: str = ""
    diagnostics: Optional[Diagnostics] = None
    runtime_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class RunResult:
    """Result of a complete filter script run."""
    
    success: bool
    script_name: str
    total_runtime_ms: float = 0.0
    steps: list[StepResult] = field(default_factory=list)
    initial_diagnostics: Optional[Diagnostics] = None
    final_diagnostics: Optional[Diagnostics] = None
    final_mesh: Optional[MockMesh] = None
    error: Optional[str] = None
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"=== Run Result: {self.script_name} ===",
            f"Status: {'SUCCESS' if self.success else 'FAILED'}",
            f"Total runtime: {self.total_runtime_ms:.1f}ms",
            f"Steps: {len(self.steps)}",
            "",
        ]
        
        for step in self.steps:
            status_icon = {
                "success": "✓",
                "warning": "⚠",
                "error": "✗",
                "skipped": "○",
                "pending": "·",
                "running": "►",
            }.get(step.status, "?")
            lines.append(f"  {status_icon} Step {step.step_number}: {step.action_name} ({step.runtime_ms:.1f}ms)")
            if step.error:
                lines.append(f"      Error: {step.error}")
        
        if self.final_diagnostics:
            lines.append("")
            lines.append("Final state:")
            lines.append(f"  Watertight: {'✓' if self.final_diagnostics.is_watertight else '✗'}")
            lines.append(f"  Issues: {len(self.final_diagnostics.issues())}")
        
        return "\n".join(lines)


class FilterScriptRunner:
    """Executes filter scripts on meshes."""
    
    def __init__(self, registry: Optional[ActionRegistry] = None):
        """Initialize runner with action registry."""
        self.registry = registry or get_action_registry()
        self._progress_callback: Optional[Callable[[int, int, str], None]] = None
        self._cancelled = False
    
    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        """Set callback for progress updates: callback(current_step, total_steps, message)."""
        self._progress_callback = callback
    
    def cancel(self):
        """Cancel the current run."""
        self._cancelled = True
    
    def _report_progress(self, step: int, total: int, message: str):
        """Report progress to callback if set."""
        if self._progress_callback:
            self._progress_callback(step, total, message)
    
    def run(self, script: FilterScript, mesh: MockMesh) -> RunResult:
        """
        Execute a filter script on a mesh.
        
        Args:
            script: The filter script to execute.
            mesh: The input mesh.
            
        Returns:
            RunResult with the execution results.
        """
        import time
        
        self._cancelled = False
        start_time = time.perf_counter()
        
        # Compute initial diagnostics
        initial_diag = compute_diagnostics(mesh)
        
        steps: list[StepResult] = []
        current_mesh = mesh
        total_steps = len(script.actions)
        
        for i, action in enumerate(script.actions):
            if self._cancelled:
                steps.append(StepResult(
                    step_number=i + 1,
                    action_name=action.name,
                    status="skipped",
                    message="Cancelled by user",
                ))
                continue
            
            self._report_progress(i + 1, total_steps, f"Running {action.name}...")
            
            # Execute action
            result = self.registry.execute(
                action.name,
                current_mesh,
                action.params,
            )
            
            if result.success:
                current_mesh = result.mesh
                steps.append(StepResult(
                    step_number=i + 1,
                    action_name=action.name,
                    status="success",
                    message=result.message,
                    diagnostics=result.diagnostics,
                    runtime_ms=result.runtime_ms,
                ))
            else:
                steps.append(StepResult(
                    step_number=i + 1,
                    action_name=action.name,
                    status="error",
                    message=result.message,
                    error=result.error,
                    runtime_ms=result.runtime_ms,
                ))
                
                # Handle error based on on_error policy
                if action.on_error == "abort":
                    # Mark remaining steps as skipped
                    for j in range(i + 1, len(script.actions)):
                        steps.append(StepResult(
                            step_number=j + 1,
                            action_name=script.actions[j].name,
                            status="skipped",
                            message="Skipped due to previous error",
                        ))
                    break
                elif action.on_error == "skip":
                    continue
                # "continue" keeps going with the current mesh
        
        total_runtime = (time.perf_counter() - start_time) * 1000
        final_diag = compute_diagnostics(current_mesh) if current_mesh else None
        
        # Determine overall success
        has_errors = any(s.status == "error" for s in steps)
        success = not has_errors
        
        self._report_progress(total_steps, total_steps, "Complete")
        
        return RunResult(
            success=success,
            script_name=script.name,
            total_runtime_ms=total_runtime,
            steps=steps,
            initial_diagnostics=initial_diag,
            final_diagnostics=final_diag,
            final_mesh=current_mesh,
            error=steps[-1].error if steps and steps[-1].status == "error" else None,
        )


def generate_filter_script(profile_name: str, model_fingerprint: str,
                          suggested_actions: list[str]) -> FilterScript:
    """
    Generate a filter script from a profile.
    
    Args:
        profile_name: Name of the detected profile.
        model_fingerprint: Fingerprint of the model.
        suggested_actions: List of suggested action names.
        
    Returns:
        Generated FilterScript.
    """
    return FilterScript(
        name=f"{profile_name}-suggested",
        version="1.0.0",
        meta=FilterScriptMeta(
            generated_by="model_scan",
            model_fingerprint=model_fingerprint,
            generator_version="0.1.0",
            description=f"Auto-generated filter script for {profile_name} profile",
            source="local",
        ),
        actions=[FilterAction(name=action) for action in suggested_actions],
    )
