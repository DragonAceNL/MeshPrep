# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Filter script loading, execution, and preset management.

A filter script defines a sequence of repair actions to apply to a mesh.
Filter scripts are the primary user-editable unit for defining repair workflows.

See docs/functional_spec.md for the complete filter script specification.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Any, Union

import trimesh

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from .actions import ActionRegistry

logger = logging.getLogger(__name__)


@dataclass
class FilterAction:
    """
    A single action in a filter script.
    
    Attributes:
        name: Action name (must be registered in ActionRegistry)
        params: Action parameters
        id: Optional stable ID for diagnostics/reporting
        enabled: Whether this action is enabled
        on_error: Error policy: "abort", "skip", or "continue"
        timeout: Optional timeout in seconds
    """
    name: str
    params: dict = field(default_factory=dict)
    id: Optional[str] = None
    enabled: bool = True
    on_error: str = "abort"
    timeout: Optional[float] = None


@dataclass
class FilterScriptMeta:
    """
    Metadata for a filter script.
    
    Tracks provenance and authorship information.
    """
    generated_by: str = "user"
    model_fingerprint: Optional[str] = None
    generator_version: str = "0.1.0"
    timestamp: Optional[str] = None
    author: str = "user"
    description: str = ""
    source: str = "local"
    tags: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "generated_by": self.generated_by,
            "model_fingerprint": self.model_fingerprint,
            "generator_version": self.generator_version,
            "timestamp": self.timestamp or datetime.now().isoformat(),
            "author": self.author,
            "description": self.description,
            "source": self.source,
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "FilterScriptMeta":
        return cls(
            generated_by=data.get("generated_by", "user"),
            model_fingerprint=data.get("model_fingerprint"),
            generator_version=data.get("generator_version", "0.1.0"),
            timestamp=data.get("timestamp"),
            author=data.get("author", "user"),
            description=data.get("description", ""),
            source=data.get("source", "local"),
            tags=data.get("tags", []),
        )


@dataclass
class FilterScript:
    """
    A complete filter script defining a repair workflow.
    
    Filter scripts are JSON/YAML documents that describe an ordered
    list of actions to run against a mesh.
    """
    name: str
    version: str = "1.0.0"
    actions: list[FilterAction] = field(default_factory=list)
    meta: FilterScriptMeta = field(default_factory=FilterScriptMeta)
    
    @classmethod
    def from_dict(cls, data: dict) -> "FilterScript":
        """Create from dictionary."""
        actions = []
        for action_data in data.get("actions", []):
            actions.append(FilterAction(
                name=action_data.get("name", ""),
                params=action_data.get("params", {}),
                id=action_data.get("id"),
                enabled=action_data.get("enabled", True),
                on_error=action_data.get("on_error", "abort"),
                timeout=action_data.get("timeout"),
            ))
        
        meta_data = data.get("meta", data.get("metadata", {}))
        meta = FilterScriptMeta.from_dict(meta_data)
        
        # Use description from top-level or meta
        if "description" in data and not meta.description:
            meta.description = data["description"]
        
        return cls(
            name=data.get("name", "unnamed"),
            version=data.get("version", "1.0.0"),
            actions=actions,
            meta=meta,
        )
    
    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "FilterScript":
        """Load from JSON file."""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        script = cls.from_dict(data)
        if script.meta.source == "local" and not script.meta.description:
            script.meta.description = f"Loaded from {path.name}"
        return script
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "FilterScript":
        """Load from YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required to load YAML filter scripts")
        
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        script = cls.from_dict(data)
        if script.meta.source == "local" and not script.meta.description:
            script.meta.description = f"Loaded from {path.name}"
        return script
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "FilterScript":
        """
        Load from file, auto-detecting format from extension.
        
        Supports .json and .yaml/.yml files.
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Filter script not found: {path}")
        
        suffix = path.suffix.lower()
        
        if suffix == ".json":
            return cls.from_json(path)
        elif suffix in (".yaml", ".yml"):
            return cls.from_yaml(path)
        else:
            # Try JSON first, then YAML
            try:
                return cls.from_json(path)
            except json.JSONDecodeError:
                if YAML_AVAILABLE:
                    return cls.from_yaml(path)
                raise ValueError(f"Unknown filter script format: {path}")
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "meta": self.meta.to_dict(),
            "actions": [
                {
                    "id": a.id,
                    "name": a.name,
                    "params": a.params,
                    "enabled": a.enabled,
                    "on_error": a.on_error,
                    "timeout": a.timeout,
                }
                for a in self.actions
            ],
        }
    
    def to_json(self, path: Union[str, Path], indent: int = 2) -> None:
        """Save to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent)
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save to YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required to save YAML filter scripts")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def validate(self) -> list[str]:
        """
        Validate the filter script.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if not self.name:
            errors.append("Filter script must have a name")
        
        for i, action in enumerate(self.actions):
            if not action.name:
                errors.append(f"Action {i}: missing action name")
            elif not ActionRegistry.exists(action.name):
                errors.append(f"Action {i}: unknown action '{action.name}'")
            
            if action.on_error not in ("abort", "skip", "continue"):
                errors.append(f"Action {i}: invalid on_error value '{action.on_error}'")
        
        return errors
    
    @property
    def description(self) -> str:
        """Convenience accessor for description."""
        return self.meta.description


@dataclass
class ActionResult:
    """Result of executing a single action."""
    action_name: str
    action_id: Optional[str]
    success: bool
    duration_ms: float
    error: Optional[str] = None
    
    # Mesh stats after action
    vertex_count: int = 0
    face_count: int = 0
    is_watertight: bool = False
    
    def to_dict(self) -> dict:
        return {
            "action_name": self.action_name,
            "action_id": self.action_id,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "is_watertight": self.is_watertight,
        }


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
    
    def to_dict(self) -> dict:
        """Convert to dictionary (excluding mesh)."""
        return {
            "success": self.success,
            "total_duration_ms": self.total_duration_ms,
            "actions_completed": self.actions_completed,
            "actions_total": self.actions_total,
            "error": self.error,
            "action_results": [r.to_dict() for r in self.action_results],
        }


class FilterScriptRunner:
    """
    Executes filter scripts on meshes.
    
    The runner processes each action in sequence, tracking results
    and handling errors according to each action's on_error policy.
    """
    
    def __init__(self, stop_on_error: bool = True):
        """
        Initialize runner.
        
        Args:
            stop_on_error: Default behavior when action fails (can be overridden per-action)
        """
        self.stop_on_error = stop_on_error
    
    def run(
        self,
        script: FilterScript,
        mesh: trimesh.Trimesh,
        progress_callback: Optional[Callable[[int, str, int], None]] = None
    ) -> FilterScriptResult:
        """
        Execute a filter script on a mesh.
        
        Args:
            script: The filter script to execute
            mesh: Input mesh
            progress_callback: Optional callback(action_index, action_name, total_actions)
            
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
            
            logger.info(f"Executing action: {action.name} with params: {action.params}")
            
            action_result = self._run_action(action, current_mesh)
            result.action_results.append(action_result)
            
            if action_result.success:
                # Update current mesh from successful action
                try:
                    current_mesh = ActionRegistry.execute(
                        action.name, current_mesh, action.params
                    )
                except Exception:
                    pass  # Already captured in action_result
                
                logger.info(
                    f"  [{i+1}/{total_actions}] {action.name}: OK "
                    f"(v={action_result.vertex_count}, f={action_result.face_count})"
                )
            else:
                logger.error(
                    f"  [{i+1}/{total_actions}] {action.name}: FAILED - {action_result.error}"
                )
                
                # Handle error according to policy
                if action.on_error == "abort":
                    result.success = False
                    result.error = f"Action '{action.name}' failed: {action_result.error}"
                    break
                elif action.on_error == "skip":
                    # Continue with current mesh
                    logger.info(f"  Skipping failed action (on_error=skip)")
                # "continue" also just continues
        
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
                action_id=action.id,
                success=True,
                duration_ms=duration_ms,
                vertex_count=len(result_mesh.vertices),
                face_count=len(result_mesh.faces),
                is_watertight=bool(result_mesh.is_watertight)
            )
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            return ActionResult(
                action_name=action.name,
                action_id=action.id,
                success=False,
                duration_ms=duration_ms,
                error=str(e)
            )


def create_filter_script(
    name: str,
    actions: list[tuple[str, dict]],
    description: str = "",
    author: str = "user"
) -> FilterScript:
    """
    Create a filter script from a list of actions.
    
    Convenience function for creating scripts programmatically.
    
    Args:
        name: Script name
        actions: List of (action_name, params) tuples
        description: Script description
        author: Author name
        
    Returns:
        FilterScript object
    """
    return FilterScript(
        name=name,
        actions=[
            FilterAction(name=action_name, params=params)
            for action_name, params in actions
        ],
        meta=FilterScriptMeta(
            description=description,
            author=author,
            timestamp=datetime.now().isoformat(),
        )
    )


# =============================================================================
# Built-in Presets
# =============================================================================

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
    description="Fill holes using pymeshfix for robust repairs (falls back to trimesh)",
    actions=[
        ("trimesh_basic", {}),
        ("fill_holes_pymeshfix", {}),
        ("fix_normals", {}),
        ("validate", {}),
    ]
)

PRESET_FULL_REPAIR = create_filter_script(
    name="full-repair",
    description="Full repair using PyMeshFix for robust hole filling and manifold fixes",
    actions=[
        ("trimesh_basic", {}),
        ("pymeshfix_repair", {"joincomp": True}),
        ("fix_normals", {}),
        ("validate", {}),
    ]
)

PRESET_MANIFOLD_REPAIR = create_filter_script(
    name="manifold-repair",
    description="Make mesh manifold for 3D printing",
    actions=[
        ("trimesh_basic", {}),
        ("make_manifold", {}),
        ("fix_normals", {}),
        ("validate", {}),
    ]
)

PRESET_AGGRESSIVE = create_filter_script(
    name="aggressive-repair",
    description="Aggressive repair: full cleanup, hole fill, manifold fix, and component cleanup",
    actions=[
        ("trimesh_basic", {}),
        ("pymeshfix_repair", {"joincomp": True}),
        ("remove_small_components", {"min_faces": 100}),
        ("fix_normals", {}),
        ("validate", {}),
    ]
)

# All built-in presets
PRESETS: dict[str, FilterScript] = {
    "basic-cleanup": PRESET_BASIC_CLEANUP,
    "fill-holes": PRESET_FILL_HOLES,
    "full-repair": PRESET_FULL_REPAIR,
    "manifold-repair": PRESET_MANIFOLD_REPAIR,
    "aggressive-repair": PRESET_AGGRESSIVE,
}


def get_preset(name: str) -> Optional[FilterScript]:
    """Get a built-in preset filter script by name."""
    return PRESETS.get(name)


def list_presets() -> list[str]:
    """List all built-in preset names."""
    return list(PRESETS.keys())


def load_presets_from_directory(directory: Union[str, Path]) -> dict[str, FilterScript]:
    """
    Load all filter scripts from a directory.
    
    Args:
        directory: Path to directory containing .json/.yaml filter scripts
        
    Returns:
        Dictionary mapping script name to FilterScript
    """
    directory = Path(directory)
    scripts = {}
    
    if not directory.exists():
        return scripts
    
    for path in directory.iterdir():
        if path.suffix.lower() in (".json", ".yaml", ".yml"):
            try:
                script = FilterScript.load(path)
                scripts[script.name] = script
                logger.debug(f"Loaded preset: {script.name} from {path}")
            except Exception as e:
                logger.warning(f"Failed to load preset from {path}: {e}")
    
    return scripts
