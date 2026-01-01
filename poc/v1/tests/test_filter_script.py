# Copyright 2025 Dragon Ace
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""Tests for filter scripts and execution."""

import pytest
import json
from pathlib import Path

from meshprep.core.mock_mesh import MockMesh
from meshprep.core.filter_script import (
    FilterScript, FilterAction, FilterScriptMeta,
    FilterScriptRunner, generate_filter_script,
)
from meshprep.core.actions import ActionRegistry, get_action_registry


class TestFilterAction:
    """Tests for FilterAction class."""
    
    def test_create_action(self):
        """Test creating a filter action."""
        action = FilterAction(
            name="fill_holes",
            params={"max_hole_size": 500},
        )
        
        assert action.name == "fill_holes"
        assert action.params["max_hole_size"] == 500
        assert action.on_error == "abort"
    
    def test_action_to_dict(self):
        """Test converting action to dictionary."""
        action = FilterAction(
            name="fill_holes",
            params={"max_hole_size": 500},
            id="step-1",
        )
        
        d = action.to_dict()
        
        assert d["name"] == "fill_holes"
        assert d["params"]["max_hole_size"] == 500
        assert d["id"] == "step-1"
    
    def test_action_from_dict(self):
        """Test creating action from dictionary."""
        data = {
            "name": "decimate",
            "params": {"target_ratio": 0.5},
            "on_error": "skip",
        }
        
        action = FilterAction.from_dict(data)
        
        assert action.name == "decimate"
        assert action.params["target_ratio"] == 0.5
        assert action.on_error == "skip"


class TestFilterScript:
    """Tests for FilterScript class."""
    
    def test_create_script(self):
        """Test creating a filter script."""
        script = FilterScript(
            name="test-script",
            version="1.0.0",
            actions=[
                FilterAction(name="trimesh_basic"),
                FilterAction(name="validate"),
            ],
        )
        
        assert script.name == "test-script"
        assert len(script.actions) == 2
    
    def test_script_to_json(self):
        """Test converting script to JSON."""
        script = FilterScript(
            name="test-script",
            actions=[FilterAction(name="validate")],
        )
        
        json_str = script.to_json()
        data = json.loads(json_str)
        
        assert data["name"] == "test-script"
        assert len(data["actions"]) == 1
    
    def test_script_from_json(self):
        """Test creating script from JSON."""
        json_str = '''
        {
            "name": "loaded-script",
            "version": "1.0.0",
            "actions": [
                {"name": "trimesh_basic"},
                {"name": "fill_holes", "params": {"max_hole_size": 1000}}
            ]
        }
        '''
        
        script = FilterScript.from_json(json_str)
        
        assert script.name == "loaded-script"
        assert len(script.actions) == 2
        assert script.actions[1].params["max_hole_size"] == 1000
    
    def test_script_save_load(self, tmp_path):
        """Test saving and loading a script."""
        script = FilterScript(
            name="save-test",
            actions=[
                FilterAction(name="trimesh_basic"),
                FilterAction(name="validate"),
            ],
        )
        
        path = tmp_path / "script.json"
        script.save(path)
        
        loaded = FilterScript.load(path)
        
        assert loaded.name == "save-test"
        assert len(loaded.actions) == 2
    
    def test_add_action(self):
        """Test adding an action."""
        script = FilterScript(name="test")
        script.add_action("trimesh_basic")
        script.add_action("validate")
        
        assert len(script.actions) == 2
        assert script.actions[0].name == "trimesh_basic"
    
    def test_remove_action(self):
        """Test removing an action."""
        script = FilterScript(
            name="test",
            actions=[
                FilterAction(name="a"),
                FilterAction(name="b"),
                FilterAction(name="c"),
            ],
        )
        
        script.remove_action(1)
        
        assert len(script.actions) == 2
        assert script.actions[0].name == "a"
        assert script.actions[1].name == "c"
    
    def test_validate_script(self):
        """Test script validation."""
        # Valid script
        valid = FilterScript(
            name="valid",
            actions=[FilterAction(name="trimesh_basic")],
        )
        errors = valid.validate()
        assert len(errors) == 0
        
        # Invalid script - no actions
        invalid = FilterScript(name="invalid", actions=[])
        errors = invalid.validate()
        assert len(errors) > 0
        
        # Invalid script - unknown action
        unknown = FilterScript(
            name="unknown",
            actions=[FilterAction(name="nonexistent_action")],
        )
        errors = unknown.validate()
        assert len(errors) > 0


class TestFilterScriptRunner:
    """Tests for FilterScriptRunner class."""
    
    def test_run_simple_script(self):
        """Test running a simple script."""
        mesh = MockMesh(
            is_watertight=False,
            hole_count=5,
            degenerate_face_count=10,
        )
        
        script = FilterScript(
            name="test",
            actions=[
                FilterAction(name="trimesh_basic"),
                FilterAction(name="fill_holes", params={"max_hole_size": 1000}),
                FilterAction(name="validate"),
            ],
        )
        
        runner = FilterScriptRunner()
        result = runner.run(script, mesh)
        
        assert result.success
        assert len(result.steps) == 3
        assert all(s.status == "success" for s in result.steps)
        assert result.final_mesh is not None
    
    def test_run_with_progress_callback(self):
        """Test progress callback."""
        mesh = MockMesh()
        script = FilterScript(
            name="test",
            actions=[
                FilterAction(name="trimesh_basic"),
                FilterAction(name="validate"),
            ],
        )
        
        progress_calls = []
        
        def callback(step, total, msg):
            progress_calls.append((step, total, msg))
        
        runner = FilterScriptRunner()
        runner.set_progress_callback(callback)
        runner.run(script, mesh)
        
        assert len(progress_calls) >= 2
    
    def test_run_with_error_abort(self):
        """Test error handling with abort policy."""
        mesh = MockMesh()
        
        # Script with unknown action
        script = FilterScript(
            name="test",
            actions=[
                FilterAction(name="trimesh_basic"),
                FilterAction(name="unknown_action"),
                FilterAction(name="validate"),
            ],
        )
        
        runner = FilterScriptRunner()
        result = runner.run(script, mesh)
        
        assert not result.success
        # Validate should be skipped
        assert result.steps[-1].status == "skipped"


class TestGenerateFilterScript:
    """Tests for filter script generation."""
    
    def test_generate_from_profile(self):
        """Test generating script from profile."""
        script = generate_filter_script(
            profile_name="holes-only",
            model_fingerprint="abc123",
            suggested_actions=["trimesh_basic", "fill_holes", "validate"],
        )
        
        assert script.name == "holes-only-suggested"
        assert len(script.actions) == 3
        assert script.meta.model_fingerprint == "abc123"
        assert script.meta.generated_by == "model_scan"


class TestActionRegistry:
    """Tests for ActionRegistry."""
    
    def test_get_action(self):
        """Test getting an action from registry."""
        registry = get_action_registry()
        
        action = registry.get("fill_holes")
        
        assert action is not None
        assert action.name == "fill_holes"
        assert action.tool == "trimesh"
    
    def test_list_actions(self):
        """Test listing all actions."""
        registry = get_action_registry()
        actions = registry.list_actions()
        
        assert len(actions) > 10
        assert all(hasattr(a, "name") for a in actions)
    
    def test_execute_action(self):
        """Test executing an action."""
        registry = get_action_registry()
        mesh = MockMesh(degenerate_face_count=10)
        
        result = registry.execute("remove_degenerate_faces", mesh, {})
        
        assert result.success
        assert result.mesh.degenerate_face_count == 0
