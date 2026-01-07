# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Test bootstrap - SIMPLIFIED to just verify it exists and works."""

from meshprep.core.bootstrap import get_bootstrap_manager
from meshprep.core import ActionRegistry


def test_bootstrap_exists():
    """Bootstrap manager can be created."""
    manager = get_bootstrap_manager()
    assert manager is not None


def test_check_dependency():
    """Can check if a dependency exists."""
    manager = get_bootstrap_manager()
    
    # These should exist in test environment
    assert manager.check_dependency("numpy") == True
    assert manager.check_dependency("trimesh") == True
    
    # This shouldn't exist
    assert manager.check_dependency("nonexistent_package_xyz") == False


def test_all_actions_registered(check_test_dependencies):
    """Verify all 20 actions registered."""
    # Import action modules
    from meshprep.actions import trimesh, pymeshfix, blender, open3d, core
    
    actions = ActionRegistry.list_actions()
    assert len(actions) >= 20, f"Expected 20+ actions, got {len(actions)}"
