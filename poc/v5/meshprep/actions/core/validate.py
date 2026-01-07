# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Validate mesh (does not modify, just checks)."""

from typing import Dict, Any, Optional

from meshprep.core.action import Action, ActionRiskLevel, register_action
from meshprep.core.mesh import Mesh
from meshprep.core.validator import Validator


@register_action
class ValidateAction(Action):
    """Validate mesh geometry (no modifications)."""
    
    name = "validate"
    description = "Validate mesh geometry (checkpoint)"
    risk_level = ActionRiskLevel.LOW
    default_params = {}
    
    def execute(self, mesh: Mesh, params: Optional[Dict[str, Any]] = None) -> Mesh:
        """Validate mesh."""
        validator = Validator()
        validation = validator.validate_geometry(mesh)
        
        self.logger.info(f"Validation: watertight={validation.is_watertight}, "
                        f"manifold={validation.is_manifold}, "
                        f"volume={validation.volume:.2f}")
        
        if validation.issues:
            for issue in validation.issues:
                self.logger.warning(f"  Issue: {issue}")
        
        if not validation.is_printable:
            self.logger.warning("Mesh is NOT printable")
        else:
            self.logger.info("Mesh is printable")
        
        return mesh
