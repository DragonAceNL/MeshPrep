# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Main repair engine orchestrating the repair process."""

from dataclasses import dataclass
from typing import Optional, List, TYPE_CHECKING
from pathlib import Path
import logging

from .mesh import Mesh
from .pipeline import Pipeline, PipelineResult
from .validator import Validator, ValidationResult

if TYPE_CHECKING:
    from meshprep.learning import HistoryTracker

logger = logging.getLogger(__name__)


@dataclass
class RepairResult:
    """Result of a repair operation."""
    success: bool
    mesh: Optional[Mesh] = None
    error: Optional[str] = None
    pipeline_used: Optional[str] = None
    attempts: int = 0
    validation: Optional[ValidationResult] = None


class RepairEngine:
    """Main engine for orchestrating mesh repair."""
    
    def __init__(
        self,
        validator: Optional[Validator] = None,
        tracker: Optional["HistoryTracker"] = None,
        max_attempts: int = 10,
    ):
        """
        Initialize repair engine.
        
        Args:
            validator: Validator instance
            tracker: History tracker for learning
            max_attempts: Maximum repair attempts
        """
        self.validator = validator or Validator()
        self.tracker = tracker
        self.max_attempts = max_attempts
        self.logger = logging.getLogger("meshprep.engine")
    
    def repair(
        self,
        mesh_path: Path,
        pipelines: Optional[List[Pipeline]] = None,
    ) -> RepairResult:
        """
        Repair a mesh file.
        
        Args:
            mesh_path: Path to mesh file
            pipelines: List of pipelines to try
            
        Returns:
            RepairResult
        """
        self.logger.info(f"Repairing {mesh_path}")
        
        # Load mesh
        try:
            if isinstance(mesh_path, Mesh):
                mesh = mesh_path
            else:
                mesh = Mesh.load(mesh_path)
        except Exception as e:
            return RepairResult(
                success=False,
                error=f"Failed to load mesh: {e}",
            )
        
        # Check if already valid
        validation = self.validator.validate_geometry(mesh)
        if validation.is_printable:
            self.logger.info("Mesh is already printable")
            return RepairResult(
                success=True,
                mesh=mesh,
                attempts=0,
                validation=ValidationResult(geometric=validation),
            )
        
        # Get pipelines
        if pipelines is None:
            pipelines = self._get_default_pipelines()
        
        # Try each pipeline
        for attempt, pipeline in enumerate(pipelines[:self.max_attempts], 1):
            self.logger.info(f"Attempt {attempt}: {pipeline.name}")
            
            result = pipeline.execute(mesh)
            
            if result.success and result.mesh:
                # Validate result
                validation = self.validator.validate_geometry(result.mesh)
                
                # Track the repair if tracker available
                if self.tracker:
                    try:
                        fingerprint = self.tracker.compute_mesh_fingerprint(mesh)
                        self.tracker.record_repair(
                            mesh_fingerprint=fingerprint,
                            pipeline_name=pipeline.name,
                            success=validation.is_printable,
                            vertex_count=result.mesh.metadata.vertex_count,
                            face_count=result.mesh.metadata.face_count,
                            duration_ms=result.duration_ms,
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to track repair: {e}")
                
                if validation.is_printable:
                    self.logger.info(f"Repair successful with {pipeline.name}")
                    return RepairResult(
                        success=True,
                        mesh=result.mesh,
                        pipeline_used=pipeline.name,
                        attempts=attempt,
                        validation=ValidationResult(geometric=validation),
                    )
        
        # All attempts failed
        return RepairResult(
            success=False,
            error=f"Failed after {len(pipelines)} attempts",
            attempts=len(pipelines),
        )
    
    def _get_default_pipelines(self) -> List[Pipeline]:
        """Get default pipelines."""
        return [
            Pipeline(
                name="light-repair",
                actions=[
                    {"name": "remove_duplicates"},
                    {"name": "fix_normals"},
                    {"name": "fill_holes"},
                ],
                description="Light repair",
            ),
            Pipeline(
                name="standard-repair",
                actions=[
                    {"name": "remove_duplicates"},
                    {"name": "fix_normals"},
                    {"name": "fill_holes"},
                    {"name": "make_watertight"},
                ],
                description="Standard repair",
            ),
            Pipeline(
                name="aggressive-repair",
                actions=[
                    {"name": "pymeshfix_clean"},
                    {"name": "fill_holes"},
                    {"name": "pymeshfix_repair"},
                ],
                description="Aggressive repair",
            ),
        ]
