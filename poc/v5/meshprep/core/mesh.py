# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""
Mesh wrapper class with metadata and feature extraction.

Wraps trimesh.Trimesh with additional functionality for repair.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import logging

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


@dataclass
class MeshMetadata:
    """Metadata associated with a mesh."""
    
    fingerprint: str = ""
    source_file: Optional[Path] = None
    profile: str = "unknown"
    
    # Geometric properties
    is_watertight: bool = False
    is_manifold: bool = False
    vertex_count: int = 0
    face_count: int = 0
    body_count: int = 1
    volume: float = 0.0
    
    # Additional context
    tags: Dict[str, Any] = field(default_factory=dict)


class Mesh:
    """Mesh wrapper with metadata and feature extraction."""
    
    def __init__(self, trimesh_obj: trimesh.Trimesh, metadata: Optional[MeshMetadata] = None):
        self._mesh = trimesh_obj
        self._metadata = metadata or MeshMetadata()
        self._update_metadata_from_mesh()
    
    @classmethod
    def load(cls, path: Path) -> "Mesh":
        """Load a mesh from file."""
        logger.info(f"Loading mesh from {path}")
        
        try:
            mesh_obj = trimesh.load(str(path), force='mesh')
            
            if isinstance(mesh_obj, trimesh.Scene):
                logger.debug("Converting Scene to single mesh")
                mesh_obj = trimesh.util.concatenate(list(mesh_obj.geometry.values()))
            
            metadata = MeshMetadata(source_file=path)
            return cls(mesh_obj, metadata)
            
        except Exception as e:
            logger.error(f"Failed to load mesh: {e}")
            raise
    
    def _update_metadata_from_mesh(self):
        """Update metadata from current mesh state."""
        try:
            self._metadata.vertex_count = len(self._mesh.vertices)
            self._metadata.face_count = len(self._mesh.faces)
            self._metadata.is_watertight = bool(self._mesh.is_watertight)
            self._metadata.is_manifold = bool(self._mesh.is_volume)
            
            try:
                self._metadata.volume = float(self._mesh.volume) if self._mesh.is_volume else 0.0
            except:
                self._metadata.volume = 0.0
            
            try:
                components = self._mesh.split(only_watertight=False)
                self._metadata.body_count = len(components)
            except:
                self._metadata.body_count = 1
                
        except Exception as e:
            logger.warning(f"Failed to update metadata: {e}")
    
    @property
    def trimesh(self) -> trimesh.Trimesh:
        """Access underlying trimesh object."""
        return self._mesh
    
    @property
    def metadata(self) -> MeshMetadata:
        """Access metadata."""
        return self._metadata
    def copy(self) -> "Mesh":
        """Create a deep copy of this mesh."""
        from copy import deepcopy
        return Mesh(
            self._mesh.copy(),
            deepcopy(self._metadata),
        )

    
    def __repr__(self) -> str:
        return f"Mesh(vertices={self._metadata.vertex_count}, faces={self._metadata.face_count})"


