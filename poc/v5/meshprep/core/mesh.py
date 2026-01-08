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
import trimesh as trimesh_lib

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
    
    def __init__(self, trimesh_obj: trimesh_lib.Trimesh, metadata: Optional[MeshMetadata] = None):
        # Always convert to regular Trimesh (not primitive) for mutability
        self._mesh = self._ensure_mutable(trimesh_obj)
        self._metadata = metadata or MeshMetadata()
        self._update_metadata_from_mesh()
    
    @staticmethod
    def _ensure_mutable(mesh_obj: trimesh_lib.Trimesh) -> trimesh_lib.Trimesh:
        """
        Ensure mesh is mutable (not a primitive with read-only arrays).
        
        Trimesh primitives (Box, Sphere, etc.) have immutable vertices/faces.
        This converts them to regular Trimesh objects with writable arrays.
        """
        # Check if it's a primitive or has read-only arrays
        try:
            if hasattr(mesh_obj, '_primitive') or not mesh_obj.vertices.flags.writeable:
                # Convert to regular Trimesh with writable arrays
                return trimesh_lib.Trimesh(
                    vertices=np.array(mesh_obj.vertices, dtype=np.float64),
                    faces=np.array(mesh_obj.faces, dtype=np.int64),
                    process=False  # Don't modify during creation
                )
        except:
            pass
        
        # Already mutable or regular mesh
        return mesh_obj
    
    @classmethod
    def load(cls, path: Path) -> "Mesh":
        """Load a mesh from file."""
        logger.info(f"Loading mesh from {path}")
        
        try:
            mesh_obj = trimesh_lib.load(str(path), force='mesh')
            
            if isinstance(mesh_obj, trimesh_lib.Scene):
                logger.debug("Converting Scene to single mesh")
                mesh_obj = trimesh_lib.util.concatenate(list(mesh_obj.geometry.values()))
            
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
    def trimesh(self) -> trimesh_lib.Trimesh:
        """Access underlying trimesh object."""
        return self._mesh
    
    @trimesh.setter
    def trimesh(self, value: trimesh_lib.Trimesh):
        """Set underlying trimesh object (ensures mutability)."""
        self._mesh = self._ensure_mutable(value)
        self._update_metadata_from_mesh()
    
    @property
    def metadata(self) -> MeshMetadata:
        """Access metadata."""
        return self._metadata
    
    def copy(self) -> "Mesh":
        """Create a deep copy of this mesh with mutable arrays."""
        from copy import deepcopy
        
        # Create new trimesh with copied, writable arrays
        new_trimesh = trimesh_lib.Trimesh(
            vertices=np.array(self._mesh.vertices, dtype=np.float64),
            faces=np.array(self._mesh.faces, dtype=np.int64),
            process=False
        )
        
        return Mesh(new_trimesh, deepcopy(self._metadata))
    
    def sample_points(self, n_points: int = 2048, include_normals: bool = True) -> Dict[str, np.ndarray]:
        """Sample points from mesh surface for ML encoding."""
        points, face_indices = trimesh_lib.sample.sample_surface(self._mesh, n_points)
        
        result = {"points": points}
        
        if include_normals:
            result["normals"] = self._mesh.face_normals[face_indices]
        
        return result
    
    def save(self, path: Path):
        """Save mesh to file."""
        self._mesh.export(str(path))
        logger.info(f"Saved mesh to {path}")
    
    def __repr__(self) -> str:
        return f"Mesh(vertices={self._metadata.vertex_count}, faces={self._metadata.face_count})"
