# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""History tracker for recording repair attempts."""

import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging
import hashlib

logger = logging.getLogger(__name__)


class HistoryTracker:
    """
    Tracks repair history in SQLite database.
    
    Records all repair attempts with mesh info, pipeline used,
    success/failure, quality scores, and timing.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize tracker."""
        if db_path is None:
            db_path = Path("learning_data") / "history.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_db()
        logger.info(f"HistoryTracker initialized: {self.db_path}")
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS repair_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    mesh_fingerprint TEXT NOT NULL,
                    mesh_file TEXT,
                    vertex_count INTEGER,
                    face_count INTEGER,
                    pipeline_name TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    quality_score REAL,
                    duration_ms REAL,
                    error_message TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS pipeline_stats (
                    pipeline_name TEXT PRIMARY KEY,
                    total_attempts INTEGER DEFAULT 0,
                    successes INTEGER DEFAULT 0,
                    avg_quality REAL,
                    avg_duration_ms REAL
                )
            ''')
            
            conn.commit()
    
    def record_repair(
        self,
        mesh_fingerprint: str,
        pipeline_name: str,
        success: bool,
        mesh_file: Optional[str] = None,
        vertex_count: Optional[int] = None,
        face_count: Optional[int] = None,
        quality_score: Optional[float] = None,
        duration_ms: Optional[float] = None,
        error_message: Optional[str] = None,
    ):
        """Record a repair attempt."""
        timestamp = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO repair_history (
                    timestamp, mesh_fingerprint, mesh_file,
                    vertex_count, face_count, pipeline_name,
                    success, quality_score, duration_ms, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp, mesh_fingerprint, mesh_file,
                vertex_count, face_count, pipeline_name,
                1 if success else 0, quality_score, duration_ms, error_message
            ))
            conn.commit()
        
        self._update_pipeline_stats(pipeline_name, success, quality_score, duration_ms)
    
    def _update_pipeline_stats(self, pipeline_name: str, success: bool, quality_score, duration_ms):
        """Update pipeline statistics."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                'SELECT total_attempts, successes, avg_quality, avg_duration_ms '
                'FROM pipeline_stats WHERE pipeline_name = ?',
                (pipeline_name,)
            ).fetchone()
            
            if row:
                total, successes, avg_quality, avg_duration = row
                total += 1
                successes += 1 if success else 0
                
                if quality_score is not None and avg_quality is not None:
                    avg_quality = (avg_quality * (total - 1) + quality_score) / total
                elif quality_score is not None:
                    avg_quality = quality_score
                
                if duration_ms is not None and avg_duration is not None:
                    avg_duration = (avg_duration * (total - 1) + duration_ms) / total
                elif duration_ms is not None:
                    avg_duration = duration_ms
                
                conn.execute('''
                    UPDATE pipeline_stats
                    SET total_attempts = ?, successes = ?, avg_quality = ?, avg_duration_ms = ?
                    WHERE pipeline_name = ?
                ''', (total, successes, avg_quality, avg_duration, pipeline_name))
            else:
                conn.execute('''
                    INSERT INTO pipeline_stats (
                        pipeline_name, total_attempts, successes, avg_quality, avg_duration_ms
                    ) VALUES (?, ?, ?, ?, ?)
                ''', (pipeline_name, 1, 1 if success else 0, quality_score, duration_ms))
            
            conn.commit()
    
    def get_pipeline_stats(self, pipeline_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a pipeline."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                'SELECT * FROM pipeline_stats WHERE pipeline_name = ?',
                (pipeline_name,)
            ).fetchone()
            
            return dict(row) if row else None
    
    def get_all_pipeline_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all pipelines."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                'SELECT * FROM pipeline_stats ORDER BY successes DESC'
            ).fetchall()
            
            return [dict(row) for row in rows]
    
    def compute_mesh_fingerprint(self, mesh) -> str:
        """Compute fingerprint for a mesh."""
        vertices_bytes = mesh.trimesh.vertices.tobytes()
        faces_bytes = mesh.trimesh.faces.tobytes()
        
        h = hashlib.sha256()
        h.update(vertices_bytes)
        h.update(faces_bytes)
        
        return f"MP:{h.hexdigest()[:12]}"
    
    def get_recent_repairs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent repair attempts."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                'SELECT * FROM repair_history ORDER BY timestamp DESC LIMIT ?',
                (limit,)
            ).fetchall()
            
            return [dict(row) for row in rows]
