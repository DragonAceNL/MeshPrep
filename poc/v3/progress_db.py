# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep � https://github.com/DragonAceNL/MeshPrep

"""
SQLite database for MeshPrep batch processing progress and results.

Provides a unified data store for:
- Batch processing progress (replaces progress.json)
- Individual model results (replaces loading from filter JSON files)
- Statistics and summaries

This enables both the live dashboard and reports index to read from
the same source of truth with atomic updates.
"""

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Database path
DB_PATH = Path(__file__).parent / "meshprep_progress.db"

# Schema version for migrations
SCHEMA_VERSION = 1

SCHEMA_SQL = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_info (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- Batch progress tracking (single row, updated frequently)
CREATE TABLE IF NOT EXISTS batch_progress (
    id INTEGER PRIMARY KEY CHECK (id = 1),  -- Only one row
    total_files INTEGER DEFAULT 0,
    processed INTEGER DEFAULT 0,
    successful INTEGER DEFAULT 0,
    failed INTEGER DEFAULT 0,
    escalations INTEGER DEFAULT 0,
    skipped INTEGER DEFAULT 0,
    precheck_skipped INTEGER DEFAULT 0,
    reconstructed INTEGER DEFAULT 0,
    
    start_time TEXT,
    last_update TEXT,
    current_file TEXT,
    current_action TEXT,
    current_step INTEGER DEFAULT 0,
    total_steps INTEGER DEFAULT 0,
    
    -- Action timeout tracking
    action_start_time TEXT,
    action_soft_timeout_s REAL DEFAULT 0,
    action_hard_timeout_s REAL DEFAULT 0,
    action_timeout_factor REAL DEFAULT 0,
    
    total_duration_ms REAL DEFAULT 0,
    avg_duration_ms REAL DEFAULT 0,
    eta_seconds REAL DEFAULT 0
);

-- Individual model results
CREATE TABLE IF NOT EXISTS model_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id TEXT UNIQUE NOT NULL,
    model_fingerprint TEXT,
    file_path TEXT,
    
    -- Status
    success INTEGER DEFAULT 0,
    filter_used TEXT,
    escalation_used INTEGER DEFAULT 0,
    error TEXT,
    
    -- Precheck
    precheck_passed INTEGER DEFAULT 0,
    precheck_skipped INTEGER DEFAULT 0,
    
    -- Reconstruction
    is_reconstruction INTEGER DEFAULT 0,
    reconstruction_method TEXT,
    geometry_loss_pct REAL DEFAULT 0,
    
    -- Original mesh stats
    original_vertices INTEGER DEFAULT 0,
    original_faces INTEGER DEFAULT 0,
    original_volume REAL DEFAULT 0,
    original_watertight INTEGER DEFAULT 0,
    original_manifold INTEGER DEFAULT 0,
    original_components INTEGER DEFAULT 1,
    original_holes INTEGER DEFAULT 0,
    original_file_size INTEGER DEFAULT 0,
    
    -- Result mesh stats
    result_vertices INTEGER DEFAULT 0,
    result_faces INTEGER DEFAULT 0,
    result_volume REAL DEFAULT 0,
    result_watertight INTEGER DEFAULT 0,
    result_manifold INTEGER DEFAULT 0,
    result_components INTEGER DEFAULT 1,
    result_holes INTEGER DEFAULT 0,
    fixed_file_size INTEGER DEFAULT 0,
    
    -- Changes
    volume_change_pct REAL DEFAULT 0,
    face_change_pct REAL DEFAULT 0,
    
    -- Timing
    duration_ms REAL DEFAULT 0,
    timestamp TEXT,
    
    -- Indexing
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_results_file_id ON model_results(file_id);
CREATE INDEX IF NOT EXISTS idx_results_success ON model_results(success);
CREATE INDEX IF NOT EXISTS idx_results_filter ON model_results(filter_used);
CREATE INDEX IF NOT EXISTS idx_results_fingerprint ON model_results(model_fingerprint);
CREATE INDEX IF NOT EXISTS idx_results_precheck_skipped ON model_results(precheck_skipped);
CREATE INDEX IF NOT EXISTS idx_results_escalation ON model_results(escalation_used);

-- Initialize batch progress with single row
INSERT OR IGNORE INTO batch_progress (id) VALUES (1);

-- Store schema version
INSERT OR REPLACE INTO schema_info (key, value) VALUES ('version', '1');
"""


@dataclass
class Progress:
    """Track overall progress of batch processing."""
    total_files: int = 0
    processed: int = 0
    successful: int = 0
    failed: int = 0
    escalations: int = 0
    skipped: int = 0
    precheck_skipped: int = 0
    reconstructed: int = 0
    
    start_time: str = ""
    last_update: str = ""
    current_file: str = ""
    current_action: str = ""
    current_step: int = 0
    total_steps: int = 0
    
    # Action timeout tracking
    action_start_time: str = ""        # When current action started
    action_soft_timeout_s: float = 0    # Soft timeout in seconds
    action_hard_timeout_s: float = 0    # Hard timeout (kill) in seconds
    action_timeout_factor: float = 0    # Learned timeout factor
    
    total_duration_ms: float = 0
    avg_duration_ms: float = 0
    eta_seconds: float = 0
    
    @property
    def percent_complete(self) -> float:
        if self.total_files == 0:
            return 0
        return (self.processed / self.total_files) * 100
    
    @property
    def success_rate(self) -> float:
        if self.processed == 0:
            return 0
        return (self.successful / self.processed) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d['percent_complete'] = self.percent_complete
        d['success_rate'] = self.success_rate
        return d


@dataclass 
class ModelResult:
    """Result for a single model processing."""
    file_id: str
    model_fingerprint: str = ""
    file_path: str = ""
    
    success: bool = False
    filter_used: str = ""
    escalation_used: bool = False
    error: str = ""
    
    precheck_passed: bool = False
    precheck_skipped: bool = False
    
    is_reconstruction: bool = False
    reconstruction_method: str = ""
    geometry_loss_pct: float = 0
    
    original_vertices: int = 0
    original_faces: int = 0
    original_volume: float = 0
    original_watertight: bool = False
    original_manifold: bool = False
    original_components: int = 1
    original_holes: int = 0
    original_file_size: int = 0
    
    result_vertices: int = 0
    result_faces: int = 0
    result_volume: float = 0
    result_watertight: bool = False
    result_manifold: bool = False
    result_components: int = 1
    result_holes: int = 0
    fixed_file_size: int = 0
    
    volume_change_pct: float = 0
    face_change_pct: float = 0
    
    duration_ms: float = 0
    timestamp: str = ""


class ProgressDatabase:
    """SQLite database for progress and results tracking."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(SCHEMA_SQL)
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    # =========================================================================
    # Progress Methods
    # =========================================================================
    
    def get_progress(self) -> Progress:
        """Get current batch progress."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM batch_progress WHERE id = 1").fetchone()
            if row:
                # Handle both old schema (without timeout fields) and new schema
                action_start_time = ""
                action_soft_timeout_s = 0.0
                action_hard_timeout_s = 0.0
                action_timeout_factor = 0.0
                
                try:
                    action_start_time = row["action_start_time"] or ""
                    action_soft_timeout_s = row["action_soft_timeout_s"] or 0.0
                    action_hard_timeout_s = row["action_hard_timeout_s"] or 0.0
                    action_timeout_factor = row["action_timeout_factor"] or 0.0
                except (IndexError, KeyError):
                    pass  # Old schema, use defaults
                
                return Progress(
                    total_files=row["total_files"] or 0,
                    processed=row["processed"] or 0,
                    successful=row["successful"] or 0,
                    failed=row["failed"] or 0,
                    escalations=row["escalations"] or 0,
                    skipped=row["skipped"] or 0,
                    precheck_skipped=row["precheck_skipped"] or 0,
                    reconstructed=row["reconstructed"] or 0,
                    start_time=row["start_time"] or "",
                    last_update=row["last_update"] or "",
                    current_file=row["current_file"] or "",
                    current_action=row["current_action"] or "",
                    current_step=row["current_step"] or 0,
                    total_steps=row["total_steps"] or 0,
                    action_start_time=action_start_time,
                    action_soft_timeout_s=action_soft_timeout_s,
                    action_hard_timeout_s=action_hard_timeout_s,
                    action_timeout_factor=action_timeout_factor,
                    total_duration_ms=row["total_duration_ms"] or 0,
                    avg_duration_ms=row["avg_duration_ms"] or 0,
                    eta_seconds=row["eta_seconds"] or 0,
                )
            return Progress()
    
    def save_progress(self, progress: Progress) -> None:
        """Save batch progress."""
        progress.last_update = datetime.now().isoformat()
        
        with self._get_connection() as conn:
            # Try to add timeout columns if they don't exist (migration)
            try:
                conn.execute("ALTER TABLE batch_progress ADD COLUMN action_start_time TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists
            try:
                conn.execute("ALTER TABLE batch_progress ADD COLUMN action_soft_timeout_s REAL DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE batch_progress ADD COLUMN action_hard_timeout_s REAL DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE batch_progress ADD COLUMN action_timeout_factor REAL DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            
            conn.execute("""
                UPDATE batch_progress SET
                    total_files = ?,
                    processed = ?,
                    successful = ?,
                    failed = ?,
                    escalations = ?,
                    skipped = ?,
                    precheck_skipped = ?,
                    reconstructed = ?,
                    start_time = ?,
                    last_update = ?,
                    current_file = ?,
                    current_action = ?,
                    current_step = ?,
                    total_steps = ?,
                    action_start_time = ?,
                    action_soft_timeout_s = ?,
                    action_hard_timeout_s = ?,
                    action_timeout_factor = ?,
                    total_duration_ms = ?,
                    avg_duration_ms = ?,
                    eta_seconds = ?
                WHERE id = 1
            """, (
                progress.total_files,
                progress.processed,
                progress.successful,
                progress.failed,
                progress.escalations,
                progress.skipped,
                progress.precheck_skipped,
                progress.reconstructed,
                progress.start_time,
                progress.last_update,
                progress.current_file,
                progress.current_action,
                progress.current_step,
                progress.total_steps,
                progress.action_start_time,
                progress.action_soft_timeout_s,
                progress.action_hard_timeout_s,
                progress.action_timeout_factor,
                progress.total_duration_ms,
                progress.avg_duration_ms,
                progress.eta_seconds,
            ))
    
    def reset_progress(self) -> None:
        """Reset progress for a new batch run."""
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE batch_progress SET
                    total_files = 0,
                    processed = 0,
                    successful = 0,
                    failed = 0,
                    escalations = 0,
                    skipped = 0,
                    precheck_skipped = 0,
                    reconstructed = 0,
                    start_time = ?,
                    last_update = ?,
                    current_file = '',
                    current_action = '',
                    current_step = 0,
                    total_steps = 0,
                    total_duration_ms = 0,
                    avg_duration_ms = 0,
                    eta_seconds = 0
                WHERE id = 1
            """, (datetime.now().isoformat(), datetime.now().isoformat()))
    
    # =========================================================================
    # Model Results Methods
    # =========================================================================
    
    def save_result(self, result: ModelResult) -> None:
        """Save or update a model result."""
        now = datetime.now().isoformat()
        
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO model_results (
                    file_id, model_fingerprint, file_path,
                    success, filter_used, escalation_used, error,
                    precheck_passed, precheck_skipped,
                    is_reconstruction, reconstruction_method, geometry_loss_pct,
                    original_vertices, original_faces, original_volume,
                    original_watertight, original_manifold, original_components,
                    original_holes, original_file_size,
                    result_vertices, result_faces, result_volume,
                    result_watertight, result_manifold, result_components,
                    result_holes, fixed_file_size,
                    volume_change_pct, face_change_pct,
                    duration_ms, timestamp, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_id) DO UPDATE SET
                    model_fingerprint = excluded.model_fingerprint,
                    file_path = excluded.file_path,
                    success = excluded.success,
                    filter_used = excluded.filter_used,
                    escalation_used = excluded.escalation_used,
                    error = excluded.error,
                    precheck_passed = excluded.precheck_passed,
                    precheck_skipped = excluded.precheck_skipped,
                    is_reconstruction = excluded.is_reconstruction,
                    reconstruction_method = excluded.reconstruction_method,
                    geometry_loss_pct = excluded.geometry_loss_pct,
                    original_vertices = excluded.original_vertices,
                    original_faces = excluded.original_faces,
                    original_volume = excluded.original_volume,
                    original_watertight = excluded.original_watertight,
                    original_manifold = excluded.original_manifold,
                    original_components = excluded.original_components,
                    original_holes = excluded.original_holes,
                    original_file_size = excluded.original_file_size,
                    result_vertices = excluded.result_vertices,
                    result_faces = excluded.result_faces,
                    result_volume = excluded.result_volume,
                    result_watertight = excluded.result_watertight,
                    result_manifold = excluded.result_manifold,
                    result_components = excluded.result_components,
                    result_holes = excluded.result_holes,
                    fixed_file_size = excluded.fixed_file_size,
                    volume_change_pct = excluded.volume_change_pct,
                    face_change_pct = excluded.face_change_pct,
                    duration_ms = excluded.duration_ms,
                    timestamp = excluded.timestamp,
                    updated_at = excluded.updated_at
            """, (
                result.file_id, result.model_fingerprint, result.file_path,
                1 if result.success else 0, result.filter_used, 
                1 if result.escalation_used else 0, result.error or "",
                1 if result.precheck_passed else 0, 1 if result.precheck_skipped else 0,
                1 if result.is_reconstruction else 0, result.reconstruction_method or "",
                result.geometry_loss_pct,
                result.original_vertices, result.original_faces, result.original_volume,
                1 if result.original_watertight else 0, 1 if result.original_manifold else 0,
                result.original_components, result.original_holes, result.original_file_size,
                result.result_vertices, result.result_faces, result.result_volume,
                1 if result.result_watertight else 0, 1 if result.result_manifold else 0,
                result.result_components, result.result_holes, result.fixed_file_size,
                result.volume_change_pct, result.face_change_pct,
                result.duration_ms, result.timestamp or now, now,
            ))
    
    def get_result(self, file_id: str) -> Optional[ModelResult]:
        """Get result for a specific model."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM model_results WHERE file_id = ?",
                (file_id,)
            ).fetchone()
            if row:
                return self._row_to_result(row)
            return None
    
    def get_all_results(
        self,
        status_filter: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[ModelResult]:
        """Get all model results with optional filtering."""
        query = "SELECT * FROM model_results"
        params: list = []
        
        if status_filter:
            if status_filter == "success":
                query += " WHERE success = 1 AND precheck_skipped = 0"
            elif status_filter == "failed":
                query += " WHERE success = 0"
            elif status_filter == "skipped":
                query += " WHERE precheck_skipped = 1"
            elif status_filter == "escalated":
                query += " WHERE escalation_used = 1"
        
        query += " ORDER BY file_id ASC"
        
        if limit:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_result(row) for row in rows]
    
    def get_results_count(self, status_filter: Optional[str] = None) -> int:
        """Get count of results with optional filtering."""
        query = "SELECT COUNT(*) FROM model_results"
        
        if status_filter:
            if status_filter == "success":
                query += " WHERE success = 1 AND precheck_skipped = 0"
            elif status_filter == "failed":
                query += " WHERE success = 0"
            elif status_filter == "skipped":
                query += " WHERE precheck_skipped = 1"
            elif status_filter == "escalated":
                query += " WHERE escalation_used = 1"
        
        with self._get_connection() as conn:
            return conn.execute(query).fetchone()[0]
    
    def get_processed_file_ids(self) -> set:
        """Get set of all processed file IDs."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT file_id FROM model_results").fetchall()
            return {row["file_id"] for row in rows}
    
    def _row_to_result(self, row: sqlite3.Row) -> ModelResult:
        """Convert database row to ModelResult."""
        return ModelResult(
            file_id=row["file_id"],
            model_fingerprint=row["model_fingerprint"] or "",
            file_path=row["file_path"] or "",
            success=bool(row["success"]),
            filter_used=row["filter_used"] or "",
            escalation_used=bool(row["escalation_used"]),
            error=row["error"] or "",
            precheck_passed=bool(row["precheck_passed"]),
            precheck_skipped=bool(row["precheck_skipped"]),
            is_reconstruction=bool(row["is_reconstruction"]),
            reconstruction_method=row["reconstruction_method"] or "",
            geometry_loss_pct=row["geometry_loss_pct"] or 0,
            original_vertices=row["original_vertices"] or 0,
            original_faces=row["original_faces"] or 0,
            original_volume=row["original_volume"] or 0,
            original_watertight=bool(row["original_watertight"]),
            original_manifold=bool(row["original_manifold"]),
            original_components=row["original_components"] or 1,
            original_holes=row["original_holes"] or 0,
            original_file_size=row["original_file_size"] or 0,
            result_vertices=row["result_vertices"] or 0,
            result_faces=row["result_faces"] or 0,
            result_volume=row["result_volume"] or 0,
            result_watertight=bool(row["result_watertight"]),
            result_manifold=bool(row["result_manifold"]),
            result_components=row["result_components"] or 1,
            result_holes=row["result_holes"] or 0,
            fixed_file_size=row["fixed_file_size"] or 0,
            volume_change_pct=row["volume_change_pct"] or 0,
            face_change_pct=row["face_change_pct"] or 0,
            duration_ms=row["duration_ms"] or 0,
            timestamp=row["timestamp"] or "",
        )
    
    # =========================================================================
    # Statistics Methods
    # =========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics from results."""
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(success) as successful,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed,
                    SUM(precheck_skipped) as precheck_skipped,
                    SUM(escalation_used) as escalations,
                    SUM(is_reconstruction) as reconstructed,
                    AVG(duration_ms) as avg_duration_ms,
                    SUM(duration_ms) as total_duration_ms
                FROM model_results
            """).fetchone()
            
            return {
                "total": row["total"] or 0,
                "successful": row["successful"] or 0,
                "failed": row["failed"] or 0,
                "precheck_skipped": row["precheck_skipped"] or 0,
                "escalations": row["escalations"] or 0,
                "reconstructed": row["reconstructed"] or 0,
                "avg_duration_ms": row["avg_duration_ms"] or 0,
                "total_duration_ms": row["total_duration_ms"] or 0,
            }


# Global instance
_db_instance: Optional[ProgressDatabase] = None


def get_progress_db() -> ProgressDatabase:
    """Get or create the global database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = ProgressDatabase()
    return _db_instance


def reset_progress_db() -> None:
    """Reset the global database instance."""
    global _db_instance
    _db_instance = None
