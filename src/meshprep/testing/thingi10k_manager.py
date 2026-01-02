# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Thingi10K Dataset Manager

This module manages the Thingi10K dataset for MeshPrep testing:
- Downloads and stores metadata in SQLite database
- Categorizes models by defect type
- Provides query interface for test fixture selection

Storage locations:
- Raw models: External directory (e.g., C:/Users/.../Thingi10K/raw_meshes/)
- Metadata DB: MeshPrep/data/thingi10k/thingi10k.db
- Test fixtures: MeshPrep/tests/fixtures/thingi10k/
"""

import json
import sqlite3
import shutil
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterator
import logging

logger = logging.getLogger(__name__)


# Default paths
DEFAULT_RAW_MESHES_PATH = Path(r"C:\Users\Dragon Ace\Source\repos\Thingi10K\raw_meshes")
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "thingi10k" / "thingi10k.db"
DEFAULT_FIXTURES_PATH = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "thingi10k"

# Licenses that allow redistribution (for including in git repository)
# These licenses allow sharing and redistribution, with or without attribution requirements
PERMISSIVE_LICENSES = [
    "Creative Commons - Attribution",           # CC-BY: Can share with attribution
    "Creative Commons - Attribution - Share Alike",  # CC-BY-SA: Can share with attribution + same license
    "Creative Commons - Public Domain Dedication",   # CC0: Public domain
    "Public Domain",                            # Public domain
    "GNU - GPL",                                # GPL: Can share with source
    "GNU - LGPL",                               # LGPL: Can share with source
    "BSD License",                              # BSD: Can share with attribution
]

# Licenses that restrict redistribution (Non-Commercial or No-Derivatives)
# These should NOT be included in the git repository
RESTRICTIVE_LICENSES = [
    "Creative Commons - Attribution - Non-Commercial",  # NC: Non-commercial only
    "Attribution - Non-Commercial - Share Alike",       # NC-SA: Non-commercial only
    "Attribution - Non-Commercial - No Derivatives",    # NC-ND: Most restrictive
    "Creative Commons - Attribution - No Derivatives",  # ND: No derivatives
    "unknown_license",                                  # Unknown: Don't risk it
]


@dataclass
class ModelMetadata:
    """Metadata for a single Thingi10K model."""
    
    file_id: int
    
    # Basic info
    name: str = ""
    thing_id: int = 0
    author: str = ""
    license: str = ""
    
    # Geometry stats
    num_vertices: int = 0
    num_faces: int = 0
    num_edges: int = 0
    volume: float = 0.0
    surface_area: float = 0.0
    
    # Bounding box
    bbox_min_x: float = 0.0
    bbox_min_y: float = 0.0
    bbox_min_z: float = 0.0
    bbox_max_x: float = 0.0
    bbox_max_y: float = 0.0
    bbox_max_z: float = 0.0
    
    # Quality flags
    is_watertight: bool = False
    is_manifold: bool = False
    is_oriented: bool = False
    is_solid: bool = False
    
    # Defect counts
    num_holes: int = 0
    num_components: int = 1
    num_degenerate_faces: int = 0
    num_duplicate_faces: int = 0
    num_self_intersections: int = 0
    
    # Computed flags
    has_holes: bool = False
    has_multiple_components: bool = False
    has_self_intersections: bool = False
    has_non_manifold: bool = False
    has_degenerate_faces: bool = False
    
    # Topology
    genus: int = 0
    euler_characteristic: int = 0
    
    # MeshPrep categorization
    primary_category: str = ""  # clean, holes, non_manifold, etc.
    categories: str = ""  # JSON array of all matching categories
    meshprep_profile: str = ""  # Suggested MeshPrep profile
    
    # File info
    file_extension: str = "stl"
    file_size_bytes: int = 0
    
    # Timestamps
    imported_at: str = ""
    analyzed_at: str = ""


# SQL schema
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS models (
    file_id INTEGER PRIMARY KEY,
    
    -- Basic info
    name TEXT,
    thing_id INTEGER,
    author TEXT,
    license TEXT,
    
    -- Geometry stats
    num_vertices INTEGER,
    num_faces INTEGER,
    num_edges INTEGER,
    volume REAL,
    surface_area REAL,
    
    -- Bounding box
    bbox_min_x REAL,
    bbox_min_y REAL,
    bbox_min_z REAL,
    bbox_max_x REAL,
    bbox_max_y REAL,
    bbox_max_z REAL,
    
    -- Quality flags
    is_watertight BOOLEAN,
    is_manifold BOOLEAN,
    is_oriented BOOLEAN,
    is_solid BOOLEAN,
    
    -- Defect counts
    num_holes INTEGER,
    num_components INTEGER,
    num_degenerate_faces INTEGER,
    num_duplicate_faces INTEGER,
    num_self_intersections INTEGER,
    
    -- Computed flags
    has_holes BOOLEAN,
    has_multiple_components BOOLEAN,
    has_self_intersections BOOLEAN,
    has_non_manifold BOOLEAN,
    has_degenerate_faces BOOLEAN,
    
    -- Topology
    genus INTEGER,
    euler_characteristic INTEGER,
    
    -- MeshPrep categorization
    primary_category TEXT,
    categories TEXT,
    meshprep_profile TEXT,
    
    -- File info
    file_extension TEXT,
    file_size_bytes INTEGER,
    
    -- Timestamps
    imported_at TEXT,
    analyzed_at TEXT
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_primary_category ON models(primary_category);
CREATE INDEX IF NOT EXISTS idx_is_watertight ON models(is_watertight);
CREATE INDEX IF NOT EXISTS idx_is_manifold ON models(is_manifold);
CREATE INDEX IF NOT EXISTS idx_has_holes ON models(has_holes);
CREATE INDEX IF NOT EXISTS idx_has_self_intersections ON models(has_self_intersections);
CREATE INDEX IF NOT EXISTS idx_num_faces ON models(num_faces);

-- Category summary view
CREATE VIEW IF NOT EXISTS category_summary AS
SELECT 
    primary_category,
    COUNT(*) as model_count,
    AVG(num_faces) as avg_faces,
    SUM(CASE WHEN is_watertight THEN 1 ELSE 0 END) as watertight_count,
    SUM(CASE WHEN is_manifold THEN 1 ELSE 0 END) as manifold_count
FROM models
GROUP BY primary_category;

-- Test results table (for storing benchmark results)
CREATE TABLE IF NOT EXISTS test_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER,
    test_run_id TEXT,
    
    -- Input state
    profile_detected TEXT,
    profile_confidence REAL,
    
    -- Repair result
    repair_success BOOLEAN,
    repair_error TEXT,
    repair_runtime_ms REAL,
    filter_script_used TEXT,
    
    -- Validation result
    is_geometrically_valid BOOLEAN,
    is_visually_unchanged BOOLEAN,
    volume_change_pct REAL,
    hausdorff_relative REAL,
    
    -- Overall
    overall_success BOOLEAN,
    
    -- Timestamp
    tested_at TEXT,
    
    FOREIGN KEY (file_id) REFERENCES models(file_id)
);

CREATE INDEX IF NOT EXISTS idx_test_run_id ON test_results(test_run_id);
CREATE INDEX IF NOT EXISTS idx_test_file_id ON test_results(file_id);
"""


class Thingi10KDatabase:
    """SQLite database for Thingi10K metadata."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(SCHEMA_SQL)
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def insert_model(self, metadata: ModelMetadata):
        """Insert or update a model's metadata."""
        data = asdict(metadata)
        
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?" for _ in data])
        update_clause = ", ".join([f"{k}=excluded.{k}" for k in data.keys()])
        
        sql = f"""
            INSERT INTO models ({columns})
            VALUES ({placeholders})
            ON CONFLICT(file_id) DO UPDATE SET {update_clause}
        """
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(sql, list(data.values()))
    
    def insert_models_batch(self, models: list[ModelMetadata]):
        """Insert multiple models efficiently."""
        if not models:
            return
        
        data_list = [asdict(m) for m in models]
        columns = ", ".join(data_list[0].keys())
        placeholders = ", ".join(["?" for _ in data_list[0]])
        
        sql = f"INSERT OR REPLACE INTO models ({columns}) VALUES ({placeholders})"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(sql, [list(d.values()) for d in data_list])
    
    def get_model(self, file_id: int) -> Optional[ModelMetadata]:
        """Get a single model by file_id."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM models WHERE file_id = ?", 
                (file_id,)
            ).fetchone()
            
            if row:
                return ModelMetadata(**dict(row))
        return None
    
    def get_models_by_category(
        self, 
        category: str, 
        limit: int = 100,
        max_faces: Optional[int] = None
    ) -> list[ModelMetadata]:
        """Get models by primary category."""
        sql = "SELECT * FROM models WHERE primary_category = ?"
        params: list = [category]
        
        if max_faces:
            sql += " AND num_faces <= ?"
            params.append(max_faces)
        
        sql += " ORDER BY num_faces ASC LIMIT ?"
        params.append(limit)
        
        with self._get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [ModelMetadata(**dict(row)) for row in rows]
    
    def get_models_by_defect(
        self,
        has_holes: Optional[bool] = None,
        has_non_manifold: Optional[bool] = None,
        has_self_intersections: Optional[bool] = None,
        has_multiple_components: Optional[bool] = None,
        is_watertight: Optional[bool] = None,
        limit: int = 100
    ) -> list[ModelMetadata]:
        """Query models by defect flags."""
        conditions = []
        params: list = []
        
        if has_holes is not None:
            conditions.append("has_holes = ?")
            params.append(has_holes)
        
        if has_non_manifold is not None:
            conditions.append("has_non_manifold = ?")
            params.append(has_non_manifold)
        
        if has_self_intersections is not None:
            conditions.append("has_self_intersections = ?")
            params.append(has_self_intersections)
        
        if has_multiple_components is not None:
            conditions.append("has_multiple_components = ?")
            params.append(has_multiple_components)
        
        if is_watertight is not None:
            conditions.append("is_watertight = ?")
            params.append(is_watertight)
        
        sql = "SELECT * FROM models"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY num_faces ASC LIMIT ?"
        params.append(limit)
        
        with self._get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [ModelMetadata(**dict(row)) for row in rows]
    
    def get_category_summary(self) -> dict[str, dict]:
        """Get summary statistics by category."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM category_summary").fetchall()
            return {row["primary_category"]: dict(row) for row in rows}
    
    def get_all_file_ids(self) -> list[int]:
        """Get all file IDs in the database."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT file_id FROM models").fetchall()
            return [row["file_id"] for row in rows]
    
    def count_models(self) -> int:
        """Get total model count."""
        with self._get_connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
    
    def export_to_json(self, output_path: Path):
        """Export all metadata to JSON."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM models").fetchall()
            data = [dict(row) for row in rows]
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported {len(data)} models to {output_path}")
    
    # Test results methods
    def insert_test_result(self, result: dict):
        """Insert a test result."""
        columns = ", ".join(result.keys())
        placeholders = ", ".join(["?" for _ in result])
        
        sql = f"INSERT INTO test_results ({columns}) VALUES ({placeholders})"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(sql, list(result.values()))
    
    def get_test_results_summary(self, test_run_id: str) -> dict:
        """Get summary of a test run."""
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN overall_success THEN 1 ELSE 0 END) as successful,
                    SUM(CASE WHEN repair_success THEN 1 ELSE 0 END) as repair_success,
                    SUM(CASE WHEN is_geometrically_valid THEN 1 ELSE 0 END) as geom_valid,
                    SUM(CASE WHEN is_visually_unchanged THEN 1 ELSE 0 END) as visually_unchanged,
                    AVG(repair_runtime_ms) as avg_runtime_ms
                FROM test_results
                WHERE test_run_id = ?
            """, (test_run_id,)).fetchone()
            
            return dict(row) if row else {}


def categorize_model(metadata: ModelMetadata) -> tuple[str, list[str]]:
    """
    Categorize a model based on its defects.
    
    Returns:
        primary_category: Main category for the model
        categories: List of all matching categories
    """
    categories = []
    
    # Check for clean models first
    if (metadata.is_watertight and 
        metadata.is_manifold and 
        not metadata.has_self_intersections and
        metadata.num_components == 1):
        categories.append("clean")
    
    # Defect categories
    if metadata.has_holes or not metadata.is_watertight:
        if metadata.num_holes > 10:
            categories.append("many_small_holes")
        else:
            categories.append("holes")
    
    if metadata.has_non_manifold or not metadata.is_manifold:
        categories.append("non_manifold")
    
    if metadata.has_self_intersections:
        categories.append("self_intersecting")
    
    if metadata.has_multiple_components or metadata.num_components > 1:
        if metadata.num_components > 5:
            categories.append("fragmented")
        else:
            categories.append("multiple_components")
    
    if metadata.has_degenerate_faces or metadata.num_degenerate_faces > 0:
        categories.append("degenerate_faces")
    
    # Complex: multiple issues
    defect_count = sum([
        metadata.has_holes,
        metadata.has_non_manifold,
        metadata.has_self_intersections,
        metadata.has_multiple_components
    ])
    if defect_count > 1:
        categories.append("complex")
    
    # Determine primary category
    priority = [
        "clean",
        "holes",
        "many_small_holes", 
        "non_manifold",
        "self_intersecting",
        "fragmented",
        "multiple_components",
        "degenerate_faces",
        "complex"
    ]
    
    primary = "unknown"
    for cat in priority:
        if cat in categories:
            primary = cat
            break
    
    return primary, categories


def map_to_meshprep_profile(metadata: ModelMetadata) -> str:
    """Map model metadata to a MeshPrep profile name."""
    
    # Clean models
    if metadata.primary_category == "clean":
        return "clean"
    
    # Holes
    if metadata.primary_category in ("holes", "many_small_holes"):
        if metadata.has_non_manifold:
            return "mesh-with-holes-and-non-manifold"
        return "holes-only"
    
    # Non-manifold
    if metadata.primary_category == "non_manifold":
        return "non-manifold"
    
    # Self-intersecting
    if metadata.primary_category == "self_intersecting":
        return "self-intersecting"
    
    # Fragmented
    if metadata.primary_category in ("fragmented", "multiple_components"):
        return "fragmented"
    
    # Complex
    if metadata.primary_category == "complex":
        return "complex-high-genus"
    
    return "unknown"


class Thingi10KManager:
    """
    High-level manager for Thingi10K dataset.
    
    Handles:
    - Importing metadata from thingi10k Python package
    - Categorizing models
    - Selecting test fixtures
    - Copying fixtures to test directory
    """
    
    def __init__(
        self,
        raw_meshes_path: Optional[Path] = None,
        db_path: Optional[Path] = None,
        fixtures_path: Optional[Path] = None
    ):
        self.raw_meshes_path = raw_meshes_path or DEFAULT_RAW_MESHES_PATH
        self.fixtures_path = fixtures_path or DEFAULT_FIXTURES_PATH
        self.db = Thingi10KDatabase(db_path)
    
    def import_from_thingi10k_package(self):
        """
        Import metadata from the thingi10k Python package.
        
        Reads directly from the CSV files bundled with the package:
        - geometry_data.csv: Geometric properties (vertices, faces, manifold, etc.)
        - contextual_data.csv: Thing metadata (name, author, license, etc.)
        - input_summary.csv: Quality flags summary
        """
        try:
            import thingi10k
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                f"Required package not installed: {e}. "
                "Install with: pip install thingi10k pandas"
            )
        
        # Find the metadata CSV files in the thingi10k package
        import importlib.util
        spec = importlib.util.find_spec("thingi10k")
        if spec is None or spec.origin is None:
            raise ImportError("Cannot find thingi10k package location")
        
        pkg_path = Path(spec.origin).parent
        metadata_path = pkg_path / "metadata"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata directory not found: {metadata_path}")
        
        logger.info(f"Reading metadata from: {metadata_path}")
        
        # Load geometry data (main data source)
        geom_file = metadata_path / "geometry_data.csv"
        if not geom_file.exists():
            raise FileNotFoundError(f"Geometry data file not found: {geom_file}")
        
        logger.info("Loading geometry_data.csv...")
        geom_df = pd.read_csv(geom_file)
        logger.info(f"  Loaded {len(geom_df)} geometry records")
        
        # Load contextual data (thing metadata)
        ctx_file = metadata_path / "contextual_data.csv"
        ctx_df = None
        if ctx_file.exists():
            logger.info("Loading contextual_data.csv...")
            ctx_df = pd.read_csv(ctx_file)
            logger.info(f"  Loaded {len(ctx_df)} contextual records")
        
        # Load input summary (quality flags AND license info)
        summary_file = metadata_path / "input_summary.csv"
        summary_df = None
        license_map = {}  # file_id -> license
        if summary_file.exists():
            logger.info("Loading input_summary.csv...")
            summary_df = pd.read_csv(summary_file)
            logger.info(f"  Loaded {len(summary_df)} summary records")
            # Build license lookup map
            for _, row in summary_df.iterrows():
                license_map[int(row['ID'])] = str(row.get('License', 'unknown_license'))
        
        # Build models from geometry data
        logger.info("Processing metadata...")
        models = []
        
        for _, row in geom_df.iterrows():
            file_id = int(row.get("file_id", 0))
            
            # Get license from summary data
            model_license = license_map.get(file_id, "unknown_license")
            
            # Compute derived flags from geometry data
            num_boundary_edges = int(row.get("num_boundary_edges", 0))
            num_components = int(row.get("num_connected_components", 1))
            vertex_manifold = bool(row.get("vertex_manifold", 0))
            edge_manifold = bool(row.get("edge_manifold", 0))
            is_oriented = bool(row.get("oriented", 0))
            is_solid = bool(row.get("solid", 0))
            num_self_intersections = int(row.get("num_self_intersections", 0))
            num_degenerate = int(row.get("num_geometrical_degenerated_faces", 0)) + \
                            int(row.get("num_combinatorial_degenerated_faces", 0))
            num_duplicated = int(row.get("num_duplicated_faces", 0))
            
            # Watertight = no boundary edges and manifold
            is_watertight = (num_boundary_edges == 0) and edge_manifold and vertex_manifold
            is_manifold = edge_manifold and vertex_manifold
            
            metadata = ModelMetadata(
                file_id=file_id,
                license=model_license,
                num_vertices=int(row.get("num_vertices", 0)),
                num_faces=int(row.get("num_faces", 0)),
                surface_area=float(row.get("total_area", 0.0)),
                is_watertight=is_watertight,
                is_manifold=is_manifold,
                is_oriented=is_oriented,
                is_solid=is_solid,
                num_holes=num_boundary_edges,  # Boundary edges indicate holes
                num_components=num_components,
                num_degenerate_faces=num_degenerate,
                num_duplicate_faces=num_duplicated,
                num_self_intersections=num_self_intersections,
                euler_characteristic=int(row.get("euler_characteristic", 0)),
                has_holes=num_boundary_edges > 0,
                has_multiple_components=num_components > 1,
                has_self_intersections=num_self_intersections > 0,
                has_non_manifold=not is_manifold,
                has_degenerate_faces=num_degenerate > 0,
                imported_at=datetime.now().isoformat()
            )
            
            # Categorize based on defects
            primary, categories = categorize_model(metadata)
            metadata.primary_category = primary
            metadata.categories = json.dumps(categories)
            metadata.meshprep_profile = map_to_meshprep_profile(metadata)
            
            models.append(metadata)
        
        logger.info(f"Importing {len(models)} models to database...")
        self.db.insert_models_batch(models)
        
        logger.info("Import complete!")
        return len(models)
    
    def import_from_directory(self):
        """
        Import metadata by scanning the raw_meshes directory.
        Uses trimesh to analyze each model.
        """
        try:
            import trimesh
        except ImportError:
            raise ImportError("trimesh not installed. Install with: pip install trimesh")
        
        stl_files = list(self.raw_meshes_path.glob("*.stl"))
        logger.info(f"Found {len(stl_files)} STL files")
        
        models = []
        for i, stl_path in enumerate(stl_files):
            if (i + 1) % 100 == 0:
                logger.info(f"Progress: {i + 1}/{len(stl_files)}")
            
            try:
                file_id = int(stl_path.stem)
                mesh = trimesh.load(str(stl_path))
                
                metadata = ModelMetadata(
                    file_id=file_id,
                    num_vertices=len(mesh.vertices),
                    num_faces=len(mesh.faces),
                    volume=float(mesh.volume) if mesh.is_volume else 0.0,
                    surface_area=float(mesh.area),
                    is_watertight=mesh.is_watertight,
                    is_manifold=mesh.is_volume,
                    num_components=len(mesh.split()),
                    file_extension="stl",
                    file_size_bytes=stl_path.stat().st_size,
                    imported_at=datetime.now().isoformat(),
                    analyzed_at=datetime.now().isoformat()
                )
                
                # Set computed flags
                metadata.has_holes = not mesh.is_watertight
                metadata.has_multiple_components = metadata.num_components > 1
                metadata.has_non_manifold = not mesh.is_volume
                
                # Categorize
                primary, categories = categorize_model(metadata)
                metadata.primary_category = primary
                metadata.categories = json.dumps(categories)
                metadata.meshprep_profile = map_to_meshprep_profile(metadata)
                
                models.append(metadata)
                
            except Exception as e:
                logger.warning(f"Error processing {stl_path.name}: {e}")
        
        logger.info(f"Importing {len(models)} models to database...")
        self.db.insert_models_batch(models)
        
        return len(models)
    
    def select_test_fixtures(
        self,
        per_category: int = 20,
        max_faces: int = 100000,
        categories: Optional[list[str]] = None,
        permissive_licenses_only: bool = True
    ) -> dict[str, list[int]]:
        """
        Select representative test fixtures for each category.
        
        Uses defect flags for more accurate selection rather than just
        primary_category, since models often have multiple issues.
        
        Args:
            per_category: Number of models per category
            max_faces: Maximum face count (for faster testing)
            categories: Categories to include (None = all)
            permissive_licenses_only: If True, only select models with
                licenses that allow redistribution (CC-BY, CC-BY-SA, 
                Public Domain, GPL). This is important for including
                fixtures in the git repository.
        
        Returns:
            Dict mapping category -> list of file_ids
        """
        if categories is None:
            categories = [
                "clean",
                "holes",
                "many_small_holes",
                "non_manifold",
                "self_intersecting",
                "fragmented",
                "multiple_components",
                "complex"
            ]
        
        selected: dict[str, list[int]] = {}
        
        # Build license filter clause if needed
        license_filter = ""
        if permissive_licenses_only:
            # Create SQL IN clause for permissive licenses
            license_list = ", ".join([f"'{lic}'" for lic in PERMISSIVE_LICENSES])
            license_filter = f" AND license IN ({license_list})"
            logger.info(f"  Filtering for permissive licenses only")
        
        with self.db._get_connection() as conn:
            for cat in categories:
                # Build query based on category
                if cat == "clean":
                    base_sql = '''
                        SELECT file_id FROM models 
                        WHERE is_watertight = 1 AND is_manifold = 1 
                        AND has_self_intersections = 0 AND num_components = 1
                        AND num_faces <= ?
                    '''
                elif cat == "holes":
                    base_sql = '''
                        SELECT file_id FROM models 
                        WHERE has_holes = 1 AND num_holes <= 10
                        AND num_faces <= ?
                    '''
                elif cat == "many_small_holes":
                    base_sql = '''
                        SELECT file_id FROM models 
                        WHERE has_holes = 1 AND num_holes > 10
                        AND num_faces <= ?
                    '''
                elif cat == "non_manifold":
                    # Non-manifold without holes (to separate from holes category)
                    base_sql = '''
                        SELECT file_id FROM models 
                        WHERE has_non_manifold = 1 AND has_holes = 0
                        AND num_faces <= ?
                    '''
                elif cat == "self_intersecting":
                    base_sql = '''
                        SELECT file_id FROM models 
                        WHERE has_self_intersections = 1
                        AND num_faces <= ?
                    '''
                elif cat == "fragmented":
                    base_sql = '''
                        SELECT file_id FROM models 
                        WHERE num_components > 5
                        AND num_faces <= ?
                    '''
                elif cat == "multiple_components":
                    base_sql = '''
                        SELECT file_id FROM models 
                        WHERE num_components > 1 AND num_components <= 5
                        AND num_faces <= ?
                    '''
                elif cat == "complex":
                    # Multiple defect types
                    base_sql = '''
                        SELECT file_id FROM models 
                        WHERE (CAST(has_holes AS INT) + CAST(has_non_manifold AS INT) + 
                               CAST(has_self_intersections AS INT) + CAST(has_multiple_components AS INT)) >= 2
                        AND num_faces <= ?
                    '''
                else:
                    # Fallback to primary_category
                    base_sql = f'''
                        SELECT file_id FROM models 
                        WHERE primary_category = '{cat}'
                        AND num_faces <= ?
                    '''
                
                # Add license filter and ordering
                sql = base_sql + license_filter + " ORDER BY num_faces ASC LIMIT ?"
                
                rows = conn.execute(sql, (max_faces, per_category)).fetchall()
                selected[cat] = [row[0] for row in rows]
                logger.info(f"  {cat}: {len(selected[cat])} models")
        
        return selected
    
    def copy_fixtures_to_test_dir(
        self,
        selected: dict[str, list[int]],
        overwrite: bool = False
    ):
        """Copy selected fixtures to the test fixtures directory."""
        
        for category, file_ids in selected.items():
            cat_dir = self.fixtures_path / category
            cat_dir.mkdir(parents=True, exist_ok=True)
            
            for file_id in file_ids:
                src = self.raw_meshes_path / f"{file_id}.stl"
                dst = cat_dir / f"{file_id}.stl"
                
                if not src.exists():
                    logger.warning(f"Source file not found: {src}")
                    continue
                
                if dst.exists() and not overwrite:
                    continue
                
                shutil.copy(src, dst)
            
            logger.info(f"Copied {len(file_ids)} files to {cat_dir}")
        
        # Save index
        index_path = self.fixtures_path / "thingi10k_index.json"
        with open(index_path, "w") as f:
            json.dump(selected, f, indent=2)
        
        logger.info(f"Saved index to {index_path}")
    
    def get_stl_path(self, file_id: int) -> Optional[Path]:
        """Get the path to a model's STL file."""
        path = self.raw_meshes_path / f"{file_id}.stl"
        return path if path.exists() else None
    
    def print_summary(self):
        """Print summary statistics."""
        total = self.db.count_models()
        summary = self.db.get_category_summary()
        
        print(f"\nThingi10K Database Summary")
        print(f"=" * 50)
        print(f"Total models: {total}")
        print(f"\nBy Category:")
        
        for cat, stats in sorted(summary.items()):
            pct = stats["model_count"] / total * 100 if total > 0 else 0
            print(f"  {cat}: {stats['model_count']} ({pct:.1f}%)")
            print(f"    Avg faces: {stats['avg_faces']:.0f}")
            print(f"    Watertight: {stats['watertight_count']}")
            print(f"    Manifold: {stats['manifold_count']}")


# CLI interface
def main():
    """Command-line interface for Thingi10K management."""
    import argparse
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    parser = argparse.ArgumentParser(description="Thingi10K Dataset Manager")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Import command
    import_parser = subparsers.add_parser("import", help="Import metadata")
    import_parser.add_argument(
        "--source", 
        choices=["package", "directory"],
        default="package",
        help="Import source"
    )
    import_parser.add_argument(
        "--raw-meshes",
        type=Path,
        default=DEFAULT_RAW_MESHES_PATH,
        help="Path to raw_meshes directory"
    )
    
    # Select command
    select_parser = subparsers.add_parser("select", help="Select test fixtures")
    select_parser.add_argument(
        "--per-category",
        type=int,
        default=20,
        help="Models per category"
    )
    select_parser.add_argument(
        "--max-faces",
        type=int,
        default=100000,
        help="Maximum face count"
    )
    select_parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy fixtures to test directory"
    )
    select_parser.add_argument(
        "--raw-meshes",
        type=Path,
        default=DEFAULT_RAW_MESHES_PATH,
        help="Path to raw_meshes directory"
    )
    
    # Summary command
    subparsers.add_parser("summary", help="Print database summary")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export to JSON")
    export_parser.add_argument(
        "--output",
        type=Path,
        default=Path("thingi10k_metadata.json"),
        help="Output file path"
    )
    
    args = parser.parse_args()
    
    raw_meshes = getattr(args, 'raw_meshes', DEFAULT_RAW_MESHES_PATH)
    manager = Thingi10KManager(raw_meshes_path=raw_meshes)
    
    if args.command == "import":
        if args.source == "package":
            count = manager.import_from_thingi10k_package()
        else:
            count = manager.import_from_directory()
        print(f"Imported {count} models")
        manager.print_summary()
    
    elif args.command == "select":
        selected = manager.select_test_fixtures(
            per_category=args.per_category,
            max_faces=args.max_faces
        )
        
        if args.copy:
            manager.copy_fixtures_to_test_dir(selected)
        else:
            print("\nSelected fixtures (use --copy to copy files):")
            for cat, ids in selected.items():
                print(f"  {cat}: {len(ids)} models")
    
    elif args.command == "summary":
        manager.print_summary()
    
    elif args.command == "export":
        manager.db.export_to_json(args.output)
        print(f"Exported to {args.output}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
