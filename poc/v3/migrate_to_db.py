# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep - https://github.com/DragonAceNL/MeshPrep

"""
Migrate existing results from CSV to SQLite database.

Run this once to populate the database from existing results.csv.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import RESULTS_CSV
from progress_db import get_progress_db, ModelResult, Progress


def migrate_csv_to_db():
    """Migrate results from CSV to SQLite database."""
    if not RESULTS_CSV.exists():
        print(f"CSV file not found: {RESULTS_CSV}")
        return
    
    db = get_progress_db()
    
    # Read CSV and insert into database
    with open(RESULTS_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        count = 0
        success_count = 0
        precheck_skipped_count = 0
        escalation_count = 0
        total_duration = 0
        
        for row in reader:
            result = ModelResult(
                file_id=row['file_id'],
                model_fingerprint=row.get('model_fingerprint', ''),
                file_path='',
                success=row['success'].lower() == 'true',
                filter_used=row.get('filter_used', ''),
                escalation_used=row.get('escalation_used', '').lower() == 'true',
                error=row.get('error', ''),
                precheck_passed=row.get('precheck_passed', '').lower() == 'true',
                precheck_skipped=row.get('precheck_skipped', '').lower() == 'true',
                original_vertices=int(row.get('original_vertices', 0) or 0),
                original_faces=int(row.get('original_faces', 0) or 0),
                original_volume=float(row.get('original_volume', 0) or 0),
                original_watertight=row.get('original_watertight', '').lower() == 'true',
                original_manifold=row.get('original_manifold', '').lower() == 'true',
                original_components=int(row.get('original_components', 1) or 1),
                original_holes=int(row.get('original_holes', 0) or 0),
                original_file_size=int(row.get('original_file_size', 0) or 0),
                result_vertices=int(row.get('result_vertices', 0) or 0),
                result_faces=int(row.get('result_faces', 0) or 0),
                result_volume=float(row.get('result_volume', 0) or 0),
                result_watertight=row.get('result_watertight', '').lower() == 'true',
                result_manifold=row.get('result_manifold', '').lower() == 'true',
                result_components=int(row.get('result_components', 1) or 1),
                result_holes=int(row.get('result_holes', 0) or 0),
                fixed_file_size=int(row.get('fixed_file_size', 0) or 0),
                volume_change_pct=float(row.get('volume_change_pct', 0) or 0),
                face_change_pct=float(row.get('face_change_pct', 0) or 0),
                duration_ms=float(row.get('duration_ms', 0) or 0),
                timestamp=row.get('timestamp', ''),
            )
            
            db.save_result(result)
            count += 1
            
            if result.success:
                success_count += 1
            if result.precheck_skipped:
                precheck_skipped_count += 1
            if result.escalation_used:
                escalation_count += 1
            total_duration += result.duration_ms
            
            if count % 100 == 0:
                print(f"  Migrated {count} results...")
    
    print(f"\nMigrated {count} results from CSV to SQLite database")
    print(f"  Successful: {success_count}")
    print(f"  Already clean: {precheck_skipped_count}")
    print(f"  Escalations: {escalation_count}")
    
    # Update progress with totals
    progress = Progress(
        total_files=count,
        processed=count,
        successful=success_count,
        failed=count - success_count,
        precheck_skipped=precheck_skipped_count,
        escalations=escalation_count,
        total_duration_ms=total_duration,
        avg_duration_ms=total_duration / count if count > 0 else 0,
    )
    db.save_progress(progress)
    print(f"\nUpdated progress: {count} processed, {success_count} successful")


if __name__ == "__main__":
    migrate_csv_to_db()
