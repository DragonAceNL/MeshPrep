# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Reports index HTML generator for MeshPrep.

Generates the index page listing all processed models
with filtering, sorting, and navigation capabilities.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from html_helpers import (
    COLORS,
    get_base_styles,
    get_stats_row_styles,
    get_filter_styles,
    get_table_styles,
    format_duration,
    get_status_info,
)

if TYPE_CHECKING:
    from test_result import TestResult

logger = logging.getLogger(__name__)


def _get_index_styles() -> str:
    """Get CSS styles specific to index page."""
    return ""  # Most styles come from helpers


def load_results_from_reports(reports_path: Path, filters_path: Path) -> List["TestResult"]:
    """Load TestResult objects from existing filter JSON files.
    
    This allows regenerating the index without re-running processing.
    
    Args:
        reports_path: Path to reports directory
        filters_path: Path to filters directory
        
    Returns:
        List of TestResult objects reconstructed from filter files
    """
    # Import here to avoid circular import
    from test_result import TestResult
    
    results = []
    
    for filter_file in filters_path.glob("*.json"):
        try:
            with open(filter_file, encoding="utf-8") as f:
                data = json.load(f)
            
            # Reconstruct TestResult from filter data
            result = TestResult(
                file_id=data.get("model_id", filter_file.stem),
                file_path=data.get("original_filename", ""),
                success=data.get("success", False),
                filter_used=data.get("filter_name", "unknown"),
                escalation_used=data.get("escalated_to_blender", False),
                model_fingerprint=data.get("model_fingerprint", ""),
                timestamp=data.get("timestamp", ""),
            )
            
            # Extract precheck info
            precheck = data.get("precheck", {})
            result.precheck_passed = precheck.get("passed", False)
            result.precheck_skipped = precheck.get("skipped", False)
            
            # Extract diagnostics
            diag = data.get("diagnostics", {})
            before = diag.get("before", {}) or {}
            after = diag.get("after", {}) or {}
            
            result.original_vertices = before.get("vertices", 0)
            result.original_faces = before.get("faces", 0)
            result.original_volume = before.get("volume", 0) or 0
            result.original_watertight = before.get("is_watertight", False)
            result.original_manifold = before.get("is_watertight", False)  # Approximation
            
            result.result_vertices = after.get("vertices", 0)
            result.result_faces = after.get("faces", 0)
            result.result_volume = after.get("volume", 0) or 0
            result.result_watertight = after.get("is_watertight", False)
            result.result_manifold = after.get("is_watertight", False)
            
            # Extract repair attempt info for duration
            repair_attempts = data.get("repair_attempts", {})
            result.duration_ms = repair_attempts.get("total_duration_ms", 0)
            
            results.append(result)
            
        except Exception as e:
            logger.warning(f"Failed to load result from {filter_file}: {e}")
            continue
    
    return results


def generate_reports_index(
    results: Optional[List["TestResult"]],
    reports_path: Path,
    filters_path: Path,
) -> None:
    """Generate an index.html in the reports folder for easy navigation.
    
    Args:
        results: List of TestResult objects (if None, loads from reports)
        reports_path: Path to reports directory
        filters_path: Path to filters directory
    """
    # If no results passed, load from existing reports
    if not results:
        results = load_results_from_reports(reports_path, filters_path)
    
    if not results:
        logger.warning("No results to generate index from")
        return
    
    index_path = reports_path / "index.html"
    
    # Sort results by file_id
    sorted_results = sorted(results, key=lambda r: r.file_id)
    
    # Calculate stats
    total = len(results)
    successful = sum(1 for r in results if r.success)
    failed = total - successful
    precheck_skipped = sum(1 for r in results if r.precheck_skipped)
    escalations = sum(1 for r in results if r.escalation_used)
    
    # Build styles
    all_styles = (
        get_base_styles() +
        get_stats_row_styles() +
        get_filter_styles() +
        get_table_styles() +
        _get_index_styles()
    )
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>MeshPrep Reports Index</title>
    <style>
{all_styles}
    </style>
</head>
<body>
    <div class="container">
        <h1>&#128203; MeshPrep Reports Index</h1>
        <p class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Total: {total} models</p>
        
        <div class="stats-row">
            <div class="stat">
                <div class="stat-value">{total}</div>
                <div class="stat-label">Total</div>
            </div>
            <div class="stat">
                <div class="stat-value success">{successful}</div>
                <div class="stat-label">Successful</div>
            </div>
            <div class="stat">
                <div class="stat-value failed">{failed}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat">
                <div class="stat-value skipped">{precheck_skipped}</div>
                <div class="stat-label">Already Clean</div>
            </div>
            <div class="stat">
                <div class="stat-value escalated">{escalations}</div>
                <div class="stat-label">Blender</div>
            </div>
        </div>
        
        <div class="filters">
            <label>Filter:</label>
            <select id="statusFilter" onchange="filterTable()">
                <option value="all">All</option>
                <option value="success">Successful</option>
                <option value="failed">Failed</option>
                <option value="skipped">Already Clean</option>
                <option value="escalated">Blender Escalation</option>
            </select>
            
            <label>Search:</label>
            <input type="text" id="searchBox" class="search-box" placeholder="Search by model ID..." onkeyup="filterTable()">
        </div>
        
        <table id="resultsTable">
            <thead>
                <tr>
                    <th onclick="sortTable(0)">Model ID</th>
                    <th onclick="sortTable(1)">Status</th>
                    <th onclick="sortTable(2)">Filter</th>
                    <th onclick="sortTable(3)">Duration</th>
                    <th onclick="sortTable(4)">Faces Before</th>
                    <th onclick="sortTable(5)">Faces After</th>
                    <th>Report</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for r in sorted_results:
        # Status
        status_class, status_text = get_status_info(r.success, r.precheck_skipped, r.escalation_used)
        
        # Determine data attribute for filtering
        if r.precheck_skipped:
            status_data = "skipped"
        elif r.success:
            status_data = "escalated" if r.escalation_used else "success"
        else:
            status_data = "failed"
        
        duration_text = format_duration(r.duration_ms)
        
        html += f"""                <tr data-status="{status_data}">
                    <td><strong>{r.file_id}</strong></td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{r.filter_used}</td>
                    <td>{duration_text}</td>
                    <td>{r.original_faces:,}</td>
                    <td>{r.result_faces:,}</td>
                    <td><a href="{r.file_id}.html">View</a></td>
                </tr>
"""
    
    html += f"""            </tbody>
        </table>
        
        <p style="margin-top: 30px; color: {COLORS['text_muted']}; font-size: 12px;">
            <a href="/dashboard">&#128202; Dashboard</a> | 
            <a href="javascript:location.reload()">Refresh</a>
        </p>
    </div>
    
    <script>
        function filterTable() {{
            const statusFilter = document.getElementById('statusFilter').value;
            const searchText = document.getElementById('searchBox').value.toLowerCase();
            const rows = document.querySelectorAll('#resultsTable tbody tr');
            
            rows.forEach(row => {{
                const status = row.getAttribute('data-status');
                const modelId = row.cells[0].textContent.toLowerCase();
                
                const statusMatch = statusFilter === 'all' || status === statusFilter;
                const searchMatch = modelId.includes(searchText);
                
                row.style.display = (statusMatch && searchMatch) ? '' : 'none';
            }});
        }}
        
        let sortDirection = {{}};
        
        function sortTable(columnIndex) {{
            const table = document.getElementById('resultsTable');
            const tbody = table.tBodies[0];
            const rows = Array.from(tbody.rows);
            
            sortDirection[columnIndex] = !sortDirection[columnIndex];
            const direction = sortDirection[columnIndex] ? 1 : -1;
            
            rows.sort((a, b) => {{
                let aVal = a.cells[columnIndex].textContent.trim();
                let bVal = b.cells[columnIndex].textContent.trim();
                
                // Try numeric comparison
                const aNum = parseFloat(aVal.replace(/,/g, ''));
                const bNum = parseFloat(bVal.replace(/,/g, ''));
                
                if (!isNaN(aNum) && !isNaN(bNum)) {{
                    return (aNum - bNum) * direction;
                }}
                
                return aVal.localeCompare(bVal) * direction;
            }});
            
            rows.forEach(row => tbody.appendChild(row));
        }}
    </script>
</body>
</html>
"""
    
    with open(index_path, "w", encoding='utf-8') as f:
        f.write(html)
    
    logger.info(f"Generated reports index: {index_path}")
