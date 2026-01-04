# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Reports index HTML generator for MeshPrep.

Generates the index page listing all processed models
with filtering, sorting, pagination, and navigation capabilities.

Now loads data from SQLite database instead of filter JSON files.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from html_helpers import (
    COLORS,
    get_base_styles,
    get_stats_row_styles,
    get_filter_styles,
    get_table_styles,
    format_duration,
    get_status_info,
)
from progress_db import get_progress_db, ModelResult

logger = logging.getLogger(__name__)

# Default pagination settings
DEFAULT_PAGE_SIZE = 50
PAGE_SIZE_OPTIONS = [25, 50, 100, 200, 500]


def _get_index_styles() -> str:
    """Get CSS styles specific to index page."""
    return f"""
        .pagination {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        .pagination button {{
            background: {COLORS['background_secondary']};
            color: {COLORS['text_primary']};
            border: 1px solid {COLORS['background_tertiary']};
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
        }}
        .pagination button:hover:not(:disabled) {{
            background: {COLORS['background_tertiary']};
            border-color: {COLORS['accent']};
        }}
        .pagination button:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}
        .pagination button.active {{
            background: {COLORS['accent']};
            color: {COLORS['background']};
            border-color: {COLORS['accent']};
        }}
        .pagination .page-info {{
            color: {COLORS['text_secondary']};
            font-size: 14px;
            padding: 0 15px;
        }}
        .pagination select {{
            background: {COLORS['background_secondary']};
            color: {COLORS['text_primary']};
            border: 1px solid {COLORS['background_tertiary']};
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 14px;
        }}
        .page-numbers {{
            display: flex;
            gap: 5px;
        }}
        .page-numbers button {{
            min-width: 40px;
        }}
        .results-info {{
            color: {COLORS['text_secondary']};
            font-size: 14px;
            margin-bottom: 15px;
        }}
"""


def load_results_from_database() -> List[ModelResult]:
    """Load all results from the SQLite database.
    
    Returns:
        List of ModelResult objects from database
    """
    db = get_progress_db()
    return db.get_all_results()


def generate_reports_index(
    results: Optional[List[ModelResult]] = None,
    reports_path: Optional[Path] = None,
    filters_path: Optional[Path] = None,  # Kept for backward compatibility, ignored
    page_size: int = DEFAULT_PAGE_SIZE,
) -> None:
    """Generate an index.html in the reports folder for easy navigation.
    
    Args:
        results: List of ModelResult objects (if None, loads from database)
        reports_path: Path to reports directory (uses default if None)
        filters_path: Ignored (kept for backward compatibility)
        page_size: Number of results per page (default: 50)
    """
    # Default reports path
    if reports_path is None:
        from config import REPORTS_PATH
        reports_path = REPORTS_PATH
    
    # Load from database if no results passed
    if not results:
        results = load_results_from_database()
    
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
    
    # Generate page size options HTML
    page_size_options_html = "\n".join(
        f'                <option value="{size}"{" selected" if size == page_size else ""}>{size}</option>'
        for size in PAGE_SIZE_OPTIONS
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
            <select id="statusFilter" onchange="applyFilters()">
                <option value="all">All</option>
                <option value="success">Successful</option>
                <option value="failed">Failed</option>
                <option value="skipped">Already Clean</option>
                <option value="escalated">Blender Escalation</option>
            </select>
            
            <label>Search:</label>
            <input type="text" id="searchBox" class="search-box" placeholder="Search by model ID..." onkeyup="applyFilters()">
            
            <label>Per page:</label>
            <select id="pageSizeSelect" onchange="changePageSize()">
{page_size_options_html}
            </select>
        </div>
        
        <div class="results-info" id="resultsInfo">
            Showing all {total} results
        </div>
        
        <div class="pagination" id="paginationTop"></div>
        
        <table id="resultsTable">
            <thead>
                <tr>
                    <th onclick="sortTable(0)">Model ID &#8645;</th>
                    <th onclick="sortTable(1)">Status &#8645;</th>
                    <th onclick="sortTable(2)">Filter &#8645;</th>
                    <th onclick="sortTable(3)">Duration &#8645;</th>
                    <th onclick="sortTable(4)">Faces Before &#8645;</th>
                    <th onclick="sortTable(5)">Faces After &#8645;</th>
                    <th>Report</th>
                </tr>
            </thead>
            <tbody id="tableBody">
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
        
        <div class="pagination" id="paginationBottom"></div>
        
        <p style="margin-top: 30px; color: {COLORS['text_muted']}; font-size: 12px;">
            <a href="/live">&#128202; Live Dashboard</a> | 
            <a href="/learning">&#129504; Learning Status</a> |
            <a href="javascript:location.reload()">Refresh</a>
        </p>
    </div>
    
    <script>
        // Pagination state
        let currentPage = 1;
        let pageSize = {page_size};
        let filteredRows = [];
        let allRows = [];
        let sortDirection = {{}};
        let currentSortColumn = -1;
        
        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {{
            allRows = Array.from(document.querySelectorAll('#tableBody tr'));
            
            // Read state from URL
            const params = new URLSearchParams(window.location.search);
            if (params.has('page')) currentPage = parseInt(params.get('page')) || 1;
            if (params.has('size')) {{
                pageSize = parseInt(params.get('size')) || {page_size};
                document.getElementById('pageSizeSelect').value = pageSize;
            }}
            if (params.has('filter')) {{
                document.getElementById('statusFilter').value = params.get('filter');
            }}
            if (params.has('search')) {{
                document.getElementById('searchBox').value = params.get('search');
            }}
            
            applyFilters();
        }});
        
        function updateURL() {{
            const params = new URLSearchParams();
            if (currentPage > 1) params.set('page', currentPage);
            if (pageSize !== {page_size}) params.set('size', pageSize);
            
            const filter = document.getElementById('statusFilter').value;
            if (filter !== 'all') params.set('filter', filter);
            
            const search = document.getElementById('searchBox').value;
            if (search) params.set('search', search);
            
            const newURL = window.location.pathname + (params.toString() ? '?' + params.toString() : '');
            window.history.replaceState({{}}, '', newURL);
        }}
        
        function applyFilters() {{
            const statusFilter = document.getElementById('statusFilter').value;
            const searchText = document.getElementById('searchBox').value.toLowerCase();
            
            filteredRows = allRows.filter(row => {{
                const status = row.getAttribute('data-status');
                const modelId = row.cells[0].textContent.toLowerCase();
                
                const statusMatch = statusFilter === 'all' || status === statusFilter;
                const searchMatch = modelId.includes(searchText);
                
                return statusMatch && searchMatch;
            }});
            
            // Reset to page 1 when filters change
            currentPage = 1;
            renderPage();
            updateURL();
        }}
        
        function changePageSize() {{
            pageSize = parseInt(document.getElementById('pageSizeSelect').value);
            currentPage = 1;
            renderPage();
            updateURL();
        }}
        
        function goToPage(page) {{
            const totalPages = Math.ceil(filteredRows.length / pageSize);
            currentPage = Math.max(1, Math.min(page, totalPages));
            renderPage();
            updateURL();
            
            // Scroll to top of table
            document.getElementById('resultsTable').scrollIntoView({{ behavior: 'smooth' }});
        }}
        
        function renderPage() {{
            const totalPages = Math.ceil(filteredRows.length / pageSize) || 1;
            const startIdx = (currentPage - 1) * pageSize;
            const endIdx = startIdx + pageSize;
            
            // Hide all rows first
            allRows.forEach(row => row.style.display = 'none');
            
            // Show only rows for current page
            filteredRows.slice(startIdx, endIdx).forEach(row => row.style.display = '');
            
            // Update results info
            const showing = filteredRows.length === 0 ? 0 : Math.min(endIdx, filteredRows.length) - startIdx;
            const infoText = filteredRows.length === allRows.length
                ? `Showing ${{startIdx + 1}}-${{Math.min(endIdx, filteredRows.length)}} of ${{filteredRows.length}} results`
                : `Showing ${{startIdx + 1}}-${{Math.min(endIdx, filteredRows.length)}} of ${{filteredRows.length}} filtered results (from ${{allRows.length}} total)`;
            document.getElementById('resultsInfo').textContent = filteredRows.length > 0 ? infoText : 'No results match your filters';
            
            // Render pagination controls
            renderPagination('paginationTop', totalPages);
            renderPagination('paginationBottom', totalPages);
        }}
        
        function renderPagination(containerId, totalPages) {{
            const container = document.getElementById(containerId);
            
            if (filteredRows.length <= pageSize) {{
                container.innerHTML = '';
                return;
            }}
            
            let html = `
                <button onclick="goToPage(1)" ${{currentPage === 1 ? 'disabled' : ''}}>&laquo; First</button>
                <button onclick="goToPage(currentPage - 1)" ${{currentPage === 1 ? 'disabled' : ''}}>&lsaquo; Prev</button>
                <div class="page-numbers">
            `;
            
            // Calculate which page numbers to show
            const maxButtons = 7;
            let startPage = Math.max(1, currentPage - Math.floor(maxButtons / 2));
            let endPage = Math.min(totalPages, startPage + maxButtons - 1);
            
            if (endPage - startPage < maxButtons - 1) {{
                startPage = Math.max(1, endPage - maxButtons + 1);
            }}
            
            if (startPage > 1) {{
                html += `<button onclick="goToPage(1)">1</button>`;
                if (startPage > 2) html += `<span style="color: #888;">...</span>`;
            }}
            
            for (let i = startPage; i <= endPage; i++) {{
                html += `<button onclick="goToPage(${{i}})" class="${{i === currentPage ? 'active' : ''}}">${{i}}</button>`;
            }}
            
            if (endPage < totalPages) {{
                if (endPage < totalPages - 1) html += `<span style="color: #888;">...</span>`;
                html += `<button onclick="goToPage(${{totalPages}})">${{totalPages}}</button>`;
            }}
            
            html += `
                </div>
                <button onclick="goToPage(currentPage + 1)" ${{currentPage === totalPages ? 'disabled' : ''}}>Next &rsaquo;</button>
                <button onclick="goToPage(${{totalPages}})" ${{currentPage === totalPages ? 'disabled' : ''}}>Last &raquo;</button>
                <span class="page-info">Page ${{currentPage}} of ${{totalPages}}</span>
            `;
            
            container.innerHTML = html;
        }}
        
        function sortTable(columnIndex) {{
            // Toggle direction
            if (currentSortColumn === columnIndex) {{
                sortDirection[columnIndex] = !sortDirection[columnIndex];
            }} else {{
                sortDirection[columnIndex] = true; // ascending by default
                currentSortColumn = columnIndex;
            }}
            
            const direction = sortDirection[columnIndex] ? 1 : -1;
            
            // Sort all rows (affects filteredRows through reference)
            allRows.sort((a, b) => {{
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
            
            // Re-append sorted rows to tbody
            const tbody = document.getElementById('tableBody');
            allRows.forEach(row => tbody.appendChild(row));
            
            // Re-apply filters to update filteredRows
            applyFilters();
        }}
    </script>
</body>
</html>
"""
    
    with open(index_path, "w", encoding='utf-8') as f:
        f.write(html)
    
    logger.info(f"Generated reports index: {index_path}")
