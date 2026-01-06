# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Local HTTP server for MeshPrep reports with MeshLab integration.

This server:
1. Serves the reports and STL files via HTTP
2. Provides API endpoints for progress data (from SQLite)
3. Provides an API endpoint to open STL files in MeshLab
4. Provides an API endpoint to rate models

Usage:
    python reports_server.py [--port 8000]
    
Then open: http://localhost:8000/reports/
"""

import argparse
import http.server
import json
import os
import shutil
import socketserver
import subprocess
import sys
import urllib.parse
from pathlib import Path

# Configuration
THINGI10K_BASE = Path(r"C:\Users\Dragon Ace\Source\repos\Thingi10K")
THINGI10K_RAW_MESHES = THINGI10K_BASE / "raw_meshes"
THINGI10K_REPORTS = THINGI10K_BASE / "reports"
THINGI10K_FIXED = THINGI10K_BASE / "fixed"
POC_V3_PATH = Path(r"C:\Users\Dragon Ace\Source\repos\MeshPrep\poc\v3")
MESHLAB_PATHS = [
    r"C:\Program Files\VCG\MeshLab\meshlab.exe",
    r"C:\Program Files (x86)\VCG\MeshLab\meshlab.exe",
    r"C:\Program Files\MeshLab\meshlab.exe",
    shutil.which("meshlab"),  # Check PATH
]


def find_meshlab() -> str | None:
    """Find MeshLab executable."""
    for path in MESHLAB_PATHS:
        if path and Path(path).exists():
            return path
    return None


class SilentTCPServer(socketserver.TCPServer):
    """TCPServer that silently handles connection errors."""
    
    def handle_error(self, request, client_address):
        """Handle errors silently for common connection issues."""
        exc_type, exc_value, _ = sys.exc_info()
        
        # Silently ignore connection abort/reset errors (browser closed connection)
        if exc_type in (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            return
        
        # For other errors, use default handling
        super().handle_error(request, client_address)


class MeshLabHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler with MeshLab integration and rating API."""
    
    def __init__(self, *args, **kwargs):
        # Don't set directory here - we'll handle routing manually
        super().__init__(*args, **kwargs)
    
    def handle(self):
        """Handle requests with connection error suppression."""
        try:
            super().handle()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            # Browser closed connection - this is normal, ignore silently
            pass
    
    def translate_path(self, path):
        """Translate URL path to filesystem path with multi-directory support."""
        # Remove query string and fragment
        path = path.split('?')[0].split('#')[0]
        path = urllib.parse.unquote(path)
        
        # Route based on path prefix
        if path == '/reports' or path == '/reports/':
            # Serve index.html from reports directory
            index_path = THINGI10K_REPORTS / "index.html"
            if not index_path.exists():
                # Generate empty index if reports dir exists but index doesn't
                self._generate_empty_reports_index()
            return str(index_path)
        elif path.startswith('/reports/'):
            # Serve from Thingi10K/reports/ directory
            rel_path = path[len('/reports/'):]
            if rel_path:
                return str(THINGI10K_REPORTS / rel_path)
            return str(THINGI10K_REPORTS / "index.html")
        elif path.startswith('/fixed/'):
            # Serve from Thingi10K/fixed/ directory
            rel_path = path[len('/fixed/'):]
            return str(THINGI10K_FIXED / rel_path)
        elif path.startswith('/raw_meshes/') or path.startswith('/meshes/'):
            # Serve from Thingi10K/raw_meshes/ directory
            if path.startswith('/raw_meshes/'):
                rel_path = path[len('/raw_meshes/')]
            else:
                rel_path = path[len('/meshes/'):]
            return str(THINGI10K_RAW_MESHES / rel_path)
        elif path.startswith('/live_dashboard') or path == '/live' or path == '/':
            # Live dashboard is the main entry point
            return str(POC_V3_PATH / 'live_dashboard.html')
        elif path.startswith('/learning') or path == '/learning-status':
            # Learning status page - generate fresh if doesn't exist
            status_path = POC_V3_PATH / 'learning_status.html'
            if not status_path.exists():
                try:
                    # Add POC v3 to path for imports
                    sys.path.insert(0, str(POC_V3_PATH))
                    from generate_learning_status import generate_learning_status_page
                    generate_learning_status_page()
                except Exception:
                    pass
            return str(status_path)
        else:
            # Default: serve from Thingi10K base for backward compatibility
            if path.startswith('/'):
                path = path[1:]
            return str(THINGI10K_BASE / path)
    
    def do_GET(self):
        """Handle GET requests."""
        parsed = urllib.parse.urlparse(self.path)
        
        # API endpoint for progress data (from SQLite)
        if parsed.path == "/api/progress" or parsed.path == "/progress.json":
            self.handle_get_progress()
            return
        
        # API endpoint to open file in MeshLab
        if parsed.path == "/api/open-meshlab":
            self.handle_open_meshlab(parsed.query)
            return
        
        # API endpoint to check MeshLab availability
        if parsed.path == "/api/meshlab-status":
            self.handle_meshlab_status()
            return
        
        # API endpoint to regenerate learning status page
        if parsed.path == "/api/refresh-learning-status":
            self.handle_refresh_learning_status()
            return
        
        # API endpoint for error logs
        if parsed.path == "/api/errors":
            self.handle_get_errors(parsed.query)
            return
        
        # Error logs HTML page
        if parsed.path == "/errors" or parsed.path == "/errors/":
            self.handle_errors_page(parsed.query)
            return
        
        # API endpoint to get rating for a model
        if parsed.path == "/api/get-rating":
            self.handle_get_rating(parsed.query)
            return
        
        # Default: serve static files
        super().do_GET()
    
    def do_POST(self):
        """Handle POST requests."""
        parsed = urllib.parse.urlparse(self.path)
        
        # API endpoint to rate a model
        if parsed.path == "/api/rate-model":
            self.handle_rate_model()
            return
        
        # Default: method not allowed
        self.send_response(405)
        self.end_headers()
    
    def handle_get_progress(self):
        """Get progress data from SQLite database."""
        try:
            # Add POC v3 to path for imports
            sys.path.insert(0, str(POC_V3_PATH))
            from progress_db import get_progress_db
            
            db = get_progress_db()
            progress = db.get_progress()
            
            # Return as JSON (compatible with live_dashboard.html)
            response = progress.to_dict()
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            print(f"[Progress] Error getting progress: {e}")
            self.send_json_error(500, f"Failed to get progress: {e}")
    
    def handle_meshlab_status(self):
        """Check if MeshLab is available."""
        meshlab = find_meshlab()
        response = {
            "available": meshlab is not None,
            "path": meshlab
        }
        
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
    
    def handle_refresh_learning_status(self):
        """Regenerate the learning status page."""
        try:
            sys.path.insert(0, str(POC_V3_PATH))
            from generate_learning_status import generate_learning_status_page
            page_path = generate_learning_status_page()
            response = {
                "success": True,
                "path": str(page_path),
                "message": "Learning status page regenerated"
            }
            self.send_response(200)
        except Exception as e:
            response = {
                "success": False,
                "error": str(e)
            }
            self.send_response(500)
        
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
    
    def handle_get_rating(self, query_string: str):
        """Get existing rating for a model by fingerprint."""
        params = urllib.parse.parse_qs(query_string)
        fingerprint = params.get("fingerprint", [None])[0]
        
        if not fingerprint:
            self.send_json_error(400, "Missing 'fingerprint' parameter")
            return
        
        try:
            # Add POC v2 to path for imports
            sys.path.insert(0, str(POC_V3_PATH.parent / "v2"))
            from meshprep_poc.quality_feedback import get_quality_engine
            
            quality_engine = get_quality_engine()
            rating = quality_engine.get_rating_by_fingerprint(fingerprint)
            
            if rating:
                response = {
                    "success": True,
                    "rating": {
                        "rating_value": rating.rating_value,
                        "user_comment": rating.user_comment,
                        "rated_at": rating.rated_at,
                        "rated_by": rating.rated_by,
                    }
                }
            else:
                response = {
                    "success": True,
                    "rating": None
                }
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            print(f"[Rating] Error getting rating: {e}")
            self.send_json_error(500, f"Failed to get rating: {e}")
    
    def handle_get_errors(self, query_string: str):
        """Get error log data as JSON with pagination."""
        params = urllib.parse.parse_qs(query_string)
        page = int(params.get("page", [1])[0])
        per_page = int(params.get("per_page", [50])[0])
        date_filter = params.get("date", [None])[0]
        category_filter = params.get("category", [None])[0]
        action_filter = params.get("action", [None])[0]
        
        try:
            # Add POC v2 to path for imports
            sys.path.insert(0, str(POC_V3_PATH.parent / "v2"))
            from meshprep_poc.error_logger import (
                get_all_error_logs,
                parse_error_log,
                get_error_summary,
            )
            from meshprep_poc.subprocess_executor import get_failure_tracker
            
            # Get all log files
            log_files = get_all_error_logs()
            
            # Collect all entries from all log files (or just the selected date)
            all_entries = []
            for log_file in log_files:
                if date_filter and date_filter not in log_file.name:
                    continue
                entries = parse_error_log(log_file)
                for entry in entries:
                    entry["_log_file"] = log_file.name
                    entry["_date"] = log_file.name.replace("errors_", "").replace(".log", "")
                all_entries.extend(entries)
            
            # Apply filters
            if category_filter:
                all_entries = [e for e in all_entries if e.get("category") == category_filter]
            if action_filter:
                all_entries = [e for e in all_entries if e.get("action") == action_filter]
            
            # Sort by timestamp (newest first)
            all_entries.sort(key=lambda x: (x.get("_date", ""), x.get("_timestamp", "")), reverse=True)
            
            # Pagination
            total = len(all_entries)
            total_pages = (total + per_page - 1) // per_page if per_page > 0 else 1
            start = (page - 1) * per_page
            end = start + per_page
            page_entries = all_entries[start:end]
            
            # Get summary stats
            categories = {}
            actions = {}
            for entry in all_entries:
                cat = entry.get("category", "unknown")
                act = entry.get("action", "unknown")
                categories[cat] = categories.get(cat, 0) + 1
                actions[act] = actions.get(act, 0) + 1
            
            # Get SQLite failure patterns
            tracker = get_failure_tracker()
            failure_stats = tracker.get_failure_stats()
            
            response = {
                "success": True,
                "page": page,
                "per_page": per_page,
                "total": total,
                "total_pages": total_pages,
                "entries": page_entries,
                "summary": {
                    "by_category": dict(sorted(categories.items(), key=lambda x: -x[1])),
                    "by_action": dict(sorted(actions.items(), key=lambda x: -x[1])),
                },
                "log_files": [f.name for f in log_files],
                "failure_patterns": failure_stats.get("patterns", [])[:10],
            }
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            print(f"[Errors] Error getting errors: {e}")
            import traceback
            traceback.print_exc()
            self.send_json_error(500, f"Failed to get errors: {e}")
    
    def handle_errors_page(self, query_string: str):
        """Serve the errors HTML page."""
        params = urllib.parse.parse_qs(query_string)
        page = int(params.get("page", [1])[0])
        
        html = self._generate_errors_html(page)
        
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))
    
    def _generate_empty_reports_index(self):
        """Generate an empty index.html file in the reports directory."""
        try:
            index_path = THINGI10K_REPORTS / "index.html"
            with open(index_path, 'w', encoding='utf-8') as f:
                f.write("<!DOCTYPE html>\n<html lang='en'>\n<head>\n")
                f.write("    <meta charset='UTF-8'>\n")
                f.write("    <meta name='viewport' content='width=device-width, initial-scale=1.0'>\n")
                f.write("    <title>MeshPrep Reports</title>\n")
                f.write("    <style>\n")
                f.write("        body { font-family: Arial, sans-serif; margin: 0; padding: 0; }\n")
                f.write("        h1 { background-color: #4CAF50; color: white; padding: 10px 0; text-align: center; }\n")
                f.write("        p { padding: 0 15px; }\n")
                f.write("    </style>\n")
                f.write("</head>\n<body>\n")
                f.write("    <h1>MeshPrep Reports</h1>\n")
                f.write("    <p>No reports found. Please run the MeshPrep test to generate reports.</p>\n")
                f.write("</body>\n</html>\n")
            
            print(f"[Reports] Created empty index.html: {index_path}")
        except Exception as e:
            print(f"[Reports] Failed to create index.html: {e}")
    
    def _generate_errors_html(self, current_page: int = 1) -> str:
        """Generate the errors page HTML with pagination."""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MeshPrep Error Logs</title>
    <style>
        :root {
            --bg-color: #1a1a2e;
            --card-bg: #16213e;
            --text-color: #eee;
            --text-muted: #888;
            --accent: #e94560;
            --success: #4ade80;
            --warning: #fbbf24;
            --error: #ef4444;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, sans-serif;
            background: var(--bg-color);
            color: var(--text-color);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        h1 {
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .nav-links {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        
        .nav-links a {
            color: var(--accent);
            text-decoration: none;
            padding: 8px 16px;
            background: var(--card-bg);
            border-radius: 6px;
            transition: all 0.2s;
        }
        
        .nav-links a:hover {
            background: var(--accent);
            color: white;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .stat-card {
            background: var(--card-bg);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        
        .stat-card .value {
            font-size: 2rem;
            font-weight: bold;
            color: var(--accent);
        }
        
        .stat-card .label {
            color: var(--text-muted);
            font-size: 0.9rem;
        }
        
        .filters {
            background: var(--card-bg);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            align-items: center;
        }
        
        .filters label {
            color: var(--text-muted);
        }
        
        .filters select, .filters input {
            background: var(--bg-color);
            color: var(--text-color);
            border: 1px solid #333;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 0.9rem;
        }
        
        .filters button {
            background: var(--accent);
            color: white;
            border: none;
            padding: 8px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9rem;
        }
        
        .filters button:hover {
            opacity: 0.9;
        }
        
        .error-table {
            width: 100%;
            border-collapse: collapse;
            background: var(--card-bg);
            border-radius: 8px;
            overflow: hidden;
        }
        
        .error-table th, .error-table td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #333;
        }
        
        .error-table th {
            background: rgba(233, 69, 96, 0.2);
            font-weight: 600;
            position: sticky;
            top: 0;
        }
        
        .error-table tr:hover {
            background: rgba(255, 255, 255, 0.05);
        }
        
        .error-table .error-msg {
            max-width: 400px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            font-family: monospace;
            font-size: 0.85rem;
            color: var(--text-muted);
        }
        
        .error-table .error-msg:hover {
            white-space: normal;
            word-break: break-word;
        }
        
        .badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 500;
        }
        
        .badge-error { background: rgba(239, 68, 68, 0.3); color: #f87171; }
        .badge-crash { background: rgba(239, 68, 68, 0.5); color: #fca5a5; }
        .badge-warning { background: rgba(251, 191, 36, 0.3); color: #fcd34d; }
        
        .badge-category { background: rgba(99, 102, 241, 0.3); color: #a5b4fc; }
        .badge-action { background: rgba(6, 182, 212, 0.3); color: #67e8f9; }
        
        .pagination {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
            margin-top: 20px;
            flex-wrap: wrap;
        }
        
        .pagination button {
            background: var(--card-bg);
            color: var(--text-color);
            border: 1px solid #333;
            padding: 8px 15px;
            border-radius: 4px;
            cursor: pointer;
        }
        
        .pagination button:hover:not(:disabled) {
            background: var(--accent);
            border-color: var(--accent);
        }
        
        .pagination button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .pagination .page-info {
            color: var(--text-muted);
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: var(--text-muted);
        }
        
        .summary-section {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .summary-card {
            background: var(--card-bg);
            padding: 15px;
            border-radius: 8px;
        }
        
        .summary-card h3 {
            margin-bottom: 10px;
            color: var(--accent);
            font-size: 1rem;
        }
        
        .summary-item {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #333;
        }
        
        .summary-item:last-child {
            border-bottom: none;
        }
        
        .timestamp {
            font-family: monospace;
            font-size: 0.85rem;
            color: var(--text-muted);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>MeshPrep Error Logs</h1>
        
        <div class="nav-links">
            <a href="/live">Dashboard</a>
            <a href="/reports/">Reports</a>
            <a href="/learning">Learning Status</a>
            <a href="/errors/" class="active">Error Logs</a>
        </div>
        
        <div class="stats-grid" id="stats">
            <div class="stat-card">
                <div class="value" id="total-errors">-</div>
                <div class="label">Total Errors</div>
            </div>
            <div class="stat-card">
                <div class="value" id="total-categories">-</div>
                <div class="label">Error Categories</div>
            </div>
            <div class="stat-card">
                <div class="value" id="total-actions">-</div>
                <div class="label">Actions with Errors</div>
            </div>
            <div class="stat-card">
                <div class="value" id="total-log-files">-</div>
                <div class="label">Log Files</div>
            </div>
        </div>
        
        <div class="filters">
            <label>Date:
                <select id="filter-date">
                    <option value="">All Dates</option>
                </select>
            </label>
            <label>Category:
                <select id="filter-category">
                    <option value="">All Categories</option>
                </select>
            </label>
            <label>Action:
                <select id="filter-action">
                    <option value="">All Actions</option>
                </select>
            </label>
            <label>Per Page:
                <select id="per-page">
                    <option value="25">25</option>
                    <option value="50" selected>50</option>
                    <option value="100">100</option>
                    <option value="200">200</option>
                </select>
            </label>
            <button onclick="applyFilters()">Apply</button>
            <button onclick="clearFilters()">Clear</button>
        </div>
        
        <div class="summary-section" id="summary-section">
            <div class="summary-card">
                <h3>Top Error Categories</h3>
                <div id="summary-categories">Loading...</div>
            </div>
            <div class="summary-card">
                <h3>Top Failing Actions</h3>
                <div id="summary-actions">Loading...</div>
            </div>
        </div>
        
        <div id="table-container">
            <div class="loading">Loading error logs...</div>
        </div>
        
        <div class="pagination" id="pagination"></div>
    </div>
    
    <script>
        let currentPage = 1;
        let currentData = null;
        
        async function loadErrors(page = 1) {
            const dateFilter = document.getElementById('filter-date').value;
            const categoryFilter = document.getElementById('filter-category').value;
            const actionFilter = document.getElementById('filter-action').value;
            const perPage = document.getElementById('per-page').value;
            
            let url = `/api/errors?page=${page}&per_page=${perPage}`;
            if (dateFilter) url += `&date=${encodeURIComponent(dateFilter)}`;
            if (categoryFilter) url += `&category=${encodeURIComponent(categoryFilter)}`;
            if (actionFilter) url += `&action=${encodeURIComponent(actionFilter)}`;
            
            try {
                const response = await fetch(url);
                const data = await response.json();
                
                if (data.success) {
                    currentData = data;
                    currentPage = data.page;
                    renderStats(data);
                    renderTable(data);
                    renderPagination(data);
                    updateFilters(data);
                    renderSummary(data);
                } else {
                    document.getElementById('table-container').innerHTML = 
                        `<div class="loading">Error: ${data.error}</div>`;
                }
            } catch (err) {
                document.getElementById('table-container').innerHTML = 
                    `<div class="loading">Error loading data: ${err.message}</div>`;
            }
        }
        
        function renderStats(data) {
            document.getElementById('total-errors').textContent = data.total.toLocaleString();
            document.getElementById('total-categories').textContent = 
                Object.keys(data.summary.by_category).length;
            document.getElementById('total-actions').textContent = 
                Object.keys(data.summary.by_action).length;
            document.getElementById('total-log-files').textContent = data.log_files.length;
        }
        
        function renderSummary(data) {
            // Categories
            const catHtml = Object.entries(data.summary.by_category)
                .slice(0, 8)
                .map(([cat, count]) => 
                    `<div class="summary-item"><span>${cat}</span><span>${count}</span></div>`
                ).join('');
            document.getElementById('summary-categories').innerHTML = catHtml || 'No data';
            
            // Actions
            const actHtml = Object.entries(data.summary.by_action)
                .slice(0, 8)
                .map(([act, count]) => 
                    `<div class="summary-item"><span>${act}</span><span>${count}</span></div>`
                ).join('');
            document.getElementById('summary-actions').innerHTML = actHtml || 'No data';
        }
        
        function renderTable(data) {
            if (data.entries.length === 0) {
                document.getElementById('table-container').innerHTML = 
                    '<div class="loading">No errors found matching the filters.</div>';
                return;
            }
            
            let html = `
                <table class="error-table">
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Date</th>
                            <th>Type</th>
                            <th>Category</th>
                            <th>Action</th>
                            <th>Model</th>
                            <th>Faces</th>
                            <th>Error Message</th>
                        </tr>
                    </thead>
                    <tbody>
            `;
            
            for (const entry of data.entries) {
                const typeClass = entry.type === 'CRASH' ? 'badge-crash' : 
                                  entry.type === 'error' ? 'badge-error' : 'badge-warning';
                
                html += `
                    <tr>
                        <td class="timestamp">${entry._timestamp || '-'}</td>
                        <td>${entry._date || '-'}</td>
                        <td><span class="badge ${typeClass}">${entry.type || '-'}</span></td>
                        <td><span class="badge badge-category">${entry.category || '-'}</span></td>
                        <td><span class="badge badge-action">${entry.action || '-'}</span></td>
                        <td>${entry.model_id || '-'}</td>
                        <td>${entry.faces?.toLocaleString() || '-'}</td>
                        <td class="error-msg" title="${escapeHtml(entry.error || '')}">${escapeHtml(entry.error || '-')}</td>
                    </tr>
                `;
            }
            
            html += '</tbody></table>';
            document.getElementById('table-container').innerHTML = html;
        }
        
        function renderPagination(data) {
            const { page, total_pages, total, per_page } = data;
            const start = (page - 1) * per_page + 1;
            const end = Math.min(page * per_page, total);
            
            let html = `
                <button onclick="loadErrors(1)" ${page === 1 ? 'disabled' : ''}>First</button>
                <button onclick="loadErrors(${page - 1})" ${page === 1 ? 'disabled' : ''}>Previous</button>
                <span class="page-info">Page ${page} of ${total_pages} (${start}-${end} of ${total})</span>
                <button onclick="loadErrors(${page + 1})" ${page === total_pages ? 'disabled' : ''}>Next</button>
                <button onclick="loadErrors(${total_pages})" ${page === total_pages ? 'disabled' : ''}>Last</button>
            `;
            
            document.getElementById('pagination').innerHTML = html;
        }
        
        function updateFilters(data) {
            // Update date filter
            const dateSelect = document.getElementById('filter-date');
            const currentDate = dateSelect.value;
            dateSelect.innerHTML = '<option value="">All Dates</option>';
            for (const file of data.log_files) {
                const date = file.replace('errors_', '').replace('.log', '');
                dateSelect.innerHTML += `<option value="${date}" ${date === currentDate ? 'selected' : ''}>${date}</option>`;
            }
            
            // Update category filter
            const catSelect = document.getElementById('filter-category');
            const currentCat = catSelect.value;
            catSelect.innerHTML = '<option value="">All Categories</option>';
            for (const cat of Object.keys(data.summary.by_category)) {
                catSelect.innerHTML += `<option value="${cat}" ${cat === currentCat ? 'selected' : ''}>${cat}</option>`;
            }
            
            // Update action filter
            const actSelect = document.getElementById('filter-action');
            const currentAct = actSelect.value;
            actSelect.innerHTML = '<option value="">All Actions</option>';
            for (const act of Object.keys(data.summary.by_action)) {
                actSelect.innerHTML += `<option value="${act}" ${act === currentAct ? 'selected' : ''}>${act}</option>`;
            }
        }
        
        function applyFilters() {
            loadErrors(1);
        }
        
        function clearFilters() {
            document.getElementById('filter-date').value = '';
            document.getElementById('filter-category').value = '';
            document.getElementById('filter-action').value = '';
            loadErrors(1);
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // Initial load
        loadErrors(1);
    </script>
</body>
</html>
'''
    
    def handle_rate_model(self):
        """Handle POST request to rate a model."""
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(body)
            
            fingerprint = data.get('fingerprint')
            file_id = data.get('file_id')
            rating = data.get('rating')
            comment = data.get('comment', '')
            
            if not fingerprint:
                self.send_json_error(400, "Missing 'fingerprint' field")
                return
            
            if not rating or rating < 1 or rating > 5:
                self.send_json_error(400, "Rating must be between 1 and 5")
                return
            
            # Normalize fingerprint format
            if not fingerprint.startswith("MP:"):
                fingerprint = f"MP:{fingerprint}"
            
            print(f"[Rating] Recording rating for {fingerprint}: {rating}/5")
            
            # Add POC v2 to path for imports
            sys.path.insert(0, str(POC_V3_PATH.parent / "v2"))
            from meshprep_poc.quality_feedback import get_quality_engine, QualityRating
            
            # Try to load additional context from filter file
            pipeline_used = "unknown"
            profile = "standard"
            escalated = False
            volume_change_pct = 0.0
            face_count_change_pct = 0.0
            model_filename = file_id or "unknown"
            
            if file_id:
                filter_path = THINGI10K_REPORTS / "filters" / f"{file_id}.json"
                if filter_path.exists():
                    try:
                        with open(filter_path, encoding='utf-8') as f:
                            filter_data = json.load(f)
                        
                        # Get the ACTUAL winning pipeline name from repair_attempts
                        # This is critical for learning - we need to know which specific
                        # pipeline produced the result, not just "slicer-repair-loop"
                        repair_attempts = filter_data.get("repair_attempts", {})
                        attempts = repair_attempts.get("attempts", [])
                        
                        # Find the successful pipeline
                        winning_pipeline = None
                        for attempt in attempts:
                            if attempt.get("success", False):
                                winning_pipeline = attempt.get("pipeline_name")
                                break
                        
                        if winning_pipeline:
                            pipeline_used = winning_pipeline
                            print(f"[Rating] Actual winning pipeline: {pipeline_used}")
                        else:
                            # Fallback to filter_name if no successful attempt found
                            pipeline_used = filter_data.get("filter_name", "unknown")
                        
                        escalated = filter_data.get("escalated_to_blender", False)
                        model_filename = filter_data.get("original_filename", file_id)
                        
                        # Detect profile from diagnostics
                        diag = filter_data.get("diagnostics", {})
                        before = diag.get("before", {}) or {}
                        after = diag.get("after", {}) or {}
                        
                        # Determine profile based on body count and issues
                        body_count = before.get("body_count", 1)
                        issues = repair_attempts.get("issues_found", [])
                        
                        if "extreme-fragmented" in issues or body_count > 1000:
                            profile = "extreme-fragmented"
                        elif "fragmented" in issues or body_count > 10:
                            profile = "fragmented"
                        elif body_count > 1:
                            profile = "multi-body"
                        else:
                            profile = "standard"
                        
                        print(f"[Rating] Detected profile: {profile}")
                        
                        if before.get("volume") and after.get("volume"):
                            vol_before = before["volume"]
                            vol_after = after["volume"]
                            if vol_before != 0:
                                volume_change_pct = ((vol_after - vol_before) / abs(vol_before)) * 100
                        
                        if before.get("faces") and after.get("faces"):
                            face_before = before["faces"]
                            face_after = after["faces"]
                            if face_before > 0:
                                face_count_change_pct = ((face_after - face_before) / face_before) * 100
                    except Exception as e:
                        print(f"[Rating] Could not load filter data: {e}")
            
            # Create and record rating
            quality_rating = QualityRating(
                model_fingerprint=fingerprint,
                model_filename=model_filename,
                rating_type="gradational",
                rating_value=rating,
                user_comment=comment if comment else None,
                rated_by="web_user",
                pipeline_used=pipeline_used,
                profile=profile,
                escalated=escalated,
                volume_change_pct=volume_change_pct,
                face_count_change_pct=face_count_change_pct,
            )
            
            quality_engine = get_quality_engine()
            quality_engine.record_rating(quality_rating)
            
            print(f"[Rating] Successfully recorded rating for {fingerprint}")
            print(f"[Rating]   Pipeline: {pipeline_used}, Profile: {profile}, Rating: {rating}/5")
            
            response = {
                "success": True,
                "message": "Rating recorded successfully",
                "fingerprint": fingerprint,
                "rating": rating,
                "pipeline": pipeline_used,
                "profile": profile,
            }
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except json.JSONDecodeError as e:
            print(f"[Rating] Invalid JSON: {e}")
            self.send_json_error(400, f"Invalid JSON: {e}")
        except Exception as e:
            print(f"[Rating] Error: {e}")
            import traceback
            traceback.print_exc()
            self.send_json_error(500, f"Failed to record rating: {e}")
    
    def handle_open_meshlab(self, query_string: str):
        """Open a file in MeshLab."""
        params = urllib.parse.parse_qs(query_string)
        file_path = params.get("file", [None])[0]
        
        if not file_path:
            self.send_json_error(400, "Missing 'file' parameter")
            return
        
        # Decode and resolve the file path
        file_path = urllib.parse.unquote(file_path)
        print(f"[MeshLab] Requested file: {file_path}")
        
        # Handle relative paths (from reports)
        if file_path.startswith("../"):
            # Relative to reports folder - need to resolve from base
            rel_path = file_path.replace("../", "", 1)  # Remove first ../
            full_path = THINGI10K_BASE / rel_path
        elif file_path.startswith("/"):
            # Absolute path from server root
            full_path = THINGI10K_BASE / file_path.lstrip("/")
        else:
            full_path = THINGI10K_BASE / file_path
        
        full_path = full_path.resolve()
        print(f"[MeshLab] Resolved path: {full_path}")
        
        # Security check: ensure file is within allowed directory
        try:
            full_path.relative_to(THINGI10K_BASE.resolve())
        except ValueError:
            self.send_json_error(403, f"Access denied: file outside allowed directory")
            return
        
        if not full_path.exists():
            self.send_json_error(404, f"File not found: {full_path}")
            return
        
        # Find MeshLab
        meshlab = find_meshlab()
        if not meshlab:
            self.send_json_error(500, "MeshLab not found on this system")
            return
        
        # Open in MeshLab
        try:
            print(f"[MeshLab] Opening: {full_path}")
            print(f"[MeshLab] Using: {meshlab}")
            
            if sys.platform == "win32":
                # Windows: use subprocess.Popen with CREATE_NEW_PROCESS_GROUP
                subprocess.Popen(
                    [meshlab, str(full_path)],
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:
                subprocess.Popen(
                    [meshlab, str(full_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            
            response = {"success": True, "file": str(full_path), "meshlab": meshlab}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
            print(f"[MeshLab] Successfully opened: {full_path.name}")
            
        except Exception as e:
            print(f"[MeshLab] ERROR: {e}")
            self.send_json_error(500, f"Failed to open MeshLab: {e}")
    
    def send_json_error(self, code: int, message: str):
        """Send a JSON error response."""
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({"error": message, "success": False}).encode())
        print(f"[ERROR] {code}: {message}")
    
    def log_message(self, format, *args):
        """Custom logging."""
        print(f"[{self.log_date_time_string()}] {format % args}")


def main():
    parser = argparse.ArgumentParser(description="MeshPrep Reports Server with MeshLab integration")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Port to listen on (default: 8000)")
    args = parser.parse_args()
    
    # Check MeshLab
    meshlab = find_meshlab()
    if meshlab:
        print(f"[OK] MeshLab found: {meshlab}")
    else:
        print("[WARN] MeshLab not found - 'Open in MeshLab' will not work")
    
    # Check paths
    if not THINGI10K_BASE.exists():
        print(f"[ERROR] Thingi10K path not found: {THINGI10K_BASE}")
        sys.exit(1)
    
    if not THINGI10K_REPORTS.exists():
        print(f"[WARN] Reports path not found: {THINGI10K_REPORTS}")
        print("       Reports will be created when you run the test.")
    
    if not POC_V3_PATH.exists():
        print(f"[WARN] POC v3 path not found: {POC_V3_PATH}")
    
    print(f"[OK] Serving from: {THINGI10K_BASE}")
    print(f"[OK] Reports: {THINGI10K_REPORTS}")
    print()
    print("=" * 60)
    print(f"Server running at: http://localhost:{args.port}/")
    print()
    print("Available URLs:")
    print(f"  Live Dashboard:   http://localhost:{args.port}/live")
    print(f"  Reports Index:    http://localhost:{args.port}/reports/")
    print(f"  Learning Status:  http://localhost:{args.port}/learning")
    print(f"  Error Logs:       http://localhost:{args.port}/errors/")
    print()
    print("Additional paths:")
    print(f"  Raw Meshes:       http://localhost:{args.port}/raw_meshes/")
    print(f"  Fixed Meshes:     http://localhost:{args.port}/fixed/")
    print()
    print("API Endpoints:")
    print(f"  GET  /api/progress                       - Get progress data (JSON)")
    print(f"  GET  /api/errors?page=1&per_page=50      - Get error logs (JSON)")
    print(f"  GET  /api/get-rating?fingerprint=MP:xxx  - Get rating for model")
    print(f"  POST /api/rate-model                     - Submit rating for model")
    print("=" * 60)
    print()
    print("Press Ctrl+C to stop")
    print()
    
    with SilentTCPServer(("", args.port), MeshLabHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
