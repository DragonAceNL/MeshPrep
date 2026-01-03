# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Local HTTP server for MeshPrep reports with MeshLab integration.

This server:
1. Serves the reports and STL files via HTTP
2. Provides an API endpoint to open STL files in MeshLab

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
THINGI10K_PATH = Path(r"C:\Users\Dragon Ace\Source\repos\Thingi10K\raw_meshes")
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


class MeshLabHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler with MeshLab integration."""
    
    def __init__(self, *args, **kwargs):
        # Don't set directory here - we'll handle routing manually
        super().__init__(*args, **kwargs)
    
    def translate_path(self, path):
        """Translate URL path to filesystem path with multi-directory support."""
        # Remove query string and fragment
        path = path.split('?')[0].split('#')[0]
        path = urllib.parse.unquote(path)
        
        # Route based on path prefix
        if path.startswith('/MeshPrep/') or path.startswith('/poc/'):
            # Serve from POC v3 directory (for dashboard, progress.json, etc.)
            if path.startswith('/MeshPrep/poc/v3/'):
                rel_path = path[len('/MeshPrep/poc/v3/'):]
            elif path.startswith('/poc/v3/'):
                rel_path = path[len('/poc/v3/'):]
            elif path.startswith('/poc/'):
                rel_path = path[len('/poc/'):]
                return str(POC_V3_PATH.parent / rel_path)
            else:
                rel_path = path[len('/MeshPrep/'):]
                return str(POC_V3_PATH.parent.parent / rel_path)
            return str(POC_V3_PATH / rel_path)
        elif path.startswith('/dashboard'):
            # Shortcut for dashboard
            if path == '/dashboard' or path == '/dashboard/':
                return str(POC_V3_PATH / 'dashboard.html')
            return str(POC_V3_PATH / path[1:])  # Remove leading /
        elif path.startswith('/live_dashboard') or path == '/live':
            return str(POC_V3_PATH / 'live_dashboard.html')
        elif path.startswith('/learning') or path == '/learning-status':
            # Learning status page - generate fresh if doesn't exist or requested
            status_path = POC_V3_PATH / 'learning_status.html'
            if not status_path.exists():
                try:
                    from generate_learning_status import generate_learning_status_page
                    generate_learning_status_page()
                except Exception:
                    pass
            return str(status_path)
        elif path.startswith('/progress.json'):
            return str(POC_V3_PATH / 'progress.json')
        else:
            # Default: serve from Thingi10K raw_meshes
            if path.startswith('/'):
                path = path[1:]
            return str(THINGI10K_PATH / path)
    
    def do_GET(self):
        """Handle GET requests."""
        parsed = urllib.parse.urlparse(self.path)
        
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
        
        # Default: serve static files
        super().do_GET()
    
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
            # Relative to reports folder - need to resolve from raw_meshes
            rel_path = file_path.replace("../", "", 1)  # Remove first ../
            full_path = THINGI10K_PATH / rel_path
        elif file_path.startswith("/"):
            # Absolute path from server root
            full_path = THINGI10K_PATH / file_path.lstrip("/")
        else:
            full_path = THINGI10K_PATH / file_path
        
        full_path = full_path.resolve()
        print(f"[MeshLab] Resolved path: {full_path}")
        
        # Security check: ensure file is within allowed directory
        try:
            full_path.relative_to(THINGI10K_PATH.resolve())
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
    if not THINGI10K_PATH.exists():
        print(f"[ERROR] Thingi10K path not found: {THINGI10K_PATH}")
        sys.exit(1)
    
    if not POC_V3_PATH.exists():
        print(f"[WARN] POC v3 path not found: {POC_V3_PATH}")
    
    print(f"[OK] Serving reports from: {THINGI10K_PATH}")
    print(f"[OK] Serving dashboard from: {POC_V3_PATH}")
    print()
    print("=" * 60)
    print(f"Server running at: http://localhost:{args.port}/")
    print()
    print("Available URLs:")
    print(f"  Reports Index:    http://localhost:{args.port}/reports/")
    print(f"  Live Dashboard:   http://localhost:{args.port}/live_dashboard.html")
    print(f"  Static Dashboard: http://localhost:{args.port}/dashboard")
    print(f"  Progress JSON:    http://localhost:{args.port}/progress.json")
    print("=" * 60)
    print()
    print("Press Ctrl+C to stop")
    print()
    
    with socketserver.TCPServer(("", args.port), MeshLabHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
