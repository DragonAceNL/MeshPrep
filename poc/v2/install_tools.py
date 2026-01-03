# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Tool installation script for MeshPrep external tools.

Automatically downloads and installs:
- Blender (for advanced mesh repair)
- PrusaSlicer (for slicer validation)
- OrcaSlicer (alternative slicer)
- SuperSlicer (alternative slicer)

Usage:
    python install_tools.py --list              # List available tools
    python install_tools.py --check             # Check installed tools
    python install_tools.py --install prusa     # Install PrusaSlicer
    python install_tools.py --install blender   # Install Blender
    python install_tools.py --install all       # Install all recommended tools
"""

import argparse
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Optional, List, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def get_tools_directory() -> Path:
    """Get the MeshPrep tools directory based on platform."""
    if os.environ.get("MESHPREP_TOOLS_DIR"):
        return Path(os.environ["MESHPREP_TOOLS_DIR"])
    
    system = platform.system()
    if system == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")))
        return base / "MeshPrep" / "tools"
    elif system == "Darwin":  # macOS
        return Path.home() / "Library" / "Application Support" / "MeshPrep" / "tools"
    else:  # Linux
        return Path.home() / ".local" / "share" / "meshprep" / "tools"


def get_downloads_directory() -> Path:
    """Get the downloads cache directory."""
    return get_tools_directory() / "downloads"


# Tool definitions with download URLs and version info
TOOL_DEFINITIONS = {
    "blender": {
        "display_name": "Blender",
        "description": "Advanced mesh repair (remesh, booleans, solidify)",
        "version": "4.2.0",
        "required": False,
        "size_mb": 400,
        "urls": {
            "Windows": "https://download.blender.org/release/Blender4.2/blender-4.2.0-windows-x64.zip",
        },
        "executable": {
            "Windows": "blender.exe",
        },
        "install_subdir": "blender",
    },
    "prusa": {
        "display_name": "PrusaSlicer",
        "description": "Slicer validation (recommended)",
        "version": "2.9.2",
        "required": False,
        "size_mb": 200,
        "urls": {
            "Windows": "https://github.com/prusa3d/PrusaSlicer/releases/download/version_2.9.2/PrusaSlicer-2.9.2-win64.zip",
        },
        "executable": {
            "Windows": "prusa-slicer-console.exe",
        },
        "install_subdir": "slicers/prusaslicer",
    },
    "orca": {
        "display_name": "OrcaSlicer",
        "description": "Slicer validation (alternative)",
        "version": "2.2.0",
        "required": False,
        "size_mb": 250,
        "urls": {
            "Windows": "https://github.com/SoftFever/OrcaSlicer/releases/download/v2.2.0/OrcaSlicer_Windows_V2.2.0.zip",
        },
        "executable": {
            "Windows": "orca-slicer.exe",
        },
        "install_subdir": "slicers/orcaslicer",
    },
}


def get_system() -> str:
    """Get the current system name."""
    return platform.system()


def download_file(url: str, dest: Path, show_progress: bool = True) -> bool:
    """Download a file from URL to destination."""
    try:
        logger.info(f"Downloading from: {url}")
        
        # Create parent directory
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        def reporthook(block_num, block_size, total_size):
            if show_progress and total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 // total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r  Progress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)
        
        urllib.request.urlretrieve(url, dest, reporthook)
        
        if show_progress:
            print()  # Newline after progress
        
        logger.info(f"Downloaded to: {dest}")
        return True
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def extract_zip(zip_path: Path, dest_dir: Path) -> bool:
    """Extract a ZIP file to destination directory."""
    try:
        logger.info(f"Extracting to: {dest_dir}")
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dest_dir)
        
        logger.info("Extraction complete")
        return True
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return False


def find_executable_in_dir(directory: Path, exe_name: str) -> Optional[Path]:
    """Find an executable file in a directory (recursively)."""
    for path in directory.rglob(exe_name):
        if path.is_file():
            return path
    return None


def install_tool(tool_name: str, force: bool = False) -> Tuple[bool, str]:
    """
    Install a tool to the MeshPrep tools directory.
    
    Returns:
        Tuple of (success, message)
    """
    if tool_name not in TOOL_DEFINITIONS:
        return False, f"Unknown tool: {tool_name}"
    
    tool = TOOL_DEFINITIONS[tool_name]
    system = get_system()
    
    if system not in tool["urls"]:
        return False, f"{tool['display_name']} is not available for {system}"
    
    tools_dir = get_tools_directory()
    install_dir = tools_dir / tool["install_subdir"] / f"{tool_name}-{tool['version']}"
    
    # Check if already installed
    if install_dir.exists() and not force:
        exe_name = tool["executable"].get(system, "")
        exe_path = find_executable_in_dir(install_dir, exe_name)
        if exe_path:
            return True, f"{tool['display_name']} is already installed at {exe_path}"
    
    # Download
    url = tool["urls"][system]
    downloads_dir = get_downloads_directory()
    downloads_dir.mkdir(parents=True, exist_ok=True)
    
    filename = url.split("/")[-1]
    download_path = downloads_dir / filename
    
    if not download_path.exists():
        logger.info(f"Downloading {tool['display_name']} {tool['version']}...")
        if not download_file(url, download_path):
            return False, f"Failed to download {tool['display_name']}"
    else:
        logger.info(f"Using cached download: {download_path}")
    
    # Clean install directory if force
    if force and install_dir.exists():
        shutil.rmtree(install_dir)
    
    # Extract
    install_dir.mkdir(parents=True, exist_ok=True)
    
    if filename.endswith(".zip"):
        if not extract_zip(download_path, install_dir):
            return False, f"Failed to extract {tool['display_name']}"
    else:
        return False, f"Unsupported archive format: {filename}"
    
    # Find executable
    exe_name = tool["executable"].get(system, "")
    exe_path = find_executable_in_dir(install_dir, exe_name)
    
    if exe_path:
        logger.info(f"Installed {tool['display_name']} to: {exe_path}")
        return True, f"Successfully installed {tool['display_name']} {tool['version']} to {exe_path}"
    else:
        return False, f"Installation completed but executable not found: {exe_name}"


def check_tool(tool_name: str) -> Tuple[bool, Optional[Path], Optional[str]]:
    """
    Check if a tool is installed.
    
    Returns:
        Tuple of (is_installed, executable_path, version)
    """
    if tool_name not in TOOL_DEFINITIONS:
        return False, None, None
    
    tool = TOOL_DEFINITIONS[tool_name]
    system = get_system()
    
    # Check MeshPrep tools directory
    tools_dir = get_tools_directory()
    tool_dir = tools_dir / tool["install_subdir"]
    
    if tool_dir.exists():
        exe_name = tool["executable"].get(system, "")
        for version_dir in tool_dir.iterdir():
            if version_dir.is_dir():
                exe_path = find_executable_in_dir(version_dir, exe_name)
                if exe_path:
                    # Try to get version
                    version = version_dir.name.split("-")[-1] if "-" in version_dir.name else None
                    return True, exe_path, version
    
    # Check system PATH
    exe_name = tool["executable"].get(system, "")
    if exe_name:
        which_result = shutil.which(exe_name)
        if which_result:
            return True, Path(which_result), None
    
    return False, None, None


def list_tools() -> None:
    """List all available tools and their status."""
    print("\nAvailable Tools:")
    print("-" * 70)
    
    for name, tool in TOOL_DEFINITIONS.items():
        is_installed, path, version = check_tool(name)
        
        status = "[INSTALLED]" if is_installed else "[NOT INSTALLED]"
        size = f"~{tool['size_mb']} MB"
        
        print(f"\n  {name}")
        print(f"    Name: {tool['display_name']}")
        print(f"    Description: {tool['description']}")
        print(f"    Version: {tool['version']}")
        print(f"    Size: {size}")
        print(f"    Status: {status}")
        
        if is_installed and path:
            print(f"    Path: {path}")
    
    print()


def check_all_tools() -> None:
    """Check status of all tools."""
    print("\nTool Status:")
    print("-" * 70)
    
    tools_dir = get_tools_directory()
    print(f"Tools directory: {tools_dir}")
    print()
    
    for name, tool in TOOL_DEFINITIONS.items():
        is_installed, path, version = check_tool(name)
        
        status = "INSTALLED" if is_installed else "NOT INSTALLED"
        print(f"  {tool['display_name']}: {status}")
        
        if is_installed and path:
            print(f"    Path: {path}")
            if version:
                print(f"    Version: {version}")
    
    print()


def install_all_recommended() -> None:
    """Install all recommended tools (Blender + PrusaSlicer)."""
    recommended = ["prusa", "blender"]
    
    print("\nInstalling recommended tools...")
    print("This will install:")
    
    total_size = 0
    for name in recommended:
        tool = TOOL_DEFINITIONS[name]
        print(f"  - {tool['display_name']} (~{tool['size_mb']} MB)")
        total_size += tool["size_mb"]
    
    print(f"\nTotal download size: ~{total_size} MB")
    print()
    
    for name in recommended:
        tool = TOOL_DEFINITIONS[name]
        print(f"\n{'=' * 60}")
        print(f"Installing {tool['display_name']}...")
        print("=" * 60)
        
        success, message = install_tool(name)
        print(f"\n{message}")
        
        if not success:
            print(f"\nWarning: Failed to install {tool['display_name']}")


def main():
    parser = argparse.ArgumentParser(
        description="Install external tools for MeshPrep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python install_tools.py --list              List available tools
  python install_tools.py --check             Check installed tools
  python install_tools.py --install prusa     Install PrusaSlicer
  python install_tools.py --install blender   Install Blender
  python install_tools.py --install all       Install all recommended tools
"""
    )
    
    parser.add_argument("--list", action="store_true", help="List available tools")
    parser.add_argument("--check", action="store_true", help="Check installed tools")
    parser.add_argument("--install", type=str, help="Install a tool (prusa, orca, blender, all)")
    parser.add_argument("--force", action="store_true", help="Force reinstall")
    parser.add_argument("--uninstall", type=str, help="Uninstall a tool")
    parser.add_argument("--clear-cache", action="store_true", help="Clear download cache")
    
    args = parser.parse_args()
    
    if args.list:
        list_tools()
        return 0
    
    if args.check:
        check_all_tools()
        return 0
    
    if args.clear_cache:
        cache_dir = get_downloads_directory()
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"Cleared download cache: {cache_dir}")
        else:
            print("Download cache is already empty")
        return 0
    
    if args.uninstall:
        tool_name = args.uninstall.lower()
        if tool_name == "all":
            tools_dir = get_tools_directory()
            if tools_dir.exists():
                shutil.rmtree(tools_dir)
                print(f"Removed all tools: {tools_dir}")
            else:
                print("No tools installed")
        elif tool_name in TOOL_DEFINITIONS:
            tool = TOOL_DEFINITIONS[tool_name]
            tools_dir = get_tools_directory()
            tool_dir = tools_dir / tool["install_subdir"]
            if tool_dir.exists():
                shutil.rmtree(tool_dir)
                print(f"Removed {tool['display_name']}: {tool_dir}")
            else:
                print(f"{tool['display_name']} is not installed")
        else:
            print(f"Unknown tool: {tool_name}")
            return 1
        return 0
    
    if args.install:
        tool_name = args.install.lower()
        
        if tool_name == "all":
            install_all_recommended()
        elif tool_name in TOOL_DEFINITIONS:
            success, message = install_tool(tool_name, force=args.force)
            print(message)
            return 0 if success else 1
        else:
            print(f"Unknown tool: {tool_name}")
            print(f"Available tools: {', '.join(TOOL_DEFINITIONS.keys())}, all")
            return 1
        return 0
    
    # Default: show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
