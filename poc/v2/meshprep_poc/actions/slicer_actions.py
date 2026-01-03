# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Slicer validation actions for mesh repair.

Validates meshes by running them through actual slicers (PrusaSlicer, OrcaSlicer,
SuperSlicer, Cura) to ensure they are truly 3D printable.
"""

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any

import trimesh

logger = logging.getLogger(__name__)


@dataclass
class SlicerResult:
    """Result of a slicer validation run."""
    success: bool
    slicer: str
    slicer_version: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    issues: List[Dict[str, Any]] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    return_code: int = -1
    # Estimates (only available on success)
    print_time_minutes: Optional[float] = None
    filament_grams: Optional[float] = None
    layer_count: Optional[int] = None


# Slicer detection paths (common installation locations)
SLICER_PATHS = {
    "prusa": [
        r"C:\Program Files\Prusa3D\PrusaSlicer\prusa-slicer-console.exe",
        r"C:\Program Files\Prusa3D\PrusaSlicer\prusa-slicer.exe",
        "/usr/bin/prusa-slicer",
        "/Applications/PrusaSlicer.app/Contents/MacOS/PrusaSlicer",
    ],
    "orca": [
        r"C:\Program Files\OrcaSlicer\orca-slicer-console.exe",
        r"C:\Program Files\OrcaSlicer\orca-slicer.exe",
        "/usr/bin/orca-slicer",
        "/Applications/OrcaSlicer.app/Contents/MacOS/OrcaSlicer",
    ],
    "superslicer": [
        r"C:\Program Files\SuperSlicer\superslicer-console.exe",
        r"C:\Program Files\SuperSlicer\superslicer.exe",
        "/usr/bin/superslicer",
        "/Applications/SuperSlicer.app/Contents/MacOS/SuperSlicer",
    ],
    "cura": [
        r"C:\Program Files\Ultimaker Cura\CuraEngine.exe",
        r"C:\Program Files\UltiMaker Cura\CuraEngine.exe",
        "/usr/bin/CuraEngine",
        "/Applications/Ultimaker Cura.app/Contents/MacOS/CuraEngine",
    ],
}

# Known slicer error patterns and their categories
SLICER_ERROR_PATTERNS = {
    "non_manifold": [
        "non-manifold",
        "non manifold",
        "not manifold",
        "manifold edge",
    ],
    "holes": [
        "open edge",
        "hole detected",
        "not watertight",
        "open mesh",
    ],
    "thin_walls": [
        "thin wall",
        "wall thickness",
        "too thin",
        "below minimum",
    ],
    "self_intersections": [
        "self-intersect",
        "self intersect",
        "overlapping",
    ],
    "degenerate": [
        "degenerate",
        "zero area",
        "invalid face",
    ],
    "normals": [
        "flipped normal",
        "inverted normal",
        "wrong orientation",
    ],
    "size": [
        "too large",
        "exceeds build volume",
        "outside print area",
    ],
}


def find_slicer(slicer_type: str = "auto") -> Optional[Path]:
    """
    Find an available slicer executable.
    
    Args:
        slicer_type: One of 'prusa', 'orca', 'superslicer', 'cura', or 'auto'
        
    Returns:
        Path to slicer executable, or None if not found
    """
    if slicer_type == "auto":
        # Try slicers in order of preference
        for stype in ["prusa", "orca", "superslicer", "cura"]:
            path = find_slicer(stype)
            if path:
                return path
        return None
    
    # Check environment variable first
    env_var = f"{slicer_type.upper()}_SLICER_PATH"
    if env_var in os.environ:
        env_path = Path(os.environ[env_var])
        if env_path.exists():
            return env_path
    
    # Check common paths
    for path_str in SLICER_PATHS.get(slicer_type, []):
        path = Path(path_str)
        if path.exists():
            return path
    
    # Try to find in PATH
    exe_name = {
        "prusa": "prusa-slicer",
        "orca": "orca-slicer",
        "superslicer": "superslicer",
        "cura": "CuraEngine",
    }.get(slicer_type)
    
    if exe_name:
        which_result = shutil.which(exe_name)
        if which_result:
            return Path(which_result)
        # Try with .exe on Windows
        which_result = shutil.which(f"{exe_name}.exe")
        if which_result:
            return Path(which_result)
    
    return None


def get_slicer_version(slicer_path: Path) -> Optional[str]:
    """Get the version string of a slicer."""
    try:
        result = subprocess.run(
            [str(slicer_path), "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        # Parse version from output
        output = result.stdout + result.stderr
        for line in output.split("\n"):
            if "version" in line.lower() or any(c.isdigit() for c in line):
                return line.strip()[:50]  # Limit length
        return None
    except Exception as e:
        logger.debug(f"Failed to get slicer version: {e}")
        return None


def is_slicer_available(slicer_type: str = "auto") -> bool:
    """Check if a slicer is available."""
    return find_slicer(slicer_type) is not None


def parse_slicer_output(stdout: str, stderr: str) -> Dict[str, Any]:
    """
    Parse slicer output to extract errors, warnings, and issues.
    
    Returns:
        Dict with 'errors', 'warnings', 'issues' lists
    """
    output = (stdout + "\n" + stderr).lower()
    
    errors = []
    warnings = []
    issues = []
    
    for line in (stdout + "\n" + stderr).split("\n"):
        line_lower = line.lower().strip()
        if not line_lower:
            continue
        
        # Detect errors
        if "error" in line_lower or "fatal" in line_lower or "failed" in line_lower:
            errors.append(line.strip())
        elif "warning" in line_lower or "warn" in line_lower:
            warnings.append(line.strip())
        
        # Categorize issues
        for category, patterns in SLICER_ERROR_PATTERNS.items():
            for pattern in patterns:
                if pattern in line_lower:
                    issues.append({
                        "type": category,
                        "message": line.strip(),
                        "pattern": pattern
                    })
                    break
    
    return {
        "errors": errors,
        "warnings": warnings,
        "issues": issues
    }


def validate_with_prusa(stl_path: Path, config_path: Optional[Path] = None, timeout: int = 120) -> SlicerResult:
    """
    Validate an STL file using PrusaSlicer.
    
    Args:
        stl_path: Path to the STL file
        config_path: Optional path to slicer config/profile
        timeout: Timeout in seconds
        
    Returns:
        SlicerResult with validation details
    """
    slicer_path = find_slicer("prusa")
    if not slicer_path:
        return SlicerResult(
            success=False,
            slicer="prusa",
            errors=["PrusaSlicer not found"]
        )
    
    # Build command
    cmd = [str(slicer_path)]
    
    if config_path and config_path.exists():
        cmd.extend(["--load", str(config_path)])
    
    # Use export-gcode to validate (writes to temp, but validates the model)
    with tempfile.TemporaryDirectory() as tmpdir:
        output_gcode = Path(tmpdir) / "output.gcode"
        cmd.extend([
            "--export-gcode",
            "--output", str(output_gcode),
            str(stl_path)
        ])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            parsed = parse_slicer_output(result.stdout, result.stderr)
            
            # Check if G-code was generated (indicates success)
            success = output_gcode.exists() and result.returncode == 0
            
            return SlicerResult(
                success=success,
                slicer="prusa-slicer",
                slicer_version=get_slicer_version(slicer_path),
                errors=parsed["errors"],
                warnings=parsed["warnings"],
                issues=parsed["issues"],
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode
            )
            
        except subprocess.TimeoutExpired:
            return SlicerResult(
                success=False,
                slicer="prusa-slicer",
                errors=[f"Slicer timed out after {timeout} seconds"]
            )
        except Exception as e:
            return SlicerResult(
                success=False,
                slicer="prusa-slicer",
                errors=[f"Slicer execution failed: {str(e)}"]
            )


def validate_with_orca(stl_path: Path, config_path: Optional[Path] = None, timeout: int = 120) -> SlicerResult:
    """
    Validate an STL file using OrcaSlicer.
    Similar to PrusaSlicer as it's a fork.
    """
    slicer_path = find_slicer("orca")
    if not slicer_path:
        return SlicerResult(
            success=False,
            slicer="orca",
            errors=["OrcaSlicer not found"]
        )
    
    cmd = [str(slicer_path)]
    
    if config_path and config_path.exists():
        cmd.extend(["--load", str(config_path)])
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_gcode = Path(tmpdir) / "output.gcode"
        cmd.extend([
            "--export-gcode",
            "--output", str(output_gcode),
            str(stl_path)
        ])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            parsed = parse_slicer_output(result.stdout, result.stderr)
            success = output_gcode.exists() and result.returncode == 0
            
            return SlicerResult(
                success=success,
                slicer="orca-slicer",
                slicer_version=get_slicer_version(slicer_path),
                errors=parsed["errors"],
                warnings=parsed["warnings"],
                issues=parsed["issues"],
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode
            )
            
        except subprocess.TimeoutExpired:
            return SlicerResult(
                success=False,
                slicer="orca-slicer",
                errors=[f"Slicer timed out after {timeout} seconds"]
            )
        except Exception as e:
            return SlicerResult(
                success=False,
                slicer="orca-slicer",
                errors=[f"Slicer execution failed: {str(e)}"]
            )


def validate_mesh(
    mesh: trimesh.Trimesh,
    slicer: str = "auto",
    config_path: Optional[Path] = None,
    timeout: int = 120
) -> SlicerResult:
    """
    Validate a mesh using an available slicer.
    
    Args:
        mesh: The trimesh mesh to validate
        slicer: Slicer to use ('prusa', 'orca', 'superslicer', 'cura', or 'auto')
        config_path: Optional path to slicer config
        timeout: Timeout in seconds
        
    Returns:
        SlicerResult with validation details
    """
    # Find available slicer
    slicer_path = find_slicer(slicer)
    if not slicer_path:
        return SlicerResult(
            success=False,
            slicer=slicer,
            errors=[f"No slicer found (tried: {slicer})"]
        )
    
    # Determine which slicer we found
    slicer_name = slicer_path.stem.lower()
    if "prusa" in slicer_name:
        slicer_type = "prusa"
    elif "orca" in slicer_name:
        slicer_type = "orca"
    elif "super" in slicer_name:
        slicer_type = "superslicer"
    else:
        slicer_type = "cura"
    
    # Export mesh to temp STL
    with tempfile.TemporaryDirectory() as tmpdir:
        stl_path = Path(tmpdir) / "mesh.stl"
        mesh.export(str(stl_path))
        
        # Run appropriate validator
        if slicer_type in ["prusa", "superslicer"]:
            return validate_with_prusa(stl_path, config_path, timeout)
        elif slicer_type == "orca":
            return validate_with_orca(stl_path, config_path, timeout)
        else:
            # Cura requires more setup, return basic result for now
            return SlicerResult(
                success=False,
                slicer="cura",
                errors=["Cura validation not yet implemented"]
            )


def validate_stl_file(
    stl_path: Path,
    slicer: str = "auto",
    config_path: Optional[Path] = None,
    timeout: int = 120
) -> SlicerResult:
    """
    Validate an STL file using an available slicer.
    
    Args:
        stl_path: Path to STL file
        slicer: Slicer to use ('prusa', 'orca', 'superslicer', 'cura', or 'auto')
        config_path: Optional path to slicer config
        timeout: Timeout in seconds
        
    Returns:
        SlicerResult with validation details
    """
    slicer_path = find_slicer(slicer)
    if not slicer_path:
        return SlicerResult(
            success=False,
            slicer=slicer,
            errors=[f"No slicer found (tried: {slicer})"]
        )
    
    # Determine slicer type
    slicer_name = slicer_path.stem.lower()
    if "prusa" in slicer_name or "super" in slicer_name:
        return validate_with_prusa(stl_path, config_path, timeout)
    elif "orca" in slicer_name:
        return validate_with_orca(stl_path, config_path, timeout)
    else:
        return SlicerResult(
            success=False,
            slicer="unknown",
            errors=["Unsupported slicer type"]
        )


# Convenience function to check slicer availability
def get_available_slicers() -> Dict[str, Path]:
    """
    Get a dictionary of all available slicers.
    
    Returns:
        Dict mapping slicer name to executable path
    """
    available = {}
    for slicer_type in ["prusa", "orca", "superslicer", "cura"]:
        path = find_slicer(slicer_type)
        if path:
            available[slicer_type] = path
    return available
