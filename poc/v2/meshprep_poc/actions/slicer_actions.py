# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Slicer validation actions for mesh repair.

Validates meshes by running them through actual slicers (PrusaSlicer, OrcaSlicer,
SuperSlicer, Cura) to ensure they are truly 3D printable.

IMPORTANT: Modern slicers (PrusaSlicer, OrcaSlicer) have built-in auto-repair
that silently fixes many mesh issues. This means:
1. A model that "passes" slicing may still have issues
2. The slicer is fixing issues internally, not the exported mesh
3. Users exporting to OTHER slicers may have problems

This module provides two validation modes:
1. STRICT MODE (--info): Uses slicer's mesh analysis WITHOUT auto-repair
2. SLICE MODE (--export-gcode): Tests if slicer can produce G-code (with auto-repair)

For MeshPrep, we use STRICT MODE to ensure the mesh itself is clean,
not just that the slicer can work around issues.
"""

import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any

import trimesh

logger = logging.getLogger(__name__)


@dataclass
class MeshInfo:
    """Raw mesh information from slicer --info command."""
    manifold: bool = False
    open_edges: int = 0
    facets_reversed: int = 0
    backwards_edges: int = 0
    number_of_parts: int = 0
    number_of_facets: int = 0
    volume: float = 0.0
    size_x: float = 0.0
    size_y: float = 0.0
    size_z: float = 0.0
    
    @property
    def is_clean(self) -> bool:
        """Check if mesh is clean (no issues detected)."""
        return (
            self.manifold and
            self.open_edges == 0 and
            self.backwards_edges == 0 and
            self.facets_reversed == 0  # Reversed facets indicate normal issues
        )
    
    @property
    def issues(self) -> List[str]:
        """List of detected issues."""
        issues = []
        if not self.manifold:
            issues.append("non-manifold")
        if self.open_edges > 0:
            issues.append(f"open_edges ({self.open_edges})")
        if self.backwards_edges > 0:
            issues.append(f"backwards_edges ({self.backwards_edges})")
        if self.facets_reversed > 0:
            issues.append(f"facets_reversed ({self.facets_reversed})")
        return issues


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
    # Mesh info (from --info mode)
    mesh_info: Optional[MeshInfo] = None
    # Estimates (only available on slice success)
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
        "manifold = no",
    ],
    "holes": [
        "open edge",
        "open_edges",
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
        "facets_reversed",
        "backwards_edges",
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


def parse_mesh_info(output: str) -> MeshInfo:
    """
    Parse the output of slicer --info command.
    
    Args:
        output: The stdout from --info command
        
    Returns:
        MeshInfo dataclass with parsed values
    """
    info = MeshInfo()
    
    for line in output.split("\n"):
        line = line.strip()
        if "=" not in line:
            continue
        
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        
        try:
            if key == "manifold":
                info.manifold = value.lower() == "yes"
            elif key == "open_edges":
                info.open_edges = int(value)
            elif key == "facets_reversed":
                info.facets_reversed = int(value)
            elif key == "backwards_edges":
                info.backwards_edges = int(value)
            elif key == "number_of_parts":
                info.number_of_parts = int(value)
            elif key == "number_of_facets":
                info.number_of_facets = int(value)
            elif key == "volume":
                info.volume = float(value)
            elif key == "size_x":
                info.size_x = float(value)
            elif key == "size_y":
                info.size_y = float(value)
            elif key == "size_z":
                info.size_z = float(value)
        except (ValueError, TypeError):
            pass
    
    return info


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


def get_mesh_info_prusa(stl_path: Path, timeout: int = 30) -> SlicerResult:
    """
    Get mesh info using PrusaSlicer --info (STRICT MODE - no auto-repair).
    
    This is the preferred validation method as it reports the actual mesh
    state WITHOUT any auto-repair.
    
    Args:
        stl_path: Path to the STL file
        timeout: Timeout in seconds
        
    Returns:
        SlicerResult with mesh_info populated
    """
    slicer_path = find_slicer("prusa")
    if not slicer_path:
        return SlicerResult(
            success=False,
            slicer="prusa",
            errors=["PrusaSlicer not found"]
        )
    
    cmd = [str(slicer_path), "--info", str(stl_path)]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        mesh_info = parse_mesh_info(result.stdout)
        
        # Build issues list from mesh_info
        issues = []
        if not mesh_info.manifold:
            issues.append({"type": "non_manifold", "message": "Mesh is not manifold"})
        if mesh_info.open_edges > 0:
            issues.append({"type": "holes", "message": f"{mesh_info.open_edges} open edges"})
        if mesh_info.backwards_edges > 0:
            issues.append({"type": "normals", "message": f"{mesh_info.backwards_edges} backwards edges"})
        if mesh_info.facets_reversed > 0:
            issues.append({"type": "normals", "message": f"{mesh_info.facets_reversed} facets reversed"})
        
        return SlicerResult(
            success=mesh_info.is_clean,
            slicer="prusa-slicer (--info)",
            slicer_version=get_slicer_version(slicer_path),
            errors=[],
            warnings=[],
            issues=issues,
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
            mesh_info=mesh_info
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


def validate_with_prusa(stl_path: Path, config_path: Optional[Path] = None, timeout: int = 120) -> SlicerResult:
    """
    Validate an STL file using PrusaSlicer (SLICE MODE - includes auto-repair).
    
    WARNING: PrusaSlicer auto-repairs many mesh issues internally. A model that
    passes this validation may still have issues when used with other slicers.
    
    For strict validation without auto-repair, use get_mesh_info_prusa() instead.
    
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
                slicer="prusa-slicer (slice mode)",
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


def validate_mesh_strict(
    mesh: trimesh.Trimesh,
    slicer: str = "auto",
    timeout: int = 30
) -> SlicerResult:
    """
    STRICT validation: Get mesh info WITHOUT slicer auto-repair.
    
    This is the recommended validation method for MeshPrep as it ensures
    the mesh itself is clean, not just that a specific slicer can fix it.
    
    Args:
        mesh: The trimesh mesh to validate
        slicer: Slicer to use (currently only 'prusa' supports --info)
        timeout: Timeout in seconds
        
    Returns:
        SlicerResult with mesh_info and issues
    """
    # Find available slicer that supports --info
    slicer_path = find_slicer(slicer)
    if not slicer_path:
        return SlicerResult(
            success=False,
            slicer=slicer,
            errors=[f"No slicer found (tried: {slicer})"]
        )
    
    # Currently only PrusaSlicer supports --info
    slicer_name = slicer_path.stem.lower()
    if "prusa" not in slicer_name and "super" not in slicer_name:
        # Fall back to slice mode for other slicers
        logger.warning(f"Slicer {slicer_name} doesn't support --info, falling back to slice mode")
        return validate_mesh(mesh, slicer=slicer, timeout=timeout)
    
    # Export mesh to temp STL
    with tempfile.TemporaryDirectory() as tmpdir:
        stl_path = Path(tmpdir) / "mesh.stl"
        mesh.export(str(stl_path))
        
        return get_mesh_info_prusa(stl_path, timeout)


def validate_mesh(
    mesh: trimesh.Trimesh,
    slicer: str = "auto",
    config_path: Optional[Path] = None,
    timeout: int = 120,
    strict: bool = True
) -> SlicerResult:
    """
    Validate a mesh using an available slicer.
    
    Args:
        mesh: The trimesh mesh to validate
        slicer: Slicer to use ('prusa', 'orca', 'superslicer', 'cura', or 'auto')
        config_path: Optional path to slicer config
        timeout: Timeout in seconds
        strict: If True, use --info mode (no auto-repair). If False, use slice mode.
        
    Returns:
        SlicerResult with validation details
    """
    # For strict mode, use --info if available
    if strict:
        result = validate_mesh_strict(mesh, slicer, timeout)
        if result.mesh_info is not None:
            return result
        # Fall through to slice mode if --info not available
    
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
    timeout: int = 120,
    strict: bool = True
) -> SlicerResult:
    """
    Validate an STL file using an available slicer.
    
    Args:
        stl_path: Path to STL file
        slicer: Slicer to use ('prusa', 'orca', 'superslicer', 'cura', or 'auto')
        config_path: Optional path to slicer config
        timeout: Timeout in seconds
        strict: If True, use --info mode (no auto-repair). If False, use slice mode.
        
    Returns:
        SlicerResult with validation details
    """
    # For strict mode, use --info if available
    if strict:
        slicer_path = find_slicer(slicer)
        if slicer_path:
            slicer_name = slicer_path.stem.lower()
            if "prusa" in slicer_name or "super" in slicer_name:
                return get_mesh_info_prusa(stl_path, timeout)
    
    # Fall back to slice mode
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


# ==============================================================================
# REGISTERED ACTIONS FOR FILTER SCRIPTS
# ==============================================================================

from .registry import register_action


@register_action(
    name="slicer_validate",
    description="Validate mesh with a slicer (STRICT mode - no auto-repair)",
    parameters={
        "slicer": "Slicer to use: 'prusa', 'orca', 'auto' (default: 'auto')",
        "timeout": "Timeout in seconds (default: 30)",
        "strict": "If True, use --info mode without auto-repair (default: True)",
    },
    risk_level="low"
)
def action_slicer_validate(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Validate mesh using a slicer (STRICT mode by default).
    
    This action does NOT modify the mesh - it only validates that the mesh
    is clean according to the slicer's mesh analysis (without auto-repair).
    If validation fails, an exception is raised.
    
    Args:
        mesh: The mesh to validate
        params: Parameters including 'slicer', 'timeout', 'strict'
        
    Returns:
        The same mesh (unmodified) if validation passes
        
    Raises:
        ValueError: If slicer validation fails
    """
    slicer = params.get("slicer", "auto")
    timeout = params.get("timeout", 30)
    strict = params.get("strict", True)
    
    result = validate_mesh(mesh, slicer=slicer, timeout=timeout, strict=strict)
    
    if not result.success:
        # Build detailed error message
        error_parts = []
        if result.errors:
            error_parts.extend(result.errors)
        if result.mesh_info and not result.mesh_info.is_clean:
            error_parts.append(f"Mesh issues: {', '.join(result.mesh_info.issues)}")
        if result.issues:
            issue_types = list(set(i['type'] for i in result.issues))
            error_parts.append(f"Issue types: {', '.join(issue_types)}")
        
        error_msg = "; ".join(error_parts) if error_parts else "Slicer validation failed"
        raise ValueError(f"Slicer validation failed: {error_msg}")
    
    logger.info(f"Slicer validation passed ({result.slicer})")
    return mesh


@register_action(
    name="slicer_validate_slice",
    description="Validate mesh can be sliced (with slicer auto-repair)",
    parameters={
        "slicer": "Slicer to use: 'prusa', 'orca', 'auto' (default: 'auto')",
        "timeout": "Timeout in seconds (default: 120)",
    },
    risk_level="low"
)
def action_slicer_validate_slice(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Validate mesh can be sliced (allows slicer auto-repair).
    
    WARNING: This mode allows the slicer to auto-repair issues, so a mesh
    that passes may still have problems with other slicers.
    
    Use slicer_validate for strict validation without auto-repair.
    """
    slicer = params.get("slicer", "auto")
    timeout = params.get("timeout", 120)
    
    result = validate_mesh(mesh, slicer=slicer, timeout=timeout, strict=False)
    
    if not result.success:
        error_msg = "; ".join(result.errors) if result.errors else "Slicer validation failed"
        raise ValueError(f"Slicer validation failed: {error_msg}")
    
    logger.info(f"Slicer slice validation passed ({result.slicer})")
    return mesh


@register_action(
    name="slicer_check",
    description="Check if mesh is clean (non-fatal, logs warnings)",
    parameters={
        "slicer": "Slicer to use: 'prusa', 'orca', 'auto' (default: 'auto')",
        "timeout": "Timeout in seconds (default: 30)",
    },
    risk_level="low"
)
def action_slicer_check(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Check mesh with a slicer (non-fatal version).
    
    Unlike slicer_validate, this action does not raise an exception on failure.
    It logs warnings but allows the pipeline to continue.
    
    Args:
        mesh: The mesh to check
        params: Parameters including 'slicer' and 'timeout'
        
    Returns:
        The same mesh (unmodified)
    """
    slicer = params.get("slicer", "auto")
    timeout = params.get("timeout", 30)
    
    # Check if any slicer is available
    if not is_slicer_available(slicer):
        logger.warning(f"No slicer available for validation (tried: {slicer})")
        return mesh
    
    result = validate_mesh(mesh, slicer=slicer, timeout=timeout, strict=True)
    
    if result.success:
        logger.info(f"Slicer check passed ({result.slicer})")
    else:
        logger.warning(f"Slicer check failed:")
        if result.mesh_info:
            for issue in result.mesh_info.issues:
                logger.warning(f"  - {issue}")
        if result.issues:
            for issue in result.issues:
                logger.warning(f"  - [{issue['type']}] {issue['message']}")
    
    return mesh
