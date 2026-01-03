# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Reproducibility and versioning support for MeshPrep.

This module provides:
- Version tracking for MeshPrep, actions, and tools
- Compatibility checking between filter scripts and current environment
- Environment snapshots for exact reproduction
- Version validation and warnings

Reproducibility Levels:
- Loose: MeshPrep version only (default)
- Standard: MeshPrep + tool versions (recommended for sharing)
- Strict: Everything + exact commits (for scientific/production)
"""

import json
import logging
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants and Paths
# ---------------------------------------------------------------------------

# Root paths
_MODULE_DIR = Path(__file__).parent
_POC_V2_DIR = _MODULE_DIR.parent
_REPO_ROOT = _POC_V2_DIR.parent.parent

# Config files
CONFIG_DIR = _REPO_ROOT / "config"
COMPATIBILITY_FILE = CONFIG_DIR / "compatibility.json"
ACTION_REGISTRY_FILE = CONFIG_DIR / "action_registry.json"
VERSION_FILE = _REPO_ROOT / "VERSION"


class ReproducibilityLevel(Enum):
    """Level of reproducibility for filter scripts and reports."""
    LOOSE = "loose"        # MeshPrep version only
    STANDARD = "standard"  # MeshPrep + tool versions
    STRICT = "strict"      # Everything + exact commits


# ---------------------------------------------------------------------------
# Version Information
# ---------------------------------------------------------------------------

@dataclass
class VersionInfo:
    """Version information for a package or tool."""
    name: str
    version: str
    min_version: Optional[str] = None
    max_version: Optional[str] = None
    recommended_version: Optional[str] = None
    path: Optional[str] = None
    
    def is_compatible(self) -> bool:
        """Check if version is within min/max bounds."""
        try:
            from packaging.version import Version
            current = Version(self.version)
            
            if self.min_version and current < Version(self.min_version):
                return False
            if self.max_version and current > Version(self.max_version):
                return False
            return True
        except ImportError:
            # packaging not available, do simple string comparison
            return True
        except Exception:
            return True
    
    def is_recommended(self) -> bool:
        """Check if version matches recommended."""
        if not self.recommended_version:
            return True
        return self.version == self.recommended_version


@dataclass
class EnvironmentSnapshot:
    """Complete snapshot of the execution environment."""
    
    # MeshPrep info
    meshprep_version: str = ""
    meshprep_commit: Optional[str] = None
    action_registry_version: str = "1.0.0"
    
    # Python environment
    python_version: str = ""
    platform_info: str = ""
    
    # Package versions
    package_versions: dict[str, str] = field(default_factory=dict)
    
    # External tools
    external_tools: dict[str, dict[str, Any]] = field(default_factory=dict)
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    reproducibility_level: str = "standard"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "meshprep_version": self.meshprep_version,
            "meshprep_commit": self.meshprep_commit,
            "action_registry_version": self.action_registry_version,
            "python_version": self.python_version,
            "platform": self.platform_info,
            "tool_versions": self.package_versions,
            "external_tools": self.external_tools,
            "timestamp": self.timestamp,
            "level": self.reproducibility_level,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "EnvironmentSnapshot":
        """Create from dictionary."""
        return cls(
            meshprep_version=data.get("meshprep_version", ""),
            meshprep_commit=data.get("meshprep_commit"),
            action_registry_version=data.get("action_registry_version", "1.0.0"),
            python_version=data.get("python_version", ""),
            platform_info=data.get("platform", ""),
            package_versions=data.get("tool_versions", {}),
            external_tools=data.get("external_tools", {}),
            timestamp=data.get("timestamp", ""),
            reproducibility_level=data.get("level", "standard"),
        )


# ---------------------------------------------------------------------------
# Compatibility Matrix
# ---------------------------------------------------------------------------

@dataclass
class CompatibilityMatrix:
    """Tool compatibility matrix for a MeshPrep version."""
    
    meshprep_version: str
    action_registry_version: str
    python: dict[str, str]  # min, max, recommended
    required_packages: dict[str, dict[str, Optional[str]]]  # package -> {min, max, recommended}
    external_tools: dict[str, dict[str, Optional[str]]]  # tool -> {min, max, recommended}
    
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "CompatibilityMatrix":
        """Load compatibility matrix from JSON file."""
        path = path or COMPATIBILITY_FILE
        
        if not path.exists():
            logger.warning(f"Compatibility file not found: {path}, using defaults")
            return cls.default()
        
        try:
            with open(path) as f:
                data = json.load(f)
            
            return cls(
                meshprep_version=data.get("meshprep_version", "0.1.0"),
                action_registry_version=data.get("action_registry_version", "1.0.0"),
                python=data.get("python", {"min": "3.11", "max": "3.12", "recommended": "3.12"}),
                required_packages=data.get("required_packages", {}),
                external_tools=data.get("external_tools", {}),
            )
        except Exception as e:
            logger.error(f"Failed to load compatibility matrix: {e}")
            return cls.default()
    
    @classmethod
    def default(cls) -> "CompatibilityMatrix":
        """Return default compatibility matrix."""
        return cls(
            meshprep_version="0.2.0",
            action_registry_version="1.0.0",
            python={"min": "3.11", "max": "3.12", "recommended": "3.12"},
            required_packages={
                "trimesh": {"min": "4.0.0", "recommended": "4.5.0", "max": None},
                "pymeshfix": {"min": "0.16.0", "recommended": "0.17.0", "max": None},
                "pymeshlab": {"min": "2023.12", "recommended": "2025.7", "max": None},
                "numpy": {"min": "1.24.0", "recommended": "2.4.0", "max": None},
            },
            external_tools={
                "blender": {"min": "3.6.0", "recommended": "4.2.0", "max": None},
                "prusaslicer": {"min": "2.6.0", "recommended": "2.8.0", "max": None},
                "orcaslicer": {"min": "2.0.0", "recommended": "2.2.0", "max": None},
            },
        )
    
    def save(self, path: Optional[Path] = None) -> None:
        """Save compatibility matrix to JSON file."""
        path = path or COMPATIBILITY_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "meshprep_version": self.meshprep_version,
            "action_registry_version": self.action_registry_version,
            "python": self.python,
            "required_packages": self.required_packages,
            "external_tools": self.external_tools,
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved compatibility matrix to {path}")


# ---------------------------------------------------------------------------
# Version Detection Functions
# ---------------------------------------------------------------------------

def get_meshprep_version() -> str:
    """Get MeshPrep version from VERSION file or package."""
    # Try VERSION file first
    if VERSION_FILE.exists():
        try:
            return VERSION_FILE.read_text().strip()
        except Exception:
            pass
    
    # Try package version
    try:
        from . import __version__
        return __version__
    except ImportError:
        pass
    
    return "0.1.0"


def get_git_commit() -> Optional[str]:
    """Get current git commit hash if in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def get_python_version() -> str:
    """Get Python version string."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def get_platform_info() -> str:
    """Get platform information string."""
    return platform.platform()


def get_package_version(package_name: str) -> Optional[str]:
    """Get installed version of a Python package."""
    try:
        from importlib.metadata import version
        return version(package_name)
    except Exception:
        # Try importing and checking __version__
        try:
            module = __import__(package_name)
            return getattr(module, "__version__", None)
        except Exception:
            return None


def get_blender_version() -> Optional[str]:
    """Get Blender version if available."""
    try:
        # Try common Blender paths
        blender_paths = [
            "blender",  # PATH
            r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
            "/Applications/Blender.app/Contents/MacOS/Blender",
            "/usr/bin/blender",
        ]
        
        for blender_path in blender_paths:
            try:
                result = subprocess.run(
                    [blender_path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    # Parse "Blender 4.2.0" from output
                    for line in result.stdout.split("\n"):
                        if line.startswith("Blender"):
                            parts = line.split()
                            if len(parts) >= 2:
                                return parts[1]
            except Exception:
                continue
    except Exception:
        pass
    return None


def get_slicer_version(slicer: str = "prusa") -> Optional[str]:
    """Get slicer version if available."""
    slicer_commands = {
        "prusa": ["prusa-slicer", "--version"],
        "orca": ["orca-slicer", "--version"],
        "cura": ["CuraEngine", "--version"],
    }
    
    if slicer not in slicer_commands:
        return None
    
    try:
        result = subprocess.run(
            slicer_commands[slicer],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            # Extract version from output
            output = result.stdout.strip() or result.stderr.strip()
            # Usually first line contains version
            return output.split("\n")[0].strip()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Environment Capture
# ---------------------------------------------------------------------------

def capture_environment(
    level: ReproducibilityLevel = ReproducibilityLevel.STANDARD,
    include_external: bool = True,
) -> EnvironmentSnapshot:
    """
    Capture current environment snapshot.
    
    Args:
        level: Reproducibility level
        include_external: Whether to check external tools (slower)
    
    Returns:
        EnvironmentSnapshot with all version information
    """
    snapshot = EnvironmentSnapshot(
        meshprep_version=get_meshprep_version(),
        python_version=get_python_version(),
        platform_info=get_platform_info(),
        reproducibility_level=level.value,
    )
    
    # Get git commit for strict level
    if level == ReproducibilityLevel.STRICT:
        snapshot.meshprep_commit = get_git_commit()
    
    # Capture package versions
    packages = ["trimesh", "pymeshfix", "pymeshlab", "numpy", "scipy", "networkx"]
    for pkg in packages:
        version = get_package_version(pkg)
        if version:
            snapshot.package_versions[pkg] = version
    
    # Capture external tools
    if include_external:
        # Blender
        blender_version = get_blender_version()
        if blender_version:
            snapshot.external_tools["blender"] = {
                "version": blender_version,
                "available": True,
            }
        else:
            snapshot.external_tools["blender"] = {
                "version": None,
                "available": False,
            }
        
        # Slicers
        for slicer in ["prusa", "orca"]:
            slicer_version = get_slicer_version(slicer)
            snapshot.external_tools[slicer + "slicer"] = {
                "version": slicer_version,
                "available": slicer_version is not None,
            }
    
    return snapshot


def export_environment(path: Path, level: ReproducibilityLevel = ReproducibilityLevel.STANDARD) -> None:
    """Export environment snapshot to file."""
    snapshot = capture_environment(level, include_external=True)
    
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(snapshot.to_dict(), f, indent=2)
    
    logger.info(f"Exported environment snapshot to {path}")


def import_environment(path: Path) -> EnvironmentSnapshot:
    """Import environment snapshot from file."""
    with open(path) as f:
        data = json.load(f)
    return EnvironmentSnapshot.from_dict(data)


# ---------------------------------------------------------------------------
# Compatibility Checking
# ---------------------------------------------------------------------------

@dataclass
class CompatibilityIssue:
    """A compatibility issue found during checking."""
    severity: str  # "error", "warning", "info"
    component: str  # e.g., "python", "trimesh", "blender"
    message: str
    current_version: Optional[str] = None
    expected_version: Optional[str] = None


@dataclass
class CompatibilityResult:
    """Result of compatibility checking."""
    compatible: bool
    issues: list[CompatibilityIssue] = field(default_factory=list)
    
    @property
    def errors(self) -> list[CompatibilityIssue]:
        return [i for i in self.issues if i.severity == "error"]
    
    @property
    def warnings(self) -> list[CompatibilityIssue]:
        return [i for i in self.issues if i.severity == "warning"]
    
    def __str__(self) -> str:
        if self.compatible and not self.issues:
            return "Environment: COMPATIBLE"
        
        lines = []
        if not self.compatible:
            lines.append("Environment: INCOMPATIBLE")
        else:
            lines.append("Environment: COMPATIBLE (with warnings)")
        
        for issue in self.issues:
            prefix = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(issue.severity, "?")
            lines.append(f"  {prefix} {issue.component}: {issue.message}")
        
        return "\n".join(lines)


def check_compatibility(
    matrix: Optional[CompatibilityMatrix] = None,
    strict: bool = False,
) -> CompatibilityResult:
    """
    Check current environment against compatibility matrix.
    
    Args:
        matrix: Compatibility matrix to check against (loads default if None)
        strict: If True, treat warnings as errors
    
    Returns:
        CompatibilityResult with issues found
    """
    if matrix is None:
        matrix = CompatibilityMatrix.load()
    
    result = CompatibilityResult(compatible=True)
    
    # Check Python version
    py_version = get_python_version()
    py_major_minor = ".".join(py_version.split(".")[:2])
    
    if matrix.python.get("min") and py_major_minor < matrix.python["min"]:
        result.issues.append(CompatibilityIssue(
            severity="error",
            component="python",
            message=f"Python {py_version} is below minimum {matrix.python['min']}",
            current_version=py_version,
            expected_version=matrix.python["min"],
        ))
        result.compatible = False
    
    if matrix.python.get("max") and py_major_minor > matrix.python["max"]:
        result.issues.append(CompatibilityIssue(
            severity="error",
            component="python",
            message=f"Python {py_version} is above maximum {matrix.python['max']}",
            current_version=py_version,
            expected_version=matrix.python["max"],
        ))
        result.compatible = False
    
    # Check required packages
    for pkg_name, constraints in matrix.required_packages.items():
        installed = get_package_version(pkg_name)
        
        if not installed:
            result.issues.append(CompatibilityIssue(
                severity="warning",
                component=pkg_name,
                message=f"{pkg_name} is not installed",
                current_version=None,
                expected_version=constraints.get("recommended"),
            ))
            if strict:
                result.compatible = False
            continue
        
        # Check version bounds
        try:
            from packaging.version import Version
            current = Version(installed)
            
            if constraints.get("min") and current < Version(constraints["min"]):
                result.issues.append(CompatibilityIssue(
                    severity="error",
                    component=pkg_name,
                    message=f"{pkg_name} {installed} is below minimum {constraints['min']}",
                    current_version=installed,
                    expected_version=constraints["min"],
                ))
                result.compatible = False
            
            if constraints.get("max") and current > Version(constraints["max"]):
                result.issues.append(CompatibilityIssue(
                    severity="warning",
                    component=pkg_name,
                    message=f"{pkg_name} {installed} is above maximum {constraints['max']}",
                    current_version=installed,
                    expected_version=constraints["max"],
                ))
                if strict:
                    result.compatible = False
            
            # Check if recommended
            if constraints.get("recommended") and installed != constraints["recommended"]:
                result.issues.append(CompatibilityIssue(
                    severity="info",
                    component=pkg_name,
                    message=f"{pkg_name} {installed} differs from recommended {constraints['recommended']}",
                    current_version=installed,
                    expected_version=constraints["recommended"],
                ))
                
        except ImportError:
            # packaging not available, skip version checking
            pass
    
    return result


def check_filter_script_compatibility(
    script_meta: dict,
    current_snapshot: Optional[EnvironmentSnapshot] = None,
) -> CompatibilityResult:
    """
    Check if a filter script is compatible with current environment.
    
    Args:
        script_meta: Filter script metadata (from meta field)
        current_snapshot: Current environment (captures if None)
    
    Returns:
        CompatibilityResult with issues found
    """
    if current_snapshot is None:
        current_snapshot = capture_environment(ReproducibilityLevel.STANDARD)
    
    result = CompatibilityResult(compatible=True)
    
    # Check MeshPrep version
    script_version = script_meta.get("meshprep_version", "")
    if script_version:
        current_version = current_snapshot.meshprep_version
        
        # Major version must match
        script_major = script_version.split(".")[0]
        current_major = current_version.split(".")[0]
        
        if script_major != current_major:
            result.issues.append(CompatibilityIssue(
                severity="error",
                component="meshprep",
                message=f"Filter script requires MeshPrep {script_version}, but {current_version} is installed",
                current_version=current_version,
                expected_version=script_version,
            ))
            result.compatible = False
        elif script_version != current_version:
            result.issues.append(CompatibilityIssue(
                severity="warning",
                component="meshprep",
                message=f"Filter script created with MeshPrep {script_version}, running on {current_version}",
                current_version=current_version,
                expected_version=script_version,
            ))
    
    # Check tool versions
    script_tools = script_meta.get("tool_versions", {})
    for tool_name, script_tool_version in script_tools.items():
        current_tool_version = current_snapshot.package_versions.get(tool_name)
        
        if current_tool_version and script_tool_version:
            if current_tool_version != script_tool_version:
                result.issues.append(CompatibilityIssue(
                    severity="warning",
                    component=tool_name,
                    message=f"Filter script used {tool_name} {script_tool_version}, current is {current_tool_version}",
                    current_version=current_tool_version,
                    expected_version=script_tool_version,
                ))
    
    return result


# ---------------------------------------------------------------------------
# Reproducibility Block for Reports
# ---------------------------------------------------------------------------

def create_reproducibility_block(
    level: ReproducibilityLevel = ReproducibilityLevel.STANDARD,
    filter_script_hash: Optional[str] = None,
    input_file_hash: Optional[str] = None,
    reproduce_command: Optional[str] = None,
) -> dict:
    """
    Create a reproducibility block for inclusion in reports.
    
    Args:
        level: Reproducibility level
        filter_script_hash: SHA256 hash of filter script
        input_file_hash: SHA256 hash of input file
        reproduce_command: Command to reproduce this run
    
    Returns:
        Dictionary suitable for JSON serialization
    """
    snapshot = capture_environment(level, include_external=True)
    
    block = snapshot.to_dict()
    block["filter_script_hash"] = filter_script_hash
    block["input_file_hash"] = input_file_hash
    block["reproduce_command"] = reproduce_command
    
    return block


# ---------------------------------------------------------------------------
# Console Output
# ---------------------------------------------------------------------------

def print_environment_check(result: Optional[CompatibilityResult] = None) -> None:
    """Print environment check results to console."""
    if result is None:
        result = check_compatibility()
    
    snapshot = capture_environment(ReproducibilityLevel.STANDARD)
    
    print()
    print("MeshPrep Environment Check")
    print("=" * 40)
    print(f"MeshPrep version: {snapshot.meshprep_version}")
    print(f"Python: {snapshot.python_version}")
    print(f"Platform: {snapshot.platform_info}")
    print()
    
    print("Required Packages:")
    matrix = CompatibilityMatrix.load()
    for pkg_name in matrix.required_packages.keys():
        version = snapshot.package_versions.get(pkg_name, "(not installed)")
        recommended = matrix.required_packages[pkg_name].get("recommended", "")
        
        if version == recommended:
            status = "OK"
        elif version == "(not installed)":
            status = "MISSING"
        else:
            status = f"(recommended: {recommended})"
        
        print(f"  {pkg_name:15} {version:12} {status}")
    
    print()
    print("External Tools:")
    for tool_name, tool_info in snapshot.external_tools.items():
        if tool_info.get("available"):
            print(f"  {tool_name:15} {tool_info.get('version', 'unknown'):12} OK")
        else:
            print(f"  {tool_name:15} (not found)")
    
    print()
    print(str(result))
    print()
