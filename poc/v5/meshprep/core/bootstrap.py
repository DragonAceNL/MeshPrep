# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Bootstrap manager for automatic dependency setup."""

import sys
import subprocess
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging

logger = logging.getLogger(__name__)


class BootstrapManager:
    """
    Manages automatic installation of dependencies.
    
    Ensures MeshPrep can run with zero manual setup by:
    - Detecting missing dependencies
    - Installing them automatically (with user consent)
    - Handling optional dependencies
    - Providing fallback options
    """
    
    # Core dependencies (required)
    CORE_DEPS = {
        "numpy": "numpy>=1.24.0",
        "trimesh": "trimesh>=4.0.0",
        "click": "click>=8.0.0",
    }
    
    # Optional dependencies (for advanced features)
    OPTIONAL_DEPS = {
        "torch": ("torch>=2.0.0", "ML prediction"),
        "pymeshfix": ("pymeshfix>=0.16.0", "PyMeshFix repair"),
        "open3d": ("open3d>=0.17.0", "Open3D reconstruction"),
    }
    
    def __init__(self, auto_install: Optional[bool] = None):
        """
        Initialize bootstrap manager.
        
        Args:
            auto_install: Auto-install missing packages (None = prompt first time)
        """
        self.config_dir = Path.home() / ".meshprep"
        self.config_file = self.config_dir / "config.json"
        
        # Load or create config
        self.config = self._load_config()
        
        if auto_install is not None:
            self.config["auto_install"] = auto_install
        
        self.missing_core = []
        self.missing_optional = []
    
    def _load_config(self) -> Dict:
        """Load configuration from disk."""
        if self.config_file.exists():
            try:
                return json.loads(self.config_file.read_text())
            except Exception as e:
                logger.warning(f"Could not load config: {e}")
        
        # Default config
        return {
            "auto_install": None,  # None = prompt first time
            "skip_optional": False,
            "first_run": True,
        }
    
    def _save_config(self):
        """Save configuration to disk."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file.write_text(json.dumps(self.config, indent=2))
    
    def check_dependency(self, module_name: str) -> bool:
        """Check if a module is installed."""
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    
    def install_package(self, package_spec: str) -> Tuple[bool, str]:
        """
        Install a package using pip.
        
        Returns:
            (success, message)
        """
        try:
            logger.info(f"Installing {package_spec}...")
            
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package_spec],
                capture_output=True,
                text=True,
                timeout=300,
            )
            
            if result.returncode == 0:
                return True, f"✓ Installed {package_spec}"
            else:
                return False, f"✗ Failed to install {package_spec}: {result.stderr}"
        
        except Exception as e:
            return False, f"✗ Error installing {package_spec}: {e}"
    
    def check_and_install_core(self, prompt: bool = True) -> bool:
        """
        Check and install core dependencies.
        
        Returns:
            True if all core deps available
        """
        # Check what's missing
        self.missing_core = []
        for module, package in self.CORE_DEPS.items():
            if not self.check_dependency(module):
                self.missing_core.append(package)
        
        if not self.missing_core:
            return True
        
        # Prompt user on first run
        if prompt and self.config.get("first_run") and self.config.get("auto_install") is None:
            print("\n" + "=" * 60)
            print("MeshPrep First-Time Setup")
            print("=" * 60)
            print("\nMissing required dependencies:")
            for pkg in self.missing_core:
                print(f"  • {pkg}")
            
            response = input("\nInstall automatically? [Y/n]: ").strip().lower()
            auto_install = response in ["", "y", "yes"]
            
            self.config["auto_install"] = auto_install
            self.config["first_run"] = False
            self._save_config()
        
        # Install if auto_install enabled
        if self.config.get("auto_install", False):
            print("\nInstalling dependencies...")
            for package in self.missing_core:
                success, message = self.install_package(package)
                print(f"  {message}")
                if not success:
                    return False
            
            return True
        else:
            print("\nPlease install manually:")
            print(f"  pip install {' '.join(self.missing_core)}")
            return False
    
    def check_optional_dependencies(self) -> Dict[str, bool]:
        """Check status of optional dependencies."""
        status = {}
        for module, (package, description) in self.OPTIONAL_DEPS.items():
            status[module] = self.check_dependency(module)
        return status
    
    def install_optional(self, module_name: str) -> bool:
        """Install an optional dependency."""
        if module_name not in self.OPTIONAL_DEPS:
            return False
        
        package, description = self.OPTIONAL_DEPS[module_name]
        success, message = self.install_package(package)
        print(message)
        return success
    
    def setup_environment(self) -> bool:
        """
        Complete environment setup.
        
        Returns:
            True if core dependencies satisfied
        """
        # Check core
        if not self.check_and_install_core():
            logger.error("Core dependencies not satisfied")
            return False
        
        # Check optional (informational only)
        if self.config.get("first_run", True):
            optional_status = self.check_optional_dependencies()
            
            missing_optional = [
                f"{pkg} ({self.OPTIONAL_DEPS[pkg][1]})"
                for pkg, available in optional_status.items()
                if not available
            ]
            
            if missing_optional:
                print("\nOptional features available with:")
                for pkg in missing_optional:
                    print(f"  • {pkg}")
                print("\nInstall with: pip install meshprep[all]")
        
        return True
    
    def get_feature_availability(self) -> Dict[str, bool]:
        """Get availability of all features."""
        return {
            "core": all(self.check_dependency(m) for m in self.CORE_DEPS.keys()),
            "ml": self.check_dependency("torch"),
            "pymeshfix": self.check_dependency("pymeshfix"),
            "open3d": self.check_dependency("open3d"),
            "blender": self._check_blender(),
        }
    
    def _check_blender(self) -> bool:
        """Check if Blender is available."""
        import shutil
        from pathlib import Path
        
        # Check if in PATH first
        if shutil.which("blender"):
            return True
        
        # Check common installation directories
        blender_dirs = [
            Path(r"C:\Program Files\Blender Foundation"),
            Path(r"C:\Program Files (x86)\Blender Foundation"),
            Path.home() / "AppData" / "Local" / "Blender Foundation",
        ]
        
        for base_dir in blender_dirs:
            if base_dir.exists():
                # Look for any Blender version folder
                for subdir in base_dir.iterdir():
                    if subdir.is_dir() and subdir.name.lower().startswith("blender"):
                        exe = subdir / "blender.exe"
                        if exe.exists():
                            return True
        
        return False
    
    def get_blender_path(self) -> Optional[Path]:
        """Get the path to the Blender executable."""
        import shutil
        from pathlib import Path
        
        # Check if in PATH first
        which_result = shutil.which("blender")
        if which_result:
            return Path(which_result)
        
        # Check common installation directories
        blender_dirs = [
            Path(r"C:\Program Files\Blender Foundation"),
            Path(r"C:\Program Files (x86)\Blender Foundation"),
            Path.home() / "AppData" / "Local" / "Blender Foundation",
        ]
        
        for base_dir in blender_dirs:
            if base_dir.exists():
                # Look for any Blender version folder (prefer higher versions)
                versions = []
                for subdir in base_dir.iterdir():
                    if subdir.is_dir() and subdir.name.lower().startswith("blender"):
                        exe = subdir / "blender.exe"
                        if exe.exists():
                            versions.append(exe)
                
                if versions:
                    # Sort by name (higher version numbers come last)
                    versions.sort(key=lambda p: p.parent.name)
                    return versions[-1]  # Return highest version
        
        return None


# Singleton instance
_bootstrap_manager = None


def get_bootstrap_manager() -> BootstrapManager:
    """Get singleton bootstrap manager."""
    global _bootstrap_manager
    if _bootstrap_manager is None:
        _bootstrap_manager = BootstrapManager()
    return _bootstrap_manager


def ensure_environment() -> bool:
    """
    Ensure environment is ready (called on first import).
    
    Returns:
        True if environment ready
    """
    manager = get_bootstrap_manager()
    return manager.setup_environment()
