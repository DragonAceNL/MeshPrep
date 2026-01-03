#!/usr/bin/env python3
# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
MeshPrep Environment Check Script.

Validates the current environment and reports version information
for reproducibility and troubleshooting.

Usage:
    python checkenv.py [--json] [--export <path>] [--strict]
"""

import argparse
import json
import sys
from pathlib import Path

# Add poc/v2 to path for importing meshprep_poc
script_dir = Path(__file__).parent
repo_root = script_dir.parent
poc_v2_dir = repo_root / "poc" / "v2"
sys.path.insert(0, str(poc_v2_dir))


def main():
    parser = argparse.ArgumentParser(
        description="MeshPrep Environment Check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of human-readable text",
    )
    parser.add_argument(
        "--export",
        type=Path,
        metavar="PATH",
        help="Export environment snapshot to file",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors (for CI)",
    )
    parser.add_argument(
        "--include-external",
        action="store_true",
        default=True,
        help="Check external tools (Blender, slicers)",
    )
    
    args = parser.parse_args()
    
    try:
        from meshprep_poc.reproducibility import (
            capture_environment,
            check_compatibility,
            export_environment,
            print_environment_check,
            ReproducibilityLevel,
            CompatibilityMatrix,
        )
    except ImportError as e:
        print(f"Error: Could not import meshprep_poc: {e}", file=sys.stderr)
        print("Make sure you're running from the MeshPrep repository root.", file=sys.stderr)
        sys.exit(1)
    
    # Capture environment
    level = ReproducibilityLevel.STRICT if args.strict else ReproducibilityLevel.STANDARD
    snapshot = capture_environment(level, include_external=args.include_external)
    
    # Check compatibility
    result = check_compatibility(strict=args.strict)
    
    # Export if requested
    if args.export:
        export_environment(args.export, level)
        print(f"Environment exported to: {args.export}")
    
    # Output
    if args.json:
        output = {
            "environment": snapshot.to_dict(),
            "compatibility": {
                "compatible": result.compatible,
                "issues": [
                    {
                        "severity": i.severity,
                        "component": i.component,
                        "message": i.message,
                        "current_version": i.current_version,
                        "expected_version": i.expected_version,
                    }
                    for i in result.issues
                ],
            },
        }
        print(json.dumps(output, indent=2))
    else:
        print()
        print("=" * 50)
        print("MeshPrep Environment Check")
        print("=" * 50)
        print()
        print(f"MeshPrep version: {snapshot.meshprep_version}")
        if snapshot.meshprep_commit:
            print(f"Git commit: {snapshot.meshprep_commit}")
        print(f"Python: {snapshot.python_version}")
        print(f"Platform: {snapshot.platform_info}")
        print()
        
        # Load compatibility matrix for recommended versions
        matrix = CompatibilityMatrix.load()
        
        print("Required Packages:")
        print("-" * 50)
        for pkg_name in matrix.required_packages.keys():
            version = snapshot.package_versions.get(pkg_name, "(not installed)")
            recommended = matrix.required_packages[pkg_name].get("recommended", "")
            
            if version == "(not installed)":
                status = "MISSING"
                marker = "!"
            elif version == recommended:
                status = "OK"
                marker = "+"
            else:
                status = f"(recommended: {recommended})"
                marker = "~"
            
            print(f"  [{marker}] {pkg_name:15} {version:12} {status}")
        
        print()
        print("External Tools:")
        print("-" * 50)
        for tool_name, tool_info in snapshot.external_tools.items():
            if tool_info.get("available"):
                version = tool_info.get("version", "unknown")
                print(f"  [+] {tool_name:15} {version:12} OK")
            else:
                print(f"  [-] {tool_name:15} (not found)")
        
        print()
        print("Compatibility Check:")
        print("-" * 50)
        if result.compatible:
            if not result.issues:
                print("  [+] Environment is COMPATIBLE")
            else:
                print("  [~] Environment is COMPATIBLE (with notes)")
        else:
            print("  [!] Environment is INCOMPATIBLE")
        
        if result.issues:
            print()
            for issue in result.issues:
                prefix = {"error": "[!]", "warning": "[~]", "info": "[i]"}.get(issue.severity, "[?]")
                print(f"  {prefix} {issue.component}: {issue.message}")
        
        print()
    
    # Exit code
    if not result.compatible:
        sys.exit(1)
    elif args.strict and result.warnings:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
