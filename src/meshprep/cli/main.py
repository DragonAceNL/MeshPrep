# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Command-line interface for MeshPrep.

Provides commands for:
- repair: Repair a single STL file
- diagnose: Analyze a mesh and show diagnostics
- validate: Check if a mesh is printable
- checkenv: Verify the environment is set up correctly
- list-presets: Show available filter script presets
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click

from meshprep import __version__
from meshprep.core import (
    load_mesh,
    save_mesh,
    compute_diagnostics,
    compute_fingerprint,
    format_diagnostics,
    validate_geometry,
    validate_repair,
    format_validation_result,
    FilterScript,
    FilterScriptRunner,
    get_preset,
    list_presets,
    PRESETS,
)


def setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )


@click.group()
@click.version_option(version=__version__, prog_name="meshprep")
def main():
    """
    MeshPrep - Automated STL cleanup pipeline for 3D printing.
    
    Use 'meshprep COMMAND --help' for more information on each command.
    """
    pass


@main.command()
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True),
              help="Path to input STL file")
@click.option("--output", "-o", "output_path", type=click.Path(),
              help="Output file path (default: <input>_repaired.stl)")
@click.option("--filter", "-f", "filter_path", type=click.Path(exists=True),
              help="Path to filter script (JSON/YAML)")
@click.option("--preset", "-p", "preset_name", type=str,
              help="Name of built-in preset to use")
@click.option("--report", "-r", "report_path", type=click.Path(),
              help="Path for JSON report output")
@click.option("--overwrite", is_flag=True, help="Overwrite existing output files")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def repair(
    input_path: str,
    output_path: Optional[str],
    filter_path: Optional[str],
    preset_name: Optional[str],
    report_path: Optional[str],
    overwrite: bool,
    verbose: bool
):
    """
    Repair a mesh file for 3D printing.
    
    Examples:
    
        meshprep repair --input model.stl
        
        meshprep repair -i model.stl -p full-repair
        
        meshprep repair -i model.stl -f my_filter.json -o fixed.stl
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    input_path = Path(input_path)
    
    # Determine output path
    if output_path:
        output_path = Path(output_path)
    else:
        output_path = input_path.parent / f"{input_path.stem}_repaired.stl"
    
    # Check if output exists
    if output_path.exists() and not overwrite:
        click.echo(f"Error: Output file exists: {output_path}")
        click.echo("Use --overwrite to replace it.")
        sys.exit(1)
    
    # Load filter script
    if filter_path:
        click.echo(f"Loading filter script: {filter_path}")
        script = FilterScript.load(filter_path)
    elif preset_name:
        script = get_preset(preset_name)
        if script is None:
            click.echo(f"Error: Unknown preset '{preset_name}'")
            click.echo(f"Available presets: {', '.join(list_presets())}")
            sys.exit(1)
        click.echo(f"Using preset: {preset_name}")
    else:
        # Default to full-repair
        script = get_preset("full-repair")
        click.echo("Using default preset: full-repair")
    
    # Validate script
    errors = script.validate()
    if errors:
        click.echo("Filter script validation errors:")
        for error in errors:
            click.echo(f"  - {error}")
        sys.exit(1)
    
    # Load mesh
    click.echo(f"Loading: {input_path}")
    try:
        mesh = load_mesh(input_path)
    except Exception as e:
        click.echo(f"Error loading mesh: {e}")
        sys.exit(1)
    
    original_mesh = mesh.copy()
    
    # Show initial diagnostics
    diag = compute_diagnostics(mesh)
    click.echo(f"  Vertices: {diag.vertex_count:,}")
    click.echo(f"  Faces: {diag.face_count:,}")
    click.echo(f"  Watertight: {diag.is_watertight}")
    
    # Run filter script
    click.echo(f"\nRunning filter script: {script.name}")
    runner = FilterScriptRunner()
    
    def progress(index: int, name: str, total: int):
        click.echo(f"  [{index+1}/{total}] {name}...")
    
    result = runner.run(script, mesh, progress_callback=progress)
    
    if not result.success:
        click.echo(f"\nRepair failed: {result.error}")
        sys.exit(1)
    
    # Validate result
    validation = validate_repair(original_mesh, result.final_mesh)
    
    click.echo(f"\nRepair completed in {result.total_duration_ms:.1f}ms")
    click.echo(f"  Watertight: {validation.geometric.is_watertight}")
    click.echo(f"  Manifold: {validation.geometric.is_manifold}")
    click.echo(f"  Volume change: {validation.fidelity.volume_change_pct:.2f}%")
    
    # Save output
    click.echo(f"\nSaving: {output_path}")
    save_mesh(result.final_mesh, output_path)
    
    # Generate report
    if report_path:
        report_path = Path(report_path)
        report = {
            "input": str(input_path),
            "output": str(output_path),
            "filter_script": script.name,
            "success": result.success,
            "duration_ms": result.total_duration_ms,
            "validation": validation.to_dict(),
            "fingerprint": compute_fingerprint(original_mesh),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        click.echo(f"Report saved: {report_path}")
    
    # Summary
    if validation.geometric.is_printable:
        click.echo("\n✓ Mesh is ready for 3D printing!")
    else:
        click.echo("\n⚠ Mesh may have issues:")
        for issue in validation.geometric.issues:
            click.echo(f"  - {issue}")


@main.command()
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True),
              help="Path to mesh file")
@click.option("--json", "-j", "json_output", is_flag=True, help="Output as JSON")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def diagnose(input_path: str, json_output: bool, verbose: bool):
    """
    Analyze a mesh and show diagnostics.
    
    Examples:
    
        meshprep diagnose --input model.stl
        
        meshprep diagnose -i model.stl --json
    """
    setup_logging(verbose)
    
    input_path = Path(input_path)
    
    click.echo(f"Loading: {input_path}")
    try:
        mesh = load_mesh(input_path)
    except Exception as e:
        click.echo(f"Error loading mesh: {e}")
        sys.exit(1)
    
    diag = compute_diagnostics(mesh)
    
    if json_output:
        click.echo(json.dumps(diag.to_dict(), indent=2))
    else:
        click.echo(format_diagnostics(diag, f"Diagnostics: {input_path.name}"))
        
        # Print summary
        click.echo("\nStatus:")
        if diag.is_watertight:
            click.echo("  ✓ Watertight")
        else:
            click.echo("  ✗ Not watertight (has holes)")
        
        if diag.is_volume:
            click.echo("  ✓ Valid volume (manifold)")
        else:
            click.echo("  ✗ Non-manifold geometry")
        
        if diag.component_count == 1:
            click.echo("  ✓ Single component")
        else:
            click.echo(f"  ⚠ Multiple components ({diag.component_count})")
        
        if diag.degenerate_face_count == 0:
            click.echo("  ✓ No degenerate faces")
        else:
            click.echo(f"  ⚠ Degenerate faces ({diag.degenerate_face_count})")


@main.command()
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True),
              help="Path to mesh file")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def validate(input_path: str, verbose: bool):
    """
    Check if a mesh is suitable for 3D printing.
    
    Examples:
    
        meshprep validate --input model.stl
    """
    setup_logging(verbose)
    
    input_path = Path(input_path)
    
    click.echo(f"Loading: {input_path}")
    try:
        mesh = load_mesh(input_path)
    except Exception as e:
        click.echo(f"Error loading mesh: {e}")
        sys.exit(1)
    
    result = validate_geometry(mesh)
    
    click.echo(f"\nValidation: {input_path.name}")
    click.echo("=" * 50)
    click.echo(f"Watertight: {'✓' if result.is_watertight else '✗'}")
    click.echo(f"Manifold: {'✓' if result.is_manifold else '✗'}")
    click.echo(f"Positive Volume: {'✓' if result.has_positive_volume else '✗'}")
    click.echo(f"Consistent Winding: {'✓' if result.is_winding_consistent else '✗'}")
    click.echo(f"No Degenerate Faces: {'✓' if result.no_degenerate_faces else '✗'}")
    click.echo("=" * 50)
    
    if result.is_printable:
        click.echo("\n✓ Mesh is ready for 3D printing!")
        sys.exit(0)
    else:
        click.echo("\n✗ Mesh is NOT ready for 3D printing")
        if result.issues:
            click.echo("\nIssues found:")
            for issue in result.issues:
                click.echo(f"  - {issue}")
        click.echo("\nRun 'meshprep repair --input <file>' to fix these issues.")
        sys.exit(1)


@main.command("list-presets")
def list_presets_cmd():
    """
    List available filter script presets.
    """
    click.echo("Available presets:\n")
    for name, script in PRESETS.items():
        click.echo(f"  {name}")
        click.echo(f"    {script.description}")
        click.echo(f"    Actions: {', '.join(a.name for a in script.actions)}")
        click.echo()


@main.command()
def checkenv():
    """
    Check if the environment is set up correctly.
    
    Verifies that all required and optional dependencies are installed.
    """
    click.echo("MeshPrep Environment Check")
    click.echo("=" * 50)
    
    # Python version
    import platform
    py_version = platform.python_version()
    click.echo(f"\nPython: {py_version}")
    
    major, minor = sys.version_info[:2]
    if major == 3 and 11 <= minor <= 12:
        click.echo("  ✓ Python version is compatible")
    else:
        click.echo("  ⚠ Python 3.11 or 3.12 recommended for full functionality")
    
    # Core dependencies
    click.echo("\nCore Dependencies:")
    
    try:
        import trimesh
        click.echo(f"  ✓ trimesh: {trimesh.__version__}")
    except ImportError:
        click.echo("  ✗ trimesh: NOT INSTALLED")
    
    try:
        import numpy
        click.echo(f"  ✓ numpy: {numpy.__version__}")
    except ImportError:
        click.echo("  ✗ numpy: NOT INSTALLED")
    
    try:
        import scipy
        click.echo(f"  ✓ scipy: {scipy.__version__}")
    except ImportError:
        click.echo("  ✗ scipy: NOT INSTALLED")
    
    # Optional dependencies
    click.echo("\nOptional Dependencies:")
    
    try:
        import pymeshfix
        click.echo("  ✓ pymeshfix: installed")
    except ImportError:
        click.echo("  ⚠ pymeshfix: NOT INSTALLED (some repair actions limited)")
    
    try:
        import meshio
        click.echo(f"  ✓ meshio: {meshio.__version__}")
    except ImportError:
        click.echo("  ○ meshio: not installed (optional)")
    
    try:
        import yaml
        click.echo("  ✓ PyYAML: installed")
    except ImportError:
        click.echo("  ○ PyYAML: not installed (JSON-only filter scripts)")
    
    # GUI dependencies
    click.echo("\nGUI Dependencies:")
    
    try:
        from PySide6 import __version__ as pyside_version
        click.echo(f"  ✓ PySide6: {pyside_version}")
    except ImportError:
        click.echo("  ○ PySide6: not installed (CLI-only mode)")
    
    # Blender check
    click.echo("\nExternal Tools:")
    
    import shutil
    blender_path = shutil.which("blender")
    if blender_path:
        click.echo(f"  ✓ Blender: {blender_path}")
    else:
        click.echo("  ○ Blender: not found (escalation unavailable)")
    
    click.echo("\n" + "=" * 50)
    click.echo("Environment check complete.")


if __name__ == "__main__":
    main()
