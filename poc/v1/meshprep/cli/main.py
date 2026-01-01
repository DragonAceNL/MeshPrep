# Copyright 2025 Dragon Ace
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""Command-line interface for MeshPrep."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..core.mock_mesh import load_mock_stl, save_mock_stl
from ..core.diagnostics import compute_diagnostics
from ..core.profiles import ProfileDetector
from ..core.filter_script import FilterScript, FilterScriptRunner, generate_filter_script


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="meshprep",
        description="MeshPrep - Automated STL Cleanup Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect profile and repair
  python -m meshprep.cli --input model.stl --output ./clean/

  # Use a specific filter script
  python -m meshprep.cli --input model.stl --filter my_filter.json

  # Use a named preset
  python -m meshprep.cli --input model.stl --preset holes-only

  # Dry-run with verbose output
  python -m meshprep.cli --input model.stl --dry-run --verbose
        """,
    )
    
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to input STL file",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./output"),
        help="Directory for cleaned STL output (default: ./output/)",
    )
    
    parser.add_argument(
        "--filter", "-f",
        type=Path,
        help="Path to a filter script (JSON/YAML) to use instead of auto-detection",
    )
    
    parser.add_argument(
        "--preset", "-p",
        type=str,
        help="Name of a preset from filters/ to use",
    )
    
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("./report.json"),
        help="Path for JSON report output (default: ./report.json)",
    )
    
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("./report.csv"),
        help="Path for CSV report output (default: ./report.csv)",
    )
    
    parser.add_argument(
        "--export-run",
        type=Path,
        help="Export reproducible run package to specified directory",
    )
    
    parser.add_argument(
        "--use-blender",
        choices=["always", "on-failure", "never"],
        default="on-failure",
        help="When to use Blender escalation (default: on-failure)",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate filter actions without writing output",
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="MeshPrep POC v0.1.0",
    )
    
    return parser


def log(message: str, level: str = "info", verbose: bool = False):
    """Log a message."""
    if level == "debug" and not verbose:
        return
    
    prefix = {
        "info": "[INFO]",
        "success": "[OK]",
        "warning": "[WARN]",
        "error": "[ERROR]",
        "debug": "[DEBUG]",
    }.get(level, "[INFO]")
    
    print(f"{prefix} {message}")


def run_cli(args: argparse.Namespace) -> int:
    """Run the CLI with parsed arguments."""
    verbose = args.verbose
    
    # Validate input file
    if not args.input.exists():
        log(f"Input file not found: {args.input}", "error")
        return 1
    
    log(f"Processing: {args.input}", "info", verbose)
    
    # Load mesh
    log("Loading model...", "info", verbose)
    mesh = load_mock_stl(args.input)
    log(f"Loaded mesh: {mesh.vertex_count} vertices, {mesh.face_count} faces", "debug", verbose)
    
    # Compute diagnostics
    log("Computing diagnostics...", "info", verbose)
    diagnostics = compute_diagnostics(mesh)
    
    if verbose:
        print("\n" + diagnostics.summary() + "\n")
    
    # Determine filter script
    script: Optional[FilterScript] = None
    
    if args.filter:
        # Load from file
        log(f"Loading filter script: {args.filter}", "info", verbose)
        script = FilterScript.load(args.filter)
    elif args.preset:
        # Load preset
        preset_path = Path("filters") / f"{args.preset}.json"
        if not preset_path.exists():
            preset_path = Path("filters") / f"{args.preset}.yaml"
        
        if preset_path.exists():
            log(f"Loading preset: {args.preset}", "info", verbose)
            script = FilterScript.load(preset_path)
        else:
            log(f"Preset not found: {args.preset}", "error")
            return 1
    else:
        # Auto-detect profile
        log("Detecting profile...", "info", verbose)
        detector = ProfileDetector()
        matches = detector.detect(diagnostics)
        
        if matches:
            match = matches[0]
            log(f"Detected profile: {match.profile.display_name} ({match.confidence:.0%})", "success")
            
            if verbose:
                for reason in match.reasons:
                    log(f"  - {reason}", "debug", verbose)
            
            script = generate_filter_script(
                match.profile.name,
                mesh.fingerprint,
                match.profile.suggested_actions,
            )
        else:
            log("Could not detect profile", "error")
            return 1
    
    log(f"Filter script: {script.name} ({len(script.actions)} actions)", "info", verbose)
    
    # Validate script
    errors = script.validate()
    if errors:
        for error in errors:
            log(error, "error")
        return 1
    
    # Run filter script
    runner = FilterScriptRunner()
    
    def progress_callback(step, total, msg):
        if verbose:
            log(f"Step {step}/{total}: {msg}", "debug", verbose)
    
    runner.set_progress_callback(progress_callback)
    
    log("Running filter script..." if not args.dry_run else "Running dry-run...", "info", verbose)
    result = runner.run(script, mesh, dry_run=args.dry_run)
    
    # Print results
    if verbose:
        print("\n" + result.summary() + "\n")
    
    if result.success:
        log(f"Filter script completed successfully ({result.total_runtime_ms:.1f}ms)", "success")
        
        if not args.dry_run:
            # Save output
            args.output.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = args.output / f"{args.input.stem}__{script.name}__{timestamp}.stl"
            
            if output_file.exists() and not args.overwrite:
                log(f"Output file exists (use --overwrite): {output_file}", "error")
                return 1
            
            save_mock_stl(result.final_mesh, output_file)
            log(f"Output saved to: {output_file}", "success")
        
        # Save report
        report = {
            "timestamp": datetime.now().isoformat(),
            "input_file": str(args.input),
            "script_name": script.name,
            "dry_run": args.dry_run,
            "success": result.success,
            "total_runtime_ms": result.total_runtime_ms,
            "steps": [
                {
                    "step": s.step_number,
                    "action": s.action_name,
                    "status": s.status,
                    "runtime_ms": s.runtime_ms,
                    "error": s.error,
                }
                for s in result.steps
            ],
            "initial_diagnostics": result.initial_diagnostics.to_dict() if result.initial_diagnostics else None,
            "final_diagnostics": result.final_diagnostics.to_dict() if result.final_diagnostics else None,
        }
        
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        log(f"Report saved to: {args.report}", "info", verbose)
        
        # CSV report
        csv_header = "filename,status,script,runtime_ms,watertight_before,watertight_after\n"
        csv_row = (
            f"{args.input.name},"
            f"{'success' if result.success else 'failed'},"
            f"{script.name},"
            f"{result.total_runtime_ms:.1f},"
            f"{result.initial_diagnostics.is_watertight if result.initial_diagnostics else 'N/A'},"
            f"{result.final_diagnostics.is_watertight if result.final_diagnostics else 'N/A'}\n"
        )
        
        if args.csv.exists():
            with open(args.csv, "a") as f:
                f.write(csv_row)
        else:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            with open(args.csv, "w") as f:
                f.write(csv_header + csv_row)
        
        # Export run package if requested
        if args.export_run:
            export_run_package(args, script, result, diagnostics)
        
        return 0
    else:
        log(f"Filter script failed: {result.error}", "error")
        return 1


def export_run_package(args: argparse.Namespace, script: FilterScript, 
                       result, diagnostics):
    """Export a reproducible run package."""
    package_dir = Path(args.export_run)
    package_dir.mkdir(parents=True, exist_ok=True)
    
    # Save filter script
    script.save(package_dir / "filter_script.json")
    
    # Save report
    import shutil
    if args.report.exists():
        shutil.copy(args.report, package_dir / "report.json")
    
    # Save diagnostics
    (package_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics.to_dict(), indent=2)
    )
    
    # Save reproduce instructions
    reproduce = f"""# MeshPrep Run Package

## Reproduce this run

```bash
python -m meshprep.cli --input <your_model.stl> --filter filter_script.json
```

## Original command

```bash
python -m meshprep.cli --input {args.input} --output {args.output}
```

## Environment

- MeshPrep POC v0.1.0
- Python 3.11+
- Generated: {datetime.now().isoformat()}
"""
    
    (package_dir / "README.md").write_text(reproduce)
    
    log(f"Run package exported to: {package_dir}", "success")


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        return run_cli(args)
    except KeyboardInterrupt:
        log("Interrupted by user", "warning")
        return 130
    except Exception as e:
        log(f"Unexpected error: {e}", "error")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
