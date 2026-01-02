#!/usr/bin/env python
# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Validate filter scripts against real Thingi10K test fixtures.

This script proves that our filter scripts actually work by:
1. Loading real STL files from the test fixtures
2. Running actual trimesh/pymeshfix operations
3. Validating that results are printable and visually unchanged
4. Reporting success rates per category

Usage:
    python validate_filters.py
    python validate_filters.py --category holes --limit 5
    python validate_filters.py --all --output results.json
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from meshprep_poc.mesh_ops import load_mesh, save_mesh, compute_diagnostics, print_diagnostics
from meshprep_poc.validation import validate_repair, print_validation_result
from meshprep_poc.filter_script import (
    FilterScriptRunner, 
    FilterScript,
    PRESETS,
    get_preset,
    list_presets
)
from meshprep_poc.actions import ActionRegistry
from meshprep_poc.report import (
    RepairReport,
    create_repair_report,
    generate_markdown_report,
    generate_json_report,
    generate_report_index,
)


# Path to test fixtures (relative to MeshPrep root)
MESHPREP_ROOT = Path(__file__).parent.parent.parent
FIXTURES_PATH = MESHPREP_ROOT / "tests" / "fixtures" / "thingi10k"


@dataclass
class TestResult:
    """Result of testing a single model."""
    file_id: str
    category: str
    filter_script: str
    
    # Repair result
    repair_success: bool
    repair_error: Optional[str]
    repair_duration_ms: float
    
    # Validation result  
    is_geometrically_valid: bool
    is_visually_unchanged: bool
    volume_change_pct: float
    hausdorff_relative: float
    
    # Original state
    original_watertight: bool
    original_volume: bool
    original_faces: int
    
    # Final state
    final_watertight: bool
    final_volume: bool
    final_faces: int
    
    @property
    def overall_success(self) -> bool:
        """Overall success = repair succeeded AND result is valid."""
        return (
            self.repair_success and 
            self.is_geometrically_valid
        )


def get_best_filter_for_category(category: str, mesh: Optional["trimesh.Trimesh"] = None) -> FilterScript:
    """
    Get the best filter script for a defect category.
    
    This maps defect categories to appropriate repair strategies.
    
    If a mesh is provided, it will check for multiple components and
    automatically use conservative repair to preserve them.
    
    Args:
        category: The defect category name
        mesh: Optional mesh to analyze for component count
        
    Returns:
        The best FilterScript for this category/mesh combination
    """
    # Check if mesh has multiple components - if so, use conservative repair
    if mesh is not None:
        try:
            components = mesh.split(only_watertight=False)
            if len(components) > 1:
                logger.info(f"Model has {len(components)} components - using conservative-repair to preserve them")
                return get_preset("conservative-repair")
        except Exception:
            pass  # Fall through to category-based selection
    
    category_to_filter = {
        "clean": get_preset("basic-cleanup"),
        # Use full-repair for holes - but only if single component (checked above)
        "holes": get_preset("full-repair"),
        "many_small_holes": get_preset("full-repair"),
        "non_manifold": get_preset("manifold-repair"),
        "self_intersecting": get_preset("full-repair"),
        # Use conservative repair for multi-component models to preserve parts
        "fragmented": get_preset("conservative-repair"),
        "multiple_components": get_preset("conservative-repair"),
        "complex": get_preset("conservative-repair"),
    }
    
    return category_to_filter.get(category, get_preset("full-repair"))


def test_single_model(
    stl_path: Path,
    category: str,
    filter_script: Optional[FilterScript] = None,
    output_dir: Optional[Path] = None,
    disable_blender_escalation: bool = False,
    generate_report: bool = False,
    report_dir: Optional[Path] = None,
) -> TestResult:
    """
    Test a single model with a filter script.
    
    Blender escalation is AUTOMATIC when:
    1. Blender is available on the system
    2. Primary repair fails to produce a printable result
    
    Args:
        stl_path: Path to STL file
        category: Defect category
        filter_script: Filter script to apply (if None, auto-selects based on mesh)
        output_dir: Optional directory to save repaired model
        disable_blender_escalation: If True, disable automatic Blender escalation
        generate_report: If True, generate a detailed Markdown report
        report_dir: Directory for report files (required if generate_report=True)
        
    Returns:
        TestResult with all metrics
    """
    file_id = stl_path.stem
    escalation_used = False
    escalation_filter_name = None
    
    try:
        # Load mesh first so we can analyze it
        original = load_mesh(stl_path)
        original_diag = compute_diagnostics(original)
        
        # Auto-select filter script if not provided, using mesh analysis
        if filter_script is None:
            filter_script = get_best_filter_for_category(category, original)
        
        action_names = [a.name for a in filter_script.actions if a.enabled]
        
        # Run filter script
        runner = FilterScriptRunner(stop_on_error=False)
        result = runner.run(filter_script, original)
        
        if not result.success or result.final_mesh is None:
            # Generate failure report if requested
            if generate_report and report_dir:
                model_report_dir = report_dir / category / file_id
                repair_report = create_repair_report(
                    original_mesh=original,
                    repaired_mesh=None,
                    input_path=stl_path,
                    output_path=None,
                    filter_script_name=filter_script.name,
                    filter_script_actions=action_names,
                    duration_ms=result.total_duration_ms,
                    success=False,
                    error_message=result.error,
                    report_dir=model_report_dir,
                    render_images=True,
                )
                generate_markdown_report(repair_report, model_report_dir / "report.md")
                generate_json_report(repair_report, model_report_dir / "report.json")
            
            return TestResult(
                file_id=file_id,
                category=category,
                filter_script=filter_script.name,
                repair_success=False,
                repair_error=result.error,
                repair_duration_ms=result.total_duration_ms,
                is_geometrically_valid=False,
                is_visually_unchanged=False,
                volume_change_pct=0,
                hausdorff_relative=0,
                original_watertight=original_diag.is_watertight,
                original_volume=original_diag.is_volume,
                original_faces=original_diag.face_count,
                final_watertight=False,
                final_volume=False,
                final_faces=0
            )
        
        # Validate result
        validation = validate_repair(original, result.final_mesh)
        
        # Compute geometry loss metrics
        import numpy as np
        original_volume = original_diag.volume if original_diag.volume > 0 else 0
        result_volume = result.final_mesh.volume if result.final_mesh.is_volume else 0
        volume_loss_pct = abs(original_volume - result_volume) / original_volume * 100 if original_volume > 0 else 0
        
        original_faces = original_diag.face_count
        result_faces = len(result.final_mesh.faces)
        face_loss_pct = (original_faces - result_faces) / original_faces * 100 if original_faces > 0 else 0
        
        # Detect significant geometry loss (even if result is "printable")
        significant_geometry_loss = (
            volume_loss_pct > 30 or  # Lost more than 30% volume
            face_loss_pct > 40  # Lost more than 40% faces
        )
        
        if significant_geometry_loss:
            logger.warning(f"  Significant geometry loss detected: volume={volume_loss_pct:.1f}%, faces={face_loss_pct:.1f}%")
        
        # Automatic Blender escalation when:
        # 1. Blender is available
        # 2. Not disabled
        # 3. Result is not printable (not watertight/manifold) OR significant geometry loss
        from meshprep_poc.actions.blender_actions import is_blender_available
        
        blender_available = is_blender_available()
        needs_escalation = (
            not disable_blender_escalation and
            blender_available and
            (
                not validation.geometric.is_printable or
                significant_geometry_loss
            ) and
            (result.final_mesh is None or len(result.final_mesh.faces) == 0 or
             not result.final_mesh.is_watertight or significant_geometry_loss)
        )
        
        if needs_escalation:
            logger.info(f"  Primary repair insufficient - auto-escalating to Blender for {file_id}...")
            
            escalation_script = get_preset("blender-remesh")
            if escalation_script:
                escalation_result = runner.run(escalation_script, original)
                
                if escalation_result.success and escalation_result.final_mesh is not None:
                    if len(escalation_result.final_mesh.faces) > 0:
                        result = escalation_result
                        validation = validate_repair(original, result.final_mesh)
                        filter_script = escalation_script
                        escalation_used = True
                        escalation_filter_name = escalation_script.name
                        action_names = [a.name for a in escalation_script.actions if a.enabled]
                        logger.info(f"  Blender escalation successful")
                else:
                    logger.warning(f"  Blender escalation failed")
        elif not validation.geometric.is_printable and not blender_available:
            logger.warning(f"  Primary repair insufficient but Blender not available for escalation")
        
        # Save repaired model if requested
        output_path = None
        if output_dir and validation.geometric.is_printable:
            output_path = output_dir / category / f"{file_id}_repaired.stl"
            save_mesh(result.final_mesh, output_path)
        
        # Generate report if requested
        if generate_report and report_dir:
            model_report_dir = report_dir / category / file_id
            repair_report = create_repair_report(
                original_mesh=original,
                repaired_mesh=result.final_mesh,
                input_path=stl_path,
                output_path=output_path,
                filter_script_name=filter_script.name,
                filter_script_actions=action_names,
                duration_ms=result.total_duration_ms,
                success=validation.geometric.is_printable,
                error_message=None,
                escalation_used=escalation_used,
                escalation_filter=escalation_filter_name,
                report_dir=model_report_dir,
                render_images=True,
            )
            generate_markdown_report(repair_report, model_report_dir / "report.md")
            generate_json_report(repair_report, model_report_dir / "report.json")
        
        final_diag = validation.repaired_diagnostics
        
        return TestResult(
            file_id=file_id,
            category=category,
            filter_script=filter_script.name,
            repair_success=True,
            repair_error=None,
            repair_duration_ms=result.total_duration_ms,
            is_geometrically_valid=validation.geometric.is_printable,
            is_visually_unchanged=validation.fidelity.is_visually_unchanged,
            volume_change_pct=validation.fidelity.volume_change_pct,
            hausdorff_relative=validation.fidelity.hausdorff_relative,
            original_watertight=original_diag.is_watertight,
            original_volume=original_diag.is_volume,
            original_faces=original_diag.face_count,
            final_watertight=final_diag.is_watertight if final_diag else False,
            final_volume=final_diag.is_volume if final_diag else False,
            final_faces=final_diag.face_count if final_diag else 0
        )
        
    except Exception as e:
        logger.error(f"Error testing {file_id}: {e}")
        return TestResult(
            file_id=file_id,
            category=category,
            filter_script=filter_script.name,
            repair_success=False,
            repair_error=str(e),
            repair_duration_ms=0,
            is_geometrically_valid=False,
            is_visually_unchanged=False,
            volume_change_pct=0,
            hausdorff_relative=0,
            original_watertight=False,
            original_volume=False,
            original_faces=0,
            final_watertight=False,
            final_volume=False,
            final_faces=0
        )


def test_category(
    category: str,
    limit: Optional[int] = None,
    output_dir: Optional[Path] = None,
    filter_script: Optional[FilterScript] = None,
    disable_blender_escalation: bool = False,
    generate_reports: bool = False,
    report_dir: Optional[Path] = None,
) -> list[TestResult]:
    """
    Test all models in a category.
    
    Blender escalation is AUTOMATIC when Blender is available and needed.
    
    Args:
        category: Category name (holes, non_manifold, etc.)
        limit: Maximum number of models to test
        output_dir: Optional directory for repaired models
        filter_script: Optional specific filter script (auto-select if None)
        disable_blender_escalation: If True, disable automatic Blender escalation
        generate_reports: If True, generate Markdown reports for each model
        report_dir: Directory for report files
        
    Returns:
        List of TestResult objects
    """
    category_dir = FIXTURES_PATH / category
    
    if not category_dir.exists():
        logger.warning(f"Category directory not found: {category_dir}")
        return []
    
    stl_files = list(category_dir.glob("*.stl"))
    
    if limit:
        stl_files = stl_files[:limit]
    
    if not stl_files:
        logger.warning(f"No STL files found in {category}")
        return []
    
    # Get filter script (for display purposes - actual selection happens per-model)
    # If user specified a filter, use it for all; otherwise we'll auto-select per model
    display_filter_name = "auto-select (per-model)"
    if filter_script is not None:
        display_filter_name = filter_script.name
    else:
        # Get a preview of what category-default would be (without mesh analysis)
        default_filter = get_best_filter_for_category(category)
        display_filter_name = f"{default_filter.name} (or conservative if multi-component)"
    
    # Check Blender availability for display
    from meshprep_poc.actions.blender_actions import is_blender_available
    blender_status = "available (auto-escalation enabled)" if is_blender_available() and not disable_blender_escalation else "not available"
    if disable_blender_escalation and is_blender_available():
        blender_status = "available (auto-escalation disabled)"
    
    logger.info(f"\nTesting category: {category}")
    logger.info(f"  Models: {len(stl_files)}")
    logger.info(f"  Filter: {display_filter_name}")
    logger.info(f"  Blender: {blender_status}")
    if generate_reports:
        logger.info(f"  Report generation: enabled")
    
    results = []
    
    for i, stl_path in enumerate(stl_files):
        logger.info(f"  [{i+1}/{len(stl_files)}] {stl_path.name}...")
        
        # Pass filter_script (could be None for auto-selection)
        result = test_single_model(
            stl_path, category, filter_script, output_dir,
            disable_blender_escalation=disable_blender_escalation,
            generate_report=generate_reports,
            report_dir=report_dir,
        )
        
        status = "OK" if result.overall_success else "FAIL"
        logger.info(f"    {status} - watertight={result.final_watertight}, vol_change={result.volume_change_pct:.2f}%")
        
        results.append(result)
    
    return results


def compute_summary(results: list[TestResult]) -> dict:
    """Compute summary statistics from results."""
    if not results:
        return {}
    
    total = len(results)
    
    # Overall metrics
    repair_success = sum(1 for r in results if r.repair_success)
    geom_valid = sum(1 for r in results if r.is_geometrically_valid)
    visual_unchanged = sum(1 for r in results if r.is_visually_unchanged)
    overall_success = sum(1 for r in results if r.overall_success)
    
    # Average metrics
    avg_duration = sum(r.repair_duration_ms for r in results) / total
    avg_vol_change = sum(abs(r.volume_change_pct) for r in results) / total
    
    # By category
    categories = set(r.category for r in results)
    by_category = {}
    
    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        cat_total = len(cat_results)
        cat_success = sum(1 for r in cat_results if r.overall_success)
        
        by_category[cat] = {
            "total": cat_total,
            "successful": cat_success,
            "success_rate": cat_success / cat_total * 100 if cat_total > 0 else 0
        }
    
    return {
        "total_models": total,
        "repair_success": repair_success,
        "repair_success_rate": repair_success / total * 100,
        "geometrically_valid": geom_valid,
        "geometrically_valid_rate": geom_valid / total * 100,
        "visually_unchanged": visual_unchanged,
        "visually_unchanged_rate": visual_unchanged / total * 100,
        "overall_success": overall_success,
        "overall_success_rate": overall_success / total * 100,
        "avg_duration_ms": avg_duration,
        "avg_volume_change_pct": avg_vol_change,
        "by_category": by_category
    }


def print_summary(summary: dict):
    """Print summary in a nice format."""
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    print(f"\nTotal models tested: {summary['total_models']}")
    print(f"\nSuccess Rates:")
    print(f"  Repair completed:      {summary['repair_success_rate']:5.1f}% ({summary['repair_success']}/{summary['total_models']})")
    print(f"  Geometrically valid:   {summary['geometrically_valid_rate']:5.1f}% ({summary['geometrically_valid']}/{summary['total_models']})")
    print(f"  Visually unchanged:    {summary['visually_unchanged_rate']:5.1f}% ({summary['visually_unchanged']}/{summary['total_models']})")
    print(f"  Overall success:       {summary['overall_success_rate']:5.1f}% ({summary['overall_success']}/{summary['total_models']})")
    
    print(f"\nPerformance:")
    print(f"  Avg duration: {summary['avg_duration_ms']:.1f}ms")
    print(f"  Avg volume change: {summary['avg_volume_change_pct']:.2f}%")
    
    print(f"\nBy Category:")
    for cat, stats in sorted(summary.get("by_category", {}).items()):
        print(f"  {cat:20s}: {stats['success_rate']:5.1f}% ({stats['successful']}/{stats['total']})")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Validate filter scripts against Thingi10K test fixtures"
    )
    
    parser.add_argument(
        "--category", "-c",
        help="Test a specific category (holes, non_manifold, etc.)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Test all categories"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Maximum models per category"
    )
    parser.add_argument(
        "--filter", "-f",
        help="Use a specific filter preset (basic-cleanup, fill-holes, full-repair, manifold-repair)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--save-repaired",
        type=Path,
        help="Directory to save repaired models"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available filter presets"
    )
    parser.add_argument(
        "--list-actions",
        action="store_true",
        help="List available actions"
    )
    parser.add_argument(
        "--no-blender",
        action="store_true",
        help="Disable automatic Blender escalation (by default, Blender is used when available and needed)"
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        help="Directory to generate Markdown reports with before/after images"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # List presets
    if args.list_presets:
        print("\nAvailable filter presets:")
        for name in list_presets():
            preset = get_preset(name)
            print(f"  {name}: {preset.description}")
        return
    
    # List actions
    if args.list_actions:
        print("\nAvailable actions:")
        for name, action in ActionRegistry.get_all().items():
            print(f"  {name}: {action.description}")
        return
    
    # Check fixtures exist
    if not FIXTURES_PATH.exists():
        logger.error(f"Fixtures not found at: {FIXTURES_PATH}")
        logger.error("Run 'python scripts/manage_fixtures.py build' first")
        sys.exit(1)
    
    # Get filter script if specified
    filter_script = None
    if args.filter:
        filter_script = get_preset(args.filter)
        if filter_script is None:
            logger.error(f"Unknown filter preset: {args.filter}")
            logger.error(f"Available: {', '.join(list_presets())}")
            sys.exit(1)
    
    # Determine categories to test
    if args.all:
        categories = [d.name for d in FIXTURES_PATH.iterdir() if d.is_dir()]
    elif args.category:
        categories = [args.category]
    else:
        # Default: test a few categories
        categories = ["clean", "holes", "non_manifold"]
    
    # Run tests
    all_results = []
    
    # Determine if we should generate reports
    generate_reports = args.report_dir is not None
    
    for category in categories:
        results = test_category(
            category,
            limit=args.limit,
            output_dir=args.save_repaired,
            filter_script=filter_script,
            disable_blender_escalation=args.no_blender,
            generate_reports=generate_reports,
            report_dir=args.report_dir,
        )
        all_results.extend(results)
    
    # Compute and print summary
    if all_results:
        summary = compute_summary(all_results)
        print_summary(summary)
        
        # Generate report index if reports were generated
        if generate_reports and args.report_dir:
            generate_report_index(args.report_dir)
            logger.info(f"\nReport index generated at: {args.report_dir / 'index.md'}")
        
        # Save results
        if args.output:
            output_data = {
                "timestamp": datetime.now().isoformat(),
                "summary": summary,
                "results": [asdict(r) for r in all_results]
            }
            
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"\nResults saved to: {args.output}")
    else:
        logger.warning("No results to summarize")


if __name__ == "__main__":
    main()
