# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Report generation for mesh repair operations.

Generates comprehensive Markdown reports with:
- Before/after 3D model images
- Detailed diagnostics comparison
- Repair action summary
- Success/failure status
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import json

import numpy as np
import trimesh

from .mesh_ops import MeshDiagnostics, compute_diagnostics
from .reproducibility import (
    ReproducibilityLevel,
    capture_environment,
    create_reproducibility_block,
)
from .quality_feedback import (
    QualityFeedbackEngine,
    QualityRating,
    QualityPrediction,
    get_quality_engine,
)

logger = logging.getLogger(__name__)


def render_mesh_image(
    mesh: trimesh.Trimesh,
    output_path: Path,
    resolution: tuple[int, int] = (800, 600),
    background_color: tuple[int, int, int, int] = (255, 255, 255, 255),
    mesh_color: tuple[int, int, int, int] = (100, 149, 237, 255),  # Cornflower blue
) -> bool:
    """
    Render a mesh to an image file.
    
    Args:
        mesh: The mesh to render
        output_path: Path to save the image (PNG)
        resolution: Image resolution (width, height)
        background_color: RGBA background color
        mesh_color: RGBA mesh color
        
    Returns:
        True if rendering succeeded, False otherwise
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a copy with the desired color
        mesh_copy = mesh.copy()
        mesh_copy.visual.face_colors = mesh_color
        
        # Create a scene with the mesh
        scene = trimesh.Scene(mesh_copy)
        
        # Try to render using pyrender or pyglet
        try:
            # Use trimesh's built-in rendering
            png_data = scene.save_image(
                resolution=resolution,
                background=background_color
            )
            
            if png_data is not None:
                with open(output_path, 'wb') as f:
                    f.write(png_data)
                logger.info(f"Rendered image to {output_path}")
                return True
        except Exception as e:
            logger.warning(f"Trimesh rendering failed: {e}")
        
        # Fallback: try matplotlib-based rendering
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            
            fig = plt.figure(figsize=(resolution[0]/100, resolution[1]/100), dpi=100)
            ax = fig.add_subplot(111, projection='3d')
            
            # Get vertices and faces
            vertices = mesh.vertices
            faces = mesh.faces
            
            # Create polygon collection
            mesh_collection = Poly3DCollection(
                vertices[faces],
                alpha=0.8,
                facecolor=[c/255 for c in mesh_color[:3]],
                edgecolor='gray',
                linewidth=0.1
            )
            ax.add_collection3d(mesh_collection)
            
            # Set axis limits
            bounds = mesh.bounds
            center = (bounds[0] + bounds[1]) / 2
            max_range = np.max(bounds[1] - bounds[0]) / 2
            
            ax.set_xlim(center[0] - max_range, center[0] + max_range)
            ax.set_ylim(center[1] - max_range, center[1] + max_range)
            ax.set_zlim(center[2] - max_range, center[2] + max_range)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # Set background color
            ax.set_facecolor([c/255 for c in background_color[:3]])
            fig.patch.set_facecolor([c/255 for c in background_color[:3]])
            
            plt.savefig(output_path, dpi=100, bbox_inches='tight', 
                       facecolor=fig.get_facecolor())
            plt.close(fig)
            
            logger.info(f"Rendered image using matplotlib to {output_path}")
            return True
            
        except ImportError:
            logger.warning("matplotlib not available for fallback rendering")
        except Exception as e:
            logger.warning(f"Matplotlib rendering failed: {e}")
        
        return False
        
    except Exception as e:
        logger.error(f"Failed to render mesh image: {e}")
        return False


@dataclass
class RepairReport:
    """Complete repair report data."""
    
    # File information
    input_file: str
    output_file: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Diagnostics
    original_diagnostics: Optional[MeshDiagnostics] = None
    repaired_diagnostics: Optional[MeshDiagnostics] = None
    
    # Repair information
    filter_script_name: str = ""
    filter_script_actions: list[str] = field(default_factory=list)
    repair_duration_ms: float = 0.0
    
    # Status
    success: bool = False
    error_message: Optional[str] = None
    escalation_used: bool = False
    escalation_filter: Optional[str] = None
    
    # Image paths (relative to report)
    before_image: Optional[str] = None
    after_image: Optional[str] = None
    
    # Model file paths (relative to report)
    before_model: Optional[str] = None
    after_model: Optional[str] = None
    
    # Reproducibility information
    reproducibility: Optional[dict[str, Any]] = None
    model_fingerprint: Optional[str] = None
    filter_script_hash: Optional[str] = None
    
    # Quality validation (Level 4)
    quality_validation: Optional[dict[str, Any]] = None


def generate_markdown_report(
    report: RepairReport,
    output_path: Path,
    index_path: Optional[str] = None,
) -> None:
    """
    Generate a comprehensive Markdown report.
    
    Args:
        report: The repair report data
        output_path: Path to save the Markdown file
        index_path: Optional relative path to the index file for navigation
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = []
    
    # Navigation bar
    if index_path:
        lines.append(f"[<- Back to Index]({index_path})")
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # Header
    lines.append(f"# MeshPrep Repair Report")
    lines.append("")
    lines.append(f"**Generated:** {report.timestamp}")
    lines.append("")
    
    # Status banner
    status_emoji = "✅" if report.success else "❌"
    status_text = "SUCCESS" if report.success else "FAILED"
    lines.append(f"## Status: {status_emoji} {status_text}")
    lines.append("")
    
    if report.error_message:
        lines.append(f"> **Error:** {report.error_message}")
        lines.append("")
    
    # File information
    lines.append("## File Information")
    lines.append("")
    lines.append(f"| Property | Value |")
    lines.append(f"|----------|-------|")
    lines.append(f"| **Input File** | `{report.input_file}` |")
    if report.output_file:
        lines.append(f"| **Output File** | `{report.output_file}` |")
    lines.append(f"| **Filter Script** | `{report.filter_script_name}` |")
    lines.append(f"| **Duration** | {report.repair_duration_ms:.1f} ms |")
    if report.escalation_used:
        lines.append(f"| **Escalation** | Yes (`{report.escalation_filter}`) |")
    lines.append("")
    
    # Before/After Images
    if report.before_image or report.after_image:
        lines.append("## Visual Comparison")
        lines.append("")
        
        # Use ./ prefix for relative paths to ensure proper resolution
        # Show images in a side-by-side table format with HTML img tags for better compatibility
        lines.append("<table>")
        lines.append("<tr><th>Before (Original)</th><th>After (Repaired)</th></tr>")
        lines.append("<tr>")
        
        if report.before_image:
            before_path = report.before_image.replace("\\", "/")
            lines.append(f'<td><img src="./{before_path}" alt="Before" width="400"/></td>')
        else:
            lines.append("<td><em>No image</em></td>")
        
        if report.after_image:
            after_path = report.after_image.replace("\\", "/")
            lines.append(f'<td><img src="./{after_path}" alt="After" width="400"/></td>')
        else:
            lines.append("<td><em>No image</em></td>")
        
        lines.append("</tr>")
        lines.append("</table>")
        lines.append("")
        
        # Also add standard Markdown images as fallback
        lines.append("<!-- Markdown fallback for viewers that don't support HTML -->")
        if report.before_image:
            before_path = report.before_image.replace("\\", "/")
            lines.append(f"")
            lines.append(f"**Before:** ![Before - Original Model](./{before_path})")
        
        if report.after_image:
            after_path = report.after_image.replace("\\", "/")
            lines.append(f"")
            lines.append(f"**After:** ![After - Repaired Model](./{after_path})")
        
        lines.append("")
    
    # Download 3D Models
    if report.before_model or report.after_model:
        lines.append("## 3D Model Files")
        lines.append("")
        lines.append("Download the STL files to compare in your favorite 3D viewer:")
        lines.append("")
        lines.append("| Model | File | Size Info |")
        lines.append("|-------|------|-----------|")
        
        if report.before_model:
            before_model_path = report.before_model.replace("\\", "/")
            orig_faces = report.original_diagnostics.face_count if report.original_diagnostics else 0
            orig_verts = report.original_diagnostics.vertex_count if report.original_diagnostics else 0
            lines.append(f"| **Before (Original)** | [📥 Download](./{before_model_path}) | {orig_verts:,} vertices, {orig_faces:,} faces |")
        
        if report.after_model:
            after_model_path = report.after_model.replace("\\", "/")
            rep_faces = report.repaired_diagnostics.face_count if report.repaired_diagnostics else 0
            rep_verts = report.repaired_diagnostics.vertex_count if report.repaired_diagnostics else 0
            lines.append(f"| **After (Repaired)** | [📥 Download](./{after_model_path}) | {rep_verts:,} vertices, {rep_faces:,} faces |")
        
        lines.append("")
    
    # Diagnostics Comparison
    lines.append("## Diagnostics Comparison")
    lines.append("")
    
    orig = report.original_diagnostics
    rep = report.repaired_diagnostics
    
    if orig and rep:
        lines.append("### Geometry")
        lines.append("")
        lines.append("| Metric | Before | After | Change |")
        lines.append("|--------|--------|-------|--------|")
        
        # Vertices
        v_change = rep.vertex_count - orig.vertex_count
        v_pct = (v_change / orig.vertex_count * 100) if orig.vertex_count > 0 else 0
        v_emoji = "🔺" if v_change > 0 else ("🔻" if v_change < 0 else "➖")
        lines.append(f"| Vertices | {orig.vertex_count:,} | {rep.vertex_count:,} | {v_emoji} {v_change:+,} ({v_pct:+.1f}%) |")
        
        # Faces
        f_change = rep.face_count - orig.face_count
        f_pct = (f_change / orig.face_count * 100) if orig.face_count > 0 else 0
        f_emoji = "🔺" if f_change > 0 else ("🔻" if f_change < 0 else "➖")
        lines.append(f"| Faces | {orig.face_count:,} | {rep.face_count:,} | {f_emoji} {f_change:+,} ({f_pct:+.1f}%) |")
        
        # Volume
        vol_change = rep.volume - orig.volume
        vol_pct = (vol_change / orig.volume * 100) if orig.volume != 0 else 0
        lines.append(f"| Volume | {orig.volume:.4f} | {rep.volume:.4f} | {vol_pct:+.1f}% |")
        
        # Surface Area
        area_change = rep.surface_area - orig.surface_area
        area_pct = (area_change / orig.surface_area * 100) if orig.surface_area > 0 else 0
        lines.append(f"| Surface Area | {orig.surface_area:.4f} | {rep.surface_area:.4f} | {area_pct:+.1f}% |")
        
        # Bounding Box
        lines.append(f"| Bbox Diagonal | {orig.bbox_diagonal:.4f} | {rep.bbox_diagonal:.4f} | — |")
        lines.append("")
        
        # Quality Flags
        lines.append("### Quality Flags")
        lines.append("")
        lines.append("| Flag | Before | After |")
        lines.append("|------|--------|-------|")
        
        def flag_emoji(before: bool, after: bool) -> tuple[str, str]:
            b = "✅" if before else "❌"
            a = "✅" if after else "❌"
            return b, a
        
        wt_b, wt_a = flag_emoji(orig.is_watertight, rep.is_watertight)
        lines.append(f"| Watertight | {wt_b} | {wt_a} |")
        
        vol_b, vol_a = flag_emoji(orig.is_volume, rep.is_volume)
        lines.append(f"| Manifold (is_volume) | {vol_b} | {vol_a} |")
        
        wind_b, wind_a = flag_emoji(orig.is_winding_consistent, rep.is_winding_consistent)
        lines.append(f"| Winding Consistent | {wind_b} | {wind_a} |")
        lines.append("")
        
        # Defects
        lines.append("### Defects")
        lines.append("")
        lines.append("| Defect | Before | After |")
        lines.append("|--------|--------|-------|")
        
        def defect_status(before: int, after: int) -> str:
            if before == 0 and after == 0:
                return "✅ None"
            elif after == 0:
                return "✅ Fixed"
            elif after < before:
                return f"⚠️ {after} (was {before})"
            elif after == before:
                return f"❌ {after}"
            else:
                return f"❌ {after} (was {before})"
        
        lines.append(f"| Boundary Edges | {orig.boundary_edge_count} | {defect_status(orig.boundary_edge_count, rep.boundary_edge_count)} |")
        lines.append(f"| Estimated Holes | {orig.hole_count} | {defect_status(orig.hole_count, rep.hole_count)} |")
        lines.append(f"| Components | {orig.component_count} | {rep.component_count} |")
        lines.append(f"| Degenerate Faces | {orig.degenerate_face_count} | {defect_status(orig.degenerate_face_count, rep.degenerate_face_count)} |")
        lines.append(f"| Euler Characteristic | {orig.euler_characteristic} | {rep.euler_characteristic} |")
        lines.append("")
    
    # Actions performed
    if report.filter_script_actions:
        lines.append("## Actions Performed")
        lines.append("")
        for i, action in enumerate(report.filter_script_actions, 1):
            lines.append(f"{i}. `{action}`")
        lines.append("")
    
    # Printability Assessment
    lines.append("## Printability Assessment")
    lines.append("")
    
    if rep:
        issues = []
        if not rep.is_watertight:
            issues.append("❌ Mesh is not watertight (has holes)")
        if not rep.is_volume:
            issues.append("❌ Mesh is not manifold")
        if rep.component_count > 1:
            issues.append(f"⚠️ Mesh has {rep.component_count} separate components")
        if rep.degenerate_face_count > 0:
            issues.append(f"⚠️ Mesh has {rep.degenerate_face_count} degenerate faces")
        
        if not issues:
            lines.append("✅ **Model appears ready for 3D printing!**")
            lines.append("")
            lines.append("The mesh is:")
            lines.append("- Watertight (no holes)")
            lines.append("- Manifold (valid topology)")
            lines.append("- Single component")
        else:
            lines.append("⚠️ **Model may have issues for 3D printing:**")
            lines.append("")
            for issue in issues:
                lines.append(f"- {issue}")
    else:
        lines.append("*(No diagnostics available)*")
    
    lines.append("")
    
    # Footer
    lines.append("---")
    lines.append("")
    lines.append(f"*Report generated by [MeshPrep](https://github.com/DragonAceNL/MeshPrep)*")
    lines.append("")
    
    # Write the file
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Markdown report saved to {output_path}")


def generate_json_report(
    report: RepairReport,
    output_path: Path,
    include_reproducibility: bool = True,
) -> None:
    """
    Generate a JSON report for programmatic consumption.
    
    Args:
        report: The repair report data
        output_path: Path to save the JSON file
        include_reproducibility: Whether to include reproducibility block
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "version": "1.0.0",
        "timestamp": report.timestamp,
        "input_file": report.input_file,
        "output_file": report.output_file,
        "filter_script": report.filter_script_name,
        "actions": report.filter_script_actions,
        "duration_ms": report.repair_duration_ms,
        "success": report.success,
        "error_message": report.error_message,
        "escalation_used": report.escalation_used,
        "escalation_filter": report.escalation_filter,
        "images": {
            "before": report.before_image,
            "after": report.after_image,
        },
        "models": {
            "before": report.before_model,
            "after": report.after_model,
        },
        "original_diagnostics": report.original_diagnostics.to_dict() if report.original_diagnostics else None,
        "repaired_diagnostics": report.repaired_diagnostics.to_dict() if report.repaired_diagnostics else None,
    }
    
    # Add reproducibility information
    if include_reproducibility:
        if report.reproducibility:
            data["reproducibility"] = report.reproducibility
        else:
            # Capture current environment
            data["reproducibility"] = create_reproducibility_block(
                level=ReproducibilityLevel.STANDARD,
                filter_script_hash=report.filter_script_hash,
                input_file_hash=report.model_fingerprint,
            )
    
    # Add fingerprint if available
    if report.model_fingerprint:
        data["model_fingerprint"] = report.model_fingerprint
    
    # Add quality validation block
    if report.quality_validation:
        data["quality_validation"] = report.quality_validation
    else:
        # Generate default unverified quality validation block with prediction
        quality_block = {
            "status": "unverified",
            "user_rating": None,
            "user_rating_type": None,
            "user_comment": None,
            "rated_by": None,
            "rated_at": None,
        }
        
        # Add automatic metrics if we have diagnostics
        if report.original_diagnostics and report.repaired_diagnostics:
            orig = report.original_diagnostics
            rep = report.repaired_diagnostics
            
            volume_change_pct = 0.0
            if orig.volume != 0:
                volume_change_pct = ((rep.volume - orig.volume) / abs(orig.volume)) * 100
            
            face_count_change_pct = 0.0
            if orig.face_count > 0:
                face_count_change_pct = ((rep.face_count - orig.face_count) / orig.face_count) * 100
            
            quality_block["automatic_metrics"] = {
                "volume_change_pct": round(volume_change_pct, 2),
                "face_count_change_pct": round(face_count_change_pct, 2),
                "hausdorff_distance": None,  # Would require mesh comparison
                "chamfer_distance": None,
                "detail_preservation": None,
                "silhouette_similarity": None,
            }
            
            # Try to get quality prediction
            try:
                quality_engine = get_quality_engine()
                # Determine profile (default to "standard" if not available)
                profile = "standard"
                prediction = quality_engine.predict_quality(
                    pipeline=report.filter_script_name,
                    profile=profile,
                    volume_change_pct=volume_change_pct,
                    face_count_change_pct=face_count_change_pct,
                    escalated=report.escalation_used,
                )
                quality_block["predicted_quality"] = {
                    "score": round(prediction.score, 2),
                    "confidence": round(prediction.confidence, 2),
                    "based_on_samples": prediction.based_on_samples,
                    "warning": prediction.warning or "No user verification - predicted score only",
                }
            except Exception as e:
                logger.debug(f"Could not generate quality prediction: {e}")
                quality_block["predicted_quality"] = {
                    "score": None,
                    "confidence": 0.0,
                    "based_on_samples": 0,
                    "warning": "Quality prediction unavailable",
                }
        
        data["quality_validation"] = quality_block
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"JSON report saved to {output_path}")


def create_repair_report(
    original_mesh: trimesh.Trimesh,
    repaired_mesh: Optional[trimesh.Trimesh],
    input_path: Path,
    output_path: Optional[Path],
    filter_script_name: str,
    filter_script_actions: list[str],
    duration_ms: float,
    success: bool,
    error_message: Optional[str] = None,
    escalation_used: bool = False,
    escalation_filter: Optional[str] = None,
    report_dir: Optional[Path] = None,
    render_images: bool = True,
) -> RepairReport:
    """
    Create a complete repair report with optional image rendering.
    
    Args:
        original_mesh: The original mesh before repair
        repaired_mesh: The repaired mesh (or None if failed)
        input_path: Path to input file
        output_path: Path to output file (if saved)
        filter_script_name: Name of filter script used
        filter_script_actions: List of action names executed
        duration_ms: Total repair duration in milliseconds
        success: Whether repair was successful
        error_message: Error message if failed
        escalation_used: Whether Blender escalation was used
        escalation_filter: Name of escalation filter if used
        report_dir: Directory to save report files
        render_images: Whether to render before/after images
        
    Returns:
        RepairReport with all data populated
    """
    report = RepairReport(
        input_file=str(input_path),
        output_file=str(output_path) if output_path else None,
        filter_script_name=filter_script_name,
        filter_script_actions=filter_script_actions,
        repair_duration_ms=duration_ms,
        success=success,
        error_message=error_message,
        escalation_used=escalation_used,
        escalation_filter=escalation_filter,
    )
    
    # Compute diagnostics
    report.original_diagnostics = compute_diagnostics(original_mesh)
    
    if repaired_mesh is not None and len(repaired_mesh.faces) > 0:
        report.repaired_diagnostics = compute_diagnostics(repaired_mesh)
    
    # Render images if requested and report_dir is provided
    if render_images and report_dir:
        report_dir.mkdir(parents=True, exist_ok=True)
        images_dir = report_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Render before image
        before_path = images_dir / "before.png"
        if render_mesh_image(original_mesh, before_path, mesh_color=(200, 100, 100, 255)):
            report.before_image = "images/before.png"
        
        # Render after image
        if repaired_mesh is not None and len(repaired_mesh.faces) > 0:
            after_path = images_dir / "after.png"
            if render_mesh_image(repaired_mesh, after_path, mesh_color=(100, 200, 100, 255)):
                report.after_image = "images/after.png"
    
    # Save model files if report_dir is provided
    if report_dir:
        report_dir.mkdir(parents=True, exist_ok=True)
        models_dir = report_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        # Save before (original) model
        try:
            before_model_path = models_dir / "before.stl"
            original_mesh.export(str(before_model_path), file_type="stl")
            report.before_model = "models/before.stl"
            logger.info(f"Saved original model to {before_model_path}")
        except Exception as e:
            logger.warning(f"Failed to save original model: {e}")
        
        # Save after (repaired) model
        if repaired_mesh is not None and len(repaired_mesh.faces) > 0:
            try:
                after_model_path = models_dir / "after.stl"
                repaired_mesh.export(str(after_model_path), file_type="stl")
                report.after_model = "models/after.stl"
                logger.info(f"Saved repaired model to {after_model_path}")
            except Exception as e:
                logger.warning(f"Failed to save repaired model: {e}")
    
    return report


def generate_report_index(
    report_dir: Path,
    title: str = "MeshPrep Repair Reports",
) -> None:
    """
    Generate an index.md file that lists all reports with navigation.
    
    Scans the report directory for all report.json files and creates
    a summary index with links to each report.
    
    Args:
        report_dir: Root directory containing all reports
        title: Title for the index page
    """
    report_dir = Path(report_dir)
    index_path = report_dir / "index.md"
    
    # Collect all reports
    reports_by_category: dict[str, list[dict]] = {}
    
    for json_file in report_dir.rglob("report.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            # Get relative path from report_dir to this report
            rel_path = json_file.parent.relative_to(report_dir)
            parts = rel_path.parts
            
            if len(parts) >= 2:
                category = parts[0]
                model_id = parts[1]
            elif len(parts) == 1:
                category = "uncategorized"
                model_id = parts[0]
            else:
                continue
            
            # Extract key info
            report_info = {
                "model_id": model_id,
                "path": str(rel_path).replace("\\", "/"),
                "success": data.get("success", False),
                "filter_script": data.get("filter_script", "unknown"),
                "duration_ms": data.get("duration_ms", 0),
                "escalation_used": data.get("escalation_used", False),
                "original": data.get("original_diagnostics", {}),
                "repaired": data.get("repaired_diagnostics", {}),
                "timestamp": data.get("timestamp", ""),
            }
            
            if category not in reports_by_category:
                reports_by_category[category] = []
            reports_by_category[category].append(report_info)
            
        except Exception as e:
            logger.warning(f"Failed to read {json_file}: {e}")
    
    # Sort categories and reports
    for category in reports_by_category:
        reports_by_category[category].sort(key=lambda x: x["model_id"])
    
    # Generate index markdown
    lines = []
    
    # Header
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().isoformat()}")
    lines.append("")
    
    # Summary statistics
    total_reports = sum(len(reports) for reports in reports_by_category.values())
    total_success = sum(
        sum(1 for r in reports if r["success"])
        for reports in reports_by_category.values()
    )
    total_escalations = sum(
        sum(1 for r in reports if r["escalation_used"])
        for reports in reports_by_category.values()
    )
    
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| **Total Reports** | {total_reports} |")
    lines.append(f"| **Successful** | {total_success} ({total_success/total_reports*100:.1f}%) |" if total_reports > 0 else "| **Successful** | 0 |")
    lines.append(f"| **Failed** | {total_reports - total_success} |")
    lines.append(f"| **Used Escalation** | {total_escalations} |")
    lines.append(f"| **Categories** | {len(reports_by_category)} |")
    lines.append("")
    
    # Quick navigation
    lines.append("## Categories")
    lines.append("")
    for category in sorted(reports_by_category.keys()):
        count = len(reports_by_category[category])
        success_count = sum(1 for r in reports_by_category[category] if r["success"])
        emoji = "✅" if success_count == count else ("⚠️" if success_count > 0 else "❌")
        lines.append(f"- [{emoji} {category}](#{category}) ({success_count}/{count} successful)")
    lines.append("")
    
    # Detailed reports by category
    for category in sorted(reports_by_category.keys()):
        reports = reports_by_category[category]
        success_count = sum(1 for r in reports if r["success"])
        
        lines.append(f"---")
        lines.append("")
        lines.append(f"## {category}")
        lines.append("")
        lines.append(f"**{success_count}/{len(reports)}** repairs successful")
        lines.append("")
        
        # Table header
        lines.append("| Status | Model | Filter | Duration | Watertight | Faces | Report |")
        lines.append("|--------|-------|--------|----------|------------|-------|--------|")
        
        for r in reports:
            status = "✅" if r["success"] else "❌"
            if r["escalation_used"]:
                status += " 🚀"  # Rocket for escalation
            
            model_id = r["model_id"]
            filter_name = r["filter_script"]
            duration = f"{r['duration_ms']:.0f}ms"
            
            # Watertight status
            orig_wt = r["original"].get("is_watertight", False)
            rep_wt = r["repaired"].get("is_watertight", False) if r["repaired"] else False
            wt_status = f"{"❌" if not orig_wt else "✅"} -> {"✅" if rep_wt else "❌"}"
            
            # Face count change
            orig_faces = r["original"].get("face_count", 0)
            rep_faces = r["repaired"].get("face_count", 0) if r["repaired"] else 0
            faces_str = f"{orig_faces:,} -> {rep_faces:,}"
            
            report_link = f"[View](./{r['path']}/report.md)"
            
            lines.append(f"| {status} | {model_id} | {filter_name} | {duration} | {wt_status} | {faces_str} | {report_link} |")
        
        lines.append("")
    
    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*Index generated by [MeshPrep](https://github.com/DragonAceNL/MeshPrep)*")
    lines.append("")
    
    # Write index file
    index_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report index saved to {index_path}")
    
    # Now update all individual reports to include back link
    for category, reports in reports_by_category.items():
        for r in reports:
            report_md_path = report_dir / r["path"] / "report.md"
            if report_md_path.exists():
                # Calculate relative path back to index
                depth = len(Path(r["path"]).parts)
                back_path = "../" * depth + "index.md"
                
                # Read and update report
                content = report_md_path.read_text(encoding="utf-8")
                
                # Check if it already has a back link
                if "[<- Back to Index]" not in content:
                    # Add navigation at the top
                    nav = f"[<- Back to Index]({back_path})\n\n---\n\n"
                    content = nav + content
                    report_md_path.write_text(content, encoding="utf-8")
                    logger.debug(f"Added navigation to {report_md_path}")
