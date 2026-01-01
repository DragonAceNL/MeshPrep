# Copyright 2025 Dragon Ace
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""Profile detection and matching for mesh analysis."""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from .diagnostics import Diagnostics


class ProfileCategory(Enum):
    """Categories of model profiles."""
    CLEAN = "Clean"
    HOLES = "Holes and Boundaries"
    FRAGMENTED = "Fragmented / Multi-Component"
    TOPOLOGY = "Topology Errors"
    NORMALS = "Normal Issues"
    SELF_INTERSECTION = "Self-Intersection"
    INTERNAL = "Internal Geometry / Hollow"
    THIN_FEATURES = "Thin Features / Wall Thickness"
    SCAN_NOISY = "Scan / Noisy Mesh"
    COMPLEX = "Complex Topology"
    SCALE = "Scale / Dimension Issues"
    PRINTABILITY = "Printability Hints"
    FINE_DETAIL = "Fine Detail / Precision"


@dataclass
class Profile:
    """A model profile with detection rules and suggested actions."""
    
    name: str
    display_name: str
    category: ProfileCategory
    description: str
    suggested_actions: list[str] = field(default_factory=list)
    
    def __hash__(self):
        return hash(self.name)


@dataclass
class ProfileMatch:
    """Result of profile detection."""
    
    profile: Profile
    confidence: float  # 0.0 - 1.0
    reasons: list[str] = field(default_factory=list)
    
    def explanation(self) -> str:
        """Generate explanation of why this profile was matched."""
        return f"{self.profile.display_name} ({self.confidence:.0%}): " + "; ".join(self.reasons)


# Define all profiles
PROFILES = {
    # Clean / Minimal Repair
    "clean": Profile(
        name="clean",
        display_name="Clean",
        category=ProfileCategory.CLEAN,
        description="Model is already printable or nearly printable.",
        suggested_actions=["trimesh_basic", "validate", "export_stl"],
    ),
    "clean-minor-issues": Profile(
        name="clean-minor-issues",
        display_name="Clean (Minor Issues)",
        category=ProfileCategory.CLEAN,
        description="Nearly clean model with minor fixable issues.",
        suggested_actions=["trimesh_basic", "remove_degenerate_faces", "recalculate_normals", "validate"],
    ),
    
    # Holes and Boundaries
    "holes-only": Profile(
        name="holes-only",
        display_name="Holes Only",
        category=ProfileCategory.HOLES,
        description="Single-component model with open holes.",
        suggested_actions=["trimesh_basic", "fill_holes", "recalculate_normals", "validate"],
    ),
    "open-bottom": Profile(
        name="open-bottom",
        display_name="Open Bottom",
        category=ProfileCategory.HOLES,
        description="Large opening on one face (flat base missing).",
        suggested_actions=["fill_holes", "recalculate_normals", "validate"],
    ),
    
    # Fragmented / Multi-Component
    "fragmented": Profile(
        name="fragmented",
        display_name="Fragmented",
        category=ProfileCategory.FRAGMENTED,
        description="Model contains many small disconnected components.",
        suggested_actions=["remove_small_components", "merge_vertices", "fill_holes", "validate"],
    ),
    "floating-components": Profile(
        name="floating-components",
        display_name="Floating Components",
        category=ProfileCategory.FRAGMENTED,
        description="Disconnected components positioned away from main part.",
        suggested_actions=["remove_small_components", "validate"],
    ),
    
    # Topology Errors
    "non-manifold": Profile(
        name="non-manifold",
        display_name="Non-Manifold",
        category=ProfileCategory.TOPOLOGY,
        description="Topology errors (non-manifold edges/vertices) present.",
        suggested_actions=["trimesh_basic", "remove_degenerate_faces", "pymeshfix_repair", "recalculate_normals", "validate"],
    ),
    "degenerate-heavy": Profile(
        name="degenerate-heavy",
        display_name="Degenerate Heavy",
        category=ProfileCategory.TOPOLOGY,
        description="Large number of degenerate (zero-area) faces.",
        suggested_actions=["remove_degenerate_faces", "merge_vertices", "fill_holes", "validate"],
    ),
    
    # Normal Issues
    "normals-inconsistent": Profile(
        name="normals-inconsistent",
        display_name="Inconsistent Normals",
        category=ProfileCategory.NORMALS,
        description="Face normals inconsistent or inverted.",
        suggested_actions=["fix_normals", "remove_degenerate_faces", "validate"],
    ),
    "inverted-normals": Profile(
        name="inverted-normals",
        display_name="Inverted Normals",
        category=ProfileCategory.NORMALS,
        description="All or most normals pointing inward.",
        suggested_actions=["flip_normals", "validate"],
    ),
    
    # Self-Intersection
    "self-intersecting": Profile(
        name="self-intersecting",
        display_name="Self-Intersecting",
        category=ProfileCategory.SELF_INTERSECTION,
        description="Mesh contains self-intersections or overlapping geometry.",
        suggested_actions=["pymeshfix_repair", "boolean_union", "validate"],
    ),
    
    # Internal Geometry / Hollow
    "hollow-porous": Profile(
        name="hollow-porous",
        display_name="Hollow / Porous",
        category=ProfileCategory.INTERNAL,
        description="Contains internal cavities, nested shells, or porous regions.",
        suggested_actions=["remove_internal_geometry", "fill_holes", "validate"],
    ),
    
    # Thin Features / Wall Thickness
    "thin-shell": Profile(
        name="thin-shell",
        display_name="Thin Shell",
        category=ProfileCategory.THIN_FEATURES,
        description="Thin walls or features that may not print reliably.",
        suggested_actions=["identify_thin_regions", "thicken_regions", "smooth_laplacian", "validate"],
    ),
    
    # Scan / Noisy Mesh
    "noisy-scan": Profile(
        name="noisy-scan",
        display_name="Noisy Scan",
        category=ProfileCategory.SCAN_NOISY,
        description="High-detail noisy mesh (3D scans) with many tiny defects.",
        suggested_actions=["decimate", "remove_degenerate_faces", "pymeshfix_repair", "smooth_laplacian", "validate"],
    ),
    "high-triangle-density": Profile(
        name="high-triangle-density",
        display_name="High Triangle Density",
        category=ProfileCategory.SCAN_NOISY,
        description="Extremely high triangle count relative to model size.",
        suggested_actions=["decimate", "validate"],
    ),
    
    # Complex Topology
    "complex-high-genus": Profile(
        name="complex-high-genus",
        display_name="Complex / High Genus",
        category=ProfileCategory.COMPLEX,
        description="High genus or complex topology requiring remeshing.",
        suggested_actions=["trimesh_basic", "pymeshfix_repair", "blender_remesh", "validate"],
    ),
    
    # Scale / Dimension Issues
    "small-part": Profile(
        name="small-part",
        display_name="Small Part",
        category=ProfileCategory.SCALE,
        description="Model is very small (may be in wrong units).",
        suggested_actions=["validate"],
    ),
    "oversized": Profile(
        name="oversized",
        display_name="Oversized",
        category=ProfileCategory.SCALE,
        description="Model exceeds target printer build volume.",
        suggested_actions=["validate"],
    ),
    
    # Printability Hints
    "overhang-heavy": Profile(
        name="overhang-heavy",
        display_name="Overhang Heavy",
        category=ProfileCategory.PRINTABILITY,
        description="Many faces with steep overhang angles.",
        suggested_actions=["validate"],
    ),
}


class ProfileDetector:
    """Detects model profiles from diagnostics."""
    
    def __init__(self, thresholds: Optional[dict] = None):
        """
        Initialize detector with optional custom thresholds.
        
        Args:
            thresholds: Custom detection thresholds.
        """
        self.thresholds = thresholds or self._default_thresholds()
    
    def _default_thresholds(self) -> dict:
        """Get default detection thresholds."""
        return {
            "normal_consistency_low": 0.8,
            "normal_consistency_very_low": 0.5,
            "degenerate_ratio_high": 0.01,
            "component_count_high": 5,
            "largest_component_low": 0.8,
            "triangle_density_high": 0.1,
            "min_thickness_low": 0.8,
            "genus_high": 5,
            "small_volume": 1.0,
            "large_volume": 1000000.0,
            "overhang_ratio_high": 0.3,
        }
    
    def detect(self, diagnostics: Diagnostics) -> list[ProfileMatch]:
        """
        Detect matching profiles for a mesh.
        
        Args:
            diagnostics: Computed diagnostics for the mesh.
            
        Returns:
            List of ProfileMatch objects sorted by confidence (highest first).
        """
        matches = []
        
        # Check for clean mesh
        if self._is_clean(diagnostics):
            matches.append(ProfileMatch(
                profile=PROFILES["clean"],
                confidence=0.95,
                reasons=["Watertight", "No topology errors", "Single component"],
            ))
        elif self._is_nearly_clean(diagnostics):
            matches.append(ProfileMatch(
                profile=PROFILES["clean-minor-issues"],
                confidence=0.85,
                reasons=["Mostly clean", "Minor issues detected"],
            ))
        
        # Check for holes
        if diagnostics.hole_count > 0 and diagnostics.component_count == 1:
            confidence = min(0.9, 0.5 + diagnostics.hole_count * 0.05)
            matches.append(ProfileMatch(
                profile=PROFILES["holes-only"],
                confidence=confidence,
                reasons=[f"{diagnostics.hole_count} holes detected", "Single component"],
            ))
        
        # Check for fragmented
        if diagnostics.component_count > self.thresholds["component_count_high"]:
            confidence = min(0.95, 0.5 + diagnostics.component_count * 0.05)
            matches.append(ProfileMatch(
                profile=PROFILES["fragmented"],
                confidence=confidence,
                reasons=[f"{diagnostics.component_count} components", 
                        f"Largest is {diagnostics.largest_component_pct:.1%}"],
            ))
        
        # Check for non-manifold
        if diagnostics.non_manifold_edge_count > 0 or diagnostics.non_manifold_vertex_count > 0:
            total_nm = diagnostics.non_manifold_edge_count + diagnostics.non_manifold_vertex_count
            confidence = min(0.95, 0.6 + total_nm * 0.02)
            matches.append(ProfileMatch(
                profile=PROFILES["non-manifold"],
                confidence=confidence,
                reasons=[f"{diagnostics.non_manifold_edge_count} non-manifold edges",
                        f"{diagnostics.non_manifold_vertex_count} non-manifold vertices"],
            ))
        
        # Check for degenerate faces
        if diagnostics.face_count > 0:
            degen_ratio = diagnostics.degenerate_face_count / diagnostics.face_count
            if degen_ratio > self.thresholds["degenerate_ratio_high"]:
                matches.append(ProfileMatch(
                    profile=PROFILES["degenerate-heavy"],
                    confidence=min(0.9, 0.5 + degen_ratio * 10),
                    reasons=[f"{diagnostics.degenerate_face_count} degenerate faces ({degen_ratio:.1%})"],
                ))
        
        # Check for normal issues
        if diagnostics.normal_consistency < self.thresholds["normal_consistency_very_low"]:
            matches.append(ProfileMatch(
                profile=PROFILES["inverted-normals"],
                confidence=0.85,
                reasons=[f"Normal consistency: {diagnostics.normal_consistency:.1%}"],
            ))
        elif diagnostics.normal_consistency < self.thresholds["normal_consistency_low"]:
            matches.append(ProfileMatch(
                profile=PROFILES["normals-inconsistent"],
                confidence=0.8,
                reasons=[f"Normal consistency: {diagnostics.normal_consistency:.1%}"],
            ))
        
        # Check for self-intersections
        if diagnostics.self_intersections:
            confidence = min(0.95, 0.6 + diagnostics.self_intersection_count * 0.01)
            matches.append(ProfileMatch(
                profile=PROFILES["self-intersecting"],
                confidence=confidence,
                reasons=[f"{diagnostics.self_intersection_count} self-intersections"],
            ))
        
        # Check for thin features
        if diagnostics.estimated_min_thickness < self.thresholds["min_thickness_low"]:
            matches.append(ProfileMatch(
                profile=PROFILES["thin-shell"],
                confidence=0.8,
                reasons=[f"Min thickness: {diagnostics.estimated_min_thickness:.2f}mm"],
            ))
        
        # Check for high triangle density (noisy scan)
        if diagnostics.triangle_density > self.thresholds["triangle_density_high"]:
            matches.append(ProfileMatch(
                profile=PROFILES["noisy-scan"],
                confidence=0.75,
                reasons=[f"High triangle density: {diagnostics.triangle_density:.3f}"],
            ))
        
        # Check for high genus
        if diagnostics.genus > self.thresholds["genus_high"]:
            matches.append(ProfileMatch(
                profile=PROFILES["complex-high-genus"],
                confidence=0.8,
                reasons=[f"Genus: {diagnostics.genus}"],
            ))
        
        # Check for nested shells
        if diagnostics.nested_shell_count > 0:
            matches.append(ProfileMatch(
                profile=PROFILES["hollow-porous"],
                confidence=0.85,
                reasons=[f"{diagnostics.nested_shell_count} nested shells"],
            ))
        
        # Check for scale issues
        if diagnostics.bbox_volume < self.thresholds["small_volume"]:
            matches.append(ProfileMatch(
                profile=PROFILES["small-part"],
                confidence=0.7,
                reasons=[f"Very small: {diagnostics.bbox_volume:.3f} mm³"],
            ))
        elif diagnostics.bbox_volume > self.thresholds["large_volume"]:
            matches.append(ProfileMatch(
                profile=PROFILES["oversized"],
                confidence=0.7,
                reasons=[f"Very large: {diagnostics.bbox_volume:.0f} mm³"],
            ))
        
        # Sort by confidence
        matches.sort(key=lambda m: m.confidence, reverse=True)
        
        # If no matches, default to clean-minor-issues
        if not matches:
            matches.append(ProfileMatch(
                profile=PROFILES["clean-minor-issues"],
                confidence=0.5,
                reasons=["No specific issues detected"],
            ))
        
        return matches
    
    def _is_clean(self, d: Diagnostics) -> bool:
        """Check if mesh is essentially clean."""
        return (
            d.is_watertight
            and d.non_manifold_edge_count == 0
            and d.non_manifold_vertex_count == 0
            and d.degenerate_face_count == 0
            and d.component_count == 1
            and not d.self_intersections
            and d.normal_consistency >= 0.95
        )
    
    def _is_nearly_clean(self, d: Diagnostics) -> bool:
        """Check if mesh is nearly clean with minor issues."""
        issues = 0
        if d.degenerate_face_count > 0 and d.degenerate_face_count < 10:
            issues += 1
        if d.normal_consistency < 0.95 and d.normal_consistency >= 0.8:
            issues += 1
        if d.duplicate_vertex_ratio > 0 and d.duplicate_vertex_ratio < 0.05:
            issues += 1
        
        return (
            d.is_watertight
            and d.non_manifold_edge_count == 0
            and d.component_count == 1
            and not d.self_intersections
            and issues <= 2
        )
    
    def get_profile(self, name: str) -> Optional[Profile]:
        """Get a profile by name."""
        return PROFILES.get(name)
    
    def list_profiles(self) -> list[Profile]:
        """Get all available profiles."""
        return list(PROFILES.values())
