# Copyright 2025 Dragon Ace
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""Tests for diagnostics and profile detection."""

import pytest

from meshprep.core.mock_mesh import MockMesh
from meshprep.core.diagnostics import Diagnostics, compute_diagnostics
from meshprep.core.profiles import ProfileDetector, Profile, PROFILES


class TestDiagnostics:
    """Tests for Diagnostics class."""
    
    def test_compute_diagnostics(self):
        """Test computing diagnostics from mesh."""
        mesh = MockMesh(
            vertex_count=1000,
            face_count=2000,
            is_watertight=True,
            hole_count=0,
        )
        
        diag = compute_diagnostics(mesh)
        
        assert diag.vertex_count == 1000
        assert diag.face_count == 2000
        assert diag.is_watertight
        assert diag.hole_count == 0
    
    def test_diagnostics_to_dict(self):
        """Test converting diagnostics to dictionary."""
        diag = Diagnostics(
            vertex_count=500,
            face_count=1000,
            is_watertight=False,
        )
        
        d = diag.to_dict()
        
        assert d["vertex_count"] == 500
        assert d["face_count"] == 1000
        assert d["is_watertight"] is False
    
    def test_diagnostics_from_dict(self):
        """Test creating diagnostics from dictionary."""
        data = {
            "vertex_count": 500,
            "face_count": 1000,
            "is_watertight": True,
        }
        
        diag = Diagnostics.from_dict(data)
        
        assert diag.vertex_count == 500
        assert diag.is_watertight
    
    def test_is_printable(self):
        """Test printability check."""
        # Printable mesh
        printable = Diagnostics(
            is_watertight=True,
            non_manifold_edge_count=0,
            non_manifold_vertex_count=0,
            component_count=1,
            self_intersections=False,
        )
        assert printable.is_printable()
        
        # Non-printable mesh
        non_printable = Diagnostics(
            is_watertight=False,
            hole_count=5,
        )
        assert not non_printable.is_printable()
    
    def test_issues(self):
        """Test issue detection."""
        diag = Diagnostics(
            is_watertight=False,
            hole_count=3,
            non_manifold_edge_count=5,
            normal_consistency=0.5,
        )
        
        issues = diag.issues()
        
        assert len(issues) >= 3
        assert any("watertight" in i.lower() for i in issues)
        assert any("manifold" in i.lower() for i in issues)
        assert any("normal" in i.lower() for i in issues)


class TestProfileDetector:
    """Tests for ProfileDetector class."""
    
    def test_detect_clean_profile(self):
        """Test detecting clean profile."""
        diag = Diagnostics(
            is_watertight=True,
            hole_count=0,
            component_count=1,
            non_manifold_edge_count=0,
            non_manifold_vertex_count=0,
            degenerate_face_count=0,
            normal_consistency=1.0,
            self_intersections=False,
        )
        
        detector = ProfileDetector()
        matches = detector.detect(diag)
        
        assert len(matches) > 0
        assert matches[0].profile.name == "clean"
        assert matches[0].confidence > 0.9
    
    def test_detect_holes_only_profile(self):
        """Test detecting holes-only profile."""
        diag = Diagnostics(
            is_watertight=False,
            hole_count=5,
            component_count=1,
            non_manifold_edge_count=0,
        )
        
        detector = ProfileDetector()
        matches = detector.detect(diag)
        
        # Should include holes-only profile
        profile_names = [m.profile.name for m in matches]
        assert "holes-only" in profile_names
    
    def test_detect_non_manifold_profile(self):
        """Test detecting non-manifold profile."""
        diag = Diagnostics(
            non_manifold_edge_count=10,
            non_manifold_vertex_count=5,
        )
        
        detector = ProfileDetector()
        matches = detector.detect(diag)
        
        profile_names = [m.profile.name for m in matches]
        assert "non-manifold" in profile_names
    
    def test_detect_fragmented_profile(self):
        """Test detecting fragmented profile."""
        diag = Diagnostics(
            component_count=15,
            largest_component_pct=0.4,
        )
        
        detector = ProfileDetector()
        matches = detector.detect(diag)
        
        profile_names = [m.profile.name for m in matches]
        assert "fragmented" in profile_names
    
    def test_matches_sorted_by_confidence(self):
        """Test that matches are sorted by confidence."""
        diag = Diagnostics(
            is_watertight=False,
            hole_count=3,
            non_manifold_edge_count=2,
        )
        
        detector = ProfileDetector()
        matches = detector.detect(diag)
        
        assert len(matches) >= 2
        for i in range(len(matches) - 1):
            assert matches[i].confidence >= matches[i + 1].confidence
    
    def test_get_profile(self):
        """Test getting profile by name."""
        detector = ProfileDetector()
        
        profile = detector.get_profile("holes-only")
        assert profile is not None
        assert profile.name == "holes-only"
        
        unknown = detector.get_profile("nonexistent")
        assert unknown is None
    
    def test_list_profiles(self):
        """Test listing all profiles."""
        detector = ProfileDetector()
        profiles = detector.list_profiles()
        
        assert len(profiles) > 10
        assert all(isinstance(p, Profile) for p in profiles)
