# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Shared HTML generation helpers for MeshPrep reports.

Contains common CSS styles, color constants, and utility functions
used across all HTML report generators.
"""

from datetime import timedelta
from typing import Optional


# =============================================================================
# Color Constants (Dark Theme)
# =============================================================================

COLORS = {
    "background": "#0f1720",
    "background_secondary": "#1b2b33",
    "background_tertiary": "#2a3a43",
    "text_primary": "#dff6fb",
    "text_secondary": "#888",
    "text_muted": "#555",
    "accent": "#4fe8c4",
    "accent_hover": "#3dd4b0",
    "success": "#2ecc71",
    "warning": "#f39c12",
    "danger": "#e74c3c",
    "info": "#3498db",
    "purple": "#9b59b6",
}


# =============================================================================
# Common CSS Styles
# =============================================================================

def get_base_styles() -> str:
    """Return base CSS styles used across all pages."""
    return f"""
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: {COLORS['background']};
            color: {COLORS['text_primary']};
            margin: 0;
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: {COLORS['accent']}; margin-bottom: 10px; }}
        h2 {{ color: {COLORS['accent']}; }}
        .subtitle {{ color: {COLORS['text_secondary']}; margin-bottom: 20px; }}
        
        a {{ color: {COLORS['accent']}; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        
        .success {{ color: {COLORS['success']}; }}
        .failed {{ color: {COLORS['danger']}; }}
        .warning {{ color: {COLORS['warning']}; }}
        .skipped {{ color: {COLORS['info']}; }}
        .escalated {{ color: {COLORS['warning']}; }}
"""


def get_stats_grid_styles() -> str:
    """Return CSS for stats grid layout."""
    return f"""
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: {COLORS['background_secondary']};
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 36px;
            font-weight: bold;
            color: {COLORS['accent']};
        }}
        .stat-label {{ color: {COLORS['text_secondary']}; margin-top: 5px; }}
"""


def get_stats_row_styles() -> str:
    """Return CSS for horizontal stats row layout."""
    return f"""
        .stats-row {{
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .stat {{
            background: {COLORS['background_secondary']};
            padding: 10px 20px;
            border-radius: 8px;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: {COLORS['accent']};
        }}
        .stat-label {{ color: {COLORS['text_secondary']}; margin-top: 5px; }}
"""


def get_table_styles() -> str:
    """Return CSS for tables."""
    return f"""
        table {{
            width: 100%;
            border-collapse: collapse;
            background: {COLORS['background_secondary']};
            border-radius: 12px;
            overflow: hidden;
        }}
        th, td {{
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid {COLORS['background_tertiary']};
        }}
        th {{
            background: {COLORS['background']};
            color: {COLORS['accent']};
            cursor: pointer;
            user-select: none;
            position: sticky;
            top: 0;
        }}
        th:hover {{ background: #1a2a35; }}
        tr:hover {{ background: {COLORS['background_tertiary']}; }}
"""


def get_nav_bar_styles() -> str:
    """Return CSS for navigation bar."""
    return f"""
        .nav-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: {COLORS['background_secondary']};
            padding: 10px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .nav-bar a {{ color: {COLORS['accent']}; text-decoration: none; padding: 5px 10px; }}
        .nav-bar a:hover {{ background: {COLORS['background_tertiary']}; border-radius: 4px; }}
        .nav-bar .disabled {{ color: {COLORS['text_muted']}; }}
"""


def get_nav_links_styles() -> str:
    """Return CSS for navigation links section."""
    return f"""
        .nav-links {{
            margin-bottom: 20px;
        }}
        .nav-links a {{
            color: {COLORS['accent']};
            text-decoration: none;
            padding: 8px 16px;
            background: {COLORS['background_secondary']};
            border-radius: 6px;
            margin-right: 10px;
        }}
        .nav-links a:hover {{
            background: {COLORS['background_tertiary']};
        }}
"""


def get_progress_bar_styles() -> str:
    """Return CSS for progress bars."""
    return f"""
        .progress-bar {{
            background: {COLORS['background_secondary']};
            border-radius: 10px;
            height: 30px;
            overflow: hidden;
            margin-bottom: 30px;
        }}
        .progress-fill {{
            background: linear-gradient(90deg, {COLORS['accent']}, {COLORS['success']});
            height: 100%;
            transition: width 0.5s;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }}
"""


def get_spinner_styles() -> str:
    """Return CSS for loading spinner animation."""
    return f"""
        .spinner {{
            width: 20px;
            height: 20px;
            border: 3px solid #333;
            border-top-color: {COLORS['accent']};
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
"""


def get_button_styles() -> str:
    """Return CSS for buttons."""
    return f"""
        .download-btn {{
            display: inline-block;
            background: {COLORS['accent']};
            color: {COLORS['background']};
            padding: 12px 24px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: bold;
            border: none;
            cursor: pointer;
        }}
        .download-btn:hover {{ background: {COLORS['accent_hover']}; }}
        .download-btn.secondary {{ background: {COLORS['background_secondary']}; color: {COLORS['accent']}; }}
        .no-file {{ color: {COLORS['text_secondary']}; padding: 12px 24px; }}
        
        .meshlab-btn {{
            background: {COLORS['purple']} !important;
            color: white !important;
        }}
        .meshlab-btn:hover {{ background: #8e44ad !important; }}
"""


def get_filter_styles() -> str:
    """Return CSS for filter controls."""
    return f"""
        .filters {{
            background: {COLORS['background_secondary']};
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
        }}
        .filters label {{ color: {COLORS['text_secondary']}; }}
        .filters select, .filters input {{
            background: {COLORS['background']};
            color: {COLORS['text_primary']};
            border: 1px solid #333;
            padding: 8px 12px;
            border-radius: 4px;
        }}
"""


def get_status_badge_styles() -> str:
    """Return CSS for status badges."""
    return f"""
        .status-badge {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            margin-bottom: 20px;
        }}
        .status-badge.success {{ background: {COLORS['success']}; color: {COLORS['background']}; }}
        .status-badge.failed {{ background: {COLORS['danger']}; color: white; }}
        .status-badge.skipped {{ background: {COLORS['info']}; color: white; }}
        .status-badge.escalated {{ background: {COLORS['warning']}; color: {COLORS['background']}; }}
"""


def get_error_box_styles() -> str:
    """Return CSS for error message boxes."""
    return f"""
        .error-box {{
            background: #2a1a1a;
            border: 1px solid {COLORS['danger']};
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 30px;
        }}
        .error-box h3 {{ color: {COLORS['danger']}; margin-top: 0; }}
        .error-box pre {{ margin: 0; white-space: pre-wrap; }}
"""


# =============================================================================
# Utility Functions
# =============================================================================

def format_duration(duration_ms: float) -> str:
    """Format duration in milliseconds to human-readable string."""
    duration_sec = duration_ms / 1000
    if duration_sec >= 60:
        return f"{duration_sec/60:.1f}m"
    return f"{duration_sec:.1f}s"


def format_eta(eta_seconds: float) -> str:
    """Format ETA seconds to human-readable string."""
    if eta_seconds <= 0:
        return "Calculating..."
    return str(timedelta(seconds=int(eta_seconds)))


def format_number(value: int) -> str:
    """Format number with thousand separators."""
    return f"{value:,}"


def format_percent(value: float, decimals: int = 1) -> str:
    """Format percentage value."""
    return f"{value:.{decimals}f}%"


def get_status_info(success: bool, precheck_skipped: bool, escalation_used: bool) -> tuple[str, str]:
    """Get status class and text for a result.
    
    Returns:
        Tuple of (css_class, display_text)
    """
    if precheck_skipped:
        return "skipped", "&#10003; Already Clean"
    elif success:
        if escalation_used:
            return "escalated", "&#10003; Fixed (Blender)"
        else:
            return "success", "&#10003; Fixed"
    else:
        return "failed", "&#10007; Failed"


def escape_html(text: str) -> str:
    """Escape special HTML characters."""
    return (text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;"))


def get_html_document_start(title: str, extra_styles: str = "", extra_scripts: str = "") -> str:
    """Generate HTML document header with common styles.
    
    Args:
        title: Page title
        extra_styles: Additional CSS to include
        extra_scripts: Additional JavaScript to include
    """
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
{get_base_styles()}
{extra_styles}
    </style>
{extra_scripts}
</head>
<body>
    <div class="container">
"""


def get_html_document_end() -> str:
    """Generate HTML document footer."""
    return """    </div>
</body>
</html>
"""
