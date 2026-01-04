# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Dashboard HTML generator for MeshPrep batch processing.

Generates the main dashboard showing progress, statistics,
and recent results during batch processing.
"""

import logging
from datetime import timedelta
from pathlib import Path
from typing import List, TYPE_CHECKING

from html_helpers import (
    COLORS,
    get_base_styles,
    get_stats_grid_styles,
    get_nav_links_styles,
    get_progress_bar_styles,
    get_spinner_styles,
    get_table_styles,
    format_duration,
    format_eta,
)

if TYPE_CHECKING:
    from run_full_test import Progress, TestResult

logger = logging.getLogger(__name__)


def _get_dashboard_styles() -> str:
    """Get CSS styles specific to dashboard."""
    return f"""
        .current-file {{
            background: {COLORS['background_secondary']};
            border-radius: 12px;
            padding: 15px 20px;
            margin-bottom: 30px;
            display: flex;
            align-items: center;
            gap: 15px;
        }}
"""


def generate_dashboard(
    progress: "Progress",
    results: List["TestResult"],
    output_path: Path,
) -> None:
    """Generate HTML dashboard for batch processing overview.
    
    Args:
        progress: Current progress tracking object
        results: List of all test results
        output_path: Path to write dashboard HTML file
    """
    # Calculate statistics
    avg_duration = sum(r.duration_ms for r in results) / len(results) if results else 0
    
    # Recent results (last 20)
    recent = results[-20:] if len(results) > 20 else results
    
    # Build styles
    all_styles = (
        get_base_styles() +
        get_stats_grid_styles() +
        get_nav_links_styles() +
        get_progress_bar_styles() +
        get_spinner_styles() +
        get_table_styles() +
        _get_dashboard_styles()
    )
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>MeshPrep Thingi10K Test Dashboard</title>
    <meta http-equiv="refresh" content="30">
    <style>
{all_styles}
    </style>
</head>
<body>
    <div class="container">
        <h1>?? MeshPrep Thingi10K Test Dashboard</h1>
        <p class="subtitle">
            Started: {progress.start_time[:19] if progress.start_time else 'Not started'} | 
            Last Update: {progress.last_update[:19] if progress.last_update else 'Never'}
            <span class="eta">| ETA: {format_eta(progress.eta_seconds)}</span>
        </p>
        
        <div class="nav-links">
            <a href="/reports/index.html">?? Reports Index</a>
            <a href="/learning">?? Learning Status</a>
            <a href="/live">?? Live Dashboard</a>
        </div>
        
        <div class="progress-bar">
            <div class="progress-fill" style="width: {progress.percent_complete:.1f}%">
                {progress.percent_complete:.1f}% ({progress.processed:,} / {progress.total_files:,})
            </div>
        </div>
        
        <div class="current-file">
            <div class="spinner"></div>
            <div style="flex: 1;">
                <div>Currently processing: <strong>{progress.current_file or 'Waiting...'}</strong></div>
                <div style="margin-top: 5px; font-size: 14px; color: {COLORS['text_secondary']};">
                    Action: <span style="color: {COLORS['accent']};">{progress.current_action or '-'}</span>
                    {f'(step {progress.current_step} of {progress.total_steps})' if progress.current_step else ''}
                </div>
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{progress.total_files:,}</div>
                <div class="stat-label">Total Models</div>
            </div>
            <div class="stat-card">
                <div class="stat-value success">{progress.successful:,}</div>
                <div class="stat-label">Successful</div>
            </div>
            <div class="stat-card">
                <div class="stat-value failed">{progress.failed:,}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value escalated">{progress.escalations:,}</div>
                <div class="stat-label">Escalations</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: {COLORS['info']};">{progress.precheck_skipped:,}</div>
                <div class="stat-label">Already Clean</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: {COLORS['purple']};">{progress.reconstructed:,}</div>
                <div class="stat-label">Reconstructed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{progress.success_rate:.1f}%</div>
                <div class="stat-label">Success Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg_duration/1000:.1f}s</div>
                <div class="stat-label">Avg Duration</div>
            </div>
        </div>
        
        <h2>Recent Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Status</th>
                    <th>Filter</th>
                    <th>Duration</th>
                    <th>Faces Before</th>
                    <th>Faces After</th>
                    <th>Report</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for r in reversed(recent):
        status_class = "success" if r.success else "failed"
        status_text = "?" if r.success else "?"
        if r.escalation_used:
            status_text += " ??"
        
        duration_text = format_duration(r.duration_ms)
        
        html += f"""                <tr>
                    <td><strong>{r.file_id}</strong></td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{r.filter_used}</td>
                    <td>{duration_text}</td>
                    <td>{r.original_faces:,}</td>
                    <td>{r.result_faces:,}</td>
                    <td><a href="/reports/{r.file_id}.html">View</a></td>
                </tr>
"""
    
    html += f"""            </tbody>
        </table>
        
        <p style="margin-top: 30px; color: {COLORS['text_muted']}; font-size: 12px;">
            Dashboard auto-refreshes every 30 seconds. 
            <a href="javascript:location.reload()">Refresh now</a>
        </p>
    </div>
</body>
</html>
"""
    
    with open(output_path, "w", encoding='utf-8') as f:
        f.write(html)
    
    logger.info(f"Generated dashboard: {output_path}")
