# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Individual model report HTML generator.

Generates detailed HTML reports for each processed mesh model,
including before/after comparison, metrics, and download links.
"""

import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from html_helpers import (
    COLORS,
    get_base_styles,
    get_nav_bar_styles,
    get_status_badge_styles,
    get_button_styles,
    get_error_box_styles,
    get_status_info,
    format_duration,
)

if TYPE_CHECKING:
    from test_result import TestResult

logger = logging.getLogger(__name__)


def _get_report_styles() -> str:
    """Get CSS styles specific to model reports."""
    return f"""
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: {COLORS['accent']}; margin-bottom: 5px; }}
        
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .info-card {{
            background: {COLORS['background_secondary']};
            padding: 15px;
            border-radius: 8px;
        }}
        .info-card .label {{ color: {COLORS['text_secondary']}; font-size: 12px; margin-bottom: 5px; }}
        .info-card .value {{ font-size: 18px; font-weight: bold; }}
        
        .comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }}
        .comparison-panel {{
            background: {COLORS['background_secondary']};
            border-radius: 12px;
            overflow: hidden;
        }}
        .comparison-panel h3 {{
            margin: 0;
            padding: 15px;
            background: {COLORS['background']};
            color: {COLORS['accent']};
        }}
        .comparison-panel img {{
            width: 100%;
            height: 400px;
            object-fit: contain;
            background: #0a0f14;
        }}
        
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            background: {COLORS['background_secondary']};
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 30px;
        }}
        .metrics-table th, .metrics-table td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid {COLORS['background_tertiary']};
        }}
        .metrics-table th {{ background: {COLORS['background']}; color: {COLORS['accent']}; }}
        
        .change-positive {{ color: {COLORS['danger']}; }}
        .change-negative {{ color: {COLORS['success']}; }}
        .change-neutral {{ color: {COLORS['text_secondary']}; }}
        
        .downloads {{
            display: flex;
            gap: 15px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }}
        
        .footer {{ color: {COLORS['text_muted']}; font-size: 12px; margin-top: 30px; }}
        
        .meshlab-status {{
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 12px;
            margin-left: 10px;
        }}
        .meshlab-available {{ background: {COLORS['success']}; color: white; }}
        .meshlab-unavailable {{ background: {COLORS['danger']}; color: white; }}
        
        /* Fingerprint box for filter script discovery */
        .fingerprint-box {{
            background: linear-gradient(135deg, {COLORS['background_secondary']} 0%, {COLORS['background_tertiary']} 100%);
            border: 2px solid {COLORS['accent']};
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 25px;
            text-align: center;
        }}
        .fingerprint-label {{
            color: {COLORS['text_secondary']};
            font-size: 14px;
            margin-bottom: 10px;
        }}
        .fingerprint-value {{
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 28px;
            font-weight: bold;
            color: {COLORS['accent']};
            background: {COLORS['background']};
            padding: 15px 25px;
            border-radius: 8px;
            display: inline-block;
            cursor: pointer;
            user-select: all;
            transition: all 0.2s;
        }}
        .fingerprint-value:hover {{
            background: #1a2530;
            transform: scale(1.02);
        }}
        .fingerprint-help {{
            margin-top: 12px;
            font-size: 13px;
            color: #666;
        }}
        .fingerprint-help a {{
            color: {COLORS['accent']};
            text-decoration: none;
            padding: 4px 8px;
            border-radius: 4px;
            background: {COLORS['background_secondary']};
        }}
        .fingerprint-help a:hover {{
            background: {COLORS['background_tertiary']};
        }}
        .copy-hint {{
            color: {COLORS['text_muted']};
            font-style: italic;
        }}
        
        /* Rating box styles */
        .rating-box {{
            background: {COLORS['background_secondary']};
            border: 2px solid {COLORS['background_tertiary']};
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 25px;
        }}
        .rating-box h3 {{
            margin: 0 0 15px 0;
            color: {COLORS['accent']};
            font-size: 18px;
        }}
        .rating-stars {{
            display: flex;
            gap: 8px;
            margin-bottom: 15px;
        }}
        .rating-star {{
            font-size: 32px;
            cursor: pointer;
            color: {COLORS['background_tertiary']};
            transition: all 0.2s;
            background: none;
            border: none;
            padding: 5px;
        }}
        .rating-star:hover,
        .rating-star.hovered {{
            color: #f1c40f;
            transform: scale(1.2);
        }}
        .rating-star.selected {{
            color: #f1c40f;
        }}
        .rating-star.selected.locked {{
            cursor: default;
        }}
        .rating-comment {{
            width: 100%;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid {COLORS['background_tertiary']};
            background: {COLORS['background']};
            color: {COLORS['text_primary']};
            font-size: 14px;
            resize: vertical;
            min-height: 60px;
            margin-bottom: 15px;
        }}
        .rating-comment:focus {{
            outline: none;
            border-color: {COLORS['accent']};
        }}
        .rating-submit {{
            background: {COLORS['accent']};
            color: white;
            border: none;
            padding: 10px 25px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            transition: all 0.2s;
        }}
        .rating-submit:hover {{
            background: #2980b9;
            transform: translateY(-2px);
        }}
        .rating-submit:disabled {{
            background: {COLORS['background_tertiary']};
            cursor: not-allowed;
            transform: none;
        }}
        .rating-status {{
            margin-top: 15px;
            padding: 10px;
            border-radius: 8px;
            font-size: 14px;
        }}
        .rating-status.success {{
            background: rgba(39, 174, 96, 0.2);
            color: {COLORS['success']};
        }}
        .rating-status.error {{
            background: rgba(231, 76, 60, 0.2);
            color: {COLORS['danger']};
        }}
        .rating-status.info {{
            background: rgba(52, 152, 219, 0.2);
            color: {COLORS['info']};
        }}
        .existing-rating {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 15px;
            background: rgba(52, 152, 219, 0.1);
            border-radius: 8px;
            margin-bottom: 15px;
        }}
        .existing-rating .stars {{
            color: #f1c40f;
            font-size: 20px;
        }}
        .existing-rating .details {{
            color: {COLORS['text_secondary']};
            font-size: 13px;
        }}
        .rating-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            color: {COLORS['text_muted']};
            margin-bottom: 10px;
            padding: 0 5px;
        }}
"""


def _get_report_scripts(fingerprint: str, file_id: str) -> str:
    """Get JavaScript for model report page."""
    return f"""
    <script>
        let selectedRating = 0;
        let isRatingLocked = false;
        
        async function openInMeshLab(filePath) {{
            const btn = event.target;
            const originalText = btn.innerHTML;
            btn.innerHTML = '&#8987; Opening...';
            btn.disabled = true;
            
            try {{
                const url = '/api/open-meshlab?file=' + encodeURIComponent(filePath);
                console.log('Requesting:', url);
                
                const response = await fetch(url);
                const data = await response.json();
                console.log('Response:', data);
                
                if (!response.ok || !data.success) {{
                    alert('Failed to open in MeshLab:\\n\\n' + (data.error || 'Unknown error'));
                }} else {{
                    btn.innerHTML = '&#10003; Opened!';
                    setTimeout(() => {{ btn.innerHTML = originalText; btn.disabled = false; }}, 2000);
                    return;
                }}
            }} catch (e) {{
                console.error('Error:', e);
                alert('Error: ' + e.message + '\\n\\nMake sure you are using the MeshPrep reports server:\\n\\ncd poc/v3\\npython reports_server.py');
            }}
            
            btn.innerHTML = originalText;
            btn.disabled = false;
        }}
        
        async function checkMeshLab() {{
            try {{
                const response = await fetch('/api/meshlab-status');
                const data = await response.json();
                const indicator = document.getElementById('meshlab-indicator');
                if (indicator) {{
                    if (data.available) {{
                        indicator.className = 'meshlab-status meshlab-available';
                        indicator.textContent = '\\u2713 MeshLab Ready';
                        indicator.title = 'MeshLab: ' + data.path;
                    }} else {{
                        indicator.className = 'meshlab-status meshlab-unavailable';
                        indicator.textContent = '\\u2717 MeshLab Not Found';
                    }}
                }}
            }} catch (e) {{
                const indicator = document.getElementById('meshlab-indicator');
                if (indicator) {{
                    indicator.className = 'meshlab-status meshlab-unavailable';
                    indicator.textContent = 'Use reports_server.py';
                    indicator.title = 'Run: python reports_server.py';
                }}
            }}
        }}
        
        function copyFingerprint() {{
            const fingerprint = '{fingerprint}';
            navigator.clipboard.writeText(fingerprint).then(() => {{
                const el = document.querySelector('.fingerprint-value');
                const original = el.textContent;
                el.textContent = '\\u2713 Copied!';
                el.style.background = '#27ae60';
                setTimeout(() => {{
                    el.textContent = original;
                    el.style.background = '#0f1720';
                }}, 1500);
            }}.catch(err => {{
                const el = document.querySelector('.fingerprint-value');
                const range = document.createRange();
                range.selectNodeContents(el);
                window.getSelection().removeAllRanges();
                window.getSelection().addRange(range);
                document.execCommand('copy');
                window.getSelection().removeAllRanges();
            }});
        }}
        
        // Rating functions
        function hoverStar(rating) {{
            if (isRatingLocked) return;
            const stars = document.querySelectorAll('.rating-star');
            stars.forEach((star, idx) => {{
                star.classList.toggle('hovered', idx < rating);
            }});
        }}
        
        function unhoverStars() {{
            if (isRatingLocked) return;
            const stars = document.querySelectorAll('.rating-star');
            stars.forEach((star, idx) => {{
                star.classList.remove('hovered');
                star.classList.toggle('selected', idx < selectedRating);
            }});
        }}
        
        function selectStar(rating) {{
            if (isRatingLocked) return;
            selectedRating = rating;
            const stars = document.querySelectorAll('.rating-star');
            stars.forEach((star, idx) => {{
                star.classList.toggle('selected', idx < rating);
            }});
            document.getElementById('submit-rating').disabled = false;
        }}
        
        async function submitRating() {{
            if (selectedRating === 0) {{
                showRatingStatus('Please select a rating (1-5 stars)', 'error');
                return;
            }}
            
            const comment = document.getElementById('rating-comment').value;
            const fingerprint = '{fingerprint}';
            const fileId = '{file_id}';
            
            const btn = document.getElementById('submit-rating');
            btn.disabled = true;
            btn.textContent = 'Submitting...';
            
            try {{
                const response = await fetch('/api/rate-model', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        fingerprint: fingerprint,
                        file_id: fileId,
                        rating: selectedRating,
                        comment: comment
                    }})
                }});
                
                const data = await response.json();
                
                if (data.success) {{
                    showRatingStatus('\\u2713 Rating saved! Thank you for your feedback.', 'success');
                    isRatingLocked = true;
                    document.querySelectorAll('.rating-star').forEach(s => s.classList.add('locked'));
                    btn.textContent = 'Rating Saved';
                    // Update existing rating display
                    loadExistingRating();
                }} else {{
                    showRatingStatus('Failed to save rating: ' + (data.error || 'Unknown error'), 'error');
                    btn.disabled = false;
                    btn.textContent = 'Submit Rating';
                }}
            }} catch (e) {{
                console.error('Rating error:', e);
                showRatingStatus('Error: ' + e.message + '. Make sure reports_server.py is running.', 'error');
                btn.disabled = false;
                btn.textContent = 'Submit Rating';
            }}
        }}
        
        function showRatingStatus(message, type) {{
            const statusEl = document.getElementById('rating-status');
            statusEl.textContent = message;
            statusEl.className = 'rating-status ' + type;
            statusEl.style.display = 'block';
        }}
        
        async function loadExistingRating() {{
            try {{
                const response = await fetch('/api/get-rating?fingerprint=' + encodeURIComponent('{fingerprint}'));
                const data = await response.json();
                
                if (data.success && data.rating) {{
                    const existingDiv = document.getElementById('existing-rating');
                    const stars = '\\u2605'.repeat(data.rating.rating_value) + '\\u2606'.repeat(5 - data.rating.rating_value);
                    let details = 'Rated on ' + new Date(data.rating.rated_at).toLocaleDateString();
                    if (data.rating.user_comment) {{
                        details += ' - "' + data.rating.user_comment + '"';
                    }}
                    existingDiv.innerHTML = '<span class="stars">' + stars + '</span><span class="details">' + details + '</span>';
                    existingDiv.style.display = 'flex';
                    
                    // Pre-select the rating
                    selectedRating = data.rating.rating_value;
                    const starBtns = document.querySelectorAll('.rating-star');
                    starBtns.forEach((star, idx) => {{
                        star.classList.toggle('selected', idx < selectedRating);
                    }});
                }}
            }} catch (e) {{
                // Silently fail - rating display is optional
                console.log('Could not load existing rating:', e.message);
            }}
        }}
        
        window.onload = function() {{
            checkMeshLab();
            loadExistingRating();
        }};
    </script>
"""


def generate_model_report(
    stl_path: Path,
    result: "TestResult",
    reports_path: Path,
    thingi10k_path: Path,
    fixed_path: Optional[Path] = None,
) -> None:
    """Generate HTML report for a single model.
    
    Args:
        stl_path: Path to the original mesh file
        result: TestResult with processing results
        reports_path: Path to reports directory
        thingi10k_path: Path to Thingi10K dataset
        fixed_path: Path to fixed model (if successful)
    """
    report_path = reports_path / f"{stl_path.stem}.html"
    
    # Get adjacent files for navigation
    all_files = sorted(thingi10k_path.glob("*.stl"))
    current_idx = next((i for i, f in enumerate(all_files) if f.name == stl_path.name), -1)
    
    prev_file = all_files[current_idx - 1] if current_idx > 0 else None
    next_file = all_files[current_idx + 1] if current_idx < len(all_files) - 1 else None
    
    # Status
    status_class, status_text = get_status_info(
        result.success, result.precheck_skipped, result.escalation_used
    )
    
    # Calculate changes
    vertex_change = result.result_vertices - result.original_vertices
    face_change = result.result_faces - result.original_faces
    face_change_pct = (face_change / result.original_faces * 100) if result.original_faces > 0 else 0
    
    # Navigation links
    prev_link = f'<a href="{prev_file.stem}.html">&lt; {prev_file.stem}</a>' if prev_file else '<span class="disabled">&lt; Previous</span>'
    next_link = f'<a href="{next_file.stem}.html">{next_file.stem} &gt;</a>' if next_file else '<span class="disabled">Next &gt;</span>'
    
    # Fixed model link
    if fixed_path and fixed_path.exists():
        fixed_rel_path = f"../fixed/{fixed_path.name}"
        fixed_link = f'''<a href="{fixed_rel_path}" class="download-btn" download>&#11015; Download Fixed</a>
            <button class="download-btn meshlab-btn" onclick="openInMeshLab('{fixed_rel_path}')">&#128065; MeshLab</button>'''
    elif result.precheck_skipped:
        fixed_link = '<span class="no-file">Original is already clean</span>'
    else:
        fixed_link = '<span class="no-file">Repair failed - no fixed model</span>'
    
    # Original model link
    original_rel_path = f'../raw_meshes/{stl_path.name}'
    
    # Duration formatting
    duration_text = format_duration(result.duration_ms)
    
    # Build styles
    all_styles = (
        get_base_styles() +
        get_nav_bar_styles() +
        get_status_badge_styles() +
        get_button_styles() +
        get_error_box_styles() +
        _get_report_styles()
    )
    
    # Build HTML
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{stl_path.stem} - MeshPrep Report</title>
    <style>
{all_styles}
    </style>
{_get_report_scripts(result.model_fingerprint, result.file_id)}
</head>
<body>
    <div class="container">
        <div class="nav-bar">
            <div>{prev_link}</div>
            <div>
                <a href="index.html">&#128209; Index</a>
                <a href="/dashboard">&#128202; Dashboard</a>
                <span id="meshlab-indicator" class="meshlab-status">Checking MeshLab...</span>
            </div>
            <div>{next_link}</div>
        </div>
        
        <h1>{stl_path.stem}</h1>
        <span class="status-badge {status_class}">{status_text}</span>
        
        <!-- Fingerprint Box -->
        <div class="fingerprint-box">
            <div class="fingerprint-label">&#128269; Model Fingerprint (search this on Reddit to find filter scripts)</div>
            <div class="fingerprint-value" onclick="copyFingerprint()" title="Click to copy">{result.model_fingerprint}</div>
            <div class="fingerprint-help">
                <a href="https://www.reddit.com/search/?q={result.model_fingerprint}" target="_blank">Search Reddit</a> | 
                <a href="https://www.google.com/search?q={result.model_fingerprint}" target="_blank">Search Google</a> |
                <a href="https://github.com/DragonAceNL/MeshPrep" target="_blank">MeshPrep GitHub</a> |
                <span class="copy-hint">Click fingerprint to copy</span>
            </div>
        </div>
        
        <!-- Rating Box -->
        <div class="rating-box">
            <h3>&#11088; Rate This Repair</h3>
            <div id="existing-rating" class="existing-rating" style="display: none;"></div>
            <div class="rating-labels">
                <span>Rejected</span>
                <span>Poor</span>
                <span>Acceptable</span>
                <span>Good</span>
                <span>Perfect</span>
            </div>
            <div class="rating-stars">
                <button class="rating-star" onmouseover="hoverStar(1)" onmouseout="unhoverStars()" onclick="selectStar(1)">&#9733;</button>
                <button class="rating-star" onmouseover="hoverStar(2)" onmouseout="unhoverStars()" onclick="selectStar(2)">&#9733;</button>
                <button class="rating-star" onmouseover="hoverStar(3)" onmouseout="unhoverStars()" onclick="selectStar(3)">&#9733;</button>
                <button class="rating-star" onmouseover="hoverStar(4)" onmouseout="unhoverStars()" onclick="selectStar(4)">&#9733;</button>
                <button class="rating-star" onmouseover="hoverStar(5)" onmouseout="unhoverStars()" onclick="selectStar(5)">&#9733;</button>
            </div>
            <textarea id="rating-comment" class="rating-comment" placeholder="Optional: Add a comment about the repair quality..."></textarea>
            <button id="submit-rating" class="rating-submit" onclick="submitRating()" disabled>Submit Rating</button>
            <div id="rating-status" class="rating-status" style="display: none;"></div>
        </div>
        
        <div class="info-grid">
            <div class="info-card">
                <div class="label">Filter Used</div>
                <div class="value">{result.filter_used}</div>
            </div>
            <div class="info-card">
                <div class="label">Duration</div>
                <div class="value">{duration_text}</div>
            </div>
            <div class="info-card">
                <div class="label">Original Faces</div>
                <div class="value">{result.original_faces:,}</div>
            </div>
            <div class="info-card">
                <div class="label">Result Faces</div>
                <div class="value">{result.result_faces:,}</div>
            </div>
        </div>
        
        <div class="downloads">
            <a href="{original_rel_path}" class="download-btn secondary" download>&#11015; Download Original</a>
            <button class="download-btn meshlab-btn" onclick="openInMeshLab('{original_rel_path}')">&#128065; MeshLab</button>
            {fixed_link}
            <a href="filters/{stl_path.stem}.json" class="download-btn secondary">&#128196; Filter Script</a>
        </div>
        
        <h2>Visual Comparison</h2>
        <div class="comparison">
            <div class="comparison-panel">
                <h3>Before</h3>
                <img src="images/{stl_path.stem}_before.png" alt="Before">
            </div>
            <div class="comparison-panel">
                <h3>After</h3>
                <img src="images/{stl_path.stem}_after.png" alt="After">
            </div>
        </div>
        
        <h2>Metrics</h2>
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Before</th>
                    <th>After</th>
                    <th>Change</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Vertices</td>
                    <td>{result.original_vertices:,}</td>
                    <td>{result.result_vertices:,}</td>
                    <td class="{'change-positive' if vertex_change > 0 else 'change-negative' if vertex_change < 0 else 'change-neutral'}">{vertex_change:+,}</td>
                </tr>
                <tr>
                    <td>Faces</td>
                    <td>{result.original_faces:,}</td>
                    <td>{result.result_faces:,}</td>
                    <td class="{'change-positive' if face_change > 0 else 'change-negative' if face_change < 0 else 'change-neutral'}">{face_change:+,} ({face_change_pct:+.1f}%)</td>
                </tr>
                <tr>
                    <td>Volume</td>
                    <td>{result.original_volume:.2f}</td>
                    <td>{result.result_volume:.2f}</td>
                    <td class="{'change-positive' if result.volume_change_pct > 5 else 'change-negative' if result.volume_change_pct < -5 else 'change-neutral'}">{result.volume_change_pct:+.1f}%</td>
                </tr>
                <tr>
                    <td>Watertight</td>
                    <td>{'&#10003; Yes' if result.original_watertight else '&#10007; No'}</td>
                    <td>{'&#10003; Yes' if result.result_watertight else '&#10007; No'}</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Manifold</td>
                    <td>{'&#10003; Yes' if result.original_manifold else '&#10007; No'}</td>
                    <td>{'&#10003; Yes' if result.result_manifold else '&#10007; No'}</td>
                    <td>-</td>
                </tr>
            </tbody>
        </table>
"""
    
    if result.error:
        html_content += f"""        <div class="error-box">
            <h3>&#9888; Error</h3>
            <pre>{result.error}</pre>
        </div>
"""
    
    html_content += f"""        <div class="footer">
            Generated: {result.timestamp}
        </div>
    </div>
</body>
</html>
"""
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
