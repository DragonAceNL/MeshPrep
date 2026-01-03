# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Learning Status Page Generator for MeshPrep.

Generates an HTML dashboard showing:
- Learning engine statistics
- Pipeline evolution statistics
- What the system has learned
- Action success rates by issue type
- Evolved pipeline details
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Add POC v2 to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "v2"))

from meshprep_poc.learning_engine import get_learning_engine

# Try to import detailed learning logger
try:
    from detailed_learning_logger import get_detailed_logger, DetailedLearningLogger
    DETAILED_LOGGING_AVAILABLE = True
except ImportError:
    DETAILED_LOGGING_AVAILABLE = False
    get_detailed_logger = None

# Try to import evolution engine
try:
    from meshprep_poc.pipeline_evolution import get_evolution_engine
    EVOLUTION_AVAILABLE = True
except ImportError:
    EVOLUTION_AVAILABLE = False
    get_evolution_engine = None

# Try to import reproducibility module
try:
    from meshprep_poc.reproducibility import (
        capture_environment,
        check_compatibility,
        ReproducibilityLevel,
        get_meshprep_version,
    )
    REPRODUCIBILITY_AVAILABLE = True
except ImportError:
    REPRODUCIBILITY_AVAILABLE = False

# Try to import filter pipelines
try:
    from meshprep_poc.filter_pipelines import PROFILE_PIPELINES, GENERIC_PIPELINES
    PIPELINES_AVAILABLE = True
except ImportError:
    PIPELINES_AVAILABLE = False
    PROFILE_PIPELINES = {}
    GENERIC_PIPELINES = []

# Try to import profile discovery
try:
    from meshprep_poc.profile_discovery import get_discovery_engine
    PROFILE_DISCOVERY_AVAILABLE = True
except ImportError:
    PROFILE_DISCOVERY_AVAILABLE = False
    get_discovery_engine = None

# Output path
STATUS_PAGE_PATH = Path(__file__).parent / "learning_status.html"


def get_learning_data() -> Dict[str, Any]:
    """Gather all learning data for the status page."""
    data = {
        "generated_at": datetime.now().isoformat(),
        "learning_engine": None,
        "evolution_engine": None,
        "detailed_analysis": None,
        "environment": None,
        "available_pipelines": None,
    }
    
    # Get environment info
    if REPRODUCIBILITY_AVAILABLE:
        try:
            snapshot = capture_environment(ReproducibilityLevel.STANDARD, include_external=False)
            compat = check_compatibility()
            data["environment"] = {
                "meshprep_version": snapshot.meshprep_version,
                "python_version": snapshot.python_version,
                "platform": snapshot.platform_info,
                "packages": snapshot.package_versions,
                "compatible": compat.compatible,
                "issues": len(compat.issues),
            }
        except Exception as e:
            data["environment"] = {"error": str(e)}
    
    # Get available pipelines
    if PIPELINES_AVAILABLE:
        try:
            profile_count = len(PROFILE_PIPELINES)
            total_pipelines = sum(len(pipes) for pipes in PROFILE_PIPELINES.values())
            generic_count = len(GENERIC_PIPELINES)
            
            # Get some sample pipelines
            sample_pipelines = []
            for profile, pipes in list(PROFILE_PIPELINES.items())[:3]:
                for pipe in pipes[:2]:
                    sample_pipelines.append({
                        "profile": profile,
                        "name": pipe.name,
                        "actions": [a["action"] for a in pipe.actions],
                        "priority": pipe.priority,
                    })
            
            data["available_pipelines"] = {
                "profile_categories": profile_count,
                "total_profile_pipelines": total_pipelines,
                "generic_pipelines": generic_count,
                "total": total_pipelines + generic_count,
                "samples": sample_pipelines,
            }
        except Exception as e:
            data["available_pipelines"] = {"error": str(e)}
    
    # Get learning engine stats
    try:
        engine = get_learning_engine()
        stats = engine.get_stats_summary()
        
        # Get detailed pipeline stats
        pipeline_details = []
        if hasattr(engine, '_get_connection'):
            with engine._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT pipeline_name, total_attempts, successes, failures,
                           total_duration_ms,
                           CAST(successes AS REAL) / NULLIF(total_attempts, 0) as success_rate,
                           total_duration_ms / NULLIF(total_attempts, 0) as avg_duration_ms
                    FROM pipeline_stats
                    WHERE total_attempts >= 1
                    ORDER BY success_rate DESC, total_attempts DESC
                    LIMIT 50
                """)
                for row in cursor.fetchall():
                    pipeline_details.append({
                        "name": row["pipeline_name"],
                        "attempts": row["total_attempts"],
                        "successes": row["successes"],
                        "failures": row["failures"],
                        "success_rate": row["success_rate"] or 0,
                        "avg_duration_ms": row["avg_duration_ms"] or 0,
                    })
                
                # Get issue pattern stats
                issue_patterns = []
                cursor.execute("""
                    SELECT pattern_key, 
                           SUM(successes) as total_successes,
                           SUM(failures) as total_failures
                    FROM pattern_pipeline_results
                    GROUP BY pattern_key
                    HAVING total_successes + total_failures >= 3
                    ORDER BY total_successes DESC
                    LIMIT 30
                """)
                for row in cursor.fetchall():
                    total = row["total_successes"] + row["total_failures"]
                    issue_patterns.append({
                        "pattern": row["pattern_key"],
                        "successes": row["total_successes"],
                        "failures": row["total_failures"],
                        "total": total,
                        "success_rate": row["total_successes"] / total if total > 0 else 0,
                    })
                
                # Get best pipeline per issue pattern
                best_per_pattern = []
                cursor.execute("""
                    SELECT pattern_key, pipeline_name, successes, failures,
                           CAST(successes AS REAL) / NULLIF(successes + failures, 0) as success_rate
                    FROM pattern_pipeline_results
                    WHERE successes + failures >= 3
                    ORDER BY pattern_key, success_rate DESC
                """)
                current_pattern = None
                for row in cursor.fetchall():
                    if row["pattern_key"] != current_pattern:
                        current_pattern = row["pattern_key"]
                        best_per_pattern.append({
                            "pattern": row["pattern_key"],
                            "best_pipeline": row["pipeline_name"],
                            "success_rate": row["success_rate"] or 0,
                            "attempts": row["successes"] + row["failures"],
                        })
                
                # Get profile stats
                profile_stats = []
                cursor.execute("""
                    SELECT profile, COUNT(*) as total,
                           SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes,
                           SUM(CASE WHEN escalated = 1 THEN 1 ELSE 0 END) as escalations
                    FROM model_results
                    GROUP BY profile
                    ORDER BY total DESC
                """)
                for row in cursor.fetchall():
                    profile_stats.append({
                        "profile": row["profile"] or "unknown",
                        "total": row["total"],
                        "successes": row["successes"],
                        "fix_rate": row["successes"] / row["total"] if row["total"] > 0 else 0,
                        "escalation_rate": row["escalations"] / row["total"] if row["total"] > 0 else 0,
                    })
                
                # Get mesh characteristic stats
                mesh_char_stats = []
                cursor.execute("""
                    SELECT characteristic, pipeline_name,
                           successes, failures,
                           CAST(successes AS REAL) / NULLIF(successes + failures, 0) as success_rate
                    FROM pipeline_mesh_stats
                    WHERE successes + failures >= 5
                    ORDER BY characteristic, success_rate DESC
                """)
                for row in cursor.fetchall():
                    mesh_char_stats.append({
                        "characteristic": row["characteristic"],
                        "pipeline": row["pipeline_name"],
                        "successes": row["successes"],
                        "failures": row["failures"],
                        "success_rate": row["success_rate"] or 0,
                    })
        
        data["learning_engine"] = {
            "stats": stats,
            "pipeline_details": pipeline_details,
            "issue_patterns": issue_patterns,
            "best_per_pattern": best_per_pattern,
            "profile_stats": profile_stats,
            "mesh_char_stats": mesh_char_stats,
            "data_path": str(engine.data_path),
        }
    except Exception as e:
        data["learning_engine"] = {"error": str(e)}
    
    # Get evolution engine stats
    if EVOLUTION_AVAILABLE:
        try:
            evolution = get_evolution_engine()
            evo_stats = evolution.get_stats_summary()
            
            # Get detailed action stats
            action_stats = []
            with evolution._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT action_name, params_json,
                           total_attempts, successes, failures,
                           CAST(successes AS REAL) / NULLIF(total_attempts, 0) as success_rate,
                           total_duration_ms / NULLIF(total_attempts, 0) as avg_duration_ms
                    FROM action_stats
                    WHERE total_attempts >= 3
                    ORDER BY success_rate DESC, total_attempts DESC
                    LIMIT 30
                """)
                for row in cursor.fetchall():
                    action_stats.append({
                        "action": row["action_name"],
                        "params": row["params_json"],
                        "attempts": row["total_attempts"],
                        "successes": row["successes"],
                        "failures": row["failures"],
                        "success_rate": row["success_rate"] or 0,
                        "avg_duration_ms": row["avg_duration_ms"] or 0,
                    })
                
                # Get action-issue stats
                action_issue_stats = []
                cursor.execute("""
                    SELECT a.action_name, i.issue_type,
                           i.successes, i.failures,
                           CAST(i.successes AS REAL) / NULLIF(i.successes + i.failures, 0) as success_rate
                    FROM action_issue_stats i
                    JOIN action_stats a ON i.action_key = a.action_key
                    WHERE i.successes + i.failures >= 3
                    ORDER BY i.issue_type, success_rate DESC
                """)
                for row in cursor.fetchall():
                    action_issue_stats.append({
                        "action": row["action_name"],
                        "issue": row["issue_type"],
                        "successes": row["successes"],
                        "failures": row["failures"],
                        "success_rate": row["success_rate"] or 0,
                    })
                
                # Get evolved pipelines
                evolved_pipelines = []
                cursor.execute("""
                    SELECT name, actions_json, generation, attempts, successes,
                           CAST(successes AS REAL) / NULLIF(attempts, 0) as success_rate,
                           total_duration_ms / NULLIF(attempts, 0) as avg_duration_ms,
                           created_at
                    FROM evolved_pipelines
                    ORDER BY success_rate DESC, attempts DESC
                    LIMIT 20
                """)
                for row in cursor.fetchall():
                    actions = json.loads(row["actions_json"])
                    evolved_pipelines.append({
                        "name": row["name"],
                        "actions": [a["action"] for a in actions],
                        "actions_detail": actions,
                        "generation": row["generation"],
                        "attempts": row["attempts"],
                        "successes": row["successes"],
                        "success_rate": row["success_rate"] or 0,
                        "avg_duration_ms": row["avg_duration_ms"] or 0,
                        "created_at": row["created_at"],
                    })
            
            data["evolution_engine"] = {
                "stats": evo_stats,
                "action_stats": action_stats,
                "action_issue_stats": action_issue_stats,
                "evolved_pipelines": evolved_pipelines,
                "data_path": str(evolution.data_path),
            }
        except Exception as e:
            data["evolution_engine"] = {"error": str(e)}
    else:
        data["evolution_engine"] = {"error": "Evolution engine not available"}
    
    # Get detailed analysis data
    if DETAILED_LOGGING_AVAILABLE:
        try:
            detailed_logger = get_detailed_logger()
            data["detailed_analysis"] = detailed_logger.get_analysis_summary()
        except Exception as e:
            data["detailed_analysis"] = {"error": str(e)}
    else:
        data["detailed_analysis"] = {"error": "Detailed logging not available"}
    
    # Get profile discovery data
    if PROFILE_DISCOVERY_AVAILABLE:
        try:
            discovery = get_discovery_engine()
            data["profile_discovery"] = discovery.get_stats_summary()
        except Exception as e:
            data["profile_discovery"] = {"error": str(e)}
    else:
        data["profile_discovery"] = {"error": "Profile discovery not available"}
    
    return data


def generate_status_page(data: Dict[str, Any]) -> str:
    """Generate the HTML status page."""
    
    # Environment section
    env_html = ""
    if data.get("environment") and "error" not in data["environment"]:
        env = data["environment"]
        compat_class = "success" if env["compatible"] else "danger"
        compat_text = "Compatible" if env["compatible"] else "Issues Found"
        
        # Build package list
        pkg_items = ""
        for pkg, ver in env.get("packages", {}).items():
            pkg_items += f'<span class="pkg-item"><code>{pkg}</code> {ver}</span>'
        
        env_html = f'''
        <div class="section">
            <h2>🖥️ Environment</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">v{env["meshprep_version"]}</div>
                    <div class="stat-label">MeshPrep Version</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{env["python_version"]}</div>
                    <div class="stat-label">Python</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value {compat_class}">{compat_text}</div>
                    <div class="stat-label">Compatibility</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(env.get("packages", {}))}</div>
                    <div class="stat-label">Packages</div>
                </div>
            </div>
            <div class="pkg-list" style="margin-top: 15px; display: flex; flex-wrap: wrap; gap: 10px;">
                {pkg_items}
            </div>
        </div>
        '''
    else:
        env_html = ''
    
    # Available pipelines section
    pipelines_html = ""
    if data.get("available_pipelines") and "error" not in data["available_pipelines"]:
        ap = data["available_pipelines"]
        
        # Build sample pipelines list
        samples_html = ""
        for p in ap.get("samples", []):
            actions_str = " → ".join(p["actions"])
            samples_html += f'''
                <div class="pipeline-sample">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <code>{p["name"]}</code>
                        <span style="color: #888; font-size: 12px;">{p["profile"]}</span>
                    </div>
                    <div style="font-size: 12px; color: #aaa;">{actions_str}</div>
                </div>
            '''
        
        pipelines_html = f'''
        <div class="section">
            <h2>🔧 Available Pipelines</h2>
            <p class="section-desc">Pre-configured repair pipelines ready to use</p>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{ap["total"]}</div>
                    <div class="stat-label">Total Pipelines</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{ap["profile_categories"]}</div>
                    <div class="stat-label">Profile Categories</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{ap["total_profile_pipelines"]}</div>
                    <div class="stat-label">Profile-Specific</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{ap["generic_pipelines"]}</div>
                    <div class="stat-label">Generic Fallbacks</div>
                </div>
            </div>
            <h3 style="margin-top: 20px; color: #4fe8c4; font-size: 14px;">Sample Pipelines</h3>
            <div style="display: grid; gap: 10px;">
                {samples_html}
            </div>
        </div>
        '''
    else:
        pipelines_html = ''
    
    # Learning engine section
    learning_html = ""
    if data["learning_engine"] and "error" not in data["learning_engine"]:
        le = data["learning_engine"]
        stats = le["stats"]
        
        # Summary cards
        learning_html += f"""
        <div class="section">
            <h2>📊 Learning Engine Summary</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{stats.get('total_models_processed', 0):,}</div>
                    <div class="stat-label">Models Processed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats.get('pipelines_tracked', 0)}</div>
                    <div class="stat-label">Pipelines Tracked</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats.get('issue_patterns_tracked', 0)}</div>
                    <div class="stat-label">Issue Patterns</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats.get('profiles_tracked', 0)}</div>
                    <div class="stat-label">Profiles Tracked</div>
                </div>
            </div>
            <p class="data-path">Data: {le.get('data_path', 'N/A')}</p>
        </div>
        """
        
        # Pipeline performance table
        if le.get("pipeline_details"):
            learning_html += """
            <div class="section">
                <h2>🔧 Pipeline Performance</h2>
                <p class="section-desc">Success rates for each repair pipeline based on learned data.</p>
                <table>
                    <thead>
                        <tr>
                            <th>Pipeline</th>
                            <th>Success Rate</th>
                            <th>Attempts</th>
                            <th>Successes</th>
                            <th>Failures</th>
                            <th>Avg Duration</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            for p in le["pipeline_details"][:20]:
                rate_class = "success" if p["success_rate"] >= 0.7 else "warning" if p["success_rate"] >= 0.4 else "danger"
                learning_html += f"""
                        <tr>
                            <td><code>{p['name']}</code></td>
                            <td class="{rate_class}">{p['success_rate']*100:.1f}%</td>
                            <td>{p['attempts']:,}</td>
                            <td>{p['successes']:,}</td>
                            <td>{p['failures']:,}</td>
                            <td>{p['avg_duration_ms']:.0f}ms</td>
                        </tr>
                """
            learning_html += """
                    </tbody>
                </table>
            </div>
            """
        
        # Best pipeline per issue pattern
        if le.get("best_per_pattern"):
            learning_html += """
            <div class="section">
                <h2>🎯 Best Pipeline Per Issue Pattern</h2>
                <p class="section-desc">The system has learned which pipelines work best for specific issue combinations.</p>
                <table>
                    <thead>
                        <tr>
                            <th>Issue Pattern</th>
                            <th>Best Pipeline</th>
                            <th>Success Rate</th>
                            <th>Attempts</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            for p in le["best_per_pattern"][:15]:
                rate_class = "success" if p["success_rate"] >= 0.7 else "warning" if p["success_rate"] >= 0.4 else "danger"
                learning_html += f"""
                        <tr>
                            <td><code>{p['pattern']}</code></td>
                            <td><code>{p['best_pipeline']}</code></td>
                            <td class="{rate_class}">{p['success_rate']*100:.1f}%</td>
                            <td>{p['attempts']}</td>
                        </tr>
                """
            learning_html += """
                    </tbody>
                </table>
            </div>
            """
        
        # Profile statistics
        if le.get("profile_stats"):
            learning_html += """
            <div class="section">
                <h2>📁 Model Profile Statistics</h2>
                <p class="section-desc">Success rates grouped by detected model profile (mesh characteristics).</p>
                <table>
                    <thead>
                        <tr>
                            <th>Profile</th>
                            <th>Total Models</th>
                            <th>Fix Rate</th>
                            <th>Escalation Rate</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            for p in le["profile_stats"]:
                fix_class = "success" if p["fix_rate"] >= 0.8 else "warning" if p["fix_rate"] >= 0.5 else "danger"
                learning_html += f"""
                        <tr>
                            <td><code>{p['profile']}</code></td>
                            <td>{p['total']:,}</td>
                            <td class="{fix_class}">{p['fix_rate']*100:.1f}%</td>
                            <td>{p['escalation_rate']*100:.1f}%</td>
                        </tr>
                """
            learning_html += """
                    </tbody>
                </table>
            </div>
            """
    else:
        error = data["learning_engine"].get("error", "Unknown error") if data["learning_engine"] else "Not initialized"
        learning_html = f"""
        <div class="section">
            <h2>📊 Learning Engine</h2>
            <div class="error-box">No learning data available yet. Error: {error}</div>
        </div>
        """
    
    # Evolution engine section
    evolution_html = ""
    if data["evolution_engine"] and "error" not in data["evolution_engine"]:
        ee = data["evolution_engine"]
        evo_stats = ee["stats"]
        
        evolution_html += f"""
        <div class="section">
            <h2>🧬 Pipeline Evolution Summary</h2>
            <div class="stats-grid">
                <div class="stat-card evolved">
                    <div class="stat-value">{evo_stats.get('total_evolved_pipelines', 0)}</div>
                    <div class="stat-label">Evolved Pipelines</div>
                </div>
                <div class="stat-card evolved">
                    <div class="stat-value">{evo_stats.get('successful_evolved_pipelines', 0)}</div>
                    <div class="stat-label">Successful (≥50%)</div>
                </div>
                <div class="stat-card evolved">
                    <div class="stat-value">{evo_stats.get('tracked_actions', 0)}</div>
                    <div class="stat-label">Actions Tracked</div>
                </div>
                <div class="stat-card evolved">
                    <div class="stat-value">Gen {evo_stats.get('current_generation', 0)}</div>
                    <div class="stat-label">Current Generation</div>
                </div>
            </div>
        </div>
        """
        
        # Action performance
        if ee.get("action_stats"):
            evolution_html += """
            <div class="section">
                <h2>⚡ Individual Action Performance</h2>
                <p class="section-desc">Success rates for individual repair actions (used to generate new combinations).</p>
                <table>
                    <thead>
                        <tr>
                            <th>Action</th>
                            <th>Parameters</th>
                            <th>Success Rate</th>
                            <th>Attempts</th>
                            <th>Avg Duration</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            for a in ee["action_stats"][:15]:
                rate_class = "success" if a["success_rate"] >= 0.7 else "warning" if a["success_rate"] >= 0.4 else "danger"
                params_display = a["params"] if a["params"] != "{}" else "-"
                evolution_html += f"""
                        <tr>
                            <td><code>{a['action']}</code></td>
                            <td><small>{params_display}</small></td>
                            <td class="{rate_class}">{a['success_rate']*100:.1f}%</td>
                            <td>{a['attempts']}</td>
                            <td>{a['avg_duration_ms']:.0f}ms</td>
                        </tr>
                """
            evolution_html += """
                    </tbody>
                </table>
            </div>
            """
        
        # Action-issue mapping
        if ee.get("action_issue_stats"):
            # Group by issue
            by_issue = {}
            for a in ee["action_issue_stats"]:
                issue = a["issue"]
                if issue not in by_issue:
                    by_issue[issue] = []
                by_issue[issue].append(a)
            
            evolution_html += """
            <div class="section">
                <h2>🔗 Best Actions Per Issue Type</h2>
                <p class="section-desc">Which actions work best for specific issues (learned from results).</p>
                <div class="issue-grid">
            """
            for issue, actions in sorted(by_issue.items()):
                top_actions = sorted(actions, key=lambda x: -x["success_rate"])[:3]
                evolution_html += f"""
                    <div class="issue-card">
                        <h4>{issue}</h4>
                        <ul>
                """
                for a in top_actions:
                    rate_class = "success" if a["success_rate"] >= 0.7 else "warning"
                    evolution_html += f"""
                            <li>
                                <code>{a['action']}</code>
                                <span class="{rate_class}">{a['success_rate']*100:.0f}%</span>
                            </li>
                    """
                evolution_html += """
                        </ul>
                    </div>
                """
            evolution_html += """
                </div>
            </div>
            """
        
        # Evolved pipelines
        if ee.get("evolved_pipelines"):
            evolution_html += """
            <div class="section">
                <h2>🧪 Evolved Pipeline Combinations</h2>
                <p class="section-desc">New pipeline combinations discovered through evolution.</p>
                <div class="evolved-list">
            """
            for p in ee["evolved_pipelines"]:
                rate_class = "success" if p["success_rate"] >= 0.7 else "warning" if p["success_rate"] >= 0.4 else "danger"
                actions_str = " → ".join(p["actions"])
                evolution_html += f"""
                    <div class="evolved-card">
                        <div class="evolved-header">
                            <span class="evolved-name">{p['name']}</span>
                            <span class="evolved-gen">Gen {p['generation']}</span>
                        </div>
                        <div class="evolved-actions">{actions_str}</div>
                        <div class="evolved-stats">
                            <span class="{rate_class}">{p['success_rate']*100:.1f}% success</span>
                            <span>{p['attempts']} attempts</span>
                            <span>{p['avg_duration_ms']:.0f}ms avg</span>
                        </div>
                    </div>
                """
            evolution_html += """
                </div>
            </div>
            """
    else:
        error = data["evolution_engine"].get("error", "Not available") if data["evolution_engine"] else "Not initialized"
        evolution_html = f"""
        <div class="section">
            <h2>🧬 Pipeline Evolution</h2>
            <div class="info-box">
                Evolution engine: {error}
                <br><br>
                Evolved pipelines are generated when standard pipelines fail. Run more batch tests to see evolution in action!
            </div>
        </div>
        """
    
    # Detailed analysis section
    detailed_html = ""
    if data.get("detailed_analysis") and "error" not in data["detailed_analysis"]:
        da = data["detailed_analysis"]
        
        detailed_html += """
        <div class="section">
            <h2>🔬 Detailed Action Analysis</h2>
            <p class="section-desc">Granular action-level data for algorithm improvement.</p>
        """
        
        # Problematic actions
        if da.get("problematic_actions"):
            detailed_html += """
            <h3>⚠️ Most Problematic Actions (High Break Rate)</h3>
            <p class="section-desc">Actions that frequently make meshes worse.</p>
            <table>
                <thead>
                    <tr>
                        <th>Action</th>
                        <th>Break Rate</th>
                        <th>Fix Rate</th>
                        <th>Times Tried</th>
                    </tr>
                </thead>
                <tbody>
            """
            for a in da["problematic_actions"]:
                break_class = "danger" if (a["break_rate"] or 0) > 0.3 else "warning" if (a["break_rate"] or 0) > 0.1 else "success"
                detailed_html += f"""
                    <tr>
                        <td><code>{a['action_name']}</code></td>
                        <td class="{break_class}">{(a['break_rate'] or 0)*100:.1f}%</td>
                        <td>{(a['fix_rate'] or 0)*100:.1f}%</td>
                        <td>{a['total']}</td>
                    </tr>
                """
            detailed_html += """
                </tbody>
            </table>
            """
        
        # Common failure modes
        if da.get("common_failure_modes"):
            detailed_html += """
            <h3>❌ Common Failure Modes</h3>
            <table>
                <thead>
                    <tr>
                        <th>Failure Type</th>
                        <th>Stage</th>
                        <th>Count</th>
                    </tr>
                </thead>
                <tbody>
            """
            for f in da["common_failure_modes"]:
                detailed_html += f"""
                    <tr>
                        <td><code>{f['failure_type']}</code></td>
                        <td>{f['failure_stage']}</td>
                        <td>{f['count']}</td>
                    </tr>
                """
            detailed_html += """
                </tbody>
            </table>
            """
        
        # Success by body count
        if da.get("success_by_body_count"):
            detailed_html += """
            <h3>📊 Success Rate by Body Count</h3>
            <p class="section-desc">How mesh fragmentation affects repair success.</p>
            <table>
                <thead>
                    <tr>
                        <th>Body Count</th>
                        <th>Total Models</th>
                        <th>Success Rate</th>
                    </tr>
                </thead>
                <tbody>
            """
            for s in da["success_by_body_count"]:
                rate_class = "success" if (s["success_rate"] or 0) >= 0.7 else "warning" if (s["success_rate"] or 0) >= 0.4 else "danger"
                detailed_html += f"""
                    <tr>
                        <td>{s['body_bucket']}</td>
                        <td>{s['total']}</td>
                        <td class="{rate_class}">{(s['success_rate'] or 0)*100:.1f}%</td>
                    </tr>
                """
            detailed_html += """
                </tbody>
            </table>
            """
        
        detailed_html += "</div>"
    else:
        error = data.get("detailed_analysis", {}).get("error", "Not available")
        detailed_html = f"""
        <div class="section">
            <h2>🔬 Detailed Action Analysis</h2>
            <div class="info-box">
                Detailed analysis: {error}
                <br><br>
                Run batch tests to collect action-level data for algorithm improvement.
            </div>
        </div>
        """
    
    # Discovered profiles section
    discovered_profiles_html = ""
    if data.get("profile_discovery") and "error" not in data["profile_discovery"]:
        dp = data["profile_discovery"]
        
        discovered_profiles_html += f"""
        <div class="section">
            <h2>\U0001F50D Discovered Profiles</h2>
            <p class="section-desc">Automatically discovered mesh profile categories through clustering</p>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{dp.get('active_profiles', 0)}</div>
                    <div class="stat-label">Active Profiles</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{dp.get('total_clusters', 0)}</div>
                    <div class="stat-label">Clusters</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{dp.get('models_with_profiles', 0):,}</div>
                    <div class="stat-label">Models Assigned</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{dp.get('avg_profile_success_rate', 0)*100:.0f}%</div>
                    <div class="stat-label">Avg Success</div>
                </div>
            </div>
        """
        
        if dp.get("top_profiles"):
            discovered_profiles_html += """
            <h3>Top Discovered Profiles</h3>
            <table>
                <thead>
                    <tr>
                        <th>Profile Name</th>
                        <th>Models</th>
                        <th>Success Rate</th>
                        <th>Best Pipeline</th>
                    </tr>
                </thead>
                <tbody>
            """
            for p in dp["top_profiles"]:
                rate_class = "success" if p["success_rate"] >= 0.7 else "warning" if p["success_rate"] >= 0.4 else "danger"
                discovered_profiles_html += f"""
                    <tr>
                        <td><code>{p['name']}</code></td>
                        <td>{p['total_models']:,}</td>
                        <td class="{rate_class}">{p['success_rate']*100:.1f}%</td>
                        <td>{p['best_pipeline'] or 'N/A'}</td>
                    </tr>
                """
            discovered_profiles_html += """
                </tbody>
            </table>
            """
        
        # Show unassigned clusters waiting for discovery
        if dp.get("unassigned_clusters", 0) > 0:
            discovered_profiles_html += f"""
            <div class="info-box" style="margin-top: 15px;">
                <strong>{dp['unassigned_clusters']}</strong> clusters with 
                <strong>{dp.get('unassigned_models', 0):,}</strong> models awaiting profile assignment.
                Run profile discovery to create new profiles.
            </div>
            """
        
        discovered_profiles_html += "</div>"
    else:
        error = data.get("profile_discovery", {}).get("error", "Not available")
        dp_data = data.get("profile_discovery", {})
        if dp_data.get("total_clusters", 0) == 0:
            discovered_profiles_html = f"""
            <div class="section">
                <h2>\U0001F50D Discovered Profiles</h2>
                <div class="info-box">
                    No profiles discovered yet. Process more models to enable automatic profile discovery.
                </div>
            </div>
            """
        else:
            discovered_profiles_html = f"""
            <div class="section">
                <h2>\U0001F50D Discovered Profiles</h2>
                <div class="info-box">
                    Profile discovery: {error}
                </div>
            </div>
            """
    
    # Generate full HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>MeshPrep Learning Status</title>
    <meta http-equiv="refresh" content="60">
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f1720;
            color: #dff6fb;
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        
        h1 {{ color: #4fe8c4; margin-bottom: 5px; }}
        h2 {{ color: #4fe8c4; margin-top: 0; border-bottom: 1px solid #2a3a43; padding-bottom: 10px; }}
        h4 {{ color: #4fe8c4; margin: 0 0 10px 0; }}
        
        .subtitle {{ color: #888; margin-bottom: 30px; }}
        .section-desc {{ color: #888; margin-top: -5px; margin-bottom: 15px; font-size: 14px; }}
        .data-path {{ color: #555; font-size: 12px; margin-top: 10px; }}
        
        .section {{
            background: #1b2b33;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 10px;
        }}
        .stat-card {{
            background: #0f1720;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }}
        .stat-card.evolved {{
            background: linear-gradient(135deg, #1a2a35 0%, #2a3a45 100%);
            border: 1px solid #4fe8c4;
        }}
        .stat-value {{
            font-size: 28px;
            font-weight: bold;
            color: #4fe8c4;
        }}
        .stat-label {{
            color: #888;
            font-size: 12px;
            margin-top: 5px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        th, td {{
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid #2a3a43;
        }}
        th {{
            background: #0f1720;
            color: #4fe8c4;
            font-weight: 600;
        }}
        tr:hover {{
            background: #2a3a43;
        }}
        
        code {{
            background: #0f1720;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 12px;
        }}
        
        .success {{ color: #2ecc71; }}
        .warning {{ color: #f39c12; }}
        .danger {{ color: #e74c3c; }}
        
        .error-box {{
            background: #2a1a1a;
            border: 1px solid #e74c3c;
            border-radius: 8px;
            padding: 15px;
            color: #e74c3c;
        }}
        .info-box {{
            background: #1a2a35;
            border: 1px solid #3498db;
            border-radius: 8px;
            padding: 15px;
            color: #dff6fb;
        }}
        
        .issue-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .issue-card {{
            background: #0f1720;
            border-radius: 8px;
            padding: 15px;
        }}
        .issue-card ul {{
            list-style: none;
            padding: 0;
            margin: 0;
        }}
        .issue-card li {{
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #2a3a43;
        }}
        .issue-card li:last-child {{
            border-bottom: none;
        }}
        
        .evolved-list {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        .evolved-card {{
            background: #0f1720;
            border-radius: 8px;
            padding: 15px;
            border-left: 3px solid #4fe8c4;
        }}
        .evolved-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
        }}
        .evolved-name {{
            font-weight: bold;
            color: #4fe8c4;
            font-size: 12px;
        }}
        .evolved-gen {{
            color: #888;
            font-size: 12px;
        }}
        .evolved-actions {{
            font-family: monospace;
            font-size: 13px;
            color: #dff6fb;
            margin-bottom: 10px;
            padding: 8px;
            background: #1b2b33;
            border-radius: 4px;
        }}
        .evolved-stats {{
            display: flex;
            gap: 15px;
            font-size: 12px;
            color: #888;
        }}
        .evolved-stats span {{
            padding: 3px 8px;
            background: #1b2b33;
            border-radius: 4px;
        }}
        
        .nav-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #1b2b33;
            padding: 10px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .nav-bar a {{
            color: #4fe8c4;
            text-decoration: none;
            padding: 5px 10px;
        }}
        .nav-bar a:hover {{
            background: #2a3a43;
            border-radius: 4px;
        }}
        
        .refresh-note {{
            color: #555;
            font-size: 12px;
            text-align: center;
            margin-top: 20px;
        }}
        
        .pkg-item {{
            background: #0f1720;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 12px;
        }}
        .pkg-item code {{
            color: #4fe8c4;
            margin-right: 5px;
        }}
        
        .pipeline-sample {{
            background: #0f1720;
            padding: 12px;
            border-radius: 8px;
            border-left: 3px solid #3498db;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="nav-bar">
            <div>
                <a href="index.html">📋 Reports Index</a>
                <a href="/dashboard">📊 Dashboard</a>
            </div>
            <div>
                <span style="color: #888;">Last updated: {data['generated_at'][:19]}</span>
            </div>
        </div>
        
        <h1>🧠 MeshPrep Learning Status</h1>
        <p class="subtitle">What the system has learned from processing models</p>
        
        {env_html}
        
        {pipelines_html}
        
        {learning_html}
        
        {evolution_html}
        
        {detailed_html}
        
        {discovered_profiles_html}
        
        <p class="refresh-note">This page auto-refreshes every 60 seconds. <a href="javascript:location.reload()">Refresh now</a></p>
    </div>
</body>
</html>
"""
    return html


def generate_learning_status_page() -> Path:
    """Generate and save the learning status page."""
    data = get_learning_data()
    html = generate_status_page(data)
    
    with open(STATUS_PAGE_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"Generated learning status page: {STATUS_PAGE_PATH}")
    return STATUS_PAGE_PATH


if __name__ == "__main__":
    generate_learning_status_page()
