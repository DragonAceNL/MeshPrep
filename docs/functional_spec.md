Functional Specification: Automated STL Cleanup Pipeline

Overview

Goal
- Provide a simple, robust, and reproducible automated pipeline that converts difficult or messy STL files into 3D-printable models with minimal or no manual intervention.
- Make the conversion workflow easy to use for non-technical Windows users via a lightweight GUI while retaining a powerful CLI for advanced users and automation.
- Make the tool extremely easy to set up and use: avoid requiring complex manual environment commands. If manual setup steps are required, present clear, step-by-step instructions in the GUI/CLI and logs. Where feasible implement automatic environment setup (virtualenv creation, dependency installation, basic tool detection) so users can get started with a single action.
- Support creation, editing, importing, exporting, and sharing of filter scripts and filter script presets so users can iterate and reuse successful repair strategies.
- Ensure the system can handle hard-to-fix models by escalating to advanced steps (e.g., Blender-based remeshing) when conservative repairs fail.
- Surface errors, warnings, and important diagnostics clearly to the user (both on-screen and in log files) so runs are debuggable and reproducible.
- Prioritize conversion quality and reproducibility over raw speed; long-running but deterministic repairs are acceptable.
- Enable community sharing and discovery of filter script presets (e.g., Reddit), including metadata and reproduction instructions so a preset can be associated with a model and found by others.
- Automatically detect a model's profile (see `docs/model_profiles.md`) and generate a suggested filter script tailored to that profile; present the suggested filter script for review and editing before execution.

Scope
- Input: a single STL file (ASCII or binary) per run. The tool will scan the selected model to produce a suggested, generic filter script that fits the model's profile; users can review and tweak the suggested filter script before applying it. The software will not accept a directory of files as the primary workflow — each model is treated individually to allow per-model tuning and reproducible presets.
- Output: cleaned STL files suitable for slicing and a CSV/JSON report with diagnostics and the chosen filter script for the model.
- Tools: Python-based stack using `trimesh`, `pymeshfix`, `meshio`. Optional escalation uses Blender headless if needed.

Non-Goals
- Provide a full-featured cloud service, hosted web app, or serverless execution model (deployment to cloud is out of scope for the initial version).
- Implement exhaustive manufacturability, structural simulation, or advanced slicing optimization — only basic geometry validations (watertightness, manifoldness, component sanity) are required.
- Target mobile platforms or provide a native macOS/Linux GUI as the primary interface in the initial release (desktop Windows GUI is required; cross-platform CLI/automation support remains a goal).
- Guarantee that every possible corrupt mesh can be fixed; extremely damaged meshes may still require manual intervention and will be reported as failures with diagnostics.

Note on advanced slicing optimization
- Advanced slicing optimization is intentionally out of scope for the initial release, but the design must explicitly accommodate it as a planned, optional feature.
- The system should expose clear extension points for slicer integrations and record all processing metadata (tool versions, profiles, and CLI parameters) so slicer-driven checks or optimizations can be added later without breaking reproducibility.
- Suggested phased approach for adding slicing features later:
  1. Validate-only integration: run a slicer in preview mode to collect diagnostics (supports, estimated filament/time, slicing errors) and include results in reports.
  2. Slice-and-validate: produce pinned-version G-code and parse layer/preview data for programmatic checks.
  3. Optimize-and-slice (opt-in): apply geometry transformations driven by slicer feedback (orientation, splits, thickening) and re-validate; require explicit user consent and provide diffs.

Requirements

Functional Requirements
1. Per-model processing: accept exactly one input STL file; automatically scan the model to produce a suggested generic filter script that matches the model profile. Allow the user to edit, save, export, and re-use that filter script as a preset.
2. Validation checks: watertightness, manifoldness, consistent normals, component count, and bounding box sanity.
3. Repair steps: remove degenerate faces, merge duplicate vertices, reorient normals, fill holes, remove tiny disconnected components.
4. Escalation: if primary repairs fail, run an advanced Blender-based pipeline (if Blender present) with remeshing and boolean cleanup.
5. Configurable filter scripts: filter scripts defined in JSON/YAML and selectable via GUI/CLI; suggested filter scripts from model scan may be created automatically.
6. Reporting: generate CSV and JSON reports detailing diagnostics, filter script attempts, runtime, and final status for the model.
7. Deterministic filenames: output files named with original name + filter script suffix.
8. Logging: progress logs with per-file detail and error handling.

Non-Functional Requirements
- Reproducibility: deterministic behavior given same inputs and filter script config.
- Extensibility: easy to add new filter actions or replace tools.
- Portability: runs on major OSs (Windows, macOS, Linux) with documented dependencies.
- Stability: handle corrupted files without crashing; log failures and continue.

Design

High-level flow
1. Startup / environment prep:
   - Run `checkenv` to detect required tools and Python dependencies.
   - If the environment is incomplete, attempt automatic setup (create virtualenv, install `requirements.txt`) where possible. If automatic setup cannot be performed, present clear, step-by-step instructions and one-click copyable commands in the GUI/CLI and logs.
   - Detect optional external tools (Blender) and record versions for reproducibility.
2. Model selection / filter source choice:
   - User selects a single STL model in the GUI or provides a single file path via CLI.
   - Present a choice: (A) Auto-detect profile and generate a suggested filter script, or (B) Use an existing filter script provided by the user (local file, pasted JSON/YAML, or URL/downloaded preset such as from Reddit).
   - If the user chooses (B) the tool loads and validates the provided filter script and skips automatic profile detection. The provided script is still shown for review and dry-run.
   - If the user chooses (A) continue to compute a diagnostics vector for the model (see `docs/model_profiles.md`).
3. Profile detection and suggested filter script generation:
   - Run the rule engine to match diagnostics to one or more model profiles and compute confidence scores.
   - Generate a suggested filter script tailored to the detected profile with metadata (model fingerprint, generator version, timestamp, reason/explanation).
4. Review and dry-run:
   - Present the suggested or provided filter script and diagnostics to the user with a short explanation of why steps were chosen and expected effect. Show the top alternative profiles if confidence is low.
   - Offer a `dry-run` option that simulates each filter action (no destructive writes) and reports intermediate diagnostics after each simulated step so the user can preview results.
5. Edit, save, and export presets:
   - Allow the user to edit the suggested filter script in the GUI or save it as a named preset (JSON/YAML). Presets include metadata (author, description, tags, preset_version).
   - Support import/export of presets for sharing and reproducibility.
6. Execute filter script actions:
   - Execute actions in the order defined by the filter script. For each action:
     - Run the registered action implementation (tool invocation or internal function).
     - Capture runtime, stdout/stderr, and diagnostics after the action.
     - If the action reports an error or diagnostics indicate worsening state beyond configurable thresholds, halt the script (or prompt user) and record failure details.
   - After the full script completes, run the validation checks.
7. Escalation and advanced repairs:
   - If validation still fails and Blender is available (or `--use-blender` set), run the Blender escalation pipeline for advanced remeshing/boolean cleanup and re-validate.
   - If escalation is disabled or fails, mark the model as failed and include detailed diagnostics and suggested manual remediation steps in the report.
8. Output, reporting, and reproducibility:
   - Export cleaned STL with deterministic filename pattern: `<origname>__<filtername>__<timestamp>.stl`.
   - Produce `report.json` (detailed per-step diagnostics, tool versions, commands, model fingerprint) and `report.csv` summary row for the run.
   - Offer `--export-run <dir>` to bundle input sample, filter script used, `report.json`, small before/after thumbnails, and `checkenv` output for sharing/reproducibility.
9. Logging and UI feedback:
   - Stream progress logs to the GUI console and save to a rotating logfile. Display clear error/warning messages and suggested next actions.
   - Provide a run summary UI showing success/failure, key diagnostics, runtime, and links to artifacts.
10. Iterate and contribute:
    - Allow users to save tuned presets and optionally contribute them (with metadata and tests) to the community presets repository (with PR workflow outlined in `CONTRIBUTING.md`).

Filter script representation
- Filter scripts are first-class JSON or YAML documents that describe an ordered list of filters (actions) to run against a single model. They are the primary user-editable unit (created, imported, exported, shared). The tool runs filters in a specific order using available repair tools to produce a 3D-printable result.

- Top-level structure (JSON example):
  {
    "name": "holes-only-suggested",
    "version": "1.0.0",
    "meta": {
      "generated_by": "model_scan",
      "model_fingerprint": "<sha256>",
      "generator_version": "0.1.0",
      "timestamp": "2025-01-01T12:00:00Z",
      "author": "auto",
      "description": "Suggested hole-filling filter script from model scan",
      "source": "local|url|community"
    },
    "actions": [
      { "id": "step-1", "name": "trimesh_basic", "params": {} },
      { "id": "step-2", "name": "fill_holes", "params": { "max_hole_size": 1000 } },
      { "id": "step-3", "name": "recalculate_normals", "params": {} },
      { "id": "step-4", "name": "validate", "params": {} }
    ]
  }

- Action model
  - `id` (optional): stable local id for diagnostics and per-step reporting.
  - `name`: action key that maps to a registered implementation (internal function, `trimesh` helper, `pymeshfix` wrapper, or external tool invocation).
  - `params`: dictionary of typed parameters for the action.
  - `timeout` (optional): override action timeout in seconds.
  - `on_error` (optional): policy for the action: `abort` (default), `skip`, or `continue`.

- Action registry
  - The driver maintains an action registry mapping `name` → implementation, documentation, supported params, and a stable version string. Implementations must declare whether they are "destructive" or safe for `dry-run` simulation.
  - New actions are added by registering them in the registry; filter scripts reference actions by `name` only.

- Metadata and provenance
  - Every filter script includes provenance metadata: `model_fingerprint`, `generator_version`, `generator_name`, `author`, `source`, and `timestamp`.
  - When a user imports a community preset (URL/Reddit), the tool records `source` and optionally a user-provided integrity hint (SHA256) and warns if preset is unsigned or from an untrusted source.

- Dry-run and simulation
  - The driver supports a `dry-run` mode that executes each action's non-destructive simulation path when available and collects intermediate diagnostics after each step. Filter script authors should mark actions as `dry_run_supported` when they provide a safe simulation.

- Validation and sandboxing
  - Imported filter scripts are validated against a JSON schema before execution: structure, action names, parameter types, and required fields.
  - Potentially unsafe actions (executing external binaries or network access) must be explicitly allowed by user consent in the GUI and will be flagged in reports.

- Determinism and reproducibility
  - Filter scripts should avoid non-deterministic operations; where randomness is needed, actions may accept a `seed` parameter. The driver records the environment (tool versions, platform, CLI args) and writes `report.json` to allow exact reproduction.

- Per-step reporting
  - For each action the run captures: start/end timestamp, runtime_ms, stdout/stderr, diagnostics snapshot (post-action), exit status, and error details if any. These entries populate `report.json` and the UI run summary.

- Versioning and compatibility
  - Filter scripts include a `version` field and the driver enforces compatibility with the action registry. If a script references an unknown or incompatible action version, execution fails with a clear message and suggested remediation (upgrade driver or adjust preset).

- Example action categories (initial set)
  - `trimesh_basic`: load + basic cleanup (merge vertices, remove degenerate faces)
  - `fill_holes`: fill small holes with configurable max size
  - `remove_small_components`: drop components below a volume or bounding-box threshold
  - `merge_vertices`: weld duplicates within `eps`
  - `recalculate_normals` / `reorient_normals` / `unify_normals`
  - `pymeshfix_repair`: run `pymeshfix` repair pass
  - `decimate` / `simplify` / `remesh` (may call Blender or internal remesher)
  - `separate_shells` / `boolean_union`
  - `identify_thin_regions` / `thicken_regions`
  - `validate`: run the full validation checks and produce diagnostics
  - `export`: write output STL with deterministic filename

- Security and community presets
  - Community presets must include metadata and an optional integrity SHA. The GUI will flag untrusted presets and require user confirmation before running.
  - Presets contributed to an official presets repo should follow a minimal review checklist (metadata, test fixture, compatibility notes) described in `CONTRIBUTING.md`.

- Storage and naming
  - Local filter scripts and presets are stored in the `filters/` directory.
  - Suggested filter scripts are written with `meta.source = "local"` and user templates may be saved with `author` and `tags`.

- Interoperability
  - Filter script format is intentionally simple JSON/YAML so third parties can generate presets (e.g., Reddit posts, bots). The driver provides an import helper to fetch a preset from a URL and validate it automatically.

- Backward-compatibility
  - Future driver versions must support older preset versions where reasonable; the driver emits warnings when using deprecated actions and suggests replacements.

Validation criteria
- `is_watertight` true
- No non-manifold edges
- Single large component (components below volume threshold removed)
- No self-intersections (best-effort check)

CLI

Command-line interface specification for `auto_fix_stl.py`:

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--input` | path | yes | — | Path to input STL file |
| `--output` | path | no | `./output/` | Directory for cleaned STL output |
| `--filter` | path | no | — | Path to a filter script (JSON/YAML) to use instead of auto-detection |
| `--preset` | string | no | — | Name of a preset from `filters/` to use |
| `--report` | path | no | `./report.json` | Path for JSON report output |
| `--csv` | path | no | `./report.csv` | Path for CSV report output |
| `--export-run` | path | no | — | Export reproducible run package to specified directory |
| `--use-blender` | choice | no | `on-failure` | When to use Blender escalation: `always`, `on-failure`, `never` |
| `--dry-run` | flag | no | false | Simulate filter actions without writing output |
| `--overwrite` | flag | no | false | Overwrite existing output files |
| `--verbose` | flag | no | false | Enable verbose logging |
| `--workers` | int | no | 1 | Number of parallel workers (reserved for future batch mode) |

Examples:
```bash
# Auto-detect profile and repair
python auto_fix_stl.py --input model.stl --output ./clean/

# Use a specific filter script
python auto_fix_stl.py --input model.stl --filter my_filter.json

# Use a named preset
python auto_fix_stl.py --input model.stl --preset holes-only

# Dry-run with verbose output
python auto_fix_stl.py --input model.stl --dry-run --verbose

# Export run package for sharing
python auto_fix_stl.py --input model.stl --export-run ./share/run1/
```

Artifacts
- Cleaned STL outputs
- `report.csv` with columns: filename, status, filter_script, attempts, time_ms, watertight_before, watertight_after, notes
- `report.json` detailed diagnostics per model and the generated/suggested filter script

Acceptance criteria
- For a curated set of difficult models, produce watertight outputs for >= 90% when using suggested and tuned presets.
- Produce logs and reports for all processed models.
- Preset sharing: at least two community-contributed presets with reproduce packages and pinned dependencies.

Testing
- Unit tests for helper functions (loading, basic cleanup, validation checks).
- Integration tests with sample STLs in `tests/fixtures/`.
- Each model profile should have at least one fixture for testing detection and filter script generation.

Implementation Plan (milestones)
1. Create GUI and CLI driver and basic `trimesh` + `pymeshfix` pipeline with model scanning and suggested filter script generation.
2. Add filter script config support and reporting.
3. Add Blender escalation script and detection.
4. Add unit and integration tests.
5. Document installation and usage in `docs/`.

Open issues / risks
- `pymeshfix` binary builds may be hard to install on some platforms.
- Very corrupted meshes may still fail and require manual intervention.
- Blender scripting can be complex to maintain across versions.

Next steps
- Confirm functional spec and tweak acceptance criteria.
- Start implementation: create `scripts/auto_fix_stl.py`, `filters/` directory with example presets, and test fixtures.

Model profiles
- We will include an initial set of model profiles (more than 10). The first release includes the ten core profiles listed in `docs/model_profiles.md`, and an expanded list of additional detectable profiles is documented there as well.

A new document `docs/model_profiles.md` explains each profile and how the system selects them automatically based on model diagnostics.

Collaborative & Sharing

Purpose
- Make it easy for the community to experiment with different filter scripts and share successful repair workflows (e.g. on Reddit).

Features
- Shareable filter scripts: store JSON/YAML presets in the `filters/` directory with metadata (author, description, tags, version).
- Reproducible run packages: an `--export-run <dir>` option bundles the input sample, filter script, `report.json`, and small before/after thumbnails so others can reproduce the run.
- Preset discovery: GUI and CLI support preset naming and metadata so presets can be associated with specific models and found externally.
- Standardized reports: include a short "how to reproduce" block with filter script name, pinned package versions (or Dockerfile), and commands used.
- Contribution workflow: require a `CONTRIBUTING.md` and PR template for adding presets (author, test case, and verification notes).

Why Blender remains optional
- Performance: Blender operations (remesh, booleans) are time-consuming; running Blender for every file would slow experimentation.
- Stability and portability: Blender scripting (`bpy`) can be brittle across versions and is heavier to install on CI or contributor machines.
- Fidelity: aggressive remeshing may change model detail; best used only when conservative repairs fail or when a preset explicitly requests it.

Recommended approach
- Keep Blender as an escalation step or as part of named aggressive presets (e.g. `aggressive-blender`) so contributors can opt-in when sharing presets.
- Provide pinned Blender versions in preset metadata or a Dockerfile to improve reproducibility.

Installation & Versioning

Purpose
- Provide an easy, up-to-date installation guide so new contributors and users can get started quickly.
- Maintain a clear, versioned environment for reproducibility and to help debug regressions as the tool evolves.

What to include in `docs/INSTALL.md` (summary)
- Quickstart: create virtualenv, `pip install -r requirements.txt`, example run command.
- Alternate install: `conda` environment instructions with exported `environment.yml` optional.
- Platform notes: Windows, macOS, Linux caveats and troubleshooting hints (e.g. common `pymeshfix` wheel issues).
- Optional tools: Blender install instructions, recommended Blender version(s), and how to verify `blender --version`.
- Docker: optional `Dockerfile` usage and how to run a reproducible containerized run.
- Troubleshooting: how to collect logs, attach `report.json`, `checkenv` output, and minimal repro files when reporting issues.

Versioning rules
- Maintain a `VERSION` file at repo root using Semantic Versioning (MAJOR.MINOR.PATCH), e.g. `0.1.0`.
- When dependencies or install steps change, update `requirements.txt`, bump `VERSION`, and add a short entry in `CHANGELOG.md`.
- Filter scripts must include a `preset_version` and either pinned dependencies or a Docker image tag to allow exact reproduction.

Environment validation tool
- Provide `scripts/checkenv.py` that prints installed package versions and checks for optional external tools (Blender). Include its output in exported run packages and CI logs.

Documentation hygiene & process
- Require PRs that change dependencies or install steps to update `docs/INSTALL.md`, `requirements.txt`, and bump `VERSION`.
- Add PR checklist in `CONTRIBUTING.md` to remind contributors to update installation docs and versioning when relevant.

Release process (brief)
- Tag releases (`vMAJOR.MINOR.PATCH`) and publish a release that includes `CHANGELOG.md` entries and the updated `VERSION` file.
- Optionally publish a Docker image with the same tag for reproducible environments.
