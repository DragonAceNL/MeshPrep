Functional Specification: Automated STL Cleanup Pipeline

Overview

Goal
- Provide a simple, robust, and reproducible automated pipeline that converts difficult or messy STL files into 3D-printable models with minimal or no manual intervention.
- Make the conversion workflow easy to use for non-technical Windows users via a lightweight GUI while retaining a powerful CLI for advanced users and automation.
- Make the tool extremely easy to set up and use: avoid requiring complex manual environment commands. If manual setup steps are required, present clear, step-by-step instructions in the GUI/CLI and logs. Where feasible implement automatic environment setup (virtualenv creation, dependency installation, basic tool detection) so users can get started with a single action.
- Support creation, editing, importing, exporting, and sharing of filter scripts and filter script presets so users can iterate and reuse successful repair strategies.
- Provide an extensive, intuitive filter script editor that makes it easy to build, modify, and understand repair workflows. The editor must include a comprehensive filter library containing all available filter actions from supported tools (`trimesh`, `pymeshfix`, `meshio`, and Blender) with clear descriptions of what each filter does, its parameters, and how it can be used to correct 3D models for printing.
- Ensure the system can handle hard-to-fix models by escalating to advanced steps (e.g., Blender-based remeshing) when conservative repairs fail.
- Surface errors, warnings, and important diagnostics clearly to the user (both on-screen and in log files) so runs are debuggable and reproducible.
- Prioritize conversion quality and reproducibility over raw speed; long-running but deterministic repairs are acceptable.
- Enable community sharing and discovery of filter script presets (e.g., Reddit), including metadata and reproduction instructions so a preset can be associated with a model and found by others.
- Automatically detect a model's profile (see `docs/model_profiles.md`) and generate a suggested filter script tailored to that profile; present the suggested filter script for review and editing before execution.

Scope
- Input: 3D mesh files in various formats. While the primary workflow targets STL files (ASCII or binary), MeshPrep accepts models in any format supported by `trimesh` or `meshio` and automatically converts them to STL for processing. The tool will scan the selected model to produce a suggested, generic filter script that fits the model's profile; users can review and tweak the suggested filter script before applying it. The software will not accept a directory of files as the primary workflow — each model is treated individually to allow per-model tuning and reproducible presets.
- Output: cleaned STL files suitable for slicing and a CSV/JSON report with diagnostics and the chosen filter script for the model.
- Tools: Python-based stack using `trimesh`, `pymeshfix`, `meshio`. External tools (Blender, slicers) are automatically detected and installed if missing — no manual setup required.

Supported Input Formats
-----------------------
MeshPrep leverages `trimesh` and `meshio` to support a wide range of 3D file formats. When a non-STL file is provided, it is automatically loaded and converted to an internal mesh representation for processing. The final output is always STL (the standard for 3D printing).

### Common 3D Printing Formats

| Format | Extensions | Tool | Notes |
|--------|------------|------|-------|
| **STL** | `.stl` | trimesh | Primary format (ASCII and binary) |
| **OBJ** | `.obj` | trimesh/meshio | Wavefront OBJ, widely supported |
| **PLY** | `.ply` | trimesh/meshio | Stanford PLY, good for scans |
| **3MF** | `.3mf` | trimesh | Modern 3D printing format (recommended over STL) |
| **GLTF/GLB** | `.gltf`, `.glb` | trimesh | Web 3D format, common from Sketchfab |
| **OFF** | `.off` | trimesh/meshio | Object File Format |
| **AMF** | `.amf` | trimesh | Additive Manufacturing Format |

### Compressed/Optimized Formats

| Format | Extensions | Tool | Notes |
|--------|------------|------|-------|
| **CTM** | `.ctm` | pymeshlab | OpenCTM compressed mesh (common on Sketchfab, CGTrader) |
| **Draco** | `.drc` | trimesh* | Google's compressed mesh format |

*Note: CTM is supported via PyMeshLab (MeshLab's Python bindings). Draco requires `trimesh[easy]` or DracoPy.

### CAD & Engineering Formats

| Format | Extensions | Tool | Notes |
|--------|------------|------|-------|
| **STEP** | `.step`, `.stp` | trimesh* | CAD interchange format (requires OpenCASCADE) |
| **IGES** | `.iges`, `.igs` | trimesh* | Legacy CAD format (requires OpenCASCADE) |
| **BREP** | `.brep` | trimesh* | Boundary representation (requires OpenCASCADE) |
| **VTK** | `.vtk`, `.vtu` | trimesh/meshio | Visualization Toolkit |
| **Gmsh** | `.msh` | meshio | Mesh generation tool |
| **NASTRAN** | `.nas`, `.bdf`, `.fem` | meshio | Finite element |
| **Abaqus** | `.inp` | meshio | Finite element |
| **EXODUS** | `.e`, `.ex2`, `.exo` | meshio | Sandia format |

*Note: STEP/IGES/BREP import requires optional OpenCASCADE dependencies (`pip install trimesh[easy]`).

### 3D Modeling Software Formats

| Format | Extensions | Tool | Notes |
|--------|------------|------|-------|
| **FBX** | `.fbx` | trimesh* | Autodesk exchange format |
| **DAE** | `.dae` | trimesh | COLLADA format |
| **3DS** | `.3ds` | trimesh | Legacy 3D Studio Max |
| **Blender** | `.blend` | blender | Requires Blender for conversion |

*Note: FBX requires Autodesk FBX SDK or conversion via Blender.

### Other Supported Formats

| Format | Extensions | Tool |
|--------|------------|------|
| DXF | `.dxf` | trimesh |
| SVG | `.svg` | trimesh/meshio |
| CGNS | `.cgns` | meshio |
| XDMF | `.xdmf`, `.xmf` | meshio |
| Tecplot | `.dat`, `.tec` | meshio |
| Netgen | `.vol` | meshio |
| TetGen | `.ele`, `.node` | meshio |
| UGRID | `.ugrid` | meshio |
| SU2 | `.su2` | meshio |
| MEDIT | `.mesh`, `.meshb` | meshio |

### Format-Specific Notes

#### CTM (OpenCTM)
CTM files are commonly found on 3D model marketplaces like Sketchfab and CGTrader. They offer excellent compression (often 10-20x smaller than STL) and are supported via PyMeshLab:

```bash
pip install pymeshlab
```

PyMeshLab provides the MeshLab engine as a Python library, which is the reference implementation for CTM and supports all compression modes (RAW, MG1, MG2).

#### STEP/IGES (CAD Files)
CAD formats like STEP and IGES are parametric and contain exact geometry (curves, surfaces). Converting to STL involves tessellation, which MeshPrep handles automatically. For best results:
- Higher tessellation = more triangles = smoother surfaces
- MeshPrep uses sensible defaults but allows overriding via `--cad-resolution`

#### 3MF (Recommended over STL)
3MF is a modern format that includes:
- Better precision than STL
- Units and scale information
- Color and texture support
- Multiple objects in one file

When possible, prefer 3MF over STL as input.

### Format Conversion Behavior

When a non-STL file is provided:

1. **Detect**: Format is detected from file extension (or magic bytes for ambiguous cases).
2. **Load**: The file is loaded using the appropriate library:
   - `trimesh` (preferred for mesh formats)
   - `meshio` (fallback, especially for FEM formats)
   - `openctm` (for CTM files)
   - Blender subprocess (for formats requiring conversion)
3. **Validate**: Basic validation ensures the mesh has vertices and faces.
4. **Convert**: The mesh is converted to trimesh's internal representation.
5. **Process**: Standard repair pipeline runs (same as STL input).
6. **Export**: Final output is always STL (binary by default).

**CLI Example:**
```bash
# Convert and fix an OBJ file
python auto_fix_stl.py --input model.obj --output ./clean/

# Convert and fix a PLY scan
python auto_fix_stl.py --input scan.ply --output ./clean/

# Convert GLTF to printable STL
python auto_fix_stl.py --input model.glb --output ./clean/

# Convert CTM from marketplace
python auto_fix_stl.py --input downloaded.ctm --output ./clean/

# Convert STEP file from CAD
python auto_fix_stl.py --input part.step --output ./clean/
```

**Report Metadata:**
The report includes the original format for traceability:
```json
{
  "input": {
    "filename": "model.ctm",
    "format": "ctm",
    "converted_from": "ctm",
    "original_vertices": 12450,
    "original_faces": 24896,
    "conversion_notes": "Loaded via PyMeshLab (MeshLab Python bindings)"
  }
}
```

### Installing Optional Format Support

For maximum format support, install optional dependencies:

```bash
# CTM support (via PyMeshLab - MeshLab Python bindings)
pip install pymeshlab

# CAD format support (STEP, IGES, BREP)
pip install trimesh[easy]  # Includes OpenCASCADE bindings

# Or install everything
pip install meshprep[all-formats]
```

Non-Goals
- Provide a full-featured cloud service, hosted web app, or serverless execution model (deployment to cloud is out of scope for the initial version).
- Implement exhaustive manufacturability, structural simulation, or advanced slicing optimization — only basic geometry validations (watertightness, manifoldness, component sanity) are required.
- Target mobile platforms or provide a native macOS/Linux GUI as the primary interface in the initial release (desktop Windows GUI is required; cross-platform CLI/automation support remains a goal).
- Guarantee that every possible corrupt mesh can be fixed; extremely damaged meshes may still require manual intervention and will be reported as failures with diagnostics.

Slicer Validation (Final Validation Step)
------------------------------------------
Slicer validation is the **recommended final validation step** that provides the definitive test for 3D printability. By default, every model processed by MeshPrep is validated through an actual slicer to ensure it is truly printable. This catches issues that geometry-only validation cannot detect.

**Design Philosophy**: MeshPrep prioritizes quality over speed. A model is not considered "fixed" until it successfully passes slicer validation. While this adds processing time, it guarantees that output models are actually printable. Users who have validated filter scripts can optionally skip slicer validation for faster batch processing.

**Slicer Validation Status in Reports**: When slicer validation is skipped, the output report and filter script metadata will clearly indicate `"slicer_validated": false`. This ensures transparency about the validation level of any exported model or shared filter script.

### Why Slicer Validation is Recommended

Geometry validation (watertight, manifold) is necessary but not sufficient for 3D printability:

| Check | Geometry Validation | Slicer Validation |
|-------|---------------------|-------------------|
| Watertight | ✅ `mesh.is_watertight` | ✅ Also checks |
| Manifold | ✅ `mesh.is_volume` | ✅ Also checks |
| **Thin walls** | ❌ Hard to detect | ✅ Warns about walls < nozzle width |
| **Overhangs** | ❌ Not checked | ✅ Calculates support requirements |
| **Bridging** | ❌ Not checked | ✅ Detects unsupported spans |
| **Printable size** | ❌ Basic bbox only | ✅ Checks against bed size |
| **Minimum feature size** | ❌ Difficult | ✅ Knows nozzle limitations |
| **Actually sliceable** | ❌ No guarantee | ✅ **Proves it works** |

### Supported Slicers

| Slicer | CLI Command | Notes |
|--------|-------------|-------|
| PrusaSlicer | `prusa-slicer --export-gcode` | Recommended, good error messages |
| SuperSlicer | `superslicer --export-gcode` | Fork of PrusaSlicer |
| OrcaSlicer | `orca-slicer --export-gcode` | Modern fork with extra features |
| Cura | `CuraEngine slice` | Requires printer definition |

### Slicer Issue Categories

The slicer reports issues in categories that map to available repair actions:

| Slicer Issue | Category | Potential Repair Actions |
|--------------|----------|-------------------------|
| "Thin walls detected" | `thin_walls` | `thicken_regions`, `blender_solidify`, `offset_surface` |
| "Non-manifold edges" | `non_manifold` | `pymeshfix_repair`, `fix_non_manifold_edges`, `blender_remesh` |
| "Self-intersecting faces" | `self_intersections` | `fix_self_intersections`, `blender_boolean_union` |
| "Open edges/holes" | `holes` | `fill_holes`, `cap_holes`, `pymeshfix_repair` |
| "Inverted normals" | `normals` | `fix_normals`, `flip_normals`, `reorient_normals` |
| "Model too large" | `size` | `scale` (user decision required) |
| "Disconnected parts" | `components` | `remove_small_components`, `boolean_union` |
| "Degenerate faces" | `degenerate` | `remove_degenerate_faces`, `trimesh_basic` |
| "Slicing failed" | `fatal` | Escalate to Blender remesh |

### Iterative Slicer-Driven Repair Loop

When slicer validation fails, MeshPrep attempts automatic repair using available filters:

```
┌─────────────────────────────────────────────────────────────────┐
│                    SLICER VALIDATION LOOP                       │
└─────────────────────────────────────────────────────────────────┘

1. Run Slicer Validation
         │
         ▼
    ┌─────────┐      YES
    │ Success?│─────────────► Done! Model is printable
    └────┬────┘
         │ NO
         ▼
2. Parse Slicer Errors/Warnings
         │
         ▼
3. Map Issues to Repair Actions
         │
         ▼
    ┌─────────────────┐      NO
    │ Actions available│─────────► Mark as FAILED
    │ for these issues?│          (include diagnostics + manual steps)
    └────────┬────────┘
             │ YES
             ▼
4. Execute Repair Actions
         │
         ▼
5. Run Geometry Validation
         │
         ▼
    ┌─────────┐      NO
    │ Still OK?│─────────► Revert and try next action
    └────┬────┘
         │ YES
         ▼
6. Increment attempt counter
         │
         ▼
    ┌─────────────────┐      YES
    │ Max attempts    │─────────► Mark as FAILED
    │ reached?        │          (exhausted all options)
    └────────┬────────┘
             │ NO
             ▼
        Go to Step 1
```

### Repair Strategy Selection

For each slicer issue, MeshPrep maintains a prioritized list of repair strategies:

```json
{
  "slicer_issue_mappings": {
    "thin_walls": {
      "strategies": [
        { "action": "offset_surface", "params": { "distance": 0.2 }, "priority": 1 },
        { "action": "blender_solidify", "params": { "thickness": 0.4 }, "priority": 2 },
        { "action": "thicken_regions", "params": { "target_thickness": 0.4 }, "priority": 3 }
      ],
      "max_attempts": 3
    },
    "non_manifold": {
      "strategies": [
        { "action": "pymeshfix_repair", "params": {}, "priority": 1 },
        { "action": "fix_non_manifold_edges", "params": {}, "priority": 2 },
        { "action": "blender_remesh", "params": { "voxel_size": 0.05 }, "priority": 3 }
      ],
      "max_attempts": 3
    },
    "holes": {
      "strategies": [
        { "action": "fill_holes", "params": { "max_hole_size": 100 }, "priority": 1 },
        { "action": "fill_holes", "params": { "max_hole_size": 1000 }, "priority": 2 },
        { "action": "pymeshfix_repair", "params": {}, "priority": 3 },
        { "action": "blender_remesh", "params": { "voxel_size": 0.05 }, "priority": 4 }
      ],
      "max_attempts": 4
    },
    "self_intersections": {
      "strategies": [
        { "action": "fix_self_intersections", "params": {}, "priority": 1 },
        { "action": "blender_boolean_union", "params": {}, "priority": 2 },
        { "action": "blender_remesh", "params": { "voxel_size": 0.05 }, "priority": 3 }
      ],
      "max_attempts": 3
    }
  },
  "global_max_attempts": 10,
  "escalate_to_blender_after": 5
}
```

### Slicer Validation Actions

#### Category: Slicer Validation

| Action | Tool | Description | Parameters |
|--------|------|-------------|------------|
| `slicer_validate` | slicer | Run slicer validation and collect diagnostics. | `slicer` (string: "prusa", "orca", "cura"), `config` (path to printer profile) |
| `slicer_check_printable` | slicer | Quick check if model slices without errors. | `slicer` (string), `timeout` (int, seconds) |
| `slicer_get_warnings` | slicer | Extract warnings and issues from slicer output. | `slicer` (string) |
| `slicer_estimate` | slicer | Get print time/filament estimates (implies successful slice). | `slicer` (string), `config` (path) |

### CLI Options for Slicer Validation

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--slicer` | choice | no | `auto` | Slicer to use: `prusa`, `orca`, `cura`, `auto` (detect first available) |
| `--slicer-config` | path | no | — | Path to slicer printer/profile config |
| `--slicer-repair` | choice | no | `auto` | Slicer-driven repair mode: `auto`, `prompt`, `never` |
| `--max-repair-attempts` | int | no | `10` | Maximum repair attempts in slicer loop |
| `--skip-slicer-validation` | flag | no | false | Skip slicer validation (only use with trusted filter scripts) |
| `--trust-filter-script` | flag | no | false | Trust the filter script and skip slicer validation (alias for --skip-slicer-validation) |

**Note**: Slicer validation is **always enabled by default**. The `--skip-slicer-validation` flag should only be used when applying a known-good filter script that has already been validated.

### Validation Levels

MeshPrep uses a three-level validation system. **Level 3 (Slicer Validated) is the default and required for all output**:

| Level | Checks | Confidence | Speed | When to Use |
|-------|--------|------------|-------|-------------|
| **1. Basic** | `is_watertight`, `is_volume` | ~80% | Fast (ms) | Internal checkpoints only |
| **2. Full Geometry** | + consistent normals, no self-intersections | ~90% | Medium (s) | Internal checkpoints only |
| **3. Slicer Validated** | + successful slicer pass | **~99%** | Slow (s-min) | **Default for all output** |

**Important**: Levels 1 and 2 are used as internal checkpoints during the repair process. The final output is always validated at Level 3 unless explicitly skipped with `--skip-slicer-validation` (only recommended when using a proven filter script).

### When Slicer Validation Can Be Skipped

Slicer validation can be skipped **only** in these scenarios:

1. **Using a verified filter script**: When applying a filter script that has been previously validated and shared by the community, users can skip re-validation for faster processing.
2. **Batch processing with known-good presets**: When processing many similar models with a proven preset.
3. **Development/testing**: When iterating on filter scripts during development.

To skip slicer validation, use: `--skip-slicer-validation` or `--trust-filter-script`

**Warning**: Skipping slicer validation means the output is not guaranteed to be printable. The report will clearly indicate that slicer validation was skipped.

### Reporting Slicer Results

Slicer validation results are included in `report.json`:

```json
{
  "slicer_validation": {
    "slicer": "prusa-slicer",
    "slicer_version": "2.7.0",
    "success": true,
    "attempts": 3,
    "issues_found": [
      { "type": "thin_walls", "message": "Wall thickness below 0.4mm", "resolved": true },
      { "type": "non_manifold", "message": "2 non-manifold edges", "resolved": true }
    ],
    "repairs_applied": [
      { "action": "offset_surface", "params": { "distance": 0.2 }, "attempt": 1 },
      { "action": "pymeshfix_repair", "params": {}, "attempt": 2 }
    ],
    "final_status": "printable",
    "estimates": {
      "print_time_minutes": 127,
      "filament_grams": 34.5,
      "layer_count": 412
    }
  }
}
```

### Failure Handling

When slicer validation fails after exhausting all repair options:

1. **Mark model as FAILED** with detailed diagnostics
2. **List unresolved issues** from slicer
3. **Suggest manual remediation steps**:
   - Which issues remain and why they couldn't be auto-fixed
   - Recommended manual edits in mesh editing software (Blender, Meshmixer)
   - Links to relevant documentation/tutorials
4. **Save intermediate artifacts**:
   - Best attempt mesh (closest to printable)
   - Full repair history
   - Slicer error logs

### Sharing Validated Filter Scripts

Once a filter script has been validated through the slicer validation process, it can be shared with the community. This allows others to:

1. **Skip the validation process**: Use `--trust-filter-script` to apply the proven filter script without re-running slicer validation.
2. **Save processing time**: Avoid the iterative repair loop for similar models.
3. **Build on proven solutions**: Start from a working filter script and customize as needed.

**Filter Script Validation Status**:

Filter scripts include a `validation` section in their metadata:

```json
{
  "name": "multi-component-fix",
  "version": "1.0.0",
  "meta": {
    "author": "community_user",
    "description": "Fixes multi-component models with holes",
    "slicer_validated": true,
    "validation_details": {
      "slicer": "prusa-slicer",
      "slicer_version": "2.7.0",
      "validated_on": "2025-01-15T10:30:00Z",
      "test_models": ["model1.stl", "model2.stl"],
      "success_rate": "100%"
    }
  },
  "actions": [...]
}
```

**When slicer validation is skipped**, the filter script will show:

```json
{
  "meta": {
    "slicer_validated": false,
    "validation_note": "Slicer validation was skipped. Print success not guaranteed."
  }
}
```

**Trust Levels**:

| Trust Level | Description | Can Skip Slicer Validation? |
|-------------|-------------|-----------------------------|
| `unvalidated` | New or modified filter script | No |
| `self-validated` | Validated by the author | Yes (with warning) |
| `community-validated` | Validated by multiple users | Yes |
| `official` | Part of MeshPrep's built-in presets | Yes |

### Future: Advanced Slicer Integration

The slicer validation framework is designed to support future enhancements:

1. **Validate-only integration** (Current): Run slicer to collect diagnostics and iterate repairs.
2. **Slice-and-validate**: Produce pinned-version G-code and parse layer/preview data for programmatic checks.
3. **Optimize-and-slice** (opt-in): Apply geometry transformations driven by slicer feedback (orientation, splits, thickening) and re-validate; require explicit user consent and provide diffs.

Requirements

Functional Requirements
1. Per-model processing: accept exactly one input STL file; automatically scan the model to produce a suggested generic filter script that matches the model profile. Allow the user to edit, save, export, and re-use that filter script as a preset.
2. Validation checks: watertightness, manifoldness, consistent normals, component count, and bounding box sanity.
3. Repair steps: remove degenerate faces, merge duplicate vertices, reorient normals, fill holes, remove tiny disconnected components.
4. Escalation: if primary repairs fail, run an advanced Blender-based pipeline (if Blender present) with remeshing and boolean cleanup.
5. **Slicer validation (strongly recommended)**: run the repaired model through a slicer (PrusaSlicer, OrcaSlicer, etc.) to verify it is truly printable. Parse slicer errors/warnings and attempt automatic repairs using available filter actions until the model passes or no repair options remain. Slicer validation is enabled by default and strongly recommended. When skipped, the report and filter script metadata will indicate `"slicer_validated": false`.
6. Configurable filter scripts: filter scripts defined in JSON/YAML and selectable via GUI/CLI; suggested filter scripts from model scan may be created automatically.
7. Reporting: generate CSV and JSON reports detailing diagnostics, filter script attempts, slicer validation results, runtime, and final status for the model.
8. Deterministic filenames: output files named with original name + filter script suffix.
9. Logging: progress logs with per-file detail and error handling.

Non-Functional Requirements
- Reproducibility: deterministic behavior given same inputs and filter script config.
- Extensibility: easy to add new filter actions or replace tools.
- Portability: runs on major OSs (Windows, macOS, Linux) with documented dependencies.
- Stability: handle corrupted files without crashing; log failures and continue.

Design

High-level flow
1. Startup / environment prep:
   - Run `checkenv` to detect required tools and Python dependencies.
   - If the environment is incomplete, attempt automatic setup (create virtualenv, install `requirements.txt`) where possible.
   - **Detect external tools (Blender, slicers)**: search configured paths, MeshPrep tools directory, system PATH, and standard installation locations.
   - **Offer automatic installation**: if Blender or a slicer is not found, prompt the user to install it automatically to the MeshPrep tools directory.
   - Record all tool versions for reproducibility.
   - If automatic setup cannot be performed, present clear, step-by-step instructions and one-click copyable commands in the GUI/CLI and logs.
2. Model selection / filter source choice:
   - User selects a single STL model in the GUI or provides a single file path via CLI.
   - Present a choice: (A) Auto-detect profile and generate a suggested filter script, or (B) Use an existing filter script provided by the user (local file, pasted JSON/YAML, or URL/downloaded preset such as from Reddit).
   - If the user chooses (B) the tool loads and validates the provided filter script and skips automatic profile detection. The provided script is still shown for review and dry-run.
   - If the user chooses (A) continue to compute a diagnostics vector for the model (see `docs/model_profiles.md`).
3. Profile detection and suggested filter script generation:
   - Run the rule engine to match diagnostics to one or more model profiles and compute confidence scores.
   - Generate a suggested filter script tailored to the detected profile with metadata (model fingerprint, generator version, timestamp, reason/explanation).
4. Review:
   - Present the suggested or provided filter script and diagnostics to the user with a short explanation of why steps were chosen and expected effect. Show the top alternative profiles if confidence is low.
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
   - If escalation is disabled or fails, continue to slicer validation (which may trigger additional repairs).
8. Slicer validation and iterative repair:
   - If a slicer is available and `--slicer` is not `none`, run slicer validation on the repaired model.
   - Parse slicer output for errors and warnings; categorize issues (thin walls, holes, non-manifold, etc.).
   - For each issue category, look up available repair strategies in the slicer issue mappings.
   - If repair actions are available:
     a. Execute the highest-priority untried action for each issue.
     b. Re-run geometry validation to ensure the repair didn't break the mesh.
     c. If geometry is still valid, re-run slicer validation.
     d. Repeat until slicer passes or max attempts reached or no actions remain.
   - If no repair actions are available or all have been exhausted, mark the model as failed and include detailed diagnostics with suggested manual remediation steps.
   - Record all repair attempts, slicer output, and final status in the report.
9. Output, reporting, and reproducibility:
   - Export cleaned STL with deterministic filename pattern: `<origname>__<filtername>__<timestamp>.stl`.
   - Produce `report.json` (detailed per-step diagnostics, tool versions, commands, model fingerprint) and `report.csv` summary row for the run.
   - Offer `--export-run <dir>` to bundle input sample, filter script used, `report.json`, small before/after thumbnails, and `checkenv` output for sharing/reproducibility.
10. Logging and UI feedback:
   - Stream progress logs to the GUI console and save to a rotating logfile. Display clear error/warning messages and suggested next actions.
   - Provide a run summary UI showing success/failure, key diagnostics, runtime, and links to artifacts.
11. Iterate and contribute:
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

Filter Script Editor
--------------------
The filter script editor is a core component of the GUI that enables users to build, modify, and understand repair workflows intuitively.

### Editor Features

- **Visual action list**: Display actions as draggable cards or list items with clear icons, names, and parameter summaries.
- **Drag-and-drop reordering**: Reorder actions by dragging; visual feedback shows where the action will be inserted.
- **Inline parameter editing**: Click an action to expand and edit its parameters with appropriate input controls (sliders, dropdowns, number fields).
- **Real-time validation**: Highlight invalid parameters or incompatible action sequences immediately.
- **Action search and filter**: Search the filter library by name, category, or keyword; filter by tool source (trimesh, pymeshfix, meshio, Blender).
- **Tooltips and documentation**: Hover over any action or parameter to see a tooltip with description; link to full documentation.
- **Undo/redo**: Support undo/redo for all editor operations.
- **Duplicate and delete**: Quickly duplicate or remove actions.
- **Collapse/expand**: Collapse actions to show only names for a compact overview; expand to edit.

### Filter Library

The filter library is a comprehensive catalog of all available filter actions, organized by category and tool source. Each entry includes:

- **Name**: The action key used in filter scripts (e.g., `fill_holes`).
- **Display name**: Human-readable name (e.g., "Fill Holes").
- **Category**: Grouping for organization (e.g., Cleanup, Repair, Validation, Export).
- **Tool source**: Which tool provides this action (`trimesh`, `pymeshfix`, `meshio`, `blender`, or `internal`).
- **Description**: What the action does and when to use it.
- **Parameters**: List of parameters with types, defaults, and descriptions.
- **Use cases**: Common scenarios where this action is helpful.


### Filter Library — Action Catalog

The following is the initial catalog of filter actions available in MeshPrep, grouped by category.

#### Category: Loading & Basic Cleanup

| Action | Tool | Description | Parameters |
|--------|------|-------------|------------|
| `load_mesh` | trimesh/meshio | Load a 3D mesh file in any supported format (STL, OBJ, PLY, 3MF, GLTF, etc.). Automatically detects format from extension. | `path` (string) |
| `load_stl` | trimesh | Load an STL file (ASCII or binary) into memory. | `path` (string) |
| `trimesh_basic` | trimesh | Load and apply basic cleanup: merge duplicate vertices, remove degenerate faces, remove infinite values. | `merge_tex` (bool), `merge_norm` (bool) |
| `merge_vertices` | trimesh | Weld duplicate vertices within a tolerance. Reduces vertex count and fixes small gaps. | `eps` (float, default 1e-8) |
| `remove_degenerate_faces` | trimesh | Remove faces with zero area or invalid topology. | `height` (float, threshold) |
| `remove_duplicate_faces` | trimesh | Remove faces that are exact duplicates. | — |
| `remove_infinite_values` | trimesh | Remove vertices or faces containing NaN or Inf values. | — |
| `remove_unreferenced_vertices` | trimesh | Remove vertices not referenced by any face. | — |

#### Category: Hole Filling

| Action | Tool | Description | Parameters |
|--------|------|-------------|------------|
| `fill_holes` | trimesh | Fill holes in the mesh up to a maximum size. Useful for closing small gaps. | `max_hole_size` (int, max edges), `method` (string: "fan", "ear") |
| `fill_holes_pymeshfix` | pymeshfix | Use pymeshfix's hole-filling algorithm for more robust repairs. | — |
| `cap_holes` | trimesh | Cap open boundaries with flat faces. Best for planar openings. | — |

#### Category: Normal Correction

| Action | Tool | Description | Parameters |
|--------|------|-------------|------------|
| `recalculate_normals` | trimesh | Recompute face normals from vertex winding order. | — |
| `reorient_normals` | trimesh | Attempt to make all face normals point outward consistently. | — |
| `unify_normals` | trimesh | Unify normals so adjacent faces have consistent orientation. | — |
| `fix_normals` | trimesh | Combine recalculate and reorient for a single-step fix. | — |
| `flip_normals` | trimesh | Invert all face normals (useful if model is inside-out). | — |

#### Category: Component Management

| Action | Tool | Description | Parameters |
|--------|------|-------------|------------|
| `remove_small_components` | trimesh | Remove disconnected components below a volume or face-count threshold. | `min_volume` (float), `min_faces` (int) |
| `keep_largest_component` | trimesh | Keep only the largest connected component; remove all others. | — |
| `separate_shells` | trimesh | Split mesh into separate shell components. | — |
| `boolean_union` | trimesh/blender | Merge overlapping shells into a single watertight mesh. | `engine` (string: "blender", "manifold") |
| `remove_internal_geometry` | trimesh | Remove components fully enclosed by the outer shell. | — |

#### Category: Repair & Manifold Fixes

| Action | Tool | Description | Parameters |
|--------|------|-------------|------------|
| `pymeshfix_repair` | pymeshfix | Run pymeshfix's automatic repair pass. Fixes many non-manifold issues. | — |
| `fix_non_manifold_edges` | trimesh | Attempt to fix non-manifold edges by splitting or removing. | — |
| `fix_non_manifold_vertices` | trimesh | Fix vertices shared by non-adjacent faces. | — |
| `stitch_boundaries` | trimesh | Stitch open boundaries where edges nearly match. | `tolerance` (float) |
| `close_cracks` | trimesh | Close small cracks by merging nearby vertices along boundaries. | `tolerance` (float) |

#### Category: Simplification & Remeshing

| Action | Tool | Description | Parameters |
|--------|------|-------------|------------|
| `decimate` | trimesh | Reduce face count while preserving shape. | `target_faces` (int), `target_ratio` (float) |
| `simplify_quadric` | trimesh | Quadric error decimation for high-quality simplification. | `target_faces` (int), `agg` (float) |
| `subdivide` | trimesh | Subdivide faces to increase mesh resolution. | `iterations` (int) |
| `remesh_blender` | blender | Use Blender's remesh modifier for uniform triangle distribution. | `voxel_size` (float), `mode` (string: "voxel", "quad") |
| `smooth_laplacian` | trimesh | Apply Laplacian smoothing to reduce noise. | `iterations` (int), `lamb` (float) |
| `smooth_taubin` | trimesh | Taubin smoothing (reduces shrinkage compared to Laplacian). | `iterations` (int), `lamb` (float), `mu` (float) |

#### Category: Geometry Analysis & Thin Features

| Action | Tool | Description | Parameters |
|--------|------|-------------|------------|
| `identify_thin_regions` | trimesh | Detect regions thinner than a threshold; report but do not modify. | `min_thickness` (float) |
| `thicken_regions` | blender | Thicken thin walls to meet a minimum printable thickness. | `target_thickness` (float) |
| `offset_surface` | trimesh | Offset the mesh surface inward or outward. | `distance` (float) |
| `hollow` | blender | Create a hollow shell with specified wall thickness. | `wall_thickness` (float) |

#### Category: Boolean & Intersection Fixes

| Action | Tool | Description | Parameters |
|--------|------|-------------|------------|
| `detect_self_intersections` | trimesh | Check for self-intersecting faces; report but do not modify. | — |
| `fix_self_intersections` | blender | Attempt to resolve self-intersections using boolean operations. | — |
| `boolean_difference` | blender | Subtract one mesh from another. | `tool_mesh` (path) |
| `boolean_intersect` | blender | Keep only the intersection of two meshes. | `tool_mesh` (path) |

#### Category: Validation & Diagnostics

| Action | Tool | Description | Parameters |
|--------|------|-------------|------------|
| `validate` | internal | Run all validation checks and produce a diagnostics report. | — |
| `check_watertight` | trimesh | Check if mesh is watertight (closed, no holes). | — |
| `check_manifold` | trimesh | Check for non-manifold edges and vertices. | — |
| `check_normals` | trimesh | Check for inconsistent or inverted normals. | — |
| `check_volume` | trimesh | Compute and report mesh volume. | — |
| `check_bounding_box` | trimesh | Report bounding box dimensions. | — |
| `compute_diagnostics` | internal | Compute full diagnostics vector for profile detection. | — |

#### Category: Export

| Action | Tool | Description | Parameters |
|--------|------|-------------|------------|
| `export_stl` | trimesh | Export mesh to STL file (binary by default). | `path` (string), `ascii` (bool) |
| `export_stl_ascii` | trimesh | Export mesh to ASCII STL file. | `path` (string) |
| `export_obj` | trimesh | Export mesh to OBJ format. | `path` (string) |
| `export_ply` | trimesh | Export mesh to PLY format. | `path` (string) |
| `export_3mf` | meshio | Export mesh to 3MF format (preferred for 3D printing). | `path` (string) |

#### Category: Blender-Specific (Escalation)

| Action | Tool | Description | Parameters |
|--------|------|-------------|------------|
| `blender_remesh` | blender | Apply Blender's voxel remesh for aggressive topology repair. | `voxel_size` (float), `adaptivity` (float) |
| `blender_decimate` | blender | Blender's decimate modifier. | `ratio` (float), `use_collapse` (bool) |
| `blender_boolean_union` | blender | Merge all mesh parts using Blender's boolean solver. | — |
| `blender_solidify` | blender | Add thickness to thin surfaces using the solidify modifier. | `thickness` (float) |
| `blender_smooth` | blender | Apply Blender's smooth modifier. | `iterations` (int), `factor` (float) |
| `blender_triangulate` | blender | Triangulate all faces (ensure STL compatibility). | — |

### Editor UX Guidelines

1. **Category sidebar**: Show filter library categories in a collapsible sidebar; clicking a category shows available actions.
2. **Drag from library**: Drag an action from the library directly into the action list at the desired position.
3. **Quick add**: Double-click an action in the library to append it to the end of the list.
4. **Parameter forms**: When an action is selected, show a parameter form with:
   - Input type matching the parameter type (slider for floats with range, checkbox for bools, dropdown for enums).
   - Default value pre-filled; reset button to restore defaults.
   - Validation feedback (red border for invalid values).
5. **Dependency hints**: If an action depends on another (e.g., `export_stl` requires a loaded mesh), show a hint or auto-insert the dependency.
7. **Template presets**: Offer one-click insertion of common action sequences (e.g., "Basic Cleanup", "Hole Fill + Normals", "Aggressive Repair").

### Filter Library Data File

The filter library is stored as a JSON file at `config/filter_library.json` with the following structure:

```json
{
  "version": "1.0.0",
  "categories": [
    {
      "id": "cleanup",
      "name": "Loading & Basic Cleanup",
      "actions": [
        {
          "name": "merge_vertices",
          "display_name": "Merge Vertices",
          "tool": "trimesh",
          "description": "Weld duplicate vertices within a tolerance.",
          "parameters": [
            {
              "name": "eps",
              "type": "float",
              "default": 1e-8,
              "description": "Distance threshold for merging."
            }
          ],

        }
      ]
    }
  ]
}
```

This file is loaded at startup and used to populate the editor's filter library UI and validate filter scripts.

Validation criteria

MeshPrep uses a three-level validation system:

**Level 1 - Basic Geometry (Required)**
- `is_watertight` true
- `is_volume` true (manifold)

**Level 2 - Full Geometry (Default)**
- All Level 1 checks
- No non-manifold edges
- Consistent winding/normals
- Single large component (components below volume threshold removed)
- No self-intersections (best-effort check)

**Level 3 - Slicer Validated (Recommended for final output)**
- All Level 2 checks
- Successful slicer pass with no errors
- No critical warnings (thin walls, unsupported features)
- Print time/filament estimates available

CLI

Command-line interface specification for `auto_fix_stl.py`:

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--input` | path | yes | — | Path to input mesh file (STL, OBJ, PLY, 3MF, GLTF, or any supported format) |
| `--output` | path | no | `./output/` | Directory for cleaned STL output |
| `--filter` | path | no | — | Path to a filter script (JSON/YAML) to use instead of auto-detection |
| `--preset` | string | no | — | Name of a preset from `filters/` to use |
| `--report` | path | no | `./report.json` | Path for JSON report output |
| `--csv` | path | no | `./report.csv` | Path for CSV report output |
| `--export-run` | path | no | — | Export reproducible run package to specified directory |
| `--use-blender` | choice | no | `on-failure` | When to use Blender escalation: `always`, `on-failure`, `never` |
| `--slicer` | choice | no | `auto` | Slicer to use: `prusa`, `orca`, `cura`, `auto` (detect first available) |
| `--slicer-config` | path | no | — | Path to slicer printer/profile config |
| `--slicer-repair` | choice | no | `auto` | Slicer-driven repair mode: `auto`, `prompt`, `never` |
| `--max-repair-attempts` | int | no | `10` | Maximum repair attempts in slicer validation loop |
| `--skip-slicer-validation` | flag | no | false | Skip slicer validation (only use with trusted filter scripts) |
| `--trust-filter-script` | flag | no | false | Trust the filter script and skip slicer validation |

| `--overwrite` | flag | no | false | Overwrite existing output files |
| `--verbose` | flag | no | false | Enable verbose logging |
| `--cad-resolution` | float | no | `0.01` | Tessellation resolution for CAD formats (STEP, IGES). Lower = finer mesh |
| `--workers` | int | no | 1 | Number of parallel workers (reserved for future batch mode) |

Examples:
```bash
# Auto-detect profile and repair (includes slicer validation by default)
python auto_fix_stl.py --input model.stl --output ./clean/

# Use a specific filter script (still validates with slicer)
python auto_fix_stl.py --input model.stl --filter my_filter.json

# Use a named preset (still validates with slicer)
python auto_fix_stl.py --input model.stl --preset holes-only

# Use PrusaSlicer specifically for validation
python auto_fix_stl.py --input model.stl --slicer prusa

# Allow more repair attempts before giving up
python auto_fix_stl.py --input model.stl --max-repair-attempts 15

# Skip slicer validation when using a trusted/proven filter script (faster)
python auto_fix_stl.py --input model.stl --preset community-verified --trust-filter-script

# Batch processing with a known-good preset (skip validation for speed)
python auto_fix_stl.py --input model.stl --preset proven-preset --skip-slicer-validation

# Export run package for sharing (includes full slicer validation)
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

Model Fingerprints
------------------

Every model file gets a unique, searchable **fingerprint** that enables community sharing and discovery of filter scripts.

### How It Works

1. **Open a model** in MeshPrep (STL, CTM, OBJ, etc.)
2. **MeshPrep displays the fingerprint**, e.g., `MP:42f3729aa758`
3. **Search for the fingerprint** on Reddit/Google to find existing filter scripts
4. **If you create a working filter script**, share it with the fingerprint so others can find it

### Fingerprint Format

```
MP:xxxxxxxxxxxx
```

- `MP:` = MeshPrep prefix (makes it searchable and identifiable)
- `xxxxxxxxxxxx` = 12 hex characters from SHA256 hash of the original file

### Key Design Decisions

- **Fingerprint is computed from the ORIGINAL FILE BYTES**, not the loaded mesh data
- **CTM files are fingerprinted as CTM** (compressed bytes), not as decompressed geometry
- **Same file download = same fingerprint** (exact matching)
- This ensures that if two people download the same model from CGTrader/Sketchfab, they get the same fingerprint

### Example Workflow

```
1. Download "spaceship.ctm" from CGTrader
2. Open in MeshPrep → Fingerprint: MP:42f3729aa758
3. Search "MP:42f3729aa758" on Reddit
4. Find post: "[MeshPrep Filter] MP:42f3729aa758 (spaceship.ctm) - fixes holes and normals"
5. Download and apply the filter script
6. Model is fixed!
```

### Sharing on Reddit

When sharing a filter script, use this format:

**Post title:**
```
[MeshPrep Filter] MP:42f3729aa758 (spaceship.ctm) - fixes holes and non-manifold edges
```

**Post body:**
```json
{
  "model_fingerprint": "MP:42f3729aa758",
  "original_filename": "spaceship.ctm",
  "filter_name": "slicer-repair-loop",
  "actions": [...]
}
```

### Filter Script Metadata

Every saved filter script includes the fingerprint and MeshPrep URL:

```json
{
  "model_id": "spaceship",
  "model_fingerprint": "MP:42f3729aa758",
  "original_filename": "spaceship.ctm",
  "original_format": "ctm",
  "filter_name": "slicer-repair-loop",
  "meshprep_version": "0.1.0",
  "meshprep_url": "https://github.com/DragonAceNL/MeshPrep",
  "sharing_note": "Search 'MP:42f3729aa758' on Reddit to find/share filter scripts"
}
```

### CLI: Display Fingerprint

```bash
# Show fingerprint for a model
python auto_fix_stl.py --fingerprint model.ctm
# Output: MP:42f3729aa758

# Search for filter scripts
python auto_fix_stl.py --search-filter model.ctm
# Opens browser to search for MP:42f3729aa758 on Reddit
```

Features
- Shareable filter scripts: store JSON/YAML presets in the `filters/` directory with metadata (author, description, tags, version, **model fingerprint**).
- Reproducible run packages: an `--export-run <dir>` option bundles the input sample, filter script, `report.json`, and small before/after thumbnails so others can reproduce the run.
- Preset discovery: GUI and CLI support preset naming and metadata so presets can be associated with specific models and found externally via fingerprint search.
- Standardized reports: include a short "how to reproduce" block with filter script name, pinned package versions (or Dockerfile), and commands used.
- Contribution workflow: require a `CONTRIBUTING.md` and PR template for adding presets (author, test case, and verification notes).

Why Blender and Slicers are Automatically Managed
- **Zero-friction setup**: Users can start using MeshPrep immediately without hunting for downloads or configuring paths.
- **Reproducibility**: Pinned versions ensure consistent behavior across runs and machines.
- **Isolation**: MeshPrep-managed installations don't conflict with existing system installations.
- **Reliability over space**: Disk space is cheap; user time and frustration are not. Installing tools automatically prioritizes a working experience.

When External Tools are Used
- **Blender**: Used for escalation when conservative repairs fail, or when a preset explicitly requests Blender operations (remesh, booleans, solidify). Not used for every file.
- **Slicers**: Used for final validation of repaired models. Strongly recommended to ensure output is truly printable. Can be skipped with `--trust-filter-script` when using proven presets.

Recommended Configuration
- Install both Blender and PrusaSlicer (~600 MB total) for full functionality.
- Users with disk constraints can disable auto-install and use system-installed tools.
- Provide pinned tool versions in preset metadata for reproducibility.

Installation & Versioning

Purpose
- Provide an easy, up-to-date installation guide so new contributors and users can get started quickly.
- Maintain a clear, versioned environment for reproducibility and to help debug regressions as the tool evolves.
- **Automatically install and manage external tools** (Blender, slicers) so users don't need to manually configure their environment.

Python Version Requirement
- **Python 3.11 or 3.12 is required** for MeshPrep.
- Python 3.13+ is not currently supported because `pymeshfix` does not have pre-built wheels for these versions.
- When pymeshfix releases wheels for newer Python versions, this constraint will be relaxed.
- The `pyproject.toml` enforces this constraint: `requires-python = ">=3.11,<3.13"`

External Tools Management
--------------------------

### Design Philosophy

MeshPrep prioritizes **reliability and simplicity over disk space savings**. External tools (Blender, slicers) are:

1. **Automatically detected** in standard installation locations and PATH
2. **Automatically installed** to a MeshPrep-managed directory if not found
3. **Version-pinned** for reproducibility
4. **Isolated** from system installations to avoid conflicts

This "just works" approach ensures users can start using MeshPrep immediately without manual setup steps.

### MeshPrep Tools Directory

External tools are installed to a dedicated directory:

| Platform | Default Location |
|----------|------------------|
| Windows | `%LOCALAPPDATA%\MeshPrep\tools\` |
| macOS | `~/Library/Application Support/MeshPrep/tools/` |
| Linux | `~/.local/share/meshprep/tools/` |

Directory structure:
```
MeshPrep/tools/
├── blender/
│   └── blender-4.2.0/           # Pinned Blender version
├── slicers/
│   ├── prusaslicer-2.8.0/       # PrusaSlicer (primary)
│   ├── orcaslicer-2.2.0/        # OrcaSlicer (alternative)
│   └── superslicer-2.5.0/       # SuperSlicer (alternative)
├── config/
│   └── tools.json               # Tool paths and versions
└── downloads/                   # Cached installers
```

### Tool Detection Order

When MeshPrep needs an external tool, it searches in this order:

1. **Explicit configuration** (`config/tools.json` or environment variable)
2. **MeshPrep-managed installation** (`tools/` directory)
3. **System PATH**
4. **Standard installation locations** (platform-specific)

If not found, MeshPrep offers to install the tool automatically.

### Supported External Tools

#### Blender

| Property | Value |
|----------|-------|
| Purpose | Advanced mesh repair (remesh, booleans, solidify) |
| Pinned Version | 4.2.0 LTS (or latest LTS) |
| Required | No (escalation only) |
| Size | ~400 MB (portable) |
| Install Method | Download portable ZIP/tar.xz |

**Detection locations (Windows):**
- `%LOCALAPPDATA%\MeshPrep\tools\blender\`
- `C:\Program Files\Blender Foundation\Blender *\`
- `C:\Program Files (x86)\Blender Foundation\Blender *\`
- PATH (`blender.exe`)

**Detection locations (macOS):**
- `~/Library/Application Support/MeshPrep/tools/blender/`
- `/Applications/Blender.app/`
- PATH (`blender`)

**Detection locations (Linux):**
- `~/.local/share/meshprep/tools/blender/`
- `/usr/bin/blender`
- `/snap/bin/blender`
- PATH (`blender`)

#### PrusaSlicer (Primary Slicer)

| Property | Value |
|----------|-------|
| Purpose | Slicer validation (recommended) |
| Pinned Version | 2.8.0 (or latest stable) |
| Required | No (but strongly recommended) |
| Size | ~200 MB |
| Install Method | Download portable ZIP |

**Detection locations (Windows):**
- `%LOCALAPPDATA%\MeshPrep\tools\slicers\prusaslicer-*\`
- `C:\Program Files\Prusa3D\PrusaSlicer\`
- PATH (`prusa-slicer.exe`, `prusa-slicer-console.exe`)

#### OrcaSlicer (Alternative)

| Property | Value |
|----------|-------|
| Purpose | Slicer validation (alternative) |
| Pinned Version | 2.2.0 (or latest stable) |
| Required | No |
| Size | ~250 MB |
| Install Method | Download portable ZIP |

**Detection locations (Windows):**
- `%LOCALAPPDATA%\MeshPrep\tools\slicers\orcaslicer-*\`
- `C:\Program Files\OrcaSlicer\`
- PATH (`orca-slicer.exe`)

#### SuperSlicer (Alternative)

| Property | Value |
|----------|-------|
| Purpose | Slicer validation (alternative) |
| Pinned Version | 2.5.0 (or latest stable) |
| Required | No |
| Size | ~200 MB |
| Install Method | Download portable ZIP |

#### Cura (Alternative)

| Property | Value |
|----------|-------|
| Purpose | Slicer validation (requires printer profile) |
| Pinned Version | 5.7.0 (or latest stable) |
| Required | No |
| Size | ~500 MB |
| Install Method | Download portable or installer |

**Note:** Cura requires a printer profile to function. MeshPrep includes a generic profile for validation purposes.

### Automatic Installation Flow

When MeshPrep starts or when a tool is needed:

```
┌─────────────────────────────────────────────────────────────────┐
│                    TOOL DETECTION FLOW                          │
└─────────────────────────────────────────────────────────────────┘

1. Check explicit config (tools.json, env vars)
         │
         ▼
    ┌─────────┐      YES
    │ Found?  │─────────────► Use configured path
    └────┬────┘
         │ NO
         ▼
2. Check MeshPrep tools directory
         │
         ▼
    ┌─────────┐      YES
    │ Found?  │─────────────► Use MeshPrep-managed install
    └────┬────┘
         │ NO
         ▼
3. Check system PATH and standard locations
         │
         ▼
    ┌─────────┐      YES
    │ Found?  │─────────────► Use system install
    └────┬────┘
         │ NO
         ▼
4. Prompt user for automatic installation
         │
         ▼
    ┌─────────────┐      YES
    │ User agrees?│─────────────► Download and install to tools/
    └──────┬──────┘
           │ NO
           ▼
    Mark tool as unavailable (log warning)
```

### Installation Implementation

#### Download Sources

| Tool | Download URL Pattern |
|------|---------------------|
| Blender | `https://download.blender.org/release/Blender{version}/blender-{version}-{platform}.zip` |
| PrusaSlicer | `https://github.com/prusa3d/PrusaSlicer/releases/download/version_{version}/PrusaSlicer-{version}+{platform}.zip` |
| OrcaSlicer | `https://github.com/SoftFever/OrcaSlicer/releases/download/v{version}/OrcaSlicer_{platform}_{version}.zip` |
| SuperSlicer | `https://github.com/supermerill/SuperSlicer/releases/download/{version}/SuperSlicer-{platform}-{version}.zip` |

#### Installation Script: `scripts/install_tools.py`

```python
# Usage examples:
python scripts/install_tools.py --list              # List available tools
python scripts/install_tools.py --install blender   # Install Blender
python scripts/install_tools.py --install prusa     # Install PrusaSlicer
python scripts/install_tools.py --install all       # Install all tools
python scripts/install_tools.py --check             # Check installed tools
python scripts/install_tools.py --update            # Update to pinned versions
```

#### Tools Configuration File: `config/tools.json`

```json
{
  "version": "1.0.0",
  "tools_directory": null,  // null = use platform default
  "auto_install": true,     // Prompt to install missing tools
  "tools": {
    "blender": {
      "enabled": true,
      "path": null,          // null = auto-detect
      "version": "4.2.0",    // Pinned version for auto-install
      "min_version": "3.6.0" // Minimum supported version
    },
    "slicer": {
      "preferred": "prusa",  // Primary slicer to use
      "fallback_order": ["orca", "superslicer", "cura"],
      "prusa": {
        "enabled": true,
        "path": null,
        "version": "2.8.0"
      },
      "orca": {
        "enabled": true,
        "path": null,
        "version": "2.2.0"
      },
      "superslicer": {
        "enabled": true,
        "path": null,
        "version": "2.5.0"
      },
      "cura": {
        "enabled": false,    // Requires printer profile setup
        "path": null,
        "version": "5.7.0"
      }
    }
  }
}
```

### Environment Variables

Users can override auto-detection with environment variables:

| Variable | Description |
|----------|-------------|
| `MESHPREP_TOOLS_DIR` | Override tools directory location |
| `MESHPREP_BLENDER_PATH` | Path to Blender executable |
| `MESHPREP_SLICER_PATH` | Path to preferred slicer executable |
| `MESHPREP_PRUSA_PATH` | Path to PrusaSlicer executable |
| `MESHPREP_ORCA_PATH` | Path to OrcaSlicer executable |
| `MESHPREP_AUTO_INSTALL` | Set to `0` to disable auto-install prompts |

### GUI Integration

The GUI provides a **Tools** settings panel:

1. **Tool Status**: Shows installed tools with version and path
2. **Install Button**: One-click install for missing tools
3. **Configure Button**: Override paths or disable tools
4. **Update Button**: Update tools to latest pinned versions
5. **Disk Usage**: Shows space used by MeshPrep-managed tools

### CLI Integration

The CLI supports tool management commands:

```bash
# Check environment (includes tool status)
python scripts/checkenv.py

# Install missing tools
python scripts/install_tools.py --install all

# Force reinstall of a specific tool
python scripts/install_tools.py --install blender --force

# Use specific tool path for a run
python scripts/auto_fix_stl.py --input model.stl --blender-path /path/to/blender
```

### Startup Behavior

On first run, MeshPrep:

1. Detects installed Python packages (trimesh, pymeshfix, meshio)
2. Detects external tools (Blender, slicers)
3. If any slicer is missing, prompts: "No slicer found. Install PrusaSlicer for slicer validation? [Y/n]"
4. If Blender is missing, notes: "Blender not found. Advanced repair features will be limited."
5. Records tool versions in `report.json` for reproducibility

### Disk Space Considerations

Total space for all tools:

| Tool | Approximate Size |
|------|------------------|
| Blender | ~400 MB |
| PrusaSlicer | ~200 MB |
| OrcaSlicer | ~250 MB |
| SuperSlicer | ~200 MB |
| **Total (recommended)** | **~600 MB** (Blender + PrusaSlicer) |
| **Total (all)** | **~1.1 GB** |

**Recommendation**: Install Blender + PrusaSlicer for full functionality.

### Cleanup and Uninstall

```bash
# Remove all MeshPrep-managed tools
python scripts/install_tools.py --uninstall all

# Remove specific tool
python scripts/install_tools.py --uninstall blender

# Clear download cache
python scripts/install_tools.py --clear-cache
```

What to include in `docs/INSTALL.md` (summary)
- Quickstart: create virtualenv, `pip install -r requirements.txt`, example run command.
- Automatic tool installation: explain that Blender and slicers are installed automatically on first use.
- Manual tool configuration: how to use existing system installations or override paths.
- Alternate install: `conda` environment instructions with exported `environment.yml` optional.
- Platform notes: Windows, macOS, Linux caveats and troubleshooting hints (e.g. common `pymeshfix` wheel issues).
- Docker: optional `Dockerfile` usage and how to run a reproducible containerized run.
- Troubleshooting: how to collect logs, attach `report.json`, `checkenv` output, and minimal repro files when reporting issues.

Versioning rules
- Maintain a `VERSION` file at repo root using Semantic Versioning (MAJOR.MINOR.PATCH), e.g. `0.1.0`.
- When dependencies or install steps change, update `requirements.txt`, bump `VERSION`, and add a short entry in `CHANGELOG.md`.
- Filter scripts must include a `preset_version` and either pinned dependencies or a Docker image tag to allow exact reproduction.
- External tool versions are pinned in `config/tools.json` and recorded in `report.json`.

Environment validation tool
- Provide `scripts/checkenv.py` that prints installed package versions and checks for external tools (Blender, slicers).
- Output includes: Python version, package versions, tool paths and versions, disk space usage.
- Include its output in exported run packages and CI logs.
- If tools are missing, output includes installation commands.

Documentation hygiene & process
- Require PRs that change dependencies or install steps to update `docs/INSTALL.md`, `requirements.txt`, and bump `VERSION`.
- Add PR checklist in `CONTRIBUTING.md` to remind contributors to update installation docs and versioning when relevant.

Release process (brief)
- Tag releases (`vMAJOR.MINOR.PATCH`) and publish a release that includes `CHANGELOG.md` entries and the updated `VERSION` file.
- Optionally publish a Docker image with the same tag for reproducible environments.
- Update pinned tool versions in `config/tools.json` when new stable versions are available.
