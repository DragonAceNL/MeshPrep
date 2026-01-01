# Copyright 2025 Dragon Ace
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""Main window for MeshPrep GUI."""

from pathlib import Path
from datetime import datetime

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget,
    QLabel, QPushButton, QFileDialog, QMessageBox, QFrame,
    QSplitter, QTextEdit, QGroupBox, QRadioButton, QButtonGroup,
    QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView,
    QApplication, QStatusBar, QToolBar,
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QFont, QAction, QIcon
from PySide6.QtSvgWidgets import QSvgWidget

from ..core.mock_mesh import MockMesh, load_mock_stl, save_mock_stl
from ..core.diagnostics import Diagnostics, compute_diagnostics
from ..core.profiles import ProfileDetector, ProfileMatch
from ..core.filter_script import FilterScript, FilterScriptRunner, generate_filter_script
from ..core.actions import get_action_registry

from .styles import apply_theme, DARK_THEME, LIGHT_THEME
from .widgets import (
    StepIndicator, DiagnosticsPanel, LogConsole, ProgressPanel, ProfileCard
)
from .filter_editor import FilterScriptEditor
from ..resources import get_logo_path, get_resource_path


class WorkerThread(QThread):
    """Worker thread for long-running operations."""
    
    progress = Signal(int, str)  # percentage, message
    step_completed = Signal(int, str, str)  # step, action, status
    finished = Signal(bool, str)  # success, result/error
    
    def __init__(self, task: str, **kwargs):
        super().__init__()
        self.task = task
        self.kwargs = kwargs
        self._cancelled = False
    
    def cancel(self):
        self._cancelled = True
    
    def run(self):
        try:
            if self.task == "scan":
                self._run_scan()
            elif self.task == "execute":
                self._run_execute()
        except Exception as e:
            self.finished.emit(False, str(e))
    
    def _run_scan(self):
        """Scan a model and detect profile."""
        path = self.kwargs.get("path")
        
        self.progress.emit(10, "Loading model...")
        mesh = load_mock_stl(Path(path))
        
        self.progress.emit(40, "Computing diagnostics...")
        diagnostics = compute_diagnostics(mesh)
        
        self.progress.emit(70, "Detecting profile...")
        detector = ProfileDetector()
        matches = detector.detect(diagnostics)
        
        self.progress.emit(100, "Complete")
        
        # Store results in kwargs for retrieval
        self.kwargs["mesh"] = mesh
        self.kwargs["diagnostics"] = diagnostics
        self.kwargs["matches"] = matches
        
        self.finished.emit(True, "Scan complete")
    
    def _run_execute(self):
        """Execute a filter script."""
        mesh = self.kwargs.get("mesh")
        script = self.kwargs.get("script")
        output_path = self.kwargs.get("output_path")
        
        runner = FilterScriptRunner()
        
        def progress_callback(step, total, msg):
            pct = int((step / total) * 100) if total > 0 else 0
            self.progress.emit(pct, msg)
            self.step_completed.emit(step, script.actions[step - 1].name if step <= len(script.actions) else "", "running")
        
        runner.set_progress_callback(progress_callback)
        
        result = runner.run(script, mesh)
        
        # Save output if successful
        if result.success and result.final_mesh and output_path:
            save_mock_stl(result.final_mesh, Path(output_path))
        
        self.kwargs["result"] = result
        self.finished.emit(result.success, result.summary())


class MainWindow(QMainWindow):
    """Main application window."""
    
    STEPS = [
        "Environment",
        "Select Model",
        "Review Script",
        "Execute",
        "Results",
    ]
    
    def __init__(self):
        super().__init__()
        
        self.mesh: MockMesh = None
        self.diagnostics: Diagnostics = None
        self.profile_matches: list[ProfileMatch] = []
        self.script: FilterScript = None
        self.output_path: Path = None
        self.worker: WorkerThread = None
        self._selected_file_path: str = None
        
        self._current_theme = "dark"
        
        self.setWindowTitle("MeshPrep - STL Cleanup Pipeline")
        self.setMinimumSize(1200, 800)
        
        self._setup_toolbar()
        self._setup_ui()
        apply_theme(self, self._current_theme)
        
        # Start at environment check
        self._go_to_step(0)
        self._simulate_env_check()
    
    def _setup_toolbar(self):
        """Set up the toolbar."""
        self.toolbar = QToolBar("Main Toolbar")
        self.toolbar.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, self.toolbar)
        
        # Theme toggle action
        self.theme_action = QAction("🌙 Dark Mode", self)
        self.theme_action.setToolTip("Toggle between dark and light mode")
        self.theme_action.triggered.connect(self._toggle_theme)
        self.toolbar.addAction(self.theme_action)
    
    def _toggle_theme(self):
        """Toggle between dark and light theme."""
        if self._current_theme == "dark":
            self._current_theme = "light"
            self.theme_action.setText("☀️ Light Mode")
        else:
            self._current_theme = "dark"
            self.theme_action.setText("🌙 Dark Mode")
        
        apply_theme(self, self._current_theme)
    
    def _setup_ui(self):
        """Set up the main UI."""
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Step indicator sidebar
        self.step_indicator = StepIndicator(self.STEPS)
        self.step_indicator.step_clicked.connect(self._go_to_step)
        self.step_indicator.setMaximumWidth(200)
        self.step_indicator.setMinimumWidth(180)
        
        # Load logo
        logo_path = get_logo_path()
        if logo_path.exists():
            self.step_indicator.set_logo(str(logo_path))
        
        main_layout.addWidget(self.step_indicator)
        
        # Main content area
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(16, 16, 16, 16)
        
        # Stacked widget for step content
        self.stack = QStackedWidget()
        self._create_step_pages()
        content_layout.addWidget(self.stack, stretch=1)
        
        # Log console (collapsible)
        self.log_console = LogConsole()
        self.log_console.setMaximumHeight(150)
        content_layout.addWidget(self.log_console)
        
        # Navigation buttons row with MeshPrep text logo
        nav_layout = QHBoxLayout()
        
        # MeshPrep text logo on the left (with container for positioning)
        text_logo_container = QWidget()
        text_logo_layout = QVBoxLayout(text_logo_container)
        text_logo_layout.setContentsMargins(0, 10, 0, 0)  # Add 3px top margin
        text_logo_layout.setSpacing(0)
        
        self.text_logo = QSvgWidget()
        text_logo_path = get_resource_path("images/MeshPrepText.svg")
        if text_logo_path.exists():
            self.text_logo.load(str(text_logo_path))
        self.text_logo.setFixedSize(225, 54)  # Adjust size as needed
        text_logo_layout.addWidget(self.text_logo)
        
        nav_layout.addWidget(text_logo_container, alignment=Qt.AlignVCenter)
        
        nav_layout.addStretch()
        
        self.prev_btn = QPushButton("← Previous")
        self.prev_btn.clicked.connect(self._go_previous)
        nav_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("Next →")
        self.next_btn.setObjectName("primary")
        self.next_btn.clicked.connect(self._go_next)
        nav_layout.addWidget(self.next_btn)
        
        content_layout.addLayout(nav_layout)
        
        main_layout.addLayout(content_layout, stretch=1)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def _create_step_pages(self):
        """Create the content pages for each step."""
        # Step 0: Environment
        self.stack.addWidget(self._create_env_page())
        
        # Step 1: Select Model
        self.stack.addWidget(self._create_select_page())
        
        # Step 2: Review Script
        self.stack.addWidget(self._create_profile_page())
        
        # Step 3: Execute
        self.stack.addWidget(self._create_execute_page())
        
        # Step 4: Results
        self.stack.addWidget(self._create_results_page())
    
    def _create_env_page(self) -> QWidget:
        """Create environment check page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        
        title = QLabel("Environment Check")
        title.setObjectName("header")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        layout.addWidget(title)
        
        desc = QLabel("Checking for required dependencies and tools...")
        desc.setObjectName("secondary")
        layout.addWidget(desc)
        
        self.env_text = QTextEdit()
        self.env_text.setReadOnly(True)
        self.env_text.setFont(QFont("Consolas", 10))
        layout.addWidget(self.env_text)
        
        self.env_status = QLabel("")
        layout.addWidget(self.env_status)
        
        btn_layout = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._simulate_env_check)
        btn_layout.addWidget(refresh_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        return page
    
    def _create_select_page(self) -> QWidget:
        """Create model selection page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        
        title = QLabel("Select Model")
        title.setObjectName("header")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        layout.addWidget(title)
        
        # File selection
        file_group = QGroupBox("STL File")
        file_layout = QHBoxLayout(file_group)
        
        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setWordWrap(True)
        file_layout.addWidget(self.file_path_label, stretch=1)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_file)
        file_layout.addWidget(browse_btn)
        
        layout.addWidget(file_group)
        
        # Filter source choice
        source_group = QGroupBox("Filter Script Source")
        source_layout = QVBoxLayout(source_group)
        
        self.source_buttons = QButtonGroup()
        
        auto_radio = QRadioButton("Auto-detect profile and generate suggested filter script")
        auto_radio.setChecked(True)
        self.source_buttons.addButton(auto_radio, 0)
        source_layout.addWidget(auto_radio)
        
        existing_radio = QRadioButton("Use existing filter script")
        self.source_buttons.addButton(existing_radio, 1)
        source_layout.addWidget(existing_radio)
        
        # Existing script options
        self.existing_options = QWidget()
        existing_layout = QHBoxLayout(self.existing_options)
        existing_layout.setContentsMargins(20, 0, 0, 0)
        
        load_btn = QPushButton("Load from file...")
        load_btn.clicked.connect(self._load_filter_script)
        existing_layout.addWidget(load_btn)
        
        self.loaded_script_label = QLabel("")
        existing_layout.addWidget(self.loaded_script_label)
        existing_layout.addStretch()
        
        source_layout.addWidget(self.existing_options)
        self.existing_options.setEnabled(False)
        
        self.source_buttons.idToggled.connect(self._on_source_changed)
        
        layout.addWidget(source_group)
        layout.addStretch()
        
        return page
    
    def _create_profile_page(self) -> QWidget:
        """Create profile detection and script editing page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        
        title = QLabel("Review Filter Script")
        title.setObjectName("header")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        layout.addWidget(title)
        
        splitter = QSplitter(Qt.Horizontal)
        
        # Left: Diagnostics
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.diagnostics_panel = DiagnosticsPanel()
        left_layout.addWidget(self.diagnostics_panel)
        
        self.profile_card = ProfileCard()
        left_layout.addWidget(self.profile_card)
        
        splitter.addWidget(left_panel)
        
        # Right: Filter script
        right_panel = QFrame()
        right_panel.setObjectName("panel")
        right_layout = QVBoxLayout(right_panel)
        
        script_header = QHBoxLayout()
        script_title = QLabel("Filter Script")
        script_title.setObjectName("header")
        script_header.addWidget(script_title)
        script_header.addStretch()
        
        edit_btn = QPushButton("Edit Script...")
        edit_btn.clicked.connect(self._edit_script)
        script_header.addWidget(edit_btn)
        
        right_layout.addLayout(script_header)
        
        self.script_preview = QTextEdit()
        self.script_preview.setReadOnly(True)
        self.script_preview.setFont(QFont("Consolas", 9))
        right_layout.addWidget(self.script_preview)
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        save_btn = QPushButton("Save Preset...")
        save_btn.clicked.connect(self._save_preset)
        action_layout.addWidget(save_btn)
        
        export_btn = QPushButton("Export...")
        export_btn.clicked.connect(self._export_script)
        action_layout.addWidget(export_btn)
        
        action_layout.addStretch()
        right_layout.addLayout(action_layout)
        
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 500])
        
        layout.addWidget(splitter)
        
        return page
    
    def _create_execute_page(self) -> QWidget:
        """Create execution page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        
        title = QLabel("Execute Filter Script")
        title.setObjectName("header")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        layout.addWidget(title)
        
        # Output path
        output_group = QGroupBox("Output")
        output_layout = QHBoxLayout(output_group)
        
        self.output_path_label = QLabel("./output/")
        output_layout.addWidget(self.output_path_label, stretch=1)
        
        browse_output_btn = QPushButton("Browse...")
        browse_output_btn.clicked.connect(self._browse_output)
        output_layout.addWidget(browse_output_btn)
        
        layout.addWidget(output_group)
        
        self.execute_progress = ProgressPanel()
        layout.addWidget(self.execute_progress)
        
        # Steps table
        self.execute_table = QTableWidget()
        self.execute_table.setColumnCount(5)
        self.execute_table.setHorizontalHeaderLabels(["Step", "Action", "Status", "Runtime", "Notes"])
        self.execute_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.execute_table)
        
        btn_layout = QHBoxLayout()
        
        self.execute_btn = QPushButton("Execute")
        self.execute_btn.setObjectName("primary")
        self.execute_btn.clicked.connect(self._run_execute)
        btn_layout.addWidget(self.execute_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_execution)
        btn_layout.addWidget(self.stop_btn)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        return page
    
    def _create_results_page(self) -> QWidget:
        """Create results page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        
        title = QLabel("Results")
        title.setObjectName("header")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        layout.addWidget(title)
        
        # Summary
        self.results_summary = QTextEdit()
        self.results_summary.setReadOnly(True)
        self.results_summary.setFont(QFont("Consolas", 10))
        layout.addWidget(self.results_summary)
        
        # Actions
        btn_layout = QHBoxLayout()
        
        open_output_btn = QPushButton("Open Output Folder")
        open_output_btn.clicked.connect(self._open_output_folder)
        btn_layout.addWidget(open_output_btn)
        
        export_run_btn = QPushButton("Export Run Package...")
        export_run_btn.clicked.connect(self._export_run_package)
        btn_layout.addWidget(export_run_btn)
        
        btn_layout.addStretch()
        
        new_run_btn = QPushButton("Start New Run")
        new_run_btn.clicked.connect(self._start_new_run)
        btn_layout.addWidget(new_run_btn)
        
        layout.addLayout(btn_layout)
        
        return page
    
    # Navigation
    def _go_to_step(self, step: int):
        """Navigate to a step."""
        self.stack.setCurrentIndex(step)
        self.step_indicator.set_current_step(step)
        
        self.prev_btn.setEnabled(step > 0)
        self.next_btn.setEnabled(step < len(self.STEPS) - 1)
        
        # Update next button text
        if step == 3:  # Execute
            self.next_btn.setText("View Results →")
        elif step == len(self.STEPS) - 1:
            self.next_btn.setText("Finish")
        else:
            self.next_btn.setText("Next →")
    
    def _go_previous(self):
        current = self.stack.currentIndex()
        if current > 0:
            self._go_to_step(current - 1)
    
    def _go_next(self):
        current = self.stack.currentIndex()
        
        # Validate current step before proceeding
        if current == 1:  # Select Model
            # Check if a file has been selected
            if not hasattr(self, "_selected_file_path") or not self._selected_file_path:
                QMessageBox.warning(self, "Warning", "Please select an STL file first.")
                return
            # Run scan if auto-detect
            if self.source_buttons.checkedId() == 0:
                self._run_scan()
                return
            # If using existing script, need both file and script
            elif self.source_buttons.checkedId() == 1:
                if not self.script:
                    QMessageBox.warning(self, "Warning", "Please load a filter script first.")
                    return
                # Load mesh and go to profile page
                from ..core.mock_mesh import load_mock_stl
                self.mesh = load_mock_stl(Path(self._selected_file_path))
                from ..core.diagnostics import compute_diagnostics
                self.diagnostics = compute_diagnostics(self.mesh)
                self.diagnostics_panel.set_diagnostics(self.diagnostics)
                self._update_script_preview()
                self.step_indicator.mark_completed(1)
                self._go_to_step(2)
                return
        
        if current < len(self.STEPS) - 1:
            self._go_to_step(current + 1)
            self.step_indicator.mark_completed(current)
    
    # Step 0: Environment
    def _simulate_env_check(self):
        """Simulate environment check."""
        self.env_text.clear()
        
        checks = [
            ("Python", "3.11.0", True),
            ("PySide6", "6.6.0", True),
            ("trimesh", "4.0.0 (MOCKED)", True),
            ("pymeshfix", "0.16.0 (MOCKED)", True),
            ("meshio", "5.3.0 (MOCKED)", True),
            ("Blender", "Not found (optional)", False),
        ]
        
        all_ok = True
        for name, version, required in checks:
            status = "✓" if "MOCKED" in version or "Not found" not in version else "○"
            self.env_text.append(f"{status} {name}: {version}")
            if required and "Not found" in version:
                all_ok = False
        
        self.env_text.append("")
        self.env_text.append("Note: This is a POC with mocked mesh libraries.")
        
        if all_ok:
            self.env_status.setText("✓ Environment ready")
            self.env_status.setStyleSheet(f"color: {DARK_THEME['success']};")
            self.step_indicator.mark_completed(0)
        else:
            self.env_status.setText("✗ Missing dependencies")
            self.env_status.setStyleSheet(f"color: {DARK_THEME['error']};")
        
        self.log_console.log("Environment check complete", "success" if all_ok else "warning")
    
    # Step 1: Select Model
    def _browse_file(self):
        """Browse for STL file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select STL File", "", "STL Files (*.stl);;All Files (*)"
        )
        if path:
            self.file_path_label.setText(path)
            self.mesh = None  # Will be loaded during scan
            self._selected_file_path = path
            self.log_console.log(f"Selected: {path}")
    
    def _on_source_changed(self, button_id: int, checked: bool):
        """Handle filter source change."""
        self.existing_options.setEnabled(button_id == 1)
    
    def _load_filter_script(self):
        """Load a filter script from file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Filter Script", "", "JSON Files (*.json);;YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if path:
            try:
                self.script = FilterScript.load(Path(path))
                self.loaded_script_label.setText(f"Loaded: {self.script.name}")
                self.log_console.log(f"Loaded filter script: {self.script.name}", "success")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load filter script: {e}")
                self.log_console.log(f"Failed to load filter script: {e}", "error")
    
    # Step 2: Review Script
    def _run_scan(self):
        """Run model scan."""
        if not hasattr(self, "_selected_file_path"):
            return
        
        self.worker = WorkerThread("scan", path=self._selected_file_path)
        self.worker.progress.connect(self._on_scan_progress)
        self.worker.finished.connect(self._on_scan_finished)
        self.worker.start()
        
        self.next_btn.setEnabled(False)
        self.status_bar.showMessage("Scanning model...")
    
    @Slot(int, str)
    def _on_scan_progress(self, pct: int, msg: str):
        self.status_bar.showMessage(f"Scanning: {msg}")
    
    @Slot(bool, str)
    def _on_scan_finished(self, success: bool, message: str):
        self.next_btn.setEnabled(True)
        
        if success:
            self.mesh = self.worker.kwargs.get("mesh")
            self.diagnostics = self.worker.kwargs.get("diagnostics")
            self.profile_matches = self.worker.kwargs.get("matches", [])
            
            # Update UI
            self.diagnostics_panel.set_diagnostics(self.diagnostics)
            
            if self.profile_matches:
                self.profile_card.set_profile(self.profile_matches[0])
                
                # Generate suggested script
                match = self.profile_matches[0]
                self.script = generate_filter_script(
                    match.profile.name,
                    self.mesh.fingerprint,
                    match.profile.suggested_actions,
                )
                self._update_script_preview()
            
            self.log_console.log("Model scan complete", "success")
            self.step_indicator.mark_completed(1)
            self._go_to_step(2)
        else:
            self.log_console.log(f"Scan failed: {message}", "error")
            QMessageBox.critical(self, "Error", f"Model scan failed: {message}")
        
        self.status_bar.showMessage("Ready")
    
    def _update_script_preview(self):
        """Update the script preview text."""
        if self.script:
            self.script_preview.setPlainText(self.script.to_json(indent=2))
    
    def _edit_script(self):
        """Open the filter script editor."""
        dialog = FilterScriptEditor(self.script, self, theme=self._current_theme)
        if dialog.exec() == FilterScriptEditor.Accepted:
            self.script = dialog.get_script()
            self._update_script_preview()
            self.log_console.log("Filter script updated")
    
    def _save_preset(self):
        """Save script as preset."""
        if not self.script:
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Preset", f"filters/{self.script.name}.json",
            "JSON Files (*.json);;YAML Files (*.yaml)"
        )
        if path:
            self.script.save(Path(path))
            self.log_console.log(f"Saved preset to: {path}", "success")
    
    def _export_script(self):
        """Export filter script."""
        self._save_preset()
    
    # Step 3: Execute
    def _browse_output(self):
        """Browse for output directory."""
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_path_label.setText(path)
            self.output_path = Path(path)
    
    def _setup_table(self, table: QTableWidget):
        """Set up execution table."""
        table.setRowCount(len(self.script.actions))
        for i, action in enumerate(self.script.actions):
            table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            table.setItem(i, 1, QTableWidgetItem(action.name))
            table.setItem(i, 2, QTableWidgetItem("pending"))
            if table.columnCount() > 3:
                table.setItem(i, 3, QTableWidgetItem(""))
            if table.columnCount() > 4:
                table.setItem(i, 4, QTableWidgetItem(""))
    
    def _run_execute(self):
        """Execute the filter script."""
        if not self.mesh or not self.script:
            return
        
        # Determine output path
        output_dir = Path(self.output_path_label.text())
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_name = self.mesh.source_path.stem if self.mesh.source_path else "model"
        output_file = output_dir / f"{source_name}__{self.script.name}__{timestamp}.stl"
        
        self._setup_table(self.execute_table)
        
        self.worker = WorkerThread("execute", mesh=self.mesh, script=self.script, output_path=output_file)
        self.worker.progress.connect(lambda p, m: self.execute_progress.set_progress(p, "Executing...", m))
        self.worker.step_completed.connect(self._on_step_completed)
        self.worker.finished.connect(self._on_execute_finished)
        self.worker.start()
        
        self.execute_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.output_path = output_file
    
    @Slot(int, str, str)
    def _on_step_completed(self, step: int, action: str, status: str):
        if step <= self.execute_table.rowCount():
            self.execute_table.setItem(step - 1, 2, QTableWidgetItem(status))
    
    def _stop_execution(self):
        """Stop execution."""
        if self.worker:
            self.worker.cancel()
    
    @Slot(bool, str)
    def _on_execute_finished(self, success: bool, message: str):
        self.execute_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        result = self.worker.kwargs.get("result")
        if result:
            for i, step in enumerate(result.steps):
                self.execute_table.setItem(i, 2, QTableWidgetItem(step.status))
                self.execute_table.setItem(i, 3, QTableWidgetItem(f"{step.runtime_ms:.1f}ms"))
                self.execute_table.setItem(i, 4, QTableWidgetItem(step.message or step.error or ""))
            
            # Update results page
            self.results_summary.setPlainText(result.summary())
        
        self.execute_progress.set_progress(100, "Complete" if success else "Failed")
        self.log_console.log(f"Execution {'complete' if success else 'failed'}", "success" if success else "error")
        
        if success:
            self.step_indicator.mark_completed(3)
            self.log_console.log(f"Output saved to: {self.output_path}", "success")
    
    # Step 4: Results
    def _open_output_folder(self):
        """Open output folder in file explorer."""
        if self.output_path and self.output_path.parent.exists():
            import subprocess
            subprocess.Popen(f'explorer "{self.output_path.parent}"')
    
    def _export_run_package(self):
        """Export run package."""
        path = QFileDialog.getExistingDirectory(self, "Export Run Package To")
        if path:
            # Create run package (simplified for POC)
            package_dir = Path(path) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            package_dir.mkdir(parents=True, exist_ok=True)
            
            if self.script:
                self.script.save(package_dir / "filter_script.json")
            
            # Write report
            report = {
                "timestamp": datetime.now().isoformat(),
                "script_name": self.script.name if self.script else "unknown",
                "output_file": str(self.output_path) if self.output_path else None,
            }
            import json
            (package_dir / "report.json").write_text(json.dumps(report, indent=2))
            
            self.log_console.log(f"Run package exported to: {package_dir}", "success")
            QMessageBox.information(self, "Success", f"Run package exported to:\n{package_dir}")
    
    def _start_new_run(self):
        """Start a new run."""
        self.mesh = None
        self.diagnostics = None
        self.profile_matches = []
        self.script = None
        self.output_path = None
        self._selected_file_path = None
        
        self.file_path_label.setText("No file selected")
        self.loaded_script_label.setText("")
        self.diagnostics_panel.clear()
        self.script_preview.clear()
        
        self.step_indicator.completed_steps.clear()
        self.step_indicator.mark_completed(0)  # Keep env check
        
        self._go_to_step(1)
