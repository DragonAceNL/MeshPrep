"""
Improved MeshPrep v1 GUI prototype
- Applies clearer layout and UX
- Simulates full high-level flow: environment check, model selection, profile detection, suggested filter generation,
  review/dry-run, execution with logs, and results/reporting.
- Uses PySide6. This is a simulation — no real mesh work performed.
"""
import sys
import json
import datetime
import random
from PySide6 import QtWidgets, QtCore, QtGui

# Colors / styles
BG = "#0f1720"
PANEL = "#111822"
ACCENT = "#4fe8c4"
TEXT = "#dff6fb"
BTN = "#1b2b33"

class EnvWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{BG}; color:{TEXT}")
        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("Environment Check")
        title.setStyleSheet("font: 14pt 'Segoe UI'; color: %s" % ACCENT)
        layout.addWidget(title)

        self.status_view = QtWidgets.QPlainTextEdit()
        self.status_view.setReadOnly(True)
        self.status_view.setFixedHeight(220)
        self.status_view.setFont(QtGui.QFont('Consolas', 10))
        self.status_view.setPlainText("Environment check not run. Click 'Auto-setup' to simulate.")
        layout.addWidget(self.status_view)

        btn_row = QtWidgets.QHBoxLayout()
        self.auto_setup_btn = QtWidgets.QPushButton("Auto-setup")
        self.auto_setup_btn.setStyleSheet("padding:8px; background:%s; color:%s" % (BTN, TEXT))
        self.manual_btn = QtWidgets.QPushButton("Show manual instructions")
        # keep backward-compatible attribute name
        self.instructions_btn = self.manual_btn
        btn_row.addWidget(self.auto_setup_btn)
        btn_row.addWidget(self.manual_btn)
        layout.addLayout(btn_row)

        layout.addStretch()

class SelectionWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{BG}; color:{TEXT}")
        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("Model selection & filter source")
        title.setStyleSheet("font: 14pt 'Segoe UI'; color: %s" % ACCENT)
        layout.addWidget(title)

        file_row = QtWidgets.QHBoxLayout()
        self.file_label = QtWidgets.QLabel("No model selected")
        self.file_btn = QtWidgets.QPushButton("Select STL...")
        file_row.addWidget(self.file_label)
        file_row.addWidget(self.file_btn)
        layout.addLayout(file_row)

        layout.addSpacing(8)
        self.radio_auto = QtWidgets.QRadioButton("Auto-detect profile and suggest filter")
        self.radio_manual = QtWidgets.QRadioButton("Use my filter script (file or URL)")
        self.radio_auto.setChecked(True)
        layout.addWidget(self.radio_auto)
        layout.addWidget(self.radio_manual)

        self.load_filter_btn = QtWidgets.QPushButton("Load filter script")
        layout.addWidget(self.load_filter_btn)
        layout.addStretch()

class SuggestWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{BG}; color:{TEXT}")
        layout = QtWidgets.QHBoxLayout(self)

        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel("Diagnostics"))
        self.diag = QtWidgets.QTextEdit()
        self.diag.setReadOnly(True)
        self.diag.setFixedHeight(200)
        left.addWidget(self.diag)

        left.addWidget(QtWidgets.QLabel("Profile suggestion"))
        self.profile_lbl = QtWidgets.QLabel("<none>")
        left.addWidget(self.profile_lbl)

        layout.addLayout(left, 2)

        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("Suggested filter script (editable)"))
        self.actions = QtWidgets.QListWidget()
        right.addWidget(self.actions)
        ah = QtWidgets.QHBoxLayout()
        self.add_act = QtWidgets.QPushButton("Add")
        self.edit_act = QtWidgets.QPushButton("Edit")
        self.del_act = QtWidgets.QPushButton("Remove")
        ah.addWidget(self.add_act)
        ah.addWidget(self.edit_act)
        ah.addWidget(self.del_act)
        right.addLayout(ah)

        self.save_preset_btn = QtWidgets.QPushButton("Save preset")
        right.addWidget(self.save_preset_btn)
        right.addStretch()

        layout.addLayout(right, 1)

        # wire simple actions
        self.add_act.clicked.connect(self._add)
        self.edit_act.clicked.connect(self._edit)
        self.del_act.clicked.connect(self._del)
        self.save_preset_btn.clicked.connect(self._save)

    def _add(self):
        text, ok = QtWidgets.QInputDialog.getText(self, "Action name", "Action:")
        if ok and text:
            self.actions.addItem(text)

    def _edit(self):
        item = self.actions.currentItem()
        if not item:
            return
        text, ok = QtWidgets.QInputDialog.getText(self, "Edit action", "Action:", text=item.text())
        if ok and text:
            item.setText(text)

    def _del(self):
        r = self.actions.currentRow()
        if r >= 0:
            self.actions.takeItem(r)

    def _save(self):
        preset = {
            "name": "preset-1",
            "meta": {"generated_by": "ui-save", "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()},
            "actions": [self.actions.item(i).text() for i in range(self.actions.count())]
        }
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save preset", "preset.json", "JSON (*.json)")
        if fname:
            with open(fname, 'w') as f:
                json.dump(preset, f, indent=2)
            QtWidgets.QMessageBox.information(self, "Saved", f"Preset saved to {fname}")

class DryRunWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{BG}; color:{TEXT}")
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Dry-run preview"))
        self.table = QtWidgets.QTableWidget(0,3)
        self.table.setHorizontalHeaderLabels(["Step","Status","Notes"])
        layout.addWidget(self.table)
        self.run_btn = QtWidgets.QPushButton("Run dry-run")
        layout.addWidget(self.run_btn)
        layout.addStretch()
        self.run_btn.clicked.connect(self.run)
        self._timer = None
        self._idx = 0

    def run(self):
        # gather actions from SuggestWidget
        main = self.parent()
        while main and not hasattr(main, 'suggest_widget'):
            main = main.parent()
        actions = []
        try:
            actions = [main.suggest_widget.actions.item(i).text() for i in range(main.suggest_widget.actions.count())]
        except Exception:
            pass
        if not actions:
            actions = ["trimesh_basic","fill_holes","validate"]
        self.table.setRowCount(0)
        for a in actions:
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r,0, QtWidgets.QTableWidgetItem(a))
            self.table.setItem(r,1, QtWidgets.QTableWidgetItem("pending"))
            self.table.setItem(r,2, QtWidgets.QTableWidgetItem(""))
        self._idx = 0
        if self._timer:
            self._timer.stop()
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._advance)
        self._timer.start(500)

    def _advance(self):
        if self._idx >= self.table.rowCount():
            if self._timer:
                self._timer.stop()
            return
        status = random.choice(["ok","ok","warn"])
        self.table.setItem(self._idx,1, QtWidgets.QTableWidgetItem(status))
        self.table.setItem(self._idx,2, QtWidgets.QTableWidgetItem("simulated"))
        self._idx += 1

class ExecuteWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{BG}; color:{TEXT}")
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Execute"))
        self.progress = QtWidgets.QProgressBar()
        layout.addWidget(self.progress)
        self.terminal = QtWidgets.QPlainTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setFixedHeight(220)
        layout.addWidget(self.terminal)
        h = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start run")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        h.addWidget(self.start_btn)
        h.addWidget(self.stop_btn)
        layout.addLayout(h)
        layout.addStretch()
        self._timer = None
        self._step = 0
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)

    def start(self):
        self.terminal.clear()
        self.progress.setValue(0)
        self._step = 0
        lines = [
            "[INFO] Loading model...",
            "[DEBUG] trimesh_basic...",
            "[INFO] fill_holes applied",
            "[INFO] pymeshfix repair complete",
            "[INFO] validation: watertight=True",
            "[INFO] exporting output"
        ]
        self._lines = lines
        if self._timer:
            self._timer.stop()
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(400)

    def _tick(self):
        if self._step < len(self._lines):
            self.terminal.appendPlainText(self._lines[self._step])
            self._step += 1
            self.progress.setValue(int((self._step/len(self._lines))*100))
        else:
            self.terminal.appendPlainText("[SUCCESS] Run complete")
            self.progress.setValue(100)
            self._timer.stop()

    def stop(self):
        if self._timer:
            self._timer.stop()
        self.terminal.appendPlainText("[INFO] Run stopped by user")

class ResultsWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{BG}; color:{TEXT}")
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Results & Export"))
        self.report_view = QtWidgets.QTextEdit()
        self.report_view.setReadOnly(True)
        layout.addWidget(self.report_view)
        self.export_btn = QtWidgets.QPushButton("Export run package (simulate)")
        layout.addWidget(self.export_btn)
        layout.addStretch()
        self.export_btn.clicked.connect(self._export)

    def _export(self):
        # simulate report
        report = {"status":"success","timestamp":datetime.datetime.now(datetime.timezone.utc).isoformat(),"notes":"Simulated run"}
        self.report_view.setPlainText(json.dumps(report, indent=2))
        QtWidgets.QMessageBox.information(self, "Export", "Simulated run package created (not really).")

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MeshPrep - Improved v1 Prototype")
        self.resize(1100, 700)
        central = QtWidgets.QWidget()
        central.setStyleSheet(f"background:{BG};")
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # Left: step list
        self.steps_list = QtWidgets.QListWidget()
        self.steps_list.addItems(["1. Environment","2. Select Model","3. Suggest Filter","4. Dry-Run","5. Execute","6. Results"])
        self.steps_list.setFixedWidth(180)
        self.steps_list.setStyleSheet("QListWidget{background:%s;color:%s;} QListWidget::item:selected{background:#22333b;}" % (PANEL, TEXT))
        layout.addWidget(self.steps_list)

        # Center: stacked pages
        self.stack = QtWidgets.QStackedWidget()
        self.env_widget = EnvWidget()
        self.select_widget = SelectionWidget()
        self.suggest_widget = SuggestWidget()
        self.dry_widget = DryRunWidget()
        self.exec_widget = ExecuteWidget()
        self.results_widget = ResultsWidget()
        for w in [self.env_widget, self.select_widget, self.suggest_widget, self.dry_widget, self.exec_widget, self.results_widget]:
            self.stack.addWidget(w)
        layout.addWidget(self.stack, 1)

        # Right: global log
        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("Global Log"))
        self.global_log = QtWidgets.QPlainTextEdit()
        self.global_log.setReadOnly(True)
        self.global_log.setFixedWidth(300)
        right.addWidget(self.global_log)
        layout.addLayout(right)

        # wiring
        self.steps_list.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.steps_list.setCurrentRow(0)
        self.env_widget.auto_setup_btn.clicked.connect(self._auto_setup)
        self.env_widget.instructions_btn.clicked.connect(self._show_instructions)
        self.select_widget.file_btn.clicked.connect(self._select_file)
        self.select_widget.load_filter_btn.clicked.connect(self._load_filter)
        # when suggested page saved/edited, update global log
        self.suggest_widget.save_preset_btn.clicked.connect(lambda: self.global_log.appendPlainText("Preset saved by user"))
        self.dry_widget.run_btn.clicked.connect(lambda: self.global_log.appendPlainText("Dry-run started"))
        self.exec_widget.start_btn.clicked.connect(lambda: self.global_log.appendPlainText("Run started"))
        self.exec_widget.stop_btn.clicked.connect(lambda: self.global_log.appendPlainText("Run stopped"))

    # simulated behaviors
    def _auto_setup(self):
        self.global_log.appendPlainText("Auto-setup initiated")
        self.env_widget.status_view.clear()
        steps = ["Checking Python","Checking trimesh","Checking pymeshfix","Checking meshio","Checking Blender (optional)","Finalizing"]
        self._env_idx = 0
        self._env_timer = QtCore.QTimer(self)
        self._env_timer.timeout.connect(lambda: self._env_step(steps))
        self._env_timer.start(500)

    def _env_step(self, steps):
        if self._env_idx < len(steps):
            s = steps[self._env_idx]
            # use appendPlainText for QPlainTextEdit
            self.env_widget.status_view.appendPlainText(f"[INFO] {s}... OK")
            self.global_log.appendPlainText(f"{s}: OK")
            self._env_idx += 1
        else:
            self._env_timer.stop()
            self.env_widget.status_view.appendPlainText("[SUCCESS] Environment ready (simulated)")
            self.global_log.appendPlainText("Environment setup complete")
            self.steps_list.setCurrentRow(1)

    def _show_instructions(self):
        QtWidgets.QMessageBox.information(self, "Manual setup",
                                          "See docs/INSTALL.md or gui/README.md for manual steps to prepare the environment.")

    def _select_file(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select STL", "", "STL Files (*.stl);;All Files (*)")
        if fname:
            self.select_widget.file_label.setText(fname)
            self.global_log.appendPlainText(f"Selected model: {fname}")
            # simulate profile detection
            self.suggest_widget.diag.setPlainText("Analyzing model...\ncomputing diagnostics...")
            QtCore.QTimer.singleShot(800, self._populate_suggest)
            self.steps_list.setCurrentRow(2)

    def _populate_suggest(self):
        # fake diagnostics and suggested actions
        diag = "watertight=False\nholes=3\ncomponents=1\nnormal_consistency=0.6\nprofile=holes-only\nconfidence=0.82"
        self.suggest_widget.diag.setPlainText(diag)
        self.suggest_widget.profile_lbl.setText("holes-only (confidence 0.82)")
        self.suggest_widget.actions.clear()
        for a in ["trimesh_basic","fill_holes(max_size=500)","recalculate_normals","validate"]:
            self.suggest_widget.actions.addItem(a)
        self.global_log.appendPlainText("Suggested filter generated")

    def _load_filter(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load filter", "", "JSON/YAML (*.json *.yaml *.yml);;All Files (*)")
        if fname:
            self.global_log.appendPlainText(f"Loaded filter script: {fname}")
            try:
                with open(fname, 'r') as f:
                    data = json.load(f)
                self.suggest_widget.actions.clear()
                for a in data.get('actions', []):
                    self.suggest_widget.actions.addItem(a)
                self.suggest_widget.diag.setPlainText(f"Loaded preset: {data.get('name','<unnamed>')}")
            except Exception:
                QtWidgets.QMessageBox.warning(self, "Load failed", "Could not load as JSON; showing filename only")
                self.suggest_widget.diag.setPlainText(f"Loaded filter: {fname}")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
