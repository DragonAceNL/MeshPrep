import sys
import random
from PySide6 import QtWidgets, QtGui, QtCore

# Sci-fi styled colors
BG = "#07101a"
PANEL = "#0b1620"
ACCENT = "#00ffd5"
TEXT = "#cfefff"
BTN = "#07202a"

class SciFiHeader(QtWidgets.QWidget):
    def __init__(self, title="MeshPrep - SciFi UI v2", parent=None):
        super().__init__(parent)
        self.setFixedHeight(72)
        self.setStyleSheet(f"background:{PANEL}; border-bottom:1px solid #113;")
        layout = QtWidgets.QHBoxLayout(self)
        logo = QtWidgets.QLabel()
        pix = QtGui.QPixmap(40,40)
        pix.fill(QtGui.QColor(5, 200, 180))
        logo.setPixmap(pix)
        title_lbl = QtWidgets.QLabel(title)
        title_lbl.setStyleSheet(f"font: 18pt 'Segoe UI'; color: {ACCENT}")
        layout.addWidget(logo)
        layout.addSpacing(8)
        layout.addWidget(title_lbl)
        layout.addStretch()

class CheckEnvPage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{BG}; color:{TEXT}")
        layout = QtWidgets.QVBoxLayout(self)
        self.status_label = QtWidgets.QLabel("Environment: unknown")
        self.status_label.setStyleSheet("font: 12pt 'Consolas';")
        self.auto_setup_btn = QtWidgets.QPushButton("Auto-setup environment")
        self.instructions_btn = QtWidgets.QPushButton("Show manual setup instructions")
        layout.addWidget(self.status_label)
        layout.addWidget(self.auto_setup_btn)
        layout.addWidget(self.instructions_btn)
        layout.addStretch()

class ModelSelectionPage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{BG}; color:{TEXT}")
        layout = QtWidgets.QVBoxLayout(self)
        self.file_label = QtWidgets.QLabel("No file selected")
        self.file_picker = QtWidgets.QPushButton("Select STL file...")
        self.radio_auto = QtWidgets.QRadioButton("Auto-detect profile and suggest filter")
        self.radio_manual = QtWidgets.QRadioButton("Use my filter script (file or URL)")
        self.radio_auto.setChecked(True)
        self.load_filter_btn = QtWidgets.QPushButton("Load filter script")
        layout.addWidget(self.file_label)
        layout.addWidget(self.file_picker)
        layout.addWidget(self.radio_auto)
        layout.addWidget(self.radio_manual)
        layout.addWidget(self.load_filter_btn)
        layout.addStretch()

class SuggestedFilterPage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{BG}; color:{TEXT}")
        layout = QtWidgets.QHBoxLayout(self)
        # Left: diagnostics
        left = QtWidgets.QVBoxLayout()
        self.diagnostics = QtWidgets.QTextEdit()
        self.diagnostics.setReadOnly(True)
        left.addWidget(QtWidgets.QLabel("Diagnostics"))
        left.addWidget(self.diagnostics)
        # Right: actions list
        right = QtWidgets.QVBoxLayout()
        self.actions_list = QtWidgets.QListWidget()
        btns = QtWidgets.QHBoxLayout()
        self.add_btn = QtWidgets.QPushButton("Add")
        self.edit_btn = QtWidgets.QPushButton("Edit")
        self.remove_btn = QtWidgets.QPushButton("Remove")
        btns.addWidget(self.add_btn)
        btns.addWidget(self.edit_btn)
        btns.addWidget(self.remove_btn)
        right.addWidget(QtWidgets.QLabel("Suggested filter script (editable)"))
        right.addWidget(self.actions_list)
        right.addLayout(btns)
        layout.addLayout(left, 2)
        layout.addLayout(right, 1)

        # wire simple add/edit/remove
        self.add_btn.clicked.connect(self.add_action)
        self.edit_btn.clicked.connect(self.edit_action)
        self.remove_btn.clicked.connect(self.remove_action)

    def add_action(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Add Action", "Action name:")
        if ok and name:
            self.actions_list.addItem(name)

    def edit_action(self):
        item = self.actions_list.currentItem()
        if not item:
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "Edit Action", "Action name:", text=item.text())
        if ok and name:
            item.setText(name)

    def remove_action(self):
        row = self.actions_list.currentRow()
        if row>=0:
            self.actions_list.takeItem(row)

class DryRunPage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{BG}; color:{TEXT}")
        layout = QtWidgets.QVBoxLayout(self)
        self.steps_table = QtWidgets.QTableWidget(0,3)
        self.steps_table.setHorizontalHeaderLabels(["Step", "Status", "Notes"])
        layout.addWidget(self.steps_table)
        self.run_dry_btn = QtWidgets.QPushButton("Run dry-run")
        layout.addWidget(self.run_dry_btn)
        layout.addStretch()
        self.run_dry_btn.clicked.connect(self.run_dry)

    def run_dry(self):
        # populate table from suggested actions if available via parent lookup
        main = self.parent().parent()  # main container hack
        suggested = None
        try:
            suggested = main.pages['suggest'].actions_list
        except Exception:
            pass
        self.steps_table.setRowCount(0)
        actions = []
        if suggested:
            actions = [suggested.item(i).text() for i in range(suggested.count())]
        if not actions:
            actions = ["trimesh_basic", "fill_holes", "recalculate_normals", "validate"]
        for a in actions:
            r = self.steps_table.rowCount()
            self.steps_table.insertRow(r)
            self.steps_table.setItem(r,0, QtWidgets.QTableWidgetItem(a))
            self.steps_table.setItem(r,1, QtWidgets.QTableWidgetItem("pending"))
            self.steps_table.setItem(r,2, QtWidgets.QTableWidgetItem(""))
        # simulate progression
        self._i = 0
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._advance)
        self._timer.start(700)

    def _advance(self):
        if self._i >= self.steps_table.rowCount():
            self._timer.stop()
            return
        # random success/fail for demo
        status = random.choice(["ok","ok","ok","warn"]) 
        self.steps_table.setItem(self._i,1, QtWidgets.QTableWidgetItem(status))
        self.steps_table.setItem(self._i,2, QtWidgets.QTableWidgetItem("Simulated result"))
        self._i += 1

class ExecutePage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{BG}; color:{TEXT}")
        layout = QtWidgets.QVBoxLayout(self)
        self.progress = QtWidgets.QProgressBar()
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.progress)
        layout.addWidget(QtWidgets.QLabel("Run log"))
        layout.addWidget(self.log_view)
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(self.start_btn)
        btns.addWidget(self.stop_btn)
        layout.addLayout(btns)
        self.start_btn.clicked.connect(self.start_run)
        self.stop_btn.clicked.connect(self.stop_run)
        self._running = False

    def start_run(self):
        if self._running:
            return
        self._running = True
        self.log_view.clear()
        self.progress.setValue(0)
        self._count = 0
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(300)

    def stop_run(self):
        if not self._running:
            return
        self._timer.stop()
        self._running = False
        self.log_view.appendPlainText("Run stopped by user")

    def _tick(self):
        # append synthetic terminal lines
        lines = [
            "[INFO] Loading model...",
            "[DEBUG] Running trimesh_basic...",
            "[INFO] fill_holes applied (12 holes)",
            "[WARN] small components removed",
            "[INFO] Running pymeshfix...",
            "[INFO] Validation: watertight=True",
            "[INFO] Exporting output..."
        ]
        if self._count < len(lines):
            self.log_view.appendPlainText(lines[self._count])
            self._count += 1
            self.progress.setValue(int((self._count/len(lines))*100))
        else:
            self.log_view.appendPlainText("[SUCCESS] Run complete")
            self.progress.setValue(100)
            self._timer.stop()
            self._running = False

class ResultsPage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{BG}; color:{TEXT}")
        layout = QtWidgets.QVBoxLayout(self)
        self.summary = QtWidgets.QTextEdit()
        self.summary.setReadOnly(True)
        self.export_btn = QtWidgets.QPushButton("Export run package")
        layout.addWidget(QtWidgets.QLabel("Run summary"))
        layout.addWidget(self.summary)
        layout.addWidget(self.export_btn)
        layout.addStretch()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MeshPrep - SciFi UI v2")
        self.resize(1200, 800)
        central = QtWidgets.QWidget()
        # ensure central background matches sci-fi dark theme
        central.setStyleSheet(f"background: {BG};")
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        self.header = SciFiHeader("MeshPrep - SciFi UI v2")
        main_layout.addWidget(self.header)
        body = QtWidgets.QHBoxLayout()
        # sidebar
        sidebar = QtWidgets.QFrame()
        sidebar.setFixedWidth(260)
        sidebar.setStyleSheet(f"background:{PANEL};")
        s_layout = QtWidgets.QVBoxLayout(sidebar)
        self.nav_buttons = {}
        for name in [("checkenv","Env"), ("select","Select"), ("suggest","Suggest"), ("dryrun","DryRun"), ("execute","Execute"), ("results","Results")]:
            key, label = name
            b = QtWidgets.QPushButton(label)
            b.setCheckable(True)
            b.setStyleSheet(f"background: {BTN}; color: {TEXT}; border: none; padding:12px; text-align:left;")
            s_layout.addWidget(b)
            self.nav_buttons[key] = b
        s_layout.addStretch()
        body.addWidget(sidebar)
        # main stack
        self.stack = QtWidgets.QStackedWidget()
        # stack background should match overall theme
        self.stack.setStyleSheet(f"background: {BG}; border: none;")
        self.pages = {}
        self.pages['checkenv'] = CheckEnvPage()
        self.pages['select'] = ModelSelectionPage()
        self.pages['suggest'] = SuggestedFilterPage()
        self.pages['dryrun'] = DryRunPage()
        self.pages['execute'] = ExecutePage()
        self.pages['results'] = ResultsPage()
        for key in ['checkenv','select','suggest','dryrun','execute','results']:
            self.stack.addWidget(self.pages[key])
        body.addWidget(self.stack)
        main_layout.addLayout(body)
        # footer
        footer = QtWidgets.QLabel("Status: Idle | Ready")
        footer.setStyleSheet(f"background:{PANEL}; color: {ACCENT}; padding:6px")
        main_layout.addWidget(footer)

        # wire navigation
        for i, key in enumerate(['checkenv','select','suggest','dryrun','execute','results']):
            btn = self.nav_buttons[key]
            btn.clicked.connect(lambda checked, idx=i: self.switch(idx))
        self.switch(0)

        # wire some interactions
        self.pages['select'].file_picker.clicked.connect(self.choose_file)
        self.pages['select'].load_filter_btn.clicked.connect(self.load_filter)
        self.pages['checkenv'].auto_setup_btn.clicked.connect(self.mock_setup)

    def switch(self, idx):
        # animate fade
        current = self.stack.currentWidget()
        next_widget = self.stack.widget(idx)
        if current is next_widget:
            return
        # simple cross-fade
        for w in (current, next_widget):
            if w:
                effect = QtWidgets.QGraphicsOpacityEffect(w)
                w.setGraphicsEffect(effect)
        anim_out = QtCore.QPropertyAnimation(current.graphicsEffect(), b"opacity") if current else None
        anim_in = QtCore.QPropertyAnimation(next_widget.graphicsEffect(), b"opacity")
        if anim_out:
            anim_out.setDuration(250)
            anim_out.setStartValue(1.0)
            anim_out.setEndValue(0.0)
        anim_in.setDuration(250)
        anim_in.setStartValue(0.0)
        anim_in.setEndValue(1.0)
        def on_out_finished():
            self.stack.setCurrentIndex(idx)
            anim_in.start()
        if anim_out:
            anim_out.finished.connect(on_out_finished)
            anim_out.start()
        else:
            self.stack.setCurrentIndex(idx)
            anim_in.start()
        # update sidebar active state
        keys = ['checkenv','select','suggest','dryrun','execute','results']
        for i,k in enumerate(keys):
            self.nav_buttons[k].setChecked(i==idx)

    def choose_file(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select STL", "", "STL Files (*.stl);;All Files (*)")
        if fname:
            self.pages['select'].file_label.setText(fname)
            # fill diagnostics mock
            self.pages['suggest'].diagnostics.setPlainText("Detected: holes=3\ncomponents=1\nwatertight=False\nprofile=holes-only\nconfidence=0.87")
            # auto populate suggested actions
            self.pages['suggest'].actions_list.clear()
            for a in ["trimesh_basic","fill_holes","recalculate_normals","validate"]:
                self.pages['suggest'].actions_list.addItem(a)

    def load_filter(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load filter script", "", "JSON/YAML (*.json *.yaml *.yml);;All Files (*)")
        if fname:
            # naive load: show filename in diagnostics
            self.pages['suggest'].diagnostics.setPlainText(f"Loaded filter: {fname}")

    def mock_setup(self):
        self.pages['checkenv'].status_label.setText("Environment: OK (mock)")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
