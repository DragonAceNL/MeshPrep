import sys
from PySide6 import QtWidgets, QtCore

class CheckEnvPage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        self.status_label = QtWidgets.QLabel("Environment check not run")
        self.auto_setup_btn = QtWidgets.QPushButton("Auto-setup environment")
        self.instructions_btn = QtWidgets.QPushButton("Show manual setup instructions")
        layout.addWidget(self.status_label)
        layout.addWidget(self.auto_setup_btn)
        layout.addWidget(self.instructions_btn)

class ModelSelectionPage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        self.file_picker = QtWidgets.QPushButton("Select STL file...")
        self.radio_auto = QtWidgets.QRadioButton("Auto-detect profile and suggest filter")
        self.radio_manual = QtWidgets.QRadioButton("Use my filter script (file or URL)")
        self.radio_auto.setChecked(True)
        layout.addWidget(self.file_picker)
        layout.addWidget(self.radio_auto)
        layout.addWidget(self.radio_manual)

class SuggestedFilterPage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
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
        layout.addLayout(left)
        layout.addLayout(right)

class DryRunPage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        self.steps_table = QtWidgets.QTableWidget(0,3)
        self.steps_table.setHorizontalHeaderLabels(["Step", "Status", "Notes"])
        layout.addWidget(self.steps_table)
        self.run_dry_btn = QtWidgets.QPushButton("Run dry-run")
        layout.addWidget(self.run_dry_btn)

class ExecutePage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
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

class ResultsPage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        self.summary = QtWidgets.QTextEdit()
        self.summary.setReadOnly(True)
        self.export_btn = QtWidgets.QPushButton("Export run package")
        layout.addWidget(QtWidgets.QLabel("Run summary"))
        layout.addWidget(self.summary)
        layout.addWidget(self.export_btn)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MeshPrep - Prototype GUI")
        self.resize(1000, 700)

        # Central widget with stacked pages
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.stack = QtWidgets.QStackedWidget()
        self.pages = {}
        self.pages['checkenv'] = CheckEnvPage()
        self.pages['select'] = ModelSelectionPage()
        self.pages['suggest'] = SuggestedFilterPage()
        self.pages['dryrun'] = DryRunPage()
        self.pages['execute'] = ExecutePage()
        self.pages['results'] = ResultsPage()
        for p in ['checkenv','select','suggest','dryrun','execute','results']:
            self.stack.addWidget(self.pages[p])

        # Navigation
        nav = QtWidgets.QHBoxLayout()
        self.prev_btn = QtWidgets.QPushButton("Previous")
        self.next_btn = QtWidgets.QPushButton("Next")
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)

        layout.addWidget(self.stack)
        layout.addLayout(nav)

        self.current_index = 0
        self.stack.setCurrentIndex(self.current_index)

        self.prev_btn.clicked.connect(self.go_prev)
        self.next_btn.clicked.connect(self.go_next)

    def go_prev(self):
        if self.current_index>0:
            self.current_index -= 1
            self.stack.setCurrentIndex(self.current_index)

    def go_next(self):
        if self.current_index < self.stack.count()-1:
            self.current_index += 1
            self.stack.setCurrentIndex(self.current_index)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
