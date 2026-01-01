# Copyright 2025 Dragon Ace
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""Filter Script Editor dialog for MeshPrep GUI."""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QSplitter, QWidget,
    QLabel, QPushButton, QListWidget, QListWidgetItem,
    QTreeWidget, QTreeWidgetItem, QLineEdit, QFrame,
    QScrollArea, QMessageBox, QDialogButtonBox,
)
from PySide6.QtCore import Qt, Signal

from ..core.actions import ActionRegistry, ActionCategory, get_action_registry
from ..core.filter_script import FilterScript, FilterAction
from .widgets import ActionCard, ParameterEditor
from .styles import apply_theme


class FilterLibraryPanel(QFrame):
    """Panel showing available filter actions."""
    
    action_selected = Signal(str)  # action name
    action_double_clicked = Signal(str)  # action name
    
    def __init__(self, registry: ActionRegistry, parent=None):
        super().__init__(parent)
        self.registry = registry
        self.setObjectName("panel")
        self._selected_action: str = None
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Header
        header = QLabel("Filter Library")
        header.setObjectName("header")
        layout.addWidget(header)
        
        # Search
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search actions...")
        self.search.textChanged.connect(self._filter_actions)
        layout.addWidget(self.search)
        
        # Category tree
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemClicked.connect(self._on_item_clicked)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.tree)
        
        # Add button
        self.add_btn = QPushButton("+ Add Selected Action")
        self.add_btn.setObjectName("primary")
        self.add_btn.setEnabled(False)
        self.add_btn.clicked.connect(self._on_add_clicked)
        layout.addWidget(self.add_btn)
        
        self._populate_tree()
    
    def _populate_tree(self):
        """Populate the tree with actions by category."""
        self.tree.clear()
        
        # Group actions by category
        categories = {}
        for action in self.registry.list_actions():
            cat = action.category
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(action)
        
        # Create tree items
        for category in ActionCategory:
            if category not in categories:
                continue
            
            cat_item = QTreeWidgetItem([category.value])
            cat_item.setFlags(cat_item.flags() & ~Qt.ItemIsSelectable)
            self.tree.addTopLevelItem(cat_item)
            
            for action in categories[category]:
                action_item = QTreeWidgetItem([f"{action.display_name} [{action.tool}]"])
                action_item.setData(0, Qt.UserRole, action.name)
                action_item.setToolTip(0, action.description)
                cat_item.addChild(action_item)
            
            cat_item.setExpanded(True)
    
    def _filter_actions(self, text: str):
        """Filter actions by search text."""
        text = text.lower()
        
        for i in range(self.tree.topLevelItemCount()):
            cat_item = self.tree.topLevelItem(i)
            visible_children = 0
            
            for j in range(cat_item.childCount()):
                action_item = cat_item.child(j)
                action_name = action_item.data(0, Qt.UserRole)
                action = self.registry.get(action_name)
                
                visible = (
                    text in action_name.lower() or
                    text in action.display_name.lower() or
                    text in action.description.lower() or
                    text in action.tool.lower()
                )
                action_item.setHidden(not visible)
                if visible:
                    visible_children += 1
            
            cat_item.setHidden(visible_children == 0)
    
    def _on_item_clicked(self, item: QTreeWidgetItem, column: int):
        action_name = item.data(0, Qt.UserRole)
        if action_name:
            self._selected_action = action_name
            self.add_btn.setEnabled(True)
            self.action_selected.emit(action_name)
        else:
            # Category item clicked
            self._selected_action = None
            self.add_btn.setEnabled(False)
    
    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        action_name = item.data(0, Qt.UserRole)
        if action_name:
            self.action_double_clicked.emit(action_name)
    
    def _on_add_clicked(self):
        """Handle Add button click."""
        if self._selected_action:
            self.action_double_clicked.emit(self._selected_action)


class ActionListPanel(QFrame):
    """Panel showing the current filter script actions."""
    
    action_selected = Signal(int)  # index
    actions_changed = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("panel")
        self.actions: list[FilterAction] = []
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Header
        header = QHBoxLayout()
        title = QLabel("Filter Script Actions")
        title.setObjectName("header")
        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)
        
        # Action list
        self.list = QListWidget()
        self.list.setDragDropMode(QListWidget.InternalMove)
        self.list.currentRowChanged.connect(self.action_selected.emit)
        self.list.model().rowsMoved.connect(self._on_rows_moved)
        layout.addWidget(self.list)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.move_up_btn = QPushButton("▲ Up")
        self.move_up_btn.clicked.connect(self._move_up)
        btn_layout.addWidget(self.move_up_btn)
        
        self.move_down_btn = QPushButton("▼ Down")
        self.move_down_btn.clicked.connect(self._move_down)
        btn_layout.addWidget(self.move_down_btn)
        
        self.delete_btn = QPushButton("× Delete")
        self.delete_btn.clicked.connect(self._delete_selected)
        btn_layout.addWidget(self.delete_btn)
        
        layout.addLayout(btn_layout)
    
    def set_actions(self, actions: list[FilterAction]):
        """Set the list of actions."""
        self.actions = list(actions)
        self._refresh_list()
    
    def get_actions(self) -> list[FilterAction]:
        """Get the current list of actions."""
        return self.actions
    
    def add_action(self, action: FilterAction):
        """Add an action to the list."""
        self.actions.append(action)
        self._refresh_list()
        self.list.setCurrentRow(len(self.actions) - 1)
        self.actions_changed.emit()
    
    def _refresh_list(self, preserve_selection: bool = False):
        """Refresh the list widget."""
        current_row = self.list.currentRow() if preserve_selection else -1
        
        self.list.clear()
        for i, action in enumerate(self.actions):
            item = QListWidgetItem(f"{i + 1}. {action.name}")
            if action.params:
                params_str = ", ".join(f"{k}={v}" for k, v in action.params.items())
                item.setToolTip(params_str)
            self.list.addItem(item)
        
        # Restore selection if requested
        if preserve_selection and 0 <= current_row < len(self.actions):
            self.list.setCurrentRow(current_row)
    
    def _move_up(self):
        row = self.list.currentRow()
        if row > 0:
            self.actions[row], self.actions[row - 1] = self.actions[row - 1], self.actions[row]
            self._refresh_list()
            self.list.setCurrentRow(row - 1)
            self.actions_changed.emit()
    
    def _move_down(self):
        row = self.list.currentRow()
        if row < len(self.actions) - 1:
            self.actions[row], self.actions[row + 1] = self.actions[row + 1], self.actions[row]
            self._refresh_list()
            self.list.setCurrentRow(row + 1)
            self.actions_changed.emit()
    
    def _delete_selected(self):
        row = self.list.currentRow()
        if 0 <= row < len(self.actions):
            self.actions.pop(row)
            self._refresh_list()
            if self.actions:
                self.list.setCurrentRow(min(row, len(self.actions) - 1))
            self.actions_changed.emit()
    
    def _on_rows_moved(self):
        # Sync actions list with widget order
        new_actions = []
        for i in range(self.list.count()):
            item = self.list.item(i)
            # Extract index from text "N. action_name"
            text = item.text()
            old_idx = int(text.split(".")[0]) - 1
            if 0 <= old_idx < len(self.actions):
                new_actions.append(self.actions[old_idx])
        
        self.actions = new_actions
        self._refresh_list()
        self.actions_changed.emit()


class ActionParameterPanel(QFrame):
    """Panel for editing action parameters."""
    
    params_changed = Signal()
    
    def __init__(self, registry: ActionRegistry, parent=None):
        super().__init__(parent)
        self.registry = registry
        self.current_action: FilterAction = None
        self.setObjectName("panel")
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        
        self.title = QLabel("Action Details")
        self.title.setObjectName("header")
        layout.addWidget(self.title)
        
        self.description = QLabel("")
        self.description.setWordWrap(True)
        self.description.setObjectName("secondary")
        layout.addWidget(self.description)
        
        # Scroll area for parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.params_container = QWidget()
        self.params_layout = QVBoxLayout(self.params_container)
        self.params_layout.setAlignment(Qt.AlignTop)
        scroll.setWidget(self.params_container)
        
        layout.addWidget(scroll, stretch=1)
        
        self.param_widgets = {}
    
    def set_action(self, action: FilterAction):
        """Set the action to edit."""
        self.current_action = action
        
        # Clear existing widgets
        for widget in self.param_widgets.values():
            widget.deleteLater()
        self.param_widgets.clear()
        
        # Clear layout
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if not action:
            self.title.setText("Action Details")
            self.description.setText("Select an action to edit its parameters.")
            return
        
        # Get action definition
        action_def = self.registry.get(action.name)
        if not action_def:
            self.title.setText(f"Unknown: {action.name}")
            self.description.setText("This action is not in the registry.")
            return
        
        self.title.setText(action_def.display_name)
        self.description.setText(action_def.description)
        
        # Create parameter widgets
        for param in action_def.parameters:
            self._create_param_widget(param, action.params.get(param.name, param.default))
        
        self.params_layout.addStretch()
    
    def _create_param_widget(self, param, current_value):
        """Create a widget for editing a parameter."""
        from PySide6.QtWidgets import QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QLineEdit
        
        # Label
        label = QLabel(f"{param.name}:")
        label.setToolTip(param.description)
        self.params_layout.addWidget(label)
        
        # Input widget
        if param.param_type == "bool":
            widget = QCheckBox()
            widget.setChecked(bool(current_value) if current_value is not None else False)
            widget.stateChanged.connect(lambda: self._on_param_changed(param.name, widget.isChecked()))
        elif param.param_type == "int":
            widget = QSpinBox()
            widget.setRange(int(param.min_value or -999999), int(param.max_value or 999999))
            if current_value is not None:
                widget.setValue(int(current_value))
            widget.valueChanged.connect(lambda v: self._on_param_changed(param.name, v))
        elif param.param_type == "float":
            widget = QDoubleSpinBox()
            widget.setRange(param.min_value or -999999.0, param.max_value or 999999.0)
            widget.setDecimals(6)
            if current_value is not None:
                widget.setValue(float(current_value))
            widget.valueChanged.connect(lambda v: self._on_param_changed(param.name, v))
        elif param.param_type == "enum":
            widget = QComboBox()
            widget.addItems(param.enum_values)
            if current_value:
                widget.setCurrentText(str(current_value))
            widget.currentTextChanged.connect(lambda v: self._on_param_changed(param.name, v))
        else:
            widget = QLineEdit()
            if current_value is not None:
                widget.setText(str(current_value))
            widget.textChanged.connect(lambda v: self._on_param_changed(param.name, v))
        
        self.params_layout.addWidget(widget)
        self.param_widgets[param.name] = widget
    
    def _on_param_changed(self, name: str, value):
        """Handle parameter value change."""
        if self.current_action:
            if value is not None and value != "":
                self.current_action.params[name] = value
            elif name in self.current_action.params:
                del self.current_action.params[name]
            self.params_changed.emit()


class FilterScriptEditor(QDialog):
    """Dialog for editing filter scripts."""
    
    def __init__(self, script: FilterScript = None, parent=None):
        super().__init__(parent)
        self.registry = get_action_registry()
        self.original_script = script
        self.script = self._copy_script(script) if script else self._new_script()
        
        self.setWindowTitle("Filter Script Editor")
        self.setMinimumSize(1000, 600)
        self._setup_ui()
        apply_theme(self, "dark")
    
    def _copy_script(self, script: FilterScript) -> FilterScript:
        """Create a deep copy of a filter script."""
        return FilterScript.from_dict(script.to_dict())
    
    def _new_script(self) -> FilterScript:
        """Create a new empty filter script."""
        return FilterScript(name="new-filter-script")
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Main splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: Filter library
        self.library_panel = FilterLibraryPanel(self.registry)
        self.library_panel.action_selected.connect(self._on_library_action_selected)
        self.library_panel.action_double_clicked.connect(self._add_action)
        splitter.addWidget(self.library_panel)
        
        # Middle panel: Action list
        self.action_list_panel = ActionListPanel()
        self.action_list_panel.set_actions(self.script.actions)
        self.action_list_panel.action_selected.connect(self._on_action_selected)
        self.action_list_panel.actions_changed.connect(self._on_actions_changed)
        splitter.addWidget(self.action_list_panel)
        
        # Right panel: Parameter editor
        self.param_panel = ActionParameterPanel(self.registry)
        self.param_panel.params_changed.connect(self._on_params_changed)
        splitter.addWidget(self.param_panel)
        
        splitter.setSizes([250, 400, 350])
        layout.addWidget(splitter)
        
        # Template buttons
        template_layout = QHBoxLayout()
        template_layout.addWidget(QLabel("Quick Templates:"))
        
        templates = [
            ("Basic Cleanup", ["trimesh_basic", "merge_vertices", "remove_degenerate_faces", "validate"]),
            ("Hole Fill", ["fill_holes", "recalculate_normals", "validate"]),
            ("Full Repair", ["trimesh_basic", "pymeshfix_repair", "fill_holes", "fix_normals", "validate"]),
        ]
        
        for name, actions in templates:
            btn = QPushButton(name)
            btn.clicked.connect(lambda checked, a=actions: self._apply_template(a))
            template_layout.addWidget(btn)
        
        template_layout.addStretch()
        layout.addLayout(template_layout)
        
        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def _on_library_action_selected(self, action_name: str):
        """Handle action selection in library."""
        pass  # Could show preview
    
    def _add_action(self, action_name: str):
        """Add an action from the library."""
        action = FilterAction(name=action_name)
        self.action_list_panel.add_action(action)
    
    def _on_action_selected(self, index: int):
        """Handle action selection in the list."""
        if 0 <= index < len(self.action_list_panel.actions):
            action = self.action_list_panel.actions[index]
            self.param_panel.set_action(action)
        else:
            self.param_panel.set_action(None)
    
    def _on_actions_changed(self):
        """Handle changes to the action list."""
        self.script.actions = self.action_list_panel.get_actions()
    
    def _on_params_changed(self):
        """Handle parameter changes."""
        # Preserve selection when refreshing due to parameter changes
        self.action_list_panel._refresh_list(preserve_selection=True)
    
    def _apply_template(self, action_names: list[str]):
        """Apply a template to the script."""
        result = QMessageBox.question(
            self, "Apply Template",
            "This will replace all current actions. Continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if result == QMessageBox.Yes:
            self.script.actions = [FilterAction(name=name) for name in action_names]
            self.action_list_panel.set_actions(self.script.actions)
    
    def get_script(self) -> FilterScript:
        """Get the edited filter script."""
        self.script.actions = self.action_list_panel.get_actions()
        return self.script
