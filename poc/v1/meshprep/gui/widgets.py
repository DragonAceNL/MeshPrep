# Copyright 2025 Dragon Ace
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""Custom widgets for MeshPrep GUI."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QListWidget, QListWidgetItem, QProgressBar,
    QTextEdit, QGroupBox, QScrollArea, QSizePolicy,
    QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QLineEdit,
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QFont, QColor

from typing import Optional, Any


class StepIndicator(QWidget):
    """Step indicator widget for wizard navigation."""
    
    step_clicked = Signal(int)
    
    def __init__(self, steps: list[str], parent=None):
        super().__init__(parent)
        self.steps = steps
        self.current_step = 0
        self.completed_steps = set()
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        
        self.step_buttons = []
        for i, step_name in enumerate(self.steps):
            btn = QPushButton(f"{i + 1}. {step_name}")
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, idx=i: self._on_step_clicked(idx))
            btn.setMinimumHeight(36)
            self.step_buttons.append(btn)
            layout.addWidget(btn)
        
        layout.addStretch()
        self._update_buttons()
    
    def _on_step_clicked(self, index: int):
        if index <= max(self.completed_steps, default=-1) + 1:
            self.step_clicked.emit(index)
    
    def set_current_step(self, step: int):
        self.current_step = step
        self._update_buttons()
    
    def mark_completed(self, step: int):
        self.completed_steps.add(step)
        self._update_buttons()
    
    def _update_buttons(self):
        for i, btn in enumerate(self.step_buttons):
            btn.setChecked(i == self.current_step)
            
            # Enable only completed steps and next step
            enabled = i <= max(self.completed_steps, default=-1) + 1
            btn.setEnabled(enabled)
            
            # Add visual indicators
            if i in self.completed_steps:
                btn.setText(f"✓ {i + 1}. {self.steps[i]}")
            elif i == self.current_step:
                btn.setText(f"► {i + 1}. {self.steps[i]}")
            else:
                btn.setText(f"  {i + 1}. {self.steps[i]}")


class DiagnosticsPanel(QFrame):
    """Panel displaying mesh diagnostics."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("panel")
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        
        title = QLabel("Diagnostics")
        title.setObjectName("header")
        layout.addWidget(title)
        
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setMinimumHeight(200)
        layout.addWidget(self.text)
    
    def set_diagnostics(self, diagnostics):
        """Set diagnostics to display."""
        if diagnostics:
            self.text.setPlainText(diagnostics.summary())
        else:
            self.text.setPlainText("No diagnostics available")
    
    def clear(self):
        self.text.clear()


class ActionCard(QFrame):
    """Card widget for displaying a filter action."""
    
    clicked = Signal()
    delete_clicked = Signal()
    move_up_clicked = Signal()
    move_down_clicked = Signal()
    
    def __init__(self, action_name: str, display_name: str, 
                 tool: str, params: dict = None, parent=None):
        super().__init__(parent)
        self.action_name = action_name
        self.display_name = display_name
        self.tool = tool
        self.params = params or {}
        self.setObjectName("panel")
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Main info
        info_layout = QVBoxLayout()
        
        name_label = QLabel(self.display_name)
        name_label.setStyleSheet("font-weight: bold;")
        info_layout.addWidget(name_label)
        
        tool_label = QLabel(f"[{self.tool}]")
        tool_label.setObjectName("secondary")
        info_layout.addWidget(tool_label)
        
        if self.params:
            params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
            params_label = QLabel(params_str)
            params_label.setObjectName("secondary")
            info_layout.addWidget(params_label)
        
        layout.addLayout(info_layout, stretch=1)
        
        # Buttons
        btn_layout = QVBoxLayout()
        
        up_btn = QPushButton("▲")
        up_btn.setMaximumWidth(30)
        up_btn.clicked.connect(self.move_up_clicked.emit)
        btn_layout.addWidget(up_btn)
        
        down_btn = QPushButton("▼")
        down_btn.setMaximumWidth(30)
        down_btn.clicked.connect(self.move_down_clicked.emit)
        btn_layout.addWidget(down_btn)
        
        del_btn = QPushButton("×")
        del_btn.setMaximumWidth(30)
        del_btn.setObjectName("danger")
        del_btn.clicked.connect(self.delete_clicked.emit)
        btn_layout.addWidget(del_btn)
        
        layout.addLayout(btn_layout)
        
        self.setCursor(Qt.PointingHandCursor)
    
    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)


class ActionListWidget(QWidget):
    """Widget for displaying and managing a list of actions."""
    
    action_selected = Signal(int)  # index
    actions_changed = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.action_cards: list[ActionCard] = []
        self._setup_ui()
    
    def _setup_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(8)
        
        # Scroll area for actions
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.container = QWidget()
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setAlignment(Qt.AlignTop)
        self.container_layout.setSpacing(8)
        
        scroll.setWidget(self.container)
        self.layout.addWidget(scroll)
    
    def set_actions(self, actions: list[dict]):
        """Set the list of actions to display."""
        self.clear()
        
        for i, action in enumerate(actions):
            card = ActionCard(
                action_name=action.get("name", "unknown"),
                display_name=action.get("display_name", action.get("name", "Unknown")),
                tool=action.get("tool", "internal"),
                params=action.get("params", {}),
            )
            card.clicked.connect(lambda idx=i: self.action_selected.emit(idx))
            card.delete_clicked.connect(lambda idx=i: self._delete_action(idx))
            card.move_up_clicked.connect(lambda idx=i: self._move_up(idx))
            card.move_down_clicked.connect(lambda idx=i: self._move_down(idx))
            
            self.action_cards.append(card)
            self.container_layout.addWidget(card)
    
    def clear(self):
        """Clear all actions."""
        for card in self.action_cards:
            card.deleteLater()
        self.action_cards.clear()
    
    def _delete_action(self, index: int):
        if 0 <= index < len(self.action_cards):
            card = self.action_cards.pop(index)
            card.deleteLater()
            self.actions_changed.emit()
    
    def _move_up(self, index: int):
        if index > 0:
            self._swap_actions(index, index - 1)
    
    def _move_down(self, index: int):
        if index < len(self.action_cards) - 1:
            self._swap_actions(index, index + 1)
    
    def _swap_actions(self, i: int, j: int):
        # This is a simplified version - full implementation would update the layout
        self.actions_changed.emit()


class ParameterEditor(QFrame):
    """Editor for action parameters."""
    
    value_changed = Signal(str, object)  # param_name, value
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("panel")
        self._setup_ui()
        self.param_widgets = {}
    
    def _setup_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(12, 12, 12, 12)
        
        self.title = QLabel("Parameters")
        self.title.setObjectName("header")
        self.layout.addWidget(self.title)
        
        self.form_container = QWidget()
        self.form_layout = QVBoxLayout(self.form_container)
        self.form_layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.form_container)
        
        self.layout.addStretch()
    
    def set_action(self, action_name: str, display_name: str, 
                   parameters: list[dict], current_values: dict):
        """Set the action to edit."""
        self.title.setText(f"Parameters: {display_name}")
        
        # Clear existing widgets
        for widget in self.param_widgets.values():
            widget.deleteLater()
        self.param_widgets.clear()
        
        # Create new widgets
        for param in parameters:
            name = param["name"]
            param_type = param.get("type", "string")
            default = param.get("default")
            description = param.get("description", "")
            current = current_values.get(name, default)
            
            # Label
            label = QLabel(f"{name}:")
            label.setToolTip(description)
            self.form_layout.addWidget(label)
            
            # Input widget based on type
            if param_type == "bool":
                widget = QCheckBox()
                widget.setChecked(bool(current))
                widget.stateChanged.connect(
                    lambda state, n=name: self.value_changed.emit(n, state == Qt.Checked)
                )
            elif param_type == "int":
                widget = QSpinBox()
                widget.setRange(
                    int(param.get("min_value", -999999)),
                    int(param.get("max_value", 999999))
                )
                if current is not None:
                    widget.setValue(int(current))
                widget.valueChanged.connect(
                    lambda val, n=name: self.value_changed.emit(n, val)
                )
            elif param_type == "float":
                widget = QDoubleSpinBox()
                widget.setRange(
                    param.get("min_value", -999999.0),
                    param.get("max_value", 999999.0)
                )
                widget.setDecimals(6)
                if current is not None:
                    widget.setValue(float(current))
                widget.valueChanged.connect(
                    lambda val, n=name: self.value_changed.emit(n, val)
                )
            elif param_type == "enum":
                widget = QComboBox()
                widget.addItems(param.get("enum_values", []))
                if current:
                    widget.setCurrentText(str(current))
                widget.currentTextChanged.connect(
                    lambda val, n=name: self.value_changed.emit(n, val)
                )
            else:  # string, path
                widget = QLineEdit()
                if current is not None:
                    widget.setText(str(current))
                widget.textChanged.connect(
                    lambda val, n=name: self.value_changed.emit(n, val)
                )
            
            self.form_layout.addWidget(widget)
            self.param_widgets[name] = widget
    
    def clear(self):
        """Clear the editor."""
        self.title.setText("Parameters")
        for widget in self.param_widgets.values():
            widget.deleteLater()
        self.param_widgets.clear()


class LogConsole(QFrame):
    """Log console widget."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("panel")
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        
        header = QHBoxLayout()
        title = QLabel("Log")
        title.setObjectName("header")
        header.addWidget(title)
        
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear)
        header.addWidget(clear_btn)
        
        layout.addLayout(header)
        
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.text)
    
    def log(self, message: str, level: str = "info"):
        """Add a log message."""
        color_map = {
            "info": "#dff6fb",
            "success": "#6bff6b",
            "warning": "#ffd93d",
            "error": "#ff6b6b",
        }
        color = color_map.get(level, "#dff6fb")
        self.text.append(f'<span style="color: {color}">{message}</span>')
    
    def clear(self):
        self.text.clear()


class ProgressPanel(QFrame):
    """Panel showing execution progress."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("panel")
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("header")
        layout.addWidget(self.status_label)
        
        self.progress = QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        
        self.detail_label = QLabel("")
        self.detail_label.setObjectName("secondary")
        layout.addWidget(self.detail_label)
    
    def set_progress(self, value: int, status: str = "", detail: str = ""):
        self.progress.setValue(value)
        if status:
            self.status_label.setText(status)
        if detail:
            self.detail_label.setText(detail)
    
    def reset(self):
        self.progress.setValue(0)
        self.status_label.setText("Ready")
        self.detail_label.setText("")


class ProfileCard(QFrame):
    """Card showing detected profile information."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("panel")
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        
        self.name_label = QLabel("No profile detected")
        self.name_label.setObjectName("header")
        layout.addWidget(self.name_label)
        
        self.confidence_label = QLabel("")
        layout.addWidget(self.confidence_label)
        
        self.description_label = QLabel("")
        self.description_label.setWordWrap(True)
        self.description_label.setObjectName("secondary")
        layout.addWidget(self.description_label)
        
        self.reasons_label = QLabel("")
        self.reasons_label.setWordWrap(True)
        layout.addWidget(self.reasons_label)
    
    def set_profile(self, profile_match):
        """Set the profile to display."""
        if profile_match:
            self.name_label.setText(profile_match.profile.display_name)
            self.confidence_label.setText(f"Confidence: {profile_match.confidence:.0%}")
            self.description_label.setText(profile_match.profile.description)
            self.reasons_label.setText("Reasons: " + "; ".join(profile_match.reasons))
        else:
            self.name_label.setText("No profile detected")
            self.confidence_label.setText("")
            self.description_label.setText("")
            self.reasons_label.setText("")
