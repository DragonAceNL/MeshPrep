# Copyright 2025 Dragon Ace
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""Theme styles for MeshPrep GUI."""

DARK_THEME = {
    "background": "#0f1720",
    "panel": "#111822",
    "accent": "#4fe8c4",
    "text": "#dff6fb",
    "text_secondary": "#8899a6",
    "button": "#1b2b33",
    "button_hover": "#243d47",
    "button_pressed": "#0d1a1f",
    "error": "#ff6b6b",
    "warning": "#ffd93d",
    "success": "#6bff6b",
    "border": "#2a3f4a",
    "selection": "#2a5a4a",
}

LIGHT_THEME = {
    "background": "#f5f5f5",
    "panel": "#ffffff",
    "accent": "#00a67d",
    "text": "#1a1a1a",
    "text_secondary": "#666666",
    "button": "#e0e0e0",
    "button_hover": "#d0d0d0",
    "button_pressed": "#c0c0c0",
    "error": "#d32f2f",
    "warning": "#f57c00",
    "success": "#388e3c",
    "border": "#cccccc",
    "selection": "#b8e6d4",
}


def get_stylesheet(theme: dict) -> str:
    """Generate Qt stylesheet from theme dictionary."""
    return f"""
    QMainWindow, QDialog {{
        background-color: {theme['background']};
        color: {theme['text']};
    }}
    
    QWidget {{
        background-color: {theme['background']};
        color: {theme['text']};
        font-family: "Segoe UI", Arial, sans-serif;
        font-size: 10pt;
    }}
    
    QFrame#panel {{
        background-color: {theme['panel']};
        border: 1px solid {theme['border']};
        border-radius: 8px;
    }}
    
    QLabel {{
        color: {theme['text']};
        background-color: transparent;
    }}
    
    QLabel#header {{
        font-size: 14pt;
        font-weight: bold;
        color: {theme['accent']};
    }}
    
    QLabel#secondary {{
        color: {theme['text_secondary']};
    }}
    
    QPushButton {{
        background-color: {theme['button']};
        color: {theme['text']};
        border: 1px solid {theme['border']};
        border-radius: 6px;
        padding: 8px 16px;
        min-width: 80px;
    }}
    
    QPushButton:hover {{
        background-color: {theme['button_hover']};
        border-color: {theme['accent']};
    }}
    
    QPushButton:pressed {{
        background-color: {theme['button_pressed']};
    }}
    
    QPushButton:disabled {{
        background-color: {theme['panel']};
        color: {theme['text_secondary']};
    }}
    
    QPushButton#primary {{
        background-color: {theme['accent']};
        color: {theme['background']};
        font-weight: bold;
    }}
    
    QPushButton#primary:hover {{
        background-color: {theme['accent']};
        opacity: 0.9;
    }}
    
    QPushButton#danger {{
        background-color: {theme['error']};
        color: white;
    }}
    
    QLineEdit, QTextEdit, QPlainTextEdit {{
        background-color: {theme['panel']};
        color: {theme['text']};
        border: 1px solid {theme['border']};
        border-radius: 4px;
        padding: 6px;
    }}
    
    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
        border-color: {theme['accent']};
    }}
    
    QComboBox {{
        background-color: {theme['button']};
        color: {theme['text']};
        border: 1px solid {theme['border']};
        border-radius: 4px;
        padding: 6px;
        min-width: 100px;
    }}
    
    QComboBox:hover {{
        border-color: {theme['accent']};
    }}
    
    QComboBox::drop-down {{
        border: none;
        width: 20px;
    }}
    
    QComboBox QAbstractItemView {{
        background-color: {theme['panel']};
        color: {theme['text']};
        border: 1px solid {theme['border']};
        selection-background-color: {theme['selection']};
    }}
    
    QListWidget {{
        background-color: {theme['panel']};
        color: {theme['text']};
        border: 1px solid {theme['border']};
        border-radius: 4px;
    }}
    
    QListWidget::item {{
        padding: 8px;
        border-bottom: 1px solid {theme['border']};
    }}
    
    QListWidget::item:selected {{
        background-color: {theme['selection']};
    }}
    
    QListWidget::item:hover {{
        background-color: {theme['button_hover']};
    }}
    
    QTableWidget {{
        background-color: {theme['panel']};
        color: {theme['text']};
        border: 1px solid {theme['border']};
        gridline-color: {theme['border']};
    }}
    
    QTableWidget::item {{
        padding: 4px;
    }}
    
    QTableWidget::item:selected {{
        background-color: {theme['selection']};
    }}
    
    QHeaderView::section {{
        background-color: {theme['button']};
        color: {theme['text']};
        padding: 8px;
        border: none;
        border-bottom: 1px solid {theme['border']};
    }}
    
    QProgressBar {{
        background-color: {theme['panel']};
        color: {theme['text']};
        border: 1px solid {theme['border']};
        border-radius: 4px;
        text-align: center;
    }}
    
    QProgressBar::chunk {{
        background-color: {theme['accent']};
        border-radius: 3px;
    }}
    
    QScrollBar:vertical {{
        background-color: {theme['panel']};
        width: 12px;
        border-radius: 6px;
    }}
    
    QScrollBar::handle:vertical {{
        background-color: {theme['button']};
        border-radius: 6px;
        min-height: 20px;
    }}
    
    QScrollBar::handle:vertical:hover {{
        background-color: {theme['button_hover']};
    }}
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    
    QScrollBar:horizontal {{
        background-color: {theme['panel']};
        height: 12px;
        border-radius: 6px;
    }}
    
    QScrollBar::handle:horizontal {{
        background-color: {theme['button']};
        border-radius: 6px;
        min-width: 20px;
    }}
    
    QTabWidget::pane {{
        background-color: {theme['panel']};
        border: 1px solid {theme['border']};
        border-radius: 4px;
    }}
    
    QTabBar::tab {{
        background-color: {theme['button']};
        color: {theme['text']};
        padding: 8px 16px;
        border: 1px solid {theme['border']};
        border-bottom: none;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
    }}
    
    QTabBar::tab:selected {{
        background-color: {theme['panel']};
        color: {theme['accent']};
    }}
    
    QTabBar::tab:hover {{
        background-color: {theme['button_hover']};
    }}
    
    QGroupBox {{
        background-color: {theme['panel']};
        border: 1px solid {theme['border']};
        border-radius: 8px;
        margin-top: 12px;
        padding-top: 12px;
    }}
    
    QGroupBox::title {{
        color: {theme['accent']};
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 4px;
    }}
    
    QCheckBox {{
        color: {theme['text']};
        spacing: 8px;
    }}
    
    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border: 2px solid {theme['border']};
        border-radius: 4px;
        background-color: {theme['panel']};
    }}
    
    QCheckBox::indicator:checked {{
        background-color: {theme['accent']};
        border-color: {theme['accent']};
    }}
    
    QRadioButton {{
        color: {theme['text']};
        spacing: 8px;
    }}
    
    QRadioButton::indicator {{
        width: 18px;
        height: 18px;
        border: 2px solid {theme['border']};
        border-radius: 9px;
        background-color: {theme['panel']};
    }}
    
    QRadioButton::indicator:checked {{
        background-color: {theme['accent']};
        border-color: {theme['accent']};
    }}
    
    QSpinBox, QDoubleSpinBox {{
        background-color: {theme['panel']};
        color: {theme['text']};
        border: 1px solid {theme['border']};
        border-radius: 4px;
        padding: 4px;
    }}
    
    QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: {theme['accent']};
    }}
    
    QSplitter::handle {{
        background-color: {theme['border']};
    }}
    
    QSplitter::handle:horizontal {{
        width: 2px;
    }}
    
    QSplitter::handle:vertical {{
        height: 2px;
    }}
    
    QToolTip {{
        background-color: {theme['panel']};
        color: {theme['text']};
        border: 1px solid {theme['border']};
        padding: 4px;
    }}
    
    QStatusBar {{
        background-color: {theme['panel']};
        color: {theme['text_secondary']};
    }}
    
    QMenuBar {{
        background-color: {theme['panel']};
        color: {theme['text']};
    }}
    
    QMenuBar::item:selected {{
        background-color: {theme['selection']};
    }}
    
    QMenu {{
        background-color: {theme['panel']};
        color: {theme['text']};
        border: 1px solid {theme['border']};
    }}
    
    QMenu::item:selected {{
        background-color: {theme['selection']};
    }}
    """


def apply_theme(widget, theme_name: str = "dark"):
    """Apply a theme to a widget."""
    theme = DARK_THEME if theme_name == "dark" else LIGHT_THEME
    widget.setStyleSheet(get_stylesheet(theme))
