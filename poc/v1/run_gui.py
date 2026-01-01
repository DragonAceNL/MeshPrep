# Copyright 2025 Dragon Ace
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""GUI entry point for MeshPrep POC."""

import sys


def main():
    """Launch the MeshPrep GUI."""
    from PySide6.QtWidgets import QApplication
    from meshprep.gui import MainWindow
    
    app = QApplication(sys.argv)
    app.setApplicationName("MeshPrep")
    app.setOrganizationName("Dragon Ace")
    app.setOrganizationDomain("github.com/DragonAceNL")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
