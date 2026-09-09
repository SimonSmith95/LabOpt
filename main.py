"""
Phase 12 — Entry Point
Launch the LabOpt PySide6 application.

Usage
-----
    python main.py
"""
import os
import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication
from main_window import MainWindow


def _resource_path(relative: str) -> str:
    """Return the absolute path to a bundled resource.

    Works both when running from source and from a PyInstaller one-dir bundle
    (where data files live under sys._MEIPASS at runtime).
    """
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative)


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("LabOpt")
    app.setOrganizationName("Lab")

    logo_path = _resource_path("LabOpt_logo.png")
    if os.path.exists(logo_path):
        app.setWindowIcon(QIcon(logo_path))

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
