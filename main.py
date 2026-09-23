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

# ── Logging must be configured before any other LabOpt import ────────────────
from app_logger import configure_logger, get_logger, log_path
configure_logger()
_log = get_logger(__name__)


def _resource_path(relative: str) -> str:
    """Return the absolute path to a bundled resource.

    Works both when running from source and from a PyInstaller one-dir bundle
    (where data files live under sys._MEIPASS at runtime).
    """
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative)


def _install_excepthook() -> None:
    """
    Replace sys.excepthook so unhandled exceptions are written to the log
    file before the default traceback is printed.

    This is the last line of defence for diagnosing crashes — the log entry
    will be visible in labopt.log even if the terminal window was closed.
    """
    _crash_log = get_logger("crash")

    def _hook(exc_type, exc_val, exc_tb):
        _crash_log.critical(
            "Unhandled exception — application may crash",
            exc_info=(exc_type, exc_val, exc_tb),
        )
        # Let the default handler print to stderr as well
        sys.__excepthook__(exc_type, exc_val, exc_tb)

    sys.excepthook = _hook


def main() -> None:
    _install_excepthook()

    _log.info("=" * 60)
    _log.info("LabOpt starting  (Python %s)", sys.version.split()[0])
    _log.info("Log file: %s", log_path())
    _log.info("=" * 60)

    from main_window import MainWindow   # deferred so logger is ready first

    app = QApplication(sys.argv)
    app.setApplicationName("LabOpt")
    app.setOrganizationName("Lab")

    logo_path = _resource_path("LabOpt_logo.png")
    if os.path.exists(logo_path):
        app.setWindowIcon(QIcon(logo_path))

    window = MainWindow()
    window.show()

    _log.info("Main window shown — entering event loop")
    exit_code = app.exec()
    _log.info("Event loop exited (code %d) — shutting down", exit_code)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
