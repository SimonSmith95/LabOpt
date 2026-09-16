"""
Phases 9 & 10 — Main Window + Warning System
Central PySide6 QMainWindow that ties all components together.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

import optuna
import pandas as pd
import sys

from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QAction, QFont, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from batch_results_dialog import BatchResultsDialog
from design_space_widget import ConvergenceWidget, DesignSpaceDialog, ParetoWidget
from power_analysis_widget import PowerAnalysisCoordinator
from csv_loader import aggregate_replicates, extract_param_defaults, infer_context_defaults, load_csv, load_trials_from_csv
from report_generator import export_plots_as_png, generate_report, write_report
from surrogate_quality import MIN_TRIALS, compute_surrogate_quality, predict_batch
from optuna_builder import (
    build_study,
    get_pareto_front,
    load_historical_trials,
    tell_batch,
)
from param_card_widget import ParamCardWidget
from constraint_dialog import ConstraintDialog
from parameter_config import (
    ObjectiveConfig, ParameterConfig, ParameterConstraint, ParameterType, StudyConfig,
)
from session_manager import SessionManager, SessionState
from worker import OptimizationWorker

# ── Warning thresholds (Phase 10) ─────────────────────────────────────────────
TPE_PARAM_WARNING_THRESHOLD = 8
MIN_TRIALS_PER_PARAM_RATIO = 2

# ── Dark theme stylesheet ──────────────────────────────────────────────────────
DARK_STYLESHEET = """
/* ── Base ── */
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
}
/* ── Dock widgets ── */
QDockWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}
QDockWidget::title {
    background-color: #2a2a3e;
    padding: 4px 8px;
    font-weight: bold;
}
/* ── Group boxes ── */
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 5px;
    margin-top: 10px;
    color: #bac2de;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    color: #89b4fa;
}
/* ── Buttons ── */
QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 5px;
    padding: 5px 12px;
}
QPushButton:hover {
    background-color: #45475a;
    border-color: #89b4fa;
}
QPushButton:pressed {
    background-color: #1e1e2e;
}
QPushButton:disabled {
    color: #585b70;
    background-color: #1e1e2e;
    border-color: #313244;
}
QPushButton:checked {
    background-color: #89b4fa;
    color: #1e1e2e;
    border-color: #89b4fa;
}
/* ── Combo boxes ── */
QComboBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 3px 8px;
    selection-background-color: #89b4fa;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox QAbstractItemView {
    background-color: #2a2a3e;
    color: #cdd6f4;
    selection-background-color: #89b4fa;
    selection-color: #1e1e2e;
    border: 1px solid #45475a;
}
/* ── Spin boxes ── */
QSpinBox, QDoubleSpinBox, QLineEdit {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 3px 6px;
    selection-background-color: #89b4fa;
}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    background-color: #585b70;
    border: 1px solid #6c6f85;
    border-radius: 2px;
    width: 16px;
}
QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #89b4fa;
    border-color: #89b4fa;
}
QSpinBox::up-button:pressed, QSpinBox::down-button:pressed,
QDoubleSpinBox::up-button:pressed, QDoubleSpinBox::down-button:pressed {
    background-color: #7aa2f7;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    width: 0;
    height: 0;
    border-left:  5px solid transparent;
    border-right: 5px solid transparent;
    border-bottom: 6px solid #cdd6f4;
}
QSpinBox::up-arrow:hover, QDoubleSpinBox::up-arrow:hover {
    border-bottom-color: #1e1e2e;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    width: 0;
    height: 0;
    border-left:  5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #cdd6f4;
}
QSpinBox::down-arrow:hover, QDoubleSpinBox::down-arrow:hover {
    border-top-color: #1e1e2e;
}
/* ── Labels ── */
QLabel {
    color: #cdd6f4;
}
/* ── Checkboxes & radio buttons ── */
QCheckBox, QRadioButton {
    color: #cdd6f4;
    spacing: 6px;
}
QCheckBox::indicator, QRadioButton::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #45475a;
    border-radius: 3px;
    background-color: #313244;
}
QCheckBox::indicator:checked {
    background-color: #89b4fa;
    border-color: #89b4fa;
}
QRadioButton::indicator {
    border-radius: 7px;
}
QRadioButton::indicator:checked {
    background-color: #89b4fa;
    border-color: #89b4fa;
}
/* ── Tables ── */
QTableWidget, QTableView {
    background-color: #1e1e2e;
    color: #cdd6f4;
    gridline-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    alternate-background-color: #24243e;
}
QTableWidget::item, QTableView::item {
    padding: 4px;
}
QTableWidget::item:selected, QTableView::item:selected {
    background-color: #89b4fa;
    color: #1e1e2e;
}
QHeaderView::section {
    background-color: #2a2a3e;
    color: #89b4fa;
    border: 1px solid #45475a;
    padding: 5px;
    font-weight: bold;
}
/* ── Scroll bars ── */
QScrollBar:vertical {
    background-color: #1e1e2e;
    width: 10px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background-color: #45475a;
    border-radius: 5px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background-color: #585b70;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal {
    background-color: #1e1e2e;
    height: 10px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal {
    background-color: #45475a;
    border-radius: 5px;
    min-width: 20px;
}
QScrollBar::handle:horizontal:hover {
    background-color: #585b70;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
/* ── Tabs ── */
QTabWidget::pane {
    border: 1px solid #45475a;
    border-radius: 4px;
    background-color: #1e1e2e;
}
QTabBar::tab {
    background-color: #2a2a3e;
    color: #bac2de;
    padding: 6px 14px;
    border: 1px solid #45475a;
    border-bottom: none;
    border-radius: 4px 4px 0 0;
}
QTabBar::tab:selected {
    background-color: #1e1e2e;
    color: #cdd6f4;
    border-bottom: 2px solid #89b4fa;
}
QTabBar::tab:hover:!selected {
    background-color: #313244;
}
/* ── Menu bar ── */
QMenuBar {
    background-color: #1e1e2e;
    color: #cdd6f4;
    border-bottom: 1px solid #313244;
}
QMenuBar::item:selected {
    background-color: #313244;
    border-radius: 3px;
}
QMenu {
    background-color: #2a2a3e;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 4px;
}
QMenu::item:selected {
    background-color: #89b4fa;
    color: #1e1e2e;
    border-radius: 3px;
}
QMenu::separator {
    height: 1px;
    background-color: #45475a;
    margin: 4px 8px;
}
/* ── Toolbar ── */
QToolBar {
    background-color: #2a2a3e;
    border-bottom: 1px solid #313244;
    spacing: 4px;
    padding: 3px;
}
QToolBar QToolButton {
    background-color: transparent;
    color: #cdd6f4;
    border-radius: 4px;
    padding: 4px 8px;
}
QToolBar QToolButton:hover {
    background-color: #313244;
}
/* ── Status bar ── */
QStatusBar {
    background-color: #181825;
    color: #bac2de;
    border-top: 1px solid #313244;
}
/* ── Scroll area ── */
QScrollArea {
    border: none;
    background-color: #1e1e2e;
}
/* ── Text browser (docs viewer) ── */
QTextBrowser {
    background-color: #1e1e2e;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    font-size: 13px;
    selection-background-color: #89b4fa;
    selection-color: #1e1e2e;
}
/* ── Dialog ── */
QDialog {
    background-color: #1e1e2e;
    color: #cdd6f4;
}
/* ── Frame separators ── */
QFrame[frameShape="4"], QFrame[frameShape="5"] {
    color: #45475a;
}
"""

# Warning / banner colours per theme
_WARN_STYLE_DARK  = (
    "background:#3d2e00; color:#ffc107; padding:6px; border-radius:4px; border:1px solid #ffc107;"
)
_WARN_STYLE_LIGHT = (
    "background:#fff3cd; color:#856404; padding:6px; border-radius:4px; border:1px solid #ffc107;"
)
_BANNER_STYLE_DARK  = (
    "background:#3d2e00; border:1px solid #ffc107; padding:8px; border-radius:4px;"
)
_BANNER_STYLE_LIGHT = (
    "background:#fff3cd; border:1px solid #ffc107; padding:8px; border-radius:4px;"
)


# ── Simple Markdown → HTML converter (for the docs viewer) ─────────────────────
def _markdown_to_html(text: str) -> str:
    """
    Convert a subset of Markdown to HTML suitable for QTextBrowser.
    Handles: headings (##-####), bold (**), inline code (`),
    fenced code blocks (```), bullet lists (- ), blockquotes (>), tables (|).
    """
    import re
    lines = text.split("\n")
    html_lines: list[str] = []
    in_code_block = False
    in_list = False
    in_table = False

    def close_list():
        nonlocal in_list
        if in_list:
            html_lines.append("</ul>")
            in_list = False

    def close_table():
        nonlocal in_table
        if in_table:
            html_lines.append("</table>")
            in_table = False

    def inline(s: str) -> str:
        """Apply inline formatting: bold, inline code."""
        # Inline code first (to avoid double-processing)
        s = re.sub(r"`([^`]+)`", r"<code style='background:#2a2a3e;padding:1px 4px;"
                   r"border-radius:3px;font-family:monospace'>\1</code>", s)
        # Bold
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        return s

    for line in lines:
        # Fenced code blocks
        if line.startswith("```"):
            if in_code_block:
                html_lines.append("</pre>")
                in_code_block = False
            else:
                close_list()
                close_table()
                html_lines.append(
                    "<pre style='background:#181825;color:#a6e3a1;"
                    "padding:10px;border-radius:6px;"
                    "border:1px solid #45475a;overflow-x:auto'>"
                )
                in_code_block = True
            continue

        if in_code_block:
            html_lines.append(line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
            continue

        # Headings
        if line.startswith("#### "):
            close_list(); close_table()
            html_lines.append(f"<h4 style='color:#cba6f7'>{inline(line[5:])}</h4>")
        elif line.startswith("### "):
            close_list(); close_table()
            html_lines.append(f"<h3 style='color:#89b4fa'>{inline(line[4:])}</h3>")
        elif line.startswith("## "):
            close_list(); close_table()
            html_lines.append(f"<h2 style='color:#89dceb;border-bottom:1px solid #45475a;"
                              f"padding-bottom:4px'>{inline(line[3:])}</h2>")
        elif line.startswith("# "):
            close_list(); close_table()
            html_lines.append(f"<h1 style='color:#89dceb'>{inline(line[2:])}</h1>")

        # Blockquotes
        elif line.startswith("> "):
            close_list(); close_table()
            html_lines.append(
                f"<blockquote style='border-left:3px solid #89b4fa;"
                f"margin:4px 0;padding:4px 12px;color:#bac2de'>"
                f"{inline(line[2:])}</blockquote>"
            )

        # Table rows
        elif line.startswith("|") and line.endswith("|"):
            stripped = line.strip("|")
            if re.match(r"^[\s\-|:]+$", stripped):
                # Separator row — skip
                continue
            cells = [c.strip() for c in stripped.split("|")]
            if not in_table:
                html_lines.append(
                    "<table style='border-collapse:collapse;width:100%;"
                    "margin:8px 0'>"
                )
                in_table = True
                tag = "th"
                style = ("background:#2a2a3e;color:#89b4fa;font-weight:bold;"
                         "padding:6px 10px;border:1px solid #45475a")
            else:
                tag = "td"
                style = "padding:5px 10px;border:1px solid #313244"
            row = "".join(f"<{tag} style='{style}'>{inline(c)}</{tag}>" for c in cells)
            html_lines.append(f"<tr>{row}</tr>")

        # Bullet list items
        elif re.match(r"^[-*] ", line):
            close_table()
            if not in_list:
                html_lines.append("<ul style='margin:4px 0 4px 20px;padding:0'>")
                in_list = True
            html_lines.append(f"<li style='margin:2px 0'>{inline(line[2:])}</li>")

        # Horizontal rules
        elif re.match(r"^---+$", line.strip()):
            close_list(); close_table()
            html_lines.append("<hr style='border:none;border-top:1px solid #45475a;margin:8px 0'>")

        # Blank lines
        elif line.strip() == "":
            close_list(); close_table()
            html_lines.append("<p></p>")

        # Normal paragraph text
        else:
            close_table()
            html_lines.append(f"<p style='margin:2px 0'>{inline(line)}</p>")

    close_list()
    close_table()

    body = "\n".join(html_lines)
    return (
        "<html><head><style>"
        "body{font-family:sans-serif;font-size:13px;margin:12px;color:#cdd6f4;"
        "background:#1e1e2e}"
        "code{font-family:monospace}"
        "</style></head>"
        f"<body>{body}</body></html>"
    )


class MainWindow(QMainWindow):
    """
    Main application window for LabOpt.

    Layout
    ------
    Menu bar   — File (session/config management) + Help
    Toolbar    — Load CSV / Reload CSV / path label
    Left dock  — Study settings (objectives, sampler, batch, warnings, Ask button)
    Centre     — Resume banner (if pending batch) + scrollable param cards
    Bottom dock— Results tabs (All Trials / Best-Pareto) + export button
    Status bar — trial count, best value, status text
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LabOpt — Lab Optimiser")
        # Cap initial size to the available screen geometry so the window never
        # opens larger than the display (common on 768-height laptops).
        _screen = QApplication.primaryScreen()
        if _screen is not None:
            _avail = _screen.availableGeometry()
            self.resize(
                min(1280, _avail.width()  - 40),
                min(820,  _avail.height() - 60),
            )
        else:
            self.resize(1280, 820)

        # ── Application state ──────────────────────────────────────────────
        self._study: Optional[optuna.Study] = None
        self._session_state: Optional[SessionState] = None
        self._df: Optional[pd.DataFrame] = None
        self._csv_path: Optional[str] = None
        self._param_cards: List[ParamCardWidget] = []
        self._worker: Optional[OptimizationWorker] = None
        self._batches_done: int = 0
        self._design_space_dlg: Optional[DesignSpaceDialog] = None

        # Surrogate validation state (Results-tab checkbox trigger)
        self._validation_results = None  # ValidationResults | None
        self._validation_worker  = None  # ValidationWorker  | None

        # Objective-row widgets [(col_name_label, direction_combo), ...]
        self._obj_row_widgets: List[tuple] = []
        # Per-column checkboxes in the objectives selector
        self._obj_col_rows: List[tuple] = []  # (col, QCheckBox, QComboBox)
        # User-defined parameter constraints (equality / inequality rules)
        self._constraints: List[ParameterConstraint] = []
        # Replicate aggregation UI widgets (rebuilt when a CSV is loaded)
        self._repl_cb: Optional[QCheckBox] = None
        self._repl_tol_le: Optional[QLineEdit] = None
        # Context variable UI: (col_name, QCheckBox) per column
        self._ctx_col_checkboxes: List[tuple] = []
        # Context condition spinboxes shown in Settings tab: {col_name: QDoubleSpinBox}
        self._context_spinboxes: dict = {}
        # Planned context group box (created once, shown/hidden dynamically)
        self._ctx_conditions_group: Optional[QGroupBox] = None
        # Planned context values passed to worker (set just before asking)
        self._planned_context: Optional[dict] = None
        # Reference to results dialog so we can collect context corrections after submit
        self._active_batch_dlg: Optional[BatchResultsDialog] = None

        # Centre stacked widget — created early so _build_left_dock (which adds
        # the power result panel as page 1) and _build_centre (which adds the
        # param-card scroll area as page 0) can both reference it.
        self._centre_stack = QStackedWidget()

        self._build_menu()
        self._build_toolbar()
        self._build_left_dock()
        self._build_centre()
        self._build_status_bar()

        # Ensure the param-card scroll area is visible at startup.
        # (_build_left_dock adds the power result panel first so it becomes
        # the QStackedWidget's default page; we correct that here.)
        self._centre_stack.setCurrentWidget(self._scroll)

        # ── Theme (dark by default, persisted via QSettings) ──────────────
        settings = QSettings()
        dark = settings.value("dark_mode", True, type=bool)
        self._dark_mode_action.setChecked(dark)
        self._apply_theme(dark)

    # ══════════════════════════════════════════════════════════════════════
    # Menu bar
    # ══════════════════════════════════════════════════════════════════════

    def _build_menu(self) -> None:
        mb = self.menuBar()

        # ── File ──────────────────────────────────────────────────────────
        file_menu = mb.addMenu("File")

        new_act = QAction("New Session…", self)
        new_act.setShortcut("Ctrl+N")
        new_act.triggered.connect(self._action_new_session)
        file_menu.addAction(new_act)

        load_sess_act = QAction("Load Session…", self)
        load_sess_act.setShortcut("Ctrl+O")
        load_sess_act.triggered.connect(self._action_load_session)
        file_menu.addAction(load_sess_act)

        self._recent_menu = file_menu.addMenu("Recent Sessions")
        self._refresh_recent_menu()

        file_menu.addSeparator()

        save_cfg_act = QAction("Save Config…", self)
        save_cfg_act.triggered.connect(self._action_save_config)
        file_menu.addAction(save_cfg_act)

        load_cfg_act = QAction("Load Config…", self)
        load_cfg_act.triggered.connect(self._action_load_config)
        file_menu.addAction(load_cfg_act)

        file_menu.addSeparator()
        exit_act = QAction("Exit", self)
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        # ── View ──────────────────────────────────────────────────────────
        view_menu = mb.addMenu("View")
        self._dark_mode_action = QAction("Dark Mode", self)
        self._dark_mode_action.setCheckable(True)
        self._dark_mode_action.setChecked(True)   # default; overridden in __init__
        self._dark_mode_action.setShortcut("Ctrl+Shift+D")
        self._dark_mode_action.setToolTip(
            "Toggle between dark (Catppuccin Mocha) and light (system default) theme."
        )
        self._dark_mode_action.triggered.connect(self._action_toggle_theme)
        view_menu.addAction(self._dark_mode_action)

        # ── Help ──────────────────────────────────────────────────────────
        help_menu = mb.addMenu("Help")

        how_to_act = QAction("How to Use…", self)
        how_to_act.setShortcut("F1")
        how_to_act.setToolTip("Open the How to Use section of the documentation.")
        how_to_act.triggered.connect(self._action_show_how_to_use)
        help_menu.addAction(how_to_act)

        help_menu.addSeparator()

        about_act = QAction("About", self)
        about_act.triggered.connect(self._action_about)
        help_menu.addAction(about_act)

    def _refresh_recent_menu(self) -> None:
        self._recent_menu.clear()
        recent = SessionManager.list_recent_sessions(5)
        for entry in recent:
            act = QAction(entry.get("study_name", "Unknown"), self)
            path = entry.get("session_path", "")
            act.triggered.connect(
                lambda _checked, p=path: self._load_session_from_path(p)
            )
            self._recent_menu.addAction(act)
        if not recent:
            dummy = QAction("(no recent sessions)", self)
            dummy.setEnabled(False)
            self._recent_menu.addAction(dummy)

    # ══════════════════════════════════════════════════════════════════════
    # Toolbar
    # ══════════════════════════════════════════════════════════════════════

    def _build_toolbar(self) -> None:
        # ── Logo toolbar — sits above the CSV toolbar ──────────────────────
        _logo_path = os.path.join(
            getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__))),
            "LabOpt_logo.png",
        )
        if os.path.exists(_logo_path):
            logo_tb = QToolBar("Logo", self)
            logo_tb.setMovable(False)
            logo_tb.setFloatable(False)
            # Stretch a spacer on each side so the logo is centered
            #_left_spacer = QWidget()
            #_left_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            #logo_tb.addWidget(_left_spacer)

            _logo_lbl = QLabel()
            _logo_lbl.setContentsMargins(12, 4, 0, 4)
            _pm = QPixmap(_logo_path)
            if not _pm.isNull():
                _pm = _pm.scaledToHeight(72, Qt.SmoothTransformation)
                _logo_lbl.setPixmap(_pm)
            _logo_lbl.setContentsMargins(0, 4, 0, 4)
            logo_tb.addWidget(_logo_lbl)
            _right_spacer = QWidget()
            _right_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            logo_tb.addWidget(_right_spacer)
            self.addToolBar(Qt.TopToolBarArea, logo_tb)
            self.addToolBarBreak(Qt.TopToolBarArea)

        # ── CSV / action toolbar ───────────────────────────────────────────
        tb = QToolBar("CSV", self)
        tb.setMovable(False)
        self.addToolBar(tb)

        load_csv_act = QAction("Load CSV…", self)
        load_csv_act.setShortcut("Ctrl+L")
        load_csv_act.triggered.connect(self._action_load_csv)
        tb.addAction(load_csv_act)

        reload_csv_act = QAction("Reload CSV", self)
        reload_csv_act.setShortcut("Ctrl+R")
        reload_csv_act.triggered.connect(self._action_reload_csv)
        tb.addAction(reload_csv_act)

        self._csv_path_label = QLabel("  No CSV loaded")
        self._csv_path_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(self._csv_path_label)

        tb.addSeparator()

        self._pending_results_btn = QPushButton("📋  Enter Pending Results…")
        self._pending_results_btn.setToolTip(
            "Reopen the results dialog for the current pending batch.\n"
            "Use this when you come back after running experiments —\n"
            "you can correct the actual compositions and enter your measurements."
        )
        self._pending_results_btn.setEnabled(False)
        self._pending_results_btn.clicked.connect(self._action_enter_pending_results)
        tb.addWidget(self._pending_results_btn)

    # ══════════════════════════════════════════════════════════════════════
    # Left dock — Study settings
    # ══════════════════════════════════════════════════════════════════════

    def _build_left_dock(self) -> None:
        self._left_dock = QDockWidget("LabOpt Controls", self)
        self._left_dock.setAllowedAreas(Qt.LeftDockWidgetArea)
        self._left_dock.setFeatures(QDockWidget.NoDockWidgetFeatures)

        # Outer container holds a QTabWidget: ⚙ Settings | 📊 Results
        outer = QWidget()
        outer.setMinimumWidth(310)
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        self._left_tabs = QTabWidget()
        outer_layout.addWidget(self._left_tabs)

        # ══════════════════════════════════════════════════════════════
        # Tab 1 — ⚙ Settings
        # Outer widget: scroll area (top, flex) + buttons (bottom, fixed)
        # This prevents the Ask button from being pushed off-screen when
        # a CSV with many columns causes the objectives group to expand.
        # ══════════════════════════════════════════════════════════════
        settings_widget = QWidget()
        settings_outer = QVBoxLayout(settings_widget)
        settings_outer.setContentsMargins(0, 0, 0, 0)
        settings_outer.setSpacing(0)

        # ── Scrollable content area ────────────────────────────────────
        _settings_scroll = QScrollArea()
        _settings_scroll.setWidgetResizable(True)
        _settings_scroll.setStyleSheet("QScrollArea { border: none; }")
        _scroll_content = QWidget()
        vbox = QVBoxLayout(_scroll_content)
        vbox.setSpacing(8)
        vbox.setContentsMargins(6, 6, 6, 6)

        # ── Objectives ────────────────────────────────────────────────
        self._obj_group = QGroupBox("Objectives")
        self._obj_inner_layout = QVBoxLayout(self._obj_group)
        self._obj_inner_layout.addWidget(
            QLabel("Load a CSV to configure objectives.")
        )
        vbox.addWidget(self._obj_group)

        # ── Sampler ───────────────────────────────────────────────────
        sampler_group = QGroupBox("Sampler")
        sf = QFormLayout(sampler_group)
        self._sampler_combo = QComboBox()
        self._sampler_combo.addItems(["TPE", "NSGAII", "Random", "GP"])
        self._sampler_combo.setToolTip(
            "TPE: good for single-objective, ≤8 parameters.\n"
            "NSGAII: recommended for multi-objective.\n"
            "Random: baseline / exploration only.\n"
            "GP (Gaussian Process): uncertainty-aware; best for smooth continuous\n"
            "  spaces with < 200 trials. Tells you where the model does not know.\n"
            "  May be slow for large datasets — switch to TPE above ~200 trials."
        )
        self._sampler_combo.currentIndexChanged.connect(self._update_warnings)
        sf.addRow("Sampler:", self._sampler_combo)
        vbox.addWidget(sampler_group)

        # ── Batch settings ────────────────────────────────────────────
        batch_group = QGroupBox("Batch Settings")
        bf = QFormLayout(batch_group)

        self._batch_size_spin = QSpinBox()
        self._batch_size_spin.setRange(1, 500)
        self._batch_size_spin.setValue(1)
        self._batch_size_spin.setToolTip(
            "How many experiments to suggest per batch (run in parallel in the lab)."
        )
        self._batch_size_spin.valueChanged.connect(self._update_warnings)
        bf.addRow("Batch size:", self._batch_size_spin)

        self._n_batches_spin = QSpinBox()
        self._n_batches_spin.setRange(1, 100_000)
        self._n_batches_spin.setValue(10)
        bf.addRow("Total batches:", self._n_batches_spin)

        self._batches_done_label = QLabel("0 / 10")
        bf.addRow("Batches done:", self._batches_done_label)

        # ── Auto-stop (Feature 12) ─────────────────────────────────────
        self._auto_stop_cb = QCheckBox("Auto-stop when converged")
        self._auto_stop_cb.setToolTip(
            "When enabled, LabOpt stops asking new batches as soon as the best\n"
            "objective value has not improved by more than the threshold over\n"
            "the last N consecutive batches.\n\n"
            "Recommended: set n_batches to a large number (e.g. 50) and rely\n"
            "on auto-stop to halt when the study has converged."
        )
        bf.addRow("", self._auto_stop_cb)

        # Threshold row (greyed-out when checkbox is off)
        _as_row = QWidget()
        _as_hl  = QHBoxLayout(_as_row)
        _as_hl.setContentsMargins(0, 0, 0, 0)
        _as_hl.setSpacing(4)
        _as_hl.addWidget(QLabel("Min improvement:"))
        self._as_improve_spin = QDoubleSpinBox()
        self._as_improve_spin.setRange(0.01, 100.0)
        self._as_improve_spin.setDecimals(2)
        self._as_improve_spin.setSingleStep(0.5)
        self._as_improve_spin.setValue(1.0)
        self._as_improve_spin.setSuffix(" %")
        self._as_improve_spin.setToolTip(
            "Minimum relative improvement required per batch to continue.\n"
            "E.g. 1.0% means: stop if the best value changes by less than 1%\n"
            "across all batches in the window."
        )
        self._as_improve_spin.setEnabled(False)
        self._as_improve_spin.setFixedWidth(80)
        _as_hl.addWidget(self._as_improve_spin)
        _as_hl.addWidget(QLabel("  over"))
        self._as_nbatches_spin = QSpinBox()
        self._as_nbatches_spin.setRange(1, 50)
        self._as_nbatches_spin.setValue(3)
        self._as_nbatches_spin.setToolTip(
            "Number of consecutive batches with no improvement before stopping."
        )
        self._as_nbatches_spin.setEnabled(False)
        self._as_nbatches_spin.setFixedWidth(52)
        _as_hl.addWidget(self._as_nbatches_spin)
        _as_hl.addWidget(QLabel("batches"))
        _as_hl.addStretch()
        bf.addRow("", _as_row)

        # Enable/disable sub-controls when checkbox is toggled
        self._auto_stop_cb.toggled.connect(
            lambda checked: (
                self._as_improve_spin.setEnabled(checked),
                self._as_nbatches_spin.setEnabled(checked),
            )
        )

        vbox.addWidget(batch_group)

        # ── Warning label ──────────────────────────────────────────────
        self._warning_label = QLabel()
        self._warning_label.setWordWrap(True)
        self._warning_label.hide()
        vbox.addWidget(self._warning_label)

        vbox.addStretch()
        _settings_scroll.setWidget(_scroll_content)
        settings_outer.addWidget(_settings_scroll, stretch=1)

        # ── Action buttons — always visible below the scroll area ──────
        _btn_container = QWidget()
        _btn_vbox = QVBoxLayout(_btn_container)
        _btn_vbox.setContentsMargins(6, 4, 6, 6)
        _btn_vbox.setSpacing(4)

        self._ask_btn = QPushButton("Ask Next Batch")
        self._ask_btn.setEnabled(False)
        bold_big = QFont()
        bold_big.setBold(True)
        bold_big.setPointSize(11)
        self._ask_btn.setFont(bold_big)
        self._ask_btn.setMinimumHeight(42)
        self._ask_btn.setToolTip("Suggest the next batch of parameters from Optuna.")
        self._ask_btn.clicked.connect(self._action_ask_next_batch)
        _btn_vbox.addWidget(self._ask_btn)

        self._pause_btn = QPushButton("Pause")
        self._pause_btn.setEnabled(False)
        self._pause_btn.setCheckable(True)
        self._pause_btn.clicked.connect(self._action_toggle_pause)
        _btn_vbox.addWidget(self._pause_btn)

        settings_outer.addWidget(_btn_container)
        self._left_tabs.addTab(settings_widget, "⚙  Settings")

        # ══════════════════════════════════════════════════════════════
        # Tab 2 — 📊 Results
        # ══════════════════════════════════════════════════════════════
        results_widget = QWidget()
        rvbox = QVBoxLayout(results_widget)
        rvbox.setContentsMargins(4, 4, 4, 4)
        rvbox.setSpacing(4)

        rbtn_row = QHBoxLayout()
        design_space_btn = QPushButton("📊  Design Space…")
        design_space_btn.setToolTip(
            "Open the Design Space + Correlation Matrix window.\n"
            "Shows historical data in parameter space and how variables\n"
            "correlate with each other and the objective."
        )
        design_space_btn.clicked.connect(self._action_open_design_space)
        rbtn_row.addWidget(design_space_btn)
        rbtn_row.addStretch()
        export_btn = QPushButton("Export CSV")
        export_btn.setToolTip("Export all completed trial results to a CSV file.")
        export_btn.clicked.connect(self._action_export_results)
        rbtn_row.addWidget(export_btn)
        export_xlsx_btn = QPushButton("Export Excel")
        export_xlsx_btn.setToolTip(
            "Export all completed trial results to an Excel (.xlsx) file.\n"
            "Numbers are stored as numbers — no formatting issues.\n"
            "Requires openpyxl (included in requirements.txt)."
        )
        export_xlsx_btn.clicked.connect(self._action_export_results_excel)
        rbtn_row.addWidget(export_xlsx_btn)
        rvbox.addLayout(rbtn_row)

        # ── Export Report + Surrogate Validation checkbox ──────────────
        rbtn_row2 = QHBoxLayout()
        export_report_btn = QPushButton("📄  Export Report…")
        export_report_btn.setToolTip(
            "Generate a self-contained HTML report of the optimisation session.\n"
            "Includes: session metadata, best results, all plots (embedded as\n"
            "PNG images), and the full trials table.\n"
            "The file works offline — no internet connection required."
        )
        export_report_btn.clicked.connect(self._action_export_report)
        rbtn_row2.addWidget(export_report_btn)
        rbtn_row2.addStretch()

        self._val_checkbox_main = QCheckBox("🔬  Surrogate Validation")
        self._val_checkbox_main.setEnabled(False)
        self._val_checkbox_main.setToolTip(
            "Run a rigorous validation of the surrogate model on the current dataset:\n"
            "  §0  Data quality check\n"
            "  §1  RF + GP hold-out and cross-validated accuracy\n"
            "  §2  Virtual BO benchmark (GP+EI vs RF+EI vs Greedy vs Random)\n"
            "  §3  Expected Improvement marginals per parameter\n"
            "  §5  Pass/fail summary\n\n"
            "Requires ≥ 30 unique rows.  Runtime: 3–10 minutes.\n"
            "Results are embedded in the HTML report export and shown\n"
            "as tabs in the Design Space dialog."
        )
        self._val_checkbox_main.toggled.connect(self._on_main_val_toggled)
        rbtn_row2.addWidget(self._val_checkbox_main)
        rvbox.addLayout(rbtn_row2)

        # ── Validation progress (hidden by default, above quality badge) ──
        self._val_progress_main = QProgressBar()
        self._val_progress_main.setRange(0, 100)
        self._val_progress_main.setValue(0)
        self._val_progress_main.setVisible(False)
        self._val_progress_main.setFixedHeight(14)
        rvbox.addWidget(self._val_progress_main)

        self._val_status_lbl_main = QLabel("")
        self._val_status_lbl_main.setStyleSheet("font-size: 10px; color: #a6e3a1;")
        self._val_status_lbl_main.setVisible(False)
        rvbox.addWidget(self._val_status_lbl_main)

        # ── Surrogate quality badge (Feature 2) ───────────────────────
        self._quality_badge = QLabel("Surrogate quality: —")
        self._quality_badge.setWordWrap(True)
        self._quality_badge.setAlignment(Qt.AlignCenter)
        self._quality_badge.setToolTip(
            "Cross-validated R² of a Random Forest fitted on all completed trials.\n"
            f"Green: R² > 0.85 (good)  |  Amber: R² 0.60–0.85 (moderate)\n"
            f"Red: R² < 0.60 (poor)    |  Grey: fewer than {MIN_TRIALS} completed trials."
        )
        self._quality_badge.setStyleSheet(
            "background-color: #45475a; color: #cdd6f4; "
            "padding: 4px 8px; border-radius: 4px;"
        )
        rvbox.addWidget(self._quality_badge)

        self._results_tabs = QTabWidget()

        self._trials_table = QTableWidget()
        self._trials_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._trials_table.horizontalHeader().setStretchLastSection(True)
        self._trials_table.setAlternatingRowColors(True)
        self._results_tabs.addTab(self._trials_table, "All Trials")

        self._pareto_table = QTableWidget()
        self._pareto_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._pareto_table.horizontalHeader().setStretchLastSection(True)
        self._pareto_table.setAlternatingRowColors(True)
        self._results_tabs.addTab(self._pareto_table, "Best / Pareto")

        self._convergence_widget = ConvergenceWidget()
        self._results_tabs.addTab(self._convergence_widget, "\U0001f4c8  Convergence")

        self._pareto_widget = ParetoWidget()
        self._results_tabs.addTab(self._pareto_widget, "📈  Pareto")

        rvbox.addWidget(self._results_tabs, stretch=1)
        self._left_tabs.addTab(results_widget, "📊  Results")

        # ══════════════════════════════════════════════════════════════
        # Tab 3 — 🔬 Power Analysis
        # Input form lives in the left dock; result panel + plots go in
        # the centre QStackedWidget (page 1, added here so _build_left_dock
        # and _build_centre can share self._centre_stack).
        # ══════════════════════════════════════════════════════════════
        self._power = PowerAnalysisCoordinator()
        self._left_tabs.addTab(self._power.input_panel, "🔬  Power")
        self._centre_stack.addWidget(self._power.result_panel)  # index 1
        self._left_tabs.currentChanged.connect(self._on_left_tab_changed)

        self._left_dock.setWidget(outer)
        self.addDockWidget(Qt.LeftDockWidgetArea, self._left_dock)
        # Give the dock a generous initial width so the results table is readable
        self.resizeDocks([self._left_dock], [480], Qt.Horizontal)

    # ══════════════════════════════════════════════════════════════════════
    # Centre — resume banner + param card scroll area
    # ══════════════════════════════════════════════════════════════════════

    def _build_centre(self) -> None:
        centre = QWidget()
        vbox = QVBoxLayout(centre)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(6)

        # Resume banner (hidden by default; style applied by _apply_theme())
        self._resume_banner = QFrame()
        self._resume_banner.setFrameShape(QFrame.StyledPanel)
        banner_h = QHBoxLayout(self._resume_banner)
        self._resume_label = QLabel()
        self._resume_label.setWordWrap(True)
        banner_h.addWidget(self._resume_label, stretch=1)
        load_updated_btn = QPushButton("Load Updated CSV…")
        load_updated_btn.clicked.connect(self._action_load_updated_csv)
        banner_h.addWidget(load_updated_btn)
        self._resume_banner.hide()
        vbox.addWidget(self._resume_banner)

        # Param card scroll area
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._cards_container = QWidget()
        self._cards_layout = QVBoxLayout(self._cards_container)
        self._cards_layout.setAlignment(Qt.AlignTop)
        self._cards_layout.setSpacing(6)

        self._placeholder_lbl = QLabel(
            "No CSV loaded.\n\nUse  File → New Session…  or  Load CSV…  to begin."
        )
        self._placeholder_lbl.setAlignment(Qt.AlignCenter)
        italic = QFont()
        italic.setItalic(True)
        self._placeholder_lbl.setFont(italic)
        self._cards_layout.addWidget(self._placeholder_lbl)

        self._scroll.setWidget(self._cards_container)
        # Add scroll area as page 0 of the centre stacked widget.
        # Page 1 (power result panel) is added in _build_left_dock().
        self._centre_stack.addWidget(self._scroll)   # index 0
        vbox.addWidget(self._centre_stack)

        self.setCentralWidget(centre)

    # ══════════════════════════════════════════════════════════════════════
    # Status bar
    # ══════════════════════════════════════════════════════════════════════

    def _build_status_bar(self) -> None:
        sb = self.statusBar()
        self._status_trials = QLabel("Trials: 0")
        self._status_best   = QLabel("Best: —")
        self._status_text   = QLabel("Status: Ready")
        for lbl in (self._status_trials, self._status_best, self._status_text):
            sb.addWidget(lbl)
            sb.addWidget(self._make_separator())

        # ── Repository credit (right-aligned permanent widget) ─────────────
        _credit = QLabel(
            '<a href="https://github.com/SimonSmith95/LabOpt" '
            'style="color:#585b70; font-size:11px; text-decoration:none;">'
            "github.com/SimonSmith95/LabOpt</a>"
        )
        _credit.setOpenExternalLinks(True)
        _credit.setContentsMargins(0, 0, 8, 0)
        sb.addPermanentWidget(_credit)

    @staticmethod
    def _make_separator() -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        return sep

    # ══════════════════════════════════════════════════════════════════════
    # CSV loading
    # ══════════════════════════════════════════════════════════════════════

    def _action_load_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Experiment Data File", "",
            "Data files (*.csv *.xlsx *.xls);;"
            "CSV files (*.csv);;"
            "Excel files (*.xlsx *.xls)"
        )
        if path:
            self._do_load_csv(path)

    def _action_reload_csv(self) -> None:
        if self._csv_path:
            self._do_load_csv(self._csv_path)
        elif self._session_state and self._session_state.csv_path:
            self._do_load_csv(self._session_state.csv_path)

    def _do_load_csv(self, path: str) -> None:
        try:
            df = load_csv(path)
        except ValueError as exc:
            QMessageBox.critical(self, "CSV Error", str(exc))
            return
        self._df = df
        self._csv_path = path
        if self._session_state:
            self._session_state.csv_path = path
        self._csv_path_label.setText(f"  {os.path.basename(path)}")
        self._populate_objectives_selector(df)
        self._set_status("CSV loaded")

    def _populate_objectives_selector(self, df: pd.DataFrame) -> None:
        # Clear existing objective widgets
        while self._obj_inner_layout.count():
            item = self._obj_inner_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._obj_col_rows = []

        self._obj_inner_layout.addWidget(QLabel("Select result column(s):"))

        for col in df.columns:
            row_w = QWidget()
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 0, 0, 0)

            cb = QCheckBox(col)
            dir_combo = QComboBox()
            dir_combo.addItems(["minimize", "maximize"])
            dir_combo.setFixedWidth(96)
            dir_combo.hide()
            cb.toggled.connect(lambda checked, c=dir_combo: c.setVisible(checked))

            row_l.addWidget(cb)
            row_l.addWidget(dir_combo)
            self._obj_inner_layout.addWidget(row_w)
            self._obj_col_rows.append((col, cb, dir_combo))

        # ── Context Variables section ─────────────────────────────────────
        self._ctx_col_checkboxes = []
        ctx_header = QLabel("Context variables (uncontrollable conditions):")
        ctx_header.setToolTip(
            "Mark columns that are measured environmental conditions you cannot\n"
            "control (e.g. humidity, atmospheric pressure, reagent purity).\n"
            "The surrogate will learn from them without suggesting their values."
        )
        ctx_header.setStyleSheet("color: #a6e3a1; font-size: 11px;")
        self._obj_inner_layout.addWidget(ctx_header)

        for col in df.columns:
            ctx_cb = QCheckBox(col)
            ctx_cb.setStyleSheet("color: #a6e3a1;")
            ctx_cb.setToolTip(
                f"Mark '{col}' as a context variable.\n"
                "It will be excluded from the parameter search space and treated\n"
                "as an environmental condition the surrogate can learn from."
            )
            self._obj_inner_layout.addWidget(ctx_cb)
            self._ctx_col_checkboxes.append((col, ctx_cb))

        # ── Replicate aggregation controls (Feature 8) ────────────────────
        repl_row = QWidget()
        repl_layout = QHBoxLayout(repl_row)
        repl_layout.setContentsMargins(0, 4, 0, 2)
        repl_layout.setSpacing(6)

        self._repl_cb = QCheckBox("Aggregate replicates")
        self._repl_cb.setToolTip(
            "When enabled, rows with numeric parameter values that agree within\n"
            "the ±tolerance are treated as replicate measurements of the same\n"
            "experiment. Their objective values are averaged, and an n_replicates\n"
            "column is added to the All Trials table.\n\n"
            "Categorical/string parameters must always match exactly.\n"
            "Tolerance is in the same units as your parameter columns.\n"
            "  1e-6 = exact-match only (default)\n"
            "  0.5  = merge rows within ±0.5 of each numeric param\n"
            "  5    = merge rows within ±5 (e.g. temperatures in K)"
        )
        repl_layout.addWidget(self._repl_cb)
        repl_layout.addWidget(QLabel("  ±"))

        self._repl_tol_le = QLineEdit("1e-6")
        self._repl_tol_le.setFixedWidth(80)
        self._repl_tol_le.setEnabled(False)
        self._repl_tol_le.setPlaceholderText("1e-6")
        self._repl_tol_le.setToolTip("Absolute ±tolerance for numeric parameters.")
        repl_layout.addWidget(self._repl_tol_le)
        repl_layout.addStretch()

        # Enable the tolerance field only when the checkbox is on
        self._repl_cb.toggled.connect(self._repl_tol_le.setEnabled)
        self._obj_inner_layout.addWidget(repl_row)

        apply_btn = QPushButton("Apply Objectives →")
        apply_btn.clicked.connect(self._action_apply_objectives)
        self._obj_inner_layout.addWidget(apply_btn)

        constraint_btn = QPushButton("📐  Constraints…")
        constraint_btn.setToolTip(
            "Define rules that constrain how parameter values relate to each other.\n"
            "Examples:\n"
            "  • CsPbI + FAPbI + MAPbI = 1  (compositional fractions)\n"
            "  • temp * time ≤ 50000  (processing budget)"
        )
        constraint_btn.clicked.connect(self._action_open_constraints)
        self._obj_inner_layout.addWidget(constraint_btn)

    # ══════════════════════════════════════════════════════════════════════
    # Apply objectives → build param cards + study
    # ══════════════════════════════════════════════════════════════════════

    def _action_apply_objectives(self) -> None:
        if self._df is None:
            QMessageBox.warning(self, "No CSV", "Load a CSV file first.")
            return

        selected = [
            (col, combo.currentText())
            for col, cb, combo in self._obj_col_rows
            if cb.isChecked()
        ]
        if not selected:
            QMessageBox.warning(
                self, "No Objectives", "Select at least one result column."
            )
            return

        objectives = [ObjectiveConfig(col, direction) for col, direction in selected]
        result_cols = [o.column_name for o in objectives]

        # Collect context variable columns (checked green checkboxes)
        # Validate mutual exclusivity with objectives
        obj_col_set = set(result_cols)
        from parameter_config import ContextConfig
        ctx_configs = []
        ctx_col_names = []
        for col, ctx_cb in self._ctx_col_checkboxes:
            if ctx_cb.isChecked():
                if col in obj_col_set:
                    QMessageBox.warning(
                        self, "Column Conflict",
                        f"Column '{col}' cannot be both an objective and a context variable.\n"
                        "Please uncheck it from one of the two sections."
                    )
                    return
                ctx_configs.append(ContextConfig(column_name=col))
                ctx_col_names.append(col)

        params = extract_param_defaults(self._df, result_cols, context_columns=ctx_col_names)

        config = StudyConfig(
            parameters=params,
            objectives=objectives,
            constraints=list(self._constraints),
            context_variables=ctx_configs,
            batch_size=self._batch_size_spin.value(),
            n_batches=self._n_batches_spin.value(),
            sampler_name=self._sampler_combo.currentText(),
        )

        # Rebuild the Current Conditions spinboxes for any context variables
        self._rebuild_context_conditions_ui(ctx_configs)

        # Create a new session if one doesn't exist yet
        if self._session_state is None:
            session_dir = (
                os.path.dirname(self._csv_path)
                if self._csv_path
                else os.path.join(os.path.expanduser("~"), "bhop_sessions")
            )
            self._session_state = SessionManager.create_new_session(
                config, self._csv_path or "", session_dir
            )

        self._session_state.study_config = config
        # Persist the updated config (including constraints) immediately so it
        # survives a restart even if the user never runs a batch.
        SessionManager.save(self._session_state)

        # Build / reload the Optuna study
        self._study = build_study(
            config,
            self._session_state.storage_path,
            self._session_state.study_name,
        )

        # ── Replicate aggregation (Feature 8) ─────────────────────────────
        repl_enabled = (
            self._repl_cb is not None and self._repl_cb.isChecked()
        )
        repl_tol = 1e-6
        if repl_enabled and self._repl_tol_le is not None:
            try:
                repl_tol = float(self._repl_tol_le.text())
            except ValueError:
                repl_tol = 1e-6

        # Persist replicate settings to config so they survive session save/load
        config.replicate_aggregation = repl_enabled
        config.replicate_tolerance   = repl_tol
        self._session_state.study_config = config
        SessionManager.save(self._session_state)

        # Build the loading DataFrame (aggregated or raw)
        param_col_names = [p.name for p in params if p.enabled]
        obj_col_names   = [o.column_name for o in objectives]
        n_raw = len(self._df)

        if repl_enabled and repl_tol > 0:
            df_to_load = aggregate_replicates(
                self._df, param_col_names, obj_col_names, tolerance=repl_tol
            )
        else:
            df_to_load = self._df.copy()
            df_to_load["n_replicates"] = 1

        n_agg = len(df_to_load)

        # Seed historical trials from (potentially aggregated) DataFrame
        trial_dicts = load_trials_from_csv(df_to_load, config)

        # Inject n_replicates into each trial dict's user_attrs so the
        # All Trials table can display it.  The injection aligns trial_dicts
        # with the valid rows from df_to_load (same filter as load_trials_from_csv).
        if "n_replicates" in df_to_load.columns:
            param_names_enabled = [p.name for p in config.parameters if p.enabled]
            filtered_rep_counts: list[int] = []
            for _, row in df_to_load.iterrows():
                if any(pd.isna(row.get(c)) for c in obj_col_names):
                    continue
                row_params = {
                    name: row[name]
                    for name in param_names_enabled
                    if name in row.index and not pd.isna(row[name])
                }
                if len(row_params) < len([n for n in param_names_enabled if n in row.index]):
                    continue
                try:
                    filtered_rep_counts.append(int(row["n_replicates"]))
                except (TypeError, ValueError):
                    filtered_rep_counts.append(1)
            for i, td in enumerate(trial_dicts):
                n_rep = filtered_rep_counts[i] if i < len(filtered_rep_counts) else 1
                td.setdefault("user_attrs", {})["n_replicates"] = n_rep

        added, skipped = load_historical_trials(self._study, trial_dicts, config)

        # Build parameter cards
        self._build_param_cards(params)
        self._ask_btn.setEnabled(True)
        self._update_warnings()
        self._refresh_results_tables()
        self._update_status_bar()
        self._refresh_recent_menu()
        self._refresh_design_space()   # show historical data, no suggestions yet
        self._update_main_val_checkbox()  # enable if ≥ 30 unique rows

        # Status message describes whether aggregation occurred
        if repl_enabled and n_agg < n_raw:
            self._set_status(
                f"Ready — {n_agg} aggregated trials loaded from {n_raw} rows "
                f"(tolerance ±{repl_tol}), {skipped} skipped."
            )
        else:
            self._set_status(
                f"Ready — {added} historical trials loaded, {skipped} skipped."
            )

    # ══════════════════════════════════════════════════════════════════════
    # Constraint editor
    # ══════════════════════════════════════════════════════════════════════

    def _action_open_constraints(self) -> None:
        """Open the ConstraintDialog so the user can add / edit / remove rules."""
        # Build the list of available parameter names from the current CSV
        # (all non-objective columns).
        param_names: List[str] = []
        if self._df is not None and self._session_state is not None:
            result_cols = {o.column_name for o in self._session_state.study_config.objectives}
            param_names = [c for c in self._df.columns if c not in result_cols]
        elif self._df is not None:
            param_names = list(self._df.columns)

        dlg = ConstraintDialog(
            constraints=list(self._constraints),
            param_names=param_names,
            parent=self,
        )
        if dlg.exec() == QDialog.Accepted:
            self._constraints = dlg.constraints
            # Persist immediately so constraints survive a restart without needing
            # the user to click "Apply Objectives" again.
            if self._session_state:
                self._session_state.study_config.constraints = list(self._constraints)
                SessionManager.save(self._session_state)
            n = len(self._constraints)
            self._set_status(
                f"{n} constraint(s) defined. "
                "Click 'Apply Objectives' to rebuild the study with these constraints."
            )

    # ══════════════════════════════════════════════════════════════════════
    # Parameter cards
    # ══════════════════════════════════════════════════════════════════════

    def _rebuild_context_conditions_ui(self, ctx_configs: list) -> None:
        """
        Create or update the "🌡 Current Conditions" group box in the Settings
        scrollable area.  Called by _action_apply_objectives whenever the set
        of context variables changes.

        If no context variables are defined the group is hidden.
        """
        # Remove the old group box from the layout if it exists
        if self._ctx_conditions_group is not None:
            self._ctx_conditions_group.hide()
            self._ctx_conditions_group.deleteLater()
            self._ctx_conditions_group = None
        self._context_spinboxes.clear()

        if not ctx_configs:
            return   # no context variables — nothing to show

        # Build the group box
        from parameter_config import ContextConfig
        grp = QGroupBox("🌡  Current Conditions")
        grp.setToolTip(
            "Enter expected environmental conditions for the NEXT batch of experiments.\n"
            "These values are used to condition the surrogate suggestions.\n"
            "You can correct them per-trial in the Batch Results Dialog if\n"
            "actual conditions differed from planned."
        )
        grp_layout = QVBoxLayout(grp)

        help_lbl = QLabel(
            "<i>Enter expected conditions before asking for the next batch.<br>"
            "Correct per-trial in the results dialog if conditions differed.</i>"
        )
        help_lbl.setWordWrap(True)
        help_lbl.setTextFormat(Qt.RichText)
        help_lbl.setStyleSheet("color: #a6e3a1; font-size: 11px;")
        grp_layout.addWidget(help_lbl)

        # One spinbox per context variable, pre-filled with historical mean
        ctx_defaults: dict = {}
        if self._df is not None:
            col_names = [c.column_name for c in ctx_configs]
            for d in infer_context_defaults(self._df, col_names):
                ctx_defaults[d["column_name"]] = d["mean"]

        form = QFormLayout()
        form.setSpacing(4)
        for cv in ctx_configs:
            spin = QDoubleSpinBox()
            spin.setRange(-1e9, 1e9)
            spin.setDecimals(4)
            spin.setSingleStep(1.0)
            default = ctx_defaults.get(cv.column_name, 0.0)
            spin.setValue(default)
            spin.setToolTip(
                f"Planned value of '{cv.column_name}' for the next experiments.\n"
                f"Historical mean: {default:.4g}"
            )
            label_text = cv.description if cv.description else cv.column_name
            form.addRow(f"{label_text}:", spin)
            self._context_spinboxes[cv.column_name] = spin

        grp_layout.addLayout(form)
        self._ctx_conditions_group = grp

        # Insert just before the warning label in the scrollable vbox.
        # The scrollable content widget's layout is the one inside _settings_scroll.
        # We find it via the warning label's parent layout.
        vbox = self._warning_label.parent().layout()
        if vbox is not None:
            # Find the index of the warning label so we can insert before it
            warn_idx = -1
            for i in range(vbox.count()):
                if vbox.itemAt(i) and vbox.itemAt(i).widget() is self._warning_label:
                    warn_idx = i
                    break
            if warn_idx >= 0:
                vbox.insertWidget(warn_idx, grp)
            else:
                vbox.addWidget(grp)
        grp.show()

    def _build_param_cards(self, params: List[ParameterConfig]) -> None:
        self._placeholder_lbl.hide()

        for card in self._param_cards:
            card.deleteLater()
        self._param_cards.clear()

        for param in params:
            card = ParamCardWidget(param, self._cards_container)
            card.config_changed.connect(self._update_warnings)
            self._cards_layout.addWidget(card)
            self._param_cards.append(card)

    # ══════════════════════════════════════════════════════════════════════
    # Phase 10 — Warning system
    # ══════════════════════════════════════════════════════════════════════

    def _update_warnings(self) -> None:
        n_enabled = sum(1 for c in self._param_cards if c.get_config().enabled)
        n_trials   = len(self._study.trials) if self._study else 0
        sampler    = self._sampler_combo.currentText()
        batch_size = self._batch_size_spin.value()

        warnings: List[str] = []

        if sampler == "TPE" and n_enabled > TPE_PARAM_WARNING_THRESHOLD:
            warnings.append(
                f"⚠ TPE may struggle with {n_enabled} parameters "
                f"(recommended ≤ {TPE_PARAM_WARNING_THRESHOLD}). "
                "Consider switching to NSGAII or reducing parameters."
            )
        if n_enabled > 0 and n_trials < MIN_TRIALS_PER_PARAM_RATIO * n_enabled:
            warnings.append(
                f"⚠ Only {n_trials} historical trials for {n_enabled} parameters. "
                "Initial suggestions may be poor — add more data first."
            )
        if sampler == "TPE" and n_trials > 0 and batch_size > n_trials:
            warnings.append(
                f"⚠ Batch size ({batch_size}) exceeds historical trial count ({n_trials}). "
                "TPE batch quality degrades without sufficient prior data."
            )

        # GP performance warning: O(n³) cost becomes significant above ~200 trials
        if sampler == "GP" and self._study:
            n_completed = sum(
                1 for t in self._study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            )
            if n_completed > 200:
                warnings.append(
                    f"⚠ GP may be slow for {n_completed} completed trials (> 200). "
                    "Consider switching to TPE for faster suggestions."
                )

        if warnings:
            self._warning_label.setText("\n\n".join(warnings))
            self._warning_label.show()
        else:
            self._warning_label.hide()

        self._batches_done_label.setText(
            f"{self._batches_done} / {self._n_batches_spin.value()}"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Ask next batch
    # ══════════════════════════════════════════════════════════════════════

    def _action_ask_next_batch(self) -> None:
        if self._study is None or self._session_state is None:
            QMessageBox.warning(
                self, "No Session", "Load a CSV and apply objectives first."
            )
            return

        # Warn confirmation
        if self._warning_label.isVisible():
            reply = QMessageBox.question(
                self,
                "Warnings Active",
                f"There are active warnings:\n\n{self._warning_label.text()}\n\n"
                "Proceed anyway?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        self._sync_config_from_ui()

        config = self._session_state.study_config
        self._worker = OptimizationWorker(
            self._study, config, SessionManager, self._session_state, parent=self
        )
        self._worker.batch_ready.connect(self._on_batch_ready)
        self._worker.batch_complete.connect(self._on_batch_complete)
        self._worker.optimization_done.connect(self._on_optimization_done)
        self._worker.error.connect(self._on_worker_error)

        # Collect planned context values from the Current Conditions spinboxes
        if self._context_spinboxes:
            self._planned_context = {
                col: spin.value()
                for col, spin in self._context_spinboxes.items()
            }
            self._worker.planned_context = self._planned_context
            # Persist planned context in session state for resume support
            if self._session_state:
                self._session_state.pending_planned_context = self._planned_context
                SessionManager.save(self._session_state)
        else:
            self._planned_context = None
            self._worker.planned_context = None

        self._ask_btn.setEnabled(False)
        self._pause_btn.setEnabled(True)
        self._set_status("Asking batch…")
        self._worker.start()

    def _sync_config_from_ui(self) -> None:
        if not self._session_state:
            return
        cfg = self._session_state.study_config
        cfg.batch_size   = self._batch_size_spin.value()
        cfg.n_batches    = self._n_batches_spin.value()
        cfg.sampler_name = self._sampler_combo.currentText()
        cfg.parameters   = [c.get_config() for c in self._param_cards]
        # Auto-stop settings (Feature 12)
        cfg.auto_stop                = self._auto_stop_cb.isChecked()
        cfg.auto_stop_min_improvement = self._as_improve_spin.value() / 100.0
        cfg.auto_stop_n_batches      = self._as_nbatches_spin.value()

    # ── Worker callbacks ───────────────────────────────────────────────────

    def _on_batch_ready(self, param_dicts: list) -> None:
        if not self._session_state or not self._session_state.pending_batch:
            return

        # Overlay the suggested points on the design space BEFORE opening the dialog
        # so the user can switch to the Design Space tab to review placement
        suggestions = [item["params"] for item in self._session_state.pending_batch]
        self._refresh_design_space(suggestions=suggestions)

        from optuna.trial import TrialState as _TS
        _existing = [
            {"number": t.number, **t.params}
            for t in self._study.trials if t.state == _TS.COMPLETE
        ]
        # Compute surrogate predictions for each suggestion so the dialog
        # can show the expected value alongside the actual input field.
        try:
            _predictions = predict_batch(
                self._study,
                self._session_state.study_config,
                suggestions,
            )
        except Exception:
            _predictions = None
        dlg = BatchResultsDialog(
            self._session_state.pending_batch,
            self._session_state.study_config.objectives,
            self._session_state,
            parent=self,
            existing_trials=_existing,
            study=self._study,
            predictions=_predictions,
            planned_context=self._planned_context,
        )
        self._active_batch_dlg = dlg   # keep ref so _on_results_submitted can call get_context_corrections()
        dlg.results_submitted.connect(self._on_results_submitted)
        if dlg.exec() != QDialog.Accepted:
            # User closed/cancelled the dialog without submitting results.
            # The pending batch was already saved to disk by the worker
            # (mark_batch_pending was called before batch_ready was emitted).
            # We must unblock the worker so it can exit cleanly, but we do NOT
            # re-enable "Ask Next Batch" — there are still WAITING trials in
            # Optuna.  Instead we put the UI into the "pending" state so the
            # user can enter results later via the toolbar button.
            if self._worker:
                self._worker.cancel_results()   # unblocks the worker thread
            self._ask_btn.setEnabled(False)
            self._pause_btn.setEnabled(False)
            # _on_optimization_done will fire next (worker exits); it will
            # detect the pending batch and skip the "All done!" message.
            # Pre-emptively update the UI so it is correct even before that signal.
            self._update_pending_btn_state()
            n_pending = (len(self._session_state.pending_batch)
                         if self._session_state and self._session_state.pending_batch
                         else 0)
            self._resume_label.setText(
                f"⚠  <b>{n_pending}</b> pending trial(s) awaiting lab results.\n"
                "Click <b>📋 Enter Pending Results…</b> in the toolbar when you "
                "have your measurements."
            )
            self._resume_banner.show()
            self._set_status(
                "Results not entered — use '📋 Enter Pending Results…' when ready."
            )

    def _on_results_submitted(self, results: list) -> None:
        # Append actual compositions + results to the CSV before telling Optuna
        self._append_results_to_csv(results)
        if self._worker:
            # Collect context corrections (actual conditions) from the dialog
            # and pass them to the worker before unblocking it.
            if self._active_batch_dlg is not None:
                try:
                    ctx_corrections = self._active_batch_dlg.get_context_corrections()
                    self._worker.pending_context_values = ctx_corrections
                except Exception:
                    self._worker.pending_context_values = None
            self._worker.submit_results(results)
        self._active_batch_dlg = None
        self._update_pending_btn_state()
        self._set_status("Submitting results…")

    def _on_batch_complete(self, done: int, total: int) -> None:
        self._batches_done = done
        self._batches_done_label.setText(f"{done} / {total}")
        self._refresh_results_tables()
        self._update_status_bar()
        # Auto-switch to Results tab so the user sees the new data immediately
        self._left_tabs.setCurrentIndex(1)
        self._set_status(f"Batch {done}/{total} complete.")

    def _on_optimization_done(self, reason: str = "completed") -> None:
        """
        Called when the worker loop ends.

        Parameters
        ----------
        reason : str
            "completed"  — all n_batches processed normally.
            "converged"  — auto-stop triggered (no improvement in N batches).
            "cancelled"  — stop() was called or the user cancelled results.
        """
        self._pause_btn.setEnabled(False)
        # Guard: if there is still a pending batch, the worker exited because
        # the user closed the results dialog without submitting.  Don't show
        # any "done" message or re-enable Ask.
        if self._session_state and self._session_state.pending_batch:
            self._ask_btn.setEnabled(False)
            self._update_pending_btn_state()
            return

        self._ask_btn.setEnabled(True)
        self._refresh_results_tables()

        if reason == "converged":
            cfg = self._session_state.study_config if self._session_state else None
            n_window = cfg.auto_stop_n_batches if cfg else 3
            pct      = (cfg.auto_stop_min_improvement * 100) if cfg else 1.0
            self._set_status(
                f"Converged after {self._batches_done} batch(es) — "
                f"best value unchanged (< {pct:.1f}%) "
                f"for {n_window} consecutive batches."
            )
            QMessageBox.information(
                self,
                "Auto-Stop: Converged",
                f"Optimisation converged after {self._batches_done} batch(es).\n\n"
                f"The best objective value has not improved by more than {pct:.1f}%\n"
                f"in the last {n_window} consecutive batches.\n\n"
                "You can continue by clicking 'Ask Next Batch' or review the results.",
            )
        elif reason == "cancelled":
            self._set_status("Optimisation paused.")
        else:   # "completed"
            total = self._n_batches_spin.value()
            self._set_status(
                f"Optimisation complete ({self._batches_done}/{total} batches)."
            )
            QMessageBox.information(self, "Done", "All batches complete!")

    def _on_worker_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Worker Error", msg)
        self._ask_btn.setEnabled(True)
        self._pause_btn.setEnabled(False)
        self._set_status("Error — see dialog.")

    def _action_toggle_pause(self, checked: bool) -> None:
        if self._worker:
            if checked:
                self._worker.pause()
                self._pause_btn.setText("Resume")
                self._set_status("Paused.")
            else:
                self._worker.resume()
                self._pause_btn.setText("Pause")
                self._set_status("Running…")

    # ══════════════════════════════════════════════════════════════════════
    # Resume path (pending batch on session load)
    # ══════════════════════════════════════════════════════════════════════

    def _action_load_updated_csv(self) -> None:
        """Resume path: user loads CSV with completed results."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Updated CSV", "", "CSV Files (*.csv)"
        )
        if not path:
            return
        try:
            df = load_csv(path)
        except ValueError as exc:
            QMessageBox.critical(self, "CSV Error", str(exc))
            return
        self._df = df
        self._csv_path = path

        if not self._session_state or not self._session_state.pending_batch:
            return

        matches = SessionManager.match_pending_to_csv(self._session_state, df)
        dlg = BatchResultsDialog(
            self._session_state.pending_batch,
            self._session_state.study_config.objectives,
            self._session_state,
            parent=self,
        )
        if matches:
            dlg.prefill_results(matches)
        dlg.results_submitted.connect(self._on_resume_results_submitted)
        dlg.exec()

    def _on_resume_results_submitted(self, results: list) -> None:
        if not self._session_state or not self._study:
            return
        trial_numbers = [r["trial_number"] for r in results]
        values_list   = [r["values"]        for r in results]
        self._append_results_to_csv(results)
        # Pass constrained params so the surrogate trains on feasible values.
        pending_by_num = {
            p["trial_number"]: p["params"]
            for p in (self._session_state.pending_batch or [])
        }
        constrained = [pending_by_num.get(n, {}) for n in trial_numbers]
        tell_batch(self._study, trial_numbers, values_list,
                   constrained, self._session_state.study_config)
        SessionManager.clear_pending_batch(self._session_state)
        self._resume_banner.hide()
        self._refresh_results_tables()
        self._update_status_bar()
        self._update_pending_btn_state()
        self._ask_btn.setEnabled(True)
        self._set_status("Results submitted. Ready for next batch.")
        QMessageBox.information(
            self, "Resumed", "Results submitted. Ready to ask the next batch."
        )

    # ══════════════════════════════════════════════════════════════════════
    # Session management
    # ══════════════════════════════════════════════════════════════════════

    def _action_new_session(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV for New Session", "", "CSV Files (*.csv)"
        )
        if path:
            self._session_state = None  # will be created on Apply Objectives
            self._do_load_csv(path)
            self._set_status("New session: configure objectives, then click Apply Objectives.")

    def _action_load_session(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Session", "", "Session Files (*_session.json)"
        )
        if path:
            self._load_session_from_path(path)

    def _load_session_from_path(self, path: str) -> None:
        try:
            state = SessionManager.load(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            return

        self._session_state = state
        cfg = state.study_config

        try:
            self._study = build_study(cfg, state.storage_path, state.study_name)
        except Exception as exc:
            QMessageBox.critical(self, "Study Load Error", str(exc))
            return

        # Restore UI from config (including constraints)
        self._constraints = list(cfg.constraints)
        self._sampler_combo.setCurrentText(cfg.sampler_name)
        self._batch_size_spin.setValue(cfg.batch_size)
        self._n_batches_spin.setValue(cfg.n_batches)
        # Auto-stop settings (Feature 12)
        self._auto_stop_cb.setChecked(cfg.auto_stop)
        self._as_improve_spin.setValue(cfg.auto_stop_min_improvement * 100.0)
        self._as_nbatches_spin.setValue(cfg.auto_stop_n_batches)
        self._as_improve_spin.setEnabled(cfg.auto_stop)
        self._as_nbatches_spin.setEnabled(cfg.auto_stop)
        self._build_param_cards(cfg.parameters)
        self._refresh_results_tables()
        self._update_warnings()
        self._update_status_bar()
        self._ask_btn.setEnabled(True)
        self._refresh_recent_menu()

        # Load CSV if path is still valid
        if state.csv_path and os.path.exists(state.csv_path):
            try:
                df = load_csv(state.csv_path)
                self._df = df
                self._csv_path = state.csv_path
                self._csv_path_label.setText(f"  {os.path.basename(state.csv_path)}")
                self._populate_objectives_selector(df)
                # Restore replicate aggregation settings from saved config
                if self._repl_cb is not None:
                    self._repl_cb.setChecked(cfg.replicate_aggregation)
                if self._repl_tol_le is not None:
                    self._repl_tol_le.setText(str(cfg.replicate_tolerance))
                    self._repl_tol_le.setEnabled(cfg.replicate_aggregation)
            except Exception:
                pass

        # Check for pending batch — update toolbar button and optional banner
        self._update_pending_btn_state()
        if state.pending_batch:
            n = len(state.pending_batch)
            self._resume_label.setText(
                f"⚠  Session has <b>{n}</b> pending trial(s) awaiting lab results.\n"
                "Click <b>📋 Enter Pending Results…</b> in the toolbar to enter your "
                "measurements, or use 'Load Updated CSV…' to auto-fill from a file."
            )
            self._resume_banner.show()
        else:
            self._resume_banner.hide()

        self._set_status(f"Session '{state.study_name}' loaded.")

    # ══════════════════════════════════════════════════════════════════════
    # Config save / load
    # ══════════════════════════════════════════════════════════════════════

    def _action_save_config(self) -> None:
        if not self._session_state:
            QMessageBox.warning(self, "No Config", "No active session to save from.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Config", "", "JSON Files (*.json)"
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self._session_state.study_config.to_dict(), fh, indent=2)

    def _action_load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Config", "", "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                cfg = StudyConfig.from_dict(json.load(fh))
        except Exception as exc:
            QMessageBox.critical(self, "Config Error", str(exc))
            return

        if self._session_state:
            self._session_state.study_config = cfg
        # Restore constraints to UI state so they are included in the next Apply
        self._constraints = list(cfg.constraints)
        self._sampler_combo.setCurrentText(cfg.sampler_name)
        self._batch_size_spin.setValue(cfg.batch_size)
        self._n_batches_spin.setValue(cfg.n_batches)
        self._build_param_cards(cfg.parameters)
        self._update_warnings()

    # ══════════════════════════════════════════════════════════════════════
    # Results tables
    # ══════════════════════════════════════════════════════════════════════

    def _refresh_results_tables(self) -> None:
        if not self._study:
            self._convergence_widget.clear()
            self._pareto_widget.clear()
            self._power.set_objective_data(None)
            return
        from optuna.trial import TrialState

        completed = [t for t in self._study.trials if t.state == TrialState.COMPLETE]
        if not completed:
            self._convergence_widget.clear()
            self._pareto_widget.clear()
            # Still pass df data so the calculator works as a planning tool
            self._refresh_power_widget(n_current=0)
            return

        obj_cols = (
            [o.column_name for o in self._session_state.study_config.objectives]
            if self._session_state
            else ["value"]
        )
        param_cols = list(completed[0].params.keys())
        # Include n_replicates column so users can see how many raw rows were merged
        all_cols   = ["Trial #"] + param_cols + obj_cols + ["n_replicates"]

        def _fill_table(table: QTableWidget, trials: list) -> None:
            table.setRowCount(len(trials))
            table.setColumnCount(len(all_cols))
            table.setHorizontalHeaderLabels(all_cols)
            for r, trial in enumerate(sorted(trials, key=lambda t: t.number)):
                table.setItem(r, 0, QTableWidgetItem(str(trial.number)))
                for ci, pname in enumerate(param_cols, start=1):
                    table.setItem(r, ci, QTableWidgetItem(str(trial.params.get(pname, ""))))
                for oi in range(len(obj_cols)):
                    if trial.values and oi < len(trial.values):
                        val = trial.values[oi]
                    elif trial.value is not None and oi == 0:
                        val = trial.value
                    else:
                        val = ""
                    table.setItem(r, len(param_cols) + 1 + oi, QTableWidgetItem(str(val)))
                # n_replicates: from user_attrs if set (historical trials with aggregation),
                # otherwise 1 (live batch tells)
                n_rep = 1
                if trial.user_attrs:
                    try:
                        n_rep = int(trial.user_attrs.get("n_replicates", 1))
                    except (TypeError, ValueError):
                        n_rep = 1
                table.setItem(
                    r, len(param_cols) + 1 + len(obj_cols),
                    QTableWidgetItem(str(n_rep)),
                )
            table.resizeColumnsToContents()

        _fill_table(self._trials_table, completed)
        _fill_table(self._pareto_table, get_pareto_front(self._study))

        # Refresh the convergence plot
        objectives = (
            self._session_state.study_config.objectives
            if self._session_state
            else []
        )
        if objectives:
            self._convergence_widget.refresh(self._study, objectives)
        else:
            self._convergence_widget.clear()

        # Refresh the Pareto scatter widget (multi-objective only)
        if objectives and len(objectives) >= 2:
            self._pareto_widget.refresh(self._study, objectives)
        else:
            self._pareto_widget.clear()

        # Refresh surrogate quality badge
        self._update_quality_badge()

        # Refresh power analysis widget
        self._refresh_power_widget(n_current=len(completed))

    # ══════════════════════════════════════════════════════════════════════
    # Feature 2 — Surrogate quality badge
    # ══════════════════════════════════════════════════════════════════════

    def _refresh_power_widget(self, n_current: int = 0) -> None:
        """Feed the power analysis coordinator with the first objective's data column."""
        if self._session_state is None or self._df is None:
            self._power.set_objective_data(None, n_current=n_current)
            return
        objectives = self._session_state.study_config.objectives
        if not objectives:
            self._power.set_objective_data(None, n_current=n_current)
            return
        obj_col = objectives[0].column_name
        if obj_col not in self._df.columns:
            self._power.set_objective_data(None, n_current=n_current)
            return
        series = self._df[obj_col].dropna()
        self._power.set_objective_data(series, n_current=n_current)

    def _on_left_tab_changed(self, index: int) -> None:
        """
        Show/hide the power result panel in the centre area.

        Index 2 = 🔬 Power tab → show result panel.
        Any other tab → show param cards.

        Uses setCurrentWidget() instead of setCurrentIndex() so the correct
        widget is always shown regardless of the order they were inserted into
        the stack (_build_left_dock runs before _build_centre, so insertion
        order is not the same as the logical page numbers in the comments).
        """
        if index == 2:
            self._centre_stack.setCurrentWidget(self._power.result_panel)
        else:
            self._centre_stack.setCurrentWidget(self._scroll)

    def _update_quality_badge(self) -> None:
        """Recompute surrogate quality and update the badge label."""
        if not self._study or not self._session_state:
            self._quality_badge.setText("Surrogate quality: —")
            self._quality_badge.setStyleSheet(
                "background-color: #45475a; color: #cdd6f4; "
                "padding: 4px 8px; border-radius: 4px;"
            )
            return

        try:
            result = compute_surrogate_quality(
                self._study, self._session_state.study_config
            )
        except Exception:
            # Never let a quality computation error surface to the user
            self._quality_badge.setText("Surrogate quality: —")
            self._quality_badge.setStyleSheet(
                "background-color: #45475a; color: #cdd6f4; "
                "padding: 4px 8px; border-radius: 4px;"
            )
            return

        status = result["status"]
        n      = result["n"]
        r2     = result.get("r2")
        rmse   = result.get("rmse")

        pearson_r = result.get("pearson_r")
        r2_str    = f"{r2:.2f}"        if r2        is not None else "n/a"
        r_str     = f"{pearson_r:.2f}" if pearson_r is not None else "n/a"
        rmse_str  = f"{rmse:.4g}"      if rmse      is not None else "n/a"

        if status == "insufficient":
            text  = f"Surrogate quality: Insufficient data ({n}/{MIN_TRIALS})"
            style = (
                "background-color: #45475a; color: #cdd6f4; "
                "padding: 4px 8px; border-radius: 4px;"
            )
        elif status == "good":
            text  = (f"✅ Surrogate: Good  "
                     f"(CV R²={r2_str}, r={r_str}, RMSE={rmse_str}, n={n})")
            style = (
                "background-color: #2ecc71; color: #1e1e2e; "
                "padding: 4px 8px; border-radius: 4px; font-weight: bold;"
            )
        elif status == "moderate":
            text  = (f"⚠ Surrogate: Moderate — more data recommended  "
                     f"(CV R²={r2_str}, r={r_str}, n={n})")
            style = (
                "background-color: #f39c12; color: #1e1e2e; "
                "padding: 4px 8px; border-radius: 4px;"
            )
        else:  # poor
            text  = (f"🔴 Surrogate: Poor — suggestions less reliable  "
                     f"(CV R²={r2_str}, r={r_str}, n={n})")
            style = (
                "background-color: #e74c3c; color: #ffffff; "
                "padding: 4px 8px; border-radius: 4px;"
            )

        self._quality_badge.setText(text)
        self._quality_badge.setStyleSheet(style)

    def _update_status_bar(self) -> None:
        if not self._study:
            return
        from optuna.trial import TrialState

        done = [t for t in self._study.trials if t.state == TrialState.COMPLETE]
        self._status_trials.setText(f"Trials: {len(done)}")
        try:
            pareto = get_pareto_front(self._study)
            if pareto:
                t = pareto[0]
                if t.value is not None:
                    self._status_best.setText(f"Best: {t.value:.4g}")
                elif t.values:
                    vals = ", ".join(f"{v:.4g}" for v in t.values)
                    self._status_best.setText(f"Best: [{vals}]")
        except Exception:
            pass

    def _action_export_results(self) -> None:
        if not self._study:
            QMessageBox.warning(self, "No Study", "No study loaded.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "", "CSV Files (*.csv)"
        )
        if not path:
            return
        from optuna.trial import TrialState

        rows = []
        obj_cols = (
            [o.column_name for o in self._session_state.study_config.objectives]
            if self._session_state
            else []
        )
        for t in self._study.trials:
            if t.state != TrialState.COMPLETE:
                continue
            row: dict = {"trial_number": t.number}
            row.update(t.params)
            if t.values:
                for i, v in enumerate(t.values):
                    col = obj_cols[i] if i < len(obj_cols) else f"objective_{i}"
                    row[col] = v
            elif t.value is not None:
                col = obj_cols[0] if obj_cols else "value"
                row[col] = t.value
            rows.append(row)

        pd.DataFrame(rows).to_csv(path, index=False)
        QMessageBox.information(self, "Exported", f"Results exported to:\n{path}")

    # ══════════════════════════════════════════════════════════════════════
    # Feature 10 — Export as Excel (.xlsx)
    # ══════════════════════════════════════════════════════════════════════

    def _action_export_results_excel(self) -> None:
        """Export all completed trial results to an Excel (.xlsx) file."""
        if not self._study:
            QMessageBox.warning(self, "No Study", "No study loaded.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results as Excel", "",
            "Excel files (*.xlsx);;All files (*)"
        )
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"
        from optuna.trial import TrialState

        rows = []
        obj_cols = (
            [o.column_name for o in self._session_state.study_config.objectives]
            if self._session_state
            else []
        )
        for t in self._study.trials:
            if t.state != TrialState.COMPLETE:
                continue
            row: dict = {"trial_number": t.number}
            row.update(t.params)
            if t.values:
                for i, v in enumerate(t.values):
                    col = obj_cols[i] if i < len(obj_cols) else f"objective_{i}"
                    row[col] = v
            elif t.value is not None:
                col = obj_cols[0] if obj_cols else "value"
                row[col] = t.value
            rows.append(row)

        try:
            pd.DataFrame(rows).to_excel(path, index=False, engine="openpyxl")
            QMessageBox.information(self, "Exported", f"Results exported to:\n{path}")
        except ImportError:
            QMessageBox.critical(
                self, "Export Error",
                "openpyxl is required for Excel export.\n"
                "Install it with:  pip install openpyxl"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", f"Failed to export:\n{exc}")

    # ══════════════════════════════════════════════════════════════════════
    # Surrogate Validation — Results-tab checkbox handlers
    # ══════════════════════════════════════════════════════════════════════

    def _update_main_val_checkbox(self) -> None:
        """Enable the Results-tab validation checkbox iff n_unique ≥ 30."""
        if self._df is None or not self._session_state:
            self._val_checkbox_main.setEnabled(False)
            return
        cfg = self._session_state.study_config
        numeric_cols = [
            p.name for p in cfg.parameters
            if p.enabled
            and p.ptype in (ParameterType.FLOAT, ParameterType.INT)
            and p.name in self._df.columns
        ]
        if not numeric_cols:
            self._val_checkbox_main.setEnabled(False)
            return
        try:
            n_unique = len(
                self._df.dropna(subset=numeric_cols)
                        .drop_duplicates(subset=numeric_cols)
            )
        except Exception:
            n_unique = len(self._df)
        enabled = n_unique >= 30
        self._val_checkbox_main.setEnabled(enabled)
        if not enabled and self._val_checkbox_main.isChecked():
            self._val_checkbox_main.blockSignals(True)
            self._val_checkbox_main.setChecked(False)
            self._val_checkbox_main.blockSignals(False)

    def _on_main_val_toggled(self, checked: bool) -> None:
        if checked:
            if self._validation_results is not None:
                # Results already exist — push to dialog if open, show status
                if self._design_space_dlg is not None:
                    try:
                        self._design_space_dlg.set_validation_results(
                            self._validation_results
                        )
                    except Exception:
                        pass
                r = self._validation_results
                self._val_status_lbl_main.setText(
                    f"✅  {r.n_pass}/{r.n_checks} checks passed ({r.summary})"
                )
                self._val_status_lbl_main.setVisible(True)
                self._val_progress_main.setValue(100)
                self._val_progress_main.setVisible(True)
                return

            # Extract data arrays
            try:
                X, y, feature_names = self._extract_xy_for_val()
            except ValueError as exc:
                QMessageBox.warning(
                    self, "Validation Error",
                    f"Cannot start validation:\n{exc}"
                )
                self._val_checkbox_main.setChecked(False)
                return

            ctx_X, ctx_names = self._extract_context_for_val()
            objectives = (
                self._session_state.study_config.objectives
                if self._session_state else []
            )
            direction   = objectives[0].direction   if objectives else "minimize"
            target_name = objectives[0].column_name if objectives else "objective"

            # Show progress UI
            self._val_progress_main.setValue(0)
            self._val_progress_main.setVisible(True)
            self._val_status_lbl_main.setText("Starting validation…")
            self._val_status_lbl_main.setVisible(True)
            self._val_checkbox_main.setText("🔬  Running…")
            self._val_checkbox_main.setEnabled(False)

            from validation_worker import ValidationWorker
            self._validation_worker = ValidationWorker(
                X=X, y=y,
                feature_names=feature_names,
                target_name=target_name,
                direction=direction,
                context_X=ctx_X,
                context_names=ctx_names,
                parent=self,
            )
            self._validation_worker.progress_updated.connect(
                self._on_main_val_progress
            )
            self._validation_worker.validation_done.connect(self._on_main_val_done)
            self._validation_worker.error_occurred.connect(self._on_main_val_error)
            self._validation_worker.start()

        else:
            if self._validation_worker is not None and \
                    self._validation_worker.isRunning():
                self._validation_worker.terminate()
                self._validation_worker.wait(3000)
            self._val_progress_main.setVisible(False)
            self._val_status_lbl_main.setVisible(False)
            self._val_checkbox_main.setText("🔬  Surrogate Validation")
            self._update_main_val_checkbox()   # re-enable if still ≥ 30 rows

    def _on_main_val_progress(self, pct: int, msg: str) -> None:
        self._val_progress_main.setValue(pct)
        self._val_status_lbl_main.setText(msg)

    def _on_main_val_done(self, results) -> None:
        self._validation_results = results
        self._val_progress_main.setValue(100)
        self._val_status_lbl_main.setText(
            f"✅  {results.n_pass}/{results.n_checks} checks passed "
            f"({results.summary})"
        )
        self._val_checkbox_main.setText("🔬  Surrogate Validation  ✅")
        self._update_main_val_checkbox()   # re-enable
        # Push results to Design Space dialog if it is open
        if self._design_space_dlg is not None:
            try:
                self._design_space_dlg.set_validation_results(results)
            except Exception:
                pass

    def _on_main_val_error(self, msg: str) -> None:
        self._val_progress_main.setVisible(False)
        self._val_status_lbl_main.setVisible(False)
        self._val_checkbox_main.setText("🔬  Surrogate Validation  ❌")
        self._update_main_val_checkbox()   # re-enable if possible
        QMessageBox.critical(
            self, "Validation Error",
            f"The validation run failed:\n\n{msg[:600]}"
        )

    def _extract_xy_for_val(self):
        """Extract X, y, feature_names from the current df + session config."""
        if self._df is None:
            raise ValueError("No CSV loaded.")
        if not self._session_state:
            raise ValueError("No objectives configured (apply objectives first).")
        cfg = self._session_state.study_config
        if not cfg.objectives:
            raise ValueError("No objectives configured.")
        feature_cols = [
            p.name for p in cfg.parameters
            if p.enabled
            and p.ptype in (ParameterType.FLOAT, ParameterType.INT)
            and p.name in self._df.columns
        ]
        if not feature_cols:
            raise ValueError("No enabled numeric feature parameters found.")
        obj_col = cfg.objectives[0].column_name
        if obj_col not in self._df.columns:
            raise ValueError(f"Objective column '{obj_col}' not found in CSV.")
        sub = self._df[feature_cols + [obj_col]].dropna()
        if len(sub) < 5:
            raise ValueError(
                f"Too few valid rows after removing NaNs ({len(sub)} rows)."
            )
        import numpy as _np
        X = sub[feature_cols].values.astype(float)
        y = sub[obj_col].values.astype(float)
        return X, y, feature_cols

    def _extract_context_for_val(self):
        """Extract context variable arrays from the current session config."""
        if self._df is None or not self._session_state:
            return None, None
        ctx_names = [
            cv.column_name
            for cv in getattr(
                self._session_state.study_config, "context_variables", []
            )
            if cv.column_name in self._df.columns
        ]
        if not ctx_names:
            return None, None
        ctx_sub = self._df[ctx_names].copy()
        ctx_sub = ctx_sub.fillna(ctx_sub.mean())
        import numpy as _np
        return ctx_sub.values.astype(float), ctx_names

    # ══════════════════════════════════════════════════════════════════════
    # Feature 9 — Export HTML report
    # ══════════════════════════════════════════════════════════════════════

    def _action_export_report(self) -> None:
        """
        Generate a self-contained HTML report and open it in the browser.

        Collects:
          - Session metadata from _session_state
          - Completed trials as a DataFrame
          - Best/Pareto trial rows
          - Matplotlib figures from the Results tab widgets and (if open)
            the Design Space dialog widgets
        Then calls generate_report() → write_report() and offers to open
        the result in the system browser.
        """
        import datetime
        import webbrowser

        # ── Collect metadata ───────────────────────────────────────────────
        if self._session_state:
            cfg          = self._session_state.study_config
            session_name = self._session_state.study_name
            sampler      = cfg.sampler_name
            n_batches    = self._batches_done
            objectives   = [
                f"{o.column_name} ({o.direction})"
                for o in cfg.objectives
            ]
        else:
            session_name = "Untitled Session"
            sampler      = "—"
            n_batches    = 0
            objectives   = []

        metadata = {
            "session_name": session_name,
            "date": datetime.date.today().isoformat(),
            "sampler": sampler,
            "n_batches": n_batches,
            "objectives": objectives,
        }

        # ── Build trials DataFrame ─────────────────────────────────────────
        trials_rows: list[dict] = []
        best_rows:   list[dict] = []
        if self._study:
            from optuna.trial import TrialState
            cfg      = self._session_state.study_config if self._session_state else None
            obj_cols = [o.column_name for o in cfg.objectives] if cfg else []
            completed = [t for t in self._study.trials if t.state == TrialState.COMPLETE]
            for t in sorted(completed, key=lambda x: x.number):
                row: dict = {"trial_number": t.number}
                row.update(t.params)
                if t.values:
                    for i, v in enumerate(t.values):
                        col = obj_cols[i] if i < len(obj_cols) else f"obj_{i}"
                        row[col] = v
                elif t.value is not None:
                    col = obj_cols[0] if obj_cols else "value"
                    row[col] = t.value
                trials_rows.append(row)

            # Best / Pareto rows
            for t in get_pareto_front(self._study):
                row = {"trial_number": t.number}
                row.update(t.params)
                if t.values:
                    for i, v in enumerate(t.values):
                        col = obj_cols[i] if i < len(obj_cols) else f"obj_{i}"
                        row[col] = v
                elif t.value is not None:
                    col = obj_cols[0] if obj_cols else "value"
                    row[col] = t.value
                best_rows.append(row)

        trials_df = pd.DataFrame(trials_rows)

        # ── Collect figures ────────────────────────────────────────────────
        # NOTE: We force-refresh each widget before collecting its figure.
        # We do NOT use canvas.isVisible() because that returns False for any
        # widget inside a QTabWidget whose tab is not currently selected —
        # which would silently drop every plot that isn't on screen right now.
        figures: dict = {}

        if self._study and self._session_state:
            from optuna.trial import TrialState as _TS_exp
            _completed = [t for t in self._study.trials if t.state == _TS_exp.COMPLETE]
            _objectives = self._session_state.study_config.objectives if self._session_state else []

            # ── Convergence plot ───────────────────────────────────────────
            if _completed and _objectives:
                try:
                    self._convergence_widget.refresh(self._study, _objectives)
                    figures["convergence"] = self._convergence_widget._fig
                except Exception:
                    pass

            # ── Pareto scatter (multi-objective only) ──────────────────────
            if _completed and len(_objectives) >= 2:
                try:
                    self._pareto_widget.refresh(self._study, _objectives)
                    figures["pareto"] = self._pareto_widget._fig
                except Exception:
                    pass

        # ── Design Space, Correlation, Importance ─────────────────────────
        # Create the dialog if the user never opened it, then force-refresh
        # so all three canvases render fresh plots for the report.
        if self._df is not None and self._session_state:
            cfg = self._session_state.study_config
            try:
                if self._design_space_dlg is None:
                    self._design_space_dlg = DesignSpaceDialog(parent=self)
                # Refresh all tabs (pairplot, correlation, importance, profiler)
                self._design_space_dlg.refresh(
                    df=self._df,
                    params=cfg.parameters,
                    objectives=cfg.objectives,
                    suggestions=None,
                )
                # Collect each sub-widget figure if its canvas was drawn
                # (placeholder hidden means data exists and figure was rendered)
                ds_w = self._design_space_dlg._widget
                if not ds_w._placeholder.isHidden() or ds_w._canvas.isHidden() is False:
                    figures["design_space"] = ds_w._fig

                corr_w = self._design_space_dlg._corr_widget
                if not corr_w._placeholder.isVisible():
                    figures["correlation"] = corr_w._fig

                imp_w = self._design_space_dlg._importance_widget
                if not imp_w._placeholder.isVisible():
                    figures["importance"] = imp_w._fig
            except Exception:
                pass

        # ── Validation results (prefer Results-tab source; fall back to dialog) ──
        _val_res = self._validation_results or (
            self._design_space_dlg._validation_results
            if self._design_space_dlg is not None else None
        )
        if _val_res is not None:
            for key, fig in _val_res.figures.items():
                figures[f"validation_{key}"] = fig

        # ── Generate and write ─────────────────────────────────────────────
        try:
            html = generate_report(metadata, trials_df, figures, best_rows)
        except Exception as exc:
            QMessageBox.critical(
                self, "Report Error",
                f"Failed to generate report:\n{exc}"
            )
            return

        # Save to the session directory (same folder as the .db file)
        report_dir = (
            os.path.dirname(self._session_state.storage_path)
            if self._session_state
            else (os.path.dirname(self._csv_path) if self._csv_path else os.getcwd())
        )
        try:
            report_path = write_report(html, report_dir)
        except Exception as exc:
            QMessageBox.critical(
                self, "Report Save Error",
                f"Failed to save report:\n{exc}"
            )
            return

        # ── Export individual PNGs (200 dpi, PowerPoint-ready) ─────────────
        plots_dir = ""
        try:
            plots_dir = export_plots_as_png(figures, report_path, dpi=200)
        except Exception:
            pass   # non-fatal — HTML report is still valid

        n_pngs = len([f for f in os.listdir(plots_dir) if f.endswith(".png")]) \
            if plots_dir and os.path.isdir(plots_dir) else 0

        self._set_status(
            f"Report saved: {os.path.basename(report_path)}"
            + (f"  |  {n_pngs} PNG(s) in …_plots/" if n_pngs else "")
        )

        # ── Offer to open in browser ───────────────────────────────────────
        png_note = (
            f"\n\n📁 Individual plots (200 dpi PNG) saved to:\n"
            f"  {os.path.basename(plots_dir)}/\n"
            f"  → drag these directly into PowerPoint, Word, etc."
        ) if n_pngs else ""

        pdf_note = (
            "\n\n💡 To export as PDF: open the HTML file in your browser "
            "→ File → Print → Save as PDF"
        )

        reply = QMessageBox.question(
            self,
            "Report Exported",
            f"HTML report saved to:\n{report_path}"
            f"{png_note}"
            f"{pdf_note}"
            f"\n\nOpen HTML report in browser now?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            try:
                webbrowser.open(report_path)
            except Exception:
                pass   # non-fatal — user can open manually

    # ══════════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════════

    def _set_status(self, msg: str) -> None:
        self._status_text.setText(f"Status: {msg}")

    def _update_pending_btn_state(self) -> None:
        """Enable the 'Enter Pending Results…' button iff there is a pending batch."""
        has_pending = bool(
            self._session_state and self._session_state.pending_batch
        )
        self._pending_results_btn.setEnabled(has_pending)

    def _action_enter_pending_results(self) -> None:
        """
        Open the BatchResultsDialog for the current pending batch.

        This is the primary way to re-enter results after closing the dialog
        (e.g. because experiments take 1–2 days).  The user can correct the
        actual composition values before submitting.
        """
        if not self._session_state or not self._session_state.pending_batch:
            QMessageBox.information(
                self,
                "No Pending Batch",
                "There is no pending batch to enter results for.\n"
                "Use 'Ask Next Batch' to generate a new suggestion.",
            )
            return
        if not self._study:
            QMessageBox.warning(
                self, "No Study",
                "Load a session first (File → Load Session…).",
            )
            return

        from optuna.trial import TrialState as _TS2
        _existing2 = [
            {"number": t.number, **t.params}
            for t in self._study.trials if t.state == _TS2.COMPLETE
        ]
        dlg = BatchResultsDialog(
            self._session_state.pending_batch,
            self._session_state.study_config.objectives,
            self._session_state,
            parent=self,
            existing_trials=_existing2,
            study=self._study,
        )
        dlg.results_submitted.connect(self._on_pending_results_submitted)
        dlg.exec()

    def _on_pending_results_submitted(self, results: list) -> None:
        """
        Handle results entered via the toolbar '📋 Enter Pending Results…' button.

        Tells Optuna, appends to CSV, clears the pending batch, and re-enables
        the 'Ask Next Batch' button.
        """
        if not self._session_state or not self._study:
            return
        trial_numbers = [r["trial_number"] for r in results]
        values_list   = [r["values"]        for r in results]
        # Append actual compositions + results to the experiment CSV
        self._append_results_to_csv(results)
        # Pass constrained params so the surrogate trains on feasible values.
        pending_by_num = {
            p["trial_number"]: p["params"]
            for p in (self._session_state.pending_batch or [])
        }
        constrained = [pending_by_num.get(n, {}) for n in trial_numbers]
        tell_batch(self._study, trial_numbers, values_list,
                   constrained, self._session_state.study_config)
        SessionManager.clear_pending_batch(self._session_state)
        self._resume_banner.hide()
        self._refresh_results_tables()
        self._update_status_bar()
        self._update_pending_btn_state()
        self._ask_btn.setEnabled(True)
        self._set_status("Results submitted. Ready for next batch.")
        QMessageBox.information(
            self, "Results Submitted",
            "Results recorded successfully. Ready to ask the next batch."
        )

    def _append_results_to_csv(self, results: list) -> None:
        """
        Append actual compositions + measured objective values to the experiment CSV.

        Parameters
        ----------
        results : list of dicts, each containing
            ``actual_params`` (edited compositions),
            ``nominal_params`` (original suggestion),
            ``values``         (measured objective values).

        The CSV is updated in-place.  Failures are non-fatal — a warning is
        shown but the submission continues.
        """
        if not self._csv_path or not os.path.exists(self._csv_path):
            return   # no CSV to append to
        if not self._session_state:
            return
        cfg = self._session_state.study_config
        obj_cols = [o.column_name for o in cfg.objectives]

        try:
            rows = []
            for r in results:
                # Prefer actual (lab-measured) compositions; fall back to nominal
                params = r.get("actual_params") or r.get("nominal_params") or {}
                row: dict = dict(params)
                for i, v in enumerate(r.get("values", [])):
                    col = obj_cols[i] if i < len(obj_cols) else f"objective_{i}"
                    row[col] = v
                rows.append(row)

            if not rows:
                return

            df_existing = pd.read_csv(self._csv_path)
            df_new      = pd.DataFrame(rows)
            # Align columns — add missing columns as NaN
            for c in df_existing.columns:
                if c not in df_new.columns:
                    df_new[c] = float("nan")
            df_new = df_new[df_existing.columns]   # same column order
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(self._csv_path, index=False)
            # Refresh our in-memory copy
            self._df = df_combined
        except Exception as exc:
            QMessageBox.warning(
                self,
                "CSV Update Warning",
                f"Results were submitted to Optuna, but could not be appended "
                f"to the CSV:\n{exc}\n\nYou can add them manually if needed.",
            )

    def _refresh_design_space(
        self, suggestions: Optional[List[dict]] = None
    ) -> None:
        """
        Push fresh data to the Design Space dialog (if it is open).

        Parameters
        ----------
        suggestions : optional list of param-value dicts for the latest
                      suggested batch; if None the plot shows historical
                      data only (suggestions are cleared).
        """
        if self._df is None or not self._session_state:
            return
        if self._design_space_dlg is None:
            return  # dialog not opened yet — nothing to update
        cfg = self._session_state.study_config
        self._design_space_dlg.refresh(
            df=self._df,
            params=cfg.parameters,
            objectives=cfg.objectives,
            suggestions=suggestions,
            study_config=cfg,
        )

    def _action_open_design_space(self) -> None:
        """
        Open (or raise) the Design Space visualisation window.

        Creates the dialog the first time; thereafter reuses the same
        window so the user can keep it open alongside LabOpt.
        """
        if self._df is None or not self._session_state:
            QMessageBox.information(
                self,
                "No Data",
                "Load a CSV and apply objectives first, then open the Design Space.",
            )
            return

        # Create once, reuse thereafter
        if self._design_space_dlg is None:
            self._design_space_dlg = DesignSpaceDialog(parent=self)

        cfg = self._session_state.study_config
        self._design_space_dlg.refresh(
            df=self._df,
            params=cfg.parameters,
            objectives=cfg.objectives,
            suggestions=None,   # suggestions already set by _refresh_design_space if pending
            study_config=cfg,
        )
        # Push any validation results from the Results tab into the dialog
        if self._validation_results is not None:
            try:
                self._design_space_dlg.set_validation_results(
                    self._validation_results
                )
            except Exception:
                pass
        self._design_space_dlg.show()
        self._design_space_dlg.raise_()
        self._design_space_dlg.activateWindow()

    def _action_about(self) -> None:
        QMessageBox.information(
            self,
            "About LabOpt",
            "LabOpt — Lab Optimisation Tool\n\n"
            "Built with Optuna + PySide6\n\n"
            "Supports:\n"
            "  • CSV-seeded historical trials\n"
            "  • Dead regions (excluded parameter zones)\n"
            "  • Multi-objective optimisation (Pareto front)\n"
            "  • Batched experiment suggestions\n"
            "  • Full session persistence (close & resume)\n"
            "  • TPE / NSGAII / Random / GP samplers",
        )

    # ══════════════════════════════════════════════════════════════════════
    # Theme management
    # ══════════════════════════════════════════════════════════════════════

    def _apply_theme(self, dark: bool) -> None:
        """Apply dark or light theme to the whole application and update themed widgets."""
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(DARK_STYLESHEET if dark else "")

        # Warning label (theme-specific amber style)
        self._warning_label.setStyleSheet(
            _WARN_STYLE_DARK if dark else _WARN_STYLE_LIGHT
        )
        # Resume banner
        self._resume_banner.setStyleSheet(
            _BANNER_STYLE_DARK if dark else _BANNER_STYLE_LIGHT
        )

        # Persist choice
        QSettings().setValue("dark_mode", dark)

    def _action_toggle_theme(self) -> None:
        """Called when the user clicks View → Dark Mode (checkable action)."""
        self._apply_theme(self._dark_mode_action.isChecked())

    # ══════════════════════════════════════════════════════════════════════
    # In-app documentation viewer
    # ══════════════════════════════════════════════════════════════════════

    def _action_show_how_to_use(self) -> None:
        """Help → How to Use… — shows §3 of DOCUMENTATION.md in a QTextBrowser."""
        doc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DOCUMENTATION.md")
        if not os.path.exists(doc_path):
            QMessageBox.warning(
                self, "Documentation Not Found",
                f"Could not find DOCUMENTATION.md at:\n{doc_path}"
            )
            return

        with open(doc_path, "r", encoding="utf-8") as fh:
            content = fh.read()

        # Extract §3 (How to Use) through end of §3 (before §4)
        start = content.find("## 3. How to Use")
        end   = content.find("\n## 4.", start) if start != -1 else -1
        if start == -1:
            section_text = content          # fallback: show everything
        elif end == -1:
            section_text = content[start:]
        else:
            section_text = content[start:end]

        html = _markdown_to_html(section_text)

        dlg = QDialog(self)
        dlg.setWindowTitle("How to Use LabOpt")
        dlg.resize(820, 640)

        vbox = QVBoxLayout(dlg)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(6)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setHtml(html)
        vbox.addWidget(browser, stretch=1)

        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(100)
        close_btn.clicked.connect(dlg.accept)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        vbox.addLayout(btn_row)

        dlg.exec()

    # ══════════════════════════════════════════════════════════════════════
    # Close event — save session before exit
    # ══════════════════════════════════════════════════════════════════════

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._session_state:
            if self._session_state.pending_batch:
                n = len(self._session_state.pending_batch)
                reply = QMessageBox.question(
                    self,
                    "Pending Trials",
                    f"You have {n} suggested trial(s) awaiting lab results.\n\n"
                    "The session will be saved automatically.\n"
                    "When your experiments are done, reopen the app, reload\n"
                    "this session, then click  📋 Enter Pending Results…  in\n"
                    "the toolbar to submit your measurements.\n\n"
                    "Close anyway?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    event.ignore()
                    return
            # Sync UI state into config then save
            self._sync_config_from_ui()
            SessionManager.save(self._session_state)

        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)

        event.accept()
