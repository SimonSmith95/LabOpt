"""
Session Browser dialog — Browse All Sessions…
Accessible via  File → Browse All Sessions…

Lists every session found under the configured session root directory,
with columns for project name, owner, date, trial count, and status.
Double-click a row (or click Open) to load that session.
"""
from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from session_manager import SessionManager, get_default_session_dir


class SessionBrowserDialog(QDialog):
    """
    Modal dialog showing all sessions found under the session root directory.

    Parameters
    ----------
    parent
        Parent widget.
    open_callback
        Callable that receives a session_path string when the user requests
        to open a session.  Typically ``main_window._load_session_from_path``.
    """

    def __init__(
        self,
        parent=None,
        open_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Browse All Sessions")
        self.setMinimumSize(860, 480)
        self.resize(980, 540)
        self._open_callback = open_callback
        self._sessions: list = []
        self._build_ui()
        self._refresh()

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ── Header label ──────────────────────────────────────────────────
        root = get_default_session_dir()
        header = QLabel(f"<b>Session root:</b> <code>{root}</code>")
        header.setTextFormat(Qt.RichText)
        header.setStyleSheet("color: #a6e3a1; font-size: 11px;")
        layout.addWidget(header)

        # ── Filter row ────────────────────────────────────────────────────
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter:"))
        self._filter_le = QLineEdit()
        self._filter_le.setPlaceholderText("Search by project name or owner…")
        self._filter_le.textChanged.connect(self._apply_filter)
        filter_row.addWidget(self._filter_le, stretch=1)
        refresh_btn = QPushButton("↻  Refresh")
        refresh_btn.setFixedWidth(90)
        refresh_btn.clicked.connect(self._refresh)
        filter_row.addWidget(refresh_btn)
        layout.addLayout(filter_row)

        # ── Table ─────────────────────────────────────────────────────────
        self._table = QTableWidget()
        self._table.setColumnCount(7)
        self._table.setHorizontalHeaderLabels([
            "Project Name", "Owner", "Created", "Trials", "Status", "DoE", "Path",
        ])
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self._table.horizontalHeader().setStretchLastSection(False)
        self._table.doubleClicked.connect(self._open_selected)
        layout.addWidget(self._table)

        # ── Button row ────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._open_btn = QPushButton("Open Session")
        self._open_btn.setEnabled(False)
        self._open_btn.clicked.connect(self._open_selected)
        btn_row.addWidget(self._open_btn)
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        # Enable Open button only when a row is selected
        self._table.selectionModel().selectionChanged.connect(
            lambda: self._open_btn.setEnabled(
                len(self._table.selectedItems()) > 0
            )
        )

    # ── Data loading ───────────────────────────────────────────────────────

    def _refresh(self) -> None:
        self._sessions = SessionManager.list_all_sessions()
        self._apply_filter(self._filter_le.text())

    def _apply_filter(self, text: str) -> None:
        lc = text.lower()
        if lc:
            filtered = [
                s for s in self._sessions
                if lc in s.get("project_name", "").lower()
                or lc in s.get("owner", "").lower()
                or lc in s.get("study_name", "").lower()
            ]
        else:
            filtered = self._sessions

        self._table.setRowCount(len(filtered))
        for r, s in enumerate(filtered):
            project = s.get("project_name") or s.get("study_name", "")
            owner = s.get("owner", "")
            ts = s.get("timestamp", "")
            # Format timestamp YYYYMMDD_HHMMSS → YYYY-MM-DD HH:MM:SS
            if len(ts) == 15 and "_" in ts:
                ts = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}  {ts[9:11]}:{ts[11:13]}:{ts[13:]}"
            n_trials = s.get("n_trials", 0)
            doe_complete = s.get("doe_complete", False)
            has_pending = s.get("has_pending_batch", False)

            if has_pending:
                status = "⏳ Pending results"
            elif n_trials == 0:
                status = "○ New"
            elif doe_complete:
                status = "🔬 BO running"
            else:
                status = "● Active"

            doe_text = "✓" if doe_complete else ""

            self._table.setItem(r, 0, QTableWidgetItem(project))
            self._table.setItem(r, 1, QTableWidgetItem(owner))
            self._table.setItem(r, 2, QTableWidgetItem(ts))
            self._table.setItem(r, 3, QTableWidgetItem(str(n_trials)))
            self._table.setItem(r, 4, QTableWidgetItem(status))
            self._table.setItem(r, 5, QTableWidgetItem(doe_text))
            path_item = QTableWidgetItem(s.get("session_path", ""))
            self._table.setItem(r, 6, path_item)

            # Store the full session path on the project-name cell for retrieval
            self._table.item(r, 0).setData(
                Qt.UserRole, s.get("session_path", "")
            )

        self._table.resizeColumnsToContents()
        # Re-apply stretch to column 0 after resize
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )

    # ── Actions ────────────────────────────────────────────────────────────

    def _open_selected(self) -> None:
        row = self._table.currentRow()
        if row < 0:
            return
        item = self._table.item(row, 0)
        if item is None:
            return
        session_path: str = item.data(Qt.UserRole) or ""
        if session_path and self._open_callback:
            self._open_callback(session_path)
            self.accept()
