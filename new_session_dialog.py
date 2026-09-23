"""
New Session dialog — two-step wizard for creating a LabOpt session.

Step 1  Project Identity:  project name (required), researcher, description,
                            session folder.
Step 2  Data Source:       load an existing CSV  or  start from scratch.

Usage
-----
    dlg = NewSessionDialog(parent=self)
    if dlg.exec() == QDialog.Accepted:
        project_name = dlg.project_name
        owner        = dlg.owner
        description  = dlg.description
        session_dir  = dlg.session_dir
        csv_path     = dlg.csv_path          # "" if scratch
        scratch      = dlg.start_from_scratch
"""
from __future__ import annotations

import os

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from session_manager import get_default_owner, get_default_session_dir


class NewSessionDialog(QDialog):
    """
    Two-step wizard that collects project identity and data source for a new
    LabOpt session.

    After ``exec() == QDialog.Accepted`` the following attributes are set:
        project_name      : str   — human-readable project display name
        owner             : str   — researcher / username
        description       : str   — optional longer description
        session_dir       : str   — root directory for the new session
        csv_path          : str   — path to the data CSV, or "" if scratch
        start_from_scratch: bool  — True if the user chose "no prior data"
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New LabOpt Session")
        self.setMinimumWidth(520)

        # Result attributes (populated on accept)
        self.project_name: str = ""
        self.owner: str = ""
        self.description: str = ""
        self.session_dir: str = get_default_session_dir()
        self.csv_path: str = ""
        self.start_from_scratch: bool = False

        self._build_ui()

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12)

        # Title
        title = QLabel("<h3>New LabOpt Session</h3>")
        title.setTextFormat(Qt.RichText)
        main_layout.addWidget(title)

        # Stacked widget: page 0 = Step 1, page 1 = Step 2
        self._stack = QStackedWidget()
        main_layout.addWidget(self._stack)

        self._stack.addWidget(self._build_step1())
        self._stack.addWidget(self._build_step2())
        self._stack.setCurrentIndex(0)

        # Navigation buttons
        nav = QHBoxLayout()
        self._back_btn = QPushButton("← Back")
        self._back_btn.setEnabled(False)
        self._back_btn.clicked.connect(self._go_back)
        nav.addWidget(self._back_btn)
        nav.addStretch()

        self._next_btn = QPushButton("Next →")
        self._next_btn.setDefault(True)
        self._next_btn.clicked.connect(self._go_next)
        nav.addWidget(self._next_btn)

        self._create_btn = QPushButton("Create Session")
        self._create_btn.setVisible(False)
        self._create_btn.clicked.connect(self._try_accept)
        nav.addWidget(self._create_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        nav.addWidget(cancel_btn)

        main_layout.addLayout(nav)

        # Step indicator
        self._step_lbl = QLabel("Step 1 of 2 — Project Identity")
        self._step_lbl.setStyleSheet("color: #89b4fa; font-size: 11px;")
        main_layout.addWidget(self._step_lbl)

    def _build_step1(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(6)

        form = QFormLayout()
        form.setSpacing(8)

        # Project Name (required)
        self._name_le = QLineEdit()
        self._name_le.setPlaceholderText("e.g. ProcessX Optimisation")
        self._name_le.setMaxLength(60)
        form.addRow("Project Name *:", self._name_le)

        name_hint = QLabel(
            "<i>Required.  Used as the directory name and session title.</i>"
        )
        name_hint.setTextFormat(Qt.RichText)
        name_hint.setStyleSheet("color: #6c6f85; font-size: 10px;")
        form.addRow("", name_hint)

        # Researcher
        self._owner_le = QLineEdit()
        self._owner_le.setText(get_default_owner())
        self._owner_le.setPlaceholderText("e.g. Alice Smith")
        self._owner_le.setMaxLength(60)
        form.addRow("Researcher:", self._owner_le)

        owner_hint = QLabel(
            "<i>Pre-filled from LABOPT_USER env var if set.  "
            "Used to isolate sessions on disk per user.</i>"
        )
        owner_hint.setTextFormat(Qt.RichText)
        owner_hint.setStyleSheet("color: #6c6f85; font-size: 10px;")
        form.addRow("", owner_hint)

        # Description
        self._desc_te = QTextEdit()
        self._desc_te.setPlaceholderText(
            "Optional — goal, materials, conditions…"
        )
        self._desc_te.setFixedHeight(64)
        form.addRow("Description:", self._desc_te)

        layout.addLayout(form)

        # Session folder group
        folder_grp = QGroupBox("Session Folder")
        folder_layout = QHBoxLayout(folder_grp)
        self._folder_le = QLineEdit()
        self._folder_le.setText(get_default_session_dir())
        self._folder_le.setReadOnly(False)
        folder_layout.addWidget(self._folder_le, stretch=1)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_folder)
        folder_layout.addWidget(browse_btn)
        layout.addWidget(folder_grp)

        env_note = QLabel(
            "<i>Set LABOPT_SESSION_DIR environment variable to pre-fill this "
            "for all users in a Docker deployment.</i>"
        )
        env_note.setTextFormat(Qt.RichText)
        env_note.setWordWrap(True)
        env_note.setStyleSheet("color: #6c6f85; font-size: 10px;")
        layout.addWidget(env_note)
        layout.addStretch()

        return page

    def _build_step2(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(10)

        instructions = QLabel(
            "Choose how to start this session:"
        )
        layout.addWidget(instructions)

        # Option A — CSV
        self._rb_csv = QRadioButton(
            "Load existing data CSV  (recommended if you have prior results)"
        )
        self._rb_csv.setChecked(True)
        self._rb_csv.toggled.connect(self._on_source_toggled)
        layout.addWidget(self._rb_csv)

        csv_row = QHBoxLayout()
        csv_row.setContentsMargins(20, 0, 0, 0)
        self._csv_le = QLineEdit()
        self._csv_le.setPlaceholderText("No file selected")
        self._csv_le.setReadOnly(True)
        csv_row.addWidget(self._csv_le, stretch=1)
        self._csv_browse_btn = QPushButton("Browse…")
        self._csv_browse_btn.clicked.connect(self._browse_csv)
        csv_row.addWidget(self._csv_browse_btn)
        layout.addLayout(csv_row)

        # Option B — scratch
        self._rb_scratch = QRadioButton(
            "Start from scratch  (no prior data — use the 🧪 DoE tab to plan experiments)"
        )
        self._rb_scratch.toggled.connect(self._on_source_toggled)
        layout.addWidget(self._rb_scratch)

        scratch_note = QLabel(
            "<i>You will define parameters manually, then use the DoE generator\n"
            "to create your first batch of experiments.</i>"
        )
        scratch_note.setTextFormat(Qt.RichText)
        scratch_note.setStyleSheet("color: #6c6f85; font-size: 10px;")
        scratch_note.setContentsMargins(20, 0, 0, 0)
        layout.addWidget(scratch_note)
        layout.addStretch()

        return page

    # ── Navigation ──────────────────────────────────────────────────────────

    def _go_next(self) -> None:
        # Validate Step 1
        if not self._name_le.text().strip():
            QMessageBox.warning(
                self, "Project Name Required",
                "Please enter a project name before continuing."
            )
            self._name_le.setFocus()
            return
        self._stack.setCurrentIndex(1)
        self._back_btn.setEnabled(True)
        self._next_btn.setVisible(False)
        self._create_btn.setVisible(True)
        self._step_lbl.setText("Step 2 of 2 — Data Source")

    def _go_back(self) -> None:
        self._stack.setCurrentIndex(0)
        self._back_btn.setEnabled(False)
        self._next_btn.setVisible(True)
        self._create_btn.setVisible(False)
        self._step_lbl.setText("Step 1 of 2 — Project Identity")

    def _try_accept(self) -> None:
        # Validate Step 2
        if self._rb_csv.isChecked() and not self._csv_le.text().strip():
            QMessageBox.warning(
                self, "No CSV Selected",
                "Please browse for a CSV file, or choose 'Start from scratch'."
            )
            return

        # Collect results
        self.project_name = self._name_le.text().strip()
        self.owner = self._owner_le.text().strip()
        self.description = self._desc_te.toPlainText().strip()
        self.session_dir = self._folder_le.text().strip() or get_default_session_dir()
        self.start_from_scratch = self._rb_scratch.isChecked()
        self.csv_path = "" if self.start_from_scratch else self._csv_le.text().strip()

        self.accept()

    # ── Helpers ────────────────────────────────────────────────────────────

    def _browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Session Folder",
            self._folder_le.text() or get_default_session_dir(),
        )
        if folder:
            self._folder_le.setText(folder)

    def _browse_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Experiment Data File",
            "",
            "Data files (*.csv *.xlsx *.xls);;"
            "CSV files (*.csv);;"
            "Excel files (*.xlsx *.xls)",
        )
        if path:
            self._csv_le.setText(path)

    def _on_source_toggled(self) -> None:
        csv_mode = self._rb_csv.isChecked()
        self._csv_le.setEnabled(csv_mode)
        self._csv_browse_btn.setEnabled(csv_mode)
