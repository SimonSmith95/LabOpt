"""
Objective Boundary Dialog
=========================
Lets the user define, edit and delete objective boundary conditions.

Each boundary is a rule over MEASURED OUTPUT columns (not parameter columns):

    expression   — arithmetic formula over measured column names,
                   e.g. "purity"  or  "impurity_A + impurity_B"
    operator     — one of  <=  >=
    target       — numeric right-hand-side threshold

A trial is considered INFEASIBLE if any boundary is violated.
The sampler (TPE / NSGAII) learns to avoid parameter regions that historically
produced infeasible results.  All data is always kept — no trials are deleted.

Unlike ParameterConstraint, there is no equality operator and no residual
parameter concept: boundaries are pure pass/fail gates on measured outputs.
"""
from __future__ import annotations

import ast
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from parameter_config import ObjectiveBoundary


# ── Catppuccin Mocha colours (matches the rest of the app) ────────────────────
_BG    = "#1e1e2e"
_FG    = "#cdd6f4"
_GRID  = "#313244"
_GREEN = "#a6e3a1"
_RED   = "#f38ba8"
_AMBER = "#f9e2af"
_BLUE  = "#89b4fa"


def _extract_column_names(expression: str) -> List[str]:
    """Return all variable names referenced in an arithmetic expression."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        return list(dict.fromkeys(
            n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name)
        ))
    except SyntaxError:
        return []


# ══════════════════════════════════════════════════════════════════════════════
# Single-boundary edit form (reused for Add and Edit)
# ══════════════════════════════════════════════════════════════════════════════

class _BoundaryEditDialog(QDialog):
    """
    Small modal dialog to create or edit a single :class:`ObjectiveBoundary`.
    """

    def __init__(
        self,
        column_names: List[str],
        boundary: Optional[ObjectiveBoundary] = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._column_names = column_names
        self.setWindowTitle("Edit Boundary" if boundary else "Add Objective Boundary")
        self.setMinimumWidth(480)
        self._result: Optional[ObjectiveBoundary] = None
        self._build_ui(boundary)

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self, b: Optional[ObjectiveBoundary]) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setSpacing(8)

        # Name
        self._name_edit = QLineEdit(b.name if b else "")
        self._name_edit.setPlaceholderText('e.g. "Purity specification"')
        form.addRow("Name:", self._name_edit)

        # Expression
        self._expr_edit = QLineEdit(b.expression if b else "")
        self._expr_edit.setPlaceholderText(
            "e.g.  purity   or   impurity_A + impurity_B"
        )
        form.addRow("Expression:", self._expr_edit)

        # Operator (only <= and >= — equalities don't make sense for boundaries)
        self._op_combo = QComboBox()
        self._op_combo.addItems(["<=", ">="])
        if b:
            self._op_combo.setCurrentText(b.operator)
        form.addRow("Operator:", self._op_combo)

        # Target
        self._target_spin = QDoubleSpinBox()
        self._target_spin.setRange(-1e12, 1e12)
        self._target_spin.setDecimals(8)
        self._target_spin.setSingleStep(0.001)
        self._target_spin.setValue(b.target if b else 0.0)
        form.addRow("Target value:", self._target_spin)

        layout.addLayout(form)

        # Help text
        help_lbl = QLabel(
            "Trials where this expression violates the limit are marked "
            "<b>infeasible</b>.  The surrogate (TPE / NSGAII) learns to steer "
            "future suggestions away from parameter regions that historically "
            "caused violations.  <i>All data is always kept — no trials are "
            "deleted.</i>"
        )
        help_lbl.setWordWrap(True)
        help_lbl.setStyleSheet(f"color: {_AMBER}; font-size: 11px;")
        layout.addWidget(help_lbl)

        # Known columns hint
        col_hint = QLabel(
            "Known measured columns: " + (", ".join(self._column_names) or "(none yet)")
        )
        col_hint.setWordWrap(True)
        col_hint.setStyleSheet(f"color: {_FG}; font-size: 10px; font-style: italic;")
        layout.addWidget(col_hint)

        # Tip about scientific notation
        tip_lbl = QLabel(
            " Scientific notation like <b>4.979e-2</b> also works."
        )
        tip_lbl.setWordWrap(True)
        tip_lbl.setStyleSheet(f"color: {_BLUE}; font-size: 10px;")
        layout.addWidget(tip_lbl)

        # Buttons
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    # ── Slot ────────────────────────────────────────────────────────────────

    def _on_accept(self) -> None:
        name = self._name_edit.text().strip()
        expr = self._expr_edit.text().strip()
        op   = self._op_combo.currentText()
        tgt  = self._target_spin.value()

        if not name:
            QMessageBox.warning(self, "Validation", "Please enter a boundary name.")
            return

        b = ObjectiveBoundary(name=name, expression=expr, operator=op, target=tgt)
        errs = b.validate_expression(self._column_names)
        if errs:
            QMessageBox.warning(
                self, "Expression Error",
                "The boundary expression has the following issues:\n\n• "
                + "\n• ".join(errs),
            )
            return

        self._result = b
        self.accept()

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def result(self) -> Optional[ObjectiveBoundary]:
        return self._result


# ══════════════════════════════════════════════════════════════════════════════
# Main Objective Boundary Manager Dialog
# ══════════════════════════════════════════════════════════════════════════════

class ObjectiveBoundaryDialog(QDialog):
    """
    Modal dialog that lists all :class:`ObjectiveBoundary` objects and
    provides Add / Edit / Remove controls.

    Usage
    -----
    ::
        dlg = ObjectiveBoundaryDialog(column_names, existing_boundaries, parent=self)
        if dlg.exec():
            config.objective_boundaries = dlg.boundaries
    """

    def __init__(
        self,
        column_names: List[str],
        boundaries: Optional[List[ObjectiveBoundary]] = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._column_names = column_names
        self._boundaries: List[ObjectiveBoundary] = list(boundaries or [])

        self.setWindowTitle("🚧  Objective Boundaries")
        self.setMinimumWidth(700)
        self.setMinimumHeight(380)
        self._build_ui()
        self._refresh_table()

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Intro text
        intro = QLabel(
            "<b>Define hard feasibility limits on measured output columns.</b><br>"
            "These boundaries are NOT objectives to minimise — they are pass/fail "
            "gates.  The surrogate learns which parameter regions historically "
            "produce violations and steers away from them.  Each experiment still "
            "requires you to measure and enter the boundary variable value(s) "
            "in the Batch Results Dialog."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {_FG}; font-size: 12px;")
        layout.addWidget(intro)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {_GRID};")
        layout.addWidget(sep)

        # ── Boundary table ────────────────────────────────────────────────
        cols = ["Name", "Expression", "Op", "Target"]
        self._table = QTableWidget(0, len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.doubleClicked.connect(self._on_edit)
        layout.addWidget(self._table, stretch=1)

        # ── Referenced columns info ───────────────────────────────────────
        self._ref_label = QLabel("")
        self._ref_label.setWordWrap(True)
        self._ref_label.setStyleSheet(f"color: {_BLUE}; font-size: 11px;")
        layout.addWidget(self._ref_label)

        # ── Toolbar ───────────────────────────────────────────────────────
        btn_row = QHBoxLayout()

        add_btn = QPushButton("＋  Add Boundary")
        add_btn.clicked.connect(self._on_add)
        btn_row.addWidget(add_btn)

        edit_btn = QPushButton("✏  Edit Selected")
        edit_btn.clicked.connect(self._on_edit)
        btn_row.addWidget(edit_btn)

        del_btn = QPushButton("🗑  Remove Selected")
        del_btn.clicked.connect(self._on_delete)
        btn_row.addWidget(del_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        # ── Status / hint ─────────────────────────────────────────────────
        self._status = QLabel("")
        self._status.setStyleSheet(f"color: {_AMBER}; font-size: 11px;")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        # ── Dialog buttons ────────────────────────────────────────────────
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    # ── Table helpers ───────────────────────────────────────────────────────

    def _refresh_table(self) -> None:
        self._table.setRowCount(0)
        all_referenced: set[str] = set()

        for b in self._boundaries:
            row = self._table.rowCount()
            self._table.insertRow(row)
            items = [b.name, b.expression, b.operator, f"{b.target:g}"]
            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if col == 2:  # operator column
                    item.setForeground(QColor(_AMBER))
                self._table.setItem(row, col, item)

            # Collect referenced column names for the info label
            for col_name in _extract_column_names(b.expression):
                all_referenced.add(col_name)

        # Update the referenced-columns info label
        if all_referenced:
            self._ref_label.setText(
                "📊  Measured columns you must enter per experiment: "
                + ", ".join(sorted(all_referenced))
            )
        else:
            self._ref_label.setText("")

        self._update_status()

    def _selected_row(self) -> int:
        rows = self._table.selectionModel().selectedRows()
        return rows[0].row() if rows else -1

    def _update_status(self) -> None:
        issues: List[str] = []
        for b in self._boundaries:
            errs = b.validate_expression(self._column_names)
            if errs:
                issues.append(f"• {b.name}: " + "; ".join(errs))
        if issues:
            self._status.setText("⚠ Issues detected:\n" + "\n".join(issues))
        else:
            self._status.setText(
                f"{len(self._boundaries)} boundary condition(s) defined."
                if self._boundaries
                else "No boundaries defined — all measured values are accepted."
            )

    # ── Slot handlers ───────────────────────────────────────────────────────

    def _on_add(self) -> None:
        dlg = _BoundaryEditDialog(self._column_names, parent=self)
        if dlg.exec() and dlg.result:
            self._boundaries.append(dlg.result)
            self._refresh_table()

    def _on_edit(self) -> None:
        row = self._selected_row()
        if row < 0:
            QMessageBox.information(self, "Edit", "Select a boundary row first.")
            return
        dlg = _BoundaryEditDialog(
            self._column_names, self._boundaries[row], parent=self
        )
        if dlg.exec() and dlg.result:
            self._boundaries[row] = dlg.result
            self._refresh_table()

    def _on_delete(self) -> None:
        row = self._selected_row()
        if row < 0:
            return
        name = self._boundaries[row].name
        reply = QMessageBox.question(
            self, "Remove",
            f"Remove boundary '{name}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            del self._boundaries[row]
            self._refresh_table()

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def boundaries(self) -> List[ObjectiveBoundary]:
        """The current list of boundaries (after the dialog is accepted)."""
        return list(self._boundaries)
