"""
Study Definition Dialog
=======================
Lets users define a LabOpt study from scratch — without needing a pre-existing CSV.

The dialog has three sections:
  • Parameters   — name, type (Float/Int/Categorical), min, max (or choices)
  • Objectives   — name, direction (minimize/maximize)
  • Context vars — name (uncontrollable environmental conditions)

On accept it returns an empty pandas DataFrame whose columns match the
definitions and a StudyConfig that can be fed directly into the app.

Usage
-----
    dlg = StudyDefinitionDialog(parent=self)
    if dlg.exec() == QDialog.Accepted:
        df        = dlg.empty_dataframe    # headers only, 0 rows
        cfg       = dlg.study_config       # StudyConfig with parameters/objectives/ctx
"""
from __future__ import annotations

import pandas as pd
from typing import List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from parameter_config import (
    ContextConfig,
    ObjectiveConfig,
    ParameterConfig,
    ParameterType,
    StudyConfig,
)


class StudyDefinitionDialog(QDialog):
    """
    Dialog for defining a study's parameters, objectives and context variables
    without loading an existing CSV file.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Define Study from Scratch")
        self.setMinimumSize(700, 560)
        self.resize(800, 640)

        self.empty_dataframe: pd.DataFrame = pd.DataFrame()
        self.study_config: StudyConfig = StudyConfig()

        self._build_ui()

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        vbox = QVBoxLayout(self)
        vbox.setSpacing(10)

        help_lbl = QLabel(
            "Define your experiment parameters, objectives, and context variables below.\n"
            "After clicking OK, LabOpt will create a session using these definitions.\n"
            "Use the 🧪 DoE tab to generate experiment points, or load an existing CSV later."
        )
        help_lbl.setWordWrap(True)
        help_lbl.setStyleSheet("color: #a6e3a1; font-size: 11px; padding: 4px;")
        vbox.addWidget(help_lbl)

        # ── Parameters ─────────────────────────────────────────────────────
        param_grp = QGroupBox("Parameters  (input variables the DoE will vary)")
        pg = QVBoxLayout(param_grp)

        self._param_table = QTableWidget(0, 5)
        self._param_table.setHorizontalHeaderLabels(
            ["Name", "Type", "Min / Choices", "Max", "Enabled"]
        )
        self._param_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._param_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self._param_table.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
        )
        self._param_table.setMinimumHeight(140)
        pg.addWidget(self._param_table)

        pb = QHBoxLayout()
        add_p = QPushButton("➕ Add Parameter")
        add_p.clicked.connect(self._add_param_row)
        del_p = QPushButton("🗑 Remove Selected")
        del_p.clicked.connect(lambda: self._remove_row(self._param_table))
        pb.addWidget(add_p)
        pb.addWidget(del_p)
        pb.addStretch()
        pg.addLayout(pb)
        pg.addWidget(QLabel(
            "Type: Float or Int → enter Min and Max numbers.  "
            "Categorical → enter comma-separated choices in 'Min / Choices'.",
            styleSheet="color: #6c6f85; font-size: 10px;"
        ))
        vbox.addWidget(param_grp)

        # ── Objectives ─────────────────────────────────────────────────────
        obj_grp = QGroupBox("Objectives  (measured result columns)")
        og = QVBoxLayout(obj_grp)

        self._obj_table = QTableWidget(0, 2)
        self._obj_table.setHorizontalHeaderLabels(["Column Name", "Direction"])
        self._obj_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._obj_table.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
        )
        self._obj_table.setMinimumHeight(80)
        og.addWidget(self._obj_table)

        ob = QHBoxLayout()
        add_o = QPushButton("➕ Add Objective")
        add_o.clicked.connect(self._add_obj_row)
        del_o = QPushButton("🗑 Remove Selected")
        del_o.clicked.connect(lambda: self._remove_row(self._obj_table))
        ob.addWidget(add_o)
        ob.addWidget(del_o)
        ob.addStretch()
        og.addLayout(ob)
        vbox.addWidget(obj_grp)

        # ── Context Variables ──────────────────────────────────────────────
        ctx_grp = QGroupBox(
            "Context Variables  (measured environmental conditions — optional)"
        )
        cg = QVBoxLayout(ctx_grp)

        self._ctx_table = QTableWidget(0, 1)
        self._ctx_table.setHorizontalHeaderLabels(["Variable Name"])
        self._ctx_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._ctx_table.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
        )
        self._ctx_table.setMinimumHeight(70)
        cg.addWidget(self._ctx_table)

        cb = QHBoxLayout()
        add_c = QPushButton("➕ Add Context Variable")
        add_c.clicked.connect(self._add_ctx_row)
        del_c = QPushButton("🗑 Remove Selected")
        del_c.clicked.connect(lambda: self._remove_row(self._ctx_table))
        cb.addWidget(add_c)
        cb.addWidget(del_c)
        cb.addStretch()
        cg.addLayout(cb)
        vbox.addWidget(ctx_grp)

        # ── Buttons ────────────────────────────────────────────────────────
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        vbox.addWidget(btns)

        # Pre-populate with one example row each
        self._add_param_row()
        self._add_obj_row()

    # ── Row management ─────────────────────────────────────────────────────

    def _add_param_row(self) -> None:
        r = self._param_table.rowCount()
        self._param_table.insertRow(r)
        self._param_table.setItem(r, 0, QTableWidgetItem(f"param_{r + 1}"))

        type_combo = QComboBox()
        type_combo.addItems(["Float", "Int", "Categorical"])
        self._param_table.setCellWidget(r, 1, type_combo)

        self._param_table.setItem(r, 2, QTableWidgetItem("0.0"))
        self._param_table.setItem(r, 3, QTableWidgetItem("1.0"))
        self._param_table.setItem(r, 4, QTableWidgetItem("Yes"))

    def _add_obj_row(self) -> None:
        r = self._obj_table.rowCount()
        self._obj_table.insertRow(r)
        self._obj_table.setItem(r, 0, QTableWidgetItem(f"objective_{r + 1}"))

        dir_combo = QComboBox()
        dir_combo.addItems(["minimize", "maximize"])
        self._obj_table.setCellWidget(r, 1, dir_combo)

    def _add_ctx_row(self) -> None:
        r = self._ctx_table.rowCount()
        self._ctx_table.insertRow(r)
        self._ctx_table.setItem(r, 0, QTableWidgetItem(f"context_{r + 1}"))

    def _remove_row(self, table: QTableWidget) -> None:
        rows = sorted(
            set(idx.row() for idx in table.selectedIndexes()), reverse=True
        )
        for r in rows:
            table.removeRow(r)

    # ── Validation & accept ────────────────────────────────────────────────

    def _on_accept(self) -> None:
        errors: List[str] = []

        # ── Parse parameters ───────────────────────────────────────────────
        params: List[ParameterConfig] = []
        param_names: List[str] = []
        for r in range(self._param_table.rowCount()):
            name_item = self._param_table.item(r, 0)
            name = name_item.text().strip() if name_item else ""
            if not name:
                errors.append(f"Parameter row {r + 1}: name is empty.")
                continue
            if name in param_names:
                errors.append(f"Duplicate parameter name: '{name}'.")
                continue
            param_names.append(name)

            type_combo = self._param_table.cellWidget(r, 1)
            type_str = type_combo.currentText() if type_combo else "Float"

            min_item = self._param_table.item(r, 2)
            max_item = self._param_table.item(r, 3)
            min_txt  = min_item.text().strip() if min_item else "0.0"
            max_txt  = max_item.text().strip() if max_item else "1.0"

            if type_str == "Categorical":
                choices = [c.strip() for c in min_txt.split(",") if c.strip()]
                if not choices:
                    errors.append(f"Parameter '{name}': no choices provided.")
                    continue
                p = ParameterConfig(
                    name=name, ptype=ParameterType.CATEGORICAL,
                    all_choices=choices, allowed_choices=choices,
                )
            else:
                try:
                    lo = float(min_txt)
                    hi = float(max_txt)
                except ValueError:
                    errors.append(f"Parameter '{name}': min/max must be numeric.")
                    continue
                if lo >= hi:
                    errors.append(f"Parameter '{name}': min must be < max.")
                    continue
                ptype = ParameterType.FLOAT if type_str == "Float" else ParameterType.INT
                p = ParameterConfig(name=name, ptype=ptype, full_min=lo, full_max=hi)

            params.append(p)

        # ── Parse objectives ───────────────────────────────────────────────
        objectives: List[ObjectiveConfig] = []
        obj_names: List[str] = []
        for r in range(self._obj_table.rowCount()):
            name_item = self._obj_table.item(r, 0)
            name = name_item.text().strip() if name_item else ""
            if not name:
                errors.append(f"Objective row {r + 1}: name is empty.")
                continue
            if name in obj_names:
                errors.append(f"Duplicate objective name: '{name}'.")
                continue
            obj_names.append(name)

            dir_combo = self._obj_table.cellWidget(r, 1)
            direction = dir_combo.currentText() if dir_combo else "minimize"
            objectives.append(ObjectiveConfig(column_name=name, direction=direction))

        # ── Parse context variables ────────────────────────────────────────
        ctx_vars: List[ContextConfig] = []
        ctx_names: List[str] = []
        for r in range(self._ctx_table.rowCount()):
            name_item = self._ctx_table.item(r, 0)
            name = name_item.text().strip() if name_item else ""
            if not name:
                continue   # silently skip empty context rows
            if name in ctx_names:
                errors.append(f"Duplicate context variable name: '{name}'.")
                continue
            ctx_names.append(name)
            ctx_vars.append(ContextConfig(column_name=name))

        # ── Minimum requirements ───────────────────────────────────────────
        if not params:
            errors.append("Define at least one parameter.")
        if not objectives:
            errors.append("Define at least one objective.")

        if errors:
            QMessageBox.warning(
                self, "Validation Errors",
                "Please fix the following issues:\n\n" + "\n".join(f"• {e}" for e in errors)
            )
            return

        # ── Build output ───────────────────────────────────────────────────
        # Empty DataFrame with all column headers — 0 rows, all columns present
        all_columns = (
            [p.name for p in params]
            + [c.column_name for c in ctx_vars]
            + [o.column_name for o in objectives]
        )
        self.empty_dataframe = pd.DataFrame(columns=all_columns)

        self.study_config = StudyConfig(
            parameters=params,
            objectives=objectives,
            context_variables=ctx_vars,
        )

        self.accept()
