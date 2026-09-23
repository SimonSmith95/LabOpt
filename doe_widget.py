"""
DoE Widget  (Phase 4)
=====================
PySide6 QWidget that provides the full Design of Experiments UI.

Added as the "🧪 DoE" tab in the LabOpt left dock.

Public API
----------
DoEWidget(parent=None)
    readiness_changed  Signal(int, int)   — (n_done, target) on any change
    start_bo_requested Signal()           — user clicked "Start Optimisation"

refresh(session_state, study)
    Update the widget with the current session and Optuna study.
    Call after CSV load, Apply Objectives, or session resume.

update_n_done(n_completed)
    Refresh just the readiness bar and Start button.
    Call after every batch completion or DoE result registration.
"""
from __future__ import annotations

import os
from typing import List, Optional

import pandas as pd
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QProgressBar,
    QAbstractItemView,
    QHeaderView,

)

from doe_math import (
    ALL_STRATEGIES,
    ALWAYS_AVAILABLE,
    PYDOE2_REQUIRED,
    coverage_metrics,
    compute_readiness_target,
    generate_doe,
    pydoe2_available,
    sobol_next_power_of_2,
)
from parameter_config import DoEState, StudyConfig
from session_manager import SessionManager, SessionState


# ── Strategy metadata ─────────────────────────────────────────────────────────
_STRATEGY_TOOLTIPS = {
    "LHS": (
        "Latin Hypercube Sampling — best general-purpose choice.\n"
        "Guarantees one point per equal-width bin in each dimension.\n"
        "Suitable for any N and mixed continuous/categorical spaces."
    ),
    "Sobol": (
        "Sobol quasi-random sequence — best joint coverage for large N.\n"
        "N is rounded up to the nearest power of 2 (e.g. 10 → 16).\n"
        "Best for continuous spaces with N ≥ 32."
    ),
    "Halton": (
        "Halton quasi-random sequence — good coverage, any N.\n"
        "Slightly less uniform than Sobol at large N but more flexible.\n"
        "Good alternative when exact power-of-2 N is inconvenient."
    ),
    "FullFactorial": (
        "Full Factorial — all combinations of factor levels.\n"
        "N grows exponentially: levels^n_factors.\n"
        "Best for ≤3 factors or categorical parameters with few choices."
    ),
    "PlackettBurman": (
        "Plackett-Burman screening design — 2-level, very efficient.\n"
        "N = next multiple of 4 ≥ k+1 (auto-computed).\n"
        "Best for quickly identifying important factors. Requires pyDOE2."
    ),
    "BoxBehnken": (
        "Box-Behnken response surface design — avoids corner points.\n"
        "Requires exactly 3–7 continuous factors. N auto-computed.\n"
        "Use when extreme factor combinations are physically impractical. Requires pyDOE2."
    ),
    "CCD": (
        "Central Composite Design — full factorial + axial (star) points.\n"
        "Requires 2–6 continuous factors. N auto-computed.\n"
        "Best for fitting a quadratic response surface model. Requires pyDOE2."
    ),
    "Random": (
        "Pure uniform random (baseline — same as Optuna's startup).\n"
        "No space-filling guarantee. Included for comparison only.\n"
        "Prefer LHS or Sobol for actual use."
    ),
}


class DoEWidget(QWidget):
    """
    Design of Experiments tab widget.

    Emits:
        readiness_changed(n_done: int, target: int)
        start_bo_requested()
    """

    readiness_changed  = Signal(int, int)
    start_bo_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._session_state: Optional[SessionState] = None
        self._study = None      # optuna.Study | None
        self._readiness_target: int = 10
        self._n_done: int = 0

        # Pending DoE points (generated but not yet registered)
        self._doe_df: Optional[pd.DataFrame] = None  # generated points
        self._obj_cols: List[str] = []               # objective column names

        self._build_ui()
        self._set_enabled(False)

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Scrollable content ────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        content = QWidget()
        vbox = QVBoxLayout(content)
        vbox.setSpacing(8)
        vbox.setContentsMargins(6, 6, 6, 6)

        # ── Surrogate Readiness Bar ───────────────────────────────────────
        readiness_grp = QGroupBox("Surrogate Readiness")
        rbox = QVBoxLayout(readiness_grp)
        self._readiness_bar = QProgressBar()
        self._readiness_bar.setRange(0, 100)
        self._readiness_bar.setValue(0)
        self._readiness_bar.setFixedHeight(18)
        rbox.addWidget(self._readiness_bar)
        self._readiness_lbl = QLabel("0 / — runs  (define parameters first)")
        self._readiness_lbl.setAlignment(Qt.AlignCenter)
        self._readiness_lbl.setStyleSheet("font-size: 11px;")
        rbox.addWidget(self._readiness_lbl)
        self._formula_lbl = QLabel("")
        self._formula_lbl.setStyleSheet("color: #6c6f85; font-size: 10px;")
        self._formula_lbl.setWordWrap(True)
        rbox.addWidget(self._formula_lbl)
        vbox.addWidget(readiness_grp)

        # ── Design Settings ───────────────────────────────────────────────
        settings_grp = QGroupBox("Design Settings")
        sf = QFormLayout(settings_grp)
        sf.setSpacing(6)

        self._strategy_combo = QComboBox()
        for s in ALL_STRATEGIES:
            self._strategy_combo.addItem(s)
            tip = _STRATEGY_TOOLTIPS.get(s, "")
            idx = self._strategy_combo.count() - 1
            self._strategy_combo.setItemData(idx, tip, Qt.ToolTipRole)
            if s in PYDOE2_REQUIRED and not pydoe2_available():
                item_model = self._strategy_combo.model()
                from PySide6.QtGui import QStandardItem
                item = item_model.item(idx)
                if item:
                    from PySide6.QtCore import Qt as _Qt
                    from PySide6.QtGui import QColor
                    item.setEnabled(False)
                    item.setForeground(QColor("#585b70"))
                    item.setToolTip(
                        tip + "\n\nInstall pyDOE2 to enable: pip install pyDOE2"
                    )
        self._strategy_combo.currentTextChanged.connect(self._on_strategy_changed)
        sf.addRow("Strategy:", self._strategy_combo)

        self._n_spin = QSpinBox()
        self._n_spin.setRange(1, 10_000)
        self._n_spin.setValue(20)
        self._n_spin.setToolTip("Number of DoE points to generate.")
        sf.addRow("N points:", self._n_spin)

        # Sobol info label (shown when Sobol is selected)
        self._sobol_lbl = QLabel("")
        self._sobol_lbl.setStyleSheet("color: #89b4fa; font-size: 10px;")
        self._sobol_lbl.hide()
        sf.addRow("", self._sobol_lbl)
        self._n_spin.valueChanged.connect(self._update_sobol_label)

        # Levels per factor (shown for FullFactorial only)
        self._levels_spin = QSpinBox()
        self._levels_spin.setRange(2, 20)
        self._levels_spin.setValue(3)
        self._levels_spin.setToolTip(
            "Number of equally-spaced levels per continuous / integer factor.\n"
            "N = levels^n_factors — grows exponentially!"
        )
        self._levels_row_label = QLabel("Levels/factor:")
        sf.addRow(self._levels_row_label, self._levels_spin)
        self._levels_spin.hide()
        self._levels_row_label.hide()

        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 99_999)
        self._seed_spin.setValue(42)
        self._seed_spin.setToolTip("Random seed — fix this for reproducible designs.")
        sf.addRow("Seed:", self._seed_spin)

        self._generate_btn = QPushButton("🎲  Generate DoE")
        self._generate_btn.setToolTip("Generate the DoE point set based on the settings above.")
        self._generate_btn.clicked.connect(self._action_generate)
        sf.addRow("", self._generate_btn)

        vbox.addWidget(settings_grp)

        # ── Coverage Metrics ──────────────────────────────────────────────
        self._coverage_grp = QGroupBox("Coverage Metrics")
        cbox = QFormLayout(self._coverage_grp)
        self._maximin_lbl = QLabel("—")
        cbox.addRow("Maximin distance:", self._maximin_lbl)
        self._discr_lbl = QLabel("—")
        cbox.addRow("L2 discrepancy:", self._discr_lbl)
        self._rand_ref_lbl = QLabel("—")
        cbox.addRow("vs. Random baseline:", self._rand_ref_lbl)
        self._pairplot_btn = QPushButton("📊  View Pairplot")
        self._pairplot_btn.setEnabled(False)
        self._pairplot_btn.clicked.connect(self._action_show_pairplot)
        cbox.addRow("", self._pairplot_btn)

        self._analysis_btn = QPushButton("📈  Full Analysis…")
        self._analysis_btn.setToolTip(
            "Open the full DoE analysis dialog with tabs for:\n"
            "  • Pairplot (coloured by objective if results exist)\n"
            "  • Marginal Uniformity\n"
            "  • Main Effects (if results entered)\n"
            "  • Parallel Coordinates (if results entered)"
        )
        self._analysis_btn.setEnabled(False)
        self._analysis_btn.clicked.connect(self._action_show_analysis)
        cbox.addRow("", self._analysis_btn)
        self._coverage_grp.hide()
        vbox.addWidget(self._coverage_grp)

        # ── DoE Points Table ──────────────────────────────────────────────
        self._points_grp = QGroupBox("DoE Points")
        pbox = QVBoxLayout(self._points_grp)

        self._table = QTableWidget()
        self._table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setAlternatingRowColors(True)
        self._table.setMinimumHeight(180)
        pbox.addWidget(self._table)

        btn_row = QHBoxLayout()
        self._import_btn = QPushButton("📥  Import Results CSV…")
        self._import_btn.setToolTip(
            "Import objective values from a CSV file that matches the DoE parameters.\n"
            "Rows are matched by parameter values."
        )
        self._import_btn.setEnabled(False)
        self._import_btn.clicked.connect(self._action_import_results)
        btn_row.addWidget(self._import_btn)

        self._export_pending_btn = QPushButton("📤  Export Pending CSV…")
        self._export_pending_btn.setToolTip(
            "Export the DoE parameter combinations (without results) as a CSV\n"
            "to take to the lab."
        )
        self._export_pending_btn.setEnabled(False)
        self._export_pending_btn.clicked.connect(self._action_export_pending)
        btn_row.addWidget(self._export_pending_btn)

        self._export_results_csv_btn = QPushButton("📊  Export Results CSV…")
        self._export_results_csv_btn.setToolTip(
            "Export the full DoE table (parameters + entered objective values)\n"
            "as a CSV file."
        )
        self._export_results_csv_btn.setEnabled(False)
        self._export_results_csv_btn.clicked.connect(self._action_export_results_csv)
        btn_row.addWidget(self._export_results_csv_btn)
        pbox.addLayout(btn_row)

        self._register_btn = QPushButton("✓  Register Results →")
        self._register_btn.setToolTip(
            "Register all rows with complete objective values into the Optuna study.\n"
            "This makes them available to the surrogate model."
        )
        self._register_btn.setEnabled(False)
        self._register_btn.clicked.connect(self._action_register_results)
        pbox.addWidget(self._register_btn)

        self._progress_lbl = QLabel("")
        self._progress_lbl.setStyleSheet("color: #a6e3a1; font-size: 10px;")
        pbox.addWidget(self._progress_lbl)

        self._export_report_btn = QPushButton("📄  Export DoE Report…")
        self._export_report_btn.setToolTip(
            "Generate a self-contained HTML report of the DoE phase:\n"
            "  • Session metadata\n"
            "  • Coverage metrics\n"
            "  • Pairplot (embedded)\n"
            "  • Full DoE table with results\n"
            "  • Readiness status"
        )
        self._export_report_btn.setEnabled(False)
        self._export_report_btn.clicked.connect(self._action_export_doe_report)
        pbox.addWidget(self._export_report_btn)

        self._points_grp.hide()
        vbox.addWidget(self._points_grp)

        vbox.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll, stretch=1)

        # ── Start Optimisation button (always visible) ─────────────────────
        btn_container = QWidget()
        bvbox = QVBoxLayout(btn_container)
        bvbox.setContentsMargins(6, 4, 6, 6)

        self._start_btn = QPushButton("▶  Start Optimisation")
        bold = QFont()
        bold.setBold(True)
        bold.setPointSize(11)
        self._start_btn.setFont(bold)
        self._start_btn.setMinimumHeight(42)
        self._start_btn.setEnabled(False)
        self._start_btn.setToolTip(
            "Hand off to Bayesian Optimisation once the readiness target is met.\n"
            "Switches to the ⚙ Settings tab and enables 'Ask Next Batch'."
        )
        self._start_btn.clicked.connect(self._action_start_bo)
        bvbox.addWidget(self._start_btn)
        outer.addWidget(btn_container)

    # ── Public API ─────────────────────────────────────────────────────────

    def refresh(
        self,
        session_state: Optional[SessionState],
        study=None,
    ) -> None:
        """
        Update the widget with the current session and Optuna study.
        Call after Apply Objectives, CSV load, or session resume.
        """
        self._session_state = session_state
        self._study = study

        if session_state is None:
            self._set_enabled(False)
            self._readiness_lbl.setText("0 / — runs  (define parameters first)")
            self._formula_lbl.setText("")
            return

        self._set_enabled(True)
        cfg = session_state.study_config
        self._obj_cols = [o.column_name for o in cfg.objectives]

        # Compute target
        rt = compute_readiness_target(cfg)
        self._readiness_target = rt["target"]
        self._formula_lbl.setText(f"Formula: {rt['formula_str']}")

        # Count current completed trials
        self._n_done = self._count_completed()

        # Restore existing DoE if any
        doe = session_state.doe_state
        if doe is not None:
            import json
            self._doe_df = pd.DataFrame(doe.points)
            self._populate_table_from_doe(doe)
            self._coverage_grp.show()
            self._points_grp.show()
            self._import_btn.setEnabled(True)
            self._export_pending_btn.setEnabled(True)
            self._register_btn.setEnabled(True)
            self._pairplot_btn.setEnabled(True)
        else:
            self._doe_df = None
            self._table.setRowCount(0)
            self._table.setColumnCount(0)
            self._coverage_grp.hide()
            self._points_grp.hide()

        self._update_readiness_ui()

    def update_n_done(self, n_completed: int) -> None:
        """Refresh the readiness bar after new trials are added."""
        self._n_done = n_completed
        self._update_readiness_ui()

    # ── Internal ───────────────────────────────────────────────────────────

    def _set_enabled(self, enabled: bool) -> None:
        self._generate_btn.setEnabled(enabled)
        self._strategy_combo.setEnabled(enabled)
        self._n_spin.setEnabled(enabled)
        self._seed_spin.setEnabled(enabled)
        self._levels_spin.setEnabled(enabled)

    def _count_completed(self) -> int:
        """Return number of completed Optuna trials in the current study."""
        if self._study is None:
            return 0
        try:
            from optuna.trial import TrialState
            return len([t for t in self._study.trials if t.state == TrialState.COMPLETE])
        except Exception:
            return 0

    def _update_readiness_ui(self) -> None:
        """Refresh bar, label, and Start button based on current n_done / target."""
        target = max(1, self._readiness_target)
        pct = min(100, int(100 * self._n_done / target))
        self._readiness_bar.setValue(pct)

        # Colour the bar
        if pct >= 100:
            colour = "#2ecc71"  # green
        elif pct >= 50:
            colour = "#f39c12"  # amber
        else:
            colour = "#e74c3c"  # red
        self._readiness_bar.setStyleSheet(
            f"QProgressBar::chunk {{ background-color: {colour}; border-radius: 4px; }}"
        )

        self._readiness_lbl.setText(
            f"{self._n_done} / {self._readiness_target} runs  ({pct}%)"
        )
        self._start_btn.setEnabled(self._n_done >= self._readiness_target)
        self.readiness_changed.emit(self._n_done, self._readiness_target)

    def _on_strategy_changed(self, strategy: str) -> None:
        """Show/hide strategy-specific controls when strategy changes."""
        is_sobol = strategy == "Sobol"
        is_ff    = strategy == "FullFactorial"
        is_fixed = strategy in ("PlackettBurman", "BoxBehnken", "CCD")

        self._sobol_lbl.setVisible(is_sobol)
        self._levels_spin.setVisible(is_ff)
        self._levels_row_label.setVisible(is_ff)
        self._n_spin.setEnabled(not is_fixed)

        if is_sobol:
            self._update_sobol_label(self._n_spin.value())
        if is_fixed:
            self._n_spin.setValue(0)  # N is auto; value doesn't matter

        # Update tooltip
        tip = _STRATEGY_TOOLTIPS.get(strategy, "")
        self._strategy_combo.setToolTip(tip)

    def _update_sobol_label(self, n: int) -> None:
        if self._strategy_combo.currentText() == "Sobol":
            rounded = sobol_next_power_of_2(max(n, 2))
            self._sobol_lbl.setText(f"N will be rounded up to {rounded}")

    # ── Actions ────────────────────────────────────────────────────────────

    def _action_generate(self) -> None:
        if self._session_state is None:
            return
        cfg = self._session_state.study_config

        strategy = self._strategy_combo.currentText()
        n_points = self._n_spin.value() if strategy not in ("PlackettBurman", "BoxBehnken", "CCD") else 20
        seed = self._seed_spin.value()
        levels = self._levels_spin.value()

        # Warn if existing DoE with partial results would be discarded
        existing_doe = self._session_state.doe_state
        if existing_doe is not None:
            n_entered = sum(1 for r in existing_doe.results if r is not None)
            if n_entered > 0:
                reply = QMessageBox.question(
                    self,
                    "Discard Existing DoE?",
                    f"A DoE with {n_entered}/{existing_doe.n_points} entered results already exists.\n"
                    "Regenerating will discard it.\n\nContinue?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return

        try:
            df, warns = generate_doe(cfg, n_points, strategy, seed=seed,
                                     levels_per_factor=levels)
        except Exception as exc:
            QMessageBox.critical(self, "DoE Generation Error", str(exc))
            return

        if warns:
            QMessageBox.information(
                self, "DoE Warnings",
                "\n\n".join(warns)
            )

        self._doe_df = df
        actual_n = len(df)

        # Create DoEState and persist
        doe_state = DoEState(
            strategy=strategy,
            n_points=actual_n,
            seed=seed,
            points=df.to_dict(orient="records"),
            results=[None] * actual_n,
        )
        self._session_state.doe_state = doe_state
        SessionManager.save(self._session_state)

        # Save doe_pending.csv
        session_dir = os.path.dirname(self._session_state.session_path)
        pending_path = os.path.join(session_dir, "doe_pending.csv")
        try:
            df.to_csv(pending_path, index=False)
        except Exception:
            pass

        # Populate table
        self._populate_table_from_doe(doe_state)

        # Compute and display coverage metrics
        metrics = coverage_metrics(df)
        if metrics["maximin_distance"] is not None:
            ref = metrics.get("random_maximin_ref") or 0.0
            pct_better = (
                f" (+{100*(metrics['maximin_distance']-ref)/max(ref,1e-9):.0f}% vs random)"
                if ref > 0 and metrics['maximin_distance'] > ref else ""
            )
            self._maximin_lbl.setText(f"{metrics['maximin_distance']:.4f}{pct_better}")
        if metrics["discrepancy"] is not None:
            ref_d = metrics.get("random_discr_ref") or 0.0
            pct_better_d = (
                f" ({100*(ref_d-metrics['discrepancy'])/max(ref_d,1e-9):.0f}% better than random)"
                if ref_d > 0 and metrics['discrepancy'] < ref_d else ""
            )
            self._discr_lbl.setText(f"{metrics['discrepancy']:.4f}{pct_better_d}")
        if metrics["random_maximin_ref"] is not None:
            self._rand_ref_lbl.setText(
                f"maximin={metrics['random_maximin_ref']:.4f},  "
                f"discr={metrics['random_discr_ref']:.4f}"
            )
        self._coverage_grp.show()
        self._points_grp.show()
        self._pairplot_btn.setEnabled(True)
        self._analysis_btn.setEnabled(True)
        self._import_btn.setEnabled(True)
        self._export_pending_btn.setEnabled(True)
        self._register_btn.setEnabled(True)
        self._export_report_btn.setEnabled(True)
        self._progress_lbl.setText(f"Generated {actual_n} points  ({strategy})")

    def _populate_table_from_doe(self, doe: DoEState) -> None:
        """Fill the table with DoE points and (optionally) existing results."""
        if self._session_state is None:
            return
        param_cols = list(doe.points[0].keys()) if doe.points else []
        obj_cols = self._obj_cols
        all_cols = ["✓"] + param_cols + obj_cols

        self._table.setColumnCount(len(all_cols))
        self._table.setHorizontalHeaderLabels(all_cols)
        self._table.setRowCount(len(doe.points))

        for r, point in enumerate(doe.points):
            # Status column
            result = doe.results[r] if r < len(doe.results) else None
            status_item = QTableWidgetItem("✓" if result is not None else "")
            status_item.setFlags(Qt.ItemIsEnabled)   # read-only
            if result is not None:
                status_item.setForeground(
                    self._table.palette().color(self._table.palette().ColorRole.Highlight)
                )
            self._table.setItem(r, 0, status_item)

            # Parameter columns (read-only)
            for ci, col in enumerate(param_cols, start=1):
                val = point.get(col, "")
                item = QTableWidgetItem(str(val) if not isinstance(val, float)
                                        else f"{val:.5g}")
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)  # read-only
                item.setToolTip(f"{col} = {val}")
                self._table.setItem(r, ci, item)

            # Objective columns (editable)
            for oi, obj in enumerate(obj_cols):
                col_idx = 1 + len(param_cols) + oi
                if result is not None and oi < len(result):
                    text = f"{result[oi]:.6g}"
                else:
                    text = ""
                item = QTableWidgetItem(text)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable)
                if text:
                    item.setBackground(
                        self._table.palette().color(self._table.palette().ColorRole.AlternateBase)
                    )
                self._table.setItem(r, col_idx, item)

        self._table.resizeColumnsToContents()
        self._table.horizontalHeader().setSectionResizeMode(
            len(all_cols) - 1, QHeaderView.Stretch
        )

    def _action_import_results(self) -> None:
        """Import objective values from a CSV file."""
        if self._session_state is None or self._session_state.doe_state is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Results CSV", "",
            "CSV Files (*.csv);;All files (*)"
        )
        if not path:
            return
        try:
            results_df = pd.read_csv(path)
        except Exception as exc:
            QMessageBox.critical(self, "Import Error", f"Cannot read CSV:\n{exc}")
            return

        doe = self._session_state.doe_state
        param_cols = list(doe.points[0].keys()) if doe.points else []
        n_matched = 0
        for r, point in enumerate(doe.points):
            # Try to find a matching row in results_df
            for _, row in results_df.iterrows():
                matched = True
                for col in param_cols:
                    if col not in row.index:
                        matched = False
                        break
                    try:
                        if abs(float(row[col]) - float(point[col])) > 1e-6:
                            matched = False
                            break
                    except (TypeError, ValueError):
                        if str(row[col]).strip() != str(point[col]).strip():
                            matched = False
                            break
                if matched:
                    # Fill objective values into the table
                    obj_values = []
                    for oi, obj in enumerate(self._obj_cols):
                        if obj in row.index and not pd.isna(row[obj]):
                            val = float(row[obj])
                            obj_values.append(val)
                            col_idx = 1 + len(param_cols) + oi
                            item = self._table.item(r, col_idx)
                            if item:
                                item.setText(f"{val:.6g}")
                    if len(obj_values) == len(self._obj_cols):
                        n_matched += 1
                    break

        QMessageBox.information(
            self, "Import Complete",
            f"Matched and filled {n_matched} / {len(doe.points)} rows."
        )
        self._register_btn.setEnabled(True)

    def _action_export_pending(self) -> None:
        """Export parameter columns only as a lab-ready CSV."""
        if self._doe_df is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Pending DoE CSV", "doe_pending.csv",
            "CSV Files (*.csv)"
        )
        if not path:
            return
        try:
            self._doe_df.to_csv(path, index=False)
            QMessageBox.information(self, "Exported", f"DoE pending CSV saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _action_export_results_csv(self) -> None:
        """Export full DoE table (parameters + objective values) as a CSV."""
        if self._session_state is None or self._session_state.doe_state is None:
            return
        doe = self._session_state.doe_state
        if not doe.points:
            return

        param_cols = list(doe.points[0].keys())
        rows = []
        for i, pt in enumerate(doe.points):
            row = dict(pt)
            result = doe.results[i] if i < len(doe.results) else None
            for oi, obj_col in enumerate(self._obj_cols):
                if result is not None and oi < len(result):
                    row[obj_col] = result[oi]
                else:
                    row[obj_col] = None
            rows.append(row)

        path, _ = QFileDialog.getSaveFileName(
            self, "Export DoE Results CSV", "doe_results.csv",
            "CSV Files (*.csv)"
        )
        if not path:
            return
        try:
            import pandas as _pd
            _pd.DataFrame(rows).to_csv(path, index=False)
            QMessageBox.information(
                self, "Exported", f"DoE results CSV saved to:\n{path}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _action_register_results(self) -> None:
        """
        Read objective values from the table, create historical Optuna trials,
        and save the updated doe_state.
        """
        if self._session_state is None or self._session_state.doe_state is None:
            return
        if self._study is None:
            QMessageBox.warning(
                self, "No Study",
                "Apply objectives first to create an Optuna study."
            )
            return

        doe = self._session_state.doe_state
        param_cols = list(doe.points[0].keys()) if doe.points else []
        n_obj = len(self._obj_cols)

        trial_dicts = []
        row_indices_with_results = []
        for r in range(len(doe.points)):
            obj_values = []
            all_filled = True
            for oi, obj_col in enumerate(self._obj_cols):
                col_idx = 1 + len(param_cols) + oi
                item = self._table.item(r, col_idx)
                text = item.text().strip() if item else ""
                if not text:
                    all_filled = False
                    break
                try:
                    obj_values.append(float(text))
                except ValueError:
                    all_filled = False
                    break

            if all_filled and obj_values:
                params = doe.points[r]
                trial_dicts.append({
                    "params": params,
                    "values": obj_values,
                    "user_attrs": {"source": "DoE"},
                })
                row_indices_with_results.append(r)
                # Update doe_state results
                doe.results[r] = obj_values

        if not trial_dicts:
            QMessageBox.information(
                self, "No Results",
                "No rows have complete objective values filled in yet.\n"
                "Enter values in the objective columns, then click Register."
            )
            return

        # Register into Optuna study
        from optuna_builder import load_historical_trials
        cfg = self._session_state.study_config
        added, skipped = load_historical_trials(self._study, trial_dicts, cfg)

        # Track registered trial numbers
        try:
            from optuna.trial import TrialState
            completed_numbers = [
                t.number for t in self._study.trials
                if t.state == TrialState.COMPLETE
            ]
            # Add the newest ones (those not already in doe_state)
            existing = set(doe.registered_trial_numbers)
            new_numbers = [n for n in completed_numbers if n not in existing]
            doe.registered_trial_numbers.extend(new_numbers)
        except Exception:
            pass

        # Mark complete if all rows registered
        if len(row_indices_with_results) == len(doe.points):
            doe.complete = True

        self._session_state.doe_state = doe
        SessionManager.save(self._session_state)

        # Update table status column
        for r in row_indices_with_results:
            item = self._table.item(r, 0)
            if item:
                item.setText("✓")

        # Update n_done
        self._n_done = self._count_completed()
        self._update_readiness_ui()

        # Enable Results CSV export now that results exist
        self._export_results_csv_btn.setEnabled(True)

        self._progress_lbl.setText(
            f"Registered {added} new trials  ({skipped} already present).  "
            f"Total: {self._n_done} / {self._readiness_target}"
        )

        QMessageBox.information(
            self, "Results Registered",
            f"Registered {added} trial(s) into the Optuna study.\n\n"
            f"Surrogate readiness: {self._n_done} / {self._readiness_target} runs\n"
            + ("✅ Ready — click 'Start Optimisation'!" if self._n_done >= self._readiness_target
               else "ℹ  Run more experiments to reach the readiness target.")
        )

    def _action_start_bo(self) -> None:
        """Emit the signal to hand off to the BO loop."""
        self.start_bo_requested.emit()

    def _action_show_pairplot(self) -> None:
        """Open a pairplot dialog for the current DoE points."""
        if self._doe_df is None or self._doe_df.empty:
            return
        try:
            from doe_visualisations import show_pairplot_dialog
            doe_state = self._session_state.doe_state if self._session_state else None
            show_pairplot_dialog(self._doe_df, doe_state, self._obj_cols, parent=self)
        except Exception as exc:
            QMessageBox.warning(self, "Pairplot Error", str(exc))

    def _action_show_analysis(self) -> None:
        """Open the full DoE analysis dialog (multi-tab)."""
        if self._doe_df is None or self._doe_df.empty:
            return
        try:
            from doe_visualisations import show_doe_analysis_dialog
            doe_state = self._session_state.doe_state if self._session_state else None
            show_doe_analysis_dialog(
                self._doe_df, doe_state, self._obj_cols, parent=self
            )
        except Exception as exc:
            QMessageBox.warning(self, "Analysis Error", str(exc))

    def _action_export_doe_report(self) -> None:
        """Generate and save a standalone HTML DoE phase report."""
        if self._session_state is None or self._session_state.doe_state is None:
            QMessageBox.information(
                self, "No DoE",
                "Generate a DoE design first, then export the report."
            )
            return

        # Ask where to save
        session_dir = os.path.dirname(self._session_state.session_path)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save DoE Report",
            os.path.join(session_dir, "doe_report.html"),
            "HTML Files (*.html)",
        )
        if not path:
            return

        try:
            from report_generator import doe_report
            cfg = self._session_state.study_config
            html = doe_report(
                doe_state=self._session_state.doe_state,
                doe_df=self._doe_df,
                config=cfg,
                session_state=self._session_state,
            )
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(html)
            import webbrowser
            reply = QMessageBox.question(
                self, "DoE Report Saved",
                f"Report saved to:\n{path}\n\nOpen in browser?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                try:
                    webbrowser.open(path)
                except Exception:
                    pass
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))
