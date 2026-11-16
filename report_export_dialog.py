"""
report_export_dialog.py
=======================
Unified report export dialog for LabOpt.

The user selects which sections to include in the report before exporting,
and chooses an export format (HTML, CSV, or Excel).

Surrogate validation can be triggered here if it has not already been run,
eliminating the need to navigate to a different tab before exporting.

Usage
-----
    dlg = ReportExportDialog(
        parent=main_window,
        session_state=self._session_state,
        study=self._study,
        df=self._df,
        validation_results=self._validation_results,
        doe_state=getattr(self._session_state, 'doe_state', None),
        export_callback=self._do_generate_report,
    )
    dlg.exec()

The ``export_callback`` receives a ``set[str]`` of section keys, the
validation results, and a format string (``"html"``, ``"csv"``, or
``"excel"``).  This keeps the heavy generation logic in main_window.py
where all the widget references live.
"""
from __future__ import annotations

from typing import Callable, Optional, Set

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)


# ── Section definitions ───────────────────────────────────────────────────────

class _Section:
    """Metadata for one report section."""
    def __init__(
        self,
        key: str,
        label: str,
        tooltip: str,
        available: bool,
        unavailable_reason: str = "",
        default_checked: bool = True,
    ) -> None:
        self.key               = key
        self.label             = label
        self.tooltip           = tooltip
        self.available         = available
        self.unavailable_reason = unavailable_reason
        self.default_checked   = default_checked
        self.checkbox: Optional[QCheckBox] = None


# ── Dialog ────────────────────────────────────────────────────────────────────

class ReportExportDialog(QDialog):
    """
    Section-selection dialog for report export.

    Parameters
    ----------
    parent             : Parent QWidget (typically MainWindow).
    session_state      : Current SessionState or None.
    study              : Current optuna.Study or None.
    df                 : Loaded DataFrame or None.
    validation_results : Completed validation results or None.
    doe_state          : Current DoE state or None.
    export_callback    : Callable[[set[str], any, str], None] — called with the
                         set of selected section keys, validation results, and the
                         chosen format string when the user clicks Export.
                         The callback does the actual report generation.
    """

    def __init__(
        self,
        parent=None,
        *,
        session_state=None,
        study=None,
        df=None,
        validation_results=None,
        doe_state=None,
        export_callback: Optional[Callable[..., None]] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Report")
        self.setMinimumWidth(520)

        self._session_state      = session_state
        self._study              = study
        self._df                 = df
        self._validation_results = validation_results
        self._doe_state          = doe_state
        self._export_callback    = export_callback

        self._sections: list[_Section] = self._build_sections()
        self._build_ui()

    # ── Section availability analysis ─────────────────────────────────────────

    def _build_sections(self) -> list[_Section]:
        """Determine which sections are available based on current app state."""
        from optuna.trial import TrialState as _TS

        # Count completed trials
        n_completed = 0
        is_multi    = False
        if self._study:
            completed = [t for t in self._study.trials if t.state == _TS.COMPLETE]
            n_completed = len(completed)
            if self._session_state:
                is_multi = len(self._session_state.study_config.objectives) >= 2

        # Count unique rows for validation threshold
        n_unique = 0
        if self._df is not None and self._session_state:
            cfg = self._session_state.study_config
            numeric_cols = [
                p.name for p in cfg.parameters
                if p.enabled and p.name in self._df.columns
            ]
            if numeric_cols:
                try:
                    n_unique = len(
                        self._df.dropna(subset=numeric_cols)
                                .drop_duplicates(subset=numeric_cols)
                    )
                except Exception:
                    n_unique = len(self._df)

        has_doe = bool(
            self._doe_state is not None
            and getattr(self._doe_state, 'n_points', 0) > 0
        )
        has_df_and_obj = (
            self._df is not None
            and self._session_state is not None
            and bool(self._session_state.study_config.objectives)
        )
        val_ok = n_unique >= 30

        sections = [
            _Section(
                key="dataset_health",
                label="Dataset Statistics",
                tooltip="Descriptive statistics for the loaded dataset "
                        "(rows, missing data, CV, noise estimates, SNR).",
                available=has_df_and_obj,
                unavailable_reason=(
                    "Requires a CSV file to be loaded with objectives configured."
                ),
            ),
            _Section(
                key="all_trials",
                label="All Trials Table",
                tooltip=(
                    "Full table of all completed experimental trials.\n"
                    "Tip: leave unchecked when there are many trials to keep\n"
                    "the report/file size small."
                ),
                available=n_completed >= 1,
                unavailable_reason=(
                    f"Requires at least 1 completed trial. Currently: {n_completed}."
                ),
                default_checked=False,   # opt-in — can be large for many trials
            ),
            _Section(
                key="best_pareto",
                label="Best / Pareto Table",
                tooltip="Table of the best trial (single-objective) or Pareto-optimal "
                        "trials (multi-objective).",
                available=n_completed >= 1,
                unavailable_reason=(
                    f"Requires at least 1 completed trial. Currently: {n_completed}."
                ),
            ),
            _Section(
                key="convergence",
                label="Convergence Plot",
                tooltip="Plot showing how the best objective value improved over trials.",
                available=n_completed >= 2,
                unavailable_reason=(
                    f"Requires at least 2 completed trials. Currently: {n_completed}."
                ),
            ),
            _Section(
                key="pareto_front",
                label="Pareto Front Plot",
                tooltip="Scatter plot of the Pareto front for multi-objective studies.",
                available=is_multi and n_completed >= 2,
                unavailable_reason=(
                    "Requires a multi-objective study with at least 2 completed trials."
                    if not is_multi else
                    f"Requires at least 2 completed trials. Currently: {n_completed}."
                ),
            ),
            _Section(
                key="design_space",
                label="Design Space Pairplot",
                tooltip="Pairplot showing parameter combinations explored so far.",
                available=n_completed >= 2,
                unavailable_reason=(
                    f"Requires at least 2 completed trials. Currently: {n_completed}."
                ),
            ),
            _Section(
                key="validation",
                label="Surrogate Validation",
                tooltip=(
                    "Cross-validated accuracy of the surrogate model (§0–§5).\n"
                    "If not already complete, validation will run before export\n"
                    "(this may take 3–10 minutes depending on dataset size)."
                ),
                available=val_ok,
                unavailable_reason=(
                    f"Requires at least 30 unique rows. Currently: {n_unique}."
                ),
                default_checked=self._validation_results is not None,
            ),
            _Section(
                key="doe_design",
                label="DoE Design",
                tooltip="Pairplot and coverage metrics for the generated DoE design.",
                available=has_doe,
                unavailable_reason=(
                    "Requires a DoE design to have been generated in the DoE tab."
                ),
            ),
        ]
        return sections

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        vbox = QVBoxLayout(self)
        vbox.setSpacing(10)
        vbox.setContentsMargins(12, 12, 12, 12)

        # ── Section selection ──────────────────────────────────────────────
        sections_grp = QGroupBox("Include in report")
        sg = QVBoxLayout(sections_grp)
        sg.setSpacing(6)

        for sec in self._sections:
            cb = QCheckBox(sec.label)
            cb.setToolTip(sec.tooltip)
            cb.setChecked(sec.available and sec.default_checked)
            cb.setEnabled(sec.available)
            if not sec.available:
                cb.setToolTip(
                    f"{sec.tooltip}\n\n"
                    f"Currently unavailable: {sec.unavailable_reason}"
                )
            sec.checkbox = cb
            sg.addWidget(cb)

        vbox.addWidget(sections_grp)

        # ── Export format ──────────────────────────────────────────────────
        format_grp = QGroupBox("Export format")
        fg = QVBoxLayout(format_grp)
        fg.setSpacing(5)

        self._fmt_btn_group = QButtonGroup(self)
        self._rb_html  = QRadioButton(
            "HTML  —  full report with charts (opens in browser)"
        )
        self._rb_csv   = QRadioButton(
            "CSV  —  data tables only  (.csv)"
        )
        self._rb_excel = QRadioButton(
            "Excel  —  data tables in separate sheets  (.xlsx)"
        )
        self._rb_html.setChecked(True)
        for rb in (self._rb_html, self._rb_csv, self._rb_excel):
            self._fmt_btn_group.addButton(rb)
            fg.addWidget(rb)

        # Warning shown when a non-HTML format is selected
        self._fmt_note = QLabel(
            "⚠  Charts and surrogate validation cannot be included in CSV or "
            "Excel exports — only table sections will be exported."
        )
        self._fmt_note.setWordWrap(True)
        self._fmt_note.setStyleSheet(
            "color: #f9e2af; font-size: 11px; font-style: italic; margin-top: 4px;"
        )
        self._fmt_note.setVisible(False)
        fg.addWidget(self._fmt_note)

        # React to format changes
        self._rb_html.toggled.connect(self._on_format_changed)
        self._rb_csv.toggled.connect(self._on_format_changed)
        self._rb_excel.toggled.connect(self._on_format_changed)

        vbox.addWidget(format_grp)

        # ── Validation note ────────────────────────────────────────────────
        val_sec = next((s for s in self._sections if s.key == "validation"), None)
        if val_sec and val_sec.available:
            if self._validation_results is not None:
                val_note = QLabel(
                    "Surrogate validation has already been run and will be "
                    "included if selected above."
                )
            else:
                val_note = QLabel(
                    "If 'Surrogate Validation' is selected, validation will run "
                    "automatically before the report is generated. This may take "
                    "several minutes."
                )
            val_note.setWordWrap(True)
            val_note.setStyleSheet("color: #6c6f85; font-size: 11px; font-style: italic;")
            vbox.addWidget(val_note)

        # ── Progress area (hidden by default) ─────────────────────────────
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        self._progress_bar.setFixedHeight(14)
        vbox.addWidget(self._progress_bar)

        self._progress_lbl = QLabel("")
        self._progress_lbl.setStyleSheet("font-size: 10px; color: #a6e3a1;")
        self._progress_lbl.setVisible(False)
        vbox.addWidget(self._progress_lbl)

        # ── Buttons ────────────────────────────────────────────────────────
        btns = QDialogButtonBox()
        self._export_btn = btns.addButton("Export HTML", QDialogButtonBox.AcceptRole)
        self._export_btn.clicked.connect(self._on_export)
        btns.addButton("Cancel", QDialogButtonBox.RejectRole)
        btns.rejected.connect(self.reject)
        vbox.addWidget(btns)

    # ── Format helpers ────────────────────────────────────────────────────────

    def _selected_format(self) -> str:
        """Return ``"html"``, ``"csv"``, or ``"excel"`` based on radio selection."""
        if self._rb_csv.isChecked():
            return "csv"
        if self._rb_excel.isChecked():
            return "excel"
        return "html"

    def _on_format_changed(self) -> None:
        """Update UI when the export format radio buttons change."""
        is_html = self._rb_html.isChecked()
        self._fmt_note.setVisible(not is_html)
        # Rename the export button to reflect chosen format
        fmt = self._selected_format()
        labels = {"html": "Export HTML", "csv": "Export CSV", "excel": "Export Excel"}
        self._export_btn.setText(labels.get(fmt, "Export"))

    # ── Export ────────────────────────────────────────────────────────────────

    def _selected_sections(self) -> Set[str]:
        return {
            sec.key
            for sec in self._sections
            if sec.checkbox is not None and sec.checkbox.isChecked()
        }

    def _on_export(self) -> None:
        """Collect selected sections, optionally run validation, then export."""
        sections = self._selected_sections()
        if not sections:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "Nothing Selected",
                "Select at least one section to include in the report."
            )
            return

        # Validation is only relevant for HTML exports
        fmt = self._selected_format()
        val_sec = next((s for s in self._sections if s.key == "validation"), None)
        needs_val = (
            fmt == "html"
            and "validation" in sections
            and self._validation_results is None
            and val_sec is not None
            and val_sec.available
        )

        if needs_val:
            self._run_validation_then_export(sections)
        else:
            self._do_export(sections)

    def _run_validation_then_export(self, sections: Set[str]) -> None:
        """Start ValidationWorker; on completion call _do_export."""
        if self._df is None or self._session_state is None:
            self._do_export(sections)
            return

        from parameter_config import ParameterType

        cfg = self._session_state.study_config
        feature_cols = [
            p.name for p in cfg.parameters
            if p.enabled
            and p.ptype in (ParameterType.FLOAT, ParameterType.INT)
            and p.name in self._df.columns
        ]
        if not feature_cols:
            self._do_export(sections)
            return

        obj_col = cfg.objectives[0].column_name if cfg.objectives else None
        if not obj_col or obj_col not in self._df.columns:
            self._do_export(sections)
            return

        import numpy as np
        sub = self._df[feature_cols + [obj_col]].dropna()
        if len(sub) < 5:
            self._do_export(sections)
            return

        X = sub[feature_cols].values.astype(float)
        y = sub[obj_col].values.astype(float)

        # Context arrays
        ctx_names = [
            cv.column_name
            for cv in getattr(cfg, "context_variables", [])
            if cv.column_name in self._df.columns
        ]
        if ctx_names:
            ctx_sub = self._df[ctx_names].copy().fillna(self._df[ctx_names].mean())
            ctx_X   = ctx_sub.values.astype(float)
        else:
            ctx_X     = None
            ctx_names = None

        direction   = cfg.objectives[0].direction   if cfg.objectives else "minimize"
        target_name = cfg.objectives[0].column_name if cfg.objectives else "objective"

        # Show progress
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self._progress_lbl.setText("Running surrogate validation…")
        self._progress_lbl.setVisible(True)
        self._export_btn.setEnabled(False)

        from validation_worker import ValidationWorker
        self._val_worker = ValidationWorker(
            X=X, y=y,
            feature_names=feature_cols,
            target_name=target_name,
            direction=direction,
            context_X=ctx_X,
            context_names=ctx_names,
            parent=self,
        )
        self._val_worker.progress_updated.connect(self._on_val_progress)
        self._val_worker.validation_done.connect(
            lambda results: self._on_val_done(results, sections)
        )
        self._val_worker.error_occurred.connect(
            lambda msg: self._on_val_error(msg, sections)
        )
        self._val_worker.start()
        self._pending_sections = sections

    def _on_val_progress(self, pct: int, msg: str) -> None:
        self._progress_bar.setValue(pct)
        self._progress_lbl.setText(msg)

    def _on_val_done(self, results, sections: Set[str]) -> None:
        self._validation_results = results
        self._progress_bar.setValue(100)
        self._progress_lbl.setText("Validation complete — generating report…")
        self._do_export(sections)

    def _on_val_error(self, msg: str, sections: Set[str]) -> None:
        self._progress_bar.setVisible(False)
        self._progress_lbl.setText(f"Validation failed: {msg[:120]}")
        self._export_btn.setEnabled(True)
        # Export without validation
        sections_no_val = sections - {"validation"}
        if sections_no_val:
            self._do_export(sections_no_val)

    def _do_export(self, sections: Set[str]) -> None:
        """Call the export callback and close the dialog."""
        self._progress_bar.setVisible(False)
        self._progress_lbl.setVisible(False)
        self._export_btn.setEnabled(True)

        if self._export_callback is not None:
            try:
                self._export_callback(
                    sections,
                    self._validation_results,
                    self._selected_format(),
                )
            except Exception as exc:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.critical(
                    self, "Export Error",
                    f"Failed to generate report:\n{exc}"
                )
                return

        self.accept()
