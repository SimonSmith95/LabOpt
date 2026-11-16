"""
dataset_health_widget.py
========================
A QWidget that displays the output of dataset_health.analyze_dataset() in a
clean, scrollable form layout. Designed to live permanently as a page in the
centre QStackedWidget when the Analysis tab's "Dataset Health" inner sub-tab
is active.

No dialogs, no popups. The widget is always present; its content updates
whenever the host calls update(report).

Design constraints (from implementation_guide_analysis_tab_and_report.txt)
---------------------------------------------------------------------------
- Show data only. No verdicts, no colour-coded pass/fail.
- Every metric label carries a tooltip that defines the metric and its formula.
- Tooltip is applied to both the label and the value widget so hovering
  anywhere on the row shows the definition.
- Use the application's existing dark-theme stylesheet. No custom overrides
  except for section header colour (#89b4fa) already in use elsewhere.
- Wrap content in a QScrollArea so it works at any window height.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QVBoxLayout,
    QWidget,
)

# ── Tooltip string constants (Section 10 of implementation guide) ─────────────

TT_ROWS = (
    "Total number of rows in the dataset after cleaning (empty rows removed)."
)

TT_COMPLETE = (
    "Number of rows where all parameter and objective columns have a value.\n"
    "Rows with any missing value in these columns cannot be used as training\n"
    "points for the surrogate model."
)

TT_MISSING = (
    "Percentage of cells with no value across all parameter and objective\n"
    "columns. Formula: (n_missing_cells / (n_rows \u00d7 n_columns)) \u00d7 100 %"
)

TT_PARAMS = (
    "Number of parameter (input) columns as currently configured."
)

TT_OBJECTIVES = (
    "Number of objective (output) columns as currently configured."
)

TT_ROWS_PER_PARAM = (
    "Ratio of complete rows to number of parameters.\n"
    "Formula: complete_rows / n_parameters"
)

TT_MEAN = (
    "Arithmetic mean of all non-missing values in this objective column."
)

TT_STD = (
    "Sample standard deviation of all non-missing values in this objective\n"
    "column. Formula: sqrt(sum((x \u2212 mean)\u00b2) / (n \u2212 1))"
)

TT_MIN_MAX = (
    "Minimum and maximum observed values in this objective column."
)

TT_MEDIAN = (
    "50th percentile (median) of the objective values."
)

TT_IQR = (
    "Interquartile Range \u2014 the difference between the 75th percentile (Q3)\n"
    "and 25th percentile (Q1). Formula: IQR = Q3 \u2212 Q1"
)

TT_CV = (
    "Coefficient of Variation \u2014 the standard deviation expressed as a\n"
    "percentage of the absolute mean. Measures relative variability.\n"
    "Formula: CV = (std / |mean|) \u00d7 100 %\n"
    "Blank when the mean is zero or near-zero."
)

TT_OUTLIERS = (
    "Number of rows where the objective value lies outside the range\n"
    "[Q1 \u2212 1.5 \u00d7 IQR,  Q3 + 1.5 \u00d7 IQR].\n"
    "This is a standard statistical definition of a mild outlier (Tukey, 1977).\n"
    "Outliers may represent genuine extreme results, measurement errors, or\n"
    "transcription mistakes \u2014 they require domain judgement to interpret."
)

TT_REPLICATE_GROUPS = (
    "Number of distinct parameter combinations for which at least two\n"
    "experimental runs were recorded. Replicate experiments allow direct\n"
    "estimation of measurement noise."
)

TT_NOISE_STD = (
    "Estimated measurement noise, computed as the pooled standard deviation\n"
    "across replicate groups (groups of rows with identical parameter values).\n"
    "Formula: sqrt( mean of within-group sample variances )\n"
    "Only available when at least 2 replicate groups, each with at least 2 runs,\n"
    "are present. Blank otherwise."
)

TT_SNR = (
    "Signal-to-Noise Ratio \u2014 the range of observed objective values divided\n"
    "by the estimated noise standard deviation.\n"
    "Formula: SNR = (max \u2212 min) / noise_std\n"
    "Only available when a noise estimate exists (see 'Noise std (replicates)').\n"
    "Blank otherwise."
)

# ── Helpers ───────────────────────────────────────────────────────────────────

_HEADER_STYLE = "color: #89b4fa; font-weight: bold; font-size: 11px;"


def _fmt(val, decimals: int = 4) -> str:
    """Format a numeric value or return an em-dash for None."""
    if val is None:
        return "\u2014"
    try:
        f = float(val)
        # Use fixed notation; trim unnecessary trailing zeros
        return f"{f:.{decimals}g}"
    except (TypeError, ValueError):
        return str(val)


def _make_separator() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.HLine)
    sep.setFrameShadow(QFrame.Sunken)
    return sep


def _metric_row(
    layout: QVBoxLayout,
    label: str,
    value: str,
    tooltip: str,
) -> None:
    """Add a label | value row with tooltip on both widgets."""
    row = QWidget()
    hl = QHBoxLayout(row)
    hl.setContentsMargins(2, 1, 2, 1)
    hl.setSpacing(8)

    lbl = QLabel(label)
    lbl.setToolTip(tooltip)
    lbl.setMinimumWidth(160)
    lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
    hl.addWidget(lbl)

    val_lbl = QLabel(value)
    val_lbl.setToolTip(tooltip)
    val_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    hl.addWidget(val_lbl)

    layout.addWidget(row)


def _section_header(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(_HEADER_STYLE)
    return lbl


# ── Main widget ───────────────────────────────────────────────────────────────

class DatasetHealthWidget(QWidget):
    """
    Displays the output of dataset_health.analyze_dataset() in a scrollable
    form.  Call update(report) to populate; call update(None) to show the
    placeholder.

    Signals
    -------
    refresh_requested : emitted when the user clicks the Refresh button.
        The host (main_window) connects this to re-run analyze_dataset and
        then calls update(new_report).
    """

    refresh_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer_vbox = QVBoxLayout(self)
        outer_vbox.setContentsMargins(8, 8, 8, 8)
        outer_vbox.setSpacing(6)

        # ── Title row ─────────────────────────────────────────────────────────
        title_row = QHBoxLayout()
        title_lbl = QLabel("Dataset Statistics")
        title_lbl.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #cdd6f4;"
        )
        title_row.addWidget(title_lbl)
        title_row.addStretch()

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setToolTip(
            "Re-compute dataset statistics from the currently loaded data."
        )
        self._refresh_btn.setFixedWidth(80)
        self._refresh_btn.clicked.connect(self.refresh_requested)
        title_row.addWidget(self._refresh_btn)
        outer_vbox.addLayout(title_row)

        # ── Placeholder ───────────────────────────────────────────────────────
        self._placeholder = QLabel(
            "Load a CSV file and apply objectives to view dataset statistics."
        )
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet(
            "color: #6c6f85; font-style: italic; font-size: 12px; padding: 40px;"
        )
        outer_vbox.addWidget(self._placeholder)

        # ── Scroll area for content ───────────────────────────────────────────
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet("QScrollArea { border: none; }")
        self._content_widget = QWidget()
        self._content_layout = QVBoxLayout(self._content_widget)
        self._content_layout.setContentsMargins(4, 4, 4, 4)
        self._content_layout.setSpacing(6)
        self._content_layout.setAlignment(Qt.AlignTop)
        self._scroll.setWidget(self._content_widget)
        self._scroll.hide()
        outer_vbox.addWidget(self._scroll, stretch=1)

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, report: Optional[dict]) -> None:
        """
        Populate the widget with the output of analyze_dataset(), or show the
        placeholder when report is None.

        Parameters
        ----------
        report : dict as returned by dataset_health.analyze_dataset(), or None.
        """
        # Clear existing content
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if report is None:
            self._placeholder.show()
            self._scroll.hide()
            return

        self._placeholder.hide()
        self._scroll.show()

        layout = self._content_layout

        # ── Section A: Dataset Overview ───────────────────────────────────────
        layout.addWidget(_section_header("Dataset Overview"))

        _metric_row(layout, "Rows",
                    _fmt(report.get("n_rows"), 0),
                    TT_ROWS)
        _metric_row(layout, "Complete rows",
                    _fmt(report.get("n_complete"), 0),
                    TT_COMPLETE)
        _metric_row(layout, "Missing (%)",
                    _fmt(report.get("missing_pct"), 2),
                    TT_MISSING)
        _metric_row(layout, "Parameters",
                    _fmt(report.get("n_params"), 0),
                    TT_PARAMS)
        _metric_row(layout, "Objectives",
                    _fmt(report.get("n_objectives"), 0),
                    TT_OBJECTIVES)
        _metric_row(layout, "Rows per parameter",
                    _fmt(report.get("rows_per_param"), 2),
                    TT_ROWS_PER_PARAM)

        # ── Section B: Per-objective statistics ───────────────────────────────
        for obj in report.get("objectives", []):
            layout.addWidget(_make_separator())
            col_name = obj.get("column", "?")
            layout.addWidget(_section_header(f"Objective: {col_name}"))

            n_valid = obj.get("n_valid", 0)
            _metric_row(layout, "Valid rows",
                        _fmt(n_valid, 0),
                        "Number of non-missing values in this objective column.")

            _metric_row(layout, "Mean",       _fmt(obj.get("mean")),       TT_MEAN)
            _metric_row(layout, "Std",        _fmt(obj.get("std")),        TT_STD)

            min_v = obj.get("min")
            max_v = obj.get("max")
            min_max_str = (
                f"{_fmt(min_v)} / {_fmt(max_v)}"
                if min_v is not None and max_v is not None
                else "\u2014"
            )
            _metric_row(layout, "Min / Max",  min_max_str,                 TT_MIN_MAX)
            _metric_row(layout, "Median",     _fmt(obj.get("median")),     TT_MEDIAN)
            _metric_row(layout, "IQR",        _fmt(obj.get("iqr")),        TT_IQR)
            _metric_row(layout, "CV (%)",     _fmt(obj.get("cv_pct"), 2),  TT_CV)

            outlier_count = obj.get("outlier_count", 0)
            _metric_row(layout, "Outliers (IQR)",
                        str(outlier_count),
                        TT_OUTLIERS)

            _metric_row(layout, "Replicate groups",
                        _fmt(obj.get("replicate_groups"), 0),
                        TT_REPLICATE_GROUPS)
            _metric_row(layout, "Noise std (replicates)",
                        _fmt(obj.get("noise_std_replicates")),
                        TT_NOISE_STD)
            _metric_row(layout, "SNR",
                        _fmt(obj.get("snr"), 3),
                        TT_SNR)

        # ── Section C: Per-parameter table ────────────────────────────────────
        parameters = report.get("parameters", [])
        if parameters:
            layout.addWidget(_make_separator())
            layout.addWidget(_section_header("Parameters"))

            tbl = QTableWidget(len(parameters), 5)
            tbl.setHorizontalHeaderLabels(
                ["Parameter", "Type", "Valid rows", "Unique values", "Missing (%)"]
            )
            tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
            tbl.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
            tbl.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
            tbl.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
            tbl.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
            tbl.setAlternatingRowColors(True)
            tbl.setEditTriggers(QTableWidget.NoEditTriggers)
            tbl.setSelectionMode(QTableWidget.NoSelection)
            tbl.verticalHeader().setVisible(False)
            # Keep table height tight
            tbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            row_h = tbl.rowHeight(0) if tbl.rowCount() > 0 else 24
            header_h = tbl.horizontalHeader().height()
            tbl.setFixedHeight(header_h + row_h * len(parameters) + 4)

            # Tooltip texts for column headers
            _col_tooltips = [
                "Parameter column name.",
                "Inferred data type (float, int, categorical, bool).",
                "Number of rows with a non-missing value.",
                "Number of distinct values observed.",
                "Percentage of rows with no value in this column.",
            ]
            for ci, tt in enumerate(_col_tooltips):
                tbl.horizontalHeaderItem(ci).setToolTip(tt)

            for ri, param in enumerate(parameters):
                tbl.setItem(ri, 0, QTableWidgetItem(str(param.get("column", ""))))
                tbl.setItem(ri, 1, QTableWidgetItem(str(param.get("dtype_inferred", ""))))
                tbl.setItem(ri, 2, QTableWidgetItem(str(param.get("n_valid", ""))))
                tbl.setItem(ri, 3, QTableWidgetItem(str(param.get("n_unique", ""))))
                tbl.setItem(ri, 4, QTableWidgetItem(_fmt(param.get("missing_pct"), 2)))

            layout.addWidget(tbl)

        layout.addStretch()
