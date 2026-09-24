"""
Power Analysis Widget
=====================
Split into three classes so the input form and the result/plot panel can
live in completely separate areas of the main window:

  PowerInputPanel(QWidget)
      Left-dock tab content — the parameter input form only.
      Emits ``params_changed(sigma, delta, alpha, power, two_sample)``
      whenever any control changes.

  PowerResultPanel(QWidget)
      Centre-area content — N= label, Cohen's d, traffic-light badge,
      and two Matplotlib plots (power curve + N vs. Δ).
      Call ``refresh(sigma, delta, alpha, power, two_sample, n_current)``
      to update.

  PowerAnalysisCoordinator
      Owns both panels and wires them together.
      Public API used by MainWindow:
        .input_panel  → the QWidget to embed in the left dock tab
        .result_panel → the QWidget to embed in the centre stacked widget
        .set_objective_data(series, n_current) → update σ estimate + badge

The pure-math functions (compute_required_n, achieved_power) live in
power_analysis_math.py and are imported here for re-export.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

# PySide6 — module-level for class inheritance / Signal
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# Pure-math functions (no Qt / matplotlib)
from power_analysis_math import achieved_power, compute_required_n  # re-exported

# ── Catppuccin Mocha palette (matches design_space_widget.py) ─────────────────
_BG     = "#1e1e2e"
_AX_BG  = "#181825"
_GRID   = "#313244"
_FG     = "#cdd6f4"
_BLUE   = "#89b4fa"
_GREEN  = "#a6e3a1"
_AMBER  = "#fab387"
_RED    = "#f38ba8"
_PURPLE = "#cba6f7"


# ══════════════════════════════════════════════════════════════════════════════
# Left-panel: input form
# ══════════════════════════════════════════════════════════════════════════════

class PowerInputPanel(QWidget):
    """
    Input form for power analysis — intended for the left dock tab.

    Signals
    -------
    params_changed(sigma, delta, alpha, power, two_sample)
        Emitted on every control change so ``PowerResultPanel`` can update.
    """

    params_changed = Signal(float, float, float, float, bool)

    _ALPHA_OPTIONS = [("0.01", 0.01), ("0.05", 0.05), ("0.10", 0.10)]
    _POWER_OPTIONS = [("0.70", 0.70), ("0.80", 0.80), ("0.90", 0.90), ("0.95", 0.95)]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sigma_from_data: Optional[float] = None
        self._build_ui()

    # ── Construction ───────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 8, 6, 8)
        root.setSpacing(8)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = QLabel("🔬  Power Analysis")
        hdr.setStyleSheet(
            f"color: {_BLUE}; font-size: 12px; font-weight: bold;"
        )
        root.addWidget(hdr)

        intro = QLabel(
            "How many experiments N are needed\n"
            "to detect an improvement Δ given\n"
            "measurement spread σ?"
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {_FG}; font-size: 10px; font-style: italic;")
        root.addWidget(intro)

        # ── Parameter form ────────────────────────────────────────────────────
        grp = QGroupBox("Parameters")
        form = QFormLayout(grp)
        form.setSpacing(6)
        form.setContentsMargins(6, 12, 6, 6)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        # Δ
        self._delta_spin = QDoubleSpinBox()
        self._delta_spin.setRange(0.0, 1e9)   # allow 0 during typing; guarded in result panel
        self._delta_spin.setDecimals(4)
        self._delta_spin.setValue(1.0)
        self._delta_spin.setSingleStep(0.1)
        self._delta_spin.setToolTip(
            "Minimum detectable effect (improvement) in the same units\n"
            "as your objective column.\n"
            "E.g. enter 5 to detect a 5 percentage-point yield improvement."
        )
        self._delta_spin.valueChanged.connect(self._emit)
        form.addRow("Min. effect (Δ):", self._delta_spin)

        # σ row
        sigma_row = QWidget()
        sigma_hl = QHBoxLayout(sigma_row)
        sigma_hl.setContentsMargins(0, 0, 0, 0)
        sigma_hl.setSpacing(4)

        self._sigma_spin = QDoubleSpinBox()
        self._sigma_spin.setRange(0.0, 1e9)   # allow 0 during typing; guarded in result panel
        self._sigma_spin.setDecimals(4)
        self._sigma_spin.setValue(1.0)
        self._sigma_spin.setSingleStep(0.1)
        self._sigma_spin.setToolTip(
            "Expected standard deviation of repeated measurements.\n"
            "Click '📊 From data' to estimate from the loaded objective column."
        )
        self._sigma_spin.valueChanged.connect(self._emit)
        sigma_hl.addWidget(self._sigma_spin)

        self._from_data_btn = QPushButton("📊 From data")
        self._from_data_btn.setFixedWidth(105)
        self._from_data_btn.setEnabled(False)
        self._from_data_btn.setToolTip(
            "Set σ = std of the loaded objective column.\n"
            "Load a CSV and apply objectives first."
        )
        self._from_data_btn.clicked.connect(self._apply_sigma_from_data)
        sigma_hl.addWidget(self._from_data_btn)
        form.addRow("Spread (σ):", sigma_row)

        # α
        self._alpha_combo = QComboBox()
        for label, _ in self._ALPHA_OPTIONS:
            self._alpha_combo.addItem(label)
        self._alpha_combo.setCurrentIndex(1)
        self._alpha_combo.setToolTip(
            "Significance level α (Type I error rate).\n"
            "0.05 → 5% chance of a false positive."
        )
        self._alpha_combo.currentIndexChanged.connect(self._emit)
        form.addRow("Significance (α):", self._alpha_combo)

        # Power
        self._power_combo = QComboBox()
        for label, _ in self._POWER_OPTIONS:
            self._power_combo.addItem(label)
        self._power_combo.setCurrentIndex(1)
        self._power_combo.setToolTip(
            "Desired statistical power (1 − β).\n"
            "0.80 → 80% chance of detecting the real effect."
        )
        self._power_combo.currentIndexChanged.connect(self._emit)
        form.addRow("Desired power (1−β):", self._power_combo)

        # Test type
        self._test_combo = QComboBox()
        self._test_combo.addItems(["One-sample / paired", "Two-sample (independent)"])
        self._test_combo.setToolTip(
            "One-sample / paired: one group vs. reference value or before/after.\n"
            "Two-sample: two independent groups — N reported *per group*."
        )
        self._test_combo.currentIndexChanged.connect(self._emit)
        form.addRow("Test type:", self._test_combo)

        root.addWidget(grp)
        root.addStretch()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_params(self) -> tuple[float, float, float, float, bool]:
        """Return (sigma, delta, alpha, power, two_sample)."""
        return (
            self._sigma_spin.value(),
            self._delta_spin.value(),
            self._ALPHA_OPTIONS[self._alpha_combo.currentIndex()][1],
            self._POWER_OPTIONS[self._power_combo.currentIndex()][1],
            self._test_combo.currentIndex() == 1,
        )

    def set_sigma_from_data(
        self, sigma: Optional[float], n_samples: int = 0
    ) -> None:
        """
        Called by the coordinator when the loaded data changes.

        Parameters
        ----------
        sigma     : estimated σ from the objective column, or None to clear.
        n_samples : number of non-NaN values used for the estimate (tooltip only).
        """
        self._sigma_from_data = sigma
        if sigma is not None and sigma > 0:
            self._from_data_btn.setEnabled(True)
            self._from_data_btn.setToolTip(
                f"Set σ = {sigma:.4g}  (std of {n_samples} values)\nClick to apply."
            )
        else:
            self._from_data_btn.setEnabled(False)
            self._from_data_btn.setToolTip(
                "Set σ to std of loaded objective column.\n"
                "Load a CSV and apply objectives first."
            )

    # ── Internal ───────────────────────────────────────────────────────────────

    def _apply_sigma_from_data(self) -> None:
        if self._sigma_from_data is not None and self._sigma_from_data > 0:
            self._sigma_spin.setValue(self._sigma_from_data)

    def _emit(self) -> None:
        """Emit params_changed with current values."""
        sigma, delta, alpha, power, two_sample = self.get_params()
        self.params_changed.emit(sigma, delta, alpha, power, two_sample)


# ══════════════════════════════════════════════════════════════════════════════
# Right/Centre-panel: result badge + plots
# ══════════════════════════════════════════════════════════════════════════════

class PowerResultPanel(QWidget):
    """
    Result display for power analysis — intended for the centre stacked widget.

    Call ``refresh(sigma, delta, alpha, power, two_sample, n_current)`` to
    recompute N, update labels/badge, and redraw the two Matplotlib plots.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._n_current: int = 0
        self._last_params: tuple = (1.0, 1.0, 0.05, 0.80, False)
        self._build_ui()

    # ── Construction ───────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # ── Title ─────────────────────────────────────────────────────────────
        title = QLabel("🔬  Power Analysis — Sample Size Result")
        title.setStyleSheet(
            f"color: {_BLUE}; font-size: 13px; font-weight: bold;"
        )
        root.addWidget(title)

        # ── Result summary row ────────────────────────────────────────────────
        summary_row = QHBoxLayout()
        summary_row.setSpacing(24)

        # N= big label
        n_col = QVBoxLayout()
        self._n_label = QLabel("N = —")
        self._n_label.setAlignment(Qt.AlignCenter)
        self._n_label.setStyleSheet(
            f"color: {_BLUE}; font-size: 32px; font-weight: bold;"
        )
        n_col.addWidget(self._n_label)
        self._n_sublabel = QLabel("experiments required")
        self._n_sublabel.setAlignment(Qt.AlignCenter)
        self._n_sublabel.setStyleSheet(f"color: {_FG}; font-size: 11px;")
        n_col.addWidget(self._n_sublabel)
        summary_row.addLayout(n_col)

        # Cohen's d + badge
        detail_col = QVBoxLayout()
        self._cohens_label = QLabel("Cohen's d = —")
        self._cohens_label.setAlignment(Qt.AlignCenter)
        self._cohens_label.setStyleSheet(f"color: {_FG}; font-size: 12px;")
        detail_col.addWidget(self._cohens_label)

        self._badge_label = QLabel("—")
        self._badge_label.setAlignment(Qt.AlignCenter)
        self._badge_label.setWordWrap(True)
        self._badge_label.setMinimumWidth(260)
        self._badge_label.setStyleSheet(
            f"background: #2a2a3e; color: {_FG}; "
            "padding: 8px 14px; border-radius: 6px; font-size: 11px;"
        )
        detail_col.addWidget(self._badge_label)
        summary_row.addLayout(detail_col)
        summary_row.addStretch()
        root.addLayout(summary_row)

        # ── Matplotlib canvas ─────────────────────────────────────────────────
        self._fig = Figure(facecolor=_BG, figsize=(9, 4))
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setStyleSheet(f"background-color: {_BG};")
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self._canvas, stretch=1)

    # ── Public API ─────────────────────────────────────────────────────────────

    def refresh(
        self,
        sigma: float,
        delta: float,
        alpha: float,
        power: float,
        two_sample: bool,
        n_current: int,
    ) -> None:
        """Recompute N and update all labels + plots."""
        self._n_current = n_current
        self._last_params = (sigma, delta, alpha, power, two_sample)

        # ── Guard: friendly hint when inputs are 0 / negative (mid-edit) ──
        # Do NOT touch the canvas here — calling draw_idle() on a cleared
        # figure schedules a C++-level repaint that races with the next
        # draw_idle() call and causes a crash in the Qt backend.
        # Instead, leave the last valid plots visible while the user edits.
        if sigma <= 0 or delta <= 0:
            self._n_label.setText("⚠")
            self._n_sublabel.setText("Δ and σ must both be > 0")
            self._cohens_label.setText("Adjust the parameters on the left")
            self._badge_label.setText("—")
            self._badge_label.setStyleSheet(
                f"background: #2a2a3e; color: {_FG}; "
                "padding: 8px 14px; border-radius: 6px; font-size: 11px;"
            )
            return   # canvas unchanged — last valid plots stay visible

        try:
            result = compute_required_n(sigma, delta, alpha, power, two_sample)
        except Exception as exc:
            # Catch all exceptions — e.g. extreme values causing numerical issues.
            # Same rule: don't touch the canvas to avoid racing draw calls.
            self._n_label.setText("⚠")
            self._n_sublabel.setText("Invalid parameters")
            self._cohens_label.setText(str(exc)[:80])
            self._badge_label.setText("—")
            self._badge_label.setStyleSheet(
                f"background: #2a2a3e; color: {_FG}; "
                "padding: 8px 14px; border-radius: 6px; font-size: 11px;"
            )
            return   # canvas unchanged

        n   = result["n"]
        d   = result["cohens_d"]
        tag = result["effect_tag"]

        per_group = " per group" if two_sample else ""
        self._n_label.setText(f"N = {n}")
        self._n_sublabel.setText(f"experiments required{per_group}")
        self._cohens_label.setText(f"Cohen's d = {d:.3f}  ({tag} effect)")

        self._update_badge(n)
        try:
            self._draw_plots(sigma, delta, alpha, power, two_sample, n)
        except Exception:
            # Never let a matplotlib/drawing error propagate to the user
            pass

    # ── Internal ───────────────────────────────────────────────────────────────

    def _update_badge(self, required_n: int) -> None:
        n_cur = self._n_current
        if n_cur == 0:
            self._badge_label.setText(
                "No trial data yet — use the inputs on the left to plan your budget."
            )
            self._badge_label.setStyleSheet(
                f"background: #2a2a3e; color: {_FG}; "
                "padding: 8px 14px; border-radius: 6px; font-size: 11px;"
            )
            return

        ratio = n_cur / required_n
        if ratio >= 1.0:
            msg                = f"✅  Adequately powered\n{n_cur} / {required_n} experiments"
            bg, fg, brd = "#1a3a2a", _GREEN, _GREEN
        elif ratio >= 0.5:
            more               = required_n - n_cur
            msg                = f"⚠  Borderline\n{n_cur} / {required_n}  (need {more} more)"
            bg, fg, brd = "#3d2e00", _AMBER, _AMBER
        else:
            more               = required_n - n_cur
            msg                = f"🔴  Underpowered\n{n_cur} / {required_n}  (need {more} more)"
            bg, fg, brd = "#3d1010", _RED, _RED

        self._badge_label.setText(msg)
        self._badge_label.setStyleSheet(
            f"background: {bg}; color: {fg}; border: 1px solid {brd}; "
            "padding: 8px 14px; border-radius: 6px; font-size: 11px;"
        )

    def _draw_plots(
        self,
        sigma: float,
        delta: float,
        alpha: float,
        power: float,
        two_sample: bool,
        required_n: int,
    ) -> None:
        self._fig.clear()
        ax1, ax2 = self._fig.subplots(1, 2)
        self._style_ax(ax1)
        self._style_ax(ax2)

        # ── Left: power curve  N → achieved power ──────────────────────────
        n_max     = max(200, required_n * 4)
        ns        = np.arange(2, n_max + 1, dtype=float)
        powers_arr = np.array([
            achieved_power(int(ni), sigma, delta, alpha, two_sample) for ni in ns
        ])

        ax1.plot(ns, powers_arr * 100.0, color=_BLUE, linewidth=1.8, zorder=3)

        chosen_pct = power * 100.0
        ax1.axhline(chosen_pct, color=_PURPLE, linewidth=1.0,
                    linestyle="--", alpha=0.75, label=f"Target {chosen_pct:.0f}%")
        ax1.axvline(required_n, color=_RED, linewidth=1.2,
                    linestyle="--", alpha=0.85, zorder=4)
        ax1.scatter([required_n], [chosen_pct], color=_RED, s=65, zorder=5,
                    edgecolors="white", linewidths=0.5)

        x_off = max(2, n_max * 0.07)
        ax1.annotate(
            f"N={required_n}\nPwr={chosen_pct:.0f}%",
            xy=(required_n, chosen_pct),
            xytext=(min(required_n + x_off, n_max * 0.85), chosen_pct - 12.0),
            color=_RED, fontsize=8,
            arrowprops=dict(arrowstyle="->", color=_RED, lw=0.8),
        )

        if self._n_current > 0:
            cur_pwr = achieved_power(self._n_current, sigma, delta, alpha, two_sample)
            ax1.axvline(self._n_current, color=_GREEN, linewidth=0.9,
                        linestyle=":", alpha=0.8,
                        label=f"Current n={self._n_current}  ({cur_pwr*100:.0f}%)")

        ax1.set_xlabel("N (experiments)", color=_FG, fontsize=9)
        ax1.set_ylabel("Achieved power (%)", color=_FG, fontsize=9)
        ax1.set_title("Power Curve", color=_FG, fontsize=10, pad=5)
        ax1.set_xlim(2, n_max)
        ax1.set_ylim(0, 104)
        ax1.grid(color=_GRID, linewidth=0.5, alpha=0.5)
        ax1.legend(loc="lower right", facecolor=_BG, edgecolor=_GRID,
                   labelcolor=_FG, fontsize=8)

        # ── Right: N vs. Δ ──────────────────────────────────────────────────
        delta_min = max(sigma * 0.04, delta * 0.25)
        delta_max = max(sigma * 3.0,  delta * 2.0)
        deltas    = np.linspace(delta_min, delta_max, 300)

        cap = 1200
        req_ns = []
        for dv in deltas:
            try:
                r = compute_required_n(sigma, float(dv), alpha, power, two_sample)
                req_ns.append(min(r["n"], cap))
            except ValueError:
                req_ns.append(np.nan)
        req_ns_arr = np.array(req_ns, dtype=float)

        ax2.plot(deltas, req_ns_arr, color=_BLUE, linewidth=1.8, zorder=3)
        ax2.axvline(delta, color=_RED, linewidth=1.2,
                    linestyle="--", alpha=0.85, zorder=4)
        ax2.axhline(min(required_n, cap), color=_PURPLE, linewidth=1.0,
                    linestyle="--", alpha=0.75)
        ax2.scatter([delta], [min(required_n, cap)], color=_RED, s=65, zorder=5,
                    edgecolors="white", linewidths=0.5)

        if required_n > cap:
            ax2.text(0.97, 0.97,
                     f"Actual N={required_n}\n(axis capped at {cap})",
                     transform=ax2.transAxes, ha="right", va="top",
                     fontsize=8, color=_AMBER)

        if self._n_current > 0:
            ax2.axhline(self._n_current, color=_GREEN, linewidth=0.9,
                        linestyle=":", alpha=0.8,
                        label=f"Current n={self._n_current}")
            ax2.legend(loc="upper right", facecolor=_BG, edgecolor=_GRID,
                       labelcolor=_FG, fontsize=8)

        ax2.set_xlabel("Min. detectable effect (Δ)", color=_FG, fontsize=9)
        ax2.set_ylabel("Required N", color=_FG, fontsize=9)
        ax2.set_title("N vs. Effect Size", color=_FG, fontsize=10, pad=5)
        ax2.set_xlim(delta_min, delta_max)
        ax2.set_ylim(bottom=0)
        ax2.grid(color=_GRID, linewidth=0.5, alpha=0.5)

        self._fig.patch.set_facecolor(_BG)
        try:
            self._fig.tight_layout(pad=1.5)
        except Exception:
            pass
        self._canvas.draw_idle()

    @staticmethod
    def _style_ax(ax) -> None:
        ax.set_facecolor(_AX_BG)
        for spine in ax.spines.values():
            spine.set_color(_GRID)
        ax.tick_params(colors=_FG, labelsize=8)
        ax.xaxis.label.set_color(_FG)
        ax.yaxis.label.set_color(_FG)
        ax.title.set_color(_FG)


# ══════════════════════════════════════════════════════════════════════════════
# Coordinator — wires input panel → result panel
# ══════════════════════════════════════════════════════════════════════════════

class PowerAnalysisCoordinator:
    """
    Owns ``input_panel`` and ``result_panel`` and wires them together.

    MainWindow usage
    ----------------
    self._power = PowerAnalysisCoordinator()

    # Place panels in the layout:
    self._left_tabs.addTab(self._power.input_panel, "🔬  Power")
    self._centre_stack.addWidget(self._power.result_panel)

    # Feed data from the app:
    self._power.set_objective_data(series, n_current=len(completed))
    """

    def __init__(self) -> None:
        self.input_panel  = PowerInputPanel()
        self.result_panel = PowerResultPanel()

        # Wire: any input change → refresh result
        self.input_panel.params_changed.connect(self._on_params_changed)

        # Initial draw with defaults
        self._trigger_refresh()

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_objective_data(
        self,
        series: Optional[pd.Series],
        n_current: int = 0,
    ) -> None:
        """
        Update σ estimate from the loaded objective column and refresh badge.

        Parameters
        ----------
        series    : Numeric objective values.  None to clear.
        n_current : Number of completed trials in the study.
        """
        self.result_panel._n_current = max(0, n_current)

        if series is not None:
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if len(numeric) >= 2:
                sigma_est = float(numeric.std(ddof=1))
                self.input_panel.set_sigma_from_data(sigma_est, len(numeric))
            else:
                self.input_panel.set_sigma_from_data(None)
        else:
            self.input_panel.set_sigma_from_data(None)

        self._trigger_refresh()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _on_params_changed(
        self,
        sigma: float,
        delta: float,
        alpha: float,
        power: float,
        two_sample: bool,
    ) -> None:
        self.result_panel.refresh(
            sigma, delta, alpha, power, two_sample,
            self.result_panel._n_current,
        )

    def _trigger_refresh(self) -> None:
        sigma, delta, alpha, power, two_sample = self.input_panel.get_params()
        self.result_panel.refresh(
            sigma, delta, alpha, power, two_sample,
            self.result_panel._n_current,
        )
