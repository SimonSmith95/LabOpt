"""
Design Space Visualisation Widget
==================================
Embeds a Matplotlib figure that shows where the historical data sits in
parameter space and how the most-recently suggested batch relates to it.

Plot type is chosen automatically from the number of enabled numeric params:

  ≤ 5  params → pairwise scatter grid (lower-triangle pairplot)
  6–12 params → parallel coordinates
  > 12 params → 1-D marginal strip (histogram + suggestion markers)

All plots use the Catppuccin Mocha dark palette to match the rest of the app.
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from PySide6.QtCore import Qt
from PySide6.QtGui import QScreen
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizeGrip,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from parameter_config import ObjectiveConfig, ParameterConfig, ParameterType

# ── Catppuccin Mocha palette ──────────────────────────────────────────────────
_BG      = "#1e1e2e"   # base background
_AX_BG   = "#181825"   # axis background (slightly darker)
_GRID    = "#313244"   # grid / spine colour
_FG      = "#cdd6f4"   # foreground text
_HIST    = "#89b4fa"   # historical data points (blue)
_PARETO  = "#a6e3a1"   # pareto-optimal points (green)
_SUGGEST = "#f38ba8"   # suggested batch points (red/pink)
_SUGGEST2 = "#fab387"  # second suggestion colour if needed (peach)


class DesignSpaceWidget(QWidget):
    """
    Embedded Matplotlib canvas that visualises the parameter design space.

    Public API
    ----------
    refresh(df, params, objectives, suggestions=None)
        Rebuild the plot with new data / suggestions.
    clear()
        Hide the canvas and show the placeholder text.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._df: Optional[pd.DataFrame] = None
        self._params: List[ParameterConfig] = []
        self._objectives: List[ObjectiveConfig] = []
        # suggestions: list of param-value dicts [{"Temperature": 200, ...}, ...]
        self._suggestions: Optional[List[dict]] = None

        # ── Pairplot page state ────────────────────────────────────────────
        self._page: int = 0        # current page index (0-based)
        _PAGE_SIZE = 5             # max params per pairplot page (class-level below)

        # constrained_layout automatically sizes axes, legends, colorbars and
        # suptitles so that nothing overflows the figure boundary.
        self._fig = Figure(facecolor=_BG, layout='constrained')
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setStyleSheet(f"background-color: {_BG};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── Toolbar ───────────────────────────────────────────────────────
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Colour by:"))
        self._obj_combo = QComboBox()
        self._obj_combo.setFixedWidth(180)
        self._obj_combo.currentIndexChanged.connect(self._redraw)
        toolbar.addWidget(self._obj_combo)

        toolbar.addWidget(QLabel("  View:"))
        self._view_combo = QComboBox()
        self._view_combo.addItems(["Auto", "Pairplot", "Parallel Coords", "Marginals"])
        self._view_combo.setFixedWidth(130)
        self._view_combo.setToolTip(
            "Auto: choose plot type based on parameter count.\n"
            "Pairplot: scatter grid (paged, 5 params per page).\n"
            "Parallel Coords: all params on parallel axes.\n"
            "Marginals: stacked 1-D histograms."
        )
        self._view_combo.currentIndexChanged.connect(self._on_view_changed)
        toolbar.addWidget(self._view_combo)

        toolbar.addStretch()

        # Page navigation — hidden when paging is not applicable
        self._prev_btn = QPushButton("◀")
        self._prev_btn.setFixedWidth(32)
        self._prev_btn.setToolTip("Previous page of parameters")
        self._prev_btn.clicked.connect(self._action_prev_page)
        toolbar.addWidget(self._prev_btn)

        self._page_label = QLabel("Page 1 / 1")
        self._page_label.setFixedWidth(90)
        self._page_label.setAlignment(Qt.AlignCenter)
        self._page_label.setStyleSheet(f"color: {_FG}; font-size: 11px;")
        toolbar.addWidget(self._page_label)

        self._next_btn = QPushButton("▶")
        self._next_btn.setFixedWidth(32)
        self._next_btn.setToolTip("Next page of parameters")
        self._next_btn.clicked.connect(self._action_next_page)
        toolbar.addWidget(self._next_btn)

        self._info_label = QLabel("")
        self._info_label.setStyleSheet(f"color: {_FG}; font-size: 11px;")
        toolbar.addWidget(self._info_label)

        refresh_btn = QPushButton("⟳")
        refresh_btn.setFixedWidth(36)
        refresh_btn.setToolTip("Redraw the design space plot.")
        refresh_btn.clicked.connect(self._redraw)
        toolbar.addWidget(refresh_btn)

        layout.addLayout(toolbar)

        # Start with page nav hidden
        self._set_page_nav_visible(False)

        # ── Canvas ────────────────────────────────────────────────────────
        layout.addWidget(self._canvas, stretch=1)

        # ── Placeholder ───────────────────────────────────────────────────
        self._placeholder = QLabel(
            "Load a CSV and apply objectives to see the design space."
        )
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet(
            f"color: {_FG}; font-size: 13px; font-style: italic;"
        )
        layout.addWidget(self._placeholder)

        self._canvas.hide()

    # ══════════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════════

    def refresh(
        self,
        df: pd.DataFrame,
        params: List[ParameterConfig],
        objectives: List[ObjectiveConfig],
        suggestions: Optional[List[dict]] = None,
    ) -> None:
        """
        Update data and redraw.

        Parameters
        ----------
        df          : Full historical DataFrame (all rows, all columns).
        params      : Parameter configurations (all types; non-numeric are skipped).
        objectives  : Objective configurations used to colour points.
        suggestions : Optional list of param-value dicts for the latest suggested
                      batch.  Each dict maps param name → value.
        """
        self._df          = df
        self._params      = params
        self._objectives  = objectives
        self._suggestions = suggestions

        # Sync objective combobox
        prev = self._obj_combo.currentText()
        self._obj_combo.blockSignals(True)
        self._obj_combo.clear()
        for o in objectives:
            self._obj_combo.addItem(o.column_name)
        if prev in [o.column_name for o in objectives]:
            self._obj_combo.setCurrentText(prev)
        self._obj_combo.blockSignals(False)

        self._redraw()

    def clear(self) -> None:
        """Remove data and show the placeholder message."""
        self._df = None
        self._params = []
        self._objectives = []
        self._suggestions = None
        self._fig.clear()
        self._canvas.draw_idle()
        self._canvas.hide()
        self._placeholder.show()

    # ══════════════════════════════════════════════════════════════════════
    # Internal: choose & dispatch to the right plot type
    # ══════════════════════════════════════════════════════════════════════

    # ── Pairplot page navigation ───────────────────────────────────────────

    _PAGE_SIZE = 4   # max parameters shown per page (pairplot & marginals)

    def _set_page_nav_visible(self, visible: bool) -> None:
        for w in (self._prev_btn, self._page_label, self._next_btn):
            w.setVisible(visible)

    def _update_page_nav(self, n_total: int) -> None:
        n_pages = max(1, math.ceil(n_total / self._PAGE_SIZE))
        self._page = max(0, min(self._page, n_pages - 1))
        self._page_label.setText(f"Page {self._page + 1} / {n_pages}")
        self._prev_btn.setEnabled(self._page > 0)
        self._next_btn.setEnabled(self._page < n_pages - 1)

    def _on_view_changed(self) -> None:
        self._page = 0
        self._redraw()

    def _action_prev_page(self) -> None:
        self._page = max(0, self._page - 1)
        self._redraw()

    def _action_next_page(self) -> None:
        self._page += 1
        self._redraw()

    # ── Main dispatch ──────────────────────────────────────────────────────

    def _redraw(self) -> None:
        if self._df is None or not self._params:
            return

        numeric = [
            p for p in self._params
            if p.enabled
            and p.ptype in (ParameterType.FLOAT, ParameterType.INT)
            and p.name in self._df.columns
        ]

        n_hist = len(self._df)
        n_sug  = len(self._suggestions) if self._suggestions else 0
        self._info_label.setText(
            f"{n_hist} historical pts" + (f"  |  {n_sug} suggested" if n_sug else "")
        )

        self._fig.clear()

        view = self._view_combo.currentText()   # "Auto" | "Pairplot" | ...
        n    = len(numeric)

        if n == 0:
            self._draw_no_numeric()
            self._set_page_nav_visible(False)
        else:
            # Determine whether to use the pairplot
            use_pairplot = (
                view == "Pairplot"
                or (view == "Auto" and n <= self._PAGE_SIZE)
            )
            use_parallel = (
                view == "Parallel Coords"
                or (view == "Auto" and not use_pairplot and n <= 12)
            )

            if use_pairplot:
                self._draw_pairplot(numeric)
                n_pages = max(1, math.ceil(n / self._PAGE_SIZE))
                self._set_page_nav_visible(n_pages > 1)
                self._update_page_nav(n)
            elif use_parallel:
                self._draw_parallel(numeric)
                self._set_page_nav_visible(False)
            else:
                self._draw_marginals(numeric)
                # Enable page navigation for marginals when there are many params
                n_marginal_pages = max(1, math.ceil(len(numeric) / self._PAGE_SIZE))
                self._set_page_nav_visible(n_marginal_pages > 1)
                self._update_page_nav(len(numeric))

        # constrained_layout (set on the Figure at creation) handles layout
        # automatically — no tight_layout() call needed here.

        self._canvas.draw_idle()
        self._placeholder.hide()
        self._canvas.show()

    # ── Colour-mapping helper ──────────────────────────────────────────────

    def _get_colormap(self) -> tuple:
        """
        Returns (ScalarMappable | None, values_array | None).

        The colormap direction: minimize → RdYlGn_r (low=green, high=red)
                                maximize → RdYlGn   (low=red, high=green)
        """
        col = self._obj_combo.currentText()
        obj = next((o for o in self._objectives if o.column_name == col), None)
        if col not in (self._df.columns if self._df is not None else []):
            return None, None

        vals = pd.to_numeric(self._df[col], errors="coerce").values
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            return None, None

        vmin, vmax = finite.min(), finite.max()
        if vmin == vmax:
            vmin -= 1; vmax += 1

        cmap_name = "RdYlGn_r" if (obj and obj.direction == "minimize") else "RdYlGn"
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        sm   = cm.ScalarMappable(cmap=cmap_name, norm=norm)
        sm.set_array(vals)
        return sm, vals

    def _style_ax(self, ax) -> None:
        """Apply dark theme styling to a single Axes."""
        ax.set_facecolor(_AX_BG)
        for spine in ax.spines.values():
            spine.set_color(_GRID)
        ax.tick_params(colors=_FG, labelsize=7)
        ax.xaxis.label.set_color(_FG)
        ax.yaxis.label.set_color(_FG)
        ax.title.set_color(_FG)

    def _add_colorbar(self, sm, ax_or_axes, label: str) -> None:
        cbar = self._fig.colorbar(sm, ax=ax_or_axes, shrink=0.6, pad=0.04, aspect=28)
        cbar.ax.yaxis.set_tick_params(color=_FG, labelsize=7)
        cbar.outline.set_edgecolor(_GRID)

        # Replace Matplotlib's floating "1e6"-style offset text with human-readable
        # suffixes (k / M / G) baked into each tick label.  This avoids the offset
        # text overlapping the rotated axis label on large-valued objectives.
        from matplotlib.ticker import FuncFormatter

        def _human(x, _):
            for mag, suf in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
                if abs(x) >= mag:
                    return f"{x / mag:.3g}{suf}"
            return f"{x:.4g}"

        cbar.ax.yaxis.set_major_formatter(FuncFormatter(_human))

        import matplotlib.pyplot as _plt
        _plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_FG)
        cbar.set_label(label, color=_FG, fontsize=8, labelpad=12)

    # ══════════════════════════════════════════════════════════════════════
    # Plot type 1: Pairwise scatter grid (≤ 5 numeric params)
    # ══════════════════════════════════════════════════════════════════════

    def _draw_pairplot(self, numeric_all: list[ParameterConfig]) -> None:
        from matplotlib.gridspec import GridSpec
        from matplotlib.ticker import FuncFormatter
        import matplotlib.pyplot as _plt

        # ── Apply page slicing ─────────────────────────────────────────────
        n_total = len(numeric_all)
        n_pages = max(1, math.ceil(n_total / self._PAGE_SIZE))
        self._page = max(0, min(self._page, n_pages - 1))
        start   = self._page * self._PAGE_SIZE
        numeric = numeric_all[start : start + self._PAGE_SIZE]
        n       = len(numeric)

        sm, vals = self._get_colormap()
        has_color = sm is not None and vals is not None

        # ── GridSpec: N plot columns + 1 narrow colorbar column ───────────
        # The colorbar gets its own column so it NEVER steals space from the
        # scatter plots.  `tight_layout()` then works correctly.
        if has_color:
            gs = GridSpec(
                n, n + 1, figure=self._fig,
                width_ratios=[1.0] * n + [0.05],
                hspace=0.08, wspace=0.08,
            )
            cax = self._fig.add_subplot(gs[:, n])   # rightmost column = colorbar
        else:
            gs  = GridSpec(n, n, figure=self._fig, hspace=0.08, wspace=0.08)
            cax = None

        # Build 2-D list of Axes
        axes = [[self._fig.add_subplot(gs[i, j]) for j in range(n)]
                for i in range(n)]

        for i, pi in enumerate(numeric):
            xi = pd.to_numeric(self._df[pi.name], errors="coerce").values

            for j, pj in enumerate(numeric):
                ax = axes[i][j]
                self._style_ax(ax)
                yj = pd.to_numeric(self._df[pj.name], errors="coerce").values

                if i == j:
                    # Diagonal — histogram
                    mask = np.isfinite(xi)
                    ax.hist(xi[mask], bins=min(20, max(5, mask.sum() // 3)),
                            color=_HIST, alpha=0.75,
                            edgecolor=_BG, linewidth=0.4)
                    if i == n - 1:
                        ax.set_xlabel(pi.name, fontsize=7)
                    if j == 0:
                        ax.set_ylabel(pi.name, fontsize=7)

                elif i < j:
                    # Upper triangle — hide
                    ax.set_visible(False)

                else:
                    # Lower triangle — scatter
                    mask = np.isfinite(xi) & np.isfinite(yj)
                    if has_color:
                        c_vals = np.where(
                            np.isfinite(vals),
                            vals,
                            np.nanmean(vals[np.isfinite(vals)]),
                        )
                        ax.scatter(
                            yj[mask], xi[mask],
                            c=c_vals[mask],
                            cmap=sm.cmap, norm=sm.norm,
                            s=20, alpha=0.70, linewidths=0,
                        )
                    else:
                        ax.scatter(yj[mask], xi[mask],
                                   c=_HIST, s=20, alpha=0.70, linewidths=0)

                    # Overlay suggestions
                    if self._suggestions:
                        sxi = [s.get(pi.name) for s in self._suggestions]
                        syj = [s.get(pj.name) for s in self._suggestions]
                        valid = [
                            (x, y) for x, y in zip(sxi, syj)
                            if x is not None and y is not None
                        ]
                        if valid:
                            vx, vy = zip(*valid)
                            ax.scatter(
                                vy, vx,
                                marker="*", s=180, c=_SUGGEST,
                                edgecolors="white", linewidths=0.6,
                                zorder=5,
                            )

                    if i == n - 1:
                        ax.set_xlabel(pj.name, fontsize=7)
                    if j == 0:
                        ax.set_ylabel(pi.name, fontsize=7)

        # ── Colorbar in its dedicated GridSpec column ──────────────────────
        if has_color and cax is not None:
            cbar = self._fig.colorbar(sm, cax=cax)
            cbar.ax.yaxis.set_tick_params(color=_FG, labelsize=7)
            cbar.outline.set_edgecolor(_GRID)

            def _human(x, _):
                for mag, suf in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
                    if abs(x) >= mag:
                        return f"{x / mag:.3g}{suf}"
                return f"{x:.4g}"

            cbar.ax.yaxis.set_major_formatter(FuncFormatter(_human))
            _plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_FG)
            cbar.set_label(self._obj_combo.currentText(),
                           color=_FG, fontsize=8, labelpad=10)

        # ── Legend (top-right, inside figure space) ────────────────────────
        legend_handles = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=_HIST, markersize=7,
                   linestyle="None", label="Historical data"),
        ]
        if self._suggestions:
            legend_handles.append(
                Line2D([0], [0], marker="*", color="w",
                       markerfacecolor=_SUGGEST, markersize=12,
                       linestyle="None", label="Suggested")
            )
        # Place legend inside the figure top-right corner.  No bbox_to_anchor
        # is used so constrained_layout can keep it within the figure boundary.
        self._fig.legend(
            handles=legend_handles,
            loc="upper right",
            facecolor=_BG, edgecolor=_GRID, labelcolor=_FG, fontsize=8,
        )

        # ── Suptitle with page info ────────────────────────────────────────
        if n_pages > 1:
            title = (
                f"Pairplot  (page {self._page + 1} / {n_pages}"
                f"  ·  params {start + 1}–{start + n} of {n_total})"
            )
        else:
            title = "Pairplot — Design Space"
        # Constrained layout handles vertical spacing — no y= offset needed
        self._fig.suptitle(title, color=_FG, fontsize=9)
        self._fig.patch.set_facecolor(_BG)

    # ══════════════════════════════════════════════════════════════════════
    # Plot type 2: Parallel coordinates (6–12 numeric params)
    # ══════════════════════════════════════════════════════════════════════

    def _draw_parallel(self, numeric: list[ParameterConfig]) -> None:
        ax = self._fig.add_subplot(111)
        self._style_ax(ax)

        n     = len(numeric)
        xs    = list(range(n))
        names = [p.name for p in numeric]

        # Build raw matrix & normalise each column to [0, 1]
        raw = np.column_stack([
            pd.to_numeric(self._df[p.name], errors="coerce").values
            for p in numeric
        ])  # shape: (n_rows, n_params)

        col_min = np.nanmin(raw, axis=0)
        col_max = np.nanmax(raw, axis=0)
        col_rng = np.where(col_max > col_min, col_max - col_min, 1.0)
        norm_raw = (raw - col_min) / col_rng   # [0, 1]

        sm, vals = self._get_colormap()
        has_color = sm is not None and vals is not None

        # Historical lines
        for row_i in range(len(self._df)):
            y = norm_raw[row_i]
            if np.any(np.isnan(y)):
                continue
            if has_color and np.isfinite(vals[row_i]):
                rgba = sm.to_rgba(vals[row_i], alpha=0.35)
            else:
                rgba = (*mcolors.to_rgb(_HIST), 0.25)
            ax.plot(xs, y, c=rgba, linewidth=0.9)

        # Axis spine lines
        for xi in xs:
            ax.axvline(xi, color=_GRID, linewidth=0.8, zorder=2)

        # Suggested lines
        if self._suggestions:
            suggest_colours = [_SUGGEST, _SUGGEST2, "#cba6f7", "#94e2d5"]
            for s_idx, s in enumerate(self._suggestions):
                sy = []
                for p_idx, p in enumerate(numeric):
                    v = s.get(p.name)
                    if v is not None:
                        sy.append((float(v) - col_min[p_idx]) / col_rng[p_idx])
                    else:
                        sy.append(np.nan)
                col = suggest_colours[s_idx % len(suggest_colours)]
                ax.plot(xs, sy, c=col, linewidth=2.8, alpha=0.95, zorder=5,
                        label=f"Suggestion #{s_idx + 1}")
                for xi, yi in zip(xs, sy):
                    if np.isfinite(yi):
                        ax.scatter(xi, yi, c=col, s=70, zorder=6,
                                   edgecolors="white", linewidths=0.5)

        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=25, ha="right", color=_FG, fontsize=8)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["min", "25%", "50%", "75%", "max"],
                           color=_FG, fontsize=7)
        ax.set_xlim(-0.3, n - 0.7)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Parallel Coordinates — Parameter Space",
                     color=_FG, fontsize=9, pad=6)

        if has_color:
            self._add_colorbar(sm, ax, self._obj_combo.currentText())

        if self._suggestions:
            ax.legend(loc="upper left", facecolor=_BG, edgecolor=_GRID,
                      labelcolor=_FG, fontsize=7)

        self._fig.patch.set_facecolor(_BG)

    # ══════════════════════════════════════════════════════════════════════
    # Plot type 3: 1-D marginal strip (> 12 numeric params)
    # ══════════════════════════════════════════════════════════════════════

    def _draw_marginals(self, numeric_all: list[ParameterConfig]) -> None:
        # ── Page slicing: show at most PAGE_SIZE parameters at a time ─────
        n_total = len(numeric_all)
        n_pages = max(1, math.ceil(n_total / self._PAGE_SIZE))
        self._page = max(0, min(self._page, n_pages - 1))
        start = self._page * self._PAGE_SIZE
        numeric = numeric_all[start : start + self._PAGE_SIZE]
        n = len(numeric)
        axes = self._fig.subplots(n, 1, squeeze=False)
        suggest_colours = [_SUGGEST, _SUGGEST2, "#cba6f7", "#94e2d5"]

        for i, p in enumerate(numeric):
            ax = axes[i][0]
            self._style_ax(ax)

            col_vals = pd.to_numeric(self._df[p.name], errors="coerce").dropna().values
            if len(col_vals) > 1:
                bins = min(25, max(5, len(col_vals) // 5))
                ax.hist(col_vals, bins=bins, color=_HIST, alpha=0.65,
                        edgecolor=_BG, linewidth=0.3)

            # Suggestion vertical lines
            if self._suggestions:
                for s_idx, s in enumerate(self._suggestions):
                    sv = s.get(p.name)
                    if sv is not None:
                        col = suggest_colours[s_idx % len(suggest_colours)]
                        ax.axvline(
                            float(sv), color=col, linewidth=2.0,
                            linestyle="--", alpha=0.9,
                            label=(f"Sug #{s_idx + 1}" if i == 0 else "_nolegend_"),
                        )

            ax.set_ylabel(p.name, fontsize=7, color=_FG,
                          rotation=0, labelpad=65, va="center")
            ax.set_yticks([])

        if self._suggestions:
            axes[0][0].legend(
                loc="upper right", facecolor=_BG,
                edgecolor=_GRID, labelcolor=_FG, fontsize=7,
            )

        axes[-1][0].set_xlabel("Parameter value", color=_FG, fontsize=8)
        if n_total > n:
            _marginal_title = (
                f"Parameter Marginals  (page {self._page + 1}/{n_pages}  ·  "
                f"params {start + 1}–{start + n} of {n_total})"
            )
        else:
            _marginal_title = "Parameter Marginal Distributions"
        self._fig.suptitle(_marginal_title, color=_FG, fontsize=9)
        self._fig.patch.set_facecolor(_BG)

    # ══════════════════════════════════════════════════════════════════════
    # Fallback: no numeric parameters
    # ══════════════════════════════════════════════════════════════════════

    def _draw_no_numeric(self) -> None:
        ax = self._fig.add_subplot(111)
        self._style_ax(ax)
        ax.text(0.5, 0.5,
                "No enabled numeric parameters found.\n"
                "Enable at least one FLOAT or INT parameter to see the design space.",
                ha="center", va="center", color=_FG, fontsize=11,
                transform=ax.transAxes, wrap=True)
        ax.set_axis_off()
        self._fig.patch.set_facecolor(_BG)


# ══════════════════════════════════════════════════════════════════════════════
# Correlation Matrix Widget
# ══════════════════════════════════════════════════════════════════════════════

class CorrelationWidget(QWidget):
    """
    Annotated Pearson / Spearman correlation heatmap for all numeric parameters
    and objective columns.

    Each cell shows the correlation coefficient.  A dashed line visually
    separates the parameter block from the objective block so you can quickly
    see which inputs drive each output.

    Public API
    ----------
    refresh(df, params, objectives)
        Rebuild the heatmap.
    clear()
        Hide the canvas and show the placeholder.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._df: Optional[pd.DataFrame] = None
        self._params: List[ParameterConfig] = []
        self._objectives: List[ObjectiveConfig] = []
        self._needs_redraw: bool = False   # lazy — only render when tab is shown

        self._fig = Figure(facecolor=_BG)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setStyleSheet(f"background-color: {_BG};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── Toolbar ───────────────────────────────────────────────────────
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Correlation method:"))
        self._method_combo = QComboBox()
        self._method_combo.addItems(["Pearson", "Spearman"])
        self._method_combo.setFixedWidth(130)
        self._method_combo.setToolTip(
            "Pearson: linear correlation (sensitive to outliers).\n"
            "Spearman: rank-based (robust to outliers, catches monotonic non-linear)."
        )
        self._method_combo.currentIndexChanged.connect(self._redraw)
        toolbar.addWidget(self._method_combo)
        toolbar.addStretch()
        refresh_btn = QPushButton("⟳  Refresh")
        refresh_btn.setFixedWidth(100)
        refresh_btn.clicked.connect(self._redraw)
        toolbar.addWidget(refresh_btn)
        layout.addLayout(toolbar)

        layout.addWidget(self._canvas, stretch=1)

        self._placeholder = QLabel(
            "Load a CSV and apply objectives to see the correlation matrix."
        )
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet(
            f"color: {_FG}; font-size: 13px; font-style: italic;"
        )
        layout.addWidget(self._placeholder)
        self._canvas.hide()

    # ── Public API ─────────────────────────────────────────────────────────

    def refresh(
        self,
        df: pd.DataFrame,
        params: List[ParameterConfig],
        objectives: List[ObjectiveConfig],
    ) -> None:
        self._df           = df
        self._params       = params
        self._objectives   = objectives
        self._needs_redraw = True   # rendered on demand when tab is selected

    def clear(self) -> None:
        self._df           = None
        self._needs_redraw = False
        self._fig.clear()
        self._canvas.draw_idle()
        self._canvas.hide()
        self._placeholder.show()

    # ── Internal ───────────────────────────────────────────────────────────

    def _redraw(self) -> None:
        if self._df is None:
            return

        # Collect numeric parameter columns + objective columns
        param_cols = [
            p.name for p in self._params
            if p.ptype in (ParameterType.FLOAT, ParameterType.INT)
            and p.name in self._df.columns
        ]
        obj_cols = [
            o.column_name for o in self._objectives
            if o.column_name in self._df.columns
        ]
        # Avoid duplicates (an objective column could share a name with a param)
        all_cols = param_cols + [c for c in obj_cols if c not in param_cols]

        if len(all_cols) < 2:
            self._canvas.hide()
            self._placeholder.setText(
                "Need at least 2 numeric columns to show a correlation matrix."
            )
            self._placeholder.show()
            return

        # Build numeric sub-dataframe
        sub = pd.DataFrame({c: pd.to_numeric(self._df[c], errors="coerce")
                            for c in all_cols})

        method = self._method_combo.currentText().lower()
        corr = sub.corr(method=method)
        n = len(all_cols)

        # ── Draw ───────────────────────────────────────────────────────────
        self._fig.clear()
        # Leave extra right margin for the colorbar
        ax = self._fig.add_axes([0.18, 0.18, 0.62, 0.72])
        ax.set_facecolor(_AX_BG)
        for spine in ax.spines.values():
            spine.set_color(_GRID)

        im = ax.imshow(
            corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto"
        )

        # Colorbar with fixed limits — no need for human formatter here
        cax = self._fig.add_axes([0.83, 0.18, 0.03, 0.72])
        cbar = self._fig.colorbar(im, cax=cax)
        cbar.ax.yaxis.set_tick_params(color=_FG, labelsize=7)
        cbar.outline.set_edgecolor(_GRID)
        import matplotlib.pyplot as _plt
        _plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_FG)
        cbar.set_label(
            f"{self._method_combo.currentText()} r",
            color=_FG, fontsize=8, labelpad=10,
        )

        # Tick labels
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(all_cols, rotation=40, ha="right",
                           fontsize=min(9, max(6, 80 // n)), color=_FG)
        ax.set_yticklabels(all_cols,
                           fontsize=min(9, max(6, 80 // n)), color=_FG)
        ax.tick_params(colors=_FG)

        # Annotate cells
        for i in range(n):
            for j in range(n):
                val = corr.values[i, j]
                if np.isfinite(val):
                    text_col = "white" if abs(val) > 0.6 else _FG
                    fs = max(5, min(9, 72 // n))
                    ax.text(
                        j, i, f"{val:.2f}",
                        ha="center", va="center",
                        fontsize=fs, color=text_col, fontweight="bold",
                    )

        # Dashed separator: parameters vs. objectives
        n_params = len(param_cols)
        if 0 < n_params < n:
            sep = n_params - 0.5
            ax.axhline(sep, color=_FG, linewidth=1.2, linestyle="--", alpha=0.45)
            ax.axvline(sep, color=_FG, linewidth=1.2, linestyle="--", alpha=0.45)
            # Small annotations
            ax.text(-0.6, sep / 2, "params", ha="right", va="center",
                    fontsize=6, color=_FG, alpha=0.7)
            ax.text(-0.6, (sep + n) / 2, "objectives", ha="right", va="center",
                    fontsize=6, color=_FG, alpha=0.7)

        ax.set_title(
            f"{self._method_combo.currentText()} Correlation Matrix  "
            "(dashed line: params | objectives)",
            color=_FG, fontsize=9, pad=8,
        )
        self._fig.patch.set_facecolor(_BG)

        self._canvas.draw_idle()
        self._placeholder.hide()
        self._canvas.show()


# ══════════════════════════════════════════════════════════════════════════════
# Convergence Widget
# ══════════════════════════════════════════════════════════════════════════════

class ConvergenceWidget(QWidget):
    """
    Matplotlib canvas showing best-value-so-far vs. cumulative trial number.

    One step-line per objective, colour-coded.  A dashed horizontal line marks
    the current global best.  For a single objective the legend is suppressed.

    Public API
    ----------
    refresh(study, objectives)
        Rebuild the convergence plot from all COMPLETE trials.
    clear()
        Hide the canvas and show the placeholder text.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._fig = Figure(facecolor=_BG)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setStyleSheet(f"background-color: {_BG};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(self._canvas, stretch=1)

        self._placeholder = QLabel(
            "Run at least one batch to see the convergence plot."
        )
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet(
            f"color: {_FG}; font-size: 13px; font-style: italic;"
        )
        layout.addWidget(self._placeholder)
        self._canvas.hide()

    # ── Public API ─────────────────────────────────────────────────────────

    def refresh(self, study, objectives: List[ObjectiveConfig]) -> None:
        """
        Redraw from all COMPLETE trials in *study*.

        Parameters
        ----------
        study      : optuna.Study — the active study object.
        objectives : list of ObjectiveConfig — used for labels and direction.
        """
        from optuna.trial import TrialState

        completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if not completed:
            self.clear()
            return

        completed.sort(key=lambda t: t.number)

        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.set_facecolor(_AX_BG)
        for spine in ax.spines.values():
            spine.set_color(_GRID)
        ax.tick_params(colors=_FG, labelsize=8)
        ax.xaxis.label.set_color(_FG)
        ax.yaxis.label.set_color(_FG)
        ax.title.set_color(_FG)
        ax.grid(color=_GRID, linewidth=0.5, alpha=0.5)

        obj_colours = [_HIST, _SUGGEST, _PARETO, _SUGGEST2, "#cba6f7", "#94e2d5"]

        for obj_idx, obj in enumerate(objectives):
            direction = obj.direction  # "minimize" | "maximize"
            trial_numbers: list[int] = []
            values: list[float] = []
            for t in completed:
                if t.values is not None and obj_idx < len(t.values):
                    v = t.values[obj_idx]
                    if v is not None and np.isfinite(v):
                        trial_numbers.append(t.number)
                        values.append(float(v))

            if not values:
                continue

            # Running best
            running_best: list[float] = []
            best = float("inf") if direction == "minimize" else float("-inf")
            for v in values:
                best = min(best, v) if direction == "minimize" else max(best, v)
                running_best.append(best)

            col = obj_colours[obj_idx % len(obj_colours)]
            arrow = "\u2193" if direction == "minimize" else "\u2191"
            label = f"{obj.column_name} ({arrow})"

            # ── Individual trial values (semi-transparent dots) ────────────
            # Shown under the running-best line so the user can see the full
            # spread of observations.  This is especially informative for
            # CSV-loaded datasets where the "trial order" is just the CSV row
            # order rather than a true sequential BO trajectory.
            ax.scatter(trial_numbers, values,
                       color=col, s=10, alpha=0.25, zorder=2,
                       linewidths=0)

            ax.step(trial_numbers, running_best, where="post",
                    color=col, linewidth=2.0, label=label, zorder=3)
            ax.scatter(trial_numbers, running_best,
                       color=col, s=18, zorder=4, alpha=0.75)
            # Global-best dashed line
            ax.axhline(running_best[-1], color=col, linewidth=0.8,
                       linestyle="--", alpha=0.45)

        ax.set_xlabel("Trial number", color=_FG, fontsize=9)
        ax.set_ylabel("Best objective value", color=_FG, fontsize=9)
        ax.set_title("Convergence \u2014 Best Value vs. Trial",
                     color=_FG, fontsize=9, pad=6)

        if len(objectives) > 1:
            ax.legend(loc="best", facecolor=_BG, edgecolor=_GRID,
                      labelcolor=_FG, fontsize=8)

        self._fig.patch.set_facecolor(_BG)
        try:
            self._fig.tight_layout()
        except Exception:
            pass

        self._canvas.draw_idle()
        self._placeholder.hide()
        self._canvas.show()

    def clear(self) -> None:
        """Reset to placeholder state."""
        self._fig.clear()
        self._canvas.draw_idle()
        self._canvas.hide()
        self._placeholder.show()


# ══════════════════════════════════════════════════════════════════════════════
# Parameter Importance Widget (Feature 3)
# ══════════════════════════════════════════════════════════════════════════════

class ImportanceWidget(QWidget):
    """
    Horizontal bar chart showing permutation importances from a Random Forest
    fitted on all completed trials.

    Minimum 15 completed rows required (guard is applied before fitting).
    Disabled parameters are excluded automatically.
    For multi-objective studies a QComboBox lets the user pick the target.

    Public API
    ----------
    refresh(df, params, objectives)
        Refit the RF and redraw (called after every batch and on session load).
    clear()
        Hide the canvas and show the placeholder text.
    """

    MIN_TRIALS = 15

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._df: Optional[pd.DataFrame] = None
        self._params: List[ParameterConfig] = []
        self._objectives: List[ObjectiveConfig] = []
        self._needs_redraw: bool = False   # lazy — only compute when tab is shown

        self._fig = Figure(facecolor=_BG)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setStyleSheet(f"background-color: {_BG};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── Toolbar ───────────────────────────────────────────────────────
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Objective:"))
        self._obj_combo = QComboBox()
        self._obj_combo.setFixedWidth(200)
        self._obj_combo.setToolTip(
            "Select which objective to compute feature importances for."
        )
        self._obj_combo.currentIndexChanged.connect(self._redraw)
        toolbar.addWidget(self._obj_combo)
        toolbar.addStretch()
        refresh_btn = QPushButton("⟳  Refresh")
        refresh_btn.setFixedWidth(100)
        refresh_btn.clicked.connect(self._redraw)
        toolbar.addWidget(refresh_btn)
        layout.addLayout(toolbar)

        layout.addWidget(self._canvas, stretch=1)

        self._placeholder = QLabel(
            f"Not enough data — need {self.MIN_TRIALS}+ completed rows."
        )
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet(
            f"color: {_FG}; font-size: 13px; font-style: italic;"
        )
        layout.addWidget(self._placeholder)
        self._canvas.hide()

    # ── Public API ─────────────────────────────────────────────────────────

    def refresh(
        self,
        df: pd.DataFrame,
        params: List[ParameterConfig],
        objectives: List[ObjectiveConfig],
    ) -> None:
        self._df         = df
        self._params     = params
        self._objectives = objectives

        # Sync objective combo
        prev = self._obj_combo.currentText()
        self._obj_combo.blockSignals(True)
        self._obj_combo.clear()
        for o in objectives:
            self._obj_combo.addItem(o.column_name)
        if prev in [o.column_name for o in objectives]:
            self._obj_combo.setCurrentText(prev)
        self._obj_combo.blockSignals(False)

        self._needs_redraw = True   # rendered on demand when tab is selected

    def clear(self) -> None:
        self._df           = None
        self._needs_redraw = False
        self._fig.clear()
        self._canvas.draw_idle()
        self._canvas.hide()
        self._placeholder.setText(
            f"Not enough data — need {self.MIN_TRIALS}+ completed rows."
        )
        self._placeholder.show()

    # ── Internal ───────────────────────────────────────────────────────────

    def _redraw(self) -> None:
        if self._df is None or not self._params or not self._objectives:
            return

        # ── Which objective? ───────────────────────────────────────────────
        obj_col = self._obj_combo.currentText()
        if not obj_col or obj_col not in self._df.columns:
            return

        # ── Enabled numeric + categorical params ───────────────────────────
        enabled_params = [
            p for p in self._params
            if p.enabled and p.name in self._df.columns
            and p.ptype in (ParameterType.FLOAT, ParameterType.INT,
                            ParameterType.CATEGORICAL, ParameterType.BOOL)
        ]
        if not enabled_params:
            self._show_placeholder("No enabled parameters found.")
            return

        param_names = [p.name for p in enabled_params]

        # ── Build feature / target matrices ────────────────────────────────
        sub = self._df[param_names + [obj_col]].copy()
        sub = sub.dropna(subset=[obj_col])   # must have an objective value

        if len(sub) < self.MIN_TRIALS:
            self._show_placeholder(
                f"Not enough data — need {self.MIN_TRIALS}+ rows with "
                f"objective values ({len(sub)} so far)."
            )
            return

        X_df = sub[param_names].copy()
        for col in X_df.columns:
            if X_df[col].dtype == object or pd.api.types.is_string_dtype(X_df[col]):
                X_df[col] = pd.Categorical(X_df[col]).codes
        X = X_df.fillna(-1).values.astype(float)
        y = pd.to_numeric(sub[obj_col], errors="coerce").values

        # Drop rows where y is still NaN after coerce
        valid = np.isfinite(y)
        X, y = X[valid], y[valid]

        if len(y) < self.MIN_TRIALS:
            self._show_placeholder(
                f"Not enough numeric rows — need {self.MIN_TRIALS}+ ({len(y)} so far)."
            )
            return

        # ── Fit RF + permutation importance ────────────────────────────────
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.inspection import permutation_importance

            rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=1)
            rf.fit(X, y)
            result = permutation_importance(
                rf, X, y, n_repeats=10, random_state=42, n_jobs=1
            )
        except Exception as exc:
            self._show_placeholder(f"Importance computation failed:\n{exc}")
            return

        means = result.importances_mean
        stds  = result.importances_std

        # Sort descending
        order = np.argsort(means)[::-1]
        sorted_names  = [param_names[i] for i in order]
        sorted_means  = means[order]
        sorted_stds   = stds[order]

        # ── Draw ───────────────────────────────────────────────────────────
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.set_facecolor(_AX_BG)
        for spine in ax.spines.values():
            spine.set_color(_GRID)
        ax.tick_params(colors=_FG, labelsize=8)
        ax.xaxis.label.set_color(_FG)
        ax.yaxis.label.set_color(_FG)
        ax.title.set_color(_FG)
        ax.grid(axis="x", color=_GRID, linewidth=0.5, alpha=0.5)

        y_pos = np.arange(len(sorted_names))
        bars = ax.barh(
            y_pos, sorted_means,
            xerr=sorted_stds,
            color=_HIST, alpha=0.85,
            error_kw={"ecolor": _FG, "capsize": 3, "linewidth": 1.0},
        )

        # Colour zero/negative bars differently (they are noise)
        for bar, m in zip(bars, sorted_means):
            if m <= 0:
                bar.set_color(_SUGGEST)
                bar.set_alpha(0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names, fontsize=max(7, min(10, 120 // len(sorted_names))))
        ax.invert_yaxis()   # most important at top
        ax.set_xlabel(
            "Mean permutation importance (decrease in R²)",
            color=_FG, fontsize=8,
        )
        obj_cfg = next((o for o in self._objectives if o.column_name == obj_col), None)
        direction_str = f"({obj_cfg.direction})" if obj_cfg else ""
        ax.set_title(
            f"Parameter Importance — {obj_col} {direction_str}\n"
            f"({len(y)} trials, RF + 10-repeat permutation)",
            color=_FG, fontsize=9, pad=6,
        )
        self._fig.patch.set_facecolor(_BG)

        try:
            self._fig.tight_layout()
        except Exception:
            pass

        self._canvas.draw_idle()
        self._placeholder.hide()
        self._canvas.show()

    def _show_placeholder(self, msg: str) -> None:
        self._fig.clear()
        self._canvas.draw_idle()
        self._canvas.hide()
        self._placeholder.setText(msg)
        self._placeholder.show()


# ══════════════════════════════════════════════════════════════════════════════
# Feature 6 — What-If Profiler Widget
# ══════════════════════════════════════════════════════════════════════════════

class ProfilerWidget(QWidget):
    """
    What-If / Prediction Profiler panel.

    Lets the user drag sliders (FLOAT/INT), pick from dropdowns (CATEGORICAL),
    or toggle checkboxes (BOOL) and see a Random Forest surrogate's predicted
    objective value update in real-time.  Analogous to JMP's Prediction Profiler.

    The RF is fit once in ``refresh()`` and cached — it is NOT refit on every
    slider movement.

    Minimum 10 completed rows are required; a placeholder is shown otherwise.

    Public API
    ----------
    refresh(df, params, objectives)
        Refit the RF and rebuild the UI controls.
    clear()
        Reset to placeholder state.
    """

    MIN_TRIALS = 10
    _N_STEPS   = 1000   # slider resolution for FLOAT / INT params

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._df: Optional[pd.DataFrame] = None
        self._params: List[ParameterConfig] = []
        self._objectives: List[ObjectiveConfig] = []

        # Fitted surrogate and encoding state (rebuilt on each refresh)
        self._rf = None
        self._enabled_params: List[ParameterConfig] = []
        self._cat_codes: dict = {}   # param_name → {category_value: int_code}
        self._needs_redraw: bool = False   # lazy — only fit RF when tab is shown

        # Per-row control widgets (rebuilt in _rebuild_controls)
        self._controls: dict = {}    # param_name → primary widget
        self._line_edits: dict = {}  # param_name → QLineEdit (FLOAT/INT only)

        # ── Outer layout ───────────────────────────────────────────────────
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ── Objective selector (shown when there are 2+ objectives) ────────
        obj_row = QHBoxLayout()
        obj_row.addWidget(QLabel("Predict objective:"))
        self._obj_combo = QComboBox()
        self._obj_combo.setFixedWidth(240)
        self._obj_combo.setToolTip("Select which objective the surrogate predicts.")
        self._obj_combo.currentIndexChanged.connect(self._on_objective_changed)
        obj_row.addWidget(self._obj_combo)
        obj_row.addStretch()
        layout.addLayout(obj_row)

        # ── Prediction result label ────────────────────────────────────────
        self._prediction_label = QLabel("Predicted — : —")
        self._prediction_label.setAlignment(Qt.AlignCenter)
        self._prediction_label.setStyleSheet(
            "background-color: #313244; "
            f"color: {_FG}; "
            "font-size: 14px; font-weight: bold; "
            "padding: 10px; border-radius: 6px; "
            "border: 1px solid #45475a;"
        )
        self._prediction_label.setMinimumHeight(42)
        layout.addWidget(self._prediction_label)

        # ── Scroll area containing the parameter controls ──────────────────
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet("QScrollArea { border: none; }")
        self._controls_container = QWidget()
        self._controls_container.setStyleSheet(
            f"background-color: {_BG};"
        )
        self._controls_layout = QGridLayout(self._controls_container)
        self._controls_layout.setContentsMargins(4, 4, 4, 4)
        self._controls_layout.setHorizontalSpacing(8)
        self._controls_layout.setVerticalSpacing(6)
        # Column stretch: name(0) fixed, slider(1) stretches, value(2) fixed
        self._controls_layout.setColumnStretch(1, 1)
        self._scroll.setWidget(self._controls_container)
        layout.addWidget(self._scroll, stretch=1)

        # ── Placeholder (shown when data is insufficient) ──────────────────
        self._placeholder = QLabel(
            f"Collect at least {self.MIN_TRIALS} data points to enable the Profiler."
        )
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet(
            f"color: {_FG}; font-size: 13px; font-style: italic;"
        )
        layout.addWidget(self._placeholder)

        # Initial visibility
        self._scroll.hide()
        self._prediction_label.hide()

    # ══════════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════════

    def refresh(
        self,
        df: pd.DataFrame,
        params: List[ParameterConfig],
        objectives: List[ObjectiveConfig],
    ) -> None:
        """Refit the RF surrogate and rebuild the control widgets."""
        self._df         = df
        self._params     = params
        self._objectives = objectives

        # Sync objective combo (preserve current selection if still valid)
        prev_obj = self._obj_combo.currentText()
        self._obj_combo.blockSignals(True)
        self._obj_combo.clear()
        for o in objectives:
            self._obj_combo.addItem(o.column_name)
        if prev_obj in [o.column_name for o in objectives]:
            self._obj_combo.setCurrentText(prev_obj)
        self._obj_combo.blockSignals(False)

        self._needs_redraw = True   # RF fit deferred until tab is shown

    def clear(self) -> None:
        """Reset to placeholder state."""
        self._df           = None
        self._rf           = None
        self._needs_redraw = False
        self._enabled_params = []
        self._cat_codes  = {}
        self._clear_controls_layout()
        self._scroll.hide()
        self._prediction_label.hide()
        self._placeholder.setText(
            f"Collect at least {self.MIN_TRIALS} data points to enable the Profiler."
        )
        self._placeholder.show()

    # ══════════════════════════════════════════════════════════════════════
    # Internal helpers
    # ══════════════════════════════════════════════════════════════════════

    def _on_objective_changed(self) -> None:
        """Called when the user picks a different objective to predict."""
        self._fit_and_rebuild()

    def _fit_and_rebuild(self) -> None:
        """Fit the RF for the selected objective, then rebuild the control grid."""
        if self._df is None or not self._params or not self._objectives:
            self._show_placeholder(
                f"Collect at least {self.MIN_TRIALS} data points to enable the Profiler."
            )
            return

        obj_col = self._obj_combo.currentText()
        if not obj_col or obj_col not in self._df.columns:
            self._show_placeholder("No valid objective selected.")
            return

        # Collect enabled parameters of supported types
        self._enabled_params = [
            p for p in self._params
            if p.enabled
            and p.name in self._df.columns
            and p.ptype in (
                ParameterType.FLOAT, ParameterType.INT,
                ParameterType.CATEGORICAL, ParameterType.BOOL,
            )
        ]
        if not self._enabled_params:
            self._show_placeholder("No enabled parameters found.")
            return

        param_names = [p.name for p in self._enabled_params]
        sub = self._df[param_names + [obj_col]].copy()
        sub = sub.dropna(subset=[obj_col])

        if len(sub) < self.MIN_TRIALS:
            self._show_placeholder(
                f"Collect at least {self.MIN_TRIALS} data points to enable the Profiler "
                f"({len(sub)} so far)."
            )
            return

        # Build feature matrix (label-encode categoricals/bools)
        X_df = sub[param_names].copy()
        self._cat_codes = {}
        for p in self._enabled_params:
            col = p.name
            if p.ptype == ParameterType.CATEGORICAL:
                cat = pd.Categorical(X_df[col])
                self._cat_codes[col] = {v: i for i, v in enumerate(cat.categories)}
                X_df[col] = cat.codes.astype(float)
            elif p.ptype == ParameterType.BOOL:
                X_df[col] = X_df[col].astype(float)

        X = X_df.fillna(-1).values.astype(float)
        y = pd.to_numeric(sub[obj_col], errors="coerce").values
        valid = np.isfinite(y)
        X, y = X[valid], y[valid]

        if len(y) < self.MIN_TRIALS:
            self._show_placeholder(
                f"Need {self.MIN_TRIALS}+ numeric rows ({len(y)} so far)."
            )
            return

        # Fit RF — cached; NOT refit on slider movement
        try:
            from sklearn.ensemble import RandomForestRegressor
            self._rf = RandomForestRegressor(
                n_estimators=200, random_state=42, n_jobs=1
            )
            self._rf.fit(X, y)
        except Exception as exc:
            self._show_placeholder(f"Surrogate fit failed:\n{exc}")
            return

        # Build the control grid and show it
        self._rebuild_controls(obj_col)
        self._placeholder.hide()
        self._scroll.show()
        self._prediction_label.show()
        # Compute initial prediction with midpoint values
        self._on_params_changed()

    def _clear_controls_layout(self) -> None:
        """Remove every widget from the controls QGridLayout."""
        while self._controls_layout.count():
            item = self._controls_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._controls   = {}
        self._line_edits = {}

    def _rebuild_controls(self, obj_col: str) -> None:
        """Create one row of controls per enabled parameter inside the scroll area."""
        self._clear_controls_layout()

        # Header row
        hdr_name = QLabel("Parameter")
        hdr_name.setStyleSheet(f"color: #89b4fa; font-weight: bold; font-size: 11px;")
        hdr_val  = QLabel("Value")
        hdr_val.setStyleSheet(f"color: #89b4fa; font-weight: bold; font-size: 11px;")
        hdr_val.setAlignment(Qt.AlignRight)
        self._controls_layout.addWidget(hdr_name, 0, 0)
        self._controls_layout.addWidget(hdr_val,  0, 2)

        for row_idx, p in enumerate(self._enabled_params, start=1):
            # Column 0: parameter name label
            name_lbl = QLabel(p.name)
            name_lbl.setStyleSheet(f"color: {_FG}; font-size: 11px;")
            name_lbl.setMinimumWidth(110)
            name_lbl.setMaximumWidth(200)
            name_lbl.setWordWrap(True)
            self._controls_layout.addWidget(name_lbl, row_idx, 0)

            if p.ptype in (ParameterType.FLOAT, ParameterType.INT):
                lo  = float(p.full_min)
                hi  = float(p.full_max)
                rng = hi - lo if hi != lo else 1.0
                init_val = (lo + hi) / 2.0

                # ── Slider (column 1, stretches) ───────────────────────────
                slider = QSlider(Qt.Horizontal)
                slider.setRange(0, self._N_STEPS)
                init_step = max(0, min(self._N_STEPS,
                                       int((init_val - lo) / rng * self._N_STEPS)))
                slider.setValue(init_step)
                slider.setMinimumWidth(80)
                slider.setStyleSheet(
                    "QSlider::groove:horizontal {"
                    "  background: #313244; height: 6px; border-radius: 3px;"
                    "}"
                    "QSlider::handle:horizontal {"
                    "  background: #89b4fa; width: 14px; height: 14px;"
                    "  border-radius: 7px; margin: -4px 0;"
                    "}"
                    "QSlider::sub-page:horizontal {"
                    "  background: #89b4fa; border-radius: 3px;"
                    "}"
                )
                self._controls_layout.addWidget(slider, row_idx, 1)

                # ── Value line-edit (column 2, fixed width) ────────────────
                le = QLineEdit()
                if p.ptype == ParameterType.INT:
                    le.setText(str(int(round(init_val))))
                else:
                    le.setText(f"{init_val:.4f}")
                le.setFixedWidth(90)
                le.setStyleSheet(
                    f"background-color: #313244; color: {_FG}; "
                    "border: 1px solid #45475a; border-radius: 4px; "
                    "padding: 2px 4px; font-size: 11px;"
                )
                self._controls_layout.addWidget(le, row_idx, 2)

                self._controls[p.name]    = slider
                self._line_edits[p.name]  = le

                # ── Wire bidirectional sync ────────────────────────────────
                # Use default-argument capture to avoid closure-over-loop-variable bug
                def _make_slider_handler(param, _slider, _le):
                    def _on_slider(val):
                        _lo  = float(param.full_min)
                        _hi  = float(param.full_max)
                        _rng = _hi - _lo if _hi != _lo else 1.0
                        actual = _lo + (val / self._N_STEPS) * _rng
                        if param.ptype == ParameterType.INT:
                            actual = round(actual)
                            _le.blockSignals(True)
                            _le.setText(str(int(actual)))
                            _le.blockSignals(False)
                        else:
                            _le.blockSignals(True)
                            _le.setText(f"{actual:.4f}")
                            _le.blockSignals(False)
                        self._on_params_changed()
                    return _on_slider

                def _make_le_handler(param, _slider, _le):
                    def _on_le_finished():
                        _lo  = float(param.full_min)
                        _hi  = float(param.full_max)
                        _rng = _hi - _lo if _hi != _lo else 1.0
                        try:
                            v = float(_le.text())
                        except ValueError:
                            v = (_lo + _hi) / 2.0
                        # Clamp to valid range
                        v = max(_lo, min(_hi, v))
                        if param.ptype == ParameterType.INT:
                            v = float(round(v))
                            _le.blockSignals(True)
                            _le.setText(str(int(v)))
                            _le.blockSignals(False)
                        else:
                            _le.blockSignals(True)
                            _le.setText(f"{v:.4f}")
                            _le.blockSignals(False)
                        step = max(0, min(self._N_STEPS,
                                          int((v - _lo) / _rng * self._N_STEPS)))
                        _slider.blockSignals(True)
                        _slider.setValue(step)
                        _slider.blockSignals(False)
                        self._on_params_changed()
                    return _on_le_finished

                slider.valueChanged.connect(_make_slider_handler(p, slider, le))
                le.editingFinished.connect(_make_le_handler(p, slider, le))

            elif p.ptype == ParameterType.CATEGORICAL:
                choices = p.allowed_choices if p.allowed_choices else p.all_choices
                combo = QComboBox()
                combo.addItems([str(c) for c in choices])
                combo.setStyleSheet(
                    "QComboBox { background-color: #313244; color: #cdd6f4; "
                    "border: 1px solid #45475a; border-radius: 4px; padding: 3px 6px; }"
                )
                combo.currentIndexChanged.connect(self._on_params_changed)
                self._controls_layout.addWidget(combo, row_idx, 1, 1, 2)
                self._controls[p.name] = combo

            elif p.ptype == ParameterType.BOOL:
                cb = QCheckBox()
                cb.setChecked(True)
                cb.toggled.connect(self._on_params_changed)
                self._controls_layout.addWidget(cb, row_idx, 1)
                self._controls[p.name] = cb

        # Push rows to top — add a stretch row below the last parameter
        self._controls_layout.setRowStretch(len(self._enabled_params) + 1, 1)

        # Update the prediction label header text
        obj_cfg = next(
            (o for o in self._objectives if o.column_name == obj_col), None
        )
        arrow = (" ↓" if obj_cfg and obj_cfg.direction == "minimize"
                 else " ↑" if obj_cfg else "")
        self._prediction_label.setText(
            f"Predicted {obj_col}{arrow}: —"
        )

    def _on_params_changed(self) -> None:
        """Read all current control values, run prediction, update label."""
        if self._rf is None or not self._enabled_params:
            return

        feature_vec: list[float] = []
        for p in self._enabled_params:
            ctrl = self._controls.get(p.name)
            if ctrl is None:
                feature_vec.append(0.0)
                continue

            if p.ptype in (ParameterType.FLOAT, ParameterType.INT):
                le = self._line_edits.get(p.name)
                try:
                    v = float(le.text()) if le else p.full_min
                except ValueError:
                    v = float(p.full_min)
                v = max(float(p.full_min), min(float(p.full_max), v))
                if p.ptype == ParameterType.INT:
                    v = float(round(v))
                feature_vec.append(v)

            elif p.ptype == ParameterType.CATEGORICAL:
                chosen   = ctrl.currentText()
                code_map = self._cat_codes.get(p.name, {})
                feature_vec.append(float(code_map.get(chosen, 0)))

            elif p.ptype == ParameterType.BOOL:
                feature_vec.append(1.0 if ctrl.isChecked() else 0.0)

        try:
            pred = float(self._rf.predict([feature_vec])[0])
        except Exception:
            return

        obj_col = self._obj_combo.currentText()
        obj_cfg = next(
            (o for o in self._objectives if o.column_name == obj_col), None
        )
        arrow = (" ↓" if obj_cfg and obj_cfg.direction == "minimize"
                 else " ↑" if obj_cfg else "")
        self._prediction_label.setText(
            f"Predicted {obj_col}{arrow}: {pred:.4g}"
        )

    def _show_placeholder(self, msg: str) -> None:
        self._clear_controls_layout()
        self._scroll.hide()
        self._prediction_label.hide()
        self._placeholder.setText(msg)
        self._placeholder.show()


# ══════════════════════════════════════════════════════════════════════════════
# Stand-alone dialog wrapper
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# Pareto Front Scatter Widget (Feature 7)
# ══════════════════════════════════════════════════════════════════════════════

class ParetoWidget(QWidget):
    """
    2D scatter plot of Objective A vs Objective B for multi-objective studies.

    Pareto-optimal points are highlighted with coloured stars; dominated points
    are shown in grey.  A step-line traces the Pareto front sorted by obj_x.

    For 3+ objectives a toolbar lets the user pick which pair to display.

    Public API
    ----------
    refresh(study, objectives)
        Rebuild the plot from all COMPLETE trials.
    clear()
        Hide the canvas and show the placeholder text.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._study = None
        self._objectives: List[ObjectiveConfig] = []

        self._fig = Figure(facecolor=_BG)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setStyleSheet(f"background-color: {_BG};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── Toolbar (objective-pair selectors, shown only for 3+ objectives) ─
        toolbar_layout = QHBoxLayout()
        toolbar_layout.addWidget(QLabel("X-axis:"))
        self._x_combo = QComboBox()
        self._x_combo.setFixedWidth(160)
        self._x_combo.currentIndexChanged.connect(self._redraw)
        toolbar_layout.addWidget(self._x_combo)
        toolbar_layout.addWidget(QLabel("  Y-axis:"))
        self._y_combo = QComboBox()
        self._y_combo.setFixedWidth(160)
        self._y_combo.currentIndexChanged.connect(self._redraw)
        toolbar_layout.addWidget(self._y_combo)
        toolbar_layout.addStretch()
        refresh_btn = QPushButton("⟳  Refresh")
        refresh_btn.setFixedWidth(100)
        refresh_btn.clicked.connect(self._redraw)
        toolbar_layout.addWidget(refresh_btn)

        # Wrap toolbar in a widget so it can be shown/hidden cleanly
        self._toolbar_widget = QWidget()
        self._toolbar_widget.setLayout(toolbar_layout)
        self._toolbar_widget.hide()
        layout.addWidget(self._toolbar_widget)

        layout.addWidget(self._canvas, stretch=1)

        self._placeholder = QLabel(
            "Pareto plot requires 2+ objectives.\n"
            "Load a multi-objective study to see the Pareto front."
        )
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet(
            f"color: {_FG}; font-size: 13px; font-style: italic;"
        )
        layout.addWidget(self._placeholder)
        self._canvas.hide()

    # ── Public API ─────────────────────────────────────────────────────────

    def refresh(self, study, objectives: List[ObjectiveConfig]) -> None:
        """
        Rebuild the Pareto scatter from all COMPLETE trials.

        Parameters
        ----------
        study      : optuna.Study — the active study object.
        objectives : list of ObjectiveConfig.
        """
        self._study = study
        self._objectives = objectives

        if len(objectives) < 2:
            self._fig.clear()
            self._canvas.draw_idle()
            self._canvas.hide()
            self._toolbar_widget.hide()
            self._placeholder.setText(
                "Pareto plot requires 2+ objectives.\n"
                "Add a second objective to see the Pareto front."
            )
            self._placeholder.show()
            return

        # Sync combo boxes for 3+ objectives (show toolbar only when needed)
        show_toolbar = len(objectives) > 2
        self._toolbar_widget.setVisible(show_toolbar)
        if show_toolbar:
            prev_x = self._x_combo.currentText()
            prev_y = self._y_combo.currentText()
            self._x_combo.blockSignals(True)
            self._y_combo.blockSignals(True)
            self._x_combo.clear()
            self._y_combo.clear()
            for o in objectives:
                self._x_combo.addItem(o.column_name)
                self._y_combo.addItem(o.column_name)
            # Restore previous selection if still valid; else default to 0/1
            if prev_x in [o.column_name for o in objectives]:
                self._x_combo.setCurrentText(prev_x)
            else:
                self._x_combo.setCurrentIndex(0)
            if prev_y in [o.column_name for o in objectives]:
                self._y_combo.setCurrentText(prev_y)
            else:
                self._y_combo.setCurrentIndex(min(1, len(objectives) - 1))
            self._x_combo.blockSignals(False)
            self._y_combo.blockSignals(False)

        self._redraw()

    def clear(self) -> None:
        """Reset to placeholder state."""
        self._study = None
        self._objectives = []
        self._fig.clear()
        self._canvas.draw_idle()
        self._canvas.hide()
        self._toolbar_widget.hide()
        self._placeholder.setText(
            "Pareto plot requires 2+ objectives.\n"
            "Load a multi-objective study to see the Pareto front."
        )
        self._placeholder.show()

    # ── Internal ───────────────────────────────────────────────────────────

    def _redraw(self) -> None:
        if self._study is None or len(self._objectives) < 2:
            return

        from optuna.trial import TrialState
        from optuna_builder import get_pareto_front

        completed = [t for t in self._study.trials if t.state == TrialState.COMPLETE]
        if not completed:
            self._canvas.hide()
            self._placeholder.setText(
                "No completed trials yet.\n"
                "Run at least one batch to see the Pareto scatter."
            )
            self._placeholder.show()
            return

        # Determine which objective indices to plot
        if len(self._objectives) > 2:
            x_name = self._x_combo.currentText()
            y_name = self._y_combo.currentText()
            x_idx = next(
                (i for i, o in enumerate(self._objectives) if o.column_name == x_name), 0
            )
            y_idx = next(
                (i for i, o in enumerate(self._objectives) if o.column_name == y_name), 1
            )
        else:
            x_idx, y_idx = 0, 1

        obj_x = self._objectives[x_idx]
        obj_y = self._objectives[y_idx]

        # Extract values for all completed trials
        all_x: list[float] = []
        all_y: list[float] = []
        for t in completed:
            vals = (
                t.values if t.values is not None
                else ([t.value] if t.value is not None else [])
            )
            if len(vals) > max(x_idx, y_idx):
                xv = vals[x_idx]
                yv = vals[y_idx]
                if xv is not None and yv is not None:
                    try:
                        all_x.append(float(xv))
                        all_y.append(float(yv))
                    except (TypeError, ValueError):
                        pass

        if not all_x:
            self._canvas.hide()
            self._placeholder.setText(
                "No valid objective values found for the selected pair."
            )
            self._placeholder.show()
            return

        # Get Pareto-optimal trials
        pareto_trials = get_pareto_front(self._study)
        pareto_x: list[float] = []
        pareto_y: list[float] = []
        for t in pareto_trials:
            vals = (
                t.values if t.values is not None
                else ([t.value] if t.value is not None else [])
            )
            if len(vals) > max(x_idx, y_idx):
                xv = vals[x_idx]
                yv = vals[y_idx]
                if xv is not None and yv is not None:
                    try:
                        pareto_x.append(float(xv))
                        pareto_y.append(float(yv))
                    except (TypeError, ValueError):
                        pass

        # ── Draw ───────────────────────────────────────────────────────────
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.set_facecolor(_AX_BG)
        for spine in ax.spines.values():
            spine.set_color(_GRID)
        ax.tick_params(colors=_FG, labelsize=8)
        ax.xaxis.label.set_color(_FG)
        ax.yaxis.label.set_color(_FG)
        ax.title.set_color(_FG)
        ax.grid(color=_GRID, linewidth=0.5, alpha=0.5)

        # All points (grey, potentially dominated)
        ax.scatter(
            all_x, all_y,
            color="grey", alpha=0.5, s=40, zorder=2,
            label=f"All trials ({len(all_x)})",
        )

        # Pareto-optimal points (green stars)
        if pareto_x:
            ax.scatter(
                pareto_x, pareto_y,
                marker="*", s=120, color=_PARETO,
                edgecolors="white", linewidths=0.5,
                zorder=5, label=f"Pareto-optimal ({len(pareto_x)})",
            )
            # Step-line tracing the Pareto front, sorted by x value
            sorted_pairs = sorted(zip(pareto_x, pareto_y), key=lambda p: p[0])
            sx, sy = zip(*sorted_pairs)
            ax.step(
                sx, sy,
                where="post", color=_PARETO, linewidth=1.5,
                alpha=0.7, linestyle="--", zorder=4,
            )

        # Axis labels include direction arrows
        x_arrow = "↑ maximize" if obj_x.direction == "maximize" else "↓ minimize"
        y_arrow = "↑ maximize" if obj_y.direction == "maximize" else "↓ minimize"
        ax.set_xlabel(f"{obj_x.column_name}  ({x_arrow})", color=_FG, fontsize=9)
        ax.set_ylabel(f"{obj_y.column_name}  ({y_arrow})", color=_FG, fontsize=9)
        ax.set_title("Pareto Front — Trade-off Scatter", color=_FG, fontsize=9, pad=6)

        ax.legend(
            loc="best", facecolor=_BG, edgecolor=_GRID,
            labelcolor=_FG, fontsize=8,
        )

        self._fig.patch.set_facecolor(_BG)
        try:
            self._fig.tight_layout()
        except Exception:
            pass

        self._canvas.draw_idle()
        self._placeholder.hide()
        self._canvas.show()


# ══════════════════════════════════════════════════════════════════════════════
# Dry-run Benchmark Worker + Widget
# ══════════════════════════════════════════════════════════════════════════════

from PySide6.QtCore import Signal as _Signal, QThread as _QThread


class _BenchmarkWorker(_QThread):
    """
    Background thread that runs one Optuna study per sampler on a benchmark
    function and emits convergence data when done.
    """

    progress = _Signal(str)          # human-readable status message
    done     = _Signal(dict)         # {sampler_name: [best_so_far, ...]}
    error    = _Signal(str)

    def __init__(
        self,
        func_name: str,
        n_inputs: int,
        n_objectives: int,
        n_context: int,
        n_trials: int,
        seed: int,
        sampler_names: list,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.func_name     = func_name
        self.n_inputs      = n_inputs
        self.n_objectives  = n_objectives
        self.n_context     = n_context
        self.n_trials      = n_trials
        self.seed          = seed
        self.sampler_names = sampler_names
        self._abort        = False

    def stop(self) -> None:
        self._abort = True

    def run(self) -> None:  # noqa: N802
        try:
            import optuna as _optuna
            from optuna.distributions import FloatDistribution
            from benchmark_math import evaluate, auto_select_benchmark, BENCHMARK_N_OBJ

            _optuna.logging.set_verbosity(_optuna.logging.ERROR)

            func = self.func_name
            if func == "Auto":
                func = auto_select_benchmark(self.n_inputs, self.n_objectives)

            # Effective dimensionality = n_inputs + n_context (context vars ignored by objective)
            d = max(self.n_inputs + self.n_context, 2)
            n_obj = BENCHMARK_N_OBJ.get(func, 1)

            # Build distributions dict
            dists = {f"x{i}": FloatDistribution(0.0, 1.0) for i in range(d)}

            # Objective callable
            def _objective(trial):
                x = [trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(d)]
                result = evaluate(x, func, n_obj=n_obj)
                if isinstance(result, list):
                    return result
                return [result]

            directions = ["minimize"] * n_obj
            results: dict = {}

            for s_name in self.sampler_names:
                if self._abort:
                    return
                self.progress.emit(f"Running {s_name} on {func}…")

                sampler = _make_benchmark_sampler(s_name, self.seed, n_obj)
                study = _optuna.create_study(
                    directions=directions,
                    sampler=sampler,
                )

                convergence: list = []
                best = float("inf") if n_obj == 1 else None
                hv_vals: list = []

                for trial_num in range(self.n_trials):
                    if self._abort:
                        return
                    try:
                        study.optimize(_objective, n_trials=1, show_progress_bar=False)
                        t = study.trials[-1]
                        if t.values:
                            if n_obj == 1:
                                v = t.values[0]
                                if v is not None and v == v:  # not NaN
                                    best = min(best, v)
                                convergence.append(best)
                            else:
                                # Track hypervolume
                                from benchmark_math import compute_hypervolume
                                all_vals = [
                                    tr.values for tr in study.trials
                                    if tr.values and all(v is not None for v in tr.values)
                                ]
                                if all_vals:
                                    hv = compute_hypervolume(all_vals)
                                else:
                                    hv = 0.0
                                convergence.append(hv)
                    except Exception:
                        if convergence:
                            convergence.append(convergence[-1])
                        else:
                            convergence.append(float("nan"))

                results[s_name] = convergence

            self.done.emit(results)

        except Exception as exc:
            self.error.emit(str(exc))


def _make_benchmark_sampler(name: str, seed: int, n_obj: int):
    """
    Return an Optuna sampler object by name.
    NSGAIISampler does not accept a *seed* keyword in all Optuna versions, so
    we try with seed first and fall back to the no-seed constructor.
    """
    import optuna as _optuna
    if name == "TPE":
        return _optuna.samplers.TPESampler(seed=seed)
    elif name == "Random":
        return _optuna.samplers.RandomSampler(seed=seed)
    elif name == "CmaEs" and n_obj == 1:
        return _optuna.samplers.CmaEsSampler(seed=seed)
    elif name == "NSGAii":
        try:
            return _optuna.samplers.NSGAIISampler(seed=seed)
        except TypeError:
            return _optuna.samplers.NSGAIISampler()
    elif name == "QMC":
        try:
            return _optuna.samplers.QMCSampler(seed=seed)
        except TypeError:
            return _optuna.samplers.QMCSampler()
    else:
        return _optuna.samplers.TPESampler(seed=seed)


class BenchmarkWidget(QWidget):
    """
    Dry-run benchmark tab.

    Lets the user configure a synthetic problem (N inputs, N objectives,
    N context), pick a benchmark function, and compare several samplers
    side-by-side via an embedded convergence plot.  No real data required.

    Public API
    ----------
    refresh(n_inputs_hint=4)
        Pre-fill the N-inputs spinbox from the current study (optional).
    clear()
        Reset the canvas to placeholder state.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._worker: Optional[_BenchmarkWorker] = None
        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        from benchmark_math import ALL_BENCHMARKS

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # ── Settings toggle (always visible) ──────────────────────────────
        self._settings_toggle_btn = QPushButton("⚙  Settings  ▸")
        self._settings_toggle_btn.setFixedHeight(24)
        self._settings_toggle_btn.setStyleSheet(
            "QPushButton { text-align: left; padding: 2px 8px; font-size: 11px; }"
        )
        self._settings_toggle_btn.setToolTip("Expand / collapse benchmark configuration.")
        self._settings_toggle_btn.clicked.connect(self._toggle_settings)
        layout.addWidget(self._settings_toggle_btn)

        # ── Controls container (hidden by default) ─────────────────────────
        self._settings_container = QWidget()
        _ctrl_outer = QVBoxLayout(self._settings_container)
        _ctrl_outer.setContentsMargins(0, 0, 0, 0)
        _ctrl_outer.setSpacing(0)

        ctrl_grp = QGroupBox("Benchmark Settings")
        ctrl_form = QGridLayout(ctrl_grp)
        ctrl_form.setSpacing(6)

        row = 0
        ctrl_form.addWidget(QLabel("N inputs:"), row, 0)
        self._n_inputs_spin = QSpinBox()
        self._n_inputs_spin.setRange(2, 20)
        self._n_inputs_spin.setValue(4)
        self._n_inputs_spin.setToolTip("Number of controllable input parameters.")
        ctrl_form.addWidget(self._n_inputs_spin, row, 1)

        ctrl_form.addWidget(QLabel("N objectives:"), row, 2)
        self._n_obj_spin = QSpinBox()
        self._n_obj_spin.setRange(1, 3)
        self._n_obj_spin.setValue(1)
        self._n_obj_spin.setToolTip(
            "1 → single-objective (Branin/Hartmann/Ackley)\n"
            "2 → bi-objective ZDT1/ZDT2\n"
            "3 → tri-objective DTLZ2"
        )
        self._n_obj_spin.valueChanged.connect(self._on_config_changed)
        ctrl_form.addWidget(self._n_obj_spin, row, 3)

        ctrl_form.addWidget(QLabel("N context:"), row, 4)
        self._n_ctx_spin = QSpinBox()
        self._n_ctx_spin.setRange(0, 5)
        self._n_ctx_spin.setValue(0)
        self._n_ctx_spin.setToolTip(
            "Extra noise dimensions appended to the input vector.\n"
            "Context variables are passed to the benchmark but ignored by the objective."
        )
        ctrl_form.addWidget(self._n_ctx_spin, row, 5)

        row += 1
        ctrl_form.addWidget(QLabel("Benchmark:"), row, 0)
        self._func_combo = QComboBox()
        self._func_combo.addItem("Auto")
        for b in ALL_BENCHMARKS:
            self._func_combo.addItem(b)
        self._func_combo.setToolTip(
            "Auto: chooses based on N inputs / N objectives.\n"
            "Branin: 2-D, 1-objective classic.\n"
            "Hartmann6: 6-D, 1-objective.\n"
            "Ackley: any-D, 1-objective, many local minima.\n"
            "ZDT1/ZDT2: 2-objective, any-D.\n"
            "DTLZ2: 3-objective, any-D."
        )
        ctrl_form.addWidget(self._func_combo, row, 1)

        ctrl_form.addWidget(QLabel("N trials:"), row, 2)
        self._n_trials_spin = QSpinBox()
        self._n_trials_spin.setRange(5, 500)
        self._n_trials_spin.setValue(50)
        self._n_trials_spin.setToolTip("Number of BO trials per sampler.")
        ctrl_form.addWidget(self._n_trials_spin, row, 3)

        ctrl_form.addWidget(QLabel("Seed:"), row, 4)
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 99999)
        self._seed_spin.setValue(42)
        ctrl_form.addWidget(self._seed_spin, row, 5)

        row += 1
        ctrl_form.addWidget(QLabel("Samplers:"), row, 0)
        self._sampler_checks: dict = {}
        sampler_col = 1
        for s in ["TPE", "Random", "CmaEs", "NSGAii", "QMC"]:
            cb = QCheckBox(s)
            cb.setChecked(s in ("TPE", "Random", "QMC"))
            self._sampler_checks[s] = cb
            ctrl_form.addWidget(cb, row, sampler_col)
            sampler_col += 1

        _ctrl_outer.addWidget(ctrl_grp)
        self._settings_container.hide()   # collapsed by default
        layout.addWidget(self._settings_container)

        # ── Run button (always visible, outside collapsible section) ──────
        self._run_btn = QPushButton("▶  Run Benchmark")
        self._run_btn.setMinimumHeight(34)
        self._run_btn.clicked.connect(self._action_run)
        layout.addWidget(self._run_btn)

        # ── Status label ──────────────────────────────────────────────────
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet(f"color: {_FG}; font-size: 11px;")
        layout.addWidget(self._status_lbl)

        # ── Canvas ────────────────────────────────────────────────────────
        self._fig = Figure(facecolor=_BG, layout="constrained")
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setStyleSheet(f"background-color: {_BG};")
        layout.addWidget(self._canvas, stretch=1)

        # ── Placeholder ───────────────────────────────────────────────────
        self._placeholder = QLabel(
            "Configure a benchmark above and click ▶ Run Benchmark."
        )
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet(
            f"color: {_FG}; font-size: 13px; font-style: italic;"
        )
        layout.addWidget(self._placeholder)
        self._canvas.hide()

        self._on_config_changed()

    def _toggle_settings(self) -> None:
        """Show/hide the collapsible settings group box."""
        if self._settings_container.isVisible():
            self._settings_container.hide()
            self._settings_toggle_btn.setText("⚙  Settings  ▸")
        else:
            self._settings_container.show()
            self._settings_toggle_btn.setText("⚙  Settings  ▾")

    def _on_config_changed(self) -> None:
        """Enable/disable CmaEs and NSGAii checkboxes based on N objectives."""
        n_obj = self._n_obj_spin.value()
        self._sampler_checks["CmaEs"].setEnabled(n_obj == 1)
        self._sampler_checks["NSGAii"].setEnabled(n_obj > 1)
        if n_obj > 1:
            self._sampler_checks["CmaEs"].setChecked(False)
        if n_obj == 1:
            self._sampler_checks["NSGAii"].setChecked(False)

    # ── Public API ─────────────────────────────────────────────────────────

    def refresh(self, n_inputs_hint: int = 4) -> None:
        """Pre-fill N inputs from the current study (called by DesignSpaceDialog)."""
        self._n_inputs_spin.setValue(max(2, n_inputs_hint))

    def clear(self) -> None:
        self._fig.clear()
        self._canvas.draw_idle()
        self._canvas.hide()
        self._placeholder.show()
        self._status_lbl.setText("")

    # ── Actions ────────────────────────────────────────────────────────────

    def _action_run(self) -> None:
        # Stop any running worker
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(2000)

        selected = [s for s, cb in self._sampler_checks.items() if cb.isChecked()]
        if not selected:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self, "No Samplers", "Select at least one sampler.")
            return

        self._run_btn.setEnabled(False)
        self._status_lbl.setText("Starting benchmark…")
        self._canvas.hide()
        self._placeholder.hide()

        self._worker = _BenchmarkWorker(
            func_name    = self._func_combo.currentText(),
            n_inputs     = self._n_inputs_spin.value(),
            n_objectives = self._n_obj_spin.value(),
            n_context    = self._n_ctx_spin.value(),
            n_trials     = self._n_trials_spin.value(),
            seed         = self._seed_spin.value(),
            sampler_names= selected,
            parent       = self,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(lambda: self._run_btn.setEnabled(True))
        self._worker.start()

    def _on_progress(self, msg: str) -> None:
        self._status_lbl.setText(msg)

    def _on_error(self, msg: str) -> None:
        self._status_lbl.setText(f"❌  Error: {msg[:120]}")
        self._run_btn.setEnabled(True)

    def _on_done(self, results: dict) -> None:
        """Draw convergence lines for all samplers."""
        from benchmark_math import auto_select_benchmark, BENCHMARK_N_OBJ

        func = self._func_combo.currentText()
        n_obj = self._n_obj_spin.value()
        if func == "Auto":
            func = auto_select_benchmark(self._n_inputs_spin.value(), n_obj)

        is_multi = BENCHMARK_N_OBJ.get(func, 1) > 1
        y_label = "Hypervolume ↑" if is_multi else "Best objective value ↓"

        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.set_facecolor(_AX_BG)
        for spine in ax.spines.values():
            spine.set_color(_GRID)
        ax.tick_params(colors=_FG, labelsize=8)
        ax.grid(color=_GRID, linewidth=0.5, alpha=0.5)

        # ── Determine winner (best final convergence value) ────────────────
        best_s_name: Optional[str] = None
        best_final: Optional[float] = None
        for s_name, conv in results.items():
            if not conv:
                continue
            final = conv[-1]
            if not np.isfinite(final):
                continue
            if best_final is None:
                best_final, best_s_name = final, s_name
            elif is_multi:
                if final > best_final:  # higher HV is better
                    best_final, best_s_name = final, s_name
            else:
                if final < best_final:  # lower value is better
                    best_final, best_s_name = final, s_name

        palette = [_HIST, _SUGGEST, _PARETO, _SUGGEST2, "#cba6f7", "#94e2d5"]
        for idx, (s_name, conv) in enumerate(results.items()):
            if not conv:
                continue
            is_winner = (s_name == best_s_name)
            xs = list(range(1, len(conv) + 1))
            col = palette[idx % len(palette)]
            lw  = 1.6 if is_winner else 0.9
            ax.step(xs, conv, where="post",
                    color=col, linewidth=lw,
                    label=s_name + (" ★" if is_winner else ""),
                    alpha=1.0 if is_winner else 0.75,
                    zorder=4 if is_winner else 3)
            # Small end-point dot
            ax.scatter(xs[-1:], conv[-1:],
                       color=col, s=40 if is_winner else 10,
                       zorder=5, alpha=1.0)
            # Annotate winner's final value
            if is_winner and np.isfinite(conv[-1]):
                ax.annotate(
                    f" {conv[-1]:.4g}",
                    xy=(xs[-1], conv[-1]),
                    color=col, fontsize=8, fontweight="bold",
                    va="center", ha="left",
                    zorder=6,
                )

        n_trials = self._n_trials_spin.value()
        n_inputs = self._n_inputs_spin.value()
        n_ctx    = self._n_ctx_spin.value()
        ctx_str  = f" + {n_ctx} ctx" if n_ctx else ""

        ax.set_xlabel("Trial number", color=_FG, fontsize=9)
        ax.set_ylabel(y_label, color=_FG, fontsize=9)
        ax.set_title(
            f"Benchmark: {func}  ·  d={n_inputs}{ctx_str}  ·  {n_obj} obj  ·  {n_trials} trials",
            color=_FG, fontsize=9, pad=6,
        )
        ax.xaxis.label.set_color(_FG)
        ax.yaxis.label.set_color(_FG)
        if results:
            ax.legend(loc="best", facecolor=_BG, edgecolor=_GRID,
                      labelcolor=_FG, fontsize=8)

        self._fig.patch.set_facecolor(_BG)
        self._canvas.draw_idle()
        self._canvas.show()
        self._placeholder.hide()
        winner_note = f"  ·  Winner: {best_s_name}" if best_s_name else ""
        self._status_lbl.setText(
            f"✅  Done — {len(results)} sampler(s) × {n_trials} trials on {func}{winner_note}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Live Sampler Comparison Worker + Widget
# ══════════════════════════════════════════════════════════════════════════════

class _SamplerCompareWorker(_QThread):
    """Background thread for compare_samplers_on_data()."""

    progress = _Signal(str, int)   # (message, pct 0–100)
    done     = _Signal(dict)
    error    = _Signal(str)

    def __init__(self, study, config, sampler_names, n_sim_trials, seed, parent=None):
        super().__init__(parent)
        self._study        = study
        self._config       = config
        self._samplers     = sampler_names
        self._n_sim        = n_sim_trials
        self._seed         = seed

    def run(self) -> None:  # noqa: N802
        try:
            from surrogate_quality import compare_samplers_on_data

            def _cb(name: str, pct: int) -> None:
                self.progress.emit(f"Comparing {name}…", pct)

            results = compare_samplers_on_data(
                study        = self._study,
                config       = self._config,
                sampler_names= self._samplers,
                n_sim_trials = self._n_sim,
                seed         = self._seed,
                progress_cb  = _cb,
            )
            self.done.emit(results)
        except Exception as exc:
            self.error.emit(str(exc))


class SamplerComparisonWidget(QWidget):
    """
    Live sampler comparison tab.

    Runs a virtual BO loop on the currently loaded real data using a Random
    Forest surrogate and compares multiple Optuna samplers side by side.

    Plots
    -----
    Top  : convergence curves (best value or hypervolume per simulated trial)
    Bottom: bar chart — CV R², Spearman ρ, RMSE per sampler

    One-click switch
    ----------------
    After running, the user selects a sampler in the combo and clicks
    "🔀 Use This Sampler" — emits `sampler_switch_requested(str)`.

    Public API
    ----------
    set_study(study, config)
        Called from DesignSpaceDialog when the active study changes.
    clear()
        Reset to placeholder state.

    Signals
    -------
    sampler_switch_requested(str)   — sampler name chosen by the user
    """

    sampler_switch_requested = _Signal(str)

    _SAMPLER_NAMES = ["TPE", "Random", "CmaEs", "NSGAii", "QMC"]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._study  = None
        self._config = None
        self._worker: Optional[_SamplerCompareWorker] = None
        self._last_results: Optional[dict] = None
        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # ── Settings toggle (always visible) ──────────────────────────────
        self._settings_toggle_btn = QPushButton("⚙  Settings  ▸")
        self._settings_toggle_btn.setFixedHeight(24)
        self._settings_toggle_btn.setStyleSheet(
            "QPushButton { text-align: left; padding: 2px 8px; font-size: 11px; }"
        )
        self._settings_toggle_btn.setToolTip("Expand / collapse comparison configuration.")
        self._settings_toggle_btn.clicked.connect(self._toggle_settings)
        layout.addWidget(self._settings_toggle_btn)

        # ── Controls container (hidden by default) ─────────────────────────
        self._settings_container = QWidget()
        _ctrl_outer = QVBoxLayout(self._settings_container)
        _ctrl_outer.setContentsMargins(0, 0, 0, 0)
        _ctrl_outer.setSpacing(0)

        ctrl_grp = QGroupBox("Comparison Settings")
        ctrl_grid = QGridLayout(ctrl_grp)
        ctrl_grid.setSpacing(6)

        ctrl_grid.addWidget(QLabel("Sim trials:"), 0, 0)
        self._n_sim_spin = QSpinBox()
        self._n_sim_spin.setRange(5, 200)
        self._n_sim_spin.setValue(30)
        self._n_sim_spin.setToolTip(
            "Number of virtual forward BO steps per sampler.\n"
            "Each step asks the sampler for a suggestion, evaluates it\n"
            "with an RF surrogate, and tells the result back."
        )
        ctrl_grid.addWidget(self._n_sim_spin, 0, 1)

        ctrl_grid.addWidget(QLabel("Seed:"), 0, 2)
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 99999)
        self._seed_spin.setValue(42)
        ctrl_grid.addWidget(self._seed_spin, 0, 3)

        ctrl_grid.addWidget(QLabel("Samplers:"), 1, 0)
        self._sampler_checks: dict = {}
        col = 1
        for s in self._SAMPLER_NAMES:
            cb = QCheckBox(s)
            cb.setChecked(s in ("TPE", "Random", "QMC"))
            self._sampler_checks[s] = cb
            ctrl_grid.addWidget(cb, 1, col)
            col += 1

        _ctrl_outer.addWidget(ctrl_grp)
        self._settings_container.hide()   # collapsed by default
        layout.addWidget(self._settings_container)

        # ── Run button (always visible, outside collapsible section) ──────
        self._run_btn = QPushButton("▶  Compare Samplers")
        self._run_btn.setMinimumHeight(34)
        self._run_btn.setEnabled(False)
        self._run_btn.setToolTip(
            "Run a virtual BO comparison using the loaded real data as ground truth.\n"
            "Requires ≥ 10 completed trials in the current study."
        )
        self._run_btn.clicked.connect(self._action_run)
        layout.addWidget(self._run_btn)

        # ── Status ────────────────────────────────────────────────────────
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet(f"color: {_FG}; font-size: 11px;")
        layout.addWidget(self._status_lbl)

        # ── Canvas ────────────────────────────────────────────────────────
        self._fig = Figure(facecolor=_BG, layout="constrained")
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setStyleSheet(f"background-color: {_BG};")
        layout.addWidget(self._canvas, stretch=1)

        # ── Switch row (hidden until results are available) ───────────────
        self._switch_grp = QGroupBox("Switch Active Sampler")
        switch_row = QHBoxLayout(self._switch_grp)
        switch_row.addWidget(QLabel("Use sampler:"))
        self._switch_combo = QComboBox()
        self._switch_combo.setFixedWidth(180)
        switch_row.addWidget(self._switch_combo)
        self._switch_btn = QPushButton("🔀  Use This Sampler")
        self._switch_btn.setFixedWidth(180)
        switch_row.addWidget(self._switch_btn)
        switch_row.addStretch()
        self._switch_grp.hide()
        layout.addWidget(self._switch_grp)

        # ── Placeholder ───────────────────────────────────────────────────
        self._placeholder = QLabel(
            "Load data and apply objectives to enable sampler comparison.\n"
            "Requires ≥ 10 completed trials in the active study."
        )
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet(
            f"color: {_FG}; font-size: 13px; font-style: italic;"
        )
        layout.addWidget(self._placeholder)
        self._canvas.hide()

    def _toggle_settings(self) -> None:
        """Show/hide the collapsible settings group box."""
        if self._settings_container.isVisible():
            self._settings_container.hide()
            self._settings_toggle_btn.setText("⚙  Settings  ▸")
        else:
            self._settings_container.show()
            self._settings_toggle_btn.setText("⚙  Settings  ▾")

    # ── Public API ─────────────────────────────────────────────────────────

    def set_study(self, study, config) -> None:
        """Called from DesignSpaceDialog when the active study/config changes."""
        self._study  = study
        self._config = config

        if study is None or config is None:
            self._run_btn.setEnabled(False)
            self._status_lbl.setText("")
            return

        # Check if there are enough completed trials
        try:
            from optuna.trial import TrialState
            n = len([t for t in study.trials if t.state == TrialState.COMPLETE])
        except Exception:
            n = 0

        if n >= 10:
            self._run_btn.setEnabled(True)
            self._status_lbl.setText(
                f"{n} completed trials available — click ▶ Compare Samplers."
            )
        else:
            self._run_btn.setEnabled(False)
            self._status_lbl.setText(
                f"Need ≥ 10 completed trials ({n} so far)."
            )

    def clear(self) -> None:
        self._study  = None
        self._config = None
        self._last_results = None
        self._run_btn.setEnabled(False)
        self._status_lbl.setText("")
        self._fig.clear()
        self._canvas.draw_idle()
        self._canvas.hide()
        self._switch_grp.hide()
        self._placeholder.setText(
            "Load data and apply objectives to enable sampler comparison.\n"
            "Requires ≥ 10 completed trials in the active study."
        )
        self._placeholder.show()

    # ── Actions ────────────────────────────────────────────────────────────

    def _action_run(self) -> None:
        if self._study is None or self._config is None:
            return

        selected = [s for s, cb in self._sampler_checks.items() if cb.isChecked()]
        if not selected:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self, "No Samplers", "Select at least one sampler.")
            return

        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(2000)

        self._run_btn.setEnabled(False)
        self._switch_grp.hide()
        self._canvas.hide()
        self._placeholder.hide()
        self._status_lbl.setText("Starting comparison…")

        self._worker = _SamplerCompareWorker(
            study        = self._study,
            config       = self._config,
            sampler_names= selected,
            n_sim_trials = self._n_sim_spin.value(),
            seed         = self._seed_spin.value(),
            parent       = self,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(lambda: self._run_btn.setEnabled(True))
        self._worker.start()

    def _on_progress(self, msg: str, pct: int) -> None:
        self._status_lbl.setText(f"{msg}  ({pct}%)")

    def _on_error(self, msg: str) -> None:
        self._status_lbl.setText(f"❌  {msg[:180]}")
        self._run_btn.setEnabled(True)

    def _on_done(self, results: dict) -> None:
        self._last_results = results

        # First sampler's data carries the shared quality metrics + holdout scatter
        first_data = next(iter(results.values()), {})
        y_true  = first_data.get("y_true_holdout")
        y_pred  = first_data.get("y_pred_holdout")
        r2      = first_data.get("r2")
        rmse    = first_data.get("rmse")
        pr      = first_data.get("pearson_r")
        status  = first_data.get("status", "insufficient")

        has_scatter = (y_true is not None and y_pred is not None
                       and len(y_true) >= 2)
        has_metrics = r2 is not None

        # Build figure: top = scatter, bottom = quality bar chart (if available)
        self._fig.clear()
        n_rows = 2 if has_metrics else 1
        axes_list = self._fig.subplots(n_rows, 1)
        if n_rows == 1:
            axes_list = [axes_list]

        # Top panel: predicted vs actual (30% hold-out)
        ax_top = axes_list[0]
        ax_top.set_facecolor(_AX_BG)
        for spine in ax_top.spines.values():
            spine.set_color(_GRID)
        ax_top.tick_params(colors=_FG, labelsize=9)
        ax_top.grid(color=_GRID, linewidth=0.5, alpha=0.5)
        ax_top.xaxis.label.set_color(_FG)
        ax_top.yaxis.label.set_color(_FG)

        if has_scatter:
            from scipy.stats import pearsonr as _pearsonr
            ax_top.scatter(y_true, y_pred, color=_HIST, s=30,
                           alpha=0.72, edgecolors="none", zorder=3)
            lo = float(min(y_true.min(), y_pred.min()))
            hi = float(max(y_true.max(), y_pred.max()))
            pad = (hi - lo) * 0.05 if hi > lo else 1.0
            ax_top.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                        color=_FG, linewidth=0.9, linestyle="--",
                        alpha=0.45, label="Perfect")
            try:
                ho_r, _ = _pearsonr(y_true, y_pred)
            except Exception:
                ho_r = float("nan")
            ho_rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
            ax_top.text(
                0.97, 0.05,
                f"r = {ho_r:.3f}\nRMSE = {ho_rmse:.4g}",
                transform=ax_top.transAxes, ha="right", va="bottom",
                fontsize=8, color=_FG,
                bbox=dict(boxstyle="round,pad=0.3",
                          fc=_AX_BG, ec=_GRID, alpha=0.8),
            )
            obj_name = (
                self._config.objectives[0].column_name
                if self._config and getattr(self._config, "objectives", [])
                else "Objective"
            )
            ax_top.set_xlabel(f"Actual  {obj_name}", fontsize=8)
            ax_top.set_ylabel(f"Predicted  {obj_name}", fontsize=8)
            ax_top.set_title(
                f"Surrogate: Predicted vs Actual  (30% hold-out, n={len(y_true)})",
                color=_FG, fontsize=9, pad=4,
            )
        else:
            ax_top.text(
                0.5, 0.5,
                "Insufficient data for hold-out scatter.\n"
                "Load ≥ 10 completed trials and run the comparison.",
                ha="center", va="center", color=_FG, fontsize=10,
                transform=ax_top.transAxes,
            )
            ax_top.set_axis_off()

        # Bottom panel: surrogate quality bar chart
        if has_metrics and n_rows == 2:
            ax_bar = axes_list[1]
            ax_bar.set_facecolor(_AX_BG)
            for spine in ax_bar.spines.values():
                spine.set_color(_GRID)
            ax_bar.tick_params(colors=_FG, labelsize=8)

            metrics = {}
            if r2   is not None: metrics["CV R\u00b2"]     = float(np.clip(r2,   -1, 1))
            if pr   is not None: metrics["Spearman \u03c1"] = float(np.clip(pr,   -1, 1))
            if rmse is not None: metrics["RMSE (norm)"]     = float(
                np.clip(rmse / (rmse + 1), 0, 1)
            )

            if metrics:
                xs_bar   = list(range(len(metrics)))
                m_vals   = list(metrics.values())
                cols_bar = [
                    _PARETO if v > 0.75 else _SUGGEST2 if v > 0.50 else _SUGGEST
                    for v in m_vals
                ]
                ax_bar.bar(xs_bar, m_vals, color=cols_bar, alpha=0.85, edgecolor=_BG)
                ax_bar.set_xticks(xs_bar)
                ax_bar.set_xticklabels(list(metrics.keys()), color=_FG, fontsize=8)
                ax_bar.set_ylim(0, 1.15)
                for xi, val in zip(xs_bar, m_vals):
                    ax_bar.text(xi, val + 0.02, f"{val:.2f}",
                                ha="center", va="bottom", color=_FG, fontsize=8)
                ax_bar.set_title(
                    f"Surrogate Quality  (status: {status})"
                    "  —  how much to trust the predicted vs actual above",
                    color=_FG, fontsize=8, pad=4,
                )
                ax_bar.xaxis.label.set_color(_FG)
                ax_bar.yaxis.label.set_color(_FG)
                ax_bar.grid(axis="y", color=_GRID, linewidth=0.4, alpha=0.5)

        self._fig.patch.set_facecolor(_BG)
        self._canvas.draw_idle()
        self._canvas.show()
        self._placeholder.hide()

        n_sim = self._n_sim_spin.value()
        self._status_lbl.setText(
            f"Done — {len(results)} sampler(s) × {n_sim} sim trials"
        )


class DesignSpaceDialog(QDialog):
    """
    Non-modal window hosting a tabbed interface:

    * **📊 Design Space** — pairplot / parallel coordinates / marginals
    * **🔗 Correlation Matrix** — annotated Pearson / Spearman heatmap
    * **📈 Importance** — permutation importance bar chart
    * **🔮 Profiler** — what-if prediction profiler (Feature 6)

    Usage
    -----
    ::

        dlg = DesignSpaceDialog(parent=main_window)
        dlg.refresh(df, params, objectives, suggestions=None)
        dlg.show()
        dlg.raise_()
    """

    # Bubbled up from SamplerComparisonWidget
    sampler_switch_requested = _Signal(str)

    def __init__(self, parent: QWidget | None = None,
                 embedded: bool = False) -> None:
        """
        Parameters
        ----------
        embedded : bool
            When True the dialog behaves as a plain QWidget embedded in a
            layout rather than as a floating window.  Bottom buttons
            (Fullscreen, Close) are hidden and no window chrome is shown.
        """
        super().__init__(parent)
        self._embedded = embedded

        if embedded:
            # Embed as a child widget — no title bar, no resize border
            self.setWindowFlags(Qt.Widget)
        else:
            self.setModal(False)
            self.setWindowTitle("Design Space — LabOpt")

            # ── Size: fit within screen ────────────────────────────────────
            screen = (
                QApplication.primaryScreen()
                if QApplication.instance() is not None
                else None
            )
            if screen is not None:
                avail = screen.availableGeometry()
                w = min(avail.width()  - 80, 1200)
                h = min(avail.height() - 80, 820)
            else:
                w, h = 1000, 720
            self.resize(w, h)
            self.setMinimumSize(500, 380)

        # ── Validation state ──────────────────────────────────────────────
        self._validation_results = None   # ValidationResults | None
        self._validation_worker  = None   # ValidationWorker  | None
        self._val_tabs_added     = False  # True once validation tabs injected
        # Canvases created in _ensure_val_tabs() — kept so showEvent() can
        # force a redraw when the dialog is reshown after being hidden.
        self._val_canvases: List[FigureCanvasQTAgg] = []
        # Stored for _extract_xy() and _extract_context()
        self._df: Optional[pd.DataFrame] = None
        self._params: List[ParameterConfig] = []
        self._objectives: List[ObjectiveConfig] = []
        self._study_config = None         # StudyConfig | None

        # ── Layout ────────────────────────────────────────────────────────
        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(6, 6, 6, 6)
        vbox.setSpacing(4)

        self._tabs = QTabWidget()

        self._widget = DesignSpaceWidget(self)
        self._tabs.addTab(self._widget, "Design Space")

        self._corr_widget = CorrelationWidget(self)
        self._tabs.addTab(self._corr_widget, "Correlation Matrix")

        self._importance_widget = ImportanceWidget(self)
        self._tabs.addTab(self._importance_widget, "Parameter Importance")

        self._profiler_widget = ProfilerWidget(self)
        self._tabs.addTab(self._profiler_widget, "Prediction Profiler")

        # ── Convergence + Pareto tabs (always present, updated externally) ──
        # These are at fixed positions 4 and 5 (before any validation tabs).
        self._convergence_widget = ConvergenceWidget(self)
        self._tabs.addTab(self._convergence_widget, "Convergence")

        self._pareto_widget = ParetoWidget(self)
        self._tabs.addTab(self._pareto_widget, "Pareto Front")

        # ── Benchmark + Sampler Compare tabs (positions 6 and 7) ──────────
        self._benchmark_widget = BenchmarkWidget(self)
        self._tabs.addTab(self._benchmark_widget, "Benchmark")

        self._sampler_compare_widget = SamplerComparisonWidget(self)
        self._sampler_compare_widget.sampler_switch_requested.connect(
            self.sampler_switch_requested
        )
        self._tabs.addTab(self._sampler_compare_widget, "Sampler Comparison")

        # ── Tab selector row: compact dropdown replaces the native tab bar ──
        # Hiding the tab bar and using a QComboBox gives a much cleaner look,
        # especially with 8+ tabs that would otherwise overflow horizontally.
        self._tabs.tabBar().hide()
        self._tab_combo = QComboBox()
        self._tab_combo.setMinimumWidth(220)
        self._tab_combo.setToolTip("Select which analysis panel to display.")
        for i in range(self._tabs.count()):
            self._tab_combo.addItem(self._tabs.tabText(i))
        # Bidirectional sync: combo → tabs and tabs → combo
        self._tab_combo.currentIndexChanged.connect(self._tabs.setCurrentIndex)
        self._tabs.currentChanged.connect(self._tab_combo.setCurrentIndex)
        _tab_row = QHBoxLayout()
        _tab_row.addWidget(QLabel("View:"))
        _tab_row.addWidget(self._tab_combo)
        _tab_row.addStretch()
        vbox.addLayout(_tab_row)

        # ── Validation header row (ABOVE the tab widget) ───────────────────
        _val_row = QHBoxLayout()
        self._val_checkbox = QCheckBox(
            "🔬  Run Surrogate Validation (requires ≥ 30 unique samples)")
        self._val_checkbox.setEnabled(False)
        self._val_checkbox.setToolTip(
            "Run a rigorous validation of the surrogate model on the current\n"
            "dataset:\n"
            "  §0  Data quality check\n"
            "  §1  RF + GP hold-out and cross-validated accuracy\n"
            "  §2  Virtual BO benchmark (GP+EI vs RF+EI vs Greedy vs Random)\n"
            "  §3  Expected Improvement marginals per parameter\n"
            "  §5  Pass/fail summary\n\n"
            "Runtime: 3–10 minutes depending on dataset size.\n"
            "Results are shown as new tabs and included in the HTML report."
        )
        self._val_checkbox.toggled.connect(self._on_validation_toggled)
        _val_row.addWidget(self._val_checkbox)
        _val_row.addStretch()

        self._val_progress = QProgressBar()
        self._val_progress.setRange(0, 100)
        self._val_progress.setValue(0)
        self._val_progress.setVisible(False)
        self._val_progress.setFixedHeight(18)
        _val_row.addWidget(self._val_progress)

        self._val_status_lbl = QLabel("")
        self._val_status_lbl.setStyleSheet("font-size: 11px; color: #a6e3a1;")
        self._val_status_lbl.setVisible(False)
        _val_row.addWidget(self._val_status_lbl)

        vbox.addLayout(_val_row)
        vbox.addWidget(self._tabs, stretch=1)

        # ── Fullscreen toggle state ───────────────────────────────────────
        self._is_fullscreen: bool = False
        self._normal_geometry = None   # QRect saved before going fullscreen

        # ── Tab-changed signal for lazy rendering ──────────────────────────
        self._tabs.currentChanged.connect(self._on_tab_selected)

        # ── Bottom row: size grip + fullscreen + close ────────────────────
        # Hidden when embedded in the main window (no floating chrome needed).
        if not embedded:
            bottom_row = QHBoxLayout()
            size_grip = QSizeGrip(self)
            bottom_row.addWidget(size_grip, alignment=Qt.AlignLeft | Qt.AlignBottom)
            bottom_row.addStretch()
            self._fullscreen_btn = QPushButton("⛶  Fullscreen")
            self._fullscreen_btn.setFixedWidth(115)
            self._fullscreen_btn.setToolTip("Expand to fill the screen.")
            self._fullscreen_btn.clicked.connect(self._toggle_fullscreen)
            bottom_row.addWidget(self._fullscreen_btn)
            close_btn = QPushButton("Close")
            close_btn.setFixedWidth(90)
            close_btn.clicked.connect(self.hide)
            bottom_row.addWidget(close_btn)
            vbox.addLayout(bottom_row)
        else:
            # No bottom buttons in embedded mode — fullscreen not applicable.
            self._fullscreen_btn = None

    # ── Fullscreen toggle ─────────────────────────────────────────────────

    def _toggle_fullscreen(self) -> None:
        """Toggle between fullscreen and the pre-fullscreen window size."""
        if not self._is_fullscreen:
            # Save geometry before expanding
            self._normal_geometry = self.geometry()
            screen = QApplication.primaryScreen()
            if screen is not None:
                self.setGeometry(screen.availableGeometry())
            else:
                self.showMaximized()
            self._is_fullscreen = True
            self._fullscreen_btn.setText("⧉  Windowed")
            self._fullscreen_btn.setToolTip("Restore to normal window size.")
        else:
            # Restore saved geometry
            if self._normal_geometry is not None:
                self.setGeometry(self._normal_geometry)
            self._is_fullscreen = False
            self._fullscreen_btn.setText("⛶  Fullscreen")
            self._fullscreen_btn.setToolTip("Expand to fill the screen.")

    # ── Lazy tab rendering ────────────────────────────────────────────────

    def _on_tab_selected(self, index: int) -> None:
        """
        Render the newly selected tab on demand.

        Tabs 1–3 (Correlation, Importance, Profiler) use expensive computations
        (RF fitting, permutation importance) that would freeze the UI if run
        eagerly.  They set _needs_redraw=True in refresh() and are only
        computed here when the user actually clicks the tab.

        Tab 0 (Design Space pairplot) always renders immediately — it is the
        active default and the computation is cheap.
        Tab 4+ (Validation figures) are static matplotlib canvases — no
        recomputation needed.
        """
        if index == 1 and getattr(self._corr_widget, '_needs_redraw', False):
            self._corr_widget._needs_redraw = False
            self._corr_widget._redraw()
        elif index == 2 and getattr(self._importance_widget, '_needs_redraw', False):
            self._importance_widget._needs_redraw = False
            self._importance_widget._redraw()
        elif index == 3 and getattr(self._profiler_widget, '_needs_redraw', False):
            self._profiler_widget._needs_redraw = False
            self._profiler_widget._fit_and_rebuild()

    # ── Public API ─────────────────────────────────────────────────────────

    def refresh(
        self,
        df: pd.DataFrame,
        params: List[ParameterConfig],
        objectives: List[ObjectiveConfig],
        suggestions: Optional[List[dict]] = None,
        study_config=None,
    ) -> None:
        """Refresh all tabs with new data.  Pass study_config for validation."""
        # Detect a meaningful data change → reset any existing validation state
        data_changed = (
            self._df is None
            or len(df) != len(self._df)
            or list(df.columns) != list(self._df.columns)
        )
        if data_changed and (self._validation_results is not None
                             or self._val_tabs_added):
            self._reset_validation_state()

        # Store for validation helpers
        self._df           = df
        self._params       = params
        self._objectives   = objectives
        self._study_config = study_config

        # Enable/disable checkbox based on n_unique
        self._update_val_checkbox_state()

        self._widget.refresh(df, params, objectives, suggestions)
        self._corr_widget.refresh(df, params, objectives)
        self._importance_widget.refresh(df, params, objectives)
        self._profiler_widget.refresh(df, params, objectives)

    def clear(self) -> None:
        """Clear all tabs and reset validation state."""
        self._reset_validation_state()
        self._df           = None
        self._params       = []
        self._objectives   = []
        self._study_config = None
        self._val_checkbox.setEnabled(False)
        self._widget.clear()
        self._corr_widget.clear()
        self._importance_widget.clear()
        self._profiler_widget.clear()

    def set_study(self, study, config) -> None:
        """
        Update the active Optuna study and config.

        Passed through to SamplerComparisonWidget so the user can run a live
        sampler comparison.  Also hints BenchmarkWidget with the number of
        enabled input parameters.
        """
        self._sampler_compare_widget.set_study(study, config)

        # Hint the benchmark widget with how many parameters the study has
        if config is not None:
            try:
                n_inputs = sum(1 for p in config.parameters if p.enabled)
            except Exception:
                n_inputs = 4
            self._benchmark_widget.refresh(n_inputs_hint=max(2, n_inputs))

    def set_validation_results(self, results) -> None:
        """
        Accept validation results from MainWindow and inject tabs (idempotent).

        Called automatically when MainWindow's Results-tab validation completes
        and the Design Space dialog is open (or about to be opened), so the 7
        validation tabs appear here without the user needing to re-run.
        """
        if results is None:
            return
        # If we have different results, reset tabs so they're re-injected
        if self._validation_results is not results:
            if self._val_tabs_added:
                # Keep base 8 tabs (0-7): Design Space, Correlation, Importance,
                # Profiler, Convergence, Pareto, Benchmark, Sampler Compare
                while self._tabs.count() > 8:
                    self._tabs.removeTab(self._tabs.count() - 1)
                self._val_tabs_added = False
            self._validation_results = results
        # Inject tabs (no-op if already done)
        self._ensure_val_tabs()
        # Update checkbox UI to show "Complete"
        self._val_checkbox.blockSignals(True)
        self._val_checkbox.setChecked(True)
        self._val_checkbox.setText("Surrogate Validation — Complete")
        self._val_checkbox.setEnabled(True)
        self._val_checkbox.blockSignals(False)
        self._val_status_lbl.setText(
            f"✅  Complete — {results.n_pass}/{results.n_checks} checks passed"
        )
        self._val_status_lbl.setVisible(True)
        self._val_progress.setValue(100)

    # ── Validation: checkbox state ─────────────────────────────────────────

    def _update_val_checkbox_state(self) -> None:
        """Enable the validation checkbox iff df has ≥ 30 unique numeric rows."""
        if self._df is None or not self._params:
            self._val_checkbox.setEnabled(False)
            return

        numeric_cols = [
            p.name for p in self._params
            if p.enabled
            and p.ptype in (ParameterType.FLOAT, ParameterType.INT)
            and p.name in self._df.columns
        ]
        if not numeric_cols:
            self._val_checkbox.setEnabled(False)
            return

        try:
            n_unique = len(
                self._df.dropna(subset=numeric_cols)
                        .drop_duplicates(subset=numeric_cols)
            )
        except Exception:
            n_unique = len(self._df)

        base_tip = (
            "Run a rigorous validation of the surrogate model on the current\n"
            "dataset:\n"
            "  §0  Data quality check\n"
            "  §1  RF + GP hold-out and cross-validated accuracy\n"
            "  §2  Virtual BO benchmark (GP+EI vs RF+EI vs Greedy vs Random)\n"
            "  §3  Expected Improvement marginals per parameter\n"
            "  §5  Pass/fail summary\n\n"
            "Runtime: 3–10 minutes depending on dataset size.\n"
            "Results are shown as new tabs and included in the HTML report."
        )
        if n_unique >= 30:
            self._val_checkbox.setEnabled(True)
            self._val_checkbox.setToolTip(base_tip)
        else:
            self._val_checkbox.setEnabled(False)
            self._val_checkbox.setToolTip(
                base_tip
                + f"\n\n⚠ Currently disabled: only {n_unique} unique rows "
                  f"(need ≥ 30)."
            )

    # ── Validation: state reset ────────────────────────────────────────────

    def _reset_validation_state(self) -> None:
        """Stop worker, remove validation tabs, reset all state."""
        if self._validation_worker is not None and self._validation_worker.isRunning():
            self._validation_worker.terminate()
            self._validation_worker.wait(3000)
            self._validation_worker = None

        # Remove validation tabs (keep base 8: Design Space, Correlation,
        #                          Importance, Profiler, Convergence, Pareto,
        #                          Benchmark, Sampler Compare)
        if self._val_tabs_added:
            while self._tabs.count() > 8:
                self._tabs.removeTab(self._tabs.count() - 1)
            # Sync the tab-selector dropdown
            if hasattr(self, "_tab_combo"):
                while self._tab_combo.count() > 8:
                    self._tab_combo.removeItem(self._tab_combo.count() - 1)

        self._validation_results = None
        self._validation_worker  = None
        self._val_tabs_added     = False
        self._val_canvases.clear()
        self._val_progress.setVisible(False)
        self._val_status_lbl.setVisible(False)
        if hasattr(self, "_val_checkbox"):
            self._val_checkbox.blockSignals(True)
            self._val_checkbox.setChecked(False)
            self._val_checkbox.setText(
                "🔬  Run Surrogate Validation (requires ≥ 30 unique samples)")
            self._val_checkbox.blockSignals(False)

    # ── Validation: checkbox toggle ────────────────────────────────────────

    def _on_validation_toggled(self, checked: bool) -> None:
        if checked:
            if self._validation_results is not None:
                # Results already exist — just show/switch to tabs
                self._ensure_val_tabs()
                return

            # Extract data arrays
            try:
                X, y, feature_names = self._extract_xy()
            except ValueError as exc:
                QMessageBox.warning(self, "Validation Error",
                                    f"Cannot start validation:\n{exc}")
                self._val_checkbox.setChecked(False)
                return

            ctx_X, ctx_names = self._extract_context()
            direction = (self._objectives[0].direction
                         if self._objectives else "minimize")
            target_name = (self._objectives[0].column_name
                           if self._objectives else "objective")

            # Show progress UI
            self._val_progress.setValue(0)
            self._val_progress.setVisible(True)
            self._val_status_lbl.setText("Starting validation…")
            self._val_status_lbl.setVisible(True)
            self._val_checkbox.setText("🔬  Running Validation — please wait…")
            self._val_checkbox.setEnabled(False)

            # Import lazily to avoid circular imports at module load time
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
            self._validation_worker.progress_updated.connect(self._on_val_progress)
            self._validation_worker.validation_done.connect(self._on_val_done)
            self._validation_worker.error_occurred.connect(self._on_val_error)
            self._validation_worker.start()

        else:
            # Unchecked — stop running worker but keep any existing results
            if self._validation_worker is not None and \
                    self._validation_worker.isRunning():
                self._validation_worker.terminate()
                self._validation_worker.wait(3000)
            self._val_progress.setVisible(False)
            self._val_status_lbl.setVisible(False)
            self._val_checkbox.setText(
                "🔬  Run Surrogate Validation (requires ≥ 30 unique samples)")
            self._val_checkbox.setEnabled(True)

    # ── Validation: worker signal handlers ────────────────────────────────

    def _on_val_progress(self, pct: int, msg: str) -> None:
        self._val_progress.setValue(pct)
        self._val_status_lbl.setText(msg)

    def _on_val_done(self, results) -> None:
        self._validation_results = results
        self._val_progress.setValue(100)
        self._val_status_lbl.setText(
            f"✅  Complete — {results.n_pass}/{results.n_checks} checks passed"
        )
        self._val_checkbox.setText("🔬  Surrogate Validation  ✅  Complete")
        self._val_checkbox.setEnabled(True)
        self._ensure_val_tabs()

    def _on_val_error(self, msg: str) -> None:
        self._val_progress.setVisible(False)
        self._val_status_lbl.setVisible(False)
        self._val_checkbox.setText("Surrogate Validation — Error")
        self._val_checkbox.setEnabled(True)
        QMessageBox.critical(
            self, "Validation Error",
            f"The validation run failed:\n\n{msg[:600]}"
        )

    # ── Validation: inject result tabs ────────────────────────────────────

    def _ensure_val_tabs(self) -> None:
        """Inject validation figures as new tabs (called at most once)."""
        if self._val_tabs_added or self._validation_results is None:
            return
        self._val_tabs_added = True

        r = self._validation_results
        tab_specs = [
            ("s0_data_summary",        "Data Quality"),
            ("s1_predicted_vs_true",   "Surrogate Accuracy"),
            ("s1_residuals",           "Residuals"),
            ("s2_bo_convergence",      "BO Benchmark"),
            ("s2_bo_convergence_norm", "BO (Normalised)"),
            ("s3_ei_marginals",        "EI Marginals"),
            ("s5_passfail",            "Validation Summary"),
        ]
        first_val_idx = self._tabs.count()
        for key, tab_label in tab_specs:
            fig = r.figures.get(key)
            if fig is None:
                continue
            canvas = FigureCanvasQTAgg(fig)
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self._val_canvases.append(canvas)   # track for showEvent redraw
            page = QWidget()
            page_vbox = QVBoxLayout(page)
            page_vbox.setContentsMargins(4, 4, 4, 4)
            page_vbox.addWidget(canvas)
            self._tabs.addTab(page, tab_label)
            self._tab_combo.addItem(tab_label)   # keep dropdown in sync

        # Switch to first validation tab
        if self._tabs.count() > first_val_idx:
            self._tabs.setCurrentIndex(first_val_idx)

    # ── showEvent: force validation canvases to repaint when reshown ──────

    def showEvent(self, event) -> None:  # noqa: N802
        """Schedule a non-blocking repaint of validation canvases when reshown.

        FigureCanvasQTAgg buffers can go stale after hide() + show().  Using
        draw_idle() (non-blocking) rather than draw() prevents the UI from
        freezing if there are many large validation figures.
        """
        super().showEvent(event)
        for canvas in self._val_canvases:
            try:
                canvas.draw_idle()
            except Exception:
                pass

    # ── Validation: data extraction helpers ──────────────────────────────

    def _extract_xy(self):
        """Convert stored df + params + objectives to numpy arrays."""
        if self._df is None:
            raise ValueError("No CSV loaded.")
        if not self._objectives:
            raise ValueError("No objective configured.")

        feature_cols = [
            p.name for p in self._params
            if p.enabled
            and p.ptype in (ParameterType.FLOAT, ParameterType.INT)
            and p.name in self._df.columns
        ]
        if not feature_cols:
            raise ValueError("No enabled numeric feature parameters found.")

        obj_col = self._objectives[0].column_name
        if obj_col not in self._df.columns:
            raise ValueError(f"Objective column '{obj_col}' not found in CSV.")

        sub = self._df[feature_cols + [obj_col]].dropna()
        if len(sub) < 5:
            raise ValueError(
                f"Too few valid rows after removing NaNs ({len(sub)} rows)."
            )

        X = sub[feature_cols].values.astype(float)
        y = sub[obj_col].values.astype(float)
        return X, y, feature_cols

    def _extract_context(self):
        """Extract context variable arrays from stored df + study_config."""
        if self._df is None or self._study_config is None:
            return None, None
        ctx_names = [
            cv.column_name
            for cv in getattr(self._study_config, "context_variables", [])
            if cv.column_name in self._df.columns
        ]
        if not ctx_names:
            return None, None
        ctx_sub = self._df[ctx_names].copy()
        ctx_sub = ctx_sub.fillna(ctx_sub.mean())
        ctx_X = ctx_sub.values.astype(float)
        return ctx_X, ctx_names
