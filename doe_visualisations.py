"""
DoE Visualisations  (Phase 5 — revised)
========================================
Matplotlib-based dialogs for exploring DoE point sets and results.

All dialogs use _PaginatedCanvasDialog which provides:
  • Max 4 plots per page with ◀ / ▶ navigation
  • Fullscreen / maximise button
  • Save PNG button
  • constrained_layout so nothing overflows the figure

Public API
----------
show_pairplot_dialog(doe_df, doe_state, obj_cols, parent=None)
show_main_effects_dialog(doe_df, doe_state, obj_cols, parent=None)
show_parallel_coords_dialog(doe_df, doe_state, obj_cols, parent=None)
show_coverage_metrics_dialog(doe_df, random_ref, parent=None)
show_marginal_uniformity_dialog(doe_df, parent=None)
show_doe_analysis_dialog(doe_df, doe_state, obj_cols, parent=None)
"""
from __future__ import annotations

import math
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from parameter_config import DoEState

# ── Matplotlib backend ────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ── Colour palette ────────────────────────────────────────────────────────────
_BG   = "#1e1e2e"
_FG   = "#cdd6f4"
_BLUE = "#89b4fa"
_RED  = "#f38ba8"
_GRN  = "#a6e3a1"
_YEL  = "#f9e2af"
_GRID = "#313244"

_PLOTS_PER_PAGE = 4   # max subplots shown per page


def _apply_dark_style(fig: Figure) -> None:
    """Apply Catppuccin Mocha dark styling to a Figure."""
    fig.patch.set_facecolor(_BG)
    for ax in fig.get_axes():
        ax.set_facecolor(_BG)
        ax.tick_params(colors=_FG)
        ax.xaxis.label.set_color(_FG)
        ax.yaxis.label.set_color(_FG)
        ax.title.set_color(_FG)
        for spine in ax.spines.values():
            spine.set_edgecolor(_GRID)
        ax.grid(color=_GRID, linewidth=0.5)


# ══════════════════════════════════════════════════════════════════════════════
# _PaginatedCanvasDialog  — reusable paginated matplotlib dialog
# ══════════════════════════════════════════════════════════════════════════════

class _PaginatedCanvasDialog(QDialog):
    """
    A QDialog that renders matplotlib pages via a callback.

    Parameters
    ----------
    title        : Window title.
    n_pages      : Total number of pages (1 = no pagination shown).
    draw_page    : Callable(fig: Figure, page: int) -> None
                   Must fill *fig* with content for the given page index.
                   Called with a fresh figure each redraw.
    parent       : Parent QWidget.
    start_page   : Initial page index (0-based, default 0).
    """

    def __init__(
        self,
        title: str,
        n_pages: int,
        draw_page: Callable[[Figure, int], None],
        parent=None,
        start_page: int = 0,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(720, 540)
        self.resize(960, 700)

        self._n_pages = max(1, n_pages)
        self._page = max(0, min(start_page, self._n_pages - 1))
        self._draw_page = draw_page

        # Fullscreen toggle state
        self._is_fullscreen: bool = False
        self._normal_geometry = None   # QRect saved before going fullscreen

        # Figure + canvas — constrained_layout keeps everything inside bounds
        self._fig = Figure(facecolor=_BG, layout='constrained')
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._build_ui()
        self._redraw()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(4)
        vbox.addWidget(self._canvas, stretch=1)

        btn_row = QHBoxLayout()

        # ── Pagination ────────────────────────────────────────────────────
        self._prev_btn = QPushButton("◀")
        self._prev_btn.setFixedWidth(36)
        self._prev_btn.setToolTip("Previous page")
        self._prev_btn.clicked.connect(self._go_prev)
        btn_row.addWidget(self._prev_btn)

        self._page_lbl = QLabel(self._page_text())
        self._page_lbl.setFixedWidth(100)
        self._page_lbl.setAlignment(Qt.AlignCenter)
        self._page_lbl.setStyleSheet(f"color: {_FG}; font-size: 11px;")
        btn_row.addWidget(self._page_lbl)

        self._next_btn = QPushButton("▶")
        self._next_btn.setFixedWidth(36)
        self._next_btn.setToolTip("Next page")
        self._next_btn.clicked.connect(self._go_next)
        btn_row.addWidget(self._next_btn)

        # Show pagination only if there are multiple pages
        for w in (self._prev_btn, self._page_lbl, self._next_btn):
            w.setVisible(self._n_pages > 1)

        btn_row.addStretch()

        # ── Fullscreen ────────────────────────────────────────────────────
        self._fullscreen_btn = QPushButton("⛶  Fullscreen")
        self._fullscreen_btn.setFixedWidth(115)
        self._fullscreen_btn.setToolTip("Expand to fill the screen.")
        self._fullscreen_btn.clicked.connect(self._toggle_fullscreen)
        btn_row.addWidget(self._fullscreen_btn)

        # ── Save PNG ──────────────────────────────────────────────────────
        save_btn = QPushButton("💾  Save PNG…")
        save_btn.setFixedWidth(120)
        save_btn.clicked.connect(self._save_png)
        btn_row.addWidget(save_btn)

        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(80)
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)

        vbox.addLayout(btn_row)

    # ── Pagination actions ─────────────────────────────────────────────────

    def _page_text(self) -> str:
        return f"Page {self._page + 1} / {self._n_pages}"

    def _go_prev(self) -> None:
        self._page = max(0, self._page - 1)
        self._redraw()

    def _go_next(self) -> None:
        self._page = min(self._n_pages - 1, self._page + 1)
        self._redraw()

    # ── Drawing ────────────────────────────────────────────────────────────

    def _redraw(self) -> None:
        self._fig.clear()
        try:
            self._draw_page(self._fig, self._page)
        except Exception as exc:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Render error:\n{exc}",
                    ha="center", va="center", color=_RED, fontsize=9,
                    transform=ax.transAxes)
            ax.set_axis_off()
        self._canvas.draw_idle()
        # Update pagination state
        self._page_lbl.setText(self._page_text())
        self._prev_btn.setEnabled(self._page > 0)
        self._next_btn.setEnabled(self._page < self._n_pages - 1)

    def _toggle_fullscreen(self) -> None:
        """Toggle between fullscreen and the saved pre-fullscreen window size."""
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import QTimer
        if not self._is_fullscreen:
            # Save current geometry before expanding
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
        QTimer.singleShot(80, self._canvas.draw_idle)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        # Debounce: only redraw once the user stops resizing (150 ms of quiet).
        # Calling draw_idle() for every pixel fired during a drag would saturate
        # the Qt event queue and cause the app to freeze.
        if not hasattr(self, "_resize_timer"):
            from PySide6.QtCore import QTimer
            self._resize_timer = QTimer(self)
            self._resize_timer.setSingleShot(True)
            self._resize_timer.timeout.connect(self._canvas.draw_idle)
        self._resize_timer.start(150)

    # ── Save PNG ───────────────────────────────────────────────────────────

    def _save_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot as PNG", "", "PNG Images (*.png)"
        )
        if path:
            try:
                self._fig.savefig(path, dpi=200, bbox_inches="tight",
                                  facecolor=_BG)
                QMessageBox.information(self, "Saved", f"PNG saved to:\n{path}")
            except Exception as exc:
                QMessageBox.warning(self, "Save Error", str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _param_pages(n_items: int) -> int:
    """Number of pages needed to show *n_items* at _PLOTS_PER_PAGE per page."""
    return max(1, math.ceil(n_items / _PLOTS_PER_PAGE))


def _page_slice(items: list, page: int) -> tuple:
    """Return (sub_list, start_idx) for *page* of *items*."""
    start = page * _PLOTS_PER_PAGE
    return items[start : start + _PLOTS_PER_PAGE], start


def _get_result_colours(doe_state: Optional[DoEState], n_points: int):
    """
    Return (colours, has_results, vmin, vmax, cmap) from doe_state results.
    colours is a list of length n_points.
    """
    has_results = False
    if doe_state is not None and any(r is not None for r in doe_state.results):
        y_vals = [
            r[0] if (r is not None and len(r) > 0) else None
            for r in doe_state.results
        ]
        numeric_y = [v for v in y_vals if v is not None]
        if numeric_y:
            vmin, vmax = min(numeric_y), max(numeric_y)
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap = cm.viridis
            colours = [
                cmap(norm(v)) if v is not None else (0.35, 0.35, 0.35, 0.6)
                for v in y_vals
            ]
            return colours, True, vmin, vmax, cmap
    # Default: all blue
    return [_BLUE] * n_points, False, 0.0, 1.0, cm.viridis


# ══════════════════════════════════════════════════════════════════════════════
# show_pairplot_dialog
# ══════════════════════════════════════════════════════════════════════════════

def show_pairplot_dialog(
    doe_df: pd.DataFrame,
    doe_state: Optional[DoEState],
    obj_cols: List[str],
    parent=None,
) -> None:
    """Paginated pairplot — 4 parameters per page in a lower-triangle grid."""
    if doe_df is None or doe_df.empty:
        QMessageBox.information(parent, "No Data", "No DoE points to display.")
        return

    numeric_cols = doe_df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        QMessageBox.information(parent, "No Numeric Columns",
                                "No numeric parameter columns to plot.")
        return

    n_total = len(numeric_cols)
    n_pages = _param_pages(n_total)
    colours, has_results, vmin, vmax, cmap = _get_result_colours(doe_state, len(doe_df))

    def draw(fig: Figure, page: int) -> None:
        cols_page, start = _page_slice(numeric_cols, page)
        d = len(cols_page)
        if d == 0:
            return
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        fig.suptitle(
            f"Pairplot  ({len(doe_df)} points"
            + (f", coloured by {obj_cols[0]}" if has_results and obj_cols else "")
            + (f"  ·  page {page + 1}/{n_pages}  ·  params {start + 1}–{start + d} of {n_total}"
               if n_pages > 1 else "")
            + ")",
            color=_FG, fontsize=10,
        )
        axes = fig.subplots(d, d)
        if d == 1:
            axes = np.array([[axes]])
        for i in range(d):
            for j in range(d):
                ax = axes[i, j]
                ax.set_facecolor(_BG)
                for spine in ax.spines.values():
                    spine.set_edgecolor(_GRID)
                ax.tick_params(colors=_FG, labelsize=5)
                if i == j:
                    ax.hist(doe_df[cols_page[i]].values, bins=min(8, len(doe_df)),
                            color=_BLUE, alpha=0.7, edgecolor=_BG)
                else:
                    ax.scatter(
                        doe_df[cols_page[j]].values,
                        doe_df[cols_page[i]].values,
                        c=colours, s=18, alpha=0.8, linewidths=0,
                    )
                if i == d - 1:
                    ax.set_xlabel(cols_page[j], color=_FG, fontsize=7)
                else:
                    ax.set_xticklabels([])
                if j == 0:
                    ax.set_ylabel(cols_page[i], color=_FG, fontsize=7)
                else:
                    ax.set_yticklabels([])
        fig.patch.set_facecolor(_BG)
        # Colorbar when results exist
        if has_results and obj_cols:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.5, pad=0.02)
            cbar.set_label(obj_cols[0], color=_FG, fontsize=7)
            cbar.ax.yaxis.set_tick_params(color=_FG, labelsize=6)
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_FG)

    title = "Pairplot" + (" — with results" if has_results else "")
    dlg = _PaginatedCanvasDialog(title, n_pages, draw, parent)
    dlg.exec()
    plt.close("all")


# ══════════════════════════════════════════════════════════════════════════════
# show_main_effects_dialog
# ══════════════════════════════════════════════════════════════════════════════

def show_main_effects_dialog(
    doe_df: pd.DataFrame,
    doe_state: Optional[DoEState],
    obj_cols: List[str],
    parent=None,
) -> None:
    """Paginated main effects — 4 parameters per page in a 2×2 grid of bar charts."""
    if doe_state is None or not obj_cols:
        QMessageBox.information(parent, "No Results",
                                "Enter experiment results first to see main effects.")
        return

    results = doe_state.results
    obj_idx = 0
    rows_idx = [i for i, r in enumerate(results) if r is not None]
    if not rows_idx:
        QMessageBox.information(parent, "No Results",
                                "No objective values entered yet.")
        return

    numeric_cols = doe_df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        QMessageBox.information(parent, "No Numeric Columns",
                                "No numeric parameter columns found.")
        return

    y_arr = np.array([results[i][obj_idx] for i in rows_idx], dtype=float)
    n_total = len(numeric_cols)
    n_pages = _param_pages(n_total)

    def draw(fig: Figure, page: int) -> None:
        cols_page, start = _page_slice(numeric_cols, page)
        n = len(cols_page)
        ncols = min(2, n)
        nrows = math.ceil(n / ncols)
        if n_pages > 1:
            fig.suptitle(
                f"Main Effects — {obj_cols[obj_idx]}  "
                f"(page {page + 1}/{n_pages}  ·  params {start + 1}–{start + n} of {n_total})",
                color=_FG, fontsize=10,
            )
        else:
            fig.suptitle(f"Main Effects — {obj_cols[obj_idx]}",
                         color=_FG, fontsize=10)
        axes_flat = fig.subplots(nrows, ncols).flatten() if n > 1 else [fig.subplots()]
        for pi, col in enumerate(cols_page):
            ax = axes_flat[pi]
            ax.set_facecolor(_BG)
            for spine in ax.spines.values():
                spine.set_edgecolor(_GRID)
            ax.tick_params(colors=_FG, labelsize=7)
            ax.grid(color=_GRID, linewidth=0.4, axis="y")
            x_sub = doe_df[col].values[rows_idx]
            n_bins = min(5, len(set(x_sub)))
            if n_bins >= 2:
                bin_edges = np.percentile(x_sub, np.linspace(0, 100, n_bins + 1))
                bin_edges = np.unique(bin_edges)
                if len(bin_edges) < 2:
                    bin_edges = np.linspace(x_sub.min(), x_sub.max(), n_bins + 1)
                means, centres = [], []
                for k in range(len(bin_edges) - 1):
                    lo_, hi_ = bin_edges[k], bin_edges[k + 1]
                    in_bin = (x_sub >= lo_) & (x_sub <= hi_)
                    if in_bin.any():
                        means.append(float(np.mean(y_arr[in_bin])))
                        centres.append((lo_ + hi_) / 2)
                if centres:
                    ax.bar(range(len(centres)), means, color=_BLUE,
                           alpha=0.8, edgecolor=_BG)
                    ax.set_xticks(range(len(centres)))
                    ax.set_xticklabels([f"{c:.3g}" for c in centres],
                                       rotation=30, ha="right", fontsize=7)
            ax.set_title(col, color=_FG, fontsize=8)
        # Hide unused subplots
        for pi in range(n, len(axes_flat)):
            axes_flat[pi].set_visible(False)
        fig.patch.set_facecolor(_BG)

    dlg = _PaginatedCanvasDialog("Main Effects Plot", n_pages, draw, parent)
    dlg.exec()
    plt.close("all")


# ══════════════════════════════════════════════════════════════════════════════
# show_parallel_coords_dialog
# ══════════════════════════════════════════════════════════════════════════════

def show_parallel_coords_dialog(
    doe_df: pd.DataFrame,
    doe_state: Optional[DoEState],
    obj_cols: List[str],
    parent=None,
) -> None:
    """Parallel coordinates (all params on one page — appropriate for this view)."""
    if doe_state is None or not obj_cols:
        QMessageBox.information(parent, "No Results",
                                "Enter experiment results to show parallel coordinates.")
        return

    results = doe_state.results
    rows_idx = [i for i, r in enumerate(results) if r is not None]
    if not rows_idx:
        QMessageBox.information(parent, "No Results",
                                "No objective values entered yet.")
        return

    numeric_cols = doe_df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        QMessageBox.information(parent, "No Numeric Columns",
                                "No numeric parameter columns found.")
        return

    y_arr = np.array([results[i][0] for i in rows_idx], dtype=float)
    X = doe_df[numeric_cols].values[rows_idx]
    X_norm = np.zeros_like(X, dtype=float)
    for ci in range(len(numeric_cols)):
        lo_, hi_ = X[:, ci].min(), X[:, ci].max()
        rng_ = hi_ - lo_
        X_norm[:, ci] = (X[:, ci] - lo_) / rng_ if rng_ > 0 else 0.5
    vmin2, vmax2 = y_arr.min(), y_arr.max()
    norm2 = mcolors.Normalize(vmin=vmin2, vmax=vmax2)
    cmap2 = cm.viridis

    def draw(fig: Figure, page: int) -> None:
        ax = fig.add_subplot(111)
        ax.set_facecolor(_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(_GRID)
        ax.tick_params(colors=_FG, labelsize=8)
        xs = list(range(len(numeric_cols)))
        for ri in range(len(rows_idx)):
            ax.plot(xs, X_norm[ri, :], c=cmap2(norm2(y_arr[ri])),
                    alpha=0.7, linewidth=1.2)
        ax.set_xticks(xs)
        ax.set_xticklabels(numeric_cols, rotation=30, ha="right",
                           fontsize=8, color=_FG)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(["min", "mid", "max"], fontsize=7, color=_FG)
        ax.set_title(
            f"Parallel Coordinates  ({len(rows_idx)} points)\n"
            f"Lines coloured by {obj_cols[0]}",
            color=_FG, fontsize=10,
        )
        sm2 = plt.cm.ScalarMappable(cmap=cmap2, norm=norm2)
        sm2.set_array([])
        cbar2 = fig.colorbar(sm2, ax=ax, shrink=0.8)
        cbar2.set_label(obj_cols[0], color=_FG, fontsize=8)
        cbar2.ax.yaxis.set_tick_params(color=_FG)
        plt.setp(cbar2.ax.yaxis.get_ticklabels(), color=_FG)
        fig.patch.set_facecolor(_BG)

    dlg = _PaginatedCanvasDialog("Parallel Coordinates", 1, draw, parent)
    dlg.exec()
    plt.close("all")


# ══════════════════════════════════════════════════════════════════════════════
# show_coverage_metrics_dialog
# ══════════════════════════════════════════════════════════════════════════════

def show_coverage_metrics_dialog(
    doe_df: pd.DataFrame,
    random_ref: Optional[dict],
    parent=None,
) -> None:
    """Bar chart comparing DoE coverage metrics vs random baseline."""
    from doe_math import coverage_metrics as _cm

    metrics = _cm(doe_df)
    maximin = metrics.get("maximin_distance") or 0.0
    discr   = metrics.get("discrepancy") or 0.0
    ref_mm  = (random_ref or {}).get("random_maximin_ref") or 0.0
    ref_d   = (random_ref or {}).get("random_discr_ref") or 0.0

    def draw(fig: Figure, page: int) -> None:
        axes = fig.subplots(1, 2)
        fig.suptitle("Space-Filling Coverage Metrics", color=_FG, fontsize=11)
        ax1, ax2 = axes
        for ax in axes:
            ax.set_facecolor(_BG)
            for spine in ax.spines.values():
                spine.set_edgecolor(_GRID)
            ax.tick_params(colors=_FG, labelsize=8)
        # Maximin (higher = better)
        ax1.bar(["DoE", "Random"], [maximin, ref_mm],
                color=[_GRN, _RED], alpha=0.8, edgecolor=_BG)
        ax1.set_title("Maximin Distance\n(higher = better)", color=_FG, fontsize=9)
        ax1.set_ylabel("Min pairwise distance", color=_FG, fontsize=8)
        for v, x in zip([maximin, ref_mm], [0, 1]):
            ax1.text(x, v + 0.001, f"{v:.4f}", ha="center", va="bottom",
                     color=_FG, fontsize=8)
        # Discrepancy (lower = better)
        ax2.bar(["DoE", "Random"], [discr, ref_d],
                color=[_GRN, _RED], alpha=0.8, edgecolor=_BG)
        ax2.set_title("L2 Discrepancy\n(lower = better)", color=_FG, fontsize=9)
        ax2.set_ylabel("L2-star discrepancy", color=_FG, fontsize=8)
        for v, x in zip([discr, ref_d], [0, 1]):
            ax2.text(x, v + 0.0001, f"{v:.4f}", ha="center", va="bottom",
                     color=_FG, fontsize=8)
        fig.patch.set_facecolor(_BG)

    dlg = _PaginatedCanvasDialog("Coverage Metrics vs Random", 1, draw, parent)
    dlg.exec()
    plt.close("all")


# ══════════════════════════════════════════════════════════════════════════════
# show_marginal_uniformity_dialog
# ══════════════════════════════════════════════════════════════════════════════

def show_marginal_uniformity_dialog(
    doe_df: pd.DataFrame,
    parent=None,
) -> None:
    """Paginated marginal uniformity histograms — 4 parameters per page (2×2 grid)."""
    numeric_cols = doe_df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        QMessageBox.information(parent, "No Data", "No numeric columns to plot.")
        return

    n_total = len(numeric_cols)
    n_pages = _param_pages(n_total)

    def draw(fig: Figure, page: int) -> None:
        cols_page, start = _page_slice(numeric_cols, page)
        n = len(cols_page)
        ncols = min(2, n)
        nrows = math.ceil(n / ncols)
        if n_pages > 1:
            fig.suptitle(
                f"Marginal Uniformity  (page {page + 1}/{n_pages}  ·  "
                f"params {start + 1}–{start + n} of {n_total})",
                color=_FG, fontsize=10,
            )
        else:
            fig.suptitle("Marginal Uniformity Check", color=_FG, fontsize=10)
        axes_flat = fig.subplots(nrows, ncols).flatten() if n > 1 else [fig.subplots()]
        for pi, col in enumerate(cols_page):
            ax = axes_flat[pi]
            ax.set_facecolor(_BG)
            for spine in ax.spines.values():
                spine.set_edgecolor(_GRID)
            ax.tick_params(colors=_FG, labelsize=6)
            vals = doe_df[col].dropna().values.astype(float)
            if len(vals) == 0:
                ax.set_visible(False)
                continue
            lo, hi = vals.min(), vals.max()
            rng = hi - lo
            vals_norm = (vals - lo) / rng if rng > 0 else vals * 0 + 0.5
            n_bins = min(8, len(vals))
            counts, edges = np.histogram(vals_norm, bins=n_bins, range=(0, 1))
            expected = len(vals) / n_bins
            centres = (edges[:-1] + edges[1:]) / 2
            bar_cols = [
                _GRN if abs(c - expected) / max(expected, 1) < 0.3
                else _YEL if abs(c - expected) / max(expected, 1) < 0.6
                else _RED for c in counts
            ]
            ax.bar(centres, counts, width=1 / n_bins * 0.9,
                   color=bar_cols, alpha=0.85, edgecolor=_BG)
            ax.axhline(expected, color=_FG, linestyle="--", linewidth=0.8)
            ax.set_title(col, color=_FG, fontsize=8)
        for pi in range(n, len(axes_flat)):
            axes_flat[pi].set_visible(False)
        fig.patch.set_facecolor(_BG)

    dlg = _PaginatedCanvasDialog("Marginal Uniformity Check", n_pages, draw, parent)
    dlg.exec()
    plt.close("all")


# ══════════════════════════════════════════════════════════════════════════════
# show_doe_analysis_dialog  (multi-tab, each tab paginated)
# ══════════════════════════════════════════════════════════════════════════════

def show_doe_analysis_dialog(
    doe_df: pd.DataFrame,
    doe_state: Optional[DoEState],
    obj_cols: List[str],
    parent=None,
) -> None:
    """
    Tabbed analysis dialog with paginated plots in each tab.
    Tabs: Pairplot | Uniformity | Main Effects (if results) | Parallel Coords (if results)
    """
    numeric_cols = doe_df.select_dtypes(include=[np.number]).columns.tolist()
    n_total = len(numeric_cols)
    n_pages_pair = _param_pages(n_total)
    n_pages_unif = _param_pages(n_total)

    has_results = (
        doe_state is not None
        and any(r is not None for r in doe_state.results)
        and bool(obj_cols)
    )
    colours, _hr, vmin, vmax, cmap = _get_result_colours(doe_state, len(doe_df))
    norm_c = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # ── Build dialog ──────────────────────────────────────────────────────
    dlg = QDialog(parent)
    dlg.setWindowTitle("DoE Analysis")
    dlg.setMinimumSize(860, 600)
    dlg.resize(1000, 720)

    vbox = QVBoxLayout(dlg)
    vbox.setContentsMargins(4, 4, 4, 4)
    vbox.setSpacing(4)

    tabs = QTabWidget()
    vbox.addWidget(tabs, stretch=1)

    # Per-tab lazy rendering: renderers are stored here; actual render is
    # deferred until the tab is first selected by the user.
    _tab_renderers: List[Callable] = []
    _tab_rendered:  List[bool]     = []

    # ── Helper: create a paginated tab ────────────────────────────────────
    def _make_tab(tab_title: str, n_pgs: int, draw_fn: Callable) -> None:
        tab_widget = QWidget()
        tab_vbox = QVBoxLayout(tab_widget)
        tab_vbox.setContentsMargins(0, 0, 0, 0)
        tab_vbox.setSpacing(2)

        fig = Figure(facecolor=_BG, layout='constrained')
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        tab_vbox.addWidget(canvas, stretch=1)

        page_holder = [0]

        # Navigation bar (shown only if multiple pages)
        nav_row = QHBoxLayout()
        prev_b = QPushButton("◀")
        prev_b.setFixedWidth(32)
        page_l = QLabel(f"1 / {n_pgs}")
        page_l.setFixedWidth(80)
        page_l.setAlignment(Qt.AlignCenter)
        page_l.setStyleSheet(f"color: {_FG}; font-size: 11px;")
        next_b = QPushButton("▶")
        next_b.setFixedWidth(32)

        def _redraw_tab():
            fig.clear()
            try:
                draw_fn(fig, page_holder[0])
            except Exception:
                pass
            canvas.draw_idle()
            page_l.setText(f"{page_holder[0] + 1} / {n_pgs}")
            prev_b.setEnabled(page_holder[0] > 0)
            next_b.setEnabled(page_holder[0] < n_pgs - 1)

        def _prev():
            page_holder[0] = max(0, page_holder[0] - 1)
            _redraw_tab()

        def _next():
            page_holder[0] = min(n_pgs - 1, page_holder[0] + 1)
            _redraw_tab()

        prev_b.clicked.connect(_prev)
        next_b.clicked.connect(_next)
        nav_row.addWidget(prev_b)
        nav_row.addWidget(page_l)
        nav_row.addWidget(next_b)
        nav_row.addStretch()

        save_b = QPushButton("💾")
        save_b.setFixedWidth(36)

        def _save():
            path, _ = QFileDialog.getSaveFileName(
                dlg, "Save PNG", "", "PNG (*.png)"
            )
            if path:
                try:
                    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=_BG)
                except Exception as exc:
                    QMessageBox.warning(dlg, "Save Error", str(exc))

        save_b.clicked.connect(_save)
        nav_row.addWidget(save_b)

        for w in (prev_b, page_l, next_b):
            w.setVisible(n_pgs > 1)

        tab_vbox.addLayout(nav_row)
        tabs.addTab(tab_widget, tab_title)
        # Register for lazy rendering — do NOT render now (avoids flash on open)
        _tab_renderers.append(_redraw_tab)
        _tab_rendered.append(False)

    # ── Tab 1: Pairplot ───────────────────────────────────────────────────
    if n_total > 0:
        def _draw_pair(fig: Figure, page: int) -> None:
            cols_p, start = _page_slice(numeric_cols, page)
            d = len(cols_p)
            title = "Pairplot" + (
                f"  ·  page {page + 1}/{n_pages_pair}  ·  params {start + 1}–{start + d} of {n_total}"
                if n_pages_pair > 1 else ""
            )
            fig.suptitle(title, color=_FG, fontsize=9)
            axes = fig.subplots(d, d)
            if d == 1:
                axes = np.array([[axes]])
            for i in range(d):
                for j in range(d):
                    ax = axes[i, j]
                    ax.set_facecolor(_BG)
                    for spine in ax.spines.values():
                        spine.set_edgecolor(_GRID)
                    ax.tick_params(colors=_FG, labelsize=5)
                    if i == j:
                        ax.hist(doe_df[cols_p[i]].values, bins=min(8, len(doe_df)),
                                color=_BLUE, alpha=0.7, edgecolor=_BG)
                    else:
                        ax.scatter(doe_df[cols_p[j]].values,
                                   doe_df[cols_p[i]].values,
                                   c=colours, s=16, alpha=0.8, linewidths=0)
                    if i == d - 1:
                        ax.set_xlabel(cols_p[j], color=_FG, fontsize=7)
                    else:
                        ax.set_xticklabels([])
                    if j == 0:
                        ax.set_ylabel(cols_p[i], color=_FG, fontsize=7)
                    else:
                        ax.set_yticklabels([])
            fig.patch.set_facecolor(_BG)
        _make_tab("Pairplot", n_pages_pair, _draw_pair)

    # ── Tab 2: Marginal Uniformity ────────────────────────────────────────
    if n_total > 0:
        def _draw_unif(fig: Figure, page: int) -> None:
            cols_p, start = _page_slice(numeric_cols, page)
            n = len(cols_p)
            ncols = min(2, n)
            nrows = math.ceil(n / ncols)
            if n_pages_unif > 1:
                fig.suptitle(
                    f"Marginal Uniformity  (page {page + 1}/{n_pages_unif}  ·  "
                    f"params {start + 1}–{start + n} of {n_total})",
                    color=_FG, fontsize=9,
                )
            else:
                fig.suptitle("Marginal Uniformity", color=_FG, fontsize=9)
            axes_flat = fig.subplots(nrows, ncols).flatten() if n > 1 else [fig.subplots()]
            for pi, col in enumerate(cols_p):
                ax = axes_flat[pi]
                ax.set_facecolor(_BG)
                for spine in ax.spines.values():
                    spine.set_edgecolor(_GRID)
                ax.tick_params(colors=_FG, labelsize=6)
                vals = doe_df[col].dropna().values.astype(float)
                if len(vals) == 0:
                    ax.set_visible(False)
                    continue
                lo, hi = vals.min(), vals.max()
                rng = hi - lo
                vals_norm = (vals - lo) / rng if rng > 0 else vals * 0 + 0.5
                n_bins = min(8, len(vals))
                counts, edges = np.histogram(vals_norm, bins=n_bins, range=(0, 1))
                expected = len(vals) / n_bins
                centres = (edges[:-1] + edges[1:]) / 2
                bar_cols = [
                    _GRN if abs(c - expected) / max(expected, 1) < 0.3
                    else _YEL if abs(c - expected) / max(expected, 1) < 0.6
                    else _RED for c in counts
                ]
                ax.bar(centres, counts, width=1 / n_bins * 0.9,
                       color=bar_cols, alpha=0.85, edgecolor=_BG)
                ax.axhline(expected, color=_FG, linestyle="--", linewidth=0.8)
                ax.set_title(col, color=_FG, fontsize=8)
            for pi in range(n, len(axes_flat)):
                axes_flat[pi].set_visible(False)
            fig.patch.set_facecolor(_BG)
        _make_tab("Uniformity", n_pages_unif, _draw_unif)

    # ── Tab 3: Main Effects (if results) ─────────────────────────────────
    if has_results and n_total > 0:
        results = doe_state.results
        rows_idx = [i for i, r in enumerate(results) if r is not None]
        y_arr = np.array([results[i][0] for i in rows_idx], dtype=float)
        n_pages_me = _param_pages(n_total)

        def _draw_me(fig: Figure, page: int) -> None:
            cols_p, start = _page_slice(numeric_cols, page)
            n = len(cols_p)
            ncols = min(2, n)
            nrows = math.ceil(n / ncols)
            if n_pages_me > 1:
                fig.suptitle(
                    f"Main Effects — {obj_cols[0]}  "
                    f"(page {page + 1}/{n_pages_me}  ·  params {start + 1}–{start + n} of {n_total})",
                    color=_FG, fontsize=9,
                )
            else:
                fig.suptitle(f"Main Effects — {obj_cols[0]}", color=_FG, fontsize=9)
            axes_flat = fig.subplots(nrows, ncols).flatten() if n > 1 else [fig.subplots()]
            for pi, col in enumerate(cols_p):
                ax = axes_flat[pi]
                ax.set_facecolor(_BG)
                for spine in ax.spines.values():
                    spine.set_edgecolor(_GRID)
                ax.tick_params(colors=_FG, labelsize=7)
                ax.grid(color=_GRID, linewidth=0.4, axis="y")
                x_sub = doe_df[col].values[rows_idx]
                n_bins = min(5, len(set(x_sub)))
                if n_bins >= 2:
                    bin_edges = np.percentile(x_sub, np.linspace(0, 100, n_bins + 1))
                    bin_edges = np.unique(bin_edges)
                    if len(bin_edges) < 2:
                        bin_edges = np.linspace(x_sub.min(), x_sub.max(), n_bins + 1)
                    means, centres = [], []
                    for k in range(len(bin_edges) - 1):
                        lo_, hi_ = bin_edges[k], bin_edges[k + 1]
                        in_bin = (x_sub >= lo_) & (x_sub <= hi_)
                        if in_bin.any():
                            means.append(float(np.mean(y_arr[in_bin])))
                            centres.append((lo_ + hi_) / 2)
                    if centres:
                        ax.bar(range(len(centres)), means, color=_BLUE,
                               alpha=0.8, edgecolor=_BG)
                        ax.set_xticks(range(len(centres)))
                        ax.set_xticklabels([f"{c:.3g}" for c in centres],
                                           rotation=30, ha="right", fontsize=7)
                ax.set_title(col, color=_FG, fontsize=8)
            for pi in range(n, len(axes_flat)):
                axes_flat[pi].set_visible(False)
            fig.patch.set_facecolor(_BG)
        _make_tab("Main Effects", n_pages_me, _draw_me)

    # ── Tab 4: Parallel Coordinates (if results) ──────────────────────────
    if has_results and n_total > 0:
        results = doe_state.results
        rows_idx2 = [i for i, r in enumerate(results) if r is not None]
        y_arr2 = np.array([results[i][0] for i in rows_idx2], dtype=float)
        X = doe_df[numeric_cols].values[rows_idx2]
        X_norm = np.zeros_like(X, dtype=float)
        for ci in range(len(numeric_cols)):
            lo_, hi_ = X[:, ci].min(), X[:, ci].max()
            rng_ = hi_ - lo_
            X_norm[:, ci] = (X[:, ci] - lo_) / rng_ if rng_ > 0 else 0.5
        norm_pc = mcolors.Normalize(vmin=y_arr2.min(), vmax=y_arr2.max())

        def _draw_pc(fig: Figure, page: int) -> None:
            ax = fig.add_subplot(111)
            ax.set_facecolor(_BG)
            for spine in ax.spines.values():
                spine.set_edgecolor(_GRID)
            ax.tick_params(colors=_FG, labelsize=8)
            xs = list(range(len(numeric_cols)))
            for ri in range(len(rows_idx2)):
                ax.plot(xs, X_norm[ri, :], c=cmap(norm_pc(y_arr2[ri])),
                        alpha=0.7, linewidth=1.2)
            ax.set_xticks(xs)
            ax.set_xticklabels(numeric_cols, rotation=30, ha="right",
                               fontsize=8, color=_FG)
            ax.set_yticks([0, 0.5, 1.0])
            ax.set_yticklabels(["min", "mid", "max"], fontsize=7, color=_FG)
            ax.set_title(
                f"Parallel Coordinates — coloured by {obj_cols[0]}",
                color=_FG, fontsize=9,
            )
            sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=norm_pc)
            sm2.set_array([])
            cbar2 = fig.colorbar(sm2, ax=ax, shrink=0.8)
            cbar2.set_label(obj_cols[0], color=_FG, fontsize=8)
            cbar2.ax.yaxis.set_tick_params(color=_FG)
            plt.setp(cbar2.ax.yaxis.get_ticklabels(), color=_FG)
            fig.patch.set_facecolor(_BG)
        _make_tab("Parallel Coords", 1, _draw_pc)

    # ── Lazy tab rendering: render first tab after dialog shows; others on demand
    def _on_tab_selected(index: int) -> None:
        if 0 <= index < len(_tab_renderers) and not _tab_rendered[index]:
            _tab_rendered[index] = True
            _tab_renderers[index]()

    tabs.currentChanged.connect(_on_tab_selected)

    # ── Bottom row: fullscreen toggle + close ─────────────────────────────
    _fs_state: dict = {"is_fullscreen": False, "normal_geometry": None}

    fs_btn = QPushButton("⛶  Fullscreen")
    fs_btn.setFixedWidth(115)
    fs_btn.setToolTip("Expand to fill the screen.")

    def _toggle_fullscreen() -> None:
        from PySide6.QtWidgets import QApplication as _QApp
        from PySide6.QtCore import QTimer as _QTimer
        if not _fs_state["is_fullscreen"]:
            _fs_state["normal_geometry"] = dlg.geometry()
            screen = _QApp.primaryScreen()
            if screen is not None:
                dlg.setGeometry(screen.availableGeometry())
            else:
                dlg.showMaximized()
            _fs_state["is_fullscreen"] = True
            fs_btn.setText("⧉  Windowed")
            fs_btn.setToolTip("Restore to normal window size.")
        else:
            if _fs_state["normal_geometry"] is not None:
                dlg.setGeometry(_fs_state["normal_geometry"])
            _fs_state["is_fullscreen"] = False
            fs_btn.setText("⛶  Fullscreen")
            fs_btn.setToolTip("Expand to fill the screen.")

    fs_btn.clicked.connect(_toggle_fullscreen)

    btn_row = QHBoxLayout()
    btn_row.addStretch()
    btn_row.addWidget(fs_btn)
    close_b = QPushButton("Close")
    close_b.setFixedWidth(80)
    close_b.clicked.connect(dlg.accept)
    btn_row.addWidget(close_b)
    vbox.addLayout(btn_row)

    # Render the first tab after the event loop starts (avoids flash on open)
    from PySide6.QtCore import QTimer as _QTimer0
    _QTimer0.singleShot(0, lambda: _on_tab_selected(0))

    dlg.exec()
    plt.close("all")
