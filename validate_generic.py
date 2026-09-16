"""
validate_generic.py
===================
Generalised surrogate validation engine — no Qt dependency.

Accepts numpy arrays, runs §0–§5 validation on an arbitrary tabular
dataset, and returns matplotlib Figure objects (not file paths).

Designed to be called from both the GUI worker (ValidationWorker) and
headless scripts.

Public API
----------
ValidationEngine(X, y, feature_names, target_name, ...)
    .run() → ValidationResults

ValidationResults
    .figures : Dict[str, matplotlib.figure.Figure]
    .metrics : Dict[str, Any]
    .passfail : List[Tuple[str, bool, str]]
    .n_pass, .n_checks, .summary
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm as sp_norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RepeatedKFold, train_test_split

warnings.filterwarnings("ignore")

# ── Configuration constants (overridable by caller) ───────────────────────────
MIN_UNIQUE_ROWS    = 30
DEFAULT_SEED       = 42
DEFAULT_N_CV_FOLDS = 5
DEFAULT_N_CV_REPS  = 3
DEFAULT_N_SEED_BO  = 8
DEFAULT_N_BO_ITER  = 25
DEFAULT_N_BO_RUNS  = 6
EI_XI              = 0.01
EI_MARGINAL_PTS    = 80
_TEST_SIZE         = 0.30

# ── Catppuccin Mocha colour scheme ────────────────────────────────────────────
_BG   = "#1e1e2e"   # base background
_AX   = "#181825"   # axes background
_FG   = "#cdd6f4"   # foreground text
_GRID = "#313244"   # grid / spine colour
_ACC  = "#89b4fa"   # accent (blue)
_GRN  = "#a6e3a1"   # green (pass)
_RED  = "#f38ba8"   # red (fail)
_AMB  = "#f9e2af"   # amber (warning)
_CYAN = "#89dceb"   # cyan

# Strategy colours for BO benchmark
_STRAT_COLORS: Dict[str, str] = {
    "GP + EI":  "#89b4fa",   # blue
    "RF + EI":  "#a6e3a1",   # green
    "Greedy":   "#f9e2af",   # amber
    "Random":   "#f38ba8",   # red
}


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationResults:
    figures: Dict[str, "matplotlib.figure.Figure"] = field(default_factory=dict)
    # Recognised figure keys:
    #   "s0_data_summary", "s1_predicted_vs_true", "s1_residuals",
    #   "s2_bo_convergence", "s2_bo_convergence_norm",
    #   "s3_ei_marginals", "s5_passfail"
    metrics:  Dict[str, Any] = field(default_factory=dict)
    passfail: List[Tuple[str, bool, str]] = field(default_factory=list)
    n_pass:   int = 0
    n_checks: int = 0
    summary:  str = "RED"   # "GREEN" | "YELLOW" | "RED"


# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def _style_ax(ax) -> None:
    """Apply Catppuccin Mocha dark theme to a single Axes."""
    ax.set_facecolor(_AX)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    ax.tick_params(colors=_FG, labelsize=8)
    ax.xaxis.label.set_color(_FG)
    ax.yaxis.label.set_color(_FG)
    ax.title.set_color(_FG)
    ax.grid(color=_GRID, linewidth=0.4, alpha=0.5)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return dict with RMSE, MAE, Pearson_r, Spearman_rho."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    try:
        pr, _ = stats.pearsonr(y_true, y_pred)
    except Exception:
        pr = 0.0
    try:
        sr, _ = stats.spearmanr(y_true, y_pred)
    except Exception:
        sr = 0.0
    return {
        "RMSE": rmse, "MAE": mae,
        "Pearson_r": float(pr), "Spearman_rho": float(sr),
    }


def make_gp(seed: int = DEFAULT_SEED) -> GaussianProcessRegressor:
    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e6))
        * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 10.0), nu=2.5)
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e6))
    )
    return GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=3,
        normalize_y=True, random_state=seed,
    )


def make_rf(seed: int = DEFAULT_SEED) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2,
        max_features=1.0, random_state=seed, n_jobs=1,
    )


def expected_improvement(
    mu: np.ndarray, sigma: np.ndarray, y_best: float, xi: float = EI_XI
) -> np.ndarray:
    """EI acquisition for minimisation."""
    sigma = np.maximum(sigma, 1e-9)
    z  = (y_best - mu - xi) / sigma
    ei = (y_best - mu - xi) * sp_norm.cdf(z) + sigma * sp_norm.pdf(z)
    return np.maximum(ei, 0.0)


def rf_predict_with_std(
    rf: RandomForestRegressor, X: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Use individual tree predictions to get mean & std from RF."""
    preds = np.array([tree.predict(X) for tree in rf.estimators_])
    return preds.mean(axis=0), preds.std(axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# BO run helper (pool-based, always minimises y_internal)
# ──────────────────────────────────────────────────────────────────────────────

def _bo_run(
    pool_X: np.ndarray, pool_y: np.ndarray,
    n_seed: int, n_iter: int, seed: int,
    mode: str = "gp_ei",
) -> np.ndarray:
    """
    One independent pool-based BO run (always minimises pool_y).

    mode: 'gp_ei' | 'rf_ei' | 'greedy' | 'random'

    Returns array of length (n_seed + actual_iters) with cumulative
    best values (running minimum).
    """
    rng = np.random.default_rng(seed)
    pool_size = len(pool_X)
    n_seed_clamped = min(n_seed, pool_size - 1)
    observed  = list(rng.choice(pool_size, size=n_seed_clamped, replace=False).tolist())
    remaining = [i for i in range(pool_size) if i not in observed]

    best_history: List[float] = [pool_y[i] for i in observed]

    n_iter_actual = min(n_iter, len(remaining))
    for _ in range(n_iter_actual):
        if not remaining:
            break

        Xobs = pool_X[observed]
        yobs = pool_y[observed]
        Xrem = pool_X[remaining]
        y_best = float(yobs.min())

        if mode == "random":
            chosen_local = int(rng.integers(0, len(remaining)))

        elif mode == "gp_ei":
            gp = make_gp(seed=seed)
            try:
                gp.fit(Xobs, yobs)
                mu, sigma = gp.predict(Xrem, return_std=True)
                ei = expected_improvement(mu, sigma, y_best)
                chosen_local = int(np.argmax(ei))
            except Exception:
                chosen_local = int(rng.integers(0, len(remaining)))

        elif mode == "rf_ei":
            rf = make_rf(seed=seed)
            rf.fit(Xobs, yobs)
            mu, sigma = rf_predict_with_std(rf, Xrem)
            ei = expected_improvement(mu, sigma, y_best)
            chosen_local = int(np.argmax(ei))

        elif mode == "greedy":
            gp = make_gp(seed=seed)
            try:
                gp.fit(Xobs, yobs)
                mu, _ = gp.predict(Xrem, return_std=True)
            except Exception:
                mu = yobs.mean() * np.ones(len(Xrem))
            chosen_local = int(np.argmin(mu))

        else:
            raise ValueError(f"Unknown BO mode: {mode!r}")

        chosen_global = remaining[chosen_local]
        observed.append(chosen_global)
        remaining.remove(chosen_global)
        best_history.append(float(pool_y[observed].min()))

    # Guarantee strict cumulative minimum
    arr = np.array(best_history)
    for i in range(1, len(arr)):
        arr[i] = min(arr[i], arr[i - 1])
    return arr


# ──────────────────────────────────────────────────────────────────────────────
# Main engine
# ──────────────────────────────────────────────────────────────────────────────

class ValidationEngine:
    """
    Runs validation sections §0–§5 on an arbitrary tabular dataset.

    Parameters
    ----------
    X              : np.ndarray  shape (n_samples, n_features)
    y              : np.ndarray  shape (n_samples,)
    feature_names  : List[str]
    target_name    : str
    direction      : "minimize" | "maximize"
    context_X      : Optional[np.ndarray]  shape (n_samples, n_ctx)
    context_names  : Optional[List[str]]
    seed           : int
    n_cv_folds     : int
    n_cv_reps      : int
    n_seed_bo      : int
    n_bo_iter      : int
    n_bo_runs      : int
    progress_cb    : Optional[Callable[[int, str], None]]
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        target_name: str,
        direction: str = "minimize",
        context_X: Optional[np.ndarray] = None,
        context_names: Optional[List[str]] = None,
        seed: int = DEFAULT_SEED,
        n_cv_folds: int = DEFAULT_N_CV_FOLDS,
        n_cv_reps:  int = DEFAULT_N_CV_REPS,
        n_seed_bo:  int = DEFAULT_N_SEED_BO,
        n_bo_iter:  int = DEFAULT_N_BO_ITER,
        n_bo_runs:  int = DEFAULT_N_BO_RUNS,
        progress_cb: Optional[Callable[[int, str], None]] = None,
    ) -> None:
        self.X             = np.asarray(X, dtype=float)
        self.y             = np.asarray(y, dtype=float)
        self.feature_names = list(feature_names)
        self.target_name   = target_name
        self.direction     = direction
        self.context_X     = (np.asarray(context_X, dtype=float)
                              if context_X is not None else None)
        self.context_names = list(context_names) if context_names else []
        self.seed          = seed
        self.n_cv_folds    = n_cv_folds
        self.n_cv_reps     = n_cv_reps
        self.n_seed_bo     = n_seed_bo
        self.n_bo_iter     = n_bo_iter
        self.n_bo_runs     = n_bo_runs
        self.progress_cb   = progress_cb

        # Internal storage for section results (passed to §5)
        self._s0: dict = {}
        self._s1: dict = {}
        self._s2: dict = {}
        self._s3: dict = {}

    # ── Progress helper ─────────────────────────────────────────────────────

    def _emit(self, pct: int, msg: str) -> None:
        if self.progress_cb is not None:
            try:
                self.progress_cb(pct, msg)
            except Exception:
                pass

    # ── Replicate aggregation ────────────────────────────────────────────────

    def _aggregate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Average replicates → one row per unique composition.
        Returns (X_agg, y_agg, X_full_agg).
        X_full_agg has context columns appended (means) if available.
        """
        try:
            feature_df = pd.DataFrame(self.X, columns=self.feature_names)
            feature_df["__y__"] = self.y

            agg = feature_df.groupby(
                self.feature_names, as_index=False, sort=False
            )["__y__"].mean()
            X_agg = agg[self.feature_names].values.astype(float)
            y_agg = agg["__y__"].values.astype(float)

            if self.context_X is not None and len(self.context_X) == len(self.X):
                ctx_names = self.context_names or [
                    f"ctx_{i}" for i in range(self.context_X.shape[1])
                ]
                ctx_df = pd.DataFrame(self.context_X, columns=ctx_names)
                merged = pd.concat(
                    [feature_df.drop("__y__", axis=1), ctx_df], axis=1
                )
                ctx_agg = merged.groupby(
                    self.feature_names, as_index=False, sort=False
                )[ctx_names].mean()
                ctx_vals = ctx_agg[ctx_names].values.astype(float)
                X_full_agg = np.concatenate([X_agg, ctx_vals], axis=1)
            else:
                X_full_agg = X_agg.copy()

            return X_agg, y_agg, X_full_agg

        except Exception:
            X_full = (
                np.concatenate([self.X, self.context_X], axis=1)
                if self.context_X is not None and len(self.context_X) == len(self.X)
                else self.X.copy()
            )
            return self.X.copy(), self.y.copy(), X_full

    # ── Main entry point ─────────────────────────────────────────────────────

    def run(self) -> ValidationResults:
        """Execute all sections and return a ValidationResults object."""
        figures: Dict[str, Any] = {}

        # §0 ── Data quality
        self._emit(0, "§0  Checking data quality…")
        try:
            fig0 = self._section0_data()
            if fig0 is not None:
                figures["s0_data_summary"] = fig0
        except Exception:
            pass
        self._emit(10, "§0  Complete")

        # §1 ── Surrogate accuracy
        self._emit(10, "§1  Training RF + GP surrogates…")
        try:
            fig1a, fig1b = self._section1_surrogate()
            if fig1a is not None:
                figures["s1_predicted_vs_true"] = fig1a
            if fig1b is not None:
                figures["s1_residuals"] = fig1b
        except Exception:
            pass
        self._emit(45, "§1  Complete")

        # §2 ── BO benchmark
        self._emit(45, "§2  Running virtual BO benchmark…")
        try:
            fig2a, fig2b = self._section2_bo_benchmark()
            if fig2a is not None:
                figures["s2_bo_convergence"] = fig2a
            if fig2b is not None:
                figures["s2_bo_convergence_norm"] = fig2b
        except Exception:
            pass
        self._emit(85, "§2  Complete")

        # §3 ── EI marginals
        self._emit(85, "§3  Computing EI marginals…")
        try:
            fig3 = self._section3_ei_marginals()
            if fig3 is not None:
                figures["s3_ei_marginals"] = fig3
        except Exception:
            pass
        self._emit(95, "§3  Complete")

        # §5 ── Pass/fail summary
        self._emit(95, "§5  Building pass/fail summary…")
        passfail = self._compute_passfail()
        try:
            fig5 = self._section5_passfail(passfail)
            if fig5 is not None:
                figures["s5_passfail"] = fig5
        except Exception:
            pass
        self._emit(100, "Complete")

        n_pass   = sum(1 for _, ok, _ in passfail if ok)
        n_checks = len(passfail)
        if n_pass == n_checks:
            summary = "GREEN"
        elif n_checks > 0 and n_pass >= n_checks * 0.75:
            summary = "YELLOW"
        else:
            summary = "RED"

        metrics = {
            **self._s0, **self._s1, **self._s2, **self._s3
        }

        return ValidationResults(
            figures=figures,
            metrics=metrics,
            passfail=passfail,
            n_pass=n_pass,
            n_checks=n_checks,
            summary=summary,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # §0  Data quality
    # ──────────────────────────────────────────────────────────────────────────

    def _section0_data(self) -> Optional[plt.Figure]:
        """
        Build a 1×2 figure:
          Left:  histogram of the target column
          Right: bar chart of within-replicate std (if any)
        """
        feature_df = pd.DataFrame(self.X, columns=self.feature_names)
        feature_df["__y__"] = self.y

        grp = feature_df.groupby(self.feature_names)["__y__"].agg(
            ["count", "mean", "std"]
        ).reset_index()
        replicated = grp[grp["count"] > 1]

        n_raw    = len(self.y)
        n_unique = len(grp)
        med_cv   = float(
            (replicated["std"] / replicated["mean"].abs()).median()
            if len(replicated) and replicated["mean"].abs().min() > 0 else 0.0
        )

        # Store for §5
        self._s0["n_raw"]    = n_raw
        self._s0["n_unique"] = n_unique
        self._s0["median_cv"] = med_cv

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.patch.set_facecolor(_BG)

        # Left: histogram of y
        ax = axes[0]
        _style_ax(ax)
        y_vals = self.y[np.isfinite(self.y)]
        bins = min(30, max(8, n_raw // 5))
        ax.hist(y_vals, bins=bins, color=_ACC, alpha=0.80,
                edgecolor=_BG, linewidth=0.4)
        mean_v = float(np.mean(y_vals))
        med_v  = float(np.median(y_vals))
        ax.axvline(mean_v, color=_GRN, linewidth=1.5, linestyle="--",
                   label=f"Mean {mean_v:.4g}")
        ax.axvline(med_v,  color=_AMB, linewidth=1.5, linestyle=":",
                   label=f"Median {med_v:.4g}")
        ax.set_xlabel(self.target_name, fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.set_title(f"Target distribution  (n={n_raw}, unique={n_unique})",
                     fontsize=9)
        ax.legend(fontsize=7, facecolor=_BG, edgecolor=_GRID, labelcolor=_FG)

        # Right: within-replicate std bar chart
        ax2 = axes[1]
        _style_ax(ax2)
        if len(replicated) > 0:
            sorted_rep = replicated.sort_values("std", ascending=False).head(40)
            std_vals = sorted_rep["std"].values
            ax2.bar(range(len(std_vals)), std_vals, color=_RED, alpha=0.80,
                    edgecolor=_BG, linewidth=0.3)
            ax2.set_xlabel("Unique composition (sorted by std)", fontsize=8)
            ax2.set_ylabel("Within-replicate std", fontsize=8)
            ax2.set_title(
                f"Replicate variability  ({len(replicated)} groups with replicates)\n"
                f"Median CV = {med_cv:.2%}",
                fontsize=9,
            )
        else:
            ax2.text(0.5, 0.5, "No replicates found\n(all compositions appear once)",
                     ha="center", va="center", color=_FG, fontsize=10,
                     transform=ax2.transAxes)
            ax2.set_title("Replicate variability", fontsize=9)

        if len(self.feature_names) > 1:
            multi_note = ""
        else:
            multi_note = ""
        fig.suptitle(
            f"§0  Data Quality — {self.target_name}",
            color=_FG, fontsize=10, y=1.01,
        )
        fig.tight_layout()
        return fig

    # ──────────────────────────────────────────────────────────────────────────
    # §1  Surrogate accuracy
    # ──────────────────────────────────────────────────────────────────────────

    def _section1_surrogate(self) -> Tuple[Optional[plt.Figure], Optional[plt.Figure]]:
        """
        Fit RF + GP (hold-out and repeated CV) and build:
          - 2×2 predicted-vs-true scatter figure
          - 1×2 residuals figure
        """
        X_agg, y_agg, X_full_agg = self._aggregate()
        n_unique = len(X_agg)
        if n_unique < 5:
            return None, None

        # For direction=="maximize" work internally with negated y
        y_internal = -y_agg if self.direction == "maximize" else y_agg.copy()

        # ── 70/30 hold-out ────────────────────────────────────────────────────
        Xtr, Xte, ytr, yte = train_test_split(
            X_full_agg, y_internal, test_size=_TEST_SIZE, random_state=self.seed
        )

        gp_ho_ok = False
        try:
            gp = make_gp(self.seed)
            gp.fit(Xtr, ytr)
            gp_pred_te = gp.predict(Xte)
            gp_ho_ok = True
        except Exception as exc:
            gp_pred_te = ytr.mean() * np.ones(len(yte))
            self._s1["gp_error"] = str(exc)

        rf = make_rf(self.seed)
        rf.fit(Xtr, ytr)
        rf_pred_te = rf.predict(Xte)

        m_rf_ho = compute_metrics(yte, rf_pred_te)
        m_gp_ho = compute_metrics(yte, gp_pred_te)

        # ── Repeated k-fold CV ────────────────────────────────────────────────
        rkf = RepeatedKFold(
            n_splits=min(self.n_cv_folds, n_unique // 2),
            n_repeats=self.n_cv_reps,
            random_state=self.seed,
        )

        rf_oof_acc = np.zeros(n_unique)
        rf_oof_cnt = np.zeros(n_unique)
        gp_oof_acc = np.zeros(n_unique)
        gp_oof_cnt = np.zeros(n_unique)

        for fold_i, (tr_i, te_i) in enumerate(rkf.split(X_full_agg)):
            rf_ = make_rf(seed=self.seed + fold_i)
            rf_.fit(X_full_agg[tr_i], y_internal[tr_i])
            rf_oof_acc[te_i] += rf_.predict(X_full_agg[te_i])
            rf_oof_cnt[te_i] += 1

            try:
                gp_ = make_gp(seed=self.seed + fold_i)
                gp_.fit(X_full_agg[tr_i], y_internal[tr_i])
                gp_oof_acc[te_i] += gp_.predict(X_full_agg[te_i])
                gp_oof_cnt[te_i] += 1
            except Exception:
                gp_oof_acc[te_i] += y_internal[tr_i].mean()
                gp_oof_cnt[te_i] += 1

        rf_oof = np.where(rf_oof_cnt > 0, rf_oof_acc / rf_oof_cnt, 0.0)
        gp_oof = np.where(gp_oof_cnt > 0, gp_oof_acc / gp_oof_cnt, 0.0)

        m_rf_cv = compute_metrics(y_internal, rf_oof)
        m_gp_cv = compute_metrics(y_internal, gp_oof)

        # Store for §5
        self._s1["holdout_rf"] = m_rf_ho
        self._s1["holdout_gp"] = m_gp_ho
        self._s1["cv_rf"]      = m_rf_cv
        self._s1["cv_gp"]      = m_gp_cv
        self._s1["y_internal"] = y_internal

        # ── Colour by objective quartile ──────────────────────────────────────
        q25, q50, q75 = np.percentile(y_internal, [25, 50, 75])
        def _quartile_color(val: float) -> str:
            if val <= q25: return "#89b4fa"   # Q1 blue
            if val <= q50: return "#89dceb"   # Q2 cyan
            if val <= q75: return "#f9e2af"   # Q3 amber
            return "#f38ba8"                  # Q4 red

        # ── Figure: 2×2 predicted vs true ────────────────────────────────────
        fig1, axes = plt.subplots(2, 2, figsize=(11, 9))
        fig1.patch.set_facecolor(_BG)
        fig1.suptitle(
            f"§1  Surrogate Accuracy — Predicted vs. True\n({self.target_name})",
            color=_FG, fontsize=10,
        )

        y_te_display = -yte if self.direction == "maximize" else yte
        plot_data = [
            (axes[0, 0], yte,       rf_pred_te, "RF  — 70/30 hold-out",  m_rf_ho),
            (axes[0, 1], yte,       gp_pred_te, "GP  — 70/30 hold-out",  m_gp_ho),
            (axes[1, 0], y_internal, rf_oof,     f"RF  — CV {self.n_cv_folds}-fold OOF", m_rf_cv),
            (axes[1, 1], y_internal, gp_oof,     f"GP  — CV {self.n_cv_folds}-fold OOF", m_gp_cv),
        ]

        for ax, yt, yp, title, met in plot_data:
            _style_ax(ax)
            colors = [_quartile_color(v) for v in yt]
            ax.scatter(yt, yp, c=colors, s=25, alpha=0.75,
                       edgecolors="none", zorder=3)
            lo = min(yt.min(), yp.min())
            hi = max(yt.max(), yp.max())
            pad = (hi - lo) * 0.05 if hi > lo else 1.0
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                    color=_FG, linewidth=0.9, linestyle="--", alpha=0.5,
                    label="Perfect")
            ax.set_xlabel("True value", fontsize=8)
            ax.set_ylabel("Predicted value", fontsize=8)
            ax.set_title(title, fontsize=9)
            ax.text(
                0.97, 0.05,
                f"r={met['Pearson_r']:.3f}\nρ={met['Spearman_rho']:.3f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color=_FG,
                bbox=dict(boxstyle="round,pad=0.3", fc=_AX, ec=_GRID, alpha=0.8),
            )

        fig1.tight_layout()

        # ── Figure: 1×2 residuals ─────────────────────────────────────────────
        fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.5))
        fig2.patch.set_facecolor(_BG)
        fig2.suptitle("§1  Residuals (CV out-of-fold)", color=_FG, fontsize=10)

        for ax, yp, title in [
            (axes2[0], rf_oof, "Random Forest"),
            (axes2[1], gp_oof, "Gaussian Process"),
        ]:
            _style_ax(ax)
            residuals = yp - y_internal
            colors = [_quartile_color(v) for v in y_internal]
            ax.scatter(y_internal, residuals, c=colors, s=25, alpha=0.75,
                       edgecolors="none", zorder=3)
            ax.axhline(0, color=_FG, linewidth=0.9, linestyle="--", alpha=0.5)
            ax.set_xlabel("True value", fontsize=8)
            ax.set_ylabel("Residual (pred − true)", fontsize=8)
            ax.set_title(title, fontsize=9)

        fig2.tight_layout()
        return fig1, fig2

    # ──────────────────────────────────────────────────────────────────────────
    # §2  BO benchmark
    # ──────────────────────────────────────────────────────────────────────────

    def _section2_bo_benchmark(self) -> Tuple[Optional[plt.Figure], Optional[plt.Figure]]:
        """
        Run N_BO_RUNS independent BO runs for four strategies and
        build two convergence figures (absolute + normalised).
        """
        X_agg, y_agg, X_full_agg = self._aggregate()
        n_unique = len(X_agg)

        # Clamp parameters for small pools
        n_seed = min(self.n_seed_bo, n_unique - 2)
        n_iter = min(self.n_bo_iter, n_unique - n_seed - 1)

        if n_seed < 2 or n_iter < 1:
            self._s2["skipped"] = (
                f"Pool too small ({n_unique} unique rows) — "
                f"need ≥ {self.n_seed_bo + 2} for BO benchmark."
            )
            return None, None

        # Work with negated y for maximisation (always minimise internally)
        pool_y = -y_agg if self.direction == "maximize" else y_agg.copy()
        pool_X = X_full_agg

        true_best_internal = float(pool_y.min())
        true_worst_internal = float(pool_y.max())

        strategies = {
            "GP + EI": "gp_ei",
            "RF + EI": "rf_ei",
            "Greedy":  "greedy",
            "Random":  "random",
        }

        all_curves: Dict[str, np.ndarray] = {}
        total_steps_expected = n_seed + n_iter

        for label, mode in strategies.items():
            curves = []
            for run_i in range(self.n_bo_runs):
                c = _bo_run(
                    pool_X, pool_y,
                    n_seed=n_seed, n_iter=n_iter,
                    seed=self.seed * 100 + run_i,
                    mode=mode,
                )
                curves.append(c)
            # Pad shorter curves to same length
            max_len = max(len(c) for c in curves)
            padded = [
                np.pad(c, (0, max_len - len(c)), mode="edge")
                for c in curves
            ]
            all_curves[label] = np.array(padded)   # (n_runs, steps)

        # Top-20% threshold (internal minimisation)
        top20_thr = float(np.percentile(pool_y, 20))

        results_s2: dict = {}
        for label, curves in all_curves.items():
            steps_found = []
            for c in curves:
                hits = np.where(c <= top20_thr)[0]
                steps_found.append(int(hits[0]) if len(hits) else np.nan)
            med = float(np.nanmedian(steps_found))
            results_s2[label] = {"median_steps_top20": med}
        self._s2.update(results_s2)

        x_axis = np.arange(1, all_curves[list(all_curves)[0]].shape[1] + 1)

        # ── Figure A: absolute convergence ───────────────────────────────────
        fig_a, ax_a = plt.subplots(figsize=(9, 5))
        fig_a.patch.set_facecolor(_BG)
        _style_ax(ax_a)

        for label, curves in all_curves.items():
            col = _STRAT_COLORS.get(label, _ACC)
            # Un-negate for display if maximising
            disp = -curves if self.direction == "maximize" else curves
            mu  = disp.mean(axis=0)
            std = disp.std(axis=0)
            ax_a.plot(x_axis, mu, label=label, color=col, linewidth=2)
            ax_a.fill_between(x_axis, mu - std, mu + std, color=col, alpha=0.12)

        global_best_display = (
            -true_best_internal if self.direction == "maximize"
            else true_best_internal
        )
        direction_arrow = "↑ best" if self.direction == "maximize" else "↓ best"
        ax_a.axhline(
            global_best_display, color=_FG, linewidth=1.0, linestyle="--",
            alpha=0.6, label=f"Global best ({global_best_display:.4g})",
        )
        ax_a.axvline(
            n_seed, color=_GRID, linewidth=1.0, linestyle=":",
            alpha=0.8, label=f"End of seed ({n_seed} pts)",
        )
        ax_a.set_xlabel("Number of evaluations", fontsize=9)
        ax_a.set_ylabel(f"Best {self.target_name} found ({direction_arrow})", fontsize=9)
        ax_a.set_title(
            f"§2  BO Benchmark — Best Value Found vs. Evaluations\n"
            f"(mean ± 1 std over {self.n_bo_runs} runs)",
            color=_FG, fontsize=9,
        )
        ax_a.legend(fontsize=8, facecolor=_BG, edgecolor=_GRID, labelcolor=_FG,
                    loc="best")
        fig_a.tight_layout()

        # ── Figure B: normalised convergence ──────────────────────────────────
        fig_b, ax_b = plt.subplots(figsize=(9, 5))
        fig_b.patch.set_facecolor(_BG)
        _style_ax(ax_b)

        scale = true_worst_internal - true_best_internal
        if scale < 1e-9:
            scale = 1.0

        for label, curves in all_curves.items():
            col = _STRAT_COLORS.get(label, _ACC)
            norm_curves = 1.0 - (curves - true_best_internal) / scale
            mu  = norm_curves.mean(axis=0)
            std = norm_curves.std(axis=0)
            ax_b.plot(x_axis, mu, label=label, color=col, linewidth=2)
            ax_b.fill_between(
                x_axis,
                np.maximum(0, mu - std),
                np.minimum(1, mu + std),
                color=col, alpha=0.12,
            )

        ax_b.axhline(
            1.0, color=_FG, linewidth=1.0, linestyle="--",
            alpha=0.6, label="Global best (1.0)",
        )
        ax_b.axvline(
            n_seed, color=_GRID, linewidth=1.0, linestyle=":",
            alpha=0.8,
        )
        ax_b.set_ylim(0, 1.08)
        ax_b.set_xlabel("Number of evaluations", fontsize=9)
        ax_b.set_ylabel("Normalised performance (1 = global best)", fontsize=9)
        ax_b.set_title(
            "§2  BO Benchmark — Normalised Performance (1 = global best)",
            color=_FG, fontsize=9,
        )
        ax_b.legend(fontsize=8, facecolor=_BG, edgecolor=_GRID, labelcolor=_FG,
                    loc="best")
        fig_b.tight_layout()

        return fig_a, fig_b

    # ──────────────────────────────────────────────────────────────────────────
    # §3  EI marginals
    # ──────────────────────────────────────────────────────────────────────────

    def _section3_ei_marginals(self) -> Optional[plt.Figure]:
        """
        Fit GP on full aggregated data and compute 1-D EI marginals per feature.
        Returns a figure with one subplot per feature.
        """
        X_agg, y_agg, X_full_agg = self._aggregate()
        n_unique = len(X_agg)
        if n_unique < 5:
            return None

        y_internal = -y_agg if self.direction == "maximize" else y_agg.copy()
        y_best = float(y_internal.min())

        # Fit GP on full data
        gp_ok = False
        try:
            gp_full = make_gp(self.seed)
            gp_full.fit(X_full_agg, y_internal)
            gp_ok = True
        except Exception as exc:
            self._s3["gp_error"] = str(exc)

        if not gp_ok:
            return None

        n_feat = len(self.feature_names)
        # Median values for all dimensions (including context)
        medians = np.median(X_full_agg, axis=0)

        # ── Determine layout ───────────────────────────────────────────────────
        n_cols = min(n_feat, 4)
        n_rows = int(np.ceil(n_feat / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(3.5 * n_cols, 3.0 * n_rows),
                                 squeeze=False)
        fig.patch.set_facecolor(_BG)
        fig.suptitle(
            "§3  Expected Improvement Marginals — Where to Explore Next?\n"
            "(other features held at their median value)",
            color=_FG, fontsize=9, y=1.01,
        )

        ei_best_vals: List[float] = []
        out_of_range_features: List[str] = []

        for d, feat_name in enumerate(self.feature_names):
            row_idx = d // n_cols
            col_idx = d % n_cols
            ax = axes[row_idx][col_idx]
            _style_ax(ax)

            x_min = float(X_agg[:, d].min())
            x_max = float(X_agg[:, d].max())

            if x_max - x_min < 1e-9:
                ax.text(0.5, 0.5, f"Constant feature:\n{feat_name}",
                        ha="center", va="center", color=_FG, fontsize=9,
                        transform=ax.transAxes)
                ax.set_title(feat_name, fontsize=9)
                ei_best_vals.append(x_min)
                continue

            sweep_x = np.linspace(x_min, x_max, EI_MARGINAL_PTS)

            # Build sweep matrix with all other dims at median
            X_sweep = np.tile(medians, (EI_MARGINAL_PTS, 1))
            X_sweep[:, d] = sweep_x   # feature d varies

            try:
                mu_sw, sigma_sw = gp_full.predict(X_sweep, return_std=True)
                ei_sw = expected_improvement(mu_sw, sigma_sw, y_best)
            except Exception:
                ax.text(0.5, 0.5, "GP prediction\nfailed",
                        ha="center", va="center", color=_RED, fontsize=9,
                        transform=ax.transAxes)
                ax.set_title(feat_name, fontsize=9)
                ei_best_vals.append(x_min)
                continue

            ax.plot(sweep_x, ei_sw, color=_ACC, linewidth=1.5)
            ax.fill_between(sweep_x, 0, ei_sw, alpha=0.25, color=_ACC)

            # Mark the current observed best value for this feature
            best_idx_obs = int(np.argmin(y_internal))
            best_x_obs = float(X_agg[best_idx_obs, d])
            ax.axvline(best_x_obs, color=_GRN, linewidth=1.0, linestyle="--",
                       alpha=0.7, label=f"Obs. best: {best_x_obs:.4g}")

            # Argmax EI
            best_ei_x = float(sweep_x[np.argmax(ei_sw)])
            ei_best_vals.append(best_ei_x)

            ax.set_xlabel(feat_name, fontsize=8)
            ax.set_ylabel("GP EI", fontsize=8)
            ax.set_title(feat_name, fontsize=9)
            ax.legend(fontsize=7, facecolor=_BG, edgecolor=_GRID, labelcolor=_FG)

            # Bounding-box check
            rng_d = x_max - x_min
            lower_d = x_min - 0.20 * rng_d
            upper_d = x_max + 0.20 * rng_d
            if not (lower_d <= best_ei_x <= upper_d):
                out_of_range_features.append(feat_name)

        # Hide unused subplots
        for d in range(n_feat, n_rows * n_cols):
            axes[d // n_cols][d % n_cols].set_visible(False)

        # Align y-axes across all subplots for comparability
        try:
            all_ylims = [ax.get_ylim() for row in axes for ax in row if ax.get_visible()]
            if all_ylims:
                y_hi = max(yl[1] for yl in all_ylims)
                for row in axes:
                    for ax in row:
                        if ax.get_visible():
                            ax.set_ylim(bottom=0, top=y_hi * 1.05)
        except Exception:
            pass

        # Store for §5
        self._s3["ei_best_vals"]          = ei_best_vals
        self._s3["out_of_range_features"] = out_of_range_features

        fig.tight_layout()
        return fig

    # ──────────────────────────────────────────────────────────────────────────
    # §5  Pass/fail summary
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_passfail(self) -> List[Tuple[str, bool, str]]:
        """Evaluate all 8 checks and return list of (description, passed, detail)."""
        checks: List[Tuple[str, bool, str]] = []

        # 1. Enough unique data
        n_unique = self._s0.get("n_unique", len(self.X))
        checks.append((
            "n_unique ≥ 30 (meaningful statistics)",
            n_unique >= MIN_UNIQUE_ROWS,
            f"n_unique = {n_unique} unique compositions",
        ))

        # 2. RF Pearson r > 0.5 (hold-out)
        rf_r = self._s1.get("holdout_rf", {}).get("Pearson_r", 0.0)
        checks.append((
            "RF Pearson r > 0.5 on hold-out",
            rf_r > 0.5,
            f"RF r = {rf_r:.3f}",
        ))

        # 3. GP Pearson r > 0.5 (hold-out)
        gp_r = self._s1.get("holdout_gp", {}).get("Pearson_r", 0.0)
        checks.append((
            "GP Pearson r > 0.5 on hold-out",
            gp_r > 0.5,
            f"GP r = {gp_r:.3f}",
        ))

        # 4. RF Spearman ρ > 0.5 (hold-out)
        rf_rho = self._s1.get("holdout_rf", {}).get("Spearman_rho", 0.0)
        checks.append((
            "RF Spearman ρ > 0.5 on hold-out",
            rf_rho > 0.5,
            f"RF ρ = {rf_rho:.3f}",
        ))

        # 5. CV RMSE < 30% of target range
        y_internal = self._s1.get("y_internal", self.y)
        y_scale = float(y_internal.max() - y_internal.min())
        if y_scale < 1e-9:
            y_scale = 1.0
        rf_rmse = self._s1.get("cv_rf", {}).get("RMSE", float("inf"))
        gp_rmse = self._s1.get("cv_gp", {}).get("RMSE", float("inf"))
        rf_pct  = rf_rmse / y_scale
        gp_pct  = gp_rmse / y_scale
        checks.append((
            "CV RMSE < 30% of target range (both models)",
            rf_pct < 0.30 and gp_pct < 0.30,
            f"RF RMSE={rf_pct:.1%} of range, GP RMSE={gp_pct:.1%} of range",
        ))

        # 6. GP+EI reaches top-20% faster than Random
        if "skipped" in self._s2:
            checks.append((
                "GP+EI BO reaches top-20% faster than Random",
                False,
                f"Skipped — {self._s2['skipped']}",
            ))
            checks.append((
                "RF+EI BO reaches top-20% faster than Random",
                False,
                f"Skipped — {self._s2['skipped']}",
            ))
        else:
            rand_steps = self._s2.get("Random", {}).get("median_steps_top20", np.nan)
            gp_steps   = self._s2.get("GP + EI", {}).get("median_steps_top20", np.nan)
            rf_steps   = self._s2.get("RF + EI", {}).get("median_steps_top20", np.nan)

            if not np.isnan(gp_steps) and not np.isnan(rand_steps):
                gp_beats = gp_steps <= rand_steps
                gp_detail = f"GP+EI median={gp_steps:.0f} steps, Random={rand_steps:.0f} steps"
            elif np.isnan(gp_steps):
                gp_beats  = False
                gp_detail = "GP+EI never reached top-20%"
            else:
                gp_beats  = True
                gp_detail = f"GP+EI={gp_steps:.0f} steps; random never reached top-20%"
            checks.append(("GP+EI BO reaches top-20% faster than Random", gp_beats, gp_detail))

            if not np.isnan(rf_steps) and not np.isnan(rand_steps):
                rf_beats = rf_steps <= rand_steps
                rf_detail = f"RF+EI median={rf_steps:.0f} steps, Random={rand_steps:.0f} steps"
            elif np.isnan(rf_steps):
                rf_beats  = False
                rf_detail = "RF+EI never reached top-20%"
            else:
                rf_beats  = True
                rf_detail = f"RF+EI={rf_steps:.0f} steps; random never reached top-20%"
            checks.append(("RF+EI BO reaches top-20% faster than Random", rf_beats, rf_detail))

        # 8. Top EI suggestion within training data bounding box (±20%)
        out_of_range = self._s3.get("out_of_range_features", [])
        if "gp_error" in self._s3 and not self._s3.get("ei_best_vals"):
            checks.append((
                "Top EI suggestions within training data bounding box",
                False,
                f"GP fitting failed: {self._s3.get('gp_error', '')[:80]}",
            ))
        else:
            in_range = len(out_of_range) == 0
            detail = (
                "All features within training range (±20%)"
                if in_range
                else f"Out of range: {', '.join(out_of_range[:5])}"
            )
            checks.append((
                "Top EI suggestions within training data bounding box",
                in_range,
                detail,
            ))

        return checks

    def _section5_passfail(
        self, passfail: List[Tuple[str, bool, str]]
    ) -> plt.Figure:
        """
        Render the pass/fail table as a matplotlib Figure.
        """
        n_checks = len(passfail)
        n_pass   = sum(1 for _, ok, _ in passfail if ok)

        if n_pass == n_checks:
            summary, sum_color = "GREEN — All checks passed ✅", _GRN
        elif n_checks > 0 and n_pass >= n_checks * 0.75:
            summary, sum_color = f"YELLOW — {n_pass}/{n_checks} checks passed ⚠", _AMB
        else:
            summary, sum_color = f"RED — {n_pass}/{n_checks} checks passed ❌", _RED

        fig_h = max(4.5, 0.45 * (n_checks + 2) + 1.5)
        fig, ax = plt.subplots(figsize=(11, fig_h))
        fig.patch.set_facecolor(_BG)
        ax.set_facecolor(_BG)
        ax.set_axis_off()

        col_labels = ["Check", "Status", "Detail"]
        cell_text  = []
        cell_colors = []
        status_colors = []

        for desc, ok, detail in passfail:
            status_txt   = "✓ PASS" if ok else "✗ FAIL"
            row_bg       = "#1a3025" if ok else "#3d1a1f"
            status_color = _GRN     if ok else _RED
            cell_text.append([desc[:70], status_txt, detail[:80]])
            cell_colors.append([row_bg, row_bg, row_bg])
            status_colors.append(status_color)

        # Summary row
        cell_text.append(["", f"{n_pass}/{n_checks} checks passed", summary])
        cell_colors.append(["#2a2a3e", "#2a2a3e", "#2a2a3e"])
        status_colors.append(sum_color)

        table = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellColors=cell_colors,
            cellLoc="left",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)

        # Style header
        for col_i in range(len(col_labels)):
            cell = table[0, col_i]
            cell.set_facecolor("#2a2a3e")
            cell.get_text().set_color(_ACC)
            cell.get_text().set_fontweight("bold")

        # Style data rows
        for row_i in range(1, n_checks + 1):
            _, ok, _ = passfail[row_i - 1]
            for col_i in range(3):
                cell = table[row_i, col_i]
                cell.get_text().set_color(_FG)
            # Status column gets special colour
            table[row_i, 1].get_text().set_color(status_colors[row_i - 1])
            table[row_i, 1].get_text().set_fontweight("bold")

        # Style summary row
        sum_row = n_checks + 1
        for col_i in range(3):
            cell = table[sum_row, col_i]
            cell.get_text().set_color(status_colors[sum_row - 1])
            cell.get_text().set_fontweight("bold")

        # Column widths
        table.auto_set_column_width([0, 1, 2])
        table.scale(1.0, 1.6)

        ax.set_title(
            f"§5  Validation Pass/Fail Summary",
            color=_FG, fontsize=10, pad=12,
        )

        fig.tight_layout()
        return fig
