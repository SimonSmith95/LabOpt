"""
dataset_health.py
=================
Pure statistical analysis of a DataFrame given objective and parameter column
names. No Qt dependencies. No thresholds. No opinions. Returns a plain dict
suitable for display, logging, or inclusion in a report.

Public API
----------
analyze_dataset(df, obj_cols, param_cols) -> dict

The returned dict contains descriptive statistics only.  The caller decides
what the numbers mean for their specific problem.

See implementation_guide_analysis_tab_and_report.txt for the full specification.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(val) -> Optional[float]:
    """Return float(val) or None if val is NaN / not finite."""
    try:
        f = float(val)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _pooled_noise_std(
    df: pd.DataFrame,
    param_cols: List[str],
    obj_col: str,
    tolerance: float = 1e-6,
) -> tuple[int, Optional[float]]:
    """
    Group rows by identical parameter values (within tolerance) and compute the
    pooled within-group standard deviation of *obj_col*.

    Returns
    -------
    (n_groups, pooled_std)
        n_groups  : number of replicate groups (each with ≥ 2 rows and ≥ 2 valid
                    objective values).
        pooled_std: sqrt of the mean within-group sample variance, or None when
                    fewer than 2 qualifying groups exist.

    Implementation notes
    --------------------
    Uses aggregate_replicates() from csv_loader to find groups, then recomputes
    per-group variance from the original (non-aggregated) data.  This avoids
    importing aggregate_replicates directly (which would create a circular
    dependency risk) by reimplementing the grouping logic inline.
    """
    if obj_col not in df.columns or not param_cols:
        return 0, None

    numeric_params = [
        c for c in param_cols
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    cat_params = [c for c in param_cols if c in df.columns and c not in numeric_params]

    # Round numeric params to tolerance grid for grouping
    work = df[param_cols + [obj_col]].copy() if param_cols else df[[obj_col]].copy()
    for col in numeric_params:
        if tolerance > 0:
            work[col] = (work[col] / tolerance).round() * tolerance

    group_cols = [c for c in param_cols if c in work.columns]
    if not group_cols:
        return 0, None

    variances: List[float] = []
    try:
        grouped = work.groupby(group_cols, dropna=False, sort=False)
        for _, grp in grouped:
            vals = grp[obj_col].dropna()
            if len(vals) >= 2:
                var = float(vals.var(ddof=1))
                if np.isfinite(var):
                    variances.append(var)
    except Exception:
        return 0, None

    n_groups = len(variances)
    if n_groups < 2:
        return n_groups, None

    pooled = float(np.sqrt(np.mean(variances)))
    return n_groups, (pooled if np.isfinite(pooled) else None)


def _infer_dtype_label(series: pd.Series) -> str:
    """Return a human-readable inferred type string for a parameter column."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return "unknown"
    if series.dtype == bool or pd.api.types.is_bool_dtype(series):
        return "bool"
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        return "categorical"
    # Numeric — check if all values are integers
    try:
        as_int = non_null.values.astype(int).astype(float)
        if np.allclose(non_null.values.astype(float), as_int, atol=0, rtol=0):
            return "int"
    except (ValueError, TypeError, OverflowError):
        pass
    return "float"


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_dataset(
    df: pd.DataFrame,
    obj_cols: List[str],
    param_cols: List[str],
) -> dict:
    """
    Compute descriptive statistics for a dataset given objective and parameter
    column names.

    Parameters
    ----------
    df         : The experiment DataFrame (after cleaning).
    obj_cols   : Names of objective (output / result) columns.
    param_cols : Names of parameter (input) columns.

    Returns
    -------
    dict — see implementation_guide_analysis_tab_and_report.txt Section 2 for
    the full schema.  All statistics are computed ignoring NaN values.
    Returns None for individual metrics that cannot be computed rather than
    raising an exception.

    Design constraints
    ------------------
    - No thresholds, no classification, no colour decisions.
    - Side-effect free.  Does not modify df.
    - Does not raise on edge cases (empty df, single row, all-NaN columns).
    """
    # ── Overview ──────────────────────────────────────────────────────────────
    n_rows = len(df)

    # Complete rows: no NaN in any param or obj column that actually exists
    all_cols = [c for c in (list(obj_cols) + list(param_cols)) if c in df.columns]
    if all_cols and n_rows > 0:
        complete_mask = df[all_cols].notna().all(axis=1)
        n_complete = int(complete_mask.sum())
    else:
        n_complete = n_rows

    # Overall missing percentage across param+obj cols
    if all_cols and n_rows > 0:
        total_cells = n_rows * len(all_cols)
        missing_cells = int(df[all_cols].isna().sum().sum())
        missing_pct = round(100.0 * missing_cells / total_cells, 2) if total_cells > 0 else 0.0
    else:
        missing_pct = 0.0

    n_params = len(param_cols)
    n_objectives = len(obj_cols)
    rows_per_param = (
        round(float(n_complete) / n_params, 2) if n_params > 0 else None
    )

    # ── Per-objective statistics ───────────────────────────────────────────────
    objectives: List[dict] = []
    for col in obj_cols:
        if col not in df.columns:
            objectives.append({
                "column": col,
                "n_valid": 0,
                "mean": None, "std": None, "min": None, "max": None,
                "median": None, "iqr": None, "cv_pct": None,
                "outlier_count": 0, "outlier_fraction": 0.0,
                "replicate_groups": 0,
                "noise_std_replicates": None,
                "snr": None,
            })
            continue

        series = df[col].dropna()
        n_valid = len(series)

        if n_valid == 0:
            objectives.append({
                "column": col,
                "n_valid": 0,
                "mean": None, "std": None, "min": None, "max": None,
                "median": None, "iqr": None, "cv_pct": None,
                "outlier_count": 0, "outlier_fraction": 0.0,
                "replicate_groups": 0,
                "noise_std_replicates": None,
                "snr": None,
            })
            continue

        vals = series.values.astype(float)
        mean_val   = _safe_float(np.mean(vals))
        std_val    = _safe_float(np.std(vals, ddof=1)) if n_valid >= 2 else None
        min_val    = _safe_float(np.min(vals))
        max_val    = _safe_float(np.max(vals))
        median_val = _safe_float(np.median(vals))

        q1 = _safe_float(np.percentile(vals, 25))
        q3 = _safe_float(np.percentile(vals, 75))
        iqr_val = (
            _safe_float(q3 - q1)
            if q1 is not None and q3 is not None
            else None
        )

        # CV: only when |mean| > near-zero threshold
        cv_pct: Optional[float] = None
        if mean_val is not None and abs(mean_val) >= 1e-9 and std_val is not None:
            cv_pct = _safe_float(100.0 * std_val / abs(mean_val))

        # Outliers: Tukey IQR method
        outlier_count = 0
        if q1 is not None and q3 is not None and iqr_val is not None:
            lo = q1 - 1.5 * iqr_val
            hi = q3 + 1.5 * iqr_val
            outlier_count = int(np.sum((vals < lo) | (vals > hi)))
        outlier_fraction = round(outlier_count / n_valid, 4) if n_valid > 0 else 0.0

        # Replicate-based noise
        valid_param_cols = [c for c in param_cols if c in df.columns]
        n_groups, noise_std = _pooled_noise_std(df, valid_param_cols, col)

        # SNR: range / noise_std
        snr: Optional[float] = None
        if (
            noise_std is not None
            and noise_std > 0
            and min_val is not None
            and max_val is not None
        ):
            obj_range = max_val - min_val
            if obj_range > 0:
                snr = _safe_float(obj_range / noise_std)

        objectives.append({
            "column":               col,
            "n_valid":              n_valid,
            "mean":                 mean_val,
            "std":                  std_val,
            "min":                  min_val,
            "max":                  max_val,
            "median":               median_val,
            "iqr":                  iqr_val,
            "cv_pct":               cv_pct,
            "outlier_count":        outlier_count,
            "outlier_fraction":     outlier_fraction,
            "replicate_groups":     n_groups,
            "noise_std_replicates": noise_std,
            "snr":                  snr,
        })

    # ── Per-parameter statistics ───────────────────────────────────────────────
    parameters: List[dict] = []
    for col in param_cols:
        if col not in df.columns:
            parameters.append({
                "column": col,
                "n_valid": 0,
                "n_unique": 0,
                "dtype_inferred": "unknown",
                "missing_pct": 100.0,
            })
            continue

        series = df[col]
        n_valid_p = int(series.notna().sum())
        n_unique  = int(series.dropna().nunique())
        col_missing_pct = round(
            100.0 * float(series.isna().sum()) / len(series), 2
        ) if len(series) > 0 else 0.0
        dtype_label = _infer_dtype_label(series)

        parameters.append({
            "column":        col,
            "n_valid":       n_valid_p,
            "n_unique":      n_unique,
            "dtype_inferred": dtype_label,
            "missing_pct":   col_missing_pct,
        })

    # ── Per-column missing (all columns in df) ────────────────────────────────
    per_column_missing: Dict[str, float] = {}
    for col in df.columns:
        pct = 0.0
        if len(df) > 0:
            pct = round(100.0 * float(df[col].isna().sum()) / len(df), 2)
        per_column_missing[col] = pct

    return {
        "n_rows":             n_rows,
        "n_complete":         n_complete,
        "missing_pct":        missing_pct,
        "n_params":           n_params,
        "n_objectives":       n_objectives,
        "rows_per_param":     rows_per_param,
        "objectives":         objectives,
        "parameters":         parameters,
        "per_column_missing": per_column_missing,
    }
