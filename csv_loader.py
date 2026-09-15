"""
Phase 2 — CSV Loader
Reads CSV data, infers parameter types, and extracts sensible defaults.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from parameter_config import AllowedSubRange, ObjectiveConfig, ParameterConfig, ParameterType, StudyConfig


def load_csv(path: str) -> pd.DataFrame:
    """Read a CSV or Excel data file and return a cleaned DataFrame.

    Supported formats
    -----------------
    * ``.csv`` — both ``','`` and ``';'`` delimiters (auto-detected).
      Handles UTF-8 BOM produced by Excel / Windows tools.
    * ``.xlsx`` / ``.xls`` — read with ``openpyxl`` (Excel workbook,
      first sheet).

    The function intentionally uses the name ``load_csv`` for backward
    compatibility; callers pass any supported path and receive a DataFrame.
    """
    import os as _os
    ext = _os.path.splitext(path)[1].lower()
    try:
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(path, engine="openpyxl")
        else:
            df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    except Exception as exc:
        raise ValueError(f"Cannot read file '{path}': {exc}") from exc
    df.columns = [str(c).strip() for c in df.columns]
    return df


def infer_column_type(series: pd.Series, categorical_threshold: int = 8) -> ParameterType:
    """
    Infer the ParameterType of a pandas Series.

    Decision order
    --------------
    1. BOOL   — dtype bool, or ≤ 2 unique values all in {0, 1, True, False}
    2. CATEGORICAL — object / string dtype
    3. CATEGORICAL — numeric but n_unique ≤ categorical_threshold
    4. INT    — numeric, all non-null values equal their integer cast
    5. FLOAT  — everything else
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return ParameterType.FLOAT

    # 1. Bool
    if series.dtype == bool or pd.api.types.is_bool_dtype(series):
        return ParameterType.BOOL
    unique_vals = set(non_null.unique())
    if unique_vals <= {0, 1, 0.0, 1.0, True, False} and len(unique_vals) <= 2:
        str_vals = {str(v).strip().lower() for v in unique_vals}
        if str_vals <= {"0", "1", "true", "false"}:
            return ParameterType.BOOL

    # 2. Object / string → categorical
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        return ParameterType.CATEGORICAL

    # 3. Few unique numeric values → categorical
    n_unique = non_null.nunique()
    if n_unique <= categorical_threshold:
        return ParameterType.CATEGORICAL

    # 4. All values equal their int cast → INT
    try:
        as_int = non_null.values.astype(int).astype(float)
        if np.allclose(non_null.values.astype(float), as_int, atol=0, rtol=0):
            return ParameterType.INT
    except (ValueError, TypeError, OverflowError):
        pass

    # 5. Default float
    return ParameterType.FLOAT


def extract_param_defaults(
    df: pd.DataFrame,
    result_columns: List[str],
    context_columns: Optional[List[str]] = None,
    categorical_threshold: int = 8,
) -> List[ParameterConfig]:
    """
    Build a ParameterConfig for each non-result, non-context column in *df*.

    - INT / FLOAT : full_min/max come from the data;
                    allowed_subranges defaults to [full_min, full_max].
    - CATEGORICAL  : all_choices = sorted unique string values; all allowed.
    - BOOL         : fixed_value = None (optimise both), choices = [True, False].

    Parameters
    ----------
    df
        The experiment DataFrame.
    result_columns
        Columns that are optimisation objectives — excluded from parameters.
    context_columns
        Columns that are uncontrollable context variables — excluded from
        parameters (they are stored separately in ContextConfig objects).
    categorical_threshold
        Numeric columns with ≤ this many unique values are treated as CATEGORICAL.
    """
    _exclude = set(result_columns) | set(context_columns or [])
    configs: List[ParameterConfig] = []
    for col in df.columns:
        if col in _exclude:
            continue

        series = df[col]
        ptype = infer_column_type(series, categorical_threshold)
        non_null = series.dropna()

        if ptype in (ParameterType.INT, ParameterType.FLOAT):
            # Guard against all-NaN columns (non_null is empty → min/max are NaN)
            if len(non_null) == 0:
                full_min, full_max = 0.0, 1.0
            else:
                full_min = float(non_null.min())
                full_max = float(non_null.max())
            # Guard against zero-width ranges (single unique value)
            if full_min == full_max:
                full_max = full_min + (1.0 if ptype == ParameterType.INT else 0.01)
            configs.append(
                ParameterConfig(
                    name=col,
                    ptype=ptype,
                    enabled=True,
                    full_min=full_min,
                    full_max=full_max,
                    allowed_subranges=[AllowedSubRange(full_min, full_max)],
                )
            )

        elif ptype == ParameterType.CATEGORICAL:
            choices = sorted(str(v) for v in non_null.unique())
            configs.append(
                ParameterConfig(
                    name=col,
                    ptype=ptype,
                    enabled=True,
                    all_choices=choices,
                    allowed_choices=choices[:],
                )
            )

        else:  # BOOL
            configs.append(
                ParameterConfig(
                    name=col,
                    ptype=ParameterType.BOOL,
                    enabled=True,
                    all_choices=["True", "False"],
                    allowed_choices=["True", "False"],
                    fixed_value=None,
                )
            )

    return configs


def load_trials_from_csv(df: pd.DataFrame, config: StudyConfig) -> List[dict]:
    """
    Convert rows in *df* that have non-null values in ALL objective columns
    into dicts suitable for adding to an Optuna study as historical trials.

    Returns
    -------
    list of {"params": {col: val, ...}, "values": [float, ...],
             "user_attrs": {"ctx_<name>": float, ...}}

    Context variable values (from config.context_variables) are stored with
    the ``ctx_`` prefix in ``user_attrs`` so the surrogate can learn from them
    without them entering the Optuna search space.

    Values list order matches config.objectives order.
    Safe to call multiple times — the caller deduplicates before adding.
    """
    obj_cols = [o.column_name for o in config.objectives]
    param_names = [p.name for p in config.parameters if p.enabled]
    ctx_cols = [c.column_name for c in getattr(config, "context_variables", [])]

    results: List[dict] = []
    for _, row in df.iterrows():
        # Skip rows with any missing objective value
        if any(pd.isna(row.get(c)) for c in obj_cols):
            continue
        params = {
            name: row[name]
            for name in param_names
            if name in row.index and not pd.isna(row[name])
        }
        # Skip rows where any enabled parameter is missing — Optuna cannot use them
        if len(params) < len([n for n in param_names if n in row.index]):
            continue
        values = [float(row[c]) for c in obj_cols]

        # Collect context variable values with ctx_ prefix
        ctx_attrs: dict = {}
        for col in ctx_cols:
            if col in row.index and not pd.isna(row[col]):
                ctx_attrs[f"ctx_{col}"] = float(row[col])

        results.append({"params": params, "values": values, "user_attrs": ctx_attrs})

    return results


def infer_context_defaults(
    df: pd.DataFrame,
    context_columns: List[str],
) -> List[Dict[str, float]]:
    """
    Return summary statistics for context variable columns.

    Used by the GUI to pre-fill the "Current Conditions" spinboxes with
    sensible default values derived from historical data.

    Parameters
    ----------
    df
        The experiment DataFrame (may contain NaN values in context columns).
    context_columns
        Names of columns that are context variables.

    Returns
    -------
    List of dicts, one per context column, with keys:
        ``column_name``, ``min``, ``max``, ``mean``
    Columns with no valid (non-NaN) data get ``min=0, max=1, mean=0``.
    """
    result = []
    for col in context_columns:
        if col not in df.columns:
            result.append({"column_name": col, "min": 0.0, "max": 1.0, "mean": 0.0})
            continue
        non_null = df[col].dropna()
        if len(non_null) == 0:
            result.append({"column_name": col, "min": 0.0, "max": 1.0, "mean": 0.0})
        else:
            result.append({
                "column_name": col,
                "min": float(non_null.min()),
                "max": float(non_null.max()),
                "mean": float(non_null.mean()),
            })
    return result


def aggregate_replicates(
    df: pd.DataFrame,
    param_cols: List[str],
    obj_cols: List[str],
    tolerance: float = 1e-6,
) -> pd.DataFrame:
    """
    Group rows where all *param_cols* values agree within *tolerance*.

    Grouping rules
    --------------
    - **Numeric** parameter columns: rounded to the nearest ``tolerance`` grid
      before grouping.  Two values ``a`` and ``b`` are considered identical when
      ``round(a / tolerance) == round(b / tolerance)``.
    - **Categorical / string** columns: must match exactly (tolerance has no
      effect on them).
    - **Objective** columns: mean of all rows in the group.
    - An ``n_replicates`` column is added with the raw row count per group.

    Special cases
    -------------
    ``tolerance <= 0``
        Returns the original DataFrame unchanged except for adding an
        ``n_replicates = 1`` column (no aggregation performed).
    Empty DataFrame
        Returns an empty copy with an integer ``n_replicates`` column.

    Parameters
    ----------
    df         : Input DataFrame.
    param_cols : Parameter column names used for grouping.
    obj_cols   : Objective column names whose values are averaged within groups.
    tolerance  : Absolute ± threshold for numeric parameters.
                 Set to 0 to disable aggregation entirely.

    Returns
    -------
    Aggregated DataFrame with the same columns as *df* plus ``n_replicates``.
    """
    if df.empty:
        result = df.copy()
        if "n_replicates" not in result.columns:
            result["n_replicates"] = pd.Series(dtype=int)
        return result

    if tolerance <= 0:
        # Aggregation disabled — return unchanged with n_replicates=1
        result = df.copy()
        result["n_replicates"] = 1
        return result

    # Separate numeric vs. categorical parameter columns
    numeric_params = [
        c for c in param_cols
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    cat_params = [c for c in param_cols if c in df.columns and c not in numeric_params]

    # Work on a copy, rounding numeric params to the tolerance grid
    work = df.copy()
    for col in numeric_params:
        work[col] = (work[col] / tolerance).round() * tolerance

    # Only group by columns that actually exist in the DataFrame
    group_cols = [c for c in param_cols if c in work.columns]
    if not group_cols:
        result = df.copy()
        result["n_replicates"] = 1
        return result

    valid_obj_cols = [c for c in obj_cols if c in work.columns]

    # Group and aggregate: mean for objectives, first for non-grouped remaining cols
    grouped = work.groupby(group_cols, dropna=False, sort=False)

    # Build per-column aggregation: mean for objectives, first for anything else
    other_cols = [
        c for c in work.columns
        if c not in group_cols and c not in valid_obj_cols
    ]
    agg_spec = {c: "mean" for c in valid_obj_cols}
    agg_spec.update({c: "first" for c in other_cols})

    if agg_spec:
        result = grouped.agg(agg_spec).reset_index()
    else:
        result = grouped.size().reset_index(name="_tmp_size_").drop(columns="_tmp_size_")

    # Restore original (un-rounded) numeric param values from the source df
    # by joining back on the rounded key — keeps the representative value clean.
    # Simpler approach: just keep the rounded values (they're within ±tolerance/2 anyway).

    # Attach replicate counts
    counts = grouped.size().reset_index(name="n_replicates")
    result = result.merge(counts, on=group_cols, how="left")
    result["n_replicates"] = result["n_replicates"].astype(int)

    return result


def append_rows_to_csv(path: str, rows: List[dict]) -> None:
    """
    Append *rows* (list of flat dicts) to the CSV at *path*.
    Creates the file if it does not exist.
    Used to persist newly submitted batch results back to the experiment CSV.
    """
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    try:
        existing = pd.read_csv(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    except FileNotFoundError:
        combined = new_df
    combined.to_csv(path, index=False)
