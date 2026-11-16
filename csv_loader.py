"""
Phase 2 — CSV Loader
Reads CSV data, infers parameter types, and extracts sensible defaults.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from parameter_config import AllowedSubRange, ObjectiveConfig, ParameterConfig, ParameterType, StudyConfig


def get_excel_sheet_names(path: str) -> List[str]:
    """Return the list of sheet names in an Excel workbook.

    Returns an empty list for non-Excel files or if openpyxl is unavailable.
    Used by the GUI to let the user choose a sheet before loading.
    """
    try:
        import os as _os
        if _os.path.splitext(path)[1].lower() not in (".xlsx", ".xls"):
            return []
        xl = pd.ExcelFile(path, engine="openpyxl")
        return list(xl.sheet_names)
    except Exception:
        return []


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a robust cleaning pipeline to a freshly-loaded DataFrame.

    Handles common issues in real-world CSV / Excel files produced by non-technical
    users without requiring any manual pre-processing:

    1. Strip whitespace from column names.
    2. Drop auto-generated "Unnamed: X" columns (Excel artefact for empty
       trailing columns saved by most spreadsheet applications).
    3. Replace Excel formula-error strings (#VALUE!, #REF!, #N/A, #NAME?,
       #NUM!, #DIV/0!, #NULL!) with NaN so they are treated as missing data.
    4. Drop rows that are entirely blank (notes, blank spacers between data blocks).
    5. Drop columns that are entirely blank (trailing empty columns).
    6. Attempt to coerce string columns that look numeric to float.
       Handles comma thousands-separators ("1,234" → 1234.0) and strips
       stray whitespace.  Only applied when ≥ 80 % of non-null values parse
       successfully, to avoid corrupting genuine categorical columns.
    7. Strip leading/trailing whitespace from all remaining string values.
    """
    # 1. Strip whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]

    # 2. Drop "Unnamed: X" columns (Excel empty-column artefact)
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)

    # 3. Replace formula-error strings with NaN
    _FORMULA_ERRORS = {"#VALUE!", "#REF!", "#N/A", "#NAME?",
                       "#NUM!", "#DIV/0!", "#NULL!"}
    df = df.replace(_FORMULA_ERRORS, float("nan"))

    # 4. Drop entirely-blank rows
    df = df.dropna(how="all").reset_index(drop=True)

    # 5. Drop entirely-blank columns
    df = df.dropna(axis=1, how="all")

    # 6. Coerce string columns that look numeric
    for col in df.columns:
        if df[col].dtype == object:
            try:
                # Strip commas (thousands separator) then attempt numeric parse
                stripped = df[col].str.replace(",", "", regex=False)
                parsed = pd.to_numeric(stripped, errors="coerce")
                n_original = int(df[col].notna().sum())
                n_parsed   = int(parsed.notna().sum())
                if n_original > 0 and n_parsed >= max(1, int(n_original * 0.8)):
                    df[col] = parsed
            except (AttributeError, TypeError):
                pass

    # 7. Strip whitespace from remaining string values
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()

    return df


def load_csv(path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Read a CSV or Excel data file and return a cleaned DataFrame.

    Supported formats
    -----------------
    * ``.csv`` — both ``','`` and ``';'`` delimiters (auto-detected).
      Handles UTF-8 BOM produced by Excel / Windows tools.
    * ``.xlsx`` / ``.xls`` — read with ``openpyxl`` (Excel workbook).
      Pass *sheet_name* to select a specific sheet; defaults to the first sheet.

    Cleaning
    --------
    After loading, :func:`_clean_dataframe` is applied automatically:
    blank rows/columns are dropped, formula-error strings become NaN,
    "Unnamed: X" columns are removed, and string-encoded numbers are coerced
    to float.  The caller receives a DataFrame ready for direct use.

    The function intentionally uses the name ``load_csv`` for backward
    compatibility; callers pass any supported path and receive a DataFrame.
    """
    import os as _os
    ext = _os.path.splitext(path)[1].lower()
    try:
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(
                path, engine="openpyxl",
                sheet_name=sheet_name if sheet_name is not None else 0,
            )
        else:
            df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    except Exception as exc:
        raise ValueError(f"Cannot read file '{path}': {exc}") from exc
    return _clean_dataframe(df)


def infer_column_type(series: pd.Series, categorical_threshold: int = 8) -> ParameterType:
    """
    Infer the ParameterType of a pandas Series.

    Decision order
    --------------
    1. BOOL        — dtype bool, or ≤ 2 unique values all in {0, 1, True, False}
    2. CATEGORICAL — object / string dtype
    3. INT         — all non-null values equal their integer cast
                     (checked BEFORE categorical threshold so that INT columns
                     with few distinct levels, e.g. [1, 2, 3, 4], are not
                     wrongly promoted to CATEGORICAL)
    4. CATEGORICAL — numeric, non-integer, n_unique ≤ categorical_threshold
    5. FLOAT       — everything else
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

    # 3. All values equal their int cast → INT (checked BEFORE categorical threshold)
    #    This ensures integer columns with few unique levels (e.g. [1, 2, 3, 4])
    #    are not mis-classified as CATEGORICAL.
    try:
        as_int = non_null.values.astype(int).astype(float)
        if np.allclose(non_null.values.astype(float), as_int, atol=0, rtol=0):
            return ParameterType.INT
    except (ValueError, TypeError, OverflowError):
        pass

    # 4. Few unique non-integer numeric values → categorical
    n_unique = non_null.nunique()
    if n_unique <= categorical_threshold:
        return ParameterType.CATEGORICAL

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


def load_trials_from_csv(
    df: pd.DataFrame,
    config: StudyConfig,
    skip_log: Optional[List[dict]] = None,
) -> List[dict]:
    """
    Convert rows in *df* that have non-null values in ALL objective columns
    into dicts suitable for adding to an Optuna study as historical trials.

    Parameters
    ----------
    df
        The experiment DataFrame (after cleaning).
    config
        Study configuration; drives which columns are objectives / parameters.
    skip_log
        Optional list that is populated in-place with one entry per skipped row.
        Each entry is a dict with keys:

        * ``row``    — 1-based row number in the original DataFrame
        * ``reason`` — short human-readable reason string
        * ``detail`` — longer explanation (which columns, etc.)

        Three skip reasons are tracked:

        * ``"Missing objective"`` — the objective column has no value (the
          experiment was planned but not yet run).
        * ``"Missing parameter"`` — one or more enabled input parameter columns
          are blank (Optuna cannot use this row).
        * ``"Duplicate"`` — the parameter values are identical to those of an
          already-loaded row (deduplicated automatically).

    Returns
    -------
    list of {"params": {col: val, ...}, "values": [float, ...],
             "user_attrs": {"ctx_<name>": float, ...},
             "boundary_values": {"bv_col": float, ...}}

    Values list order matches config.objectives order.
    Safe to call multiple times — the caller deduplicates before adding.
    """
    obj_cols    = [o.column_name for o in config.objectives]
    param_names = [p.name for p in config.parameters if p.enabled]
    ctx_cols    = [c.column_name for c in getattr(config, "context_variables", [])]
    bnd_cols    = [bv.column_name for bv in getattr(config, "boundary_variables", [])]

    results: List[dict] = []
    # Track fingerprints of already-loaded rows so duplicates can be reported
    # with the row number of the original that this one duplicates.
    seen_fingerprints: dict = {}   # fingerprint → 1-based row number

    for row_idx, (_, row) in enumerate(df.iterrows()):
        display_row = row_idx + 1   # 1-based for user messages

        # ── 1. Check for missing objective values ──────────────────────────
        missing_obj = [c for c in obj_cols if pd.isna(row.get(c))]
        if missing_obj:
            if skip_log is not None:
                plural = len(missing_obj) > 1
                skip_log.append({
                    "row": display_row,
                    "reason": "Missing objective",
                    "detail": (
                        f"Column{'s' if plural else ''} "
                        f"{', '.join(repr(c) for c in missing_obj)} "
                        f"{'have' if plural else 'has'} no value — "
                        "the experiment may not have been run yet."
                    ),
                })
            continue

        # ── 2. Check for missing enabled parameter values ──────────────────
        params = {
            name: row[name]
            for name in param_names
            if name in row.index and not pd.isna(row[name])
        }
        missing_params = [
            n for n in param_names
            if n in row.index and pd.isna(row[n])
        ]
        if len(params) < len([n for n in param_names if n in row.index]):
            if skip_log is not None:
                plural = len(missing_params) > 1
                skip_log.append({
                    "row": display_row,
                    "reason": "Missing parameter",
                    "detail": (
                        f"Column{'s' if plural else ''} "
                        f"{', '.join(repr(c) for c in missing_params)} "
                        f"{'are' if plural else 'is'} blank — "
                        "all enabled parameter columns must have a value."
                    ),
                })
            continue

        # ── 3. Check for duplicate parameter fingerprint ───────────────────
        fp = str(sorted((k, str(v)) for k, v in params.items()))
        if fp in seen_fingerprints:
            if skip_log is not None:
                skip_log.append({
                    "row": display_row,
                    "reason": "Duplicate",
                    "detail": (
                        f"Identical parameter values to row {seen_fingerprints[fp]}. "
                        "Only the first occurrence is loaded."
                    ),
                })
            continue
        seen_fingerprints[fp] = display_row

        # ── Build trial dict ───────────────────────────────────────────────
        values = [float(row[c]) for c in obj_cols]

        ctx_attrs: dict = {}
        for col in ctx_cols:
            if col in row.index and not pd.isna(row[col]):
                ctx_attrs[f"ctx_{col}"] = float(row[col])

        bnd_vals: dict = {}
        for col in bnd_cols:
            if col in row.index and not pd.isna(row[col]):
                try:
                    bnd_vals[col] = float(row[col])
                except (TypeError, ValueError):
                    pass

        trial_dict: dict = {
            "params": params,
            "values": values,
            "user_attrs": ctx_attrs,
        }
        if bnd_vals:
            trial_dict["boundary_values"] = bnd_vals

        results.append(trial_dict)

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
