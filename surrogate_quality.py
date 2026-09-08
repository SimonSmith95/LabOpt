"""
Surrogate Quality Indicator (Feature 2)
========================================
Computes a repeated k-fold cross-validated R² and Pearson r using a Random
Forest trained on all completed trials.  The result is used to show a coloured
quality badge in the Results tab.

Public API
----------
compute_surrogate_quality(study, config) -> dict
    Keys: status ("good" | "moderate" | "poor" | "insufficient"),
          n (int), r2 (float | None), rmse (float | None),
          pearson_r (float | None)

predict_batch(study, config, suggestions) -> list[float | None]
    Train an RF on all completed trials and return predicted first-objective
    values for each dict in *suggestions*.  Returns None entries if there is
    insufficient data (< MIN_TRIALS) or a prediction fails.

Thresholds
----------
  R² > 0.75  → "good"      (green)
  R² 0.50–0.75 → "moderate"  (amber)
  R² < 0.50  → "poor"      (red)
  < MIN_TRIALS completed   → "insufficient" (grey)
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold, cross_val_score

from parameter_config import ParameterConfig, ParameterType, StudyConfig


MIN_TRIALS = 10           # minimum completed trials required for a rating
GOOD_THRESHOLD = 0.75     # R² above this → "good"
MODERATE_THRESHOLD = 0.50 # R² above this → "moderate"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_param_value(trial_params: dict, param: ParameterConfig):
    """
    Extract a parameter value from an Optuna trial's params dict.

    Handles both single-range parameters (stored under ``param.name``) and
    multi-range parameters (stored under ``param.name__in_range_X``).
    Returns None if the parameter is not present.
    """
    # Single-range: stored directly
    if param.name in trial_params:
        return trial_params[param.name]
    # Multi-range: find the in_range key
    prefix = f"{param.name}__in_range_"
    for key, val in trial_params.items():
        if key.startswith(prefix):
            return val
    return None


def _build_X(trials_params: List[dict], config: StudyConfig) -> np.ndarray:
    """
    Build a numeric feature matrix from a list of param dicts.

    - INT / FLOAT  : stored as-is (NaN → -1 sentinel)
    - CATEGORICAL  : label-encoded; string sort is *numeric-aware* for float-
                     valued categoricals so codes preserve order
    - Multi-range  : reconstructed via _get_param_value
    - Missing      : filled with -1
    """
    enabled = [p for p in config.parameters if p.enabled]
    rows = []
    for params in trials_params:
        row = {}
        for p in enabled:
            row[p.name] = _get_param_value(params, p)
        rows.append(row)

    X_df = pd.DataFrame(rows)

    for p in enabled:
        col = X_df[p.name]
        if p.ptype == ParameterType.CATEGORICAL:
            # Sort choices numerically if possible, otherwise lexicographically
            unique = list(col.dropna().unique())
            try:
                sorted_choices = sorted(unique, key=lambda x: float(x))
            except (ValueError, TypeError):
                sorted_choices = sorted(unique, key=lambda x: str(x))
            order_map = {v: i for i, v in enumerate(sorted_choices)}
            X_df[p.name] = col.map(order_map)

    return X_df.fillna(-1).values.astype(float)


def _build_y(completed, config: StudyConfig) -> np.ndarray:
    """Extract the first-objective values from completed Optuna trials."""
    y = np.array([
        t.values[0] if t.values else (t.value if t.value is not None else np.nan)
        for t in completed
    ], dtype=float)

    # Flip sign for maximise so RF always sees a minimisation target — this
    # gives a more interpretable R² (correct direction of fit).
    if config.objectives[0].direction == "maximize":
        y = -y

    return y


# ── Public API ────────────────────────────────────────────────────────────────

def compute_surrogate_quality(study, config: StudyConfig) -> dict:
    """
    Evaluate how well a Random Forest can predict the first objective using
    repeated k-fold cross-validation (5 splits × 5 repeats = 25 estimates).

    RepeatedKFold is used instead of a single KFold pass to produce a stable
    R² estimate that is not affected by a single unlucky random split.

    Parameters
    ----------
    study  : optuna.Study
    config : StudyConfig

    Returns
    -------
    dict with keys:
        status    : "good" | "moderate" | "poor" | "insufficient"
        n         : number of completed trials used
        r2        : mean repeated-CV R²,  clipped to [-1, 1]  (None if insufficient)
        pearson_r : Pearson r between CV predictions and true values (None if insufficient)
        rmse      : training RMSE on full dataset (None if insufficient)
    """
    from optuna.trial import TrialState
    from scipy.stats import pearsonr

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    n = len(completed)

    if n < MIN_TRIALS:
        return {"status": "insufficient", "n": n,
                "r2": None, "rmse": None, "pearson_r": None}

    # ── Feature matrix ─────────────────────────────────────────────────────
    trial_params = [t.params for t in completed]
    X = _build_X(trial_params, config)

    # ── Target vector ──────────────────────────────────────────────────────
    y = _build_y(completed, config)
    valid_mask = ~np.isnan(y)
    X, y = X[valid_mask], y[valid_mask]

    if len(y) < MIN_TRIALS:
        return {"status": "insufficient", "n": len(y),
                "r2": None, "rmse": None, "pearson_r": None}

    # ── Repeated cross-validated R² ────────────────────────────────────────
    # RepeatedKFold (5 splits × 5 repeats = 25 scores) gives a stable
    # estimate unaffected by any single unlucky partition.
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
    scores = cross_val_score(rf, X, y, cv=rkf, scoring="r2")
    r2 = float(np.clip(np.mean(scores), -1.0, 1.0))

    # ── Training RMSE and Pearson r ────────────────────────────────────────
    rf.fit(X, y)
    y_pred = rf.predict(X)
    rmse = float(np.sqrt(np.mean((y_pred - y) ** 2)))

    # Use leave-one-out CV predictions for an honest Pearson r
    # (avoids over-optimism from training Pearson r)
    try:
        from sklearn.model_selection import cross_val_predict, KFold
        y_cv_pred = cross_val_predict(
            RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1),
            X, y,
            cv=KFold(n_splits=5, shuffle=True, random_state=0),
        )
        pearson_r, _ = pearsonr(y, y_cv_pred)
        pearson_r = float(np.clip(pearson_r, -1.0, 1.0))
    except Exception:
        pearson_r = None

    # ── Traffic-light status ───────────────────────────────────────────────
    if r2 > GOOD_THRESHOLD:
        status = "good"
    elif r2 > MODERATE_THRESHOLD:
        status = "moderate"
    else:
        status = "poor"

    return {"status": status, "n": n,
            "r2": r2, "rmse": rmse, "pearson_r": pearson_r}


def predict_batch(
    study,
    config: StudyConfig,
    suggestions: List[dict],
) -> List[Optional[float]]:
    """
    Train an RF on all completed trials and return predicted first-objective
    values for each suggestion.

    Parameters
    ----------
    study       : optuna.Study
    config      : StudyConfig
    suggestions : list of param-value dicts (app-level names, not Optuna internal)

    Returns
    -------
    List of floats (one per suggestion).  An entry is None if the suggestion's
    feature vector cannot be built or if there is insufficient training data.
    The sign convention follows the objective direction (values are always in
    the original objective space, not sign-flipped).
    """
    from optuna.trial import TrialState

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if len(completed) < MIN_TRIALS:
        return [None] * len(suggestions)

    trial_params = [t.params for t in completed]
    X_train = _build_X(trial_params, config)
    y_train = _build_y(completed, config)

    valid_mask = ~np.isnan(y_train)
    X_train, y_train = X_train[valid_mask], y_train[valid_mask]

    if len(y_train) < MIN_TRIALS:
        return [None] * len(suggestions)

    try:
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
        rf.fit(X_train, y_train)
    except Exception:
        return [None] * len(suggestions)

    results: List[Optional[float]] = []
    for sug in suggestions:
        try:
            X_pred = _build_X([sug], config)
            pred = float(rf.predict(X_pred)[0])
            # Un-flip if we flipped for "maximize"
            if config.objectives[0].direction == "maximize":
                pred = -pred
            results.append(pred)
        except Exception:
            results.append(None)

    return results
