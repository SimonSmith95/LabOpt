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
          pearson_r (float | None),
          y_true_holdout (np.ndarray | None), y_pred_holdout (np.ndarray | None)

predict_batch(study, config, suggestions) -> list[float | None]
    Train an RF on all completed trials and return predicted first-objective
    values for each dict in *suggestions*.  Returns None entries if there is
    insufficient data (< MIN_TRIALS) or a prediction fails.

Thresholds
----------
  R² > 0.75  -> "good"      (green)
  R² 0.50-0.75 -> "moderate"  (amber)
  R² < 0.50  -> "poor"      (red)
  < MIN_TRIALS completed   -> "insufficient" (grey)
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold, cross_val_score

from parameter_config import ParameterConfig, ParameterType, StudyConfig


MIN_TRIALS = 10           # minimum completed trials required for a rating

# Five-tier R² thresholds (descending)
STRONG_THRESHOLD   = 0.85  # R² above this -> "strong"
GOOD_THRESHOLD     = 0.70  # R² above this -> "good"
USEFUL_THRESHOLD   = 0.50  # R² above this -> "useful"
MODERATE_THRESHOLD = 0.30  # R² above this -> "moderate"
# R² <= 0.30 -> "weak"


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


def _build_X(
    trials_params: List[dict],
    config: StudyConfig,
    ctx_attrs_list: Optional[List[dict]] = None,
) -> np.ndarray:
    """
    Build a numeric feature matrix from a list of param dicts.

    - INT / FLOAT  : stored as-is (NaN -> -1 sentinel)
    - CATEGORICAL  : label-encoded; string sort is *numeric-aware* for float-
                     valued categoricals so codes preserve order
    - Multi-range  : reconstructed via _get_param_value
    - Missing      : filled with -1

    Parameters
    ----------
    trials_params
        List of Optuna trial.params dicts.
    config
        StudyConfig -- defines enabled parameters and context variables.
    ctx_attrs_list
        Optional list of trial.user_attrs dicts (one per trial).  When
        supplied, context variable values (stored as ``ctx_<name>`` keys)
        are appended as extra numeric columns to the feature matrix.
        Entries that are missing or non-numeric are imputed with -1.
    """
    enabled = [p for p in config.parameters if p.enabled]
    ctx_vars = getattr(config, "context_variables", [])

    rows = []
    for i, params in enumerate(trials_params):
        row = {}
        for p in enabled:
            row[p.name] = _get_param_value(params, p)

        # Append context variable values
        if ctx_attrs_list is not None and ctx_vars:
            uattrs = ctx_attrs_list[i] if i < len(ctx_attrs_list) else {}
            for cv in ctx_vars:
                key = f"ctx_{cv.column_name}"
                raw = uattrs.get(key)
                try:
                    row[cv.column_name] = float(raw) if raw is not None else None
                except (TypeError, ValueError):
                    row[cv.column_name] = None

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

    # Flip sign for maximise so RF always sees a minimisation target -- this
    # gives a more interpretable R² (correct direction of fit).
    if config.objectives[0].direction == "maximize":
        y = -y

    return y


# ── Public API ────────────────────────────────────────────────────────────────

def compute_surrogate_quality(study, config: StudyConfig) -> dict:
    """
    Evaluate how well a Random Forest can predict the first objective using
    repeated k-fold cross-validation (5 splits x 5 repeats = 25 estimates).

    RepeatedKFold is used instead of a single KFold pass to produce a stable
    R² estimate that is not affected by a single unlucky random split.

    Parameters
    ----------
    study  : optuna.Study
    config : StudyConfig

    Returns
    -------
    dict with keys:
        status          : "good" | "moderate" | "poor" | "insufficient"
        n               : number of completed trials used
        r2              : mean repeated-CV R², clipped to [-1, 1] (None if insufficient)
        pearson_r       : Pearson r between CV predictions and true values (None if insufficient)
        rmse            : training RMSE on full dataset (None if insufficient)
        y_true_holdout  : np.ndarray of hold-out true values (None if insufficient)
        y_pred_holdout  : np.ndarray of hold-out RF predictions (None if insufficient)
    """
    from optuna.trial import TrialState
    from scipy.stats import pearsonr

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    n = len(completed)

    if n < MIN_TRIALS:
        return {"status": "insufficient", "n": n,
                "r2": None, "rmse": None, "pearson_r": None,
                "y_true_holdout": None, "y_pred_holdout": None}

    # Feature matrix (params + context user_attrs)
    trial_params = [t.params for t in completed]
    ctx_attrs = [dict(t.user_attrs) for t in completed]
    X = _build_X(trial_params, config, ctx_attrs_list=ctx_attrs)

    # Target vector
    y = _build_y(completed, config)
    valid_mask = ~np.isnan(y)
    X, y = X[valid_mask], y[valid_mask]

    if len(y) < MIN_TRIALS:
        return {"status": "insufficient", "n": len(y),
                "r2": None, "rmse": None, "pearson_r": None,
                "y_true_holdout": None, "y_pred_holdout": None}

    # Repeated cross-validated R²
    # RepeatedKFold (5 splits x 5 repeats = 25 scores) gives a stable
    # estimate unaffected by any single unlucky partition.
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
    scores = cross_val_score(rf, X, y, cv=rkf, scoring="r2")
    r2 = float(np.clip(np.mean(scores), -1.0, 1.0))

    # Training RMSE and Pearson r
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

    # Five-tier status
    if r2 > STRONG_THRESHOLD:
        status = "strong"
    elif r2 > GOOD_THRESHOLD:
        status = "good"
    elif r2 > USEFUL_THRESHOLD:
        status = "useful"
    elif r2 > MODERATE_THRESHOLD:
        status = "moderate"
    else:
        status = "weak"

    # 70/30 hold-out scatter data for predicted-vs-actual plots.
    # Computed here so SamplerComparisonWidget can display the scatter
    # without a separate training call.
    y_true_ho: Optional[np.ndarray] = None
    y_pred_ho: Optional[np.ndarray] = None
    try:
        from sklearn.model_selection import train_test_split as _tts
        X_tr, X_te, y_tr, y_te = _tts(X, y, test_size=0.30, random_state=42)
        if len(X_tr) >= 3 and len(X_te) >= 1:
            rf_ho = RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=1
            )
            rf_ho.fit(X_tr, y_tr)
            y_pred_ho = rf_ho.predict(X_te)
            y_true_ho = y_te
    except Exception:
        pass

    return {"status": status, "n": n,
            "r2": r2, "rmse": rmse, "pearson_r": pearson_r,
            "y_true_holdout": y_true_ho, "y_pred_holdout": y_pred_ho}


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


# ── Sampler comparison on real data ──────────────────────────────────────────

def compare_samplers_on_data(
    study,
    config: StudyConfig,
    sampler_names: List[str],
    n_sim_trials: int = 30,
    seed: int = 42,
    progress_cb=None,
) -> dict:
    """
    Compare BO samplers by running a *virtual* forward BO loop on real data.

    For each sampler:
      1. Build a fresh in-memory Optuna study seeded with all completed trials.
      2. Ask the sampler for *n_sim_trials* new suggestions.
      3. Evaluate each suggestion using an RF surrogate trained on the real data.
      4. Tell the values back to the study.
      5. Record running best (or hypervolume for multi-objective) at each step.

    Parameters
    ----------
    study         : optuna.Study -- the active study with completed trials.
    config        : StudyConfig
    sampler_names : list of sampler key strings, e.g. ["TPE", "Random", "CmaEs"]
                    Supported: "TPE", "Random", "CmaEs", "NSGAii", "QMC"
    n_sim_trials  : number of simulated forward BO steps per sampler.
    seed          : random seed.
    progress_cb   : optional callable(sampler_name: str, pct: int) for progress.

    Returns
    -------
    dict keyed by sampler_name, each value is a dict:
        {
          "convergence":    list[float],      running best per trial (single-obj)
                                              OR running hypervolume (multi-obj)
          "final_best":     float | None,     last value in convergence
          "hypervolume":    float | None,     final HV for multi-obj, else None
          "r2":             float | None,     CV R² from compute_surrogate_quality
          "rmse":           float | None,
          "pearson_r":      float | None,
          "status":         str,              "good"|"moderate"|"poor"|"insufficient"
          "y_true_holdout": np.ndarray|None,  hold-out true values for scatter plot
          "y_pred_holdout": np.ndarray|None,  hold-out RF predictions for scatter plot
        }
    """
    import optuna as _optuna
    from optuna.trial import TrialState
    from sklearn.ensemble import RandomForestRegressor
    from benchmark_math import compute_hypervolume

    _optuna.logging.set_verbosity(_optuna.logging.ERROR)

    # Completed real trials
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    n_real = len(completed)
    is_multi = len(config.objectives) > 1

    # Train surrogate on real data (shared across all samplers)
    trial_params = [t.params for t in completed]
    ctx_attrs    = [dict(t.user_attrs) for t in completed]
    X_train = _build_X(trial_params, config, ctx_attrs_list=ctx_attrs)

    # For multi-objective: build a surrogate per objective
    y_trains: List[Optional[np.ndarray]] = []
    rfs: List[Optional[RandomForestRegressor]] = []

    for obj_idx, obj in enumerate(config.objectives):
        y = np.array([
            t.values[obj_idx] if (t.values and obj_idx < len(t.values)) else np.nan
            for t in completed
        ], dtype=float)
        if obj.direction == "maximize":
            y = -y
        valid = np.isfinite(y)
        if valid.sum() < MIN_TRIALS:
            y_trains.append(None)
            rfs.append(None)
            continue
        X_v, y_v = X_train[valid], y[valid]
        rf = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1)
        rf.fit(X_v, y_v)
        y_trains.append(y_v)
        rfs.append(rf)

    # Helper: predict objective values for a param dict
    def _predict(params_dict: dict) -> List[float]:
        X_pred = _build_X([params_dict], config)
        preds = []
        for obj_idx, (rf, obj) in enumerate(zip(rfs, config.objectives)):
            if rf is None:
                preds.append(float("nan"))
                continue
            p = float(rf.predict(X_pred)[0])
            if obj.direction == "maximize":
                p = -p
            preds.append(p)
        return preds

    # Build sampler objects
    def _make_sampler(name: str):
        if name == "TPE":
            return _optuna.samplers.TPESampler(seed=seed)
        elif name == "Random":
            return _optuna.samplers.RandomSampler(seed=seed)
        elif name == "CmaEs":
            return _optuna.samplers.CmaEsSampler(seed=seed)
        elif name == "NSGAii":
            return _optuna.samplers.NSGAIISampler(seed=seed)
        elif name == "QMC":
            return _optuna.samplers.QMCSampler(seed=seed)
        else:
            return _optuna.samplers.TPESampler(seed=seed)

    # Reference point for hypervolume
    if is_multi:
        all_real_vals = [
            t.values for t in completed
            if t.values and len(t.values) == len(config.objectives)
        ]
        if all_real_vals:
            ref_pt = list(np.array(all_real_vals).max(axis=0) * 1.1 + 1e-6)
        else:
            ref_pt = [1.1] * len(config.objectives)
    else:
        ref_pt = None

    # Per-sampler quality metrics + holdout scatter (computed once, shared)
    sq = compute_surrogate_quality(study, config)

    results: dict = {}
    n_samplers = len(sampler_names)

    for s_idx, s_name in enumerate(sampler_names):
        if progress_cb:
            progress_cb(s_name, int(100 * s_idx / n_samplers))

        try:
            # Clone study distributions from any completed trial
            if not completed:
                results[s_name] = {
                    "convergence": [], "final_best": None,
                    "hypervolume": None, "r2": sq["r2"],
                    "rmse": sq["rmse"], "pearson_r": sq["pearson_r"],
                    "status": sq["status"],
                    "y_true_holdout": sq.get("y_true_holdout"),
                    "y_pred_holdout": sq.get("y_pred_holdout"),
                }
                continue

            ref_trial = completed[0]
            distributions = ref_trial.distributions

            # Create fresh in-memory study with the REAL objective directions so
            # model-based samplers (TPE, CmaEs) optimise in the correct direction.
            sim_directions = [obj.direction for obj in config.objectives]
            sim_study = _optuna.create_study(
                directions=sim_directions,
                sampler=_make_sampler(s_name),
            )

            # Seed with all real trials
            for t in completed:
                sim_study.add_trial(
                    _optuna.trial.create_trial(
                        params=t.params,
                        distributions=t.distributions,
                        values=list(t.values) if t.values else [0.0],
                    )
                )

            # Forward BO loop
            convergence: List[float] = []
            all_obj_vals: List[List[float]] = []

            # Direction for single-objective convergence tracking
            dir0 = config.objectives[0].direction if config.objectives else "minimize"
            is_minimize = (dir0 == "minimize")

            # Track running best for single-obj.  Use None until the first
            # finite surrogate prediction is obtained -- this avoids +/-inf
            # sentinel values (which break matplotlib autoscaling).
            best_so_far = None

            for _ in range(n_sim_trials):
                try:
                    trial = sim_study.ask(distributions)
                    params_dict = dict(trial.params)
                    preds = _predict(params_dict)
                    sim_study.tell(trial, preds)
                except Exception:
                    # Keep last valid value; if none yet, skip this trial.
                    if convergence:
                        convergence.append(convergence[-1])
                    continue

                if is_multi:
                    all_obj_vals.append(preds)
                    hv = compute_hypervolume(all_obj_vals, ref_pt)
                    convergence.append(hv)
                else:
                    val = preds[0] if preds else float("nan")
                    if np.isfinite(val):
                        if best_so_far is None:
                            best_so_far = val
                        elif is_minimize:
                            best_so_far = min(best_so_far, val)
                        else:
                            best_so_far = max(best_so_far, val)
                    if best_so_far is not None:
                        convergence.append(best_so_far)

            final_best = convergence[-1] if convergence else None
            hv_final = convergence[-1] if (is_multi and convergence) else None

            results[s_name] = {
                "convergence":    convergence,
                "final_best":     final_best,
                "hypervolume":    hv_final,
                "r2":             sq["r2"],
                "rmse":           sq["rmse"],
                "pearson_r":      sq["pearson_r"],
                "status":         sq["status"],
                "y_true_holdout": sq.get("y_true_holdout"),
                "y_pred_holdout": sq.get("y_pred_holdout"),
            }

        except Exception as exc:
            results[s_name] = {
                "convergence":    [],
                "final_best":     None,
                "hypervolume":    None,
                "r2":             None,
                "rmse":           None,
                "pearson_r":      None,
                "status":         "error",
                "error":          str(exc),
                "y_true_holdout": None,
                "y_pred_holdout": None,
            }

    if progress_cb:
        progress_cb("done", 100)

    return results
