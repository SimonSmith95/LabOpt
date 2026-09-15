"""
Contextual Bayesian Optimisation — Surrogate & Acquisition Module
=================================================================
Provides ``ContextualSurrogate``, an RF-based acquisition function that
conditions suggestions on fixed, uncontrollable environmental context
(e.g. ambient humidity, atmospheric pressure).

The surrogate trains on (controllable_params + context_vars) → objective,
but only optimises over controllable_params at suggestion time — context is
held fixed to the user-provided current conditions.

Design notes
------------
* No PySide6 dependency — fully usable in headless scripting.
* Requires sklearn ≥ 1.4 and scipy ≥ 1.7 (both in requirements.txt).
* Minimal data threshold: MIN_TRIALS (15) completed trials required.
* Missing context values in historical trials are imputed with the column
  mean (same imputation is applied at suggestion time using training stats).
* Multi-objective: one RF per objective, EI scores multiplied together.
"""
from __future__ import annotations

import json
import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
from optuna.trial import TrialState

from parameter_config import ParameterConfig, ParameterType, StudyConfig

logger = logging.getLogger(__name__)

# Minimum completed trials needed before the contextual sampler activates
MIN_TRIALS = 15

# Number of Latin-Hypercube candidates to generate during suggest()
N_CANDIDATES = 2000


# ──────────────────────────────────────────────────────────────────────────────
# ContextualSurrogate
# ──────────────────────────────────────────────────────────────────────────────

class ContextualSurrogate:
    """
    Random Forest surrogate that includes context variables as features.

    Usage
    -----
    ::

        surrogate = ContextualSurrogate(config)
        if surrogate.fit(study):
            suggestions = surrogate.suggest({"humidity": 65.0}, n_return=3)
            # suggestions: [{"temperature": 152.4, ...}, ...]

    Parameters
    ----------
    config : StudyConfig
        Full study configuration.  Must contain at least one entry in
        ``config.context_variables`` for context-aware suggestions.
    """

    def __init__(self, config: StudyConfig) -> None:
        self.config = config
        # One RF model per objective (for multi-objective support)
        self._models: List = []           # List[RandomForestRegressor]
        self._feature_names: List[str] = []   # controllable param names + ctx names
        self._ctx_names: List[str] = []        # context variable column names
        self._ctx_means: Dict[str, float] = {} # imputation means for missing ctx
        self._y_bests: List[float] = []        # best value per objective
        self._directions: List[str] = []       # "minimize" / "maximize" per obj
        self._param_bounds: List[Tuple] = []   # (low, high, ptype) per enabled param
        self._is_fitted: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, study: optuna.Study) -> bool:
        """
        Fit one Random Forest per objective on all COMPLETE trials.

        Parameters
        ----------
        study : optuna.Study

        Returns
        -------
        True  if there were ≥ MIN_TRIALS completed trials and fitting succeeded.
        False if there is insufficient data (caller should fall back to standard
              Optuna sampling).
        """
        from sklearn.ensemble import RandomForestRegressor

        complete_trials = [
            t for t in study.trials if t.state == TrialState.COMPLETE
        ]
        if len(complete_trials) < MIN_TRIALS:
            logger.debug(
                "ContextualSurrogate.fit: only %d complete trials, need %d.",
                len(complete_trials), MIN_TRIALS,
            )
            return False

        # ── Feature names ─────────────────────────────────────────────────────
        enabled_params = [p for p in self.config.parameters if p.enabled]
        self._ctx_names = [c.column_name for c in self.config.context_variables]
        self._feature_names = (
            [p.name for p in enabled_params] + self._ctx_names
        )

        # ── Build feature matrix X and target matrix Y ───────────────────────
        n_params = len(enabled_params)
        n_ctx = len(self._ctx_names)
        n_obj = len(self.config.objectives)

        X_rows: List[List[float]] = []
        Y_rows: List[List[float]] = []

        for trial in complete_trials:
            # Controllable param values (decoded to external names)
            ext_params = _decode_trial_params(trial, self.config)
            row_x: List[float] = []
            for p in enabled_params:
                val = ext_params.get(p.name)
                if val is None:
                    row_x.append(float("nan"))
                else:
                    try:
                        row_x.append(float(val))
                    except (TypeError, ValueError):
                        row_x.append(float("nan"))

            # Context values from user_attrs
            for ctx_col in self._ctx_names:
                key = f"ctx_{ctx_col}"
                raw = trial.user_attrs.get(key)
                if raw is None:
                    row_x.append(float("nan"))
                else:
                    try:
                        row_x.append(float(raw))
                    except (TypeError, ValueError):
                        row_x.append(float("nan"))

            # Objective values
            if n_obj == 1:
                y_val = trial.value
                if y_val is None or (isinstance(y_val, float) and math.isnan(y_val)):
                    continue
                row_y = [float(y_val)]
            else:
                if trial.values is None or len(trial.values) < n_obj:
                    continue
                row_y = [float(v) for v in trial.values[:n_obj]]

            X_rows.append(row_x)
            Y_rows.append(row_y)

        if len(X_rows) < MIN_TRIALS:
            return False

        X = np.array(X_rows, dtype=float)  # (n_trials, n_features)
        Y = np.array(Y_rows, dtype=float)  # (n_trials, n_obj)

        # ── Impute NaN with column means ─────────────────────────────────────
        col_means = np.nanmean(X, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        # Store ctx column means for imputation at suggest() time
        for i, ctx_col in enumerate(self._ctx_names):
            feat_idx = n_params + i
            self._ctx_means[ctx_col] = float(col_means[feat_idx])

        # ── Store param bounds for candidate generation ────────────────────────
        self._param_bounds = []
        for p in enabled_params:
            if p.ptype in (ParameterType.INT, ParameterType.FLOAT):
                lo = p.full_min
                hi = p.full_max
                # Use allowed_subranges extremes if available
                if p.allowed_subranges:
                    lo = min(r.low for r in p.allowed_subranges)
                    hi = max(r.high for r in p.allowed_subranges)
            elif p.ptype in (ParameterType.CATEGORICAL, ParameterType.BOOL):
                choices = p.allowed_choices or p.all_choices
                lo, hi = 0.0, max(0.0, float(len(choices) - 1))
            else:
                lo, hi = 0.0, 1.0
            self._param_bounds.append((lo, hi, p.ptype))

        # ── Fit one RF per objective ──────────────────────────────────────────
        self._models = []
        self._y_bests = []
        self._directions = [o.direction for o in self.config.objectives]

        for obj_idx in range(n_obj):
            y_col = Y[:, obj_idx]
            rf = RandomForestRegressor(
                n_estimators=100,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=1,
            )
            rf.fit(X, y_col)
            self._models.append(rf)

            direction = self._directions[obj_idx]
            if direction == "minimize":
                self._y_bests.append(float(np.min(y_col)))
            else:
                self._y_bests.append(float(np.max(y_col)))

        self._is_fitted = True
        logger.debug(
            "ContextualSurrogate fitted on %d trials, %d features, %d objectives.",
            len(X_rows), len(self._feature_names), n_obj,
        )
        return True

    def suggest(
        self,
        current_context: dict,
        n_return: int = 1,
        n_candidates: int = N_CANDIDATES,
    ) -> List[dict]:
        """
        Return *n_return* controllable parameter dicts optimised for the given
        *current_context* using Expected Improvement.

        Parameters
        ----------
        current_context : dict
            Map of context column name → observed float value.
            e.g. ``{"humidity_pct": 65.0, "atm_pressure_hpa": 1013.2}``
        n_return : int
            Number of suggestions to return.
        n_candidates : int
            Number of Latin-Hypercube candidates to evaluate.

        Returns
        -------
        List of ``{param_name: value, ...}`` dicts, sorted by EI (best first).
        Falls back to random sampling if fitting has not been done.
        """
        if not self._is_fitted:
            logger.warning("ContextualSurrogate.suggest() called before fit().")
            return []

        enabled_params = [p for p in self.config.parameters if p.enabled]
        n_params = len(enabled_params)

        # ── Generate LHS candidates over controllable parameter space ─────────
        try:
            from scipy.stats.qmc import LatinHypercube
            sampler = LatinHypercube(d=n_params, seed=42)
            lhs = sampler.random(n=n_candidates)  # (n_candidates, n_params) in [0,1]
        except Exception:
            # Fallback to uniform random if scipy LHS not available
            rng = np.random.default_rng(42)
            lhs = rng.random((n_candidates, n_params))

        # Scale to parameter ranges
        candidates = np.empty_like(lhs)
        for j, (lo, hi, ptype) in enumerate(self._param_bounds):
            scaled = lo + lhs[:, j] * (hi - lo)
            if ptype == ParameterType.INT:
                scaled = np.round(scaled).clip(lo, hi)
            candidates[:, j] = scaled

        # ── Build feature matrix with context fixed ────────────────────────────
        n_ctx = len(self._ctx_names)
        ctx_row = np.empty(n_ctx)
        for i, ctx_col in enumerate(self._ctx_names):
            raw = current_context.get(ctx_col)
            if raw is None:
                ctx_row[i] = self._ctx_means.get(ctx_col, 0.0)
            else:
                try:
                    ctx_row[i] = float(raw)
                except (TypeError, ValueError):
                    ctx_row[i] = self._ctx_means.get(ctx_col, 0.0)

        # Tile context across all candidates
        ctx_block = np.tile(ctx_row, (n_candidates, 1))   # (n_candidates, n_ctx)
        X_cand = np.hstack([candidates, ctx_block])        # (n_candidates, n_features)

        # ── Score via EI (product across objectives for multi-obj) ─────────────
        ei_combined = np.ones(n_candidates)
        for obj_idx, rf in enumerate(self._models):
            # Tree-ensemble uncertainty via std of individual tree predictions
            tree_preds = np.array(
                [tree.predict(X_cand) for tree in rf.estimators_],
                dtype=float,
            )  # (n_estimators, n_candidates)
            mu = tree_preds.mean(axis=0)
            sigma = tree_preds.std(axis=0) + 1e-9   # avoid division by zero

            ei = self._ei(mu, sigma, self._y_bests[obj_idx], self._directions[obj_idx])
            ei_combined *= ei

        # ── Rank and decode top candidates ─────────────────────────────────────
        n_return = max(1, min(n_return, n_candidates))
        top_indices = np.argsort(ei_combined)[::-1][:n_return]

        results: List[dict] = []
        for idx in top_indices:
            param_dict: dict = {}
            for j, p in enumerate(enabled_params):
                val = float(candidates[idx, j])
                if p.ptype == ParameterType.INT:
                    val = int(round(val))
                elif p.ptype in (ParameterType.CATEGORICAL, ParameterType.BOOL):
                    choices = p.allowed_choices or p.all_choices
                    choice_idx = int(round(val))
                    choice_idx = max(0, min(choice_idx, len(choices) - 1))
                    val = choices[choice_idx]  # type: ignore[assignment]
                param_dict[p.name] = val
            results.append(param_dict)

        return results

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _ei(
        mu: np.ndarray,
        sigma: np.ndarray,
        y_best: float,
        direction: str,
    ) -> np.ndarray:
        """
        Expected Improvement acquisition function.

        For *minimize*:  EI = E[max(y_best − Y, 0)]
        For *maximize*:  EI = E[max(Y − y_best, 0)]

        Uses the closed-form Gaussian EI formula.
        """
        from scipy.stats import norm

        if direction == "minimize":
            z = (y_best - mu) / sigma
        else:
            z = (mu - y_best) / sigma

        ei = sigma * (z * norm.cdf(z) + norm.pdf(z))
        ei = np.where(sigma > 1e-10, ei, 0.0)
        return ei


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _decode_trial_params(trial: optuna.trial.FrozenTrial, config: StudyConfig) -> dict:
    """
    Decode internal Optuna trial.params → user-facing parameter names.

    Handles both single-subrange params (key = param name) and
    multi-subrange params ({name}__range_idx / {name}__in_range_{idx}).
    Also checks ``trial.user_attrs["ctx_suggested_params"]`` — when the
    contextual sampler was used, the actual suggested values are stored there.
    """
    # If this trial was generated by the contextual sampler, prefer those params
    ctx_json = trial.user_attrs.get("ctx_suggested_params")
    if ctx_json:
        try:
            return json.loads(ctx_json)
        except Exception:
            pass

    result: dict = {}
    raw = trial.params
    for param in config.parameters:
        if not param.enabled:
            continue
        name = param.name
        subranges = getattr(param, "allowed_subranges", [])
        if len(subranges) <= 1:
            if name in raw:
                result[name] = raw[name]
        else:
            range_idx = raw.get(f"{name}__range_idx")
            if range_idx is not None:
                val = raw.get(f"{name}__in_range_{range_idx}")
                if val is not None:
                    result[name] = val
    return result
