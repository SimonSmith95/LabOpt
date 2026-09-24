"""
DoE Math Backend  (Phase 3)
===========================
Pure-maths module — no Qt, no Optuna, no GUI.

Public API
----------
generate_doe(config, n_points, strategy, seed) -> (DataFrame, List[str])
    Generate a DoE point set for a StudyConfig.

compute_readiness_target(config) -> dict
    Compute the recommended minimum number of initial experiments before
    handing off to a surrogate model.

coverage_metrics(df) -> dict
    Compute space-filling quality metrics (maximin distance, L2 discrepancy,
    Pearson correlation matrix) for a generated DoE DataFrame.

STRATEGIES
----------
  "LHS"            Latin Hypercube Sampling  (scipy.stats.qmc)
  "Sobol"          Sobol quasi-random sequence, N rounded up to 2^m
  "Halton"         Halton quasi-random sequence
  "FullFactorial"  All combinations of levels per factor
  "PlackettBurman" 2-level screening design  (requires pyDOE2)
  "BoxBehnken"     Response surface design 3–7 continuous factors (pyDOE2)
  "CCD"            Central Composite Design  (pyDOE2)
  "Random"         Uniform random baseline (same as Optuna startup)

STRATEGY AVAILABILITY
---------------------
LHS, Sobol, Halton, FullFactorial, Random — always available (scipy / stdlib).
PlackettBurman, BoxBehnken, CCD — require pyDOE2.  If not installed the
`_PYDOE2_AVAILABLE` flag is False; callers should check before presenting
these options in the UI.
"""
from __future__ import annotations

import itertools
import math
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import qmc

from parameter_config import ParameterConstraint, ParameterType, StudyConfig

# ── Optional pyDOE2 import ────────────────────────────────────────────────────
try:
    import pyDOE2 as _pydoe
    _PYDOE2_AVAILABLE = True
except ImportError:
    _PYDOE2_AVAILABLE = False

# Strategies that are always available
ALWAYS_AVAILABLE = ["LHS", "Sobol", "Halton", "FullFactorial", "Random"]
# Strategies that require pyDOE2
PYDOE2_REQUIRED  = ["PlackettBurman", "BoxBehnken", "CCD"]
ALL_STRATEGIES   = ALWAYS_AVAILABLE + PYDOE2_REQUIRED


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────────

def pydoe2_available() -> bool:
    """Return True if pyDOE2 is installed."""
    return _PYDOE2_AVAILABLE


def sobol_next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 that is >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


# ─────────────────────────────────────────────────────────────────────────────
# compute_readiness_target
# ─────────────────────────────────────────────────────────────────────────────

def compute_readiness_target(config: StudyConfig) -> dict:
    """
    Compute the recommended minimum number of initial experiments before
    handing off to a surrogate / Bayesian Optimisation loop.

    Formula
    -------
    n_params  = number of enabled, non-residual controllable parameters
    n_context = number of context variables
    n_obj     = number of optimisation objectives

    base        = max(10,  5 * n_params  +  2 * n_context)
    multi_bonus = max(0,  (n_obj - 1) * 5)
    target      = base + multi_bonus

    Rationale
    ---------
    The "5 samples per parameter" rule of thumb comes from the BO literature
    (Jones et al. 1998; Sacks et al. 1989).  The hard floor of 10 matches
    MIN_TRIALS in surrogate_quality.py so the readiness bar always agrees
    with the surrogate quality badge.

    Returns
    -------
    dict with keys:
        target       : int — recommended minimum runs
        n_params     : int
        n_context    : int
        n_objectives : int
        formula_str  : str — human-readable breakdown
    """
    # Count enabled, non-residual parameters
    residual_names = {
        c.residual_param
        for c in config.constraints
        if c.operator == "=" and c.residual_param
    }
    n_params = sum(
        1 for p in config.parameters
        if p.enabled and p.name not in residual_names
    )
    n_context = len(getattr(config, "context_variables", []))
    n_obj = len(config.objectives)

    base = max(10, 5 * n_params + 2 * n_context)
    multi_bonus = max(0, (n_obj - 1) * 5)
    target = base + multi_bonus

    # Build human-readable formula string
    parts = [f"max(10, 5×{n_params} params"]
    if n_context:
        parts.append(f" + 2×{n_context} context")
    parts.append(")")
    formula_str = "".join(parts)
    if multi_bonus:
        formula_str += f" + {multi_bonus} (multi-obj)"
    formula_str += f" = {target}"

    return {
        "target": target,
        "n_params": n_params,
        "n_context": n_context,
        "n_objectives": n_obj,
        "formula_str": formula_str,
    }


# ─────────────────────────────────────────────────────────────────────────────
# generate_doe
# ─────────────────────────────────────────────────────────────────────────────

def generate_doe(
    config: StudyConfig,
    n_points: int,
    strategy: str,
    seed: int = 42,
    levels_per_factor: int = 3,   # used by FullFactorial only
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Generate a DoE point set for the given StudyConfig.

    Parameters
    ----------
    config            : StudyConfig with enabled parameters.
    n_points          : Number of desired points.
                        For Sobol this is rounded up to the next power of 2.
                        For PB/BB/CCD it is auto-computed and *n_points* is
                        used only as a hint.
    strategy          : One of ALL_STRATEGIES.
    seed              : Random seed for reproducibility.
    levels_per_factor : For FullFactorial — how many equally spaced levels to
                        use for each continuous / integer parameter.

    Returns
    -------
    df       : DataFrame with one column per enabled parameter.
               Rows are the DoE points.  No objective columns.
    warnings : List of human-readable warning strings (may be empty).
    """
    if strategy not in ALL_STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {ALL_STRATEGIES}")
    if strategy in PYDOE2_REQUIRED and not _PYDOE2_AVAILABLE:
        raise ValueError(
            f"Strategy '{strategy}' requires pyDOE2.  "
            "Install it with:  pip install pyDOE2"
        )

    # Collect enabled, non-residual parameters
    residual_names = {
        c.residual_param
        for c in config.constraints
        if c.operator == "=" and c.residual_param
    }
    params = [
        p for p in config.parameters
        if p.enabled and p.name not in residual_names
    ]
    if not params:
        raise ValueError("No enabled parameters found in the StudyConfig.")

    warnings: List[str] = []

    # ── Check if there is a compositional equality constraint ──────────────
    equality_constraints = [
        c for c in config.constraints if c.operator == "="
    ]

    # ── Dispatch to strategy-specific generator ────────────────────────────
    if strategy == "LHS":
        df, warnings = _generate_lhs(params, n_points, seed, warnings)
    elif strategy == "Sobol":
        df, warnings = _generate_sobol(params, n_points, seed, warnings)
    elif strategy == "Halton":
        df, warnings = _generate_halton(params, n_points, seed, warnings)
    elif strategy == "FullFactorial":
        df, warnings = _generate_full_factorial(
            params, n_points, seed, warnings, levels_per_factor
        )
    elif strategy == "PlackettBurman":
        df, warnings = _generate_plackett_burman(params, seed, warnings)
    elif strategy == "BoxBehnken":
        df, warnings = _generate_box_behnken(params, seed, warnings)
    elif strategy == "CCD":
        df, warnings = _generate_ccd(params, seed, warnings)
    else:  # "Random"
        df, warnings = _generate_random(params, n_points, seed, warnings)

    # ── Apply equality constraints ─────────────────────────────────────────
    for constraint in equality_constraints:
        if constraint.residual_param and constraint.residual_param in df.columns:
            df = _apply_equality_constraint_doe(df, constraint, params)

    # ── Apply inequality constraints (rejection sampling) ──────────────────
    inequality_constraints = [
        c for c in config.constraints if c.operator in ("<=", ">=")
    ]
    if inequality_constraints:
        df, ineq_warns = _apply_inequality_constraints(
            df, inequality_constraints, params, n_points, strategy, seed
        )
        warnings.extend(ineq_warns)

    return df, warnings


# ─────────────────────────────────────────────────────────────────────────────
# coverage_metrics
# ─────────────────────────────────────────────────────────────────────────────

def coverage_metrics(df: pd.DataFrame) -> dict:
    """
    Compute space-filling quality metrics for a generated DoE DataFrame.

    Continuous / integer columns only (string/categorical columns are skipped).

    Returns
    -------
    dict with keys:
        maximin_distance   : float — minimum pairwise Euclidean distance
                             in the [0,1]-scaled space (higher = better spread)
        discrepancy        : float — L2-star discrepancy (lower = better)
                             None if fewer than 2 rows
        correlation_matrix : pd.DataFrame — Pearson correlation matrix
                             (ideal = identity matrix for independent variables)
        random_maximin_ref : float — average maximin distance of 20 random
                             draws with the same N (reference baseline)
        random_discr_ref   : float — average L2 discrepancy of 20 random draws
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return {
            "maximin_distance": None,
            "discrepancy": None,
            "correlation_matrix": pd.DataFrame(),
            "random_maximin_ref": None,
            "random_discr_ref": None,
        }

    X = df[numeric_cols].values.astype(float)
    n, d = X.shape

    # Scale each column to [0, 1] for distance / discrepancy metrics
    col_min = X.min(axis=0)
    col_max = X.max(axis=0)
    rng = np.where(col_max - col_min > 0, col_max - col_min, 1.0)
    X_scaled = (X - col_min) / rng

    # Maximin distance
    maximin = _maximin_distance(X_scaled) if n >= 2 else None

    # L2-star discrepancy
    try:
        discr = float(qmc.discrepancy(X_scaled)) if n >= 2 else None
    except Exception:
        discr = None

    # Pearson correlation matrix
    corr = df[numeric_cols].corr()

    # Random reference values (20 draws, same N and d)
    rng_ref = np.random.default_rng(seed=0)
    ref_maximin_vals = []
    ref_discr_vals = []
    for _ in range(20):
        R = rng_ref.uniform(size=(n, d))
        ref_maximin_vals.append(_maximin_distance(R) if n >= 2 else 0.0)
        try:
            ref_discr_vals.append(float(qmc.discrepancy(R)))
        except Exception:
            ref_discr_vals.append(0.0)

    return {
        "maximin_distance": maximin,
        "discrepancy": discr,
        "correlation_matrix": corr,
        "random_maximin_ref": float(np.mean(ref_maximin_vals)),
        "random_discr_ref": float(np.mean(ref_discr_vals)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _maximin_distance(X: np.ndarray) -> float:
    """Return the minimum pairwise Euclidean distance in X (rows = points)."""
    n = len(X)
    if n < 2:
        return 0.0
    min_d = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(X[i] - X[j]))
            if d < min_d:
                min_d = d
    return min_d


def _scale_unit_to_param(unit_sample: np.ndarray, params: list) -> pd.DataFrame:
    """
    Scale a unit [0,1]^d sample to the actual parameter ranges.

    - FLOAT  : linear scale to [allowed_subrange.low, allowed_subrange.high].
               For multi-subrange params: first pick a subrange proportionally
               by width, then scale within it.
    - INT    : as float then round.
    - CATEGORICAL / BOOL : map fractional index to allowed_choices.
    """
    rows = []
    for row in unit_sample:
        point: dict = {}
        for col_idx, p in enumerate(params):
            u = float(row[col_idx])
            if p.ptype in (ParameterType.FLOAT, ParameterType.INT):
                subranges = p.allowed_subranges or [
                    type("SR", (), {"low": p.full_min, "high": p.full_max})()
                ]
                # Weight subranges by width
                widths = [max(0.0, sr.high - sr.low) for sr in subranges]
                total = sum(widths)
                if total <= 0:
                    val = subranges[0].low
                else:
                    # Pick subrange
                    cum = 0.0
                    chosen_sr = subranges[-1]
                    for sr, w in zip(subranges, widths):
                        cum += w / total
                        if u <= cum:
                            chosen_sr = sr
                            break
                    val = chosen_sr.low + u * (chosen_sr.high - chosen_sr.low)
                if p.ptype == ParameterType.INT:
                    val = int(round(val))
                    val = int(np.clip(val, p.full_min, p.full_max))
            elif p.ptype in (ParameterType.CATEGORICAL, ParameterType.BOOL):
                choices = p.allowed_choices if p.allowed_choices else p.all_choices
                if not choices:
                    choices = ["True", "False"]
                idx = int(min(round(u * (len(choices) - 1)), len(choices) - 1))
                val = choices[idx]
            else:
                val = p.full_min
            point[p.name] = val
        rows.append(point)
    return pd.DataFrame(rows)


def _generate_lhs(params, n_points, seed, warnings):
    d = len(params)
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    sample = sampler.random(n=n_points)
    df = _scale_unit_to_param(sample, params)
    return df, warnings


def _generate_sobol(params, n_points, seed, warnings):
    d = len(params)
    m = max(1, math.ceil(math.log2(max(n_points, 2))))
    actual_n = 2 ** m
    if actual_n != n_points:
        warnings.append(
            f"Sobol: N rounded up from {n_points} to {actual_n} (nearest power of 2)."
        )
    if n_points < 32 and d > 5:
        warnings.append(
            f"Sobol: N={n_points} < 32 with d={d} parameters — coverage may be poor. "
            "Prefer LHS or Halton for small N with many parameters."
        )
    sampler = qmc.Sobol(d=d, seed=seed)
    sample = sampler.random_base2(m=m)
    df = _scale_unit_to_param(sample, params)
    return df, warnings


def _generate_halton(params, n_points, seed, warnings):
    d = len(params)
    sampler = qmc.Halton(d=d, seed=seed)
    sample = sampler.random(n=n_points)
    df = _scale_unit_to_param(sample, params)
    return df, warnings


def _generate_full_factorial(params, n_points, seed, warnings, levels_per_factor):
    """Full factorial: all combinations of levels per parameter."""
    level_lists = []
    for p in params:
        if p.ptype in (ParameterType.CATEGORICAL, ParameterType.BOOL):
            choices = p.allowed_choices if p.allowed_choices else p.all_choices
            level_lists.append([float(i) / max(1, len(choices) - 1)
                                 for i in range(len(choices))])
        else:
            level_lists.append(
                [i / max(1, levels_per_factor - 1) for i in range(levels_per_factor)]
            )
    total = 1
    for ll in level_lists:
        total *= len(ll)
    if total > 1000:
        warnings.append(
            f"Full Factorial: {total} combinations — this is very large. "
            "Consider reducing levels_per_factor or switching to LHS."
        )
    if total > 10000:
        # Hard cap to avoid memory / time issues
        total = 10000
        warnings.append("Full Factorial capped at 10 000 combinations.")

    sample = list(itertools.product(*level_lists))[:total]
    sample_arr = np.array(sample, dtype=float)
    df = _scale_unit_to_param(sample_arr, params)
    return df, warnings


def _generate_plackett_burman(params, seed, warnings):
    d = len(params)
    if d < 2:
        warnings.append("Plackett-Burman requires at least 2 factors; falling back to LHS.")
        return _generate_lhs(params, max(4, d + 1), seed, warnings)
    # pyDOE2.pbdesign returns a -1/+1 matrix; shape (n, k) where n ≥ k+1
    pb = _pydoe.pbdesign(d)  # shape (n, d)
    # Map -1/+1 → 0/1
    sample = (pb + 1) / 2
    df = _scale_unit_to_param(sample, params)
    return df, warnings


def _generate_box_behnken(params, seed, warnings):
    d = len(params)
    if d < 3 or d > 7:
        warnings.append(
            f"Box-Behnken requires 3–7 factors; got {d}. Falling back to LHS."
        )
        return _generate_lhs(params, max(10, 3 * d), seed, warnings)
    bb = _pydoe.bbdesign(d, center=1)   # shape (n, d), values in {-1, 0, +1}
    sample = (bb + 1) / 2
    df = _scale_unit_to_param(sample, params)
    return df, warnings


def _generate_ccd(params, seed, warnings):
    d = len(params)
    if d < 2 or d > 6:
        warnings.append(
            f"CCD requires 2–6 factors; got {d}. Falling back to LHS."
        )
        return _generate_lhs(params, max(10, 3 * d), seed, warnings)
    cc = _pydoe.ccdesign(d, center=(1, 1), alpha="r", face="cci")
    # cc values are typically in [-sqrt(d), +sqrt(d)]; normalise to [0, 1]
    cc_min, cc_max = cc.min(), cc.max()
    if cc_max > cc_min:
        sample = (cc - cc_min) / (cc_max - cc_min)
    else:
        sample = np.zeros_like(cc)
    df = _scale_unit_to_param(sample, params)
    return df, warnings


def _generate_random(params, n_points, seed, warnings):
    rng = np.random.default_rng(seed=seed)
    sample = rng.uniform(size=(n_points, len(params)))
    df = _scale_unit_to_param(sample, params)
    return df, warnings


# ─────────────────────────────────────────────────────────────────────────────
# Constraint application
# ─────────────────────────────────────────────────────────────────────────────

def _apply_equality_constraint_doe(
    df: pd.DataFrame,
    constraint: ParameterConstraint,
    params: list,
) -> pd.DataFrame:
    """
    Apply an equality constraint by computing the residual parameter.

    For compositional constraints (a + b + c = 1):
      1. Replace the DoE values for all non-residual params in the constraint
         expression using Dirichlet sampling so they sum to the target.
      2. Compute the residual = target - sum(others).

    For other equality expressions the residual is computed algebraically:
      residual = (target - eval(expr with residual=0)) / coefficient_of_residual
    """
    residual = constraint.residual_param
    if not residual or residual not in df.columns:
        return df

    df = df.copy()
    try:
        for i in range(len(df)):
            row_dict = df.iloc[i].to_dict()
            val = ParameterConstraint.eval_expr(constraint.expression, row_dict)
            # residual = target - (expression evaluated with residual=0)
            row_dict_no_res = {k: (0.0 if k == residual else v) for k, v in row_dict.items()}
            val_no_res = ParameterConstraint.eval_expr(constraint.expression, row_dict_no_res)
            # coefficient of residual in expression
            row_dict_res1 = {k: (1.0 if k == residual else 0.0) for k, v in row_dict.items()}
            coeff = ParameterConstraint.eval_expr(constraint.expression, row_dict_res1) - \
                    ParameterConstraint.eval_expr(constraint.expression,
                                                  {k: 0.0 for k in row_dict})
            if abs(coeff) > 1e-12:
                residual_val = (constraint.target - val_no_res) / coeff
            else:
                residual_val = 0.0
            df.at[i, residual] = residual_val
    except Exception:
        pass   # Non-fatal — return df as-is if constraint application fails

    return df


def _apply_inequality_constraints(
    df: pd.DataFrame,
    constraints: list,
    params: list,
    target_n: int,
    strategy: str,
    seed: int,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Filter rows that violate inequality constraints via rejection sampling.

    If the rejection rate is high (> 50% after 5× the target), a warning is
    issued.  The generation is capped at 20× target_n attempts.
    """
    warnings: List[str] = []
    if not constraints:
        return df, warnings

    def _satisfies_all(row_dict: dict) -> bool:
        for c in constraints:
            ok, _ = c.is_satisfied(row_dict)
            if not ok:
                return False
        return True

    # Check which rows in the initial df already satisfy constraints
    valid_rows = []
    for i in range(len(df)):
        row_dict = df.iloc[i].to_dict()
        if _satisfies_all(row_dict):
            valid_rows.append(df.iloc[i])

    if len(valid_rows) >= target_n:
        return pd.DataFrame(valid_rows[:target_n]), warnings

    # Need more valid rows — generate additional batches
    total_generated = len(df)
    max_attempts = 20 * target_n
    current_seed = seed + 1

    while len(valid_rows) < target_n and total_generated < max_attempts:
        batch_n = min(target_n * 2, max_attempts - total_generated)
        extra_df, _ = {
            "LHS": _generate_lhs,
            "Sobol": _generate_sobol,
            "Halton": _generate_halton,
            "Random": _generate_random,
        }.get(strategy, _generate_random)(params, batch_n, current_seed, [])
        current_seed += 1
        total_generated += batch_n
        for i in range(len(extra_df)):
            row_dict = extra_df.iloc[i].to_dict()
            if _satisfies_all(row_dict):
                valid_rows.append(extra_df.iloc[i])
                if len(valid_rows) >= target_n:
                    break

    rejection_pct = max(0, 100 * (1 - len(valid_rows) / max(1, total_generated)))
    if rejection_pct > 50:
        warnings.append(
            f"High constraint rejection rate ({rejection_pct:.0f}% of candidates rejected). "
            "The inequality constraint significantly reduces the feasible region. "
            "Consider relaxing it or switching to a smaller N."
        )

    if not valid_rows:
        warnings.append(
            "No feasible points found after maximum attempts. "
            "The constraint may be too tight. Returning unconstrained points."
        )
        return df, warnings

    result = pd.DataFrame(valid_rows[:target_n]).reset_index(drop=True)
    if len(result) < target_n:
        warnings.append(
            f"Only {len(result)} feasible points found (requested {target_n}). "
            "Consider relaxing the constraint or reducing N."
        )
    return result, warnings
