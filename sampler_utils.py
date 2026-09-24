"""
Phase 4 — Sampler Utilities
Dead-region-aware parameter suggestion helpers for Optuna.

Strategy
--------
For parameters with multiple allowed sub-ranges, we use a two-level suggestion:
  1. Suggest WHICH sub-range to use (categorical, weighted by size).
  2. Suggest the value within that sub-range (int/float).

This keeps the internal representation transparent to TPE's Parzen estimator
and avoids slow rejection sampling.
"""
from __future__ import annotations

from typing import List, Tuple

import optuna

from parameter_config import AllowedSubRange


# ──────────────────────────────────────────────────────────────────────────────
# Public suggestion helpers
# ──────────────────────────────────────────────────────────────────────────────

def subrange_suggest_int(
    trial: optuna.Trial,
    name: str,
    subranges: List[AllowedSubRange],
    step: int = 1,
) -> int:
    """
    Suggest an integer from the union of *subranges*.

    If there is only one sub-range, behaves exactly like trial.suggest_int().
    With multiple sub-ranges a categorical choice picks the sub-range (weighted
    by the number of integers it contains), then suggest_int is called within
    the chosen range.
    """
    if not subranges:
        raise ValueError(f"No allowed sub-ranges for parameter '{name}'.")

    if len(subranges) == 1:
        return trial.suggest_int(name, int(subranges[0].low), int(subranges[0].high), step=step)

    # Weight each range by its integer count
    sizes = [max(1, (int(r.high) - int(r.low)) // step + 1) for r in subranges]
    weighted_indices = _build_weighted_list(list(range(len(subranges))), sizes)

    chosen_idx = trial.suggest_categorical(f"{name}__range_idx", weighted_indices)
    chosen = subranges[chosen_idx]
    return trial.suggest_int(
        f"{name}__in_range_{chosen_idx}",
        int(chosen.low),
        int(chosen.high),
        step=step,
    )


def subrange_suggest_float(
    trial: optuna.Trial,
    name: str,
    subranges: List[AllowedSubRange],
) -> float:
    """
    Suggest a float from the union of *subranges*.

    Same two-level strategy as subrange_suggest_int, weighted by interval width.
    """
    if not subranges:
        raise ValueError(f"No allowed sub-ranges for parameter '{name}'.")

    if len(subranges) == 1:
        return trial.suggest_float(name, subranges[0].low, subranges[0].high)

    widths = [max(1e-12, r.high - r.low) for r in subranges]
    total = sum(widths)
    # Scale to integers with resolution 1000 for the weighted categorical list
    int_weights = [max(1, round(w / total * 1000)) for w in widths]
    weighted_indices = _build_weighted_list(list(range(len(subranges))), int_weights)

    chosen_idx = trial.suggest_categorical(f"{name}__range_idx", weighted_indices)
    chosen = subranges[chosen_idx]
    return trial.suggest_float(
        f"{name}__in_range_{chosen_idx}",
        chosen.low,
        chosen.high,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Dead-region computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_allowed_subranges(
    full_min: float,
    full_max: float,
    dead_regions: List[Tuple[float, float]],
) -> List[AllowedSubRange]:
    """
    Return the complement of *dead_regions* within [full_min, full_max].

    Dead regions are merged and clipped before inversion, so overlapping or
    out-of-bounds inputs are handled gracefully.

    Raises
    ------
    ValueError if the dead regions cover the entire [full_min, full_max] range.
    """
    if not dead_regions:
        return [AllowedSubRange(full_min, full_max)]

    # Clip each dead region to the full range and drop empty ones
    clipped: List[Tuple[float, float]] = []
    for lo, hi in dead_regions:
        lo = max(lo, full_min)
        hi = min(hi, full_max)
        if lo < hi:
            clipped.append((lo, hi))

    if not clipped:
        return [AllowedSubRange(full_min, full_max)]

    # Sort and merge overlapping dead regions
    clipped.sort(key=lambda r: r[0])
    merged: List[Tuple[float, float]] = []
    for lo, hi in clipped:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))

    # Build complement within [full_min, full_max]
    allowed: List[AllowedSubRange] = []
    cursor = full_min
    for dead_lo, dead_hi in merged:
        if cursor < dead_lo:
            allowed.append(AllowedSubRange(cursor, dead_lo))
        cursor = dead_hi
    if cursor < full_max:
        allowed.append(AllowedSubRange(cursor, full_max))

    if not allowed:
        raise ValueError(
            "Dead regions cover the entire parameter range. No allowed values remain."
        )

    return allowed


def validate_subranges(
    subranges: List[AllowedSubRange],
    full_min: float,
    full_max: float,
) -> List[str]:
    """
    Return a list of human-readable error messages for any problems found.
    An empty list means the configuration is valid.
    """
    errors: List[str] = []

    if not subranges:
        errors.append("No allowed sub-ranges defined.")
        return errors

    for i, r in enumerate(subranges):
        label = f"Sub-range {i + 1}"
        if r.low >= r.high:
            errors.append(f"{label}: low ({r.low}) must be strictly less than high ({r.high}).")
        if r.low < full_min:
            errors.append(f"{label}: low ({r.low}) is below the full minimum ({full_min}).")
        if r.high > full_max:
            errors.append(f"{label}: high ({r.high}) exceeds the full maximum ({full_max}).")

    # Check pairwise overlaps on sorted copy
    sorted_r = sorted(subranges, key=lambda r: r.low)
    for i in range(len(sorted_r) - 1):
        if sorted_r[i].high > sorted_r[i + 1].low:
            errors.append(
                f"Sub-ranges {i + 1} and {i + 2} overlap "
                f"({sorted_r[i].high} > {sorted_r[i + 1].low})."
            )

    return errors


# ──────────────────────────────────────────────────────────────────────────────
# Near-duplicate detection (Feature 4)
# ──────────────────────────────────────────────────────────────────────────────

from parameter_config import ParameterType, StudyConfig  # noqa: E402  (after optuna import)

NEAR_DUPLICATE_THRESHOLD = 0.05   # 5 % of normalised parameter range


def find_near_duplicates(
    suggestions: list[dict],
    existing_trials: list[dict],
    config: StudyConfig,
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> list[dict]:
    """
    Compare each suggested parameter set against all completed trials using
    normalised Euclidean distance.

    Parameters
    ----------
    suggestions     : list of param-value dicts for the new batch.
    existing_trials : list of dicts — each must contain a ``"number"`` key
                      (the actual Optuna trial number) plus one key per
                      parameter.  Any key named ``"number"`` is treated as
                      the trial identifier and is excluded from distance
                      computation.
    config          : StudyConfig supplying parameter ranges / types.
    threshold       : distance below which a suggestion is flagged (0.05 = 5 %).

    Returns
    -------
    List of dicts, one per flagged suggestion::

        {
            "suggestion_idx"      : int,   # 0-based index into *suggestions*
            "message"             : str,   # human-readable warning
            "closest_trial_number": int,   # actual Optuna trial number
            "closest_params"      : dict,  # param values of the closest trial
            "distance"            : float, # normalised distance (0–1)
        }

    An empty list means no near-duplicates were found.
    """
    if not existing_trials:
        return []

    numeric_params = [
        p for p in config.parameters
        if p.enabled and p.ptype in (ParameterType.INT, ParameterType.FLOAT)
    ]
    cat_params = [
        p for p in config.parameters
        if p.enabled and p.ptype == ParameterType.CATEGORICAL
    ]

    def _norm(val: float, p) -> float:
        rng = p.full_max - p.full_min
        return (val - p.full_min) / rng if rng > 0 else 0.0

    n_active = len(numeric_params) + len(cat_params)
    if n_active == 0:
        return []

    results: list[dict] = []

    for s_idx, suggestion in enumerate(suggestions):
        min_dist = float("inf")
        closest_num = -1
        closest_params: dict = {}

        for existing in existing_trials:
            trial_number = existing.get("number", -1)
            # Exclude the "number" sentinel from distance computation
            params_only = {k: v for k, v in existing.items() if k != "number"}

            dist_sq = 0.0
            for p in numeric_params:
                sv = suggestion.get(p.name)
                ev = params_only.get(p.name)
                if sv is not None and ev is not None:
                    dist_sq += (_norm(float(sv), p) - _norm(float(ev), p)) ** 2
            for p in cat_params:
                sv = suggestion.get(p.name)
                ev = params_only.get(p.name)
                if sv is not None and ev is not None:
                    dist_sq += 0.0 if sv == ev else 1.0

            dist = (dist_sq / n_active) ** 0.5
            if dist < min_dist:
                min_dist = dist
                closest_num = trial_number
                closest_params = params_only

        if min_dist < threshold:
            # Build a short human-readable composition string
            comp_parts = []
            for p in numeric_params:
                v = closest_params.get(p.name)
                if v is not None:
                    comp_parts.append(f"{p.name}={float(v):.4g}")
            for p in cat_params:
                v = closest_params.get(p.name)
                if v is not None:
                    comp_parts.append(f"{p.name}={v}")
            comp_str = ", ".join(comp_parts) if comp_parts else "—"

            msg = (
                f"Suggestion {s_idx + 1} is very similar to "
                f"Optuna trial #{closest_num} ({comp_str}), "
                f"distance {min_dist * 100:.1f}%."
            )
            results.append({
                "suggestion_idx":       s_idx,
                "message":              msg,
                "closest_trial_number": closest_num,
                "closest_params":       closest_params,
                "distance":             min_dist,
            })

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_weighted_list(indices: List[int], weights: List[int]) -> List[int]:
    """Return a flat list where index i appears weights[i] times."""
    result: List[int] = []
    for idx, w in zip(indices, weights):
        result.extend([idx] * w)
    return result
