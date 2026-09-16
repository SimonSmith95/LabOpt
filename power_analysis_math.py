"""
power_analysis_math.py
======================
Pure statistical-power functions with no Qt or matplotlib dependencies.
Imported by both power_analysis_widget.py and test_power_analysis.py.

Functions
---------
compute_required_n(sigma, delta, alpha, power, two_sample) -> dict
    Required sample size for a two-sided t-test.

achieved_power(n, sigma, delta, alpha, two_sample) -> float
    Achieved statistical power for a given N.
"""
from __future__ import annotations

import math

import numpy as np


def compute_required_n(
    sigma: float,
    delta: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sample: bool = False,
) -> dict:
    """
    Compute the minimum sample size for a two-sided t-test.

    Parameters
    ----------
    sigma      : Standard deviation of the measurement (spread Y).
    delta      : Minimum detectable effect / improvement (X).
    alpha      : Desired significance level (Type I error rate). Default 0.05.
    power      : Desired statistical power (1 − β). Default 0.80.
    two_sample : If True, use the two-sample independent-groups formula and
                 return N *per group*.  If False (default), use the one-sample
                 or paired-sample formula.

    Returns
    -------
    dict
        n          : int   — required sample size (per group when two_sample=True).
        cohens_d   : float — effect size = delta / sigma.
        effect_tag : str   — "Small" | "Medium" | "Large" | "Very large".

    Raises
    ------
    ValueError
        If sigma ≤ 0, delta ≤ 0, or alpha/power are outside (0, 1).
    """
    from scipy.stats import norm

    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma!r}")
    if delta <= 0:
        raise ValueError(f"delta must be > 0, got {delta!r}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    if not (0.0 < power < 1.0):
        raise ValueError(f"power must be in (0, 1), got {power!r}")

    z_a = norm.ppf(1.0 - alpha / 2.0)
    z_b = norm.ppf(power)
    cohens_d = delta / sigma
    n_raw = ((z_a + z_b) / cohens_d) ** 2
    if two_sample:
        n_raw *= 2.0          # need N per group
    n = max(2, math.ceil(n_raw))

    # Effect-size classification (Cohen 1988)
    if cohens_d >= 0.8:
        tag = "Very large"
    elif cohens_d >= 0.5:
        tag = "Large"
    elif cohens_d >= 0.2:
        tag = "Medium"
    else:
        tag = "Small"

    return {"n": n, "cohens_d": cohens_d, "effect_tag": tag}


def achieved_power(
    n: int,
    sigma: float,
    delta: float,
    alpha: float = 0.05,
    two_sample: bool = False,
) -> float:
    """
    Compute the achieved statistical power for a given sample size N.

    The non-centrality parameter is:
        ncp = d × sqrt(n_eff)
    where n_eff = n/2 for two-sample tests, n for one-sample.

    Returns
    -------
    float in [0, 1]
    """
    from scipy.stats import norm

    cohens_d = delta / sigma
    z_a = norm.ppf(1.0 - alpha / 2.0)
    n_eff = n / 2.0 if two_sample else float(n)
    ncp = cohens_d * math.sqrt(n_eff)
    # Two-sided power: P(reject H0 | H1 true)
    pwr = 1.0 - norm.cdf(z_a - ncp) + norm.cdf(-z_a - ncp)
    return float(np.clip(pwr, 0.0, 1.0))
