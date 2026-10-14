"""
Benchmark Math
==============
Pure-numpy implementations of standard BO benchmark functions.

No Qt, no Optuna, no file I/O — safe to import from any thread.

Public API
----------
evaluate(x, func_name) -> float | list[float]
    Evaluate *func_name* at point *x* (1-D array in [0,1]^d).
    Returns a scalar for single-objective functions, a list for multi-objective.

AUTO_SELECT_BENCHMARK(n_inputs, n_objectives) -> str
    Choose an appropriate benchmark name given the problem dimensionality.

BENCHMARK_BOUNDS, BENCHMARK_N_OBJ
    Dicts mapping benchmark name → (lo, hi) per-dimension bounds and n_objectives.

Available benchmarks
--------------------
Single-objective (returns scalar):
  "Branin"     — 2-D classic (padded to any d ≥ 2 with ignore vars)
  "Hartmann6"  — 6-D (padded to any d ≥ 6 or projected to 6-D if d < 6)
  "Ackley"     — any d
  "Rastrigin"  — any d

Multi-objective (returns list):
  "ZDT1"       — 2 objectives, any d ≥ 2
  "ZDT2"       — 2 objectives, any d ≥ 2
  "DTLZ2"      — k objectives, any d ≥ k
"""
from __future__ import annotations

import math
from typing import List, Union

import numpy as np

# ── Benchmark catalogue ───────────────────────────────────────────────────────

# n_objectives for each benchmark
BENCHMARK_N_OBJ: dict[str, int] = {
    "Branin":    1,
    "Hartmann6": 1,
    "Ackley":    1,
    "Rastrigin": 1,
    "ZDT1":      2,
    "ZDT2":      2,
    "DTLZ2":     3,   # default 3 objectives; evaluate() accepts n_obj override
}

# All inputs are assumed to be in [0, 1]^d (unit hypercube)
BENCHMARK_BOUNDS: dict[str, tuple] = {
    b: (0.0, 1.0) for b in BENCHMARK_N_OBJ
}

SINGLE_OBJ_BENCHMARKS = [k for k, v in BENCHMARK_N_OBJ.items() if v == 1]
MULTI_OBJ_BENCHMARKS  = [k for k, v in BENCHMARK_N_OBJ.items() if v > 1]
ALL_BENCHMARKS        = SINGLE_OBJ_BENCHMARKS + MULTI_OBJ_BENCHMARKS


# ── Auto-selection ────────────────────────────────────────────────────────────

def auto_select_benchmark(n_inputs: int, n_objectives: int) -> str:
    """
    Return a sensible benchmark name for (n_inputs, n_objectives).

    Rules
    -----
    n_objectives == 1:
        n_inputs <= 2  → "Branin"
        n_inputs <= 6  → "Hartmann6"  (padded if necessary)
        else           → "Ackley"
    n_objectives == 2:
        → "ZDT1"
    n_objectives >= 3:
        → "DTLZ2"
    """
    if n_objectives == 1:
        if n_inputs <= 2:
            return "Branin"
        elif n_inputs <= 6:
            return "Hartmann6"
        else:
            return "Ackley"
    elif n_objectives == 2:
        return "ZDT1"
    else:
        return "DTLZ2"


# ── Main entry point ──────────────────────────────────────────────────────────

def evaluate(
    x: np.ndarray,
    func_name: str,
    n_obj: int = 0,
) -> Union[float, List[float]]:
    """
    Evaluate benchmark *func_name* at unit-hypercube point *x*.

    Parameters
    ----------
    x        : 1-D array of length d in [0, 1]^d.
    func_name: One of ALL_BENCHMARKS.
    n_obj    : Override the number of objectives for DTLZ2 (default: 3).

    Returns
    -------
    float for single-objective benchmarks.
    list[float] for multi-objective benchmarks.
    """
    x = np.asarray(x, dtype=float).ravel()

    if func_name == "Branin":
        return _branin(x)
    elif func_name == "Hartmann6":
        return _hartmann6(x)
    elif func_name == "Ackley":
        return _ackley(x)
    elif func_name == "Rastrigin":
        return _rastrigin(x)
    elif func_name == "ZDT1":
        return _zdt1(x)
    elif func_name == "ZDT2":
        return _zdt2(x)
    elif func_name == "DTLZ2":
        k = max(3, n_obj) if n_obj > 0 else 3
        return _dtlz2(x, n_obj=k)
    else:
        raise ValueError(f"Unknown benchmark '{func_name}'. Choose from {ALL_BENCHMARKS}.")


def lower_is_better(func_name: str) -> bool:
    """Return True for all benchmarks (all are minimisation problems)."""
    return True


# ── Single-objective implementations ─────────────────────────────────────────

def _branin(x: np.ndarray) -> float:
    """
    Branin-Hoo function.  Global minimum ≈ 0.397887 at three points.
    Input: x in [0,1]^d.  Uses first 2 dimensions; extra dims are ignored.
    Scaled from x1∈[−5,10], x2∈[0,15].
    """
    d = len(x)
    x1 = -5.0 + 15.0 * x[0]
    x2 = 15.0 * x[1] if d >= 2 else 7.5
    a, b, c = 1.0, 5.1 / (4 * math.pi ** 2), 5.0 / math.pi
    r, s, t = 6.0, 10.0, 1.0 / (8 * math.pi)
    return float(a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2
                 + s * (1 - t) * math.cos(x1) + s)


def _hartmann6(x: np.ndarray) -> float:
    """
    Hartmann 6-D function.  Global minimum ≈ −3.32237.
    Input: x in [0,1]^d.  If d < 6, pad with 0.5.  If d > 6, use first 6.
    """
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [10.0,  3.0, 17.0,  3.5,  1.7,  8.0],
        [0.05, 10.0, 17.0,  0.1,  8.0, 14.0],
        [3.0,   3.5,  1.7, 10.0, 17.0,  8.0],
        [17.0,  8.0,  0.05, 10.0, 0.1, 14.0],
    ])
    P = 1e-4 * np.array([
        [1312, 1696, 5569, 124,  8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381 ],
    ])
    # Build the 6-D point
    x6 = np.full(6, 0.5)
    x6[:min(6, len(x))] = x[:min(6, len(x))]

    outer = 0.0
    for i in range(4):
        inner = np.sum(A[i] * (x6 - P[i]) ** 2)
        outer += alpha[i] * math.exp(-inner)
    return float(-outer)


def _ackley(x: np.ndarray) -> float:
    """
    Ackley function.  Global minimum = 0 at the origin.
    Input: x in [0,1]^d (scaled to [−32.768, 32.768]).
    """
    xs = -32.768 + 65.536 * x
    d = len(xs)
    a, b, c = 20.0, 0.2, 2 * math.pi
    sum1 = np.sum(xs ** 2)
    sum2 = np.sum(np.cos(c * xs))
    return float(-a * math.exp(-b * math.sqrt(sum1 / d))
                 - math.exp(sum2 / d)
                 + a + math.e)


def _rastrigin(x: np.ndarray) -> float:
    """
    Rastrigin function.  Global minimum = 0 at the origin.
    Input: x in [0,1]^d (scaled to [−5.12, 5.12]).
    """
    xs = -5.12 + 10.24 * x
    d = len(xs)
    return float(10 * d + np.sum(xs ** 2 - 10 * np.cos(2 * math.pi * xs)))


# ── Multi-objective implementations ──────────────────────────────────────────

def _zdt1(x: np.ndarray) -> List[float]:
    """
    ZDT1 (Zitzler-Deb-Thiele).  Pareto front: f2 = 1 − sqrt(f1).
    Input: x in [0,1]^d (d ≥ 2).  f1 = x[0]; g uses x[1:].
    """
    d = max(len(x), 2)
    x_full = np.ones(d) * 0.5
    x_full[:len(x)] = x
    f1 = float(x_full[0])
    if d > 1:
        g = 1.0 + 9.0 * np.mean(x_full[1:])
    else:
        g = 1.0
    h = 1.0 - math.sqrt(f1 / g)
    f2 = g * h
    return [f1, float(f2)]


def _zdt2(x: np.ndarray) -> List[float]:
    """
    ZDT2.  Pareto front: f2 = 1 − f1².
    """
    d = max(len(x), 2)
    x_full = np.ones(d) * 0.5
    x_full[:len(x)] = x
    f1 = float(x_full[0])
    if d > 1:
        g = 1.0 + 9.0 * np.mean(x_full[1:])
    else:
        g = 1.0
    h = 1.0 - (f1 / g) ** 2
    f2 = g * h
    return [f1, float(f2)]


def _dtlz2(x: np.ndarray, n_obj: int = 3) -> List[float]:
    """
    DTLZ2.  Pareto front lies on the unit sphere.
    Input: x in [0,1]^d where d ≥ n_obj.
    """
    n_obj = max(2, n_obj)
    d = max(len(x), n_obj)
    x_full = np.ones(d) * 0.5
    x_full[:len(x)] = x

    # g function uses the last (d - n_obj + 1) variables
    k = d - n_obj + 1
    xm = x_full[n_obj - 1:]
    g = float(np.sum((xm - 0.5) ** 2))

    # Objectives
    fs = []
    for i in range(n_obj):
        f = (1.0 + g)
        for j in range(n_obj - 1 - i):
            f *= math.cos(0.5 * math.pi * x_full[j])
        if i > 0:
            f *= math.sin(0.5 * math.pi * x_full[n_obj - 1 - i])
        fs.append(float(f))
    return fs


# ── Hypervolume helper ────────────────────────────────────────────────────────

def compute_hypervolume(
    pareto_values: list,
    reference_point: list | None = None,
) -> float:
    """
    Compute the hypervolume indicator for a set of multi-objective values.

    Uses a simple sweep-line algorithm (exact, works for 2 objectives)
    or a Monte Carlo approximation for ≥ 3 objectives.

    Parameters
    ----------
    pareto_values    : list of lists — each inner list is one objective vector.
    reference_point  : worst-case reference point.  Defaults to 1.1 × max per obj.

    Returns
    -------
    float — hypervolume (higher = better Pareto front coverage).
    """
    if not pareto_values:
        return 0.0

    pts = np.array(pareto_values, dtype=float)  # shape (n, m)
    n, m = pts.shape

    if reference_point is None:
        ref = pts.max(axis=0) * 1.1 + 1e-6
    else:
        ref = np.array(reference_point, dtype=float)

    if m == 2:
        return _hv_2d(pts, ref)
    else:
        return _hv_mc(pts, ref, n_samples=50_000)


def _hv_2d(pts: np.ndarray, ref: np.ndarray) -> float:
    """
    Exact 2-D hypervolume via sweep line.

    Standard algorithm:
      Sort Pareto front by f1 ascending (f2 is then descending for a true front).
      For each point i the contribution is:
        width  = f1[i+1] - f1[i]   (or ref[0] - f1[i] for the last point)
        height = ref[1] - f2[i]
    """
    # Sort by first objective ascending, then filter dominated points
    order = np.argsort(pts[:, 0])
    pts = pts[order]
    front = [pts[0]]
    for p in pts[1:]:
        if p[1] < front[-1][1]:
            front.append(p)
    front = np.array(front)

    hv = 0.0
    n = len(front)
    for i in range(n):
        # Width: distance to next Pareto point, or to reference on the last segment
        if i < n - 1:
            width = front[i + 1, 0] - front[i, 0]
        else:
            width = ref[0] - front[i, 0]
        height = ref[1] - front[i, 1]
        hv += max(0.0, width) * max(0.0, height)
    return float(hv)


def _hv_mc(pts: np.ndarray, ref: np.ndarray, n_samples: int = 50_000) -> float:
    """Monte Carlo hypervolume approximation for m ≥ 3."""
    rng = np.random.default_rng(seed=0)
    lo = pts.min(axis=0)
    hi = ref
    vol = np.prod(np.maximum(hi - lo, 1e-12))
    samples = rng.uniform(lo, hi, size=(n_samples, pts.shape[1]))
    # A sample is dominated if there exists a point that is ≤ it in all dims
    dominated = np.zeros(n_samples, dtype=bool)
    for p in pts:
        dominated |= np.all(samples >= p, axis=1)
    return float(vol * dominated.mean())
