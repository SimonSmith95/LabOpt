"""
generate_synthetic_data.py
==========================
Generates synthetic benchmark CSV files used by validate_synthetic.py.

Run once:
    python generate_synthetic_data.py

Output
------
  test_data/branin_grid.csv        — 100 rows, 2 params (x1, x2)
  test_data/hartmann6_samples.csv  — 80 rows,  6 params (x1–x6)
"""
import os
import numpy as np
import pandas as pd

os.makedirs("test_data", exist_ok=True)
rng = np.random.default_rng(42)


# ── Branin ─────────────────────────────────────────────────────────────────
# f(x1, x2) — three global minima all ≈ 0.397
# x1 ∈ [−5, 10],  x2 ∈ [0, 15]

def branin(x1, x2):
    a, b, c = 1.0, 5.1 / (4 * np.pi ** 2), 5.0 / np.pi
    r, s, t = 6.0, 10.0, 1.0 / (8 * np.pi)
    return (
        a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2
        + s * (1 - t) * np.cos(x1)
        + s
    )


N_BRANIN = 300
x1 = rng.uniform(-5, 10, N_BRANIN)
x2 = rng.uniform(0, 15, N_BRANIN)
noise = rng.normal(0, 0.2, N_BRANIN)   # lower noise → higher achievable R²
y = np.array([branin(a, b) for a, b in zip(x1, x2)]) + noise

df_b = pd.DataFrame({"x1": x1.round(5), "x2": x2.round(5),
                      "branin_value": y.round(5)})
df_b.to_csv("test_data/branin_grid.csv", index=False)
print(f"branin_grid.csv:        {len(df_b)} rows | "
      f"branin_value: min={y.min():.3f}  max={y.max():.3f}")
print(f"  True global minimum ~= 0.397  (at ~3 equivalent x1,x2 locations)")


# ── Hartmann-6 ─────────────────────────────────────────────────────────────
# 6 independent parameters in [0, 1]
# Global minimum ≈ −3.322 at (0.2017, 0.1500, 0.4769, 0.2753, 0.3117, 0.6573)

ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
A = np.array([
    [10,   3,   17,  3.5, 1.7, 8  ],
    [0.05, 10,  17,  0.1, 8,   14 ],
    [3,    3.5, 1.7, 10,  17,  8  ],
    [17,   8,   0.05,10,  0.1, 14 ],
], dtype=float)
P = 1e-4 * np.array([
    [1312, 1696, 5569,  124, 8283, 5886],
    [2329, 4135, 8307, 3736, 1004, 9991],
    [2348, 1451, 3522, 2883, 3047, 6650],
    [4047, 8828, 8732, 5743, 1091,  381],
], dtype=float)


def hartmann6(x):
    x = np.asarray(x, dtype=float)
    total = 0.0
    for i in range(4):
        total -= ALPHA[i] * np.exp(-np.sum(A[i] * (x - P[i]) ** 2))
    return total


N_HARTMANN = 250   # need ~40x dimensionality for a reasonable RF fit in 6D
X_h = rng.uniform(0, 1, (N_HARTMANN, 6))
noise_h = rng.normal(0, 0.02, N_HARTMANN)  # very low noise so signal dominates
y_h = np.array([hartmann6(x) for x in X_h]) + noise_h

cols = {f"x{i + 1}": X_h[:, i].round(6) for i in range(6)}
cols["hartmann_value"] = y_h.round(6)
df_h = pd.DataFrame(cols)
df_h.to_csv("test_data/hartmann6_samples.csv", index=False)
print(f"\nhartmann6_samples.csv:  {len(df_h)} rows | "
      f"hartmann_value: min={y_h.min():.4f}  max={y_h.max():.4f}")
print(f"  True global minimum ≈ −3.3224  "
      f"(at x=[0.2017, 0.1500, 0.4769, 0.2753, 0.3117, 0.6573])")

# Verify: noiseless value at known optimum
opt = [0.20169, 0.15001, 0.47687, 0.27533, 0.31165, 0.65730]
print(f"  Noiseless value at optimum: {hartmann6(opt):.5f}  (expected −3.32237)")

print("\nDone — files written to test_data/")
