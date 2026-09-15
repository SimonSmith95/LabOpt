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
import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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


# ── Water boiling point — context variable demo ────────────────────────────
#
# SCENARIO
# --------
# You are optimising the heating temperature of a water-based process.
# The *controllable* parameter is:
#   temperature_C  — the heater set-point [75 – 135 °C]
# The *context* variable (uncontrollable) is:
#   pressure_kPa   — local atmospheric pressure [50 – 200 kPa]
#                    This varies with altitude and weather; you cannot change it.
#
# The OBJECTIVE is energy efficiency:  maximise  efficiency_score  [0 – 1]
#   • Below the boiling point → no boiling, very low score
#   • Just at or slightly above boiling → near-perfect score
#   • Far above boiling → wasted energy, score drops
#
# GROUND TRUTH  (Dühring / Antoine approximation):
#   T_boil(P) [°C] ≈ 100 + 28.94 × log10(P / 101.325)
#
# Key reference points for easy manual validation (from this dataset's formula):
#   P =  50 kPa  (sub-atmospheric)        → T_boil ≈  91 °C
#   P =  70 kPa  (sub-atmospheric)        → T_boil ≈  95 °C
#   P =  90 kPa  (just below 1 atm)       → T_boil ≈  99 °C
#   P = 101.325  (sea level, 1 atm)       → T_boil = 100 °C   ← common knowledge
#   P = 120 kPa  (slightly above atm)     → T_boil ≈ 102 °C
#   P = 150 kPa  (elevated pressure)      → T_boil ≈ 105 °C
#   P = 200 kPa  (high pressure)          → T_boil ≈ 109 °C
#
# NOTE: The formula T_boil = 100 + 28.94*log10(P/101.325) is a simplified
# approximation — not exact physical chemistry.  It is internally consistent
# within this dataset, so the optimizer converges correctly.
# The only value you need from memory to validate is: 101.325 kPa → 100 °C.
#
# DEAD ZONE: pressure_kPa ∈ (95, 108) is left empty so the model must
# interpolate to validate at ~1 atm without having seen that data.
#
# The efficiency curve peaks sharply at T = T_boil(P) and is asymmetric:
#   below boiling → score × 0.2 (exponential decay)
#   above boiling → score × 1.0 (Gaussian, σ = 4 °C)
# This means:
#   - The optimal temperature IS the local boiling point
#   - Being 5 °C too cold is 5× worse than being 5 °C too hot
# ─────────────────────────────────────────────────────────────────────────────

def boiling_point_c(pressure_kpa: np.ndarray) -> np.ndarray:
    """Antoine-approximated boiling point of water [°C] given pressure [kPa]."""
    return 100.0 + 28.94 * np.log10(pressure_kpa / 101.325)


def efficiency_score(temp_c: np.ndarray, pressure_kpa: np.ndarray) -> np.ndarray:
    """
    Energy efficiency score [0, 1].
    Peaks at T = T_boil(P); asymmetric around that point.
    """
    T_boil = boiling_point_c(pressure_kpa)
    delta = temp_c - T_boil          # positive = above boiling, negative = below

    sigma_above = 4.0   # °C — Gaussian half-width above boiling
    sigma_below = 3.0   # °C — sharper penalty below boiling

    score = np.where(
        delta >= 0,
        np.exp(-0.5 * (delta / sigma_above) ** 2),
        0.2 * np.exp(-0.5 * (delta / sigma_below) ** 2),
    )
    return score


rng_w = np.random.default_rng(7)   # different seed from above

N_WATER = 300

# Sample pressure in two bands — leave 95–108 kPa empty (≈ 1 atm zone)
p_low  = rng_w.uniform(50,  95, N_WATER // 2)   # sub-atmospheric
p_high = rng_w.uniform(108, 200, N_WATER - N_WATER // 2)  # super-atmospheric
pressure = np.concatenate([p_low, p_high])
rng_w.shuffle(pressure)

# Temperature: sample broadly across the full range
temperature = rng_w.uniform(75, 135, N_WATER)

# Compute efficiency scores, add a little measurement noise
true_scores = efficiency_score(temperature, pressure)
noise_w = rng_w.normal(0, 0.015, N_WATER)      # ±1.5 pp noise
scores = np.clip(true_scores + noise_w, 0.0, 1.0)

df_w = pd.DataFrame({
    "temperature_C": temperature.round(2),
    "pressure_kPa":  pressure.round(2),
    "efficiency_score": scores.round(4),
})
df_w.to_csv("test_data/water_boiling_context.csv", index=False)

print(f"\nwater_boiling_context.csv:  {len(df_w)} rows")
print(f"  temperature_C : {temperature.min():.1f} – {temperature.max():.1f} °C")
print(f"  pressure_kPa  : 50–95 kPa  ∪  108–200 kPa  (gap at ≈1 atm is intentional)")
print(f"  efficiency_score: min={scores.min():.3f}  max={scores.max():.3f}")
print()
print("  How to use in LabOpt:")
print("    1. Load water_boiling_context.csv")
print("    2. Set efficiency_score as objective  → maximize")
print("    3. Mark pressure_kPa as CONTEXT variable")
print("    4. Click 'Apply Objectives'")
print("    5. In 'Current Conditions' enter a pressure (e.g. 101.325 for sea level)")
print("    6. Ask Next Batch — the suggestion should be ≈ 100 °C at 101.325 kPa")
print()
print("  Quick validation reference (ground truth boiling points):")
for p_ref in [50, 70, 90, 101.325, 120, 150, 200]:
    print(f"    P = {p_ref:7.3f} kPa  →  T_boil ≈ {boiling_point_c(np.array([p_ref]))[0]:.1f} °C")
