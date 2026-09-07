"""
Smoke-test for csv_loader fixes.
Run from the repo root:
    python test_data/_smoke_test.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from csv_loader import load_csv, infer_column_type, extract_param_defaults

PASS = "PASS"
FAIL = "FAIL"

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  {status} {label}" + (f"  [{detail}]" if detail else ""))
    return condition

results = []

# ── 1. Comma-delimited ──────────────────────────────────────────────────────
print("\n[1] test_comma.csv  (comma delimiter, NaN in param + objective)")
df = load_csv("test_data/test_comma.csv")
results.append(check("Loaded without error", df is not None))
results.append(check("6 columns", len(df.columns) == 6, str(df.columns.tolist())))
results.append(check("No BOM on first column", not df.columns[0].startswith("\ufeff")))
results.append(check("NaN present in 'pressure'", df["pressure"].isna().any()))
results.append(check("NaN present in 'yield'", df["yield"].isna().any()))

# ── 2. Semicolon-delimited ───────────────────────────────────────────────────
print("\n[2] test_semicolon.csv  (semicolon delimiter — was broken before fix)")
df2 = load_csv("test_data/test_semicolon.csv")
results.append(check("Loaded without error", df2 is not None))
results.append(check("6 columns (not 1)", len(df2.columns) == 6, str(df2.columns.tolist())))
results.append(check("Same data as comma version", df.shape == df2.shape))

# ── 3. Mixed types ───────────────────────────────────────────────────────────
print("\n[3] test_mixed_types.csv  (float / int / categorical / bool)")
df3 = load_csv("test_data/test_mixed_types.csv")
from parameter_config import ParameterType
types = {col: infer_column_type(df3[col]) for col in df3.columns}
results.append(check("temperature -> FLOAT", types["temperature"] == ParameterType.FLOAT, str(types["temperature"])))
results.append(check("num_layers -> INT",    types["num_layers"]  == ParameterType.INT,   str(types["num_layers"])))
results.append(check("material -> CATEGORICAL", types["material"] == ParameterType.CATEGORICAL, str(types["material"])))
results.append(check("use_catalyst -> BOOL", types["use_catalyst"] == ParameterType.BOOL, str(types["use_catalyst"])))

# ── 4. NaN objectives ────────────────────────────────────────────────────────
print("\n[4] test_nan_objectives.csv  (some objective rows are NaN — should be skipped)")
df4 = load_csv("test_data/test_nan_objectives.csv")
results.append(check("Loaded without error", df4 is not None))
results.append(check("NaN rows present in objective", df4["Instability_index"].isna().any()))
complete_rows = df4.dropna(subset=["Instability_index"])
results.append(check("4 complete rows out of 7", len(complete_rows) == 4, f"{len(complete_rows)} complete"))

# ── 5. All-NaN column ────────────────────────────────────────────────────────
print("\n[5] test_all_nan_column.csv  (one entire parameter column is NaN)")
df5 = load_csv("test_data/test_all_nan_column.csv")
results.append(check("Loaded without error", df5 is not None))
results.append(check("extra_param is all NaN", df5["extra_param"].isna().all()))
configs5 = extract_param_defaults(df5, result_columns=["Instability_index"])
extra_cfg = next((c for c in configs5 if c.name == "extra_param"), None)
results.append(check("all-NaN column gets safe fallback range [0, 1]",
                      extra_cfg is not None and extra_cfg.full_min == 0.0 and extra_cfg.full_max == 1.0,
                      f"full_min={getattr(extra_cfg,'full_min',None)}, full_max={getattr(extra_cfg,'full_max',None)}"))

# ── 6. UTF-8 BOM ─────────────────────────────────────────────────────────────
print("\n[6] test_bom.csv  (UTF-8 BOM header from Windows/Excel)")
df6 = load_csv("test_data/test_bom.csv")
results.append(check("Loaded without error", df6 is not None))
results.append(check("First column has no BOM prefix", not df6.columns[0].startswith("\ufeff"),
                      repr(df6.columns[0])))
results.append(check("First column is 'CsPbI'", df6.columns[0] == "CsPbI", repr(df6.columns[0])))

# ── Summary ──────────────────────────────────────────────────────────────────
total = len(results)
passed = sum(results)
print(f"\n{'='*50}")
print(f"Result: {passed}/{total} checks passed")
if passed == total:
    print("ALL CHECKS PASSED")
else:
    print("SOME CHECKS FAILED — review output above")
