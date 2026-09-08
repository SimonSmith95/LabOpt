import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold, cross_val_score, KFold, cross_val_predict
from scipy.stats import pearsonr

cases = [
    ("test_data/branin_grid.csv",       ["x1", "x2"],                          "branin_value"),
    ("test_data/hartmann6_samples.csv", ["x1","x2","x3","x4","x5","x6"],       "hartmann_value"),
]

for fname, xcols, ycol in cases:
    df = pd.read_csv(fname)
    X  = df[xcols].values
    y  = df[ycol].values
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
    scores = cross_val_score(rf, X, y, cv=rkf, scoring="r2")
    y_cv  = cross_val_predict(rf, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=0))
    r, _  = pearsonr(y, y_cv)
    print(f"{fname}  n={len(df)}  R2={np.mean(scores):.3f}+/-{np.std(scores):.3f}  r={r:.3f}")
