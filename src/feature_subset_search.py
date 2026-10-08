"""Exhaustive feature-subset search for var1 (report §3.2).

For every non-empty subset of x1..x6 (63 subsets), every degree 1..7 and a small
Ridge alpha grid, compute the repeated 5-fold CV MSE. Shows that no subset of the
inputs comes close to using all six features.

Usage:  python src/feature_subset_search.py   (writes results/var1_feature_subset_search.csv)
"""
import itertools
from pathlib import Path

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
ROLL = "IMT2024027"
CV = RepeatedKFold(n_splits=5, n_repeats=3, random_state=0)
ALPHAS = [0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2]
DEGREES = range(1, 8)


def main():
    tr = pd.read_csv(ROOT / f"{ROLL}_train_var1.csv")
    feats = [c for c in tr.columns if c != "y"]
    y = tr["y"].values
    (ROOT / "results").mkdir(exist_ok=True)
    rows = []
    for k in range(1, len(feats) + 1):
        for S in itertools.combinations(feats, k):
            X = tr[list(S)].values
            for d in DEGREES:
                for a in ALPHAS:
                    model = make_pipeline(PolynomialFeatures(d, include_bias=False), StandardScaler(),
                                          Ridge(alpha=a, solver="auto" if a > 0 else "svd"))
                    s = -cross_val_score(model, X, y, cv=CV, scoring="neg_mean_squared_error", n_jobs=-1)
                    rows.append(dict(features="+".join(S), degree=d, alpha=a, cv_mse=s.mean(), cv_std=s.std()))
        df = pd.DataFrame(rows)
        top = df.loc[df.cv_mse.idxmin()]
        print(f"best with <= {k} features: {top.features} degree={top.degree} alpha={top.alpha:g} "
              f"CV-MSE={top.cv_mse:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results" / "var1_feature_subset_search.csv", index=False)


if __name__ == "__main__":
    main()
