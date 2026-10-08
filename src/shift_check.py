"""Covariate-shift check for var1.

The var1 test inputs are clipped to +/-1 far more often than the training inputs
(about 65% of test rows have >=3 clipped coordinates vs 30% in train). This script re-scores the
best candidate configurations from train.py with out-of-fold errors that are
importance-weighted so that the training rows mimic the test distribution of
"number of clipped coordinates" (k = 0..6).
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold, cross_val_predict

from common import ROLL, OUT, build

warnings.filterwarnings("ignore", category=ConvergenceWarning)
ROOT = Path(__file__).resolve().parents[1]


def main():
    for var, cands in ((1, [(3, "lasso", 7e-3), (4, "lasso", 7e-3), (5, "lasso", 5e-3), (5, "lasso", 7e-3),
                            (5, "lasso", 1e-2), (5, "ridge", 30), (6, "lasso", 1e-2), (7, "lasso", 1e-2)]),
                       (2, [(8, "ridge", 0.1), (10, "ridge", 1), (12, "ridge", 3)])):
        tr = pd.read_csv(ROOT / f"{ROLL}_train_var{var}.csv")
        te = pd.read_csv(ROOT / f"{ROLL}_test_var{var}.csv")
        feats = [c for c in tr.columns if c != "y"]
        X, y = tr[feats].values, tr["y"].values
        k_tr = (np.abs(X) == 1).sum(1)
        k_te = (np.abs(te[feats].values) == 1).sum(1)
        n = len(feats) + 1
        p_tr = np.bincount(k_tr, minlength=n) / len(k_tr)
        p_te = np.bincount(k_te, minlength=n) / len(k_te)
        w = p_te[k_tr] / np.maximum(p_tr[k_tr], 1e-12)
        print(f"\nvar{var}: P(k clipped) train={np.round(p_tr, 3).tolist()} test={np.round(p_te, 3).tolist()}")
        rows = []
        for d, reg, a in cands:
            errs = []
            for seed in range(3):
                oof = cross_val_predict(build(d, reg, a), X, y, cv=KFold(5, shuffle=True, random_state=seed), n_jobs=-1)
                errs.append((oof - y) ** 2)
            e = np.mean(errs, 0)
            by_k = {f"k{k}": e[k_tr == k].mean() for k in range(n) if (k_tr == k).sum() >= 10}
            rows.append(dict(degree=d, reg=reg, alpha=a, mse=e.mean(), shift_weighted_mse=np.average(e, weights=w), **by_k))
            print(f"  deg={d} {reg:5s} alpha={a:<6g} MSE={e.mean():.4f} shift-weighted={rows[-1]['shift_weighted_mse']:.4f}",
                  " ".join(f"{k}={v:.3f}" for k, v in by_k.items()), flush=True)
        pd.DataFrame(rows).to_csv(OUT / f"var{var}_shift_check.csv", index=False)


if __name__ == "__main__":
    main()
