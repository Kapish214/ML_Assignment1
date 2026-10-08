"""Inference: load a saved model and write the test-set predictions (no retraining).

Usage:  python src/predict.py            # both problems
        python src/predict.py 1          # only var1
Writes IMT2024027_pred_var{1,2}.csv in the repository root.
"""
import sys

import joblib
import pandas as pd

from common import MODELS, ROLL, ROOT, load


def predict(var):
    saved = joblib.load(MODELS / f"var{var}.joblib")
    _, te = load(var)
    pred = saved["model"].predict(te[saved["features"]].values)
    path = ROOT / f"{ROLL}_pred_var{var}.csv"
    pd.DataFrame({"y": pred}).to_csv(path, index=False)
    print(f"var{var}: wrote {len(pred)} predictions to {path.name}")


if __name__ == "__main__":
    for v in ([int(a) for a in sys.argv[1:]] or [1, 2]):
        predict(v)
