"""Problem 2 (var2): thermal-reservoir Thermal Anomaly Score, inputs x1, x2, x3.

Selects degree (1-20), penalty (Ridge/Lasso) and alpha by repeated 5-fold CV, trains the final
model, saves it to models/var2.joblib and writes IMT2024027_pred_var2.csv.

Usage:  python src/train_var2.py
"""
from common import main

CONFIG = dict(degrees=range(1, 21), lasso_max_degree=20,
              ridge=[1e-6, 1e-3, 1e-2, 1e-1, 0.3, 1, 3, 10, 30],
              lasso=[3e-4, 1e-3, 3e-3])

if __name__ == "__main__":
    main(2, CONFIG)
