"""Problem 1 (var1): steam-turbine Net Power Score, inputs x1..x6.

Selects degree (1-10), penalty (Ridge/Lasso) and alpha by repeated 5-fold CV, trains the final
model, saves it to models/var1.joblib and writes IMT2024027_pred_var1.csv.

Usage:  python src/train_var1.py
"""
from common import main

CONFIG = dict(degrees=range(1, 11), lasso_max_degree=8,
              ridge=[1e-6, 1e-2, 1e-1, 1, 3, 10, 30, 100, 300],
              lasso=[1e-3, 3e-3, 5e-3, 7e-3, 1e-2, 1.5e-2, 2e-2, 3e-2])

if __name__ == "__main__":
    main(1, CONFIG)
