# ML Assignment 1 — Polynomial Regression (IMT2024027)

Polynomial regression for two personalised datasets: steam-turbine Net Power Score (`var1`) and thermal-reservoir anomaly score (`var2`).

| Problem | Inputs | Chosen model |
|---|---|---|
| var1 | x1…x6 | degree-5 polynomial, Lasso α = 0.007 |
| var2 | x1, x2, x3 | degree-10 polynomial, Ridge α = 1 |

## Layout

```
IMT2024027_{train,test}_var{1,2}.csv   input data
src/common.py                          shared code: model pipeline, CV scoring, selection / training routine
src/train_var1.py                      problem 1: selection, training, saves model, writes predictions
src/train_var2.py                      problem 2: same for problem 2
src/predict.py                         inference only: loads a saved model and writes the prediction CSV
src/shift_check.py                     covariate-shift-weighted validation (report §5)
src/feature_subset_search.py           exhaustive feature-subset search for var1 (report §3.2)
models/var{1,2}.joblib                 saved fitted models (sklearn pipelines)
models/var{1,2}_coefficients.csv       the polynomial terms and their (standardised) weights
models/var{1,2}_intercept.txt          intercepts
requirements.txt                       Python dependencies
```

## Run

```bash
pip install -r requirements.txt

# Inference only (seconds): uses the saved models in models/
python src/predict.py            # writes IMT2024027_pred_var1.csv and IMT2024027_pred_var2.csv

# Full retraining (about 10-12 min each); also writes results/ (CV tables, plots, logs)
python src/train_var1.py
python src/train_var2.py

# Optional extra checks
python src/feature_subset_search.py    # ~10 min, report §3.2
cd src && python shift_check.py        # report §5
```

Model: `PolynomialFeatures(d) -> StandardScaler -> Ridge/Lasso(α)`. Degree, penalty type and α are chosen jointly by 5-fold × 2-repeat cross-validation on the training data only, then the best model is refit on the full training set to predict the test set. Everything is seeded (`random_state = 0`).

To compute a prediction from the coefficient files by hand: `y = intercept + Σ_j coef_j · (term_j − mean_j) / scale_j`, where `term_j` is the j-th monomial listed in `var{N}_coefficients.csv`.
