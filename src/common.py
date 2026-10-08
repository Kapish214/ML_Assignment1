"""Shared code for both problems: paths, model pipeline, CV scoring and the model-selection routine.

Model family (pure polynomial regression):
    PolynomialFeatures(degree) -> StandardScaler -> Ridge(alpha) | Lasso(alpha)
"""
import json, time, warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_val_predict, cross_val_score, KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parents[1]
ROLL = "IMT2024027"
OUT = ROOT / "results"
MODELS = ROOT / "models"
SEED = 0
CV = RepeatedKFold(n_splits=5, n_repeats=2, random_state=SEED)


def build(degree, reg, alpha):
    est = Ridge(alpha=alpha) if reg == "ridge" else Lasso(alpha=alpha, max_iter=50000, tol=1e-4)
    return make_pipeline(PolynomialFeatures(degree, include_bias=False), StandardScaler(), est)


def cv_score(X, y, degree, reg, alpha):
    s = -cross_val_score(build(degree, reg, alpha), X, y, cv=CV,
                         scoring="neg_mean_squared_error", n_jobs=-1)
    return s.mean(), s.std(ddof=1) / np.sqrt(len(s))


def load(var):
    tr = pd.read_csv(ROOT / f"{ROLL}_train_var{var}.csv")
    te = pd.read_csv(ROOT / f"{ROLL}_test_var{var}.csv")
    return tr, te


def save_model(var, model, feats):
    """Save the fitted pipeline (joblib) and a readable table of its polynomial coefficients.

    The prediction is  intercept + sum_j coef_j * (term_j - mean_j) / scale_j,
    where term_j is the j-th monomial of the inputs.
    """
    MODELS.mkdir(exist_ok=True)
    joblib.dump(dict(model=model, features=feats), MODELS / f"var{var}.joblib")
    poly, scaler, est = model[0], model[1], model[2]
    pd.DataFrame(dict(term=poly.get_feature_names_out(feats), coef_standardised=est.coef_,
                      mean=scaler.mean_, scale=scaler.scale_)).to_csv(MODELS / f"var{var}_coefficients.csv", index=False)
    (MODELS / f"var{var}_intercept.txt").write_text(f"{float(est.intercept_)!r}\n")


def run(var, cfg, log):
    """Select (degree, penalty, alpha) by CV, validate, refit on all training data, predict the test set."""
    OUT.mkdir(exist_ok=True)
    tr, te = load(var)
    feats = [c for c in tr.columns if c != "y"]
    X, y = tr[feats].values, tr["y"].values
    log(f"\n=== var{var}: train {tr.shape}, test {te.shape}, Var(y)={y.var():.4f}")

    # 1) Grid search: degree x regulariser x alpha
    rows = []
    for d in cfg["degrees"]:
        grid = [("ridge", a) for a in cfg["ridge"]]
        if d <= cfg["lasso_max_degree"]:
            grid += [("lasso", a) for a in cfg["lasso"]]
        for reg, a in grid:
            m, se = cv_score(X, y, d, reg, a)
            rows.append(dict(degree=d, reg=reg, alpha=a, n_terms=build(d, reg, a)[0].fit(X[:2]).n_output_features_,
                             cv_mse=m, cv_se=se))
        g = pd.DataFrame(rows)
        b = g[g.degree == d].sort_values("cv_mse").iloc[0]
        log(f"  degree {d:2d} ({int(b.n_terms):5d} terms): best {b.reg:5s} alpha={b.alpha:<8g} CV-MSE={b.cv_mse:.4f}")
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / f"var{var}_cv_grid.csv", index=False)

    best = grid.sort_values("cv_mse").iloc[0]
    d, reg, a = int(best.degree), best.reg, float(best.alpha)
    # one-standard-error rule, reported for reference
    per_deg = grid.loc[grid.groupby("degree").cv_mse.idxmin()]
    one_se = int(per_deg[per_deg.cv_mse <= best.cv_mse + best.cv_se].degree.min())
    log(f"Selected: degree={d}, {reg}, alpha={a:g}, CV-MSE={best.cv_mse:.4f} (+/-{best.cv_se:.4f}); "
        f"1-SE rule smallest degree = {one_se}")

    # 2) Feature ablation at the selected configuration (drop one input at a time)
    abl = [dict(dropped="none", cv_mse=float(best.cv_mse))]
    for f in feats:
        keep = [c for c in feats if c != f]
        m, _ = cv_score(tr[keep].values, y, d, reg, a)
        abl.append(dict(dropped=f, cv_mse=m))
        log(f"  drop {f}: CV-MSE={m:.4f}")
    pd.DataFrame(abl).to_csv(OUT / f"var{var}_feature_ablation.csv", index=False)

    # 3) Independent 80/20 hold-out check of the selected configuration
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.2, random_state=SEED)
    hb = build(d, reg, a).fit(Xa, ya).predict(Xb)
    holdout = dict(mse=float(mean_squared_error(yb, hb)), r2=float(r2_score(yb, hb)))
    log(f"Hold-out 20%: MSE={holdout['mse']:.4f}, R2={holdout['r2']:.4f}")

    # 4) Out-of-fold predictions for the parity plot
    oof = cross_val_predict(build(d, reg, a), X, y, cv=KFold(5, shuffle=True, random_state=SEED), n_jobs=-1)

    # 5) Refit on all training data, save the model, predict test
    model = build(d, reg, a).fit(X, y)
    save_model(var, model, feats)
    ptr = model.predict(X)
    pred = model.predict(te[feats].values)
    pd.DataFrame({"y": pred}).to_csv(ROOT / f"{ROLL}_pred_var{var}.csv", index=False)
    coef = model[-1].coef_
    result = dict(features=feats, degree=d, regulariser=reg, alpha=a, n_terms=int(best.n_terms),
                  nonzero_terms=int(np.sum(np.abs(coef) > 1e-10)),
                  cv_mse=float(best.cv_mse), cv_se=float(best.cv_se), cv_r2=float(1 - best.cv_mse / y.var()),
                  oof_r2=float(r2_score(y, oof)), oof_mse=float(mean_squared_error(y, oof)),
                  holdout=holdout, one_se_degree=one_se,
                  train_mse=float(mean_squared_error(y, ptr)), train_r2=float(r2_score(y, ptr)))
    log(f"FINAL var{var}: {json.dumps(result)}")
    (OUT / f"var{var}_final.json").write_text(json.dumps(result, indent=2))

    # Plots
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for r, c in (("ridge", "#2a6fdb"), ("lasso", "#d9822b")):
        s = grid[grid.reg == r].groupby("degree").cv_mse.min()
        ax.plot(s.index, s.values, "o-", color=c, label=f"{r.capitalize()} (best alpha per degree)")
    ax.axvline(d, color="grey", ls="--", lw=1)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_yscale("log"); ax.set_xlabel("polynomial degree"); ax.set_ylabel("5-fold CV MSE (log)")
    ax.set_title(f"var{var}: CV error vs degree"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / f"var{var}_cv_vs_degree.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(y, oof, s=6, alpha=.5, color="#2a6fdb")
    lo, hi = min(y.min(), oof.min()), max(y.max(), oof.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("actual y"); ax.set_ylabel("out-of-fold predicted y")
    ax.set_title(f"var{var}: R2={result['oof_r2']:.4f}")
    fig.tight_layout(); fig.savefig(OUT / f"var{var}_parity.png", dpi=150); plt.close(fig)
    return result


def main(var, cfg):
    """Entry point used by train_var1.py / train_var2.py."""
    OUT.mkdir(exist_ok=True)
    logf = open(OUT / f"var{var}_train_log.txt", "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n"); logf.flush()

    t0 = time.time()
    run(var, cfg, log)
    log(f"\nDone in {time.time() - t0:.0f}s")
