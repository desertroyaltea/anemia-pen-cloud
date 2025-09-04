#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Optional, Dict

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
import joblib

# LightGBM
try:
    from lightgbm import LGBMRegressor
except Exception as e:
    raise RuntimeError("Please install lightgbm: pip install lightgbm") from e

FEATURE_COLUMNS = [
    "R_norm_p50","a_mean","R_p50","R_p10","RG","S_p50",
    "gray_p90","gray_kurt","gray_std","gray_mean",
    "B_mean","B_p10","B_p75","G_kurt"
]

# ---- Simple split-conformal wrapper ----
class ConformalRegressor:
    def __init__(self, base_estimator, q: float):
        self.base = base_estimator
        self.q = float(q)  # quantile of |residual|
    def predict(self, X):
        return self.base.predict(X)
    def predict_interval(self, X, alpha: float = 0.1):
        y = self.base.predict(X)
        q = self.q if np.isfinite(self.q) else 0.0
        return np.c_[y - q, y + q]

def nested_cv_evaluate(X, y, groups, build_search, outer_folds=5, inner_folds=3, random_state=42):
    if groups is not None:
        outer = GroupKFold(n_splits=outer_folds)
        inner_builder = lambda: GroupKFold(n_splits=inner_folds)
    else:
        outer = KFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
        inner_builder = lambda: KFold(n_splits=inner_folds, shuffle=True, random_state=random_state)

    outer_metrics = []
    models = []

    for fold, (tr_idx, te_idx) in enumerate(outer.split(X, y, groups), 1):
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]
        gtr = groups.iloc[tr_idx] if groups is not None else None

        search = build_search(inner_builder(), gtr)
        search.fit(Xtr, ytr, **({"groups": gtr} if gtr is not None else {}))
        best = search.best_estimator_
        yhat = best.predict(Xte)

        mae = mean_absolute_error(yte, yhat)
        rmse = mean_squared_error(yte, yhat, squared=False)
        r2 = r2_score(yte, yhat)

        outer_metrics.append({"fold": fold, "MAE": mae, "RMSE": rmse, "R2": r2, "best_params": search.best_params_})
        models.append(best)

    return outer_metrics, models

def build_pipelines():
    # Preprocess: impute + robust scale
    pre = ColumnTransformer(
        [("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("sc", RobustScaler())]), FEATURE_COLUMNS)],
        remainder="drop"
    )

    pipelines = {
        "elastic": Pipeline([("pre", pre), ("reg", ElasticNet(max_iter=2000))]),
        "svr":     Pipeline([("pre", pre), ("reg", SVR())]),
        "lgbm":    Pipeline([("pre", pre), ("reg", LGBMRegressor(n_estimators=600, objective="l2", random_state=42))]),
    }

    param_dists = {
        "elastic": {"reg__alpha": np.logspace(-3, 1, 30), "reg__l1_ratio": np.linspace(0.0, 1.0, 21)},
        "svr":     {"reg__C": np.logspace(-1, 3, 30), "reg__gamma": np.logspace(-4, 0, 30), "reg__epsilon": np.logspace(-3, -0.3, 20)},
        "lgbm":    {"reg__num_leaves": [15, 31, 63, 127],
                    "reg__min_child_samples": [5, 10, 20, 40],
                    "reg__learning_rate": [0.01, 0.03, 0.05, 0.1],
                    "reg__feature_fraction": [0.7, 0.8, 0.9, 1.0]}
    }
    return pipelines, param_dists

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--out_model", default="outputs/models/anemia_pen.joblib")
    ap.add_argument("--out_card",  default="outputs/models/model_card.json")
    ap.add_argument("--outer_folds", type=int, default=5)
    ap.add_argument("--inner_folds", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.10, help="Conformal miscoverage (0.10 -> 90% interval)")
    args = ap.parse_args()

    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features_csv)
    for col in ["hb"] + FEATURE_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing column in features_csv: {col}")

    X = df[FEATURE_COLUMNS].copy()
    y = df["hb"].astype(float)
    groups = df["SubjectID"] if "SubjectID" in df.columns else None

    pipelines, param_dists = build_pipelines()

    # Build a nested-CV search factory
    def search_builder(cv_inner, inner_groups):
        def make(pipe, space):
            return RandomizedSearchCV(
                estimator=pipe,
                param_distributions=space,
                n_iter=40,
                scoring="neg_mean_absolute_error",
                cv=cv_inner,
                random_state=42,
                n_jobs=-1,
                refit=True
            )
        return {
            name: make(pipelines[name], param_dists[name]) for name in pipelines
        }

    # Evaluate each model family in nested CV
    all_metrics = {}
    best_family = None
    best_mae = np.inf
    best_models_for_family = None

    for name in pipelines:
        def build_search(cv_inner, inner_groups):
            return search_builder(cv_inner, inner_groups)[name]
        metrics, models = nested_cv_evaluate(X, y, groups, build_search,
                                             outer_folds=args.outer_folds, inner_folds=args.inner_folds)
        all_metrics[name] = metrics
        mae_mean = float(np.mean([m["MAE"] for m in metrics]))
        if mae_mean < best_mae:
            best_mae = mae_mean
            best_family = name
            best_models_for_family = models

    # Refit final model on all data with a larger search, then split for conformal calibration
    print(f"[SELECT] Best family: {best_family} (outer-CV MAE ≈ {best_mae:.3f})")
    from sklearn.model_selection import train_test_split
    if groups is not None:
        # calibration split at subject level
        uniq = df["SubjectID"].dropna().unique()
        tr_sid, cal_sid = train_test_split(uniq, test_size=0.2, random_state=42)
        tr_mask = df["SubjectID"].isin(tr_sid)
        cal_mask = df["SubjectID"].isin(cal_sid)
    else:
        tr_mask, cal_mask = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)

    Xtr, ytr = X[tr_mask], y[tr_mask]
    Xcal, ycal = X[cal_mask], y[cal_mask]

    # Bigger inner search on training set
    cv_inner = GroupKFold(n_splits=5) if groups is not None else KFold(n_splits=5, shuffle=True, random_state=42)
    rs = RandomizedSearchCV(
        estimator=pipelines[best_family],
        param_distributions=param_dists[best_family],
        n_iter=80, cv=cv_inner, n_jobs=-1, random_state=42,
        scoring="neg_mean_absolute_error", refit=True
    )
    rs.fit(Xtr, ytr, **({"groups": df.loc[tr_mask, "SubjectID"]} if groups is not None else {}))
    final_model = rs.best_estimator_

    # Conformal calibration
    cal_pred = final_model.predict(Xcal)
    resid = np.abs(ycal - cal_pred)
    q = float(np.quantile(resid, 1.0 - args.alpha))

    conformal = ConformalRegressor(final_model, q)

    bundle = {
        "model": conformal,
        "meta": {
            "family": best_family,
            "outer_cv": all_metrics,
            "alpha": args.alpha,
            "q_abs_resid": q,
            "features": FEATURE_COLUMNS,
            "preprocessing": "SimpleImputer(median)+RobustScaler in pipeline",
            "lightgbm_version": "see environment",
        }
    }
    joblib.dump(bundle, args.out_model)

    # Model card
    card = {
        "n_rows": int(len(df)),
        "features": FEATURE_COLUMNS,
        "target": "hb",
        "best_family": best_family,
        "outer_cv_summary": {
            k: { "MAE_mean": float(np.mean([m["MAE"] for m in v])),
                 "RMSE_mean": float(np.mean([m["RMSE"] for m in v])),
                 "R2_mean": float(np.mean([m["R2"] for m in v]))} for k, v in all_metrics.items()
        },
        "alpha": args.alpha,
        "conformal_q": q
    }
    Path(args.out_card).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_card).write_text(json.dumps(card, indent=2))
    print(f"[OK] Saved model to {args.out_model}")
    print(f"[OK] Saved model card to {args.out_card}")

if __name__ == "__main__":
    main()
