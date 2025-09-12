#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_models.py  — feature selection + versioned outputs
--------------------------------------------------------
- Trains two models on a features CSV: RandomForestClassifier (status) and RandomForestRegressor (hb).
- Uses model-based feature selection (top-k by RF importance) and picks k that IMPROVES CV score.
- Saves each run to models/run_YYYYMMDD_HHMMSS/.

USAGE:
  python train_models.py --csv features_dataset.csv --test-size 0.30 --seed 42

Outputs in models/run_YYYYMMDD_HHMMSS/:
  - anemia_rf.joblib
  - hb_rf.joblib
  - metrics.json
  - clf_features.json            (selected features for classifier)
  - reg_features.json            (selected features for regressor)
  - feature_names_all.json       (all usable features before selection)
  - cv_results_classification.csv
  - cv_results_regression.csv
  - confusion_matrix.png
  - README.txt
"""

import argparse, json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, r2_score, mean_absolute_error, mean_squared_error
)

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--test-size", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=-1)
    return p.parse_args()

def coerce_numeric_block(df: pd.DataFrame) -> (pd.DataFrame, dict):
    """Coerce all columns to numeric where possible. Return cleaned df + report of coerced values."""
    non_numeric_report = {}
    X = df.copy()
    for c in X.columns:
        if X[c].dtype == object:
            # normalize decimal commas if any
            X[c] = X[c].astype(str).str.replace(",", ".", regex=False)
        before_na = X[c].isna().sum()
        coerced = pd.to_numeric(X[c], errors="coerce")
        after_na = coerced.isna().sum()
        added = int(after_na - before_na)
        if added > 0:
            non_numeric_report[c] = added
        X[c] = coerced
    # inf handling + median impute
    X = X.replace([np.inf, -np.inf], np.nan)
    med = X.median(numeric_only=True)
    X = X.fillna(med)
    return X, non_numeric_report

def grid_k_list(n_features: int):
    # sensible k grid capped to feature count
    base = [5, 8, 10, 12, 15, 18, 20, 25, 30]
    ks = [k for k in base if k <= n_features]
    if n_features not in ks:
        ks.append(n_features)  # allow "all features"
    return ks

def main():
    args = parse_args()
    path = Path(args.csv)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    # Versioned output dir
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path("models") / f"run_{run_id}"
    outdir.mkdir(parents=True, exist_ok=True)

    # Load + header normalization
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    colmap = {c.lower(): c for c in df.columns}
    if "status" not in colmap or "hb" not in colmap:
        raise ValueError(f"CSV must include 'hb' and 'status' (any casing). Found: {df.columns.tolist()}")
    status_col = colmap["status"]
    hb_col     = colmap["hb"]

    # Build feature matrix candidates (drop known non-features)
    drop_cols = {status_col, hb_col, "filename", "Filename"}
    feature_cols_all = [c for c in df.columns if c not in drop_cols]

    # Coerce numerics robustly
    X_all_raw = df[feature_cols_all].copy()
    X_all, non_numeric_report = coerce_numeric_block(X_all_raw)

    # Targets
    y_cls = (pd.to_numeric(df[status_col], errors="coerce").fillna(0).astype(float) > 0.5).astype(int)
    y_reg = pd.to_numeric(df[hb_col], errors="coerce").astype(float)
    if y_cls.nunique() < 2:
        raise ValueError("status has only one class; need both 0 and 1.")

    # Train/test split
    X_train_all, X_test_all, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
        X_all, y_cls, y_reg, test_size=args.test_size, random_state=args.seed, stratify=y_cls
    )

    # ----- Classification: baseline with ALL features -----
    clf_base = RandomForestClassifier(random_state=args.seed, n_jobs=args.n_jobs)
    param_grid_cls = {
        "n_estimators": [300, 500, 800],
        "max_depth": [None, 8, 12, 18],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"],
        "bootstrap": [True],
    }
    cv_cls = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    gs_cls_all = GridSearchCV(clf_base, param_grid_cls, cv=cv_cls, scoring="roc_auc",
                              n_jobs=args.n_jobs, refit=True, verbose=0)
    gs_cls_all.fit(X_train_all, y_cls_train)
    best_clf_all = gs_cls_all.best_estimator_
    base_cv_auc  = float(gs_cls_all.best_score_)

    # Feature importances from baseline best model
    importances_cls = pd.Series(best_clf_all.feature_importances_, index=X_train_all.columns).sort_values(ascending=False)
    k_candidates = grid_k_list(n_features=X_train_all.shape[1])

    # Search best k by CV ROC-AUC
    best_k_cls = X_train_all.shape[1]
    best_cv_auc = base_cv_auc
    best_clf_k = best_clf_all
    best_feat_cls = list(X_train_all.columns)

    for k in k_candidates:
        topk = list(importances_cls.index[:k])
        gs_k = GridSearchCV(clf_base, param_grid_cls, cv=cv_cls, scoring="roc_auc",
                            n_jobs=args.n_jobs, refit=True, verbose=0)
        gs_k.fit(X_train_all[topk], y_cls_train)
        if gs_k.best_score_ > best_cv_auc + 1e-6:  # strictly improve
            best_cv_auc  = float(gs_k.best_score_)
            best_k_cls   = k
            best_clf_k   = gs_k.best_estimator_
            best_feat_cls= topk

    # Fit final classifier on selected features
    X_train_clf = X_train_all[best_feat_cls]
    X_test_clf  = X_test_all[best_feat_cls]
    best_clf_k.fit(X_train_clf, y_cls_train)
    y_pred_cls  = best_clf_k.predict(X_test_clf)
    y_proba_cls = best_clf_k.predict_proba(X_test_clf)[:,1] if hasattr(best_clf_k,"predict_proba") else None

    cls_metrics = {
        "cv_baseline_auc_all_feats": base_cv_auc,
        "cv_best_auc": best_cv_auc,
        "selected_k": best_k_cls,
        "selected_features": best_feat_cls,
        "holdout_accuracy": float(accuracy_score(y_cls_test, y_pred_cls)),
        "holdout_precision": float(precision_score(y_cls_test, y_pred_cls, zero_division=0)),
        "holdout_recall": float(recall_score(y_cls_test, y_pred_cls, zero_division=0)),
        "holdout_f1": float(f1_score(y_cls_test, y_pred_cls, zero_division=0)),
        "holdout_roc_auc": float(roc_auc_score(y_cls_test, y_proba_cls)) if y_proba_cls is not None else None,
        "cv_best_params": best_clf_k.get_params(),
    }

    # Confusion matrix
    cm = confusion_matrix(y_cls_test, y_pred_cls)
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (Anemia Classifier)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks([0,1],[0,1]); plt.yticks([0,1],[0,1])
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, f"{v}", ha="center", va="center")
    plt.tight_layout()
    (outdir / "confusion_matrix.png").write_bytes(fig.canvas.tostring_rgb()) if False else plt.savefig(outdir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    # Save CV table for "all features" search as reference
    pd.DataFrame(gs_cls_all.cv_results_).to_csv(outdir / "cv_results_classification.csv", index=False)
    joblib.dump(best_clf_k, outdir / "anemia_rf.joblib")
    with open(outdir / "clf_features.json","w") as f:
        json.dump(best_feat_cls, f, indent=2)

    # ----- Regression: baseline with ALL features -----
    rgr_base = RandomForestRegressor(random_state=args.seed, n_jobs=args.n_jobs)
    param_grid_reg = {
        "n_estimators": [500, 800, 1200],
        "max_depth": [None, 10, 14, 18],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"],
        "bootstrap": [True],
    }
    cv_reg = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    gs_reg_all = GridSearchCV(rgr_base, param_grid_reg, cv=cv_reg, scoring="r2",
                              n_jobs=args.n_jobs, refit=True, verbose=0)
    gs_reg_all.fit(X_train_all, y_reg_train)
    best_rgr_all = gs_reg_all.best_estimator_
    base_cv_r2   = float(gs_reg_all.best_score_)

    # Importances + k-search
    importances_reg = pd.Series(best_rgr_all.feature_importances_, index=X_train_all.columns).sort_values(ascending=False)
    k_candidates_reg = grid_k_list(n_features=X_train_all.shape[1])

    best_k_reg = X_train_all.shape[1]
    best_cv_r2 = base_cv_r2
    best_rgr_k = best_rgr_all
    best_feat_reg = list(X_train_all.columns)

    for k in k_candidates_reg:
        topk = list(importances_reg.index[:k])
        gs_k = GridSearchCV(rgr_base, param_grid_reg, cv=cv_reg, scoring="r2",
                            n_jobs=args.n_jobs, refit=True, verbose=0)
        gs_k.fit(X_train_all[topk], y_reg_train)
        if gs_k.best_score_ > best_cv_r2 + 1e-6:
            best_cv_r2  = float(gs_k.best_score_)
            best_k_reg  = k
            best_rgr_k  = gs_k.best_estimator_
            best_feat_reg = topk

    # Fit final regressor
    X_train_reg = X_train_all[best_feat_reg]
    X_test_reg  = X_test_all[best_feat_reg]
    best_rgr_k.fit(X_train_reg, y_reg_train)
    y_pred_reg = best_rgr_k.predict(X_test_reg)

    reg_metrics = {
        "cv_baseline_r2_all_feats": base_cv_r2,
        "cv_best_r2": best_cv_r2,
        "selected_k": best_k_reg,
        "selected_features": best_feat_reg,
        "holdout_r2": float(r2_score(y_reg_test, y_pred_reg)),
        "holdout_mae": float(mean_absolute_error(y_reg_test, y_pred_reg)),
        "holdout_rmse": float(rmse(y_reg_test, y_pred_reg)),
        "cv_best_params": best_rgr_k.get_params(),
    }

    # Save CV table for "all features" search as reference
    pd.DataFrame(gs_reg_all.cv_results_).to_csv(outdir / "cv_results_regression.csv", index=False)
    joblib.dump(best_rgr_k, outdir / "hb_rf.joblib")
    with open(outdir / "reg_features.json","w") as f:
        json.dump(best_feat_reg, f, indent=2)

    # ----- Save meta & summaries -----
    with open(outdir / "feature_names_all.json","w") as f:
        json.dump(list(X_all.columns), f, indent=2)

    report = {
        "n_samples": int(len(df)),
        "n_features_all": int(X_all.shape[1]),
        "non_numeric_values_coerced": non_numeric_report,
        "classification": cls_metrics,
        "regression": reg_metrics,
        "test_size": args.test_size,
        "random_state": args.seed,
        "run_dir": str(outdir)
    }
    with open(outdir / "metrics.json","w") as f:
        json.dump(report, f, indent=2)

    with open(outdir / "README.txt","w") as f:
        f.write(
            "This folder contains a single model run.\n"
            "Classifier uses features from clf_features.json.\n"
            "Regressor uses features from reg_features.json.\n"
            "Compare metrics.json across runs and pick the best.\n"
        )

    print(f"\nSaved models & reports to: {outdir.resolve()}")
    print(json.dumps({
        "cv_auc_all": base_cv_auc, "cv_auc_best": best_cv_auc, "clf_k": best_k_cls,
        "cv_r2_all": base_cv_r2, "cv_r2_best": best_cv_r2, "reg_k": best_k_reg
    }, indent=2))

if __name__ == "__main__":
    main()
