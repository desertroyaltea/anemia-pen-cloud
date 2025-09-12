#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train two Random-Forest models on your full feature set:
- Classification (Status: 1=Anemia, 0=Not Anemia)
- Regression (hb: hemoglobin)

Usage:
  python train_rf_models.py --csv final_training_data.csv \
    --test-size 0.30 --seed 42

Outputs:
  anemia_rf.joblib, hb_rf.joblib, metrics.json, feature_names.json,
  confusion_matrix.png, cv_results_classification.csv, cv_results_regression.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, r2_score, mean_absolute_error, mean_squared_error
)
import joblib
import matplotlib.pyplot as plt


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)


def parse_args():
    p = argparse.ArgumentParser(description="Train RF models for anemia (Status) and Hb.")
    p.add_argument("--csv", required=True, help="Path to training CSV (must contain 'hb' and 'Status'/'status').")
    p.add_argument("--test-size", type=float, default=0.30, help="Holdout ratio (default 0.30).")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for RF/GridSearch (default -1).")
    return p.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.csv)
    if not data_path.exists():
        raise FileNotFoundError(f"CSV not found: {data_path}")

    # ---- Load data ----
    df = pd.read_csv(data_path)

    # Normalize column names we rely on
    colmap = {c: c for c in df.columns}
    # Accept either 'Status' or 'status'
    if "Status" in df.columns:
        status_col = "Status"
    elif "status" in df.columns:
        status_col = "status"
    else:
        raise ValueError(f"Expected 'Status' or 'status' column in {data_path.name}. Found: {df.columns.tolist()}")

    if "hb" not in df.columns:
        raise ValueError(f"Expected 'hb' column in {data_path.name}. Found: {df.columns.tolist()}")

    # Features = everything except the targets
    feature_cols = [c for c in df.columns if c not in ("hb", status_col)]
    X = df[feature_cols].copy()

    # Targets
    y_cls = (df[status_col].astype(float) > 0.5).astype(int)
    y_reg = df["hb"].astype(float)

    # ---- Train/Test split ----
    X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
        X, y_cls, y_reg, test_size=args.test_size, random_state=args.seed, stratify=y_cls
    )

    # ===========================
    # Classification: RandomForest
    # ===========================
    clf_base = RandomForestClassifier(random_state=args.seed, n_jobs=args.n_jobs)
    param_grid_cls = {
        "n_estimators": [300, 500, 800],
        "max_depth": [None, 8, 12, 18],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"],
        "bootstrap": [True]
    }
    cv_cls = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    gs_cls = GridSearchCV(
        clf_base, param_grid_cls, cv=cv_cls, scoring="roc_auc",
        n_jobs=args.n_jobs, refit=True, verbose=0
    )
    gs_cls.fit(X_train, y_cls_train)
    clf = gs_cls.best_estimator_

    # Holdout evaluation
    y_pred_cls = clf.predict(X_test)
    y_proba_cls = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

    cls_metrics = {
        "holdout_accuracy": float(accuracy_score(y_cls_test, y_pred_cls)),
        "holdout_precision": float(precision_score(y_cls_test, y_pred_cls, zero_division=0)),
        "holdout_recall": float(recall_score(y_cls_test, y_pred_cls, zero_division=0)),
        "holdout_f1": float(f1_score(y_cls_test, y_pred_cls, zero_division=0)),
        "holdout_roc_auc": float(roc_auc_score(y_cls_test, y_proba_cls)) if y_proba_cls is not None else None,
        "cv_best_params": gs_cls.best_params_,
        "cv_best_score_roc_auc": float(gs_cls.best_score_),
    }

    # Confusion matrix plot
    cm = confusion_matrix(y_cls_test, y_pred_cls)
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (Anemia Classifier)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks([0, 1], [0, 1]); plt.yticks([0, 1], [0, 1])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, f"{v}", ha="center", va="center")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=160)
    plt.close(fig)

    # Save CV table (optional)
    pd.DataFrame(gs_cls.cv_results_).to_csv("cv_results_classification.csv", index=False)

    # Save classifier
    joblib.dump(clf, "anemia_rf.joblib")

    # =========================
    # Regression: RandomForest
    # =========================
    rgr_base = RandomForestRegressor(random_state=args.seed, n_jobs=args.n_jobs)
    param_grid_reg = {
        "n_estimators": [500, 800, 1200],
        "max_depth": [None, 10, 14, 18],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"],
        "bootstrap": [True]
    }
    cv_reg = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    gs_reg = GridSearchCV(
        rgr_base, param_grid_reg, cv=cv_reg, scoring="r2",
        n_jobs=args.n_jobs, refit=True, verbose=0
    )
    gs_reg.fit(X_train, y_reg_train)
    rgr = gs_reg.best_estimator_

    # Holdout evaluation
    y_pred_reg = rgr.predict(X_test)
    reg_metrics = {
        "holdout_r2": float(r2_score(y_reg_test, y_pred_reg)),
        "holdout_mae": float(mean_absolute_error(y_reg_test, y_pred_reg)),
        "holdout_rmse": float(rmse(y_reg_test, y_pred_reg)),
        "cv_best_params": gs_reg.best_params_,
        "cv_best_score_r2": float(gs_reg.best_score_),
    }

    # Save CV table (optional)
    pd.DataFrame(gs_reg.cv_results_).to_csv("cv_results_regression.csv", index=False)

    # Save regressor
    joblib.dump(rgr, "hb_rf.joblib")

    # =========================
    # Combined report & features
    # =========================
    report = {
        "n_samples": int(len(df)),
        "n_features": len(feature_cols),
        "features_preview": feature_cols[:10],
        "classification": cls_metrics,
        "regression": reg_metrics,
        "test_size": args.test_size,
        "random_state": args.seed
    }
    with open("metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    with open("feature_names.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    print("Saved artifacts:")
    print(" - anemia_rf.joblib")
    print(" - hb_rf.joblib")
    print(" - metrics.json")
    print(" - feature_names.json")
    print(" - confusion_matrix.png")
    print(" - cv_results_classification.csv")
    print(" - cv_results_regression.csv")


if __name__ == "__main__":
    main()
