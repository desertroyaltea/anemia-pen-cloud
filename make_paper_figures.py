#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_paper_figures.py

Generates publication-ready figures for the latest (or selected) model run.

USAGE EXAMPLES
--------------
# Auto-detect latest run under ./models and use features_dataset_fast.csv if present
python make_paper_figures.py

# Specify a run dir and a specific CSV
python make_paper_figures.py --run-dir "models/run_20250912_152931" --csv "features_dataset_fast.csv"

Outputs a 'figures/' subfolder inside the run directory with PNGs listed in the message above.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, KFold, train_test_split, cross_val_predict
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
    r2_score, mean_absolute_error
)
from sklearn.inspection import PartialDependenceDisplay

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

VASC_FEATURES = [
    "vessel_area_fraction", "mean_vesselness", "p90_vesselness",
    "skeleton_len_per_area", "branchpoint_density", "tortuosity_mean"
]

KEY_COLOR_FEATURES = [
    "R_p10","R_p50","R_norm_p50","a_mean","RG",
    "gray_mean","gray_kurt","gray_p90","S_p50","B_p10","B_mean","B_p75","gray_std"
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models-root", default="models", help="Root folder containing run_* subfolders")
    p.add_argument("--run-dir", default=None, help="Specific run directory (e.g., models/run_YYYYMMDD_HHMMSS)")
    p.add_argument("--csv", default=None, help="Features CSV (default: features_dataset_fast.csv if exists else features_dataset.csv)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def find_latest_run(models_root: Path) -> Path:
    runs = sorted([p for p in models_root.glob("run_*") if p.is_dir()], key=lambda p: p.name, reverse=True)
    if not runs:
        raise FileNotFoundError(f"No run_* folders found under {models_root}")
    return runs[0]

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype(str).str.replace(",", ".", regex=False)
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    return X

def main():
    args = parse_args()
    models_root = Path(args.models_root)

    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run(models_root)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # Choose CSV
    if args.csv:
        csv_path = Path(args.csv)
    else:
        cand1 = Path("features_dataset_fast.csv")
        cand2 = Path("features_dataset.csv")
        csv_path = cand1 if cand1.exists() else cand2
    if not csv_path.exists():
        raise FileNotFoundError(f"Features CSV not found: {csv_path}")

    # Load artifacts from the run
    metrics = load_json(run_dir / "metrics.json")
    clf_features = load_json(run_dir / "clf_features.json")
    reg_features = load_json(run_dir / "reg_features.json")
    clf = joblib.load(run_dir / "anemia_rf.joblib")
    rgr = joblib.load(run_dir / "hb_rf.joblib")

    # Load dataset and build matrices
    df = pd.read_csv(csv_path)
    # normalize label cols
    cols_map = {c.lower(): c for c in df.columns}
    if "status" not in cols_map or "hb" not in cols_map:
        raise ValueError("CSV must include 'status' and 'hb' columns")
    status_col = cols_map["status"]
    hb_col = cols_map["hb"]

    # coerce numeric for features
    drop_cols = {status_col, hb_col, "filename", "Filename"}
    df_num = coerce_numeric(df.drop(columns=[c for c in df.columns if c in drop_cols], errors="ignore"))
    # Keep only columns that exist in df for selected feature sets
    clf_feats_in_df = [f for f in clf_features if f in df_num.columns]
    reg_feats_in_df = [f for f in reg_features if f in df_num.columns]

    Xc = df_num[clf_feats_in_df].copy()
    Xr = df_num[reg_feats_in_df].copy()
    yc = (pd.to_numeric(df[status_col], errors="coerce").fillna(0).astype(float) > 0.5).astype(int).values
    yr = pd.to_numeric(df[hb_col], errors="coerce").astype(float).values

    # figures folder
    figdir = run_dir / "figures"
    figdir.mkdir(exist_ok=True)

    # -------------------------
    # Figure 01: class distribution
    # -------------------------
    plt.figure()
    classes, counts = np.unique(yc, return_counts=True)
    plt.bar([str(int(c)) for c in classes], counts)
    plt.xlabel("Status"); plt.ylabel("Count")
    plt.title("Dataset Class Distribution")
    plt.tight_layout()
    plt.savefig(figdir / "Figure_01_dataset_class_distribution.png"); plt.close()

    # -------------------------
    # Figure 02: feature correlation heatmap (top 20 by variance)
    # -------------------------
    df_corr = df_num.copy()
    # choose up to 20 columns with highest variance for readability
    topk = df_corr.var().sort_values(ascending=False).head(20).index.tolist()
    C = df_corr[topk].corr(numeric_only=True)
    plt.figure(figsize=(8,6))
    im = plt.imshow(C, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(topk)), topk, rotation=90)
    plt.yticks(range(len(topk)), topk)
    plt.title("Feature Correlation (Top-Variance Features)")
    plt.tight_layout()
    plt.savefig(figdir / "Figure_02_feature_correlation_heatmap.png"); plt.close()

    # -------------------------
    # Figure 03: key color features by status (boxplots)
    # -------------------------
    plt.figure(figsize=(10,5))
    feats = [f for f in KEY_COLOR_FEATURES if f in df_num.columns]
    data0 = [df_num.loc[yc==0, f] for f in feats]
    data1 = [df_num.loc[yc==1, f] for f in feats]
    positions0 = np.arange(len(feats)) - 0.2
    positions1 = np.arange(len(feats)) + 0.2
    bp0 = plt.boxplot(data0, positions=positions0, widths=0.35, patch_artist=True)
    bp1 = plt.boxplot(data1, positions=positions1, widths=0.35, patch_artist=True)
    for b in bp0['boxes']: b.set_alpha(0.5)
    for b in bp1['boxes']: b.set_alpha(0.5)
    plt.xticks(range(len(feats)), feats, rotation=90)
    plt.legend([bp0["boxes"][0], bp1["boxes"][0]], ["Not Anemic (0)", "Anemic (1)"], loc="best")
    plt.ylabel("Value")
    plt.title("Key Color Features by Status")
    plt.tight_layout()
    plt.savefig(figdir / "Figure_03_key_color_features_by_status_boxplots.png"); plt.close()

    # Prepare a consistent holdout split for figures needing a single split
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.30, random_state=metrics.get("random_state", 42), stratify=yc)
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.30, random_state=metrics.get("random_state", 42), stratify=yc)

    # -------------------------
    # Figure 04: ROC (CV predictions)
    # -------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=metrics.get("random_state", 42))
    # Refit a clone on Xc for CV predictions to avoid contaminating the loaded model
    from sklearn.base import clone
    clf_cv = clone(clf)
    y_scores = cross_val_predict(clf_cv, Xc, yc, cv=cv, method="predict_proba", n_jobs=-1)[:,1]
    fpr, tpr, _ = roc_curve(yc, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],"--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Classifier ROC (5-fold CV)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(figdir / "Figure_04_classifier_ROC_CV.png"); plt.close()

    # -------------------------
    # Figure 05: Precision-Recall (CV predictions)
    # -------------------------
    precision, recall, _ = precision_recall_curve(yc, y_scores)
    ap = average_precision_score(yc, y_scores)
    plt.figure()
    plt.plot(recall, precision, lw=2, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Classifier Precision-Recall (5-fold CV)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(figdir / "Figure_05_classifier_PR_CV.png"); plt.close()

    # -------------------------
    # Figure 06: Confusion matrix (holdout)
    # -------------------------
    # Fit loaded model on train split and evaluate on test split
    clf_holdout = clone(clf).fit(Xc_tr, yc_tr)
    y_pred = clf_holdout.predict(Xc_te)
    cm = confusion_matrix(yc_te, y_pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    fig, ax = plt.subplots()
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix (Holdout)")
    plt.tight_layout()
    plt.savefig(figdir / "Figure_06_classifier_confusion_matrix_holdout.png"); plt.close()

    # -------------------------
    # Figure 07: Classifier feature importance
    # -------------------------
    clf_imp = getattr(clf, "feature_importances_", None)
    if clf_imp is None:
        clf_imp = clone(clf).fit(Xc_tr, yc_tr).feature_importances_
    order = np.argsort(clf_imp)[::-1]
    feats_plot = np.array(clf_feats_in_df)[order]
    imps_plot = np.array(clf_imp)[order]
    plt.figure(figsize=(7,5))
    plt.barh(feats_plot[:20][::-1], imps_plot[:20][::-1])
    plt.title("Classifier Feature Importance")
    plt.tight_layout()
    plt.savefig(figdir / "Figure_07_classifier_feature_importance.png"); plt.close()

    # -------------------------
    # Figure 08: Partial dependence (top 4 classifier features)
    # -------------------------
    top4 = feats_plot[:4].tolist()
    fig, ax = plt.subplots(figsize=(8,6))
    clf_pd = clone(clf).fit(Xc_tr, yc_tr)
    PartialDependenceDisplay.from_estimator(clf_pd, Xc_tr, features=top4, ax=ax)
    plt.tight_layout()
    plt.savefig(figdir / "Figure_08_classifier_partial_dependence_top4.png"); plt.close()

    # -------------------------
    # Figure 09: Calibration curve (classifier)
    # -------------------------
    from sklearn.calibration import calibration_curve
    y_proba_hold = clf_holdout.predict_proba(Xc_te)[:,1]
    prob_true, prob_pred = calibration_curve(yc_te, y_proba_hold, n_bins=10, strategy="quantile")
    plt.figure()
    plt.plot([0,1],[0,1],"--", lw=1)
    plt.plot(prob_pred, prob_true, marker="o")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration Curve (Holdout)")
    plt.tight_layout()
    plt.savefig(figdir / "Figure_09_classifier_calibration_curve.png"); plt.close()

    # -------------------------
    # Figure 10: Regression predicted vs actual (holdout)
    # -------------------------
    rgr_holdout = clone(rgr).fit(Xr_tr, yr_tr)
    y_pred_r = rgr_holdout.predict(Xr_te)
    r2 = r2_score(yr_te, y_pred_r)
    mae = mean_absolute_error(yr_te, y_pred_r)
    plt.figure()
    plt.scatter(yr_te, y_pred_r, s=18, alpha=0.8)
    lims = [min(yr_te.min(), y_pred_r.min()), max(yr_te.max(), y_pred_r.max())]
    plt.plot(lims, lims, "--")
    plt.xlabel("Actual Hb (g/dL)"); plt.ylabel("Predicted Hb (g/dL)")
    plt.title(f"Regression: Predicted vs Actual (Holdout)\nR²={r2:.3f}, MAE={mae:.2f}")
    plt.tight_layout()
    plt.savefig(figdir / "Figure_10_regression_pred_vs_actual_holdout.png"); plt.close()

    # -------------------------
    # Figure 11: Regression residuals histogram (holdout)
    # -------------------------
    resid = y_pred_r - yr_te
    plt.figure()
    plt.hist(resid, bins=20, edgecolor="white")
    plt.xlabel("Residual (Pred - Actual) g/dL"); plt.ylabel("Count")
    plt.title("Regression Residuals (Holdout)")
    plt.tight_layout()
    plt.savefig(figdir / "Figure_11_regression_residuals_histogram.png"); plt.close()

    # -------------------------
    # Figure 12: Regression residuals vs fitted
    # -------------------------
    plt.figure()
    plt.scatter(y_pred_r, resid, s=18, alpha=0.8)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Fitted (Predicted Hb) g/dL"); plt.ylabel("Residuals (g/dL)")
    plt.title("Residuals vs Fitted (Holdout)")
    plt.tight_layout()
    plt.savefig(figdir / "Figure_12_regression_residuals_vs_fitted.png"); plt.close()

    # -------------------------
    # Figure 13: Regression feature importance
    # -------------------------
    rgr_imp = getattr(rgr, "feature_importances_", None)
    if rgr_imp is None:
        rgr_imp = clone(rgr).fit(Xr_tr, yr_tr).feature_importances_
    order_r = np.argsort(rgr_imp)[::-1]
    feats_r_plot = np.array(reg_feats_in_df)[order_r]
    imps_r_plot = np.array(rgr_imp)[order_r]
    plt.figure(figsize=(7,5))
    plt.barh(feats_r_plot[:20][::-1], imps_r_plot[:20][::-1])
    plt.title("Regression Feature Importance")
    plt.tight_layout()
    plt.savefig(figdir / "Figure_13_regression_feature_importance.png"); plt.close()

    # -------------------------
    # Figure 14: Vascularity feature distributions
    # -------------------------
    feats_vasc = [f for f in VASC_FEATURES if f in df_num.columns]
    n = len(feats_vasc)
    cols = 3
    rows = int(np.ceil(n/cols)) if n>0 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(10, max(3, rows*2.8)))
    if n == 0:
        axes = np.atleast_1d(axes)
        axes[0].text(0.5,0.5,"No vascularity features present", ha="center")
        axes[0].axis("off")
    else:
        axes = axes.ravel()
        for i, f in enumerate(feats_vasc):
            ax = axes[i]
            ax.hist(df_num[f], bins=20, edgecolor="white")
            ax.set_title(f)
        for j in range(i+1, rows*cols):
            axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(figdir / "Figure_14_vascularity_feature_distributions.png"); plt.close()

    print(f"Saved figures to: {figdir.resolve()}")

if __name__ == "__main__":
    main()
