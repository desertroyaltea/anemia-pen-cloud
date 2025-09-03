#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 2: Extract features from CROPPED conjunctiva images with gray-world white balance.
- Reads labels.xlsx (Sheet1 by default) with columns: image_path, hb
- Matches rows to files in --images directory
- Applies gray-world WB to each crop, then computes requested features
- Writes TWO files:
    outputs/features_wb_all.csv       (all 14 features + hb)
    outputs/features_wb_compact.csv   (a compact subset useful for SPSS)

Usage example:
  python step2_extract_features_wb.py --images cropped_images --labels labels.xlsx \
    --sheet Sheet1 --image-col image_path --hb-col hb --out outputs

Requirements:
  numpy, pandas, scipy, opencv-python-headless (or opencv-python)
"""

import argparse
import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy.stats import kurtosis

# ---------- Features we will compute (14 total) ----------
ALL_FEATURES = [
    "R_norm_p50",          # median of R/(R+G)
    "a_mean",              # Lab a* mean
    "R_p50", "R_p10",      # red channel median & 10th pct
    "RG",                  # mean(R)/mean(G)
    "S_p50",               # median saturation
    "gray_p90", "gray_kurt", "gray_std", "gray_mean",  # grayscale stats
    "B_mean", "B_p10", "B_p75",                        # blue channel
    "G_kurt"              # green channel kurtosis
]

# Compact subset we’ll export separately so you can use a smaller, stabler set in SPSS.
COMPACT_FEATURES = [
    # Pick ONE of {RG, a_mean} in SPSS (keep both here so you can choose)
    "RG", "a_mean",
    "B_p10",
    "gray_std",
    # Optional helpers:
    "gray_p90",
    "R_p50"
]

def parse_args():
    ap = argparse.ArgumentParser(description="Extract WB-normalized features from cropped images.")
    ap.add_argument("--images", type=str, required=True, help="Folder with CROPPED images (from Step 1).")
    ap.add_argument("--labels", type=str, default="labels.xlsx", help="Path to labels Excel/CSV (image_path,hb).")
    ap.add_argument("--sheet", type=str, default="Sheet1", help="Sheet name if Excel (ignored for CSV).")
    ap.add_argument("--image-col", type=str, default="image_path", help="Column name for image filename.")
    ap.add_argument("--hb-col", type=str, default="hb", help="Column name for hemoglobin.")
    ap.add_argument("--out", type=str, default="outputs", help="Output folder.")
    return ap.parse_args()

# ---------------- Utility: robust Hb parsing ----------------
def parse_hb(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    # tolerate comma decimals
    s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return np.nan

# ---------------- Gray-world white balance ----------------
def gray_world_wb(bgr):
    """
    Scale each channel so that its mean equals the global mean of means.
    Works on uint8 BGR, returns uint8 BGR.
    """
    if bgr.dtype != np.uint8:
        bgr = bgr.astype(np.uint8, copy=False)

    img = bgr.astype(np.float32)
    # Compute per-channel means
    mB, mG, mR = [img[..., c].mean() + 1e-6 for c in range(3)]
    target = (mB + mG + mR) / 3.0
    sB, sG, sR = target / mB, target / mG, target / mR
    img[..., 0] = np.clip(img[..., 0] * sB, 0, 255)
    img[..., 1] = np.clip(img[..., 1] * sG, 0, 255)
    img[..., 2] = np.clip(img[..., 2] * sR, 0, 255)
    return img.astype(np.uint8)

# ---------------- Feature computation ----------------
def pct(x, q):
    return float(np.percentile(x, q))

def safe_kurt(x):
    # Excess kurtosis (Fisher), unbiased
    return float(kurtosis(x, fisher=True, bias=False))

def compute_features(bgr):
    """
    Compute the 14 requested features AFTER gray-world WB.
    Returns dict {feature_name: value}
    """
    # WB first
    bgr_wb = gray_world_wb(bgr)

    # Split channels
    B = bgr_wb[..., 0].astype(np.float32)
    G = bgr_wb[..., 1].astype(np.float32)
    R = bgr_wb[..., 2].astype(np.float32)

    # Grayscale & other spaces
    gray = cv2.cvtColor(bgr_wb, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(bgr_wb, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(bgr_wb, cv2.COLOR_BGR2Lab).astype(np.float32)

    H = hsv[..., 0]
    S = hsv[..., 1]
    V = hsv[..., 2]
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]

    # 1) R_norm_p50 = median of R/(R+G)
    R_norm = R / (R + G + 1e-6)
    R_norm_p50 = pct(R_norm, 50)

    # 2) a_mean (Lab a*)
    a_mean = float(a.mean())

    # 3) R_p50, R_p10
    R_p50 = pct(R, 50)
    R_p10 = pct(R, 10)

    # 4) RG = mean(R)/mean(G)
    RG = float(R.mean() / (G.mean() + 1e-6))

    # 5) S_p50
    S_p50 = pct(S, 50)

    # 6) gray stats
    gray_p90 = pct(gray, 90)
    gray_kurt = safe_kurt(gray.ravel())
    gray_std = float(gray.std(ddof=1))
    gray_mean = float(gray.mean())

    # 7) B stats
    B_mean = float(B.mean())
    B_p10 = pct(B, 10)
    B_p75 = pct(B, 75)

    # 8) G_kurt
    G_kurt = safe_kurt(G.ravel())

    feats = {
        "R_norm_p50": R_norm_p50,
        "a_mean": a_mean,
        "R_p50": R_p50,
        "R_p10": R_p10,
        "RG": RG,
        "S_p50": S_p50,
        "gray_p90": gray_p90,
        "gray_kurt": gray_kurt,
        "gray_std": gray_std,
        "gray_mean": gray_mean,
        "B_mean": B_mean,
        "B_p10": B_p10,
        "B_p75": B_p75,
        "G_kurt": G_kurt,
    }
    return feats

# ---------------- Main ----------------
def main():
    args = parse_args()
    img_dir = Path(args.images)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load labels (Excel or CSV)
    labels_path = Path(args.labels)
    if labels_path.suffix.lower() in [".xlsx", ".xls"]:
        df_lab = pd.read_excel(labels_path, sheet_name=args.sheet)
    else:
        df_lab = pd.read_csv(labels_path)

    if args.image_col not in df_lab.columns or args.hb_col not in df_lab.columns:
        raise ValueError(f"Labels file must contain columns '{args.image_col}' and '{args.hb_col}'")

    # Normalize filename and parse Hb
    df_lab = df_lab[[args.image_col, args.hb_col]].copy()
    df_lab[args.image_col] = df_lab[args.image_col].astype(str).apply(lambda p: os.path.basename(p.strip()))
    df_lab[args.hb_col] = df_lab[args.hb_col].apply(parse_hb)

    # Build quick index of available cropped images by basename and by stem
    files = list(img_dir.glob("*"))
    name_to_path = {f.name: f for f in files}
    stem_to_path = {}
    for f in files:
        stem_to_path[f.stem] = f
        # also map common crop suffixes
        if f.stem.endswith("_crop"):
            stem_to_path[f.stem[:-5]] = f  # map base -> *_crop

    rows = []
    missing = 0
    unreadable = 0
    non_numeric = 0

    for _, row in df_lab.iterrows():
        name = row[args.image_col]
        hb = row[args.hb_col]

        if pd.isna(hb):
            non_numeric += 1
            continue

        # Try direct match, then stem match, then stem + "_crop"
        path = name_to_path.get(name)
        if path is None:
            base = Path(name).stem
            path = stem_to_path.get(base, None)
            if path is None:
                alt = f"{base}_crop"
                path = stem_to_path.get(alt, None)

        if path is None or not path.exists():
            missing += 1
            continue

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            unreadable += 1
            continue

        feats = compute_features(bgr)
        out_row = {
            "image": path.name,
            "hb": float(hb),
        }
        out_row.update({k: feats[k] for k in ALL_FEATURES})
        rows.append(out_row)

    if len(rows) == 0:
        raise RuntimeError("No rows processed. Check image folder and label names.")

    df_all = pd.DataFrame(rows)

    # Save ALL features
    all_csv = out_dir / "features_wb_all.csv"
    df_all.to_csv(all_csv, index=False)

    # Try Excel (optional)
    all_xlsx = out_dir / "features_wb_all.xlsx"
    try:
        df_all.to_excel(all_xlsx, index=False)
    except Exception as e:
        print(f"Note: Could not write Excel ({e}). CSV is available at {all_csv}")

    # Save COMPACT subset (includes both RG and a_mean so you can choose one in SPSS)
    keep_cols = ["image", "hb"] + [c for c in COMPACT_FEATURES if c in df_all.columns]
    df_compact = df_all[keep_cols].copy()
    cmp_csv = out_dir / "features_wb_compact.csv"
    df_compact.to_csv(cmp_csv, index=False)

    cmp_xlsx = out_dir / "features_wb_compact.xlsx"
    try:
        df_compact.to_excel(cmp_xlsx, index=False)
    except Exception as e:
        print(f"Note: Could not write Excel ({e}). CSV is available at {cmp_csv}")

    print(f"\nProcessed {len(rows)} rows.")
    print(f"Missing image files: {missing}")
    print(f"Unreadable image files: {unreadable}")
    print(f"Non-numeric Hb rows: {non_numeric}")
    print(f"Wrote: {all_csv}")
    print(f"Wrote: {cmp_csv}")

if __name__ == "__main__":
    main()
