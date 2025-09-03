#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 2: Extract 14 features from cropped conjunctiva images and merge with true Hb.
Inputs:
  - Cropped images: ./cropped_images/*.jpg|*.png|...
  - Labels Excel:   ./labels.xlsx  (Sheet1) with columns: image_path, hb
Outputs:
  - ./outputs/features.csv
  - ./outputs/features.xlsx
  - ./outputs/step2_missing.csv     (images missing label or unreadable)

Columns in features files:
  A: filename
  B: hb  (true Hb from labels)
  C..: features (exactly 14 below)

Features (14):
  1)  R_norm_p50    (median of per-pixel R / (R+G+B))
  2)  a_mean        (CIELAB a* mean, OpenCV Lab -> subtract 128)
  3)  R_p50         (median of R)
  4)  R_p10         (10th percentile of R)
  5)  RG            (mean(R) / mean(G), epsilon-safe)
  6)  S_p50         (median saturation; HSV S / 255 -> 0..1)
  7)  gray_p90      (90th percentile of grayscale)
  8)  gray_kurt     (kurtosis of grayscale; Pearson (fisher=False), bias=False)
  9)  gray_std      (std of grayscale)
 10)  gray_mean     (mean of grayscale)
 11)  B_mean        (mean of B)
 12)  B_p10         (10th percentile of B)
 13)  B_p75         (75th percentile of B)
 14)  G_kurt        (kurtosis of G; Pearson, bias=False)
"""

import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import cv2
from scipy.stats import kurtosis

# --------- Defaults (edit paths if needed) --------- #
CROPS_DIR = Path("cropped_images")
LABELS_XLSX = Path("labels.xlsx")
LABELS_SHEET = "Sheet1"
LABEL_IMG_COL = "image_path"
LABEL_HB_COL = "hb"

OUT_DIR = Path("outputs")
FEATURES_CSV = OUT_DIR / "features.csv"
FEATURES_XLSX = OUT_DIR / "features.xlsx"
MISSING_CSV = OUT_DIR / "step2_missing.csv"

# Feature list & output order (exactly 14)
FEATURE_COLUMNS = [
    "R_norm_p50",
    "a_mean",
    "R_p50",
    "R_p10",
    "RG",
    "S_p50",
    "gray_p90",
    "gray_kurt",
    "gray_std",
    "gray_mean",
    "B_mean",
    "B_p10",
    "B_p75",
    "G_kurt",
]
# --------------------------------------------------- #


def list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    files = [p for p in folder.rglob("*") if p.suffix.lower() in exts]
    files.sort()
    return files


def load_image_rgb(path: Path) -> np.ndarray:
    """Read image as RGB float32 in [0,255]. Raises on failure."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cv2.imread returned None (unreadable image)")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    return rgb


def compute_features(rgb: np.ndarray) -> Dict[str, float]:
    """
    Compute the 14 required features on the cropped RGB image.
    Channels in 0..255 float32.
    """
    # Split channels
    R = rgb[..., 0]
    G = rgb[..., 1]
    B = rgb[..., 2]

    # Grayscale (OpenCV luminance-like)
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

    # HSV for saturation (S in 0..255 -> normalize to 0..1 for S_p50)
    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    S = hsv[..., 1].astype(np.float32) / 255.0

    # Lab for a* (OpenCV Lab is scaled: L[0..100]->[0..255], a*[-128..127]->[0..255] with +128 offset)
    lab = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2Lab)
    a = lab[..., 1].astype(np.float32) - 128.0  # back to a* approx

    # 1) R_norm_p50: median of (R / (R+G+B))
    denom = R + G + B + 1e-6
    R_norm = R / denom
    R_norm_p50 = float(np.percentile(R_norm, 50))

    # 2) a_mean
    a_mean = float(np.mean(a))

    # 3) R_p50
    R_p50 = float(np.percentile(R, 50))

    # 4) R_p10
    R_p10 = float(np.percentile(R, 10))

    # 5) RG (mean R / mean G)
    RG = float((np.mean(R)) / (np.mean(G) + 1e-6))

    # 6) S_p50
    S_p50 = float(np.percentile(S, 50))

    # 7) gray_p90
    gray_p90 = float(np.percentile(gray, 90))

    # 8) gray_kurt (Pearson, i.e., normal=3)
    gray_kurt = float(kurtosis(gray.ravel(), fisher=False, bias=False, nan_policy="omit"))

    # 9) gray_std
    gray_std = float(np.std(gray, ddof=0))

    # 10) gray_mean
    gray_mean = float(np.mean(gray))

    # 11) B_mean
    B_mean = float(np.mean(B))

    # 12) B_p10
    B_p10 = float(np.percentile(B, 10))

    # 13) B_p75
    B_p75 = float(np.percentile(B, 75))

    # 14) G_kurt (Pearson)
    G_kurt = float(kurtosis(G.ravel(), fisher=False, bias=False, nan_policy="omit"))

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


def load_labels(xlsx_path: Path, sheet: str, img_col: str, hb_col: str) -> Dict[str, float]:
    """
    Read labels.xlsx and build {filename -> hb}.
    - image_path can be 'Image_003.png' or a path; we normalize to basename.
    - hb must be decimal with '.' (user confirmed).
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    if img_col not in df.columns or hb_col not in df.columns:
        raise ValueError(f"Expected columns '{img_col}' and '{hb_col}' in {xlsx_path} sheet '{sheet}'")

    # Normalize filename keys
    def to_name(x: Any) -> str:
        try:
            return Path(str(x)).name
        except Exception:
            return str(x)

    df = df.copy()
    df[img_col] = df[img_col].map(to_name)

    # Coerce hb to numeric (dot decimals per user)
    df[hb_col] = pd.to_numeric(df[hb_col], errors="coerce")
    df = df.dropna(subset=[img_col, hb_col])

    # Deduplicate: keep the first occurrence if duplicates exist
    df = df.drop_duplicates(subset=[img_col], keep="first")

    mapping = dict(zip(df[img_col].astype(str), df[hb_col].astype(float)))
    return mapping


def ensure_outdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Step 2: Extract 14 features and merge with Hb labels.")
    parser.add_argument("--crops-dir", type=str, default=str(CROPS_DIR), help="Folder with cropped images from Step 1")
    parser.add_argument("--labels-xlsx", type=str, default=str(LABELS_XLSX), help="Path to labels.xlsx")
    parser.add_argument("--labels-sheet", type=str, default=LABELS_SHEET, help="Sheet name in labels.xlsx")
    parser.add_argument("--labels-img-col", type=str, default=LABEL_IMG_COL, help="Image filename/path column name")
    parser.add_argument("--labels-hb-col", type=str, default=LABEL_HB_COL, help="Hb column name")
    args = parser.parse_args()

    crops_dir = Path(args.crops_dir)
    labels_xlsx = Path(args.labels_xlsx)
    labels_sheet = args.labels_sheet
    img_col = args.labels_img_col
    hb_col = args.labels_hb_col

    ensure_outdir(OUT_DIR)

    if not crops_dir.exists():
        print(f"ERROR: Cropped images folder not found: {crops_dir}")
        return

    if not labels_xlsx.exists():
        print(f"ERROR: Labels Excel not found: {labels_xlsx}")
        return

    # Load labels mapping
    labels_map = load_labels(labels_xlsx, labels_sheet, img_col, hb_col)

    # Iterate crops
    images = list_images(crops_dir)
    if not images:
        print(f"No images found in {crops_dir}")
        return

    rows: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []

    print(f"Found {len(images)} cropped images. Extracting features...")

    for idx, p in enumerate(images, 1):
        fname = p.name
        hb = labels_map.get(fname, None)
        if hb is None:
            missing.append({"filename": fname, "issue": "no_label"})
            print(f"[{idx}/{len(images)}] SKIP (no label): {fname}")
            continue

        try:
            rgb = load_image_rgb(p)
            feats = compute_features(rgb)
            row = {"filename": fname, "hb": float(hb)}
            # maintain specified order
            for col in FEATURE_COLUMNS:
                row[col] = feats[col]
            rows.append(row)
            if idx % 25 == 0 or idx == len(images):
                print(f"[{idx}/{len(images)}] processed ...")
        except Exception as e:
            missing.append({"filename": fname, "issue": f"unreadable_or_feature_error: {e}"})
            print(f"[{idx}/{len(images)}] ERROR {fname} -> {e}")

    # Build DataFrame in the requested column layout
    if not rows:
        print("No rows extracted (check labels match filenames).")
        # Still write empty shells for consistency
        pd.DataFrame(columns=["filename", "hb"] + FEATURE_COLUMNS).to_csv(FEATURES_CSV, index=False)
        pd.DataFrame(columns=["filename", "hb"] + FEATURE_COLUMNS).to_excel(FEATURES_XLSX, index=False)
    else:
        df = pd.DataFrame(rows, columns=["filename", "hb"] + FEATURE_COLUMNS)
        # Save
        df.to_csv(FEATURES_CSV, index=False)
        try:
            df.to_excel(FEATURES_XLSX, index=False)
        except Exception as e:
            print(f"WARNING: Could not write Excel file: {e}")
        print(f"\nExtracted features for {len(df)} images.")
        print(f"Wrote: {FEATURES_CSV}")
        if FEATURES_XLSX.exists():
            print(f"Wrote: {FEATURES_XLSX}")

    # Write missing report
    dfm = pd.DataFrame(missing)
    dfm.to_csv(MISSING_CSV, index=False)
    if len(dfm) > 0:
        print(f"Some images skipped. See: {MISSING_CSV}")
    else:
        print("All processed images had labels.")

if __name__ == "__main__":
    main()
