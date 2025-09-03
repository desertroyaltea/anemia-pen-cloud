#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 4: Predict Hb on NEW full-eye images using:
  - Roboflow detection (requests-only) to crop conjunctiva (with EXIF rotation fix),
  - The same 14 handcrafted features (as in Step 2),
  - An SPSS-exported PMML model (e.g., hemo.xml) scored via PyPMML.

Inputs (choose one):
  --image <path-to-one-image>
  --images-dir <folder-with-images>    (batch mode)

Other key args:
  --pmml hemo.xml                      (SPSS PMML file you exported)
  --api-key <roboflow_api_key>         (or set ROBOFLOW_API_KEY env var)
  --model-id eye-conjunctiva-detector/2
  --confidence 25                      (0..100, default ~0.25)
  --save-crops                         (store cropped conjunctiva under outputs/infer_crops)
  --rmse 1.7                           (optional: report pred ± 1.96*RMSE)

Outputs:
  - CSV: outputs/step4_predictions.csv
  - (optional) crops: outputs/infer_crops/<filename>

Notes:
  * If no detection is found, the image is skipped and logged.
  * This script does not need labels.xlsx.
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import cv2
from PIL import Image, ImageOps
from scipy.stats import kurtosis
from pypmml import Model

# -------- Defaults -------- #
DEFAULT_MODEL_ID = "eye-conjunctiva-detector/2"
DEFAULT_CONF_0_100 = 25                 # 0..100
DEFAULT_CLASS_NAME = "conjunctiva"
OUT_DIR = Path("outputs")
PRED_CSV = OUT_DIR / "step4_predictions.csv"
CROPS_DIR = OUT_DIR / "infer_crops"
NO_DET_CSV = OUT_DIR / "step4_no_detections.csv"

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
# -------------------------- #


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    files = [p for p in folder.rglob("*") if p.suffix.lower() in exts]
    files.sort()
    return files


def pil_exif_upright(path: Path) -> Image.Image:
    """Open image and physically apply EXIF Orientation (so pixels are upright)."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def pil_to_jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def roboflow_detect_b64(
    b64_str: str,
    model_id: str,
    api_key: str,
    confidence_0_100: int = DEFAULT_CONF_0_100,
    timeout: int = 60,
) -> Dict[str, Any]:
    """
    Hosted API (Serverless) endpoint using base64 body:
      POST https://detect.roboflow.com/<model_id>?api_key=...&confidence=...
    """
    url = f"https://detect.roboflow.com/{model_id}"
    params = {"api_key": api_key, "confidence": str(confidence_0_100), "format": "json"}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    resp = requests.post(url, params=params, data=b64_str, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def select_best_box(preds: List[Dict[str, Any]], target_class: str) -> Optional[Dict[str, Any]]:
    if not preds:
        return None
    target = [p for p in preds if str(p.get("class", "")).lower() == target_class.lower()]
    if target:
        return max(target, key=lambda p: p.get("confidence", 0.0))
    return max(preds, key=lambda p: p.get("confidence", 0.0))


def crop_from_box(pil_img: Image.Image, box: Dict[str, Any]) -> Image.Image:
    """
    Roboflow box: center (x,y), width, height in pixels; coords align with this upright PIL image.
    """
    x = float(box["x"]); y = float(box["y"])
    w = float(box["width"]); h = float(box["height"])
    left = int(round(x - w / 2.0))
    top = int(round(y - h / 2.0))
    right = int(round(x + w / 2.0))
    bottom = int(round(y + h / 2.0))

    left = max(0, left); top = max(0, top)
    right = min(pil_img.width, right); bottom = min(pil_img.height, bottom)
    if right <= left or bottom <= top:
        return pil_img.copy()
    return pil_img.crop((left, top, right, bottom))


def compute_features_from_pil(pil_img: Image.Image) -> Dict[str, float]:
    """Compute the 14 features (same definitions as Step 2) from a PIL RGB image."""
    rgb = np.array(pil_img, dtype=np.uint8)  # HxWx3 RGB uint8

    # Split channels float32
    R = rgb[..., 0].astype(np.float32)
    G = rgb[..., 1].astype(np.float32)
    B = rgb[..., 2].astype(np.float32)

    # Grayscale via OpenCV
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # HSV Saturation (0..1)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    S = hsv[..., 1].astype(np.float32) / 255.0

    # Lab a* (subtract 128)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
    a = lab[..., 1].astype(np.float32) - 128.0

    # 1) R_norm_p50
    denom = R + G + B + 1e-6
    R_norm = R / denom
    R_norm_p50 = float(np.percentile(R_norm, 50))

    # 2) a_mean
    a_mean = float(np.mean(a))

    # 3) R_p50
    R_p50 = float(np.percentile(R, 50))

    # 4) R_p10
    R_p10 = float(np.percentile(R, 10))

    # 5) RG
    RG = float((np.mean(R)) / (np.mean(G) + 1e-6))

    # 6) S_p50
    S_p50 = float(np.percentile(S, 50))

    # 7) gray_p90
    gray_p90 = float(np.percentile(gray, 90))

    # 8) gray_kurt (Pearson)
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

    return {
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


def predict_one_with_pmml(
    pil_full: Image.Image,
    model: Model,
    api_key: str,
    model_id: str,
    conf_0_100: int,
    class_name: str,
    save_crop_path: Optional[Path] = None,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Returns (hb_pred or None, meta dict).
    meta includes detection info and errors if any.
    """
    # Encode upright full image to base64 for Roboflow
    try:
        jpg_bytes = pil_to_jpeg_bytes(pil_full, quality=90)
        b64 = base64.b64encode(jpg_bytes).decode("utf-8")
    except Exception as e:
        return None, {"error": f"encode_base64_failed: {e}"}

    # Detect
    try:
        resp = roboflow_detect_b64(b64, model_id=model_id, api_key=api_key, confidence_0_100=conf_0_100)
        preds = resp.get("predictions", [])
        best = select_best_box(preds, target_class=class_name)
        if best is None:
            return None, {"error": "no_detection"}
    except Exception as e:
        return None, {"error": f"roboflow_error: {e}"}

    # Crop
    try:
        crop = crop_from_box(pil_full, best)
        if save_crop_path is not None:
            crop.save(save_crop_path)
    except Exception as e:
        return None, {"error": f"crop_failed: {e}"}

    # Features
    try:
        feats = compute_features_from_pil(crop)
    except Exception as e:
        return None, {"error": f"feature_error: {e}"}

    # Score via PMML
    try:
        df = pd.DataFrame([{k: float(feats[k]) for k in FEATURE_COLUMNS}])
        scored = model.predict(df)
        # PyPMML returns a DataFrame; for regression, predicted value is commonly in
        # a column named like 'predicted_hb' or just the target 'hb'. We'll try both.
        if "predicted_hb" in scored.columns:
            pred = float(scored.loc[0, "predicted_hb"])
        elif "hb" in scored.columns:
            pred = float(scored.loc[0, "hb"])
        else:
            # some exports use 'prediction' or similar
            # try the last numeric column as fallback
            num_cols = [c for c in scored.columns if np.issubdtype(scored[c].dtype, np.number)]
            if not num_cols:
                return None, {"error": f"unexpected_scored_columns: {list(scored.columns)}"}
            pred = float(scored.loc[0, num_cols[-1]])
    except Exception as e:
        return None, {"error": f"pmml_predict_error: {e}"}

    meta = {
        "class": best.get("class", ""),
        "confidence": float(best.get("confidence", 0.0)),
        "x": float(best.get("x", 0.0)),
        "y": float(best.get("y", 0.0)),
        "width": float(best.get("width", 0.0)),
        "height": float(best.get("height", 0.0)),
    }
    return pred, meta


def main():
    parser = argparse.ArgumentParser(description="Step 4: Predict Hb from a full-eye image using Roboflow + SPSS PMML.")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", type=str, help="Path to one full-eye image")
    g.add_argument("--images-dir", type=str, help="Folder of images to batch predict")

    parser.add_argument("--pmml", type=str, default="hemo.xml", help="Path to SPSS PMML file")
    parser.add_argument("--api-key", type=str, default=os.getenv("ROBOFLOW_API_KEY", ""), help="Roboflow API key")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help="Roboflow model id")
    parser.add_argument("--confidence", type=int, default=DEFAULT_CONF_0_100, help="Threshold 0..100 (default 25≈0.25)")
    parser.add_argument("--class-name", type=str, default=DEFAULT_CLASS_NAME, help="Preferred class name to crop")
    parser.add_argument("--save-crops", action="store_true", help=f"Save crops under {CROPS_DIR}")
    parser.add_argument("--rmse", type=float, default=None, help="Optional RMSE to report ±1.96*RMSE interval")

    args = parser.parse_args()

    # Checks
    pmml_path = Path(args.pmml)
    if not pmml_path.exists():
        print(f"ERROR: PMML file not found: {pmml_path}", file=sys.stderr)
        sys.exit(2)

    api_key = args.api_key.strip()
    if not api_key:
        print("ERROR: Missing Roboflow API key. Pass --api-key or set ROBOFLOW_API_KEY.", file=sys.stderr)
        sys.exit(2)

    # Load model
    try:
        model = Model.load(str(pmml_path))
    except Exception as e:
        print(f"ERROR: Failed to load PMML model: {e}", file=sys.stderr)
        sys.exit(2)

    ensure_dirs(OUT_DIR)
    if args.save_crops:
        ensure_dirs(CROPS_DIR)

    # Collect images
    if args.image:
        images = [Path(args.image)]
    else:
        images = list_images(Path(args.images_dir))

    if not images:
        print("No images found to score.", file=sys.stderr)
        sys.exit(0)

    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for i, img_path in enumerate(images, 1):
        try:
            pil_full = pil_exif_upright(img_path)
        except Exception as e:
            skipped.append({"filename": img_path.name, "error": f"open_or_exif_failed: {e}"})
            print(f"[{i}/{len(images)}] ERROR opening -> {img_path.name}: {e}")
            continue

        crop_save = (CROPS_DIR / img_path.name) if args.save_crops else None
        pred, meta = predict_one_with_pmml(
            pil_full=pil_full,
            model=model,
            api_key=api_key,
            model_id=args.model_id,
            conf_0_100=args.confidence,
            class_name=args.class_name,
            save_crop_path=crop_save,
        )

        if pred is None:
            skipped.append({"filename": img_path.name, **meta})
            print(f"[{i}/{len(images)}] SKIP {img_path.name} -> {meta.get('error')}")
            continue

        # Optional interval: pred ± 1.96*RMSE
        lo = hi = ""
        if args.rmse is not None and args.rmse > 0:
            half = 1.96 * args.rmse
            lo = float(pred - half)
            hi = float(pred + half)

        row = {
            "filename": img_path.name,
            "hb_pred": float(pred),
            "hb_lower": lo,
            "hb_upper": hi,
            "det_class": meta.get("class", ""),
            "det_conf": meta.get("confidence", ""),
            "det_x": meta.get("x", ""),
            "det_y": meta.get("y", ""),
            "det_w": meta.get("width", ""),
            "det_h": meta.get("height", ""),
        }
        rows.append(row)
        if args.rmse is not None and args.rmse > 0:
            print(f"[{i}/{len(images)}] {img_path.name}: Hb={pred:.2f}  (±{1.96*args.rmse:.2f})")
        else:
            print(f"[{i}/{len(images)}] {img_path.name}: Hb={pred:.2f}")

        time.sleep(0.1)

    # Write outputs
    df = pd.DataFrame(rows, columns=[
        "filename","hb_pred","hb_lower","hb_upper",
        "det_class","det_conf","det_x","det_y","det_w","det_h"
    ])
    df.to_csv(PRED_CSV, index=False)

    pd.DataFrame(skipped).to_csv(NO_DET_CSV, index=False)

    print("\nDone.")
    print(f"- Predictions: {PRED_CSV.resolve()}")
    print(f"- Skipped/no-detections: {NO_DET_CSV.resolve()}")
    if args.save_crops:
        print(f"- Crops saved to: {CROPS_DIR.resolve()}")


if __name__ == "__main__":
    main()
