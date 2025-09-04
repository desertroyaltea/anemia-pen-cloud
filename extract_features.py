#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, io, sys, json, argparse, base64
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import requests
import cv2
from scipy.stats import kurtosis

FEATURE_COLUMNS = [
    "R_norm_p50", "a_mean", "R_p50", "R_p10", "RG", "S_p50",
    "gray_p90", "gray_kurt", "gray_std", "gray_mean",
    "B_mean", "B_p10", "B_p75", "G_kurt",
]

def exif_upright(pil_img: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(pil_img).convert("RGB")

def to_b64_jpeg(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def roboflow_detect(pil_img: Image.Image, api_key: str, model_id: str, conf_0_100: int = 25) -> Optional[Dict[str, Any]]:
    url = f"https://detect.roboflow.com/{model_id}"
    params = {"api_key": api_key, "confidence": str(conf_0_100), "format": "json"}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    b64 = to_b64_jpeg(pil_img)
    r = requests.post(url, params=params, data=b64, headers=headers, timeout=60)
    r.raise_for_status()
    js = r.json()
    preds = js.get("predictions", [])
    if not preds:
        return None
    # prefer 'conjunctiva' or 'palpebral'
    preferred = [p for p in preds if p.get("class", "").lower() in ("conjunctiva", "palpebral")]
    best = max(preferred or preds, key=lambda p: p.get("confidence", 0.0))
    return best

def crop_from_box(pil_img: Image.Image, box: Dict[str, Any]) -> Image.Image:
    x = float(box["x"]); y = float(box["y"])
    w = float(box["width"]); h = float(box["height"])
    L = int(round(x - w/2)); T = int(round(y - h/2))
    R = int(round(x + w/2)); B = int(round(y + h/2))
    L = max(0, L); T = max(0, T)
    R = min(pil_img.width, R); B = min(pil_img.height, B)
    if R <= L or B <= T:  # degenerate
        return pil_img.copy()
    return pil_img.crop((L, T, R, B))

def compute_features_from_pil(pil_img: Image.Image) -> Dict[str, float]:
    # Exactly the same recipe the app uses
    rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    R = rgb[..., 0].astype(np.float32)
    G = rgb[..., 1].astype(np.float32)
    B = rgb[..., 2].astype(np.float32)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hsv  = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    S    = hsv[..., 1].astype(np.float32) / 255.0  # 0..1
    lab  = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
    a    = lab[..., 1].astype(np.float32) - 128.0  # center a*

    denom = R + G + B + 1e-6
    R_norm = R / denom

    feats = {
        "R_norm_p50": float(np.percentile(R_norm, 50)),
        "a_mean":     float(np.mean(a)),
        "R_p50":      float(np.percentile(R, 50)),
        "R_p10":      float(np.percentile(R, 10)),
        "RG":         float(np.mean(R) / (np.mean(G) + 1e-6)),
        "S_p50":      float(np.percentile(S, 50)),
        "gray_p90":   float(np.percentile(gray, 90)),
        "gray_kurt":  float(kurtosis(gray.ravel(), fisher=False, bias=False, nan_policy="omit")),
        "gray_std":   float(np.std(gray, ddof=0)),
        "gray_mean":  float(np.mean(gray)),
        "B_mean":     float(np.mean(B)),
        "B_p10":      float(np.percentile(B, 10)),
        "B_p75":      float(np.percentile(B, 75)),
        "G_kurt":     float(kurtosis(G.ravel(), fisher=False, bias=False, nan_policy="omit")),
    }
    return feats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--labels_csv", required=True, help="CSV with columns: filename,hb[,SubjectID]")
    ap.add_argument("--out_csv", default="features.csv")
    ap.add_argument("--save_crops_dir", default=None, help="Optional dir to save cropped conjunctiva")
    ap.add_argument("--roboflow_model", default=os.getenv("ROBOFLOW_MODEL", "eye-conjunctiva-detector/2"))
    ap.add_argument("--confidence", type=int, default=25)
    ap.add_argument("--no_roboflow", action="store_true", help="Skip detection; use full image")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    labels = pd.read_csv(args.labels_csv)
    labels["filename"] = labels["filename"].astype(str)

    rf_key = os.getenv("ROBOFLOW_API_KEY", "")
    use_rf = (not args.no_roboflow) and bool(rf_key)

    out_rows = []
    crops_dir = Path(args.save_crops_dir) if args.save_crops_dir else None
    if crops_dir:
        crops_dir.mkdir(parents=True, exist_ok=True)

    for i, row in labels.iterrows():
        fname = row["filename"]
        hb    = float(row["hb"])
        sid   = row["SubjectID"] if "SubjectID" in row else None
        path  = images_dir / fname
        if not path.exists():
            print(f"[WARN] Missing image: {path}")
            continue

        try:
            pil_full = exif_upright(Image.open(path))
            if use_rf:
                try:
                    box = roboflow_detect(pil_full, rf_key, args.roboflow_model, args.confidence)
                    crop = crop_from_box(pil_full, box) if box else pil_full
                except Exception as e:
                    print(f"[WARN] Roboflow failed for {fname}: {e}; using full image")
                    crop = pil_full
            else:
                crop = pil_full

            feats = compute_features_from_pil(crop)
            rec = {"filename": fname, "hb": hb}
            if sid is not None and not (pd.isna(sid)):
                rec["SubjectID"] = sid
            rec.update(feats)
            out_rows.append(rec)

            if crops_dir:
                crop.save(crops_dir / fname)

        except Exception as e:
            print(f"[ERROR] {fname}: {e}")

    df = pd.DataFrame(out_rows, columns=(["filename","hb","SubjectID"] if "SubjectID" in labels.columns else ["filename","hb"]) + FEATURE_COLUMNS)
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] Wrote {args.out_csv} with {len(df)} rows.")

if __name__ == "__main__":
    main()
