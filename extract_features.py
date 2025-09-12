#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extract_features.py
-------------------
Pipeline:
1) Read FULL-eye images from ./Images
2) Use Roboflow detect API to crop conjunctiva -> ./images_cropped
3) Compute features per cropped image:
   - 14 baseline features (color+gray stats)
   - Vascularity features (Frangi vesselness + skeletonization)
4) Join with labels.csv (filename,hb,status)
5) Save features CSV: features_dataset.csv

USAGE:
python extract_features.py --api-key <ROBOFLOW_API_KEY> \
  --model-id eye-conjunctiva-detector/2 \
  --class-name conjunctiva \
  --images-dir Images --out-crops images_cropped \
  --labels labels.csv --out-csv features_dataset.csv

pip install numpy pandas pillow requests opencv-python scikit-image scipy
"""

import io, os, time, base64, argparse
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageOps
import cv2
from scipy.stats import kurtosis
from scipy.ndimage import convolve
from skimage import exposure, filters, morphology, measure
from skimage.morphology import skeletonize

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", required=True)
    p.add_argument("--model-id", default="eye-conjunctiva-detector/2")
    p.add_argument("--class-name", default="conjunctiva")
    p.add_argument("--conf", type=int, default=25)
    p.add_argument("--images-dir", default="Images")
    p.add_argument("--out-crops", default="images_cropped")
    p.add_argument("--labels", default="labels.csv")
    p.add_argument("--out-csv", default="features_dataset.csv")
    p.add_argument("--sleep", type=float, default=0.2)
    p.add_argument("--black-ridges", type=lambda s: s.lower() != "false", default=True)
    p.add_argument("--min-size", type=int, default=50)
    p.add_argument("--area-threshold", type=int, default=50)
    return p.parse_args()

def exif_upright(pil_img: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(pil_img).convert("RGB")

def pil_to_jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()

def to_b64_jpeg(img: Image.Image) -> str:
    return base64.b64encode(pil_to_jpeg_bytes(img, quality=90)).decode("utf-8")

def roboflow_detect_b64(b64_str: str, model_id: str, api_key: str, conf_0_100: int) -> Dict[str, Any]:
    url = f"https://detect.roboflow.com/{model_id}"
    params = {"api_key": api_key, "confidence": str(conf_0_100), "format": "json"}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    resp = requests.post(url, params=params, data=b64_str, headers=headers, timeout=60)
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
    x = float(box["x"]); y = float(box["y"])
    w = float(box["width"]); h = float(box["height"])
    left = int(round(x - w / 2.0)); top = int(round(y - h / 2.0))
    right = int(round(x + w / 2.0)); bottom = int(round(y + h / 2.0))
    left = max(0, left); top = max(0, top)
    right = min(pil_img.width, right); bottom = min(pil_img.height, bottom)
    if right <= left or bottom <= top:
        return pil_img.copy()
    return pil_img.crop((left, top, right, bottom))

def compute_baseline_features(pil_img: Image.Image) -> Dict[str, float]:
    rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    R = rgb[..., 0].astype(np.float32)
    G = rgb[..., 1].astype(np.float32)
    B = rgb[..., 2].astype(np.float32)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    S = hsv[..., 1].astype(np.float32) / 255.0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
    a = lab[..., 1].astype(np.float32) - 128.0

    denom = R + G + B + 1e-6
    R_norm = R / denom

    feats = {
        "R_norm_p50": float(np.percentile(R_norm, 50)),
        "a_mean": float(np.mean(a)),
        "R_p50": float(np.percentile(R, 50)),
        "R_p10": float(np.percentile(R, 10)),
        "RG": float(np.mean(R) / (np.mean(G) + 1e-6)),
        "S_p50": float(np.percentile(S, 50)),
        "gray_p90": float(np.percentile(gray, 90)),
        "gray_kurt": float(kurtosis(gray.ravel(), fisher=False, bias=False, nan_policy="omit")),
        "gray_std": float(np.std(gray, ddof=0)),
        "gray_mean": float(np.mean(gray)),
        "B_mean": float(np.mean(B)),
        "B_p10": float(np.percentile(B, 10)),
        "B_p75": float(np.percentile(B, 75)),
        "G_kurt": float(kurtosis(G.ravel(), fisher=False, bias=False, nan_policy="omit")),
    }
    return feats

def vascularity_features_from_conjunctiva(rgb_u8: np.ndarray,
                                          black_ridges: bool=True,
                                          min_size:int=50,
                                          area_threshold:int=50) -> Dict[str, float]:
    g = rgb_u8[...,1].astype(np.uint8)
    g_eq = exposure.equalize_adapthist(g, clip_limit=0.01)  # CLAHE [0,1]
    vmap = filters.frangi(
    g_eq,
    sigmas=np.arange(1, 6, 1),
    alpha=0.5, beta=0.5,
    black_ridges=black_ridges
)
    vmap = (vmap - vmap.min()) / (np.ptp(vmap) + 1e-8)


    thr = filters.threshold_otsu(vmap)
    mask = vmap > thr
    mask = morphology.remove_small_objects(mask, min_size=min_size)
    mask = morphology.remove_small_holes(mask, area_threshold=area_threshold)

    skel = skeletonize(mask)

    H, W = mask.shape
    area = float(H*W)

    vessel_area_fraction = float(mask.sum()) / area
    mean_vesselness      = float(vmap.mean())
    p90_vesselness       = float(np.percentile(vmap, 90))

    skeleton_length      = float(skel.sum())
    skeleton_len_per_area= skeleton_length / area

    neigh = convolve(skel.astype(np.uint8), np.ones((3,3), dtype=np.uint8), mode='constant', cval=0)
    branches = ((skel) & (neigh >= 4))
    branchpoint_density = float(branches.sum()) / area

    lbl = measure.label(skel, connectivity=2)
    torts = []
    for region in measure.regionprops(lbl):
        coords = np.array(region.coords)
        if coords.shape[0] < 10:
            continue
        path_len = float(coords.shape[0])
        pmin, pmax = coords.min(0), coords.max(0)
        chord = np.linalg.norm(pmax - pmin) + 1e-8
        torts.append(path_len / chord)
    tortuosity_mean = float(np.mean(torts)) if torts else 1.0

    return {
        "vessel_area_fraction": vessel_area_fraction,
        "mean_vesselness": mean_vesselness,
        "p90_vesselness": p90_vesselness,
        "skeleton_len_per_area": skeleton_len_per_area,
        "branchpoint_density": branchpoint_density,
        "tortuosity_mean": tortuosity_mean,
    }

def main():
    args = parse_args()

    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_crops); out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = Path(args.labels)
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.csv not found at {labels_path.resolve()}")
    labels_df = pd.read_csv(labels_path)
    req = {"filename","hb","status"}
    if not req.issubset(labels_df.columns):
        raise ValueError(f"labels.csv must contain {req}")

    rows = []
    files = sorted([p for p in images_dir.glob("*") if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp")])
    for i, img_path in enumerate(files, 1):
        try:
            pil_full = Image.open(img_path)
            pil_full = exif_upright(pil_full)

            b64 = to_b64_jpeg(pil_full)
            rf_json = roboflow_detect_b64(b64, model_id=args.model_id, api_key=args.api_key, conf_0_100=int(args.conf))
            preds = rf_json.get("predictions", [])
            best = select_best_box(preds, target_class=args.class_name)
            if best is None:
                print(f"[{i}/{len(files)}] {img_path.name}: NO DETECTION")
                continue

            crop = crop_from_box(pil_full, best)
            crop_path = out_dir / img_path.name
            crop.save(crop_path)

            base_feats = compute_baseline_features(crop)
            rgb = np.array(crop.convert("RGB"), dtype=np.uint8)
            vas_feats = vascularity_features_from_conjunctiva(
                rgb, black_ridges=args.black_ridges, min_size=args.min_size, area_threshold=args.area_threshold
            )

            row = {"filename": img_path.name}
            row.update(base_feats)
            row.update(vas_feats)
            rows.append(row)

            print(f"[{i}/{len(files)}] {img_path.name}: OK")
            time.sleep(args.sleep)
        except Exception as e:
            print(f"[{i}/{len(files)}] {img_path.name}: ERROR {e}")

    feat_df = pd.DataFrame(rows)
    merged = labels_df.merge(feat_df, on="filename", how="inner")
    out_csv = Path(args.out_csv)
    merged.to_csv(out_csv, index=False)
    print(f"Saved features CSV: {out_csv.resolve()}")
    print(f"Cropped images at: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
