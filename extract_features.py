#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extract_features.py (glare-aware + optional augmentation)

Pipeline:
1) Read full-eye images from --images-dir (default: ./Images)
2) Use Roboflow detect API to crop conjunctiva -> --out-crops (default: ./images_cropped)
3) Detect glare on each crop, inpaint it, and compute features on the inpainted image:
   - 14 baseline color/gray features
   - 6 vascularity features (Frangi vesselness + skeletonization)
   - glare_frac (fraction of pixels flagged as glare pre-inpaint)
4) Join with labels.csv (filename,hb,status) and write --out-csv (default: ./features_dataset.csv)

Optional synthetic glare augmentation:
--augment-glare N   -> for each image, create N synthetic glare variants, mask+inpaint, extract features,
                       and append rows (filename becomes "<stem>#aug{i><ext>"). NOTE: these rows will
                       only join if your labels.csv contains matching augmented filenames.

USAGE (PowerShell example):
python .\extract_features.py `
  --api-key "YOUR_ROBOFLOW_API_KEY" `
  --model-id "eye-conjunctiva-detector/2" `
  --class-name "conjunctiva" `
  --images-dir ".\Images" `
  --out-crops ".\images_cropped" `
  --labels ".\labels.csv" `
  --out-csv ".\features_dataset.csv" `
  --augment-glare 0

Dependencies:
pip install numpy pandas pillow requests opencv-python scikit-image scipy
"""

import io
import time
import base64
import argparse
import random
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


# ---------------- CLI ---------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", required=True, help="Roboflow API key")
    p.add_argument("--model-id", default="eye-conjunctiva-detector/2", help="Roboflow model id (workspace/model:version)")
    p.add_argument("--class-name", default="conjunctiva", help="Target class name in Roboflow predictions")
    p.add_argument("--conf", type=int, default=25, help="Confidence threshold (0-100) for Roboflow")
    p.add_argument("--images-dir", default="Images", help="Folder with full-eye images")
    p.add_argument("--out-crops", default="images_cropped", help="Folder to save conjunctiva crops")
    p.add_argument("--labels", default="labels.csv", help="CSV with columns: filename,hb,status")
    p.add_argument("--out-csv", default="features_dataset.csv", help="Output CSV with merged labels + features")
    p.add_argument("--sleep", type=float, default=0.15, help="Sleep between Roboflow calls (sec)")
    p.add_argument("--black-ridges", type=lambda s: s.lower() != "false", default=True,
                   help="Frangi: set False if vessels appear bright")
    p.add_argument("--min-size", type=int, default=50, help="Min object size for vessel mask cleanup")
    p.add_argument("--area-threshold", type=int, default=50, help="Min hole area to fill in vessel mask")
    p.add_argument("--augment-glare", type=int, default=0,
                   help="Per image, generate N synthetic glare variants (then mask+inpaint+extract)")
    return p.parse_args()


# ------------- Helpers ------------- #
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


# -------- Glare handling -------- #
def detect_glare_mask(rgb: np.ndarray) -> np.ndarray:
    """
    Heuristic glare mask: high V (HSV) + low S, and near-maximum grayscale.
    Returns a binary mask {0,1} (uint8).
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    S = hsv[..., 1] / 255.0
    V = hsv[..., 2] / 255.0

    mask_hsv = (V > 0.90) & (S < 0.25)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    hi = float(np.quantile(gray, 0.995))  # top 0.5% brightest
    mask_gray = gray >= hi

    mask = (mask_hsv | mask_gray).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

def inpaint_glare(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Inpaint glare with OpenCV's Telea algorithm."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask_u8 = (mask.astype(np.uint8) * 255)
    out = cv2.inpaint(bgr, mask_u8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

def add_synthetic_glare(rgb: np.ndarray, n_spots: int = 1) -> np.ndarray:
    """Overlay bright, soft-edged ellipses to simulate glare."""
    h, w, _ = rgb.shape
    aug = rgb.copy()
    for _ in range(n_spots):
        cx = random.randint(int(0.2 * w), int(0.8 * w))
        cy = random.randint(int(0.2 * h), int(0.8 * h))
        ax = random.randint(max(6, w // 25), max(10, w // 12))
        ay = random.randint(max(6, h // 25), max(10, h // 12))
        angle = random.randint(0, 180)
        mask = np.zeros((h, w), np.uint8)
        cv2.ellipse(mask, (cx, cy), (ax, ay), angle, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=ax / 3.0, sigmaY=ay / 3.0)
        m = (mask.astype(np.float32) / 255.0)[..., None]
        aug = (aug.astype(np.float32) * (1 - m) + 255.0 * m).clip(0, 255).astype(np.uint8)
    return aug


# -------- Feature extractors -------- #
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
                                          black_ridges: bool = True,
                                          min_size: int = 50,
                                          area_threshold: int = 50) -> Dict[str, float]:
    """
    Vesselness on CLAHE-equalized green channel + skeleton metrics.
    """
    g = rgb_u8[..., 1].astype(np.uint8)
    g_eq = exposure.equalize_adapthist(g, clip_limit=0.01)  # CLAHE in [0,1]
    vmap = filters.frangi(
        g_eq, sigmas=np.arange(1, 6, 1),
        alpha=0.5, beta=0.5, black_ridges=black_ridges
    )
    # NumPy 2.0-safe normalization
    vmap = (vmap - vmap.min()) / (np.ptp(vmap) + 1e-8)

    thr = filters.threshold_otsu(vmap)
    mask = vmap > thr
    mask = morphology.remove_small_objects(mask, min_size=min_size)
    mask = morphology.remove_small_holes(mask, area_threshold=area_threshold)

    skel = skeletonize(mask)

    H, W = mask.shape
    area = float(H * W)

    vessel_area_fraction = float(mask.sum()) / area
    mean_vesselness = float(vmap.mean())
    p90_vesselness = float(np.percentile(vmap, 90))

    skeleton_length = float(skel.sum())
    skeleton_len_per_area = skeleton_length / area

    neigh = convolve(skel.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode='constant', cval=0)
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


# ------------- Main ------------- #
def process_one_crop(crop_img: Image.Image, args, filename_tag: str) -> Dict[str, float]:
    """
    Glare mask + inpaint, then compute features on the inpainted crop.
    Returns a dict with 'filename' and feature columns.
    """
    rgb = np.array(crop_img.convert("RGB"), dtype=np.uint8)

    # Detect glare and inpaint
    glare_mask = detect_glare_mask(rgb)
    if glare_mask.sum() > 0:
        rgb_proc = inpaint_glare(rgb, glare_mask)
    else:
        rgb_proc = rgb

    # Features on inpainted image
    base_feats = compute_baseline_features(Image.fromarray(rgb_proc))
    vas_feats = vascularity_features_from_conjunctiva(
        rgb_proc, black_ridges=args.black_ridges,
        min_size=args.min_size, area_threshold=args.area_threshold
    )

    row = {"filename": filename_tag, "glare_frac": float(glare_mask.mean())}
    row.update(base_feats)
    row.update(vas_feats)
    return row


def main():
    args = parse_args()

    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_crops); out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = Path(args.labels)
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.csv not found at {labels_path.resolve()}")

    # Load labels and normalize headers (any casing accepted)
    labels_df = pd.read_csv(labels_path)
    labels_df.columns = [c.strip() for c in labels_df.columns]
    lowermap = {c.lower(): c for c in labels_df.columns}
    need = ["filename", "hb", "status"]
    if not all(k in lowermap for k in need):
        raise ValueError(f"labels.csv must contain headers: filename,hb,status (any casing). Found: {labels_df.columns.tolist()}")
    labels_df = labels_df.rename(columns={
        lowermap["filename"]: "filename",
        lowermap["hb"]: "hb",
        lowermap["status"]: "status"
    })

    rows = []
    files = sorted([
        p for p in images_dir.glob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    ])

    for i, img_path in enumerate(files, 1):
        try:
            pil_full = Image.open(img_path)
            pil_full = exif_upright(pil_full)

            # Roboflow detect → best conjunctiva box
            b64 = to_b64_jpeg(pil_full)
            rf_json = roboflow_detect_b64(b64, model_id=args.model_id, api_key=args.api_key, conf_0_100=int(args.conf))
            preds = rf_json.get("predictions", [])
            best = select_best_box(preds, target_class=args.class_name)
            if best is None:
                print(f"[{i}/{len(files)}] {img_path.name}: NO DETECTION")
                continue

            # Crop and save crop
            crop = crop_from_box(pil_full, best)
            crop_path = out_dir / img_path.name
            crop.save(crop_path)

            # (A) Original crop → glare mask+inpaint → features
            row_main = process_one_crop(crop, args, img_path.name)
            rows.append(row_main)

            # (B) Optional synthetic-glare augmentation
            for j in range(max(0, int(args.augment_glare))):
                rgb_aug = add_synthetic_glare(np.array(crop.convert("RGB"), dtype=np.uint8),
                                              n_spots=random.randint(1, 3))
                aug_img = Image.fromarray(rgb_aug)
                tag = f"{img_path.stem}#aug{j}{img_path.suffix}"
                row_aug = process_one_crop(aug_img, args, tag)
                rows.append(row_aug)

            print(f"[{i}/{len(files)}] {img_path.name}: OK (crop, glare-inpaint, features{f', +{args.augment_glare} aug' if args.augment_glare>0 else ''})")
            time.sleep(args.sleep)
        except Exception as e:
            print(f"[{i}/{len(files)}] {img_path.name}: ERROR {e}")

    # Build DataFrame and join with labels
    feat_df = pd.DataFrame(rows)

    # NOTE: Augmented rows (with '#augX') will ONLY join if your labels.csv contains those names.
    # If you prefer to include augmented rows with the same labels as the original image automatically,
    # we can map by stem and expand labels—just say the word.
    merged = labels_df.merge(feat_df, on="filename", how="inner")

    out_csv = Path(args.out_csv)
    merged.to_csv(out_csv, index=False)
    print(f"Saved features CSV: {out_csv.resolve()}")
    print(f"Cropped images at: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
