#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlit app for Conjunctiva-based Anemia Screening & Hb Estimation

This version is locked to the FAST extraction models trained WITHOUT Age/GENDER:
- Run folder: models/run_20250912_173617
- Uses glare masking + inpainting and vascularity features
- Extracts EXACT features required by the models

Requirements:
  streamlit numpy pandas pillow requests opencv-python scikit-image scipy joblib
"""

import os
import io
import json
import base64
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import requests
import joblib

# Imaging / features
import cv2
from scipy.stats import kurtosis
from scipy.ndimage import convolve
from skimage import exposure, filters, morphology, measure
from skimage.morphology import skeletonize

# ---------- MODEL ARTIFACTS (locked to this run) ---------- #
RUN_DIR = Path("models") / "run_20250912_173617"
ANEMIA_MODEL_PATH = RUN_DIR / "anemia_rf.joblib"
HB_MODEL_PATH     = RUN_DIR / "hb_rf.joblib"
CLF_FEATS_PATH    = RUN_DIR / "clf_features.json"
REG_FEATS_PATH    = RUN_DIR / "reg_features.json"

# Fallbacks (used only if JSON files are not found in RUN_DIR)
FALLBACK_CLF_FEATURES = [
    "R_p50","R_norm_p50","R_p10","a_mean","RG","gray_mean",
    "gray_kurt","S_p50","gray_p90","B_p10","glare_frac","G_kurt",
    "B_mean","mean_vesselness","B_p75","p90_vesselness","tortuosity_mean","gray_std"
]
FALLBACK_REG_FEATURES = [
    "a_mean","R_norm_p50","RG","R_p50","R_p10","S_p50","tortuosity_mean",
    "gray_mean","gray_kurt","B_p10","glare_frac","G_kurt","gray_std",
    "skeleton_len_per_area","B_p75","vessel_area_fraction","gray_p90",
    "B_mean","branchpoint_density","p90_vesselness","mean_vesselness"
]

# ---------- ROBOFLOW SETTINGS ---------- #
DEFAULT_MODEL_ID  = "eye-conjunctiva-detector/2"
DEFAULT_CLASS     = "conjunctiva"
DEFAULT_CONF      = 25  # 0-100

# ---------- UTILS ---------- #
def exif_upright(pil_img: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(pil_img).convert("RGB")

def pil_to_jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()

def to_b64_jpeg(img: Image.Image) -> str:
    return base64.b64encode(pil_to_jpeg_bytes(img, quality=90)).decode("utf-8")

def roboflow_detect_b64(b64_str: str, model_id: str, api_key: str, conf_0_100: int, timeout: float = 60.0) -> dict:
    url = f"https://detect.roboflow.com/{model_id}"
    params = {"api_key": api_key, "confidence": str(conf_0_100), "format": "json"}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    resp = requests.post(url, params=params, data=b64_str, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

def select_best_box(preds, target_class: str):
    if not preds:
        return None
    target = [p for p in preds if str(p.get("class", "")).lower() == target_class.lower()]
    if target:
        return max(target, key=lambda p: p.get("confidence", 0.0))
    return max(preds, key=lambda p: p.get("confidence", 0.0))

def crop_from_box(pil_img: Image.Image, box: dict) -> Image.Image:
    x = float(box["x"]); y = float(box["y"])
    w = float(box["width"]); h = float(box["height"])
    left = int(round(x - w / 2.0)); top = int(round(y - h / 2.0))
    right = int(round(x + w / 2.0)); bottom = int(round(y + h / 2.0))
    left = max(0, left); top = max(0, top)
    right = min(pil_img.width, right); bottom = min(pil_img.height, bottom)
    if right <= left or bottom <= top:
        return pil_img.copy()
    return pil_img.crop((left, top, right, bottom))

# ---------- GLARE HELPERS ---------- #
def detect_glare_mask(rgb: np.ndarray) -> np.ndarray:
    """
    Heuristic glare mask: high V (HSV) + low S, and near-maximum grayscale.
    Returns binary mask {0,1} uint8.
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
    """OpenCV Telea inpainting over glare mask."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask_u8 = (mask.astype(np.uint8) * 255)
    out = cv2.inpaint(bgr, mask_u8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

# ---------- FEATURE EXTRACTION ---------- #
def compute_baseline_features(pil_img: Image.Image) -> dict:
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
        "R_p50": float(np.percentile(R, 50)),
        "R_norm_p50": float(np.percentile(R_norm, 50)),
        "a_mean": float(np.mean(a)),
        "R_p10": float(np.percentile(R, 10)),
        "gray_mean": float(np.mean(gray)),
        "RG": float(np.mean(R) / (np.mean(G) + 1e-6)),
        "gray_kurt": float(kurtosis(gray.ravel(), fisher=False, bias=False, nan_policy="omit")),
        "gray_p90": float(np.percentile(gray, 90)),
        "S_p50": float(np.percentile(S, 50)),
        "B_p10": float(np.percentile(B, 10)),
        "B_mean": float(np.mean(B)),
        "gray_std": float(np.std(gray, ddof=0)),
        "B_p75": float(np.percentile(B, 75)),
        "G_kurt": float(kurtosis(rgb[...,1].ravel(), fisher=False, bias=False, nan_policy="omit")),
    }
    return feats

def vascularity_features_from_conjunctiva(rgb_u8: np.ndarray,
                                          black_ridges: bool = True,
                                          min_size: int = 50,
                                          area_threshold: int = 50) -> dict:
    """
    Vesselness on CLAHE-equalized green channel + skeleton metrics.
    NumPy 2.0-safe normalization.
    """
    g = rgb_u8[..., 1].astype(np.uint8)
    g_eq = exposure.equalize_adapthist(g, clip_limit=0.01)  # CLAHE [0,1]
    vmap = filters.frangi(
        g_eq, sigmas=np.arange(1, 6, 1),
        alpha=0.5, beta=0.5, black_ridges=black_ridges
    )
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

def extract_all_features_from_crop(crop_img: Image.Image) -> dict:
    rgb = np.array(crop_img.convert("RGB"), dtype=np.uint8)
    glare_mask = detect_glare_mask(rgb)
    if glare_mask.sum() > 0:
        rgb_proc = inpaint_glare(rgb, glare_mask)
    else:
        rgb_proc = rgb

    base = compute_baseline_features(Image.fromarray(rgb_proc))
    vas  = vascularity_features_from_conjunctiva(rgb_proc, black_ridges=True, min_size=50, area_threshold=50)
    out = {"glare_frac": float(glare_mask.mean())}
    out.update(base); out.update(vas)
    return out

# ---------- MODEL LOADING (cached) ---------- #
@st.cache_resource(show_spinner=False)
def load_artifacts(run_dir: Path):
    if not run_dir.exists():
        raise FileNotFoundError(f"Run folder not found: {run_dir}")
    clf = joblib.load(run_dir / "anemia_rf.joblib")
    rgr = joblib.load(run_dir / "hb_rf.joblib")
    # Prefer JSONs from the run; otherwise use fallbacks locked to this model family
    try:
        with open(run_dir / "clf_features.json", "r", encoding="utf-8") as f:
            clf_features = json.load(f)
    except Exception:
        clf_features = FALLBACK_CLF_FEATURES
    try:
        with open(run_dir / "reg_features.json", "r", encoding="utf-8") as f:
            reg_features = json.load(f)
    except Exception:
        reg_features = FALLBACK_REG_FEATURES
    return clf, rgr, clf_features, reg_features

# ---------- STREAMLIT UI ---------- #
st.set_page_config(page_title="Conjunctiva Anemia Screener + Hb Estimator (No Age/GENDER)", layout="centered")

st.title("Conjunctiva Anemia Screening & Hb Estimation")
st.caption("Using FAST extraction models trained **without** Age/GENDER • "
           "Artifacts loaded from **models/run_20250912_173617**.")

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Task", ["Screen for Anemia", "Estimate Hb"], index=0)

    # Classification display options
    thresh = st.slider("Anemia decision threshold (P[anemia])", 0.10, 0.90, 0.50, 0.01)

    st.subheader("Roboflow Detection")
    rf_api_key = st.text_input("API Key", value=os.getenv("ROBOFLOW_API_KEY", ""), type="password")
    model_id = st.text_input("Model ID", value=DEFAULT_MODEL_ID)
    target_class = st.text_input("Target Class", value=DEFAULT_CLASS)
    conf = st.slider("Detection confidence (0–100)", 0, 100, DEFAULT_CONF, 1)
    st.caption("Tip: set environment variable ROBOFLOW_API_KEY for convenience.")

uploaded = st.file_uploader("Upload a full-eye image", type=["jpg","jpeg","png","bmp","tif","tiff","webp"])

# ---------- MAIN ACTION ---------- #
if uploaded is not None:
    # Load models & feature lists
    try:
        clf, rgr, clf_features, reg_features = load_artifacts(RUN_DIR)
    except Exception as e:
        st.error(f"Failed to load models/features from {RUN_DIR}: {e}")
        st.stop()

    # Read image and run Roboflow detect
    try:
        pil_full = Image.open(uploaded)
        pil_full = exif_upright(pil_full)
        st.image(pil_full, caption="Uploaded Image (uprighted)", use_container_width=True)

        if not rf_api_key:
            st.warning("Enter your **Roboflow API Key** in the sidebar.")
            st.stop()

        b64 = to_b64_jpeg(pil_full)
        rf_json = roboflow_detect_b64(b64, model_id=model_id, api_key=rf_api_key, conf_0_100=int(conf))
        preds = rf_json.get("predictions", [])
        best = select_best_box(preds, target_class=target_class)
        if best is None:
            st.error("No conjunctiva detected. Try a clearer photo or lower the confidence.")
            st.stop()

        crop = crop_from_box(pil_full, best)
        st.image(crop, caption="Detected Conjunctiva (crop)", use_container_width=True)

    except Exception as e:
        st.error(f"Roboflow detection/cropping failed: {e}")
        st.stop()

    # Extract features from crop (with glare inpaint)
    try:
        feats = extract_all_features_from_crop(crop)
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        st.stop()

    # Build per-task feature vectors and predict
    def build_vector(required_feats, feat_dict):
        x = []
        missing = []
        for f in required_feats:
            val = feat_dict.get(f, None)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                x.append(0.0)  # fallback; for exact training behavior, we can add saved medians later
                missing.append(f)
            else:
                x.append(float(val))
        return np.array([x], dtype=np.float32), missing

    st.divider()
    if mode == "Screen for Anemia":
        Xc, miss_c = build_vector(clf_features, feats)
        if miss_c:
            st.info(f"Some classifier features missing; using 0.0 for: {', '.join(miss_c)}")

        try:
            if hasattr(clf, "predict_proba"):
                p1 = float(clf.predict_proba(Xc)[0, 1])
            else:
                score = float(clf.decision_function(Xc)[0])
                p1 = 1.0 / (1.0 + np.exp(-score))
            label = "Possible Anemia" if p1 >= thresh else "Likely Not Anemia"
            st.subheader("Anemia Screening Result")
            st.markdown(f"**{label}**  •  P(anemia) = **{p1:.2%}**  •  Threshold = {thresh:.2f}")
        except Exception as e:
            st.error(f"Classification failed: {e}")

    else:
        Xr, miss_r = build_vector(reg_features, feats)
        if miss_r:
            st.info(f"Some regression features missing; using 0.0 for: {', '.join(miss_r)}")
        try:
            hb_pred = float(rgr.predict(Xr)[0])
            st.subheader("Hemoglobin (Hb) Estimate")
            st.markdown(f"**Estimated Hb: {hb_pred:.2f} g/dL**")
        except Exception as e:
            st.error(f"Regression failed: {e}")

    # Show feature table actually used (for transparency)
    with st.expander("Show extracted feature values"):
        df_show = pd.DataFrame([feats]).T
        df_show.columns = ["value"]
        st.dataframe(df_show, use_container_width=True)

st.caption("Note: For screening/estimation only — not a diagnostic device.")
