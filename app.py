#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io, os, base64, json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageOps, ImageDraw
import cv2
from scipy.stats import kurtosis
from scipy.ndimage import convolve
import streamlit as st
import joblib
import pickle

# NEW: vesselness tools
from skimage import exposure, filters, morphology, measure
from skimage.morphology import skeletonize

# -------------------- Settings -------------------- #
DEFAULT_MODEL_ID = "eye-conjunctiva-detector/2"
DEFAULT_CLASS_NAME = "conjunctiva"
DEFAULT_CONF_0_100 = 25
# --------------------------------------------------- #

st.set_page_config(page_title="Anemia Screening & Hb Estimation", layout="wide")

# ---------- Utilities ----------
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

def draw_box_overlay(pil_img: Image.Image, box: Dict[str, Any]) -> Image.Image:
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    x = float(box["x"]); y = float(box["y"])
    w = float(box["width"]); h = float(box["height"])
    left = int(round(x - w / 2.0)); top = int(round(y - h / 2.0))
    right = int(round(x + w / 2.0)); bottom = int(round(y + h / 2.0))
    draw.rectangle([(left, top), (right, bottom)], outline=(255, 0, 0), width=3)
    txt = f"{box.get('class','?')} {box.get('confidence',0.0):.2f}"
    draw.text((left + 4, top + 4), txt, fill=(255, 0, 0))
    return img

# ---------- Feature extraction ----------
def compute_baseline_features(pil_img: Image.Image) -> Dict[str, float]:
    """
    14 baseline features (color/gray stats)
    """
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
    """
    Classical vesselness + skeletonization; no manual labels required.
    """
    g = rgb_u8[...,1].astype(np.uint8)
    g_eq = exposure.equalize_adapthist(g, clip_limit=0.01)  # CLAHE in [0,1]
    vmap = filters.frangi(
        g_eq,
        sigmas=np.arange(1, 6, 1),
        alpha=0.5, beta=0.5,
        black_ridges=black_ridges
    )
    # NumPy 2.0 safe normalization
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

    # simple tortuosity proxy
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

def compute_all_features(pil_img: Image.Image) -> Dict[str, float]:
    base = compute_baseline_features(pil_img)
    rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    vas  = vascularity_features_from_conjunctiva(rgb, black_ridges=True, min_size=50, area_threshold=50)
    all_feats = {**base, **vas}
    return all_feats

# ---------- Model loading ----------
def find_latest_run_dir(models_root: Path) -> Optional[Path]:
    if not models_root.exists():
        return None
    run_dirs = sorted([p for p in models_root.iterdir() if p.is_dir() and p.name.startswith("run_")])
    return run_dirs[-1] if run_dirs else None

def load_model_bundle() -> Tuple[Optional[Any], Optional[Any], List[str], List[str], str]:
    """
    Returns:
      anemia_model, hb_model, clf_cols, reg_cols, status_message
    Strategy:
      - Prefer newest models/run_*/ (joblib + feature jsons)
      - Fall back to legacy .model files only if nothing else exists
    """
    models_root = Path("models")
    run_dir = find_latest_run_dir(models_root)
    if run_dir:
        clf_path = run_dir / "anemia_rf.joblib"
        reg_path = run_dir / "hb_rf.joblib"
        clf_cols_path = run_dir / "clf_features.json"
        reg_cols_path = run_dir / "reg_features.json"
        try:
            clf = joblib.load(clf_path)
            reg = joblib.load(reg_path)
            clf_cols = json.loads(clf_cols_path.read_text()) if clf_cols_path.exists() else []
            reg_cols = json.loads(reg_cols_path.read_text()) if reg_cols_path.exists() else []
            msg = f"Loaded models from {run_dir}"
            return clf, reg, clf_cols, reg_cols, msg
        except Exception as e:
            # fall through to legacy loader
            msg = f"Failed to load {run_dir} bundle: {e}. Trying legacy paths…"
    else:
        msg = "No models/run_* folder found. Trying legacy paths…"

    # Legacy KNIME/other (these usually fail); try anyway then fallback to .joblib in CWD
    def _try(primary_path: str, fallbacks: List[str]) -> Tuple[Optional[Any], str]:
        try:
            return joblib.load(primary_path), f"Loaded: {primary_path}"
        except Exception as e:
            for c in fallbacks:
                if Path(c).exists():
                    try:
                        return joblib.load(c), f"Primary load failed ({type(e).__name__}); fell back to {c}"
                    except Exception:
                        continue
            return None, f"Failed to load {primary_path}: {e}"

    anemia_model, anemia_msg = _try(
        "anemia_classification_model.model",
        ["anemia_rf.joblib", "anemia_classifier.joblib"]
    )
    hb_model, hb_msg = _try(
        "hb_regression_model.model",
        ["hb_rf.joblib", "hb_regressor.joblib"]
    )

    # If we got both, require feature lists (else assume baseline 14)
    clf_cols = []
    reg_cols = []
    if anemia_model is not None and Path("clf_features.json").exists():
        clf_cols = json.loads(Path("clf_features.json").read_text())
    if hb_model is not None and Path("reg_features.json").exists():
        reg_cols = json.loads(Path("reg_features.json").read_text())

    status = "; ".join([msg, anemia_msg, hb_msg])
    return anemia_model, hb_model, clf_cols, reg_cols, status

def align_features(row_dict: Dict[str, float], required_cols: List[str]) -> pd.DataFrame:
    """
    Build a 1-row DataFrame with exactly 'required_cols' in this order.
    Missing features are filled with 0. Extra keys are ignored.
    """
    data = {c: float(row_dict.get(c, 0.0)) for c in required_cols}
    return pd.DataFrame([data], columns=required_cols)

# ---------- UI ----------
st.title("Anemia Screening & Hb Estimation")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "Roboflow API Key",
        value=os.getenv("ROBOFLOW_API_KEY", ""),
        type="password",
        help="Stored only in your session. Or set env var ROBOFLOW_API_KEY.",
    )
    model_id = st.text_input("Roboflow Model ID", value=DEFAULT_MODEL_ID)
    class_name = st.text_input("Target class", value=DEFAULT_CLASS_NAME, help="Class to prefer when multiple detections")
    conf = st.slider("Confidence threshold (0–100)", min_value=1, max_value=100, value=DEFAULT_CONF_0_100)

    st.markdown("---")
    task_option = st.radio("Choose your task:", ("Estimate Hb", "Screen for Anemia"), horizontal=False)

uploaded = st.file_uploader("Upload a full-eye photo", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"])

colL, colR = st.columns([1, 1])
with colL: st.subheader("1) Original / Detection")
with colR: st.subheader("2) Cropped Conjunctiva")

process = st.button("Run Prediction", type="primary", disabled=(uploaded is None))

# --------- Load models once ---------
@st.cache_resource(show_spinner=True)
def _load_models():
    return load_model_bundle()

anemia_model, hb_model, clf_cols, reg_cols, load_msg = _load_models()

if anemia_model is None or hb_model is None:
    st.info(load_msg)

# ----------------- Main action ----------------- #
if process:
    if uploaded is None:
        st.warning("Please upload an image."); st.stop()
    if not api_key:
        st.error("Missing Roboflow API key."); st.stop()
    if anemia_model is None and task_option == "Screen for Anemia":
        st.error("Anemia model not loaded."); st.stop()
    if hb_model is None and task_option == "Estimate Hb":
        st.error("Hb model not loaded."); st.stop()

    # Open + upright
    try:
        pil_full = Image.open(io.BytesIO(uploaded.read()))
        pil_full = exif_upright(pil_full)
    except Exception as e:
        st.error(f"Failed to open image: {e}"); st.stop()

    # Detect
    try:
        b64 = to_b64_jpeg(pil_full)
        rf_json = roboflow_detect_b64(b64, model_id=model_id, api_key=api_key, conf_0_100=int(conf))
        preds = rf_json.get("predictions", [])
        best = select_best_box(preds, target_class=class_name)
        if best is None:
            st.warning("No detection found. Try lowering the confidence threshold or using a clearer image."); st.stop()
    except Exception as e:
        st.error(f"Roboflow error: {e}"); st.stop()

    # Draw & crop
    try:
        overlay = draw_box_overlay(pil_full, best)
        crop = crop_from_box(pil_full, best)
    except Exception as e:
        st.error(f"Cropping error: {e}"); st.stop()

    with colL:
        st.image(overlay, caption=f"Detection: {best.get('class','?')} (conf {best.get('confidence',0.0):.2f})", use_container_width=True)
    with colR:
        st.image(crop, caption="Cropped conjunctiva", use_container_width=True)

    # Features (baseline + vascularity)
    try:
        feats_all = compute_all_features(crop)
    except Exception as e:
        st.error(f"Feature extraction error: {e}"); st.stop()

    st.markdown("---"); st.subheader("3) Result")

    # Column lists for each model (fallback to baseline 14 if json lists missing)
    baseline14 = [
        "R_norm_p50", "a_mean", "R_p50", "R_p10", "RG", "S_p50",
        "gray_p90", "gray_kurt", "gray_std", "gray_mean",
        "B_mean", "B_p10", "B_p75", "G_kurt"
    ]
    clf_required = clf_cols if clf_cols else baseline14
    reg_required = reg_cols if reg_cols else baseline14

    if task_option == "Estimate Hb":
        try:
            X = align_features(feats_all, reg_required)
            pred = float(hb_model.predict(X)[0])
            st.metric(label="Estimated Hb (g/dL)", value=f"{pred:.2f}")
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        try:
            X = align_features(feats_all, clf_required)
            pred = int(anemia_model.predict(X)[0])
            if pred == 1:
                st.warning("**Result:** Possible Anemia")
            else:
                st.success("**Result:** Not Anemic")
        except Exception as e:
            st.error(f"Prediction error: {e}")

    # Show feature values we computed (for transparency)
    with st.expander("Extracted features (baseline + vascularity)"):
        st.dataframe(pd.DataFrame([feats_all]).T.rename(columns={0: "value"}))

    with st.expander("Advanced / Debug info"):
        st.write("Model load:", load_msg)
        st.write("Roboflow best box:", best)
        st.write("Classifier expects columns:", clf_required)
        st.write("Regressor expects columns:", reg_required)
