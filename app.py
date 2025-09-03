#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import base64
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageOps, ImageDraw
import cv2
from scipy.stats import kurtosis
import streamlit as st

# HEIC/HEIF support for iPhone photos (registers a PIL opener)
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass  # If pillow-heif not installed, HEIC won't work; we'll warn later when needed.

# PMML (SPSS model) runtime
from pypmml import Model

# -------------------- Settings -------------------- #
DEFAULT_PMML_PATH = Path("hemo.xml")                     # your exported SPSS PMML in repo root
DEFAULT_MODEL_ID = "eye-conjunctiva-detector/2"          # Roboflow model id
DEFAULT_CLASS_NAME = "conjunctiva"
DEFAULT_CONF_0_100 = 25                                  # 0..100 (≈ 0.25)
FEATURE_COLUMNS = [
    "R_norm_p50", "a_mean", "R_p50", "R_p10", "RG", "S_p50",
    "gray_p90", "gray_kurt", "gray_std", "gray_mean",
    "B_mean", "B_p10", "B_p75", "G_kurt",
]
# --------------------------------------------------- #

st.set_page_config(page_title="Anemia Pen (PMML)", layout="wide")
st.title("🖊️ Anemia Pen — SPSS/PMML Edition")

# ---------------- Helper functions ----------------- #
def exif_upright(pil_img: Image.Image) -> Image.Image:
    """Physically apply EXIF orientation so pixels are upright."""
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

def compute_features_from_pil(pil_img: Image.Image) -> Dict[str, float]:
    """Compute the 14 features exactly as in Step 2."""
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

@st.cache_resource(show_spinner=False)
def load_pmml(path_str: str) -> Model:
    return Model.load(path_str)

# Auto-load PMML ON STARTUP (no user action)
MODEL: Optional[Model] = None
PMML_LOAD_ERROR: Optional[str] = None
try:
    if DEFAULT_PMML_PATH.exists():
        MODEL = load_pmml(str(DEFAULT_PMML_PATH))
    else:
        PMML_LOAD_ERROR = f"PMML file not found at {DEFAULT_PMML_PATH}. Commit hemo.xml or upload per-session."
except Exception as _e:
    PMML_LOAD_ERROR = f"Failed to load PMML: {_e}"

def score_pmml(model: Model, feats: Dict[str, float]) -> Tuple[float, Dict[str, Any]]:
    df = pd.DataFrame([{k: float(feats[k]) for k in FEATURE_COLUMNS}])
    scored = model.predict(df)
    # Try common columns for regression output
    if "predicted_hb" in scored.columns:
        pred = float(scored.loc[0, "predicted_hb"])
    elif "hb" in scored.columns:
        pred = float(scored.loc[0, "hb"])
    else:
        num_cols = [c for c in scored.columns if np.issubdtype(scored[c].dtype, np.number)]
        if not num_cols:
            raise RuntimeError(f"Unexpected PMML output columns: {list(scored.columns)}")
        pred = float(scored.loc[0, num_cols[-1]])
    extra = {"columns": list(scored.columns)}
    return pred, extra

def get_uploaded_image_bytes() -> Tuple[Optional[bytes], Optional[str]]:
    """
    Returns (image_bytes, source) where source is 'camera' or 'upload'.
    Prefers camera if both provided.
    """
    # Camera capture (works on iPhone Safari). Returns a BytesIO-like object.
    cam = st.camera_input("Take a photo (camera)", help="On iPhone, allow camera access")
    if cam is not None:
        return cam.getvalue(), "camera"

    # Fallback: file upload. Include HEIC for iPhones.
    up = st.file_uploader(
        "…or upload an image",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp", "heic", "heif"],
        accept_multiple_files=False,
    )
    if up is not None:
        return up.read(), "upload"

    return None, None

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "Roboflow API Key",
        value=(st.secrets.get("ROBOFLOW_API_KEY") or os.getenv("ROBOFLOW_API_KEY", "")),
        type="password",
        help="Stored only in your session. You can also set ROBOFLOW_API_KEY in app secrets.",
    )
    model_id = st.text_input("Roboflow Model ID", value=DEFAULT_MODEL_ID)
    class_name = st.text_input("Target class", value=DEFAULT_CLASS_NAME)
    conf = st.slider("Confidence threshold (0–100)", min_value=1, max_value=100, value=DEFAULT_CONF_0_100)
    rmse = st.number_input("Optional RMSE for CI (±1.96×RMSE)", min_value=0.0, value=0.0, step=0.1, help="Leave 0 to hide CI")

    st.markdown("---")
    if PMML_LOAD_ERROR:
        st.error(PMML_LOAD_ERROR)
    else:
        st.success(f"PMML auto-loaded: {DEFAULT_PMML_PATH.name}")

colL, colR = st.columns([1, 1])
with colL:
    st.subheader("1) Original / Detection")
with colR:
    st.subheader("2) Cropped Conjunctiva")

img_bytes, source = get_uploaded_image_bytes()
process = st.button("Estimate Hb", type="primary", disabled=(img_bytes is None))

# ----------------- Main action ----------------- #
if process:
    if img_bytes is None:
        st.warning("No image received. On iPhone, try the camera button again and allow camera access.")
        st.stop()
    if not api_key:
        st.error("Missing Roboflow API key.")
        st.stop()
    if MODEL is None:
        st.error("PMML model is not loaded. Please commit hemo.xml or fix loading error in sidebar.")
        st.stop()

    # Read & fix orientation (support HEIC if pillow-heif is installed)
    try:
        pil_full = Image.open(io.BytesIO(img_bytes))
        pil_full = exif_upright(pil_full)
    except Exception as e:
        st.error(f"Failed to open image. If this is HEIC, ensure pillow-heif and libheif are installed. Error: {e}")
        st.stop()

    # Detect via Roboflow (send the upright image)
    try:
        b64 = to_b64_jpeg(pil_full)
        rf_json = roboflow_detect_b64(b64, model_id=model_id, api_key=api_key, conf_0_100=int(conf))
        preds = rf_json.get("predictions", [])
        best = select_best_box(preds, target_class=class_name)
        if best is None:
            st.warning("No detection found. Try lowering the confidence threshold or using a clearer image.")
            st.stop()
    except Exception as e:
        st.error(f"Roboflow error: {e}")
        st.stop()

    # Draw overlay + crop
    try:
        overlay = draw_box_overlay(pil_full, best)
        crop = crop_from_box(pil_full, best)
    except Exception as e:
        st.error(f"Cropping error: {e}")
        st.stop()

    with colL:
        st.image(overlay, caption=f"Detection: {best.get('class','?')} (conf {best.get('confidence',0.0):.2f}) — source: {source}", use_container_width=True)
    with colR:
        st.image(crop, caption="Cropped conjunctiva", use_container_width=True)

    # Features
    try:
        feats = compute_features_from_pil(crop)
    except Exception as e:
        st.error(f"Feature extraction error: {e}")
        st.stop()

    # Score
    try:
        pred_hb, extra = score_pmml(MODEL, feats)
    except Exception as e:
        st.error(f"PMML scoring error: {e}")
        st.stop()

    # CI (optional)
    lower = upper = None
    if rmse and rmse > 0:
        half = 1.96 * float(rmse)
        lower = pred_hb - half
        upper = pred_hb + half

    st.markdown("---")
    st.subheader("3) Result")

    if lower is not None:
        st.metric(label="Estimated Hb (g/dL)", value=f"{pred_hb:.2f}", delta=f"95% ≈ [{lower:.2f}, {upper:.2f}]")
    else:
        st.metric(label="Estimated Hb (g/dL)", value=f"{pred_hb:.2f}")

    # Show parameters (features)
    st.markdown("### Extracted Parameters (14)")
    feat_df = pd.DataFrame([feats], columns=FEATURE_COLUMNS)
    st.dataframe(feat_df, use_container_width=True, height=420)

    # Tiny debug section
    with st.expander("Advanced / Debug info"):
        st.write("Roboflow best box:", best)
        st.write("PMML output columns:", extra.get("columns"))

st.markdown("---")
st.caption("Tip: Use the **camera** on iPhone or upload an image. PMML model auto-loads from hemo.xml.")
