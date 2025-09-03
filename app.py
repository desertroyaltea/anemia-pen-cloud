#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import base64
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageOps, ImageDraw
import cv2
from scipy.stats import kurtosis
import streamlit as st
from pypmml import Model

# -------------------- Settings -------------------- #
DEFAULT_PMML_PATH = Path("hemo.xml")                     # your exported SPSS PMML
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

# -------------------- UI -------------------- #
st.title("🖊️ Anemia Pen — SPSS/PMML Edition")

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
    rmse = st.number_input("Optional RMSE for CI (±1.96×RMSE)", min_value=0.0, value=0.0, step=0.1, help="Leave 0 to hide CI")

    st.markdown("---")
    st.caption("Model file (SPSS → PMML)")
    pmml_source = st.radio("Load PMML from…", ["Path", "Upload"], horizontal=True)
    pmml_path_str = ""
    pmml_tmp_file: Optional[Path] = None
    if pmml_source == "Path":
        pmml_path_str = st.text_input("PMML path", value=str(DEFAULT_PMML_PATH))
    else:
        up = st.file_uploader("Upload .pmml / .xml", type=["pmml", "xml"])
        if up is not None:
            pmml_tmp_file = Path(st.secrets.get("_tmp_pmml_name_", "uploaded_hemo.xml"))
            # write to a temp file in working dir (Streamlit reload-safe)
            with open(pmml_tmp_file, "wb") as f:
                f.write(up.read())
            pmml_path_str = str(pmml_tmp_file)

    load_btn = st.button("Load PMML model", use_container_width=True)

    model_loaded: Optional[Model] = None
    if load_btn:
        try:
            model_loaded = load_pmml(pmml_path_str)
            st.success(f"PMML loaded: {pmml_path_str}")
        except Exception as e:
            st.error(f"Failed to load PMML: {e}")

# If user already pressed the button in a previous run, try to use cache:
if "model_loaded_flag" not in st.session_state:
    st.session_state["model_loaded_flag"] = False
if load_btn:
    st.session_state["model_loaded_flag"] = (model_loaded is not None)

# Try to use cached model if path string is available and user had loaded before
cached_model: Optional[Model] = None
if st.session_state["model_loaded_flag"] and pmml_path_str:
    try:
        cached_model = load_pmml(pmml_path_str)
    except Exception:
        cached_model = None

uploaded = st.file_uploader("Upload a full-eye photo", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"])

colL, colR = st.columns([1, 1])
with colL:
    st.subheader("1) Original / Detection")
with colR:
    st.subheader("2) Cropped Conjunctiva")

process = st.button("Estimate Hb", type="primary", disabled=(uploaded is None))

# ----------------- Main action ----------------- #
if process:
    if uploaded is None:
        st.warning("Please upload an image.")
        st.stop()
    if not api_key:
        st.error("Missing Roboflow API key.")
        st.stop()
    if cached_model is None:
        st.error("PMML model is not loaded. Load it from the sidebar first.")
        st.stop()

    # Read & fix orientation
    try:
        pil_full = Image.open(io.BytesIO(uploaded.read()))
        pil_full = exif_upright(pil_full)
    except Exception as e:
        st.error(f"Failed to open image: {e}")
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
        st.image(overlay, caption=f"Detection: {best.get('class','?')} (conf {best.get('confidence',0.0):.2f})", use_container_width=True)
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
        pred_hb, extra = score_pmml(cached_model, feats)
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

# Footer
st.markdown("---")
st.caption("Tip: Load PMML (SPSS) from sidebar → upload an eye photo → Estimate Hb.")
