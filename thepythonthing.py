# app.py — Anemia Pen (Surrogate-only)

import io
import json
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageOps
from scipy.stats import kurtosis
import streamlit as st
import joblib

# -------------------- Constants -------------------- #
ROBOPFLOW_MODEL_ID = "eye-conjunctiva-detector/2"  # your deployed detector
DEFAULT_CONFIDENCE = 0.25
TIMEOUT = 25

SURROGATE_PATH = Path("outputs/models/hemo_surrogate.joblib")  # <— ONLY THIS MODEL
FEATURE_ORDER = [
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

# -------------------- UI -------------------- #
st.set_page_config(page_title="Anemia Pen", page_icon="🖊️", layout="wide")
st.title("🖊️ Anemia Pen — Hemoglobin Estimator (Surrogate Model)")

# Sidebar: API key + options
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("Roboflow API Key", value=st.secrets.get("ROBOFLOW_API_KEY", ""), type="password")
conf = st.sidebar.slider("Detection confidence threshold", 0.05, 0.9, DEFAULT_CONFIDENCE, 0.05)
show_debug = st.sidebar.checkbox("Show debug details", value=False)

# -------------------- Utilities -------------------- #

@st.cache_resource(show_spinner=False)
def load_surrogate():
    if not SURROGATE_PATH.exists():
        st.error(f"Surrogate model not found at `{SURROGATE_PATH}`.")
        st.stop()
    try:
        model = joblib.load(SURROGATE_PATH)
    except Exception as e:
        st.error(f"Failed to load surrogate model: {e}")
        st.stop()
    return model

def pil_fix_orientation(pil_img: Image.Image) -> Image.Image:
    """Auto-rotate based on EXIF and convert to RGB."""
    try:
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception:
        pass
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return pil_img

def pil_to_jpeg_bytes(pil_img: Image.Image, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

def roboflow_detect_and_crop(pil_img: Image.Image, api_key: str, conf_thr: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Sends the upright RGB image to Roboflow via multipart/form-data ('file' part),
    returns (crop_bgr, overlay_bgr, det_info).
    """
    if not api_key:
        raise RuntimeError("Roboflow API key is required.")

    # JPEG bytes for multipart upload
    jpg_bytes = pil_to_jpeg_bytes(pil_img)
    url = f"https://detect.roboflow.com/{ROBOPFLOW_MODEL_ID}?api_key={api_key}&confidence={conf_thr:.2f}"
    files = {"file": ("image.jpg", jpg_bytes, "image/jpeg")}

    # Call RF
    r = requests.post(url, files=files, timeout=TIMEOUT)
    if r.status_code >= 500:
        raise RuntimeError(f"Roboflow server error [{r.status_code}]: {r.text}")
    if r.status_code != 200:
        raise RuntimeError(
            f"Roboflow HTTP {r.status_code}. Body: {r.text[:300]}"
        )

    data = r.json()
    preds = data.get("predictions", [])
    if not preds:
        raise RuntimeError("No conjunctiva detected.")
    # pick best box by confidence
    best = max(preds, key=lambda p: float(p.get("confidence", 0)))
    x, y = float(best["x"]), float(best["y"])
    w, h = float(best["width"]), float(best["height"])

    # Make cv2 images
    rgb = np.array(pil_img)  # HWC, RGB uint8
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Compute crop coords
    H, W = rgb.shape[:2]
    x1 = int(max(0, x - w / 2))
    y1 = int(max(0, y - h / 2))
    x2 = int(min(W, x + w / 2))
    y2 = int(min(H, y + h / 2))
    crop_bgr = bgr[y1:y2, x1:x2].copy()

    # Overlay for visualization
    overlay = bgr.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
    label = f"{best.get('class','conjunctiva')} {best.get('confidence',0):.2f}"
    cv2.putText(overlay, label, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    det_info = {
        "x": x, "y": y, "width": w, "height": h,
        "confidence": float(best.get("confidence", 0)),
        "class": best.get("class", ""), "class_id": best.get("class_id", None),
    }
    return crop_bgr, overlay, det_info

def percentiles_uint8(channel: np.ndarray, plist=(10, 25, 50, 75, 90)) -> Dict[int, float]:
    vals = np.percentile(channel.astype(np.float32), plist)
    return {p: float(vals[i]) for i, p in enumerate(plist)}

def extract_14_features(crop_bgr: np.ndarray) -> Dict[str, float]:
    """
    Returns the 14-feature dict in the exact FEATURE_ORDER.
    """
    if crop_bgr.size == 0:
        raise ValueError("Empty crop for feature extraction.")

    # Basic channels
    b = crop_bgr[:, :, 0].astype(np.float32)
    g = crop_bgr[:, :, 1].astype(np.float32)
    r = crop_bgr[:, :, 2].astype(np.float32)

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)

    # Robust stats
    R_p = percentiles_uint8(r, (10, 25, 50, 75, 90))
    G_p = percentiles_uint8(g, (10, 25, 50, 75, 90))
    B_p = percentiles_uint8(b, (10, 25, 50, 75, 90))
    GR_p = percentiles_uint8(gray, (10, 25, 50, 75, 90))

    R_mean = float(r.mean())
    G_mean = float(g.mean())
    B_mean = float(b.mean())

    R_norm_p50 = float((R_p[50] / (G_p[50] + 1e-6)))  # normalized by green median (avoids divide-by-zero)
    RG = float(R_mean / (G_mean + 1e-6))

    S = hsv[:, :, 1].astype(np.float32)
    S_p50 = float(np.percentile(S, 50))

    gray_mean = float(gray.mean())
    gray_std = float(gray.std(ddof=0))
    # Fisher’s definition (normal==0). Bias False to match SciPy default behavior used earlier.
    gray_kurt = float(kurtosis(gray, fisher=True, bias=False))
    gray_p90 = float(GR_p[90])

    a_mean = float(lab[:, :, 1].mean())  # Lab a* channel mean
    G_kurt = float(kurtosis(g, fisher=True, bias=False))

    feat = {
        "R_norm_p50": R_norm_p50,
        "a_mean": a_mean,
        "R_p50": float(R_p[50]),
        "R_p10": float(R_p[10]),
        "RG": RG,
        "S_p50": S_p50,
        "gray_p90": gray_p90,
        "gray_kurt": gray_kurt,
        "gray_std": gray_std,
        "gray_mean": gray_mean,
        "B_mean": B_mean,
        "B_p10": float(B_p[10]),
        "B_p75": float(B_p[75]),
        "G_kurt": G_kurt,
    }
    # Ensure all 14 present:
    for k in FEATURE_ORDER:
        if k not in feat:
            raise KeyError(f"Missing feature: {k}")
        if not np.isfinite(feat[k]):
            raise ValueError(f"Non-finite value in {k}: {feat[k]}")
    return feat

def predict_surrogate(model, feat_dict: Dict[str, float]) -> Tuple[float, Dict[str, Any]]:
    df = pd.DataFrame([[feat_dict[k] for k in FEATURE_ORDER]], columns=FEATURE_ORDER)
    y = model.predict(df)
    pred = float(y[0])
    debug = {"backend": "joblib_sklearn", "used_columns": FEATURE_ORDER}
    return pred, debug

def to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

# -------------------- Main App Flow -------------------- #
model = load_surrogate()
st.success(f"Loaded surrogate model: `{SURROGATE_PATH.name}`")

col_l, col_r = st.columns([1, 1])

with col_l:
    st.subheader("1) Upload an eye photo")
    up = st.file_uploader("Upload image (JPG/PNG) or use your phone camera", type=["jpg", "jpeg", "png"])
    if up:
        try:
            pil = Image.open(up)
            pil = pil_fix_orientation(pil)
            st.image(pil, caption="Original (auto-oriented)", use_container_width=True)
        except Exception as e:
            st.error(f"Failed to open image: {e}")
            st.stop()
    else:
        st.info("Please upload a photo to proceed.")
        st.stop()

with col_r:
    st.subheader("2) Detect & crop conjunctiva")
    if not api_key:
        st.warning("Enter your Roboflow API key in the sidebar.")
        st.stop()

    try:
        crop_bgr, overlay_bgr, det_info = roboflow_detect_and_crop(pil, api_key, conf)
        st.image(to_pil(overlay_bgr), caption="Detection overlay", use_container_width=True)
        st.image(to_pil(crop_bgr), caption="Cropped conjunctiva", use_container_width=True)
        if show_debug:
            st.json({"det_info": det_info})
    except Exception as e:
        st.error(f"Detection/Cropping failed: {e}")
        st.stop()

st.subheader("3) Extract features & estimate Hb (surrogate)")
try:
    feats = extract_14_features(crop_bgr)
    if show_debug:
        st.json({"features": feats})
except Exception as e:
    st.error(f"Feature extraction failed: {e}")
    st.stop()

try:
    hb_pred, dbg = predict_surrogate(model, feats)
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# Show result
st.metric(label="Estimated Hemoglobin (g/dL)", value=f"{hb_pred:.2f}")

# Optional: show inputs
with st.expander("Extracted parameters (14 features)"):
    feat_df = pd.DataFrame([feats])
    # Streamlit <=1.50: st.dataframe does not support use_column_width
    st.dataframe(feat_df, height=420)

if show_debug:
    st.subheader("Debug")
    st.json(dbg)
