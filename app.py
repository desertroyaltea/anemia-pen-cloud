# app.py
# Anemia Pen — Streamlit app with Roboflow crop + 14-feature extraction + PMML prediction
# - Auto loads hemo3.xml (falls back to hemo.xml), but lets you switch/create a custom model
# - Fixes "constant prediction" by strictly selecting the correct PMML output column
#   and validating input feature names against the PMML MiningSchema
# - Supports both file upload and camera capture on iPhone/Android

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import cv2
import requests
from PIL import Image, ImageOps
from scipy.stats import kurtosis
import streamlit as st
from pypmml import Model

# -------------------- Settings -------------------- #
DEFAULT_PMML_PRIMARY = Path("hemo3.xml")
DEFAULT_PMML_FALLBACK = Path("hemo.xml")
DEFAULT_RMSE = 1.7
ROBOFLOW_MODEL_ID = "eye-conjunctiva-detector/2"  # you can change from sidebar if needed
DEFAULT_CONFIDENCE = 0.25  # detection threshold
# -------------------------------------------------- #


# -------------------- Utilities -------------------- #
def pil_fix_orientation(pil_img: Image.Image) -> Image.Image:
    """Apply EXIF orientation (important for mobile-photos)."""
    try:
        return ImageOps.exif_transpose(pil_img)
    except Exception:
        return pil_img


def pil_to_jpeg_bytes(pil_img: Image.Image, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return rgb_to_bgr(rgb)


def crop_box(img_w: int, img_h: int, cx: float, cy: float, w: float, h: float) -> Tuple[int, int, int, int]:
    """Convert Roboflow center format (x,y,w,h) to clipped [x1,y1,x2,y2] box."""
    x1 = int(round(cx - w / 2.0))
    y1 = int(round(cy - h / 2.0))
    x2 = int(round(cx + w / 2.0))
    y2 = int(round(cy + h / 2.0))
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))
    if x2 <= x1: x2 = min(img_w - 1, x1 + 1)
    if y2 <= y1: y2 = min(img_h - 1, y1 + 1)
    return x1, y1, x2, y2


# -------------------- Roboflow (pure HTTP) -------------------- #
def roboflow_detect_and_crop(
    api_key: str,
    model_id: str,
    image_bytes: bytes,
    conf_threshold: float = DEFAULT_CONFIDENCE
) -> Tuple[Image.Image, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Send oriented JPEG bytes to Roboflow, take top-confidence 'conjunctiva', crop & return:
    - orig PIL (orientation-corrected)
    - crop BGR (numpy)
    - overlay RGB (rect on original)
    - det_info (dict with bbox/conf/labels/json)
    Raises RuntimeError on no detections.
    """
    # Decode once just to get W/H and to generate overlay later
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pil = pil_fix_orientation(pil)
    w, h = pil.size

    url = f"https://detect.roboflow.com/{model_id}"
    params = {
        "api_key": api_key,
        "confidence": conf_threshold,
        "format": "json",
    }
    files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
    r = requests.post(url, params=params, files=files, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Roboflow error {r.status_code}: {r.text[:300]}")

    data = r.json()
    preds = data.get("predictions", []) or []
    if not preds:
        raise RuntimeError("No detections returned by Roboflow.")

    # choose highest-confidence detection (as agreed)
    preds = sorted(preds, key=lambda p: p.get("confidence", 0.0), reverse=True)
    top = preds[0]

    # Center-format to box
    cx = float(top.get("x", w/2))
    cy = float(top.get("y", h/2))
    bw = float(top.get("width", w))
    bh = float(top.get("height", h))
    x1, y1, x2, y2 = crop_box(w, h, cx, cy, bw, bh)

    # Crop
    crop_pil = pil.crop((x1, y1, x2, y2))
    crop_bgr = pil_to_bgr(crop_pil)

    # Overlay
    overlay = pil.copy()
    draw = cv2.rectangle(
        img=rgb_to_bgr(np.array(overlay)),
        pt1=(x1, y1), pt2=(x2, y2),
        color=(0, 255, 0), thickness=max(2, int(round(0.003 * (w + h))))
    )
    overlay_rgb = bgr_to_rgb(draw)

    det_info = {
        "image_w": w, "image_h": h,
        "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "confidence": float(top.get("confidence", 0.0)),
        "class": top.get("class", ""),
        "all_predictions": preds,
        "raw_json": data
    }
    return pil, crop_bgr, overlay_rgb, det_info


# -------------------- 14-feature extractor -------------------- #
def extract_features_14(crop_bgr: np.ndarray) -> Dict[str, float]:
    """
    EXACT 14 features (names must match your training):
      R_norm_p50, a_mean, R_p50, R_p10, RG, S_p50,
      gray_p90, gray_kurt, gray_std, gray_mean,
      B_mean, B_p10, B_p75, G_kurt
    """
    if crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3:
        raise ValueError("Expected BGR color image for feature extraction.")

    H, W, _ = crop_bgr.shape
    # BGR channels
    B = crop_bgr[:, :, 0].astype(np.float32).reshape(-1)
    G = crop_bgr[:, :, 1].astype(np.float32).reshape(-1)
    R = crop_bgr[:, :, 2].astype(np.float32).reshape(-1)

    # RGB stats
    R_p50 = np.percentile(R, 50)
    R_p10 = np.percentile(R, 10)
    RG = float(np.mean(R) / (np.mean(G) + 1e-8))

    # Normalize R by gray mean (like previous pipelines), then median
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32).reshape(-1)
    gray_mean = float(np.mean(gray))
    R_norm = R / (gray_mean + 1e-8)
    R_norm_p50 = float(np.percentile(R_norm, 50))

    # HSV
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1].astype(np.float32).reshape(-1)
    S_p50 = float(np.percentile(S, 50))

    # LAB a*
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    a_channel = lab[:, :, 1].astype(np.float32).reshape(-1)
    a_mean = float(np.mean(a_channel))

    # Gray tails & spread
    gray_p90 = float(np.percentile(gray, 90))
    gray_std = float(np.std(gray, ddof=0))
    # Using Fisher=False to match “excess=False” (raw kurtosis, like earlier)
    gray_kurt = float(kurtosis(gray, fisher=False, bias=False))

    # Blue channel stats
    B_mean = float(np.mean(B))
    B_p10 = float(np.percentile(B, 10))
    B_p75 = float(np.percentile(B, 75))

    # Green kurtosis (raw)
    G_kurt = float(kurtosis(G, fisher=False, bias=False))

    feats = {
        "R_norm_p50": float(R_norm_p50),
        "a_mean": float(a_mean),
        "R_p50": float(R_p50),
        "R_p10": float(R_p10),
        "RG": float(RG),
        "S_p50": float(S_p50),
        "gray_p90": float(gray_p90),
        "gray_kurt": float(gray_kurt),
        "gray_std": float(gray_std),
        "gray_mean": float(gray_mean),
        "B_mean": float(B_mean),
        "B_p10": float(B_p10),
        "B_p75": float(B_p75),
        "G_kurt": float(G_kurt),
    }
    return feats


# -------------------- PMML helpers (strict) -------------------- #
PRED_COL_CANDIDATES = [
    "MLP_PredictedValue",     # SPSS NN default
    "Predicted_hb",
    "predicted_hb",
    "Prediction",
    "PredictedValue",
]


def validate_pmml_inputs(model: Model, feats: Dict[str, float]) -> Optional[str]:
    """Ensure feature names match PMML MiningSchema input names (when available)."""
    try:
        expected = list(model.inputNames)
    except Exception:
        # Some PMMLs may not expose inputNames; skip strict check.
        return None

    provided = list(feats.keys())
    missing = [x for x in expected if x not in feats]
    extras = [x for x in provided if x not in expected]

    issues = []
    if missing:
        issues.append(f"Missing features required by PMML: {missing}")
    if extras:
        issues.append(f"Unexpected features provided: {extras}")

    return " | ".join(issues) if issues else None


def predict_hb_pmml(model: Model, feature_row: Dict[str, float]) -> Tuple[float, Dict[str, Any]]:
    """
    Predict and return (hb_pred, debug_dict).
    We ONLY accept numeric candidate columns in PRED_COL_CANDIDATES.
    If none found, we raise with the visible list of output columns.
    """
    df = pd.DataFrame([feature_row])
    out = model.predict(df)

    chosen_col = None
    y_pred = None
    for c in PRED_COL_CANDIDATES:
        if c in out.columns and np.issubdtype(out[c].dtype, np.number):
            chosen_col = c
            y_pred = float(out[c].iloc[0])
            break

    if y_pred is None:
        raise RuntimeError(
            "Could not find a numeric prediction column in PMML output. "
            f"Tried: {PRED_COL_CANDIDATES}. Output columns: {list(out.columns)}. "
            "Open your PMML and confirm your predicted field name."
        )

    dbg = {
        "pmml_output_columns": list(out.columns),
        "used_column": chosen_col,
        "raw_output_row0": out.iloc[0].to_dict()
    }
    return y_pred, dbg


# -------------------- Streamlit caching -------------------- #
@st.cache_resource(show_spinner=False)
def load_pmml(path_bytes_key: Tuple[str, Optional[int]]) -> Model:
    """
    Cache PMML model. Use (path, size) tuple as the cache key so re-uploads reload.
    If path is an existing file, load from disk; otherwise treat it as raw bytes saved to a temp.
    """
    path_str, byte_len = path_bytes_key
    path = Path(path_str)
    if path.exists():
        return Model.load(str(path))
    else:
        # it points to a temp file we saved for an upload
        return Model.load(path_str)


# -------------------- UI -------------------- #
st.set_page_config(page_title="Anemia Pen", page_icon="🖊️", layout="wide")
st.title("🖊️ Anemia Pen — HB Estimator")

with st.sidebar:
    st.header("Settings")

    # API key for Roboflow
    api_key = st.text_input("Roboflow API Key", type="password", value="", help="Required to crop the conjunctiva.")
    model_id = st.text_input("Roboflow Model ID", value=ROBOFLOW_MODEL_ID)
    conf = st.slider("Detection confidence", 0.05, 0.9, float(DEFAULT_CONFIDENCE), 0.05)

    # RMSE to show a ± band
    rmse_val = st.number_input("Assumed RMSE (g/dL)", min_value=0.1, max_value=5.0, value=DEFAULT_RMSE, step=0.1)

    st.markdown("---")
    st.subheader("Model (PMML)")

    # Discover local defaults
    available_defaults = []
    if DEFAULT_PMML_PRIMARY.exists():
        available_defaults.append(("hemo3.xml (default)", str(DEFAULT_PMML_PRIMARY)))
    if DEFAULT_PMML_FALLBACK.exists():
        available_defaults.append(("hemo.xml", str(DEFAULT_PMML_FALLBACK)))

    # Which default file to start with?
    default_choice_label = "hemo3.xml (default)" if available_defaults and available_defaults[0][0].startswith("hemo3") else (available_defaults[0][0] if available_defaults else "—")

    pmml_choice = st.selectbox(
        "Choose a local PMML file",
        options=[label for (label, _) in available_defaults] + ["(Upload PMML...)"],
        index=0 if available_defaults else len(available_defaults)
    )

    uploaded_pmml = None
    chosen_pmml_path = None

    if pmml_choice == "(Upload PMML...)":
        uploaded_pmml = st.file_uploader("Upload PMML (.xml)", type=["xml"], accept_multiple_files=False, key="pmml_upl")
        if uploaded_pmml is not None:
            # Save upload to a temp file to get a filesystem path for pypmml
            tmp_path = Path(st.session_state.get("pmml_tmp_path", "uploaded_model.xml"))
            tmp_path.write_bytes(uploaded_pmml.getvalue())
            st.session_state["pmml_tmp_path"] = str(tmp_path)
            chosen_pmml_path = str(tmp_path)
            st.caption(f"Using uploaded PMML: {tmp_path.name}")
        else:
            if DEFAULT_PMML_PRIMARY.exists():
                chosen_pmml_path = str(DEFAULT_PMML_PRIMARY)
                st.caption("No upload yet → using hemo3.xml by default.")
            elif DEFAULT_PMML_FALLBACK.exists():
                chosen_pmml_path = str(DEFAULT_PMML_FALLBACK)
                st.caption("No upload yet → using hemo.xml (fallback).")
    else:
        # Map selection back to path
        label_to_path = {label: path for (label, path) in available_defaults}
        chosen_pmml_path = label_to_path.get(pmml_choice)

    if not chosen_pmml_path:
        st.error("No PMML selected. Please upload a PMML or place hemo3.xml/hemo.xml in the app folder.")
        st.stop()

    # Load/cached PMML
    pmml_size = Path(chosen_pmml_path).stat().st_size if Path(chosen_pmml_path).exists() else None
    MODEL = load_pmml((chosen_pmml_path, pmml_size))
    st.success(f"Loaded PMML: {Path(chosen_pmml_path).name}")

    with st.expander("Model inputs (from PMML)", expanded=False):
        try:
            st.write(list(MODEL.inputNames))
        except Exception:
            st.write("Model did not expose inputNames (ok). Make sure your 14 feature names match.")

st.markdown("#### 1) Provide an eye photo")

# Camera input (mobile-friendly) OR file uploader
col_cam, col_up = st.columns(2)
with col_cam:
    cam_img = st.camera_input("Take a photo (mobile works best)", key="cam", help="Use this on iPhone/Android.")
with col_up:
    up_img = st.file_uploader("...or upload an image", type=["jpg", "jpeg", "png"], accept_multiple_files=False, key="uploader")

src = cam_img or up_img
if not src:
    st.info("Take a photo or upload a file to begin.")
    st.stop()

# Read, fix orientation, convert to JPEG bytes for Roboflow
raw_bytes = src.getvalue()
pil = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
pil = pil_fix_orientation(pil)
jpeg_bytes = pil_to_jpeg_bytes(pil)

st.markdown("#### 2) Crop conjunctiva with Roboflow")
if not api_key:
    st.warning("Enter your Roboflow API key in the sidebar to crop.")
    st.stop()

try:
    orig_pil, crop_bgr, overlay_rgb, det_info = roboflow_detect_and_crop(
        api_key=api_key,
        model_id=model_id,
        image_bytes=jpeg_bytes,
        conf_threshold=conf
    )
except Exception as e:
    st.error(f"Roboflow error: {e}")
    st.stop()

c1, c2 = st.columns(2)
with c1:
    st.image(orig_pil, caption="Original (orientation-corrected)", use_container_width=True)
with c2:
    st.image(overlay_rgb, caption=f"Detection overlay • conf={det_info.get('confidence', 0):.3f}", use_container_width=True)

st.markdown("#### 3) Extract features (14)")
feats = extract_features_14(crop_bgr)

# Basic NaN/inf checks
vals = list(feats.values())
if any([pd.isna(v) or np.isinf(v) for v in vals]):
    st.error("Feature extraction produced NaN/Inf values. Try another photo.")
    st.json(feats)
    st.stop()

feat_df = pd.DataFrame([feats]).T.reset_index()
feat_df.columns = ["feature", "value"]
st.dataframe(feat_df, height=420)

st.markdown("#### 4) Predict hemoglobin (PMML)")

# Strict schema check vs PMML inputs (when available)
schema_err = validate_pmml_inputs(MODEL, feats)
if schema_err:
    st.error(f"PMML input mismatch: {schema_err}")
    with st.expander("Provided 14 features"):
        st.json(feats)
    st.stop()

try:
    pred_hb, dbg = predict_hb_pmml(MODEL, feats)
except Exception as e:
    st.error(f"PMML prediction failed: {e}")
    with st.expander("Provided 14 features"):
        st.json(feats)
    st.stop()

lo = pred_hb - 1.96 * rmse_val
hi = pred_hb + 1.96 * rmse_val

st.success(f"**Estimated Hb: {pred_hb:.2f} g/dL**  \n95% band (±1.96×RMSE @ {rmse_val:.1f}): **{lo:.2f} – {hi:.2f}**")

c3, c4 = st.columns(2)
with c3:
    st.image(bgr_to_rgb(crop_bgr), caption="Conjunctiva crop", use_container_width=True)
with c4:
    with st.expander("PMML debug", expanded=False):
        st.json(dbg)
        st.caption("If the wrong column is used or inputs mismatch, predictions can collapse to a constant. This view shows the column we actually used.")

st.caption("Tip: If different images still give identical predictions, open the PMML debug expander. Make sure `used_column` is your real prediction field (e.g., `MLP_PredictedValue`) and the 14 feature names match the PMML MiningSchema exactly.")
