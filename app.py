# app.py — Anemia Pen (PMML) — full file

from __future__ import annotations
import io
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import time, base64, json
import pandas as pd
import cv2
import requests
from PIL import Image, ImageOps
import streamlit as st

# PMML loader
from pypmml import Model  # make sure pypmml is in requirements.txt

# -------------------- Config -------------------- #
DEFAULT_PMML_NAME = "hemo3.xml"  # <-- auto-selected on load
PMML_GLOB = "*.xml"

ROBOFLOW_MODEL_ID = "eye-conjunctiva-detector/2"
ROBOFLOW_API_URL = "https://detect.roboflow.com"
DEFAULT_API_KEY = "jMhyBQxeQvj69nttV0mN"

CONF_THRESHOLD = 0.25  # ignore detections below this
PRED_CLASS = None      # or "conjunctiva" if your model has multiple classes

# -------------------- Small utils -------------------- #
def exif_fix(pil_img: Image.Image) -> Image.Image:
    """Honor EXIF orientation (prevents 90° rotated uploads from phones)."""
    try:
        return ImageOps.exif_transpose(pil_img)
    except Exception:
        return pil_img

def pil_to_jpeg_bytes(pil_img: Image.Image, quality: int = 95) -> bytes:
    """Encode PIL image to JPEG bytes."""
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

def pil_to_cv_bgr(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL -> OpenCV BGR."""
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def cv_bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR -> PIL."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def draw_box(img_bgr: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    out = img_bgr.copy()
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 3)
    return out

def downscale_if_needed(pil_img, max_side: int = 1600):
    """Constrain largest side to max_side while keeping aspect ratio."""
    w, h = pil_img.size
    m = max(w, h)
    if m <= max_side:
        return pil_img
    scale = max_side / float(m)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return pil_img.resize((new_w, new_h), Image.LANCZOS)



def safe_kurtosis(x: np.ndarray) -> float:
    """Fisher kurtosis (excess), bias=True. No SciPy needed."""
    x = x.astype(np.float64).ravel()
    n = x.size
    if n < 4:
        return float("nan")
    m = np.mean(x)
    s2 = np.mean((x - m) ** 2)
    if s2 <= 0:
        return 0.0
    m4 = np.mean((x - m) ** 4)
    # excess kurtosis (Fisher) = μ4/σ4 - 3
    return float(m4 / (s2 ** 2) - 3.0)

# -------------------- Feature extraction (14) -------------------- #
FEATURE_ORDER: List[str] = [
    "R_norm_p50",   # median of per-pixel (R / (G+eps))
    "a_mean",       # Lab a* mean
    "R_p50",        # red median
    "R_p10",        # red 10th pct
    "RG",           # mean(R) / mean(G)
    "S_p50",        # HSV S median (0-255)
    "gray_p90",     # gray 90th pct
    "gray_kurt",    # gray kurtosis (excess)
    "gray_std",     # gray std
    "gray_mean",    # gray mean
    "B_mean",       # blue mean
    "B_p10",        # blue 10th pct
    "B_p75",        # blue 75th pct
    "G_kurt",       # green kurtosis (excess)
]

def extract_14_features(crop_bgr: np.ndarray) -> Dict[str, float]:
    b, g, r = cv2.split(crop_bgr.astype(np.float64))
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV).astype(np.float64)
    h, s, v = cv2.split(hsv)
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    lch, a, b_lab = cv2.split(lab)

    eps = 1e-6
    r_over_g = r / (g + eps)

    feats = {
        "R_norm_p50": float(np.percentile(r_over_g, 50)),
        "a_mean":     float(np.mean(a)),
        "R_p50":      float(np.percentile(r, 50)),
        "R_p10":      float(np.percentile(r, 10)),
        "RG":         float(np.mean(r) / (np.mean(g) + eps)),
        "S_p50":      float(np.percentile(s, 50)),
        "gray_p90":   float(np.percentile(gray, 90)),
        "gray_kurt":  float(safe_kurtosis(gray)),
        "gray_std":   float(np.std(gray, ddof=0)),
        "gray_mean":  float(np.mean(gray)),
        "B_mean":     float(np.mean(b)),
        "B_p10":      float(np.percentile(b, 10)),
        "B_p75":      float(np.percentile(b, 75)),
        "G_kurt":     float(safe_kurtosis(g)),
    }
    # hard cast to clean python floats
    return {k: float(v) if v is not None else np.nan for k, v in feats.items()}

# -------------------- Roboflow detection -------------------- #
def roboflow_detect_and_crop(
    pil_img: Image.Image,
    api_key: str,
    model_id: str,
    conf: float = CONF_THRESHOLD,
    pred_class: Optional[str] = PRED_CLASS,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Send image to Roboflow with multiple fallbacks; return (crop_bgr, overlay_bgr, det_info).
    Robust against 5xx via alternate upload methods and endpoints.
    """
    # 1) Normalize orientation + size (iPhone originals can be huge & EXIF-rotated)
    pil_fixed = exif_fix(pil_img)
    pil_small = downscale_if_needed(pil_fixed, max_side=1600)

    jpeg_bytes = pil_to_jpeg_bytes(pil_small, quality=92)
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    datauri = f"image=data:image/jpeg;base64,{b64}"

    attempts_log = []

    def _try(url, payload_kind, **kwargs):
        params = {"api_key": api_key, "confidence": f"{conf:.2f}"}
        try:
            resp = requests.post(url, params=params, timeout=45, **kwargs)
            attempts_log.append({
                "kind": payload_kind,
                "url": url,
                "status": resp.status_code,
                "body": resp.text[:500],
            })
            if resp.status_code >= 500:
                # server error: move to next fallback
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            attempts_log.append({
                "kind": payload_kind,
                "url": url,
                "error": str(e),
            })
            return None

    # Try detect.roboflow.com first
    base1 = f"https://detect.roboflow.com/{model_id}"
    out = (
        _try(base1, "octet-stream", headers={"Content-Type": "application/octet-stream"}, data=jpeg_bytes)
        or _try(base1, "multipart-file", files={"file": ("image.jpg", jpeg_bytes, "image/jpeg")})
        or _try(base1, "form-base64", headers={"Content-Type": "application/x-www-form-urlencoded"}, data=datauri)
    )

    # Fallback to serverless if needed
    if out is None:
        base2 = f"https://serverless.roboflow.com/{model_id}"
        out = (
            _try(base2, "serverless-form-base64", headers={"Content-Type": "application/x-www-form-urlencoded"}, data=datauri)
            or _try(base2, "serverless-multipart-file", files={"file": ("image.jpg", jpeg_bytes, "image/jpeg")})
        )

    if out is None:
        # Everything failed – raise with a concise summary
        brief = json.dumps(attempts_log, ensure_ascii=False)[:1200]
        raise RuntimeError(f"All Roboflow calls failed. Attempts: {brief}")

    preds = out.get("predictions", []) or []
    if pred_class:
        preds = [p for p in preds if p.get("class") == pred_class]

    if not preds:
        brief = json.dumps(out, ensure_ascii=False)[:600]
        raise RuntimeError(f"No detection above threshold (conf>={conf}). Raw: {brief}")

    # Pick highest-confidence detection
    best = max(preds, key=lambda p: p.get("confidence", 0.0))

    # Roboflow x,y are centers in the coords of the SENT image
    W, H = pil_small.size
    cx, cy = float(best["x"]), float(best["y"])
    bw, bh = float(best["width"]), float(best["height"])
    x0 = int(round(cx - bw / 2))
    y0 = int(round(cy - bh / 2))
    w = int(round(bw))
    h = int(round(bh))

    # Clamp to bounds
    x0 = max(0, min(x0, W - 1))
    y0 = max(0, min(y0, H - 1))
    w = max(1, min(w, W - x0))
    h = max(1, min(h, H - y0))

    full_bgr = pil_to_cv_bgr(pil_small)
    crop_bgr = full_bgr[y0:y0+h, x0:x0+w].copy()
    overlay_bgr = draw_box(full_bgr, (x0, y0, w, h))

    det_info = {
        "attempts": attempts_log,
        "box_xywh": {"x": x0, "y": y0, "w": w, "h": h},
        "confidence": float(best.get("confidence", 0.0)),
        "pred_class": best.get("class", None),
        "sent_size": {"W": W, "H": H},
        "raw_pred": best,
    }
    return crop_bgr, overlay_bgr, det_info

with st.expander("Detection debug", expanded=False):
    st.write(det_info)  # this will include every attempt with status/body


# -------------------- PMML: strict input build + predict -------------------- #
def get_pmml_input_names(model: Model) -> List[str]:
    # pypmml usually has .inputNames; if not, fall back to hard-coded 14
    try:
        names = list(model.inputNames)  # type: ignore[attr-defined]
        if names:
            return names
    except Exception:
        pass
    return FEATURE_ORDER[:]  # our 14

def predict_pmml_strict(pmml_model: Model, feats: Dict[str, float], pmml_input_names: List[str]):
    """
    Order columns by PMML schema, cast to float64, guard against all-NaN,
    then predict and pick the numeric output.
    Returns: (y_hat: float, dbg: dict)
    """
    # 1) order & cast
    ordered = []
    for nm in pmml_input_names:
        val = feats.get(nm, np.nan)
        try:
            ordered.append(float(val))
        except Exception:
            ordered.append(np.nan)

    df = pd.DataFrame([ordered], columns=pmml_input_names).apply(pd.to_numeric, errors="coerce").astype("float64")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    nan_ct = int(df.isna().sum(axis=1).iloc[0])
    if nan_ct == len(pmml_input_names):
        raise RuntimeError("All PMML inputs are NaN. Check feature names / types.")

    out = pmml_model.predict(df)

    # choose output column
    candidates = ["predicted_hb", "MLP_PredictedValue", "PredictedValue"]
    used = next((c for c in candidates if c in out.columns), None)
    if used is None:
        num_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
        if not num_cols:
            raise RuntimeError(f"PMML output has no numeric columns: {list(out.columns)}")
        used = num_cols[0]

    y = float(out.iloc[0][used])

    dbg = {
        "pmml_output_columns": list(out.columns),
        "used_column": used,
        "raw_output_row0": out.iloc[0].to_dict(),
        "input_row_model_order": {k: (None if pd.isna(v) else float(v)) for k, v in zip(df.columns, df.iloc[0])},
        "input_dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "input_nan_count": nan_ct,
    }
    return y, dbg

# -------------------- Streamlit UI -------------------- #
st.set_page_config(page_title="Anemia Pen (PMML)", layout="wide")

st.title("Anemia Pen – PMML Inference")

# Sidebar: API key & PMML picker
with st.sidebar:
    st.markdown("### Settings")

    api_key = st.text_input("Roboflow API Key", value=DEFAULT_API_KEY, type="password")
    rmse = st.number_input("RMSE (for ± interval)", value=1.70, min_value=0.0, step=0.05, format="%.2f")

    # Discover PMMLs in repo root; default to hemo3.xml if present
    pmml_files = sorted([p.name for p in Path(".").glob(PMML_GLOB)])
    if not pmml_files:
        st.error("No PMML (*.xml) found in the app folder.")
        st.stop()

    default_idx = 0
    if DEFAULT_PMML_NAME in pmml_files:
        default_idx = pmml_files.index(DEFAULT_PMML_NAME)

    pmml_choice = st.selectbox("Select PMML model", pmml_files, index=default_idx)

    st.caption("Tip: On iPhone, you can tap *Take Photo* in the uploader (use JPEG/PNG).")

# Load PMML
pmml_path = Path(pmml_choice)
try:
    pmml_model = Model.load(str(pmml_path))
except Exception as e:
    st.error(f"Failed to load PMML `{pmml_path}`: {e}")
    st.stop()

pmml_inputs = get_pmml_input_names(pmml_model)

st.write(f"**Loaded PMML:** `{pmml_path.name}`")
with st.expander("PMML input schema", expanded=False):
    st.json(pmml_inputs)

# Uploader (mobile camera should offer Take Photo when using JPG/PNG)
uploaded = st.file_uploader(
    "Upload an eye photo (full eye; conjunctiva visible)",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    accept_multiple_files=False,
)

if not uploaded:
    st.info("Upload a photo to run detection → crop → features → PMML prediction.")
    st.stop()

# Read image, EXIF-fix
try:
    pil_orig = Image.open(uploaded).convert("RGB")
    pil_orig = exif_fix(pil_orig)
except Exception as e:
    st.error(f"Could not read image: {e}")
    st.stop()

# Run detection & crop
try:
    crop_bgr, overlay_bgr, det_info = roboflow_detect_and_crop(
        pil_orig, api_key=api_key, model_id=ROBOFLOW_MODEL_ID, conf=CONF_THRESHOLD, pred_class=PRED_CLASS
    )
except Exception as e:
    st.error(f"Detection/Cropping failed: {e}")
    st.stop()

# Extract features
feats = extract_14_features(crop_bgr)

# Layout
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Original**")
    st.image(pil_orig, use_container_width=True)
with col2:
    st.markdown("**Detection**")
    st.image(cv_bgr_to_pil(overlay_bgr), use_container_width=True)
with col3:
    st.markdown("**Crop used for features**")
    st.image(cv_bgr_to_pil(crop_bgr), use_container_width=True)

# Show features
feat_df = pd.DataFrame([feats], columns=FEATURE_ORDER).T.reset_index()
feat_df.columns = ["feature", "value"]
st.markdown("### Extracted Parameters")
st.dataframe(feat_df, height=420, use_container_width=True)

# Predict via PMML (STRICT)
try:
    y_hat, pmml_dbg = predict_pmml_strict(pmml_model, feats, pmml_inputs)
except Exception as e:
    st.error(f"PMML prediction failed: {e}")
    st.stop()

# Show prediction + interval
lo = y_hat - 1.96 * rmse
hi = y_hat + 1.96 * rmse
st.markdown("### Hemoglobin Estimate")
st.metric(label="Predicted Hb (g/dL)", value=f"{y_hat:.2f}", delta=None)
st.caption(f"Approx. 95% interval: **{lo:.2f}** to **{hi:.2f}**  (RMSE={rmse:.2f})")

# Debug block (helps catch 'constant output' issues)
with st.expander("PMML debug", expanded=False):
    st.json({
        "loaded_pmml": str(pmml_path),
        "pmml_input_names": pmml_inputs,
        **pmml_dbg
    })
