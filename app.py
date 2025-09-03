# app.py
# Streamlit app for Anemia Pen — PMML model, Roboflow crop, feature extraction, prediction
# - Defaults to hemo3.xml but lets you switch models from the sidebar
# - Supports camera capture (mobile) or file upload
# - Uses pure requests for Roboflow serverless detection
# - Shows crop, detection overlay, predicted Hb, CI (±RMSE), and extracted features

from __future__ import annotations

import io
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import cv2
import requests
from PIL import Image, ExifTags
from scipy.stats import kurtosis
import streamlit as st
from pypmml import Model


# -------------------- Settings -------------------- #
DEFAULT_PMML_PATH = Path("hemo3.xml")          # default model (auto-selected)
DEFAULT_MODEL_ID = "eye-conjunctiva-detector/2"
DEFAULT_CONF = 0.25
DEFAULT_RMSE = 1.7                             # used for ±RMSE interval
SAVE_CROPS_DIR = Path("cropped_images_app")    # optional save dir for crops (created if needed)

# -------------------- Utilities -------------------- #

def exif_autorotate(pil_img: Image.Image) -> Image.Image:
    """Auto-rotate a PIL image using EXIF orientation if present."""
    try:
        exif = pil_img.getexif()
        if not exif:
            return pil_img
        orientation_key = None
        for k, v in ExifTags.TAGS.items():
            if v == 'Orientation':
                orientation_key = k
                break
        if orientation_key is None or orientation_key not in exif:
            return pil_img

        orientation = exif.get(orientation_key)
        if orientation == 3:
            pil_img = pil_img.rotate(180, expand=True)
        elif orientation == 6:
            pil_img = pil_img.rotate(270, expand=True)
        elif orientation == 8:
            pil_img = pil_img.rotate(90, expand=True)
    except Exception:
        # If anything goes wrong, just return original
        pass
    return pil_img


def pil_to_jpg_bytes(pil_img: Image.Image, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def ensure_rgb_pil(data: bytes) -> Image.Image:
    """Load bytes -> PIL (RGB) and EXIF auto-rotate."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = exif_autorotate(img)
    return img


def pil_to_cv2_bgr(pil_img: Image.Image) -> np.ndarray:
    """PIL (RGB) -> OpenCV BGR uint8."""
    arr = np.array(pil_img)  # RGB
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv2_bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def draw_box_overlay(bgr_img: np.ndarray, box: Tuple[int, int, int, int], color=(0, 255, 0), thickness=2) -> np.ndarray:
    x1, y1, x2, y2 = box
    out = bgr_img.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out


def safe_clip_box(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w - 1, x + w)
    y2 = min(img_h - 1, y + h)
    return x1, y1, x2, y2


def list_pmml_files() -> List[Path]:
    """Find PMML XMLs in repo root (you can add more dirs if needed)."""
    root_xmls = sorted(Path(".").glob("*.xml"))
    # Optionally include a 'models' folder:
    models_dir = Path("models")
    if models_dir.exists():
        root_xmls += sorted(models_dir.glob("*.xml"))
    # unique by name/path order preserved
    seen = set()
    uniq = []
    for p in root_xmls:
        if p.resolve() not in seen:
            uniq.append(p)
            seen.add(p.resolve())
    return uniq


# -------------------- Roboflow detection -------------------- #

@dataclass
class Detection:
    cls: str
    conf: float
    x: float
    y: float
    w: float
    h: float


def roboflow_detect_crop(pil_img: Image.Image, api_key: str, model_id: str, conf_thresh: float = 0.25
                         ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Send image (after EXIF auto-rotate) to Roboflow serverless detection, pick the highest-confidence
    'conjunctiva' box >= conf_thresh, return (crop_bgr, overlay_rgb, det_info).
    """
    url = f"https://detect.roboflow.com/{model_id}"
    params = {"api_key": api_key, "confidence": str(conf_thresh)}
    files = {"file": ("image.jpg", pil_to_jpg_bytes(pil_img), "application/octet-stream")}

    try:
        resp = requests.post(url, params=params, files=files, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        st.error(f"Roboflow request failed: {e}")
        return None, None, None

    try:
        data = resp.json()
    except Exception:
        st.error("Roboflow returned a non-JSON response.")
        return None, None, None

    preds = data.get("predictions", [])
    if not preds:
        st.warning("No detections returned.")
        return None, None, None

    # pick highest-conf conjunctiva (or highest overall if class field missing)
    dets = []
    for p in preds:
        cls = p.get("class", "")
        conf = float(p.get("confidence", 0.0))
        if cls and cls.lower() != "conjunctiva":
            # keep only conjunctiva if class labels exist
            continue
        dets.append(Detection(
            cls=cls or "conjunctiva",
            conf=conf,
            x=float(p.get("x", 0.0)),
            y=float(p.get("y", 0.0)),
            w=float(p.get("width", 0.0)),
            h=float(p.get("height", 0.0)),
        ))

    if not dets:
        # If nothing labeled as conjunctiva, fallback to highest confidence of any class
        for p in preds:
            dets.append(Detection(
                cls=p.get("class", "object"),
                conf=float(p.get("confidence", 0.0)),
                x=float(p.get("x", 0.0)),
                y=float(p.get("y", 0.0)),
                w=float(p.get("width", 0.0)),
                h=float(p.get("height", 0.0)),
            ))

    dets.sort(key=lambda d: d.conf, reverse=True)
    best = dets[0]

    # convert cx,cy,w,h to int box (x1,y1,x2,y2)
    rgb_w, rgb_h = pil_img.size
    cx, cy, bw, bh = best.x, best.y, best.w, best.h
    x1f = cx - bw / 2.0
    y1f = cy - bh / 2.0
    x2f = cx + bw / 2.0
    y2f = cy + bh / 2.0

    x1 = int(round(x1f))
    y1 = int(round(y1f))
    x2 = int(round(x2f))
    y2 = int(round(y2f))

    # clip to image bounds
    x1 = max(0, min(x1, rgb_w - 1))
    y1 = max(0, min(y1, rgb_h - 1))
    x2 = max(0, min(x2, rgb_w - 1))
    y2 = max(0, min(y2, rgb_h - 1))

    if x2 <= x1 or y2 <= y1:
        st.error("Invalid crop box after detection.")
        return None, None, None

    # Make BGR copies for drawing / cropping
    bgr = pil_to_cv2_bgr(pil_img)
    crop_bgr = bgr[y1:y2, x1:x2].copy()
    overlay_bgr = draw_box_overlay(bgr, (x1, y1, x2, y2))
    overlay_rgb = cv2_bgr_to_rgb(overlay_bgr)

    det_info = {
        "class": best.cls,
        "confidence": best.conf,
        "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "image_w": rgb_w,
        "image_h": rgb_h,
    }
    return crop_bgr, overlay_rgb, det_info


# -------------------- Feature extraction (14 features) -------------------- #

def channel_stats(c: np.ndarray) -> Dict[str, float]:
    """Return summary stats for a single channel (uint8)."""
    c_flat = c.reshape(-1).astype(np.float32)
    c_flat = c_flat[~np.isnan(c_flat)]
    p10 = float(np.percentile(c_flat, 10))
    p50 = float(np.percentile(c_flat, 50))
    p75 = float(np.percentile(c_flat, 75))
    mean = float(c_flat.mean())
    std = float(c_flat.std(ddof=0))
    mn = float(c_flat.min())
    mx = float(c_flat.max())
    # For kurtosis we keep fisher=True (excess kurtosis) to match typical stats
    krt = float(kurtosis(c_flat, fisher=True, bias=False)) if c_flat.size > 3 else 0.0
    return {
        "mean": mean, "std": std, "min": mn, "p10": p10, "p50": p50, "p75": p75, "p90": float(np.percentile(c_flat, 90)), "max": mx,
        "skew": 0.0,  # unused in the selected 14 features, placeholder
        "kurt": krt,
    }


def extract_features_14(bgr: np.ndarray) -> Dict[str, float]:
    """
    Compute exactly these 14 features:
    R_norm_p50, a_mean, R_p50, R_p10, RG, S_p50,
    gray_p90, gray_kurt, gray_std, gray_mean,
    B_mean, B_p10, B_p75, G_kurt
    """
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Expected BGR image")

    # Basic channels
    b = bgr[:, :, 0].astype(np.uint8)
    g = bgr[:, :, 1].astype(np.uint8)
    r = bgr[:, :, 2].astype(np.uint8)

    # Grayscale
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # HSV for saturation
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32)

    # Lab for a*
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    a_lab = lab[:, :, 1].astype(np.float32)  # OpenCV: 0..255, 128 ~ 0
    a_mean = float(a_lab.mean() - 128.0)

    # Stats
    r_stats = channel_stats(r)
    g_stats = channel_stats(g)
    b_stats = channel_stats(b)

    gray_flat = gray.reshape(-1).astype(np.float32)
    gray_mean = float(gray_flat.mean())
    gray_std = float(gray_flat.std(ddof=0))
    gray_p90 = float(np.percentile(gray_flat, 90))
    gray_kurt = float(kurtosis(gray_flat, fisher=True, bias=False)) if gray_flat.size > 3 else 0.0

    # Ratios
    r_mean = r_stats["mean"]
    g_mean = g_stats["mean"]
    RG = float(r_mean / (g_mean + 1e-6))

    # Normalized red (R / (R+G)), median
    r_plus_g = r.astype(np.float32) + g.astype(np.float32) + 1e-6
    r_norm = (r.astype(np.float32) / r_plus_g).reshape(-1)
    R_norm_p50 = float(np.percentile(r_norm, 50))

    # Saturation median
    S_p50 = float(np.percentile(s.reshape(-1), 50))

    feats = {
        "R_norm_p50": R_norm_p50,
        "a_mean": a_mean,
        "R_p50": r_stats["p50"],
        "R_p10": r_stats["p10"],
        "RG": RG,
        "S_p50": S_p50,
        "gray_p90": gray_p90,
        "gray_kurt": gray_kurt,
        "gray_std": gray_std,
        "gray_mean": gray_mean,
        "B_mean": b_stats["mean"],
        "B_p10": b_stats["p10"],
        "B_p75": b_stats["p75"],
        "G_kurt": g_stats["kurt"],
    }
    return feats


# -------------------- PMML load & predict -------------------- #

@st.cache_resource(show_spinner=False)
def load_pmml(path_str: str) -> Model:
    return Model.load(path_str)


def predict_hb_pmml(model: Model, feature_row: Dict[str, float]) -> Tuple[float, Dict[str, Any]]:
    """
    Run PMML model on a single-row feature dict.
    Returns (pred_value, debug_info).
    """
    df = pd.DataFrame([feature_row])

    out = model.predict(df)
    # Try common output column names:
    candidates = [
        "predicted_hb", "Predicted_hb", "Prediction", "prediction", "PredictedValue",
        "MLP_PredictedValue", "hb"
    ]
    y_pred = None
    chosen_col = None
    for c in candidates:
        if c in out.columns:
            y_pred = float(out[c].iloc[0])
            chosen_col = c
            break
    if y_pred is None:
        # fallback: pick the last numeric column
        num_cols = [c for c in out.columns if np.issubdtype(out[c].dtype, np.number)]
        if not num_cols:
            raise RuntimeError(f"PMML output has no numeric columns: {list(out.columns)}")
        chosen_col = num_cols[-1]
        y_pred = float(out[chosen_col].iloc[0])

    dbg = {"pmml_output_columns": list(out.columns), "used_column": chosen_col}
    return y_pred, dbg


# -------------------- Streamlit UI -------------------- #

st.set_page_config(page_title="Anemia Pen (PMML)", layout="wide")
st.title("🖊️ Anemia Pen — PMML Model")

with st.sidebar:
    st.header("Settings")

    # Roboflow API key
    api_key = st.text_input(
        "Roboflow API Key",
        value=os.environ.get("ROBOFLOW_API_KEY", ""),
        type="password",
        help="Required to run the detector. Kept only in your session.",
    )
    model_id = st.text_input("Roboflow Model ID", value=DEFAULT_MODEL_ID, help="e.g., eye-conjunctiva-detector/2")
    conf_thresh = st.slider("Detection confidence threshold", 0.0, 1.0, DEFAULT_CONF, 0.01)

    st.markdown("---")
    # PMML selection (defaults to hemo3.xml if present)
    pmml_files = list_pmml_files()
    if not pmml_files:
        st.error("No PMML (.xml) models found in the repo. Please add `hemo3.xml`.")
        pmml_choice = None
    else:
        default_idx = next((i for i, p in enumerate(pmml_files) if p.name.lower() == "hemo3.xml"), 0)
        pmml_choice = st.selectbox(
            "PMML model file",
            options=pmml_files,
            index=default_idx,
            format_func=lambda p: p.name,
            help="Default is hemo3.xml; switch here any time."
        )

    rmse = st.number_input("Display uncertainty (±RMSE, g/dL)", min_value=0.0, value=DEFAULT_RMSE, step=0.1)

    save_crops = st.checkbox("Save crops to disk", value=False, help=f"Saves to `{SAVE_CROPS_DIR}`")

# Load selected model
MODEL: Optional[Model] = None
if pmml_choice is not None:
    try:
        MODEL = load_pmml(str(pmml_choice))
        st.sidebar.success(f"Loaded PMML: {pmml_choice.name}")
    except Exception as e:
        st.sidebar.error(f"Failed to load {pmml_choice.name}: {e}")

colL, colR = st.columns([1, 1])

with colL:
    st.subheader("Input")
    # Camera capture (mobile-friendly) & file uploader; camera takes precedence if used
    cam_img = st.camera_input("Take a photo (mobile)", help="Use your phone camera here")
    upl = st.file_uploader("…or upload an image", type=["jpg", "jpeg", "png"])

    # Keep whichever the user provided (camera first)
    uploaded_bytes: Optional[bytes] = None
    src_name: str = ""
    if cam_img is not None:
        uploaded_bytes = cam_img.getvalue()
        src_name = cam_img.name or "camera.jpg"
    elif upl is not None:
        uploaded_bytes = upl.read()
        src_name = upl.name

    if uploaded_bytes:
        try:
            pil_orig = ensure_rgb_pil(uploaded_bytes)
            st.image(pil_orig, caption=f"Original ({src_name})", use_container_width=True)
        except Exception as e:
            st.error(f"Failed to read image: {e}")
            uploaded_bytes = None

with colR:
    st.subheader("Prediction")

    if st.button("Estimate Hb", type="primary", disabled=(uploaded_bytes is None or MODEL is None or not api_key.strip())):
        if uploaded_bytes is None:
            st.warning("Please upload or capture an image.")
        elif MODEL is None:
            st.warning("PMML model is not loaded.")
        elif not api_key.strip():
            st.warning("Please enter your Roboflow API key in the sidebar.")
        else:
            with st.spinner("Detecting conjunctiva and extracting features…"):
                pil_img = ensure_rgb_pil(uploaded_bytes)

                crop_bgr, overlay_rgb, det_info = roboflow_detect_crop(
                    pil_img=pil_img,
                    api_key=api_key.strip(),
                    model_id=model_id.strip(),
                    conf_thresh=conf_thresh
                )

                if crop_bgr is None:
                    st.stop()

                # Show detection overlay & crop
                st.image(overlay_rgb, caption="Detection overlay", use_container_width=True)
                st.image(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB), caption="Conjunctiva crop", use_container_width=True)

                # Optionally save crop
                if save_crops:
                    SAVE_CROPS_DIR.mkdir(parents=True, exist_ok=True)
                    out_path = SAVE_CROPS_DIR / f"crop_{Path(src_name).stem}.jpg"
                    cv2.imwrite(str(out_path), crop_bgr[:, :, ::-1])  # write RGB as BGR order fix
                    st.caption(f"Saved: {out_path}")

                # Extract features (14)
                feats = extract_features_14(crop_bgr)
                feat_df = pd.DataFrame([feats]).T
                feat_df.columns = ["value"]

            # Predict
            with st.spinner("Running PMML model…"):
                try:
                    pred_hb, dbg = predict_hb_pmml(MODEL, feats)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.json({"features": feats})
                    st.stop()

            # Show results
            st.success(f"Estimated Hb: **{pred_hb:.2f} g/dL**")
            lo = pred_hb - rmse
            hi = pred_hb + rmse
            st.caption(f"Uncertainty band (±RMSE={rmse:.2f}): **[{lo:.2f}, {hi:.2f}] g/dL**")

            # Detection info
            if det_info:
                st.write(
                    f"Detection: class={det_info['class']}, conf={det_info['confidence']:.2f}, "
                    f"box={det_info['box']}"
                )

            # Extracted parameters
            st.markdown("**Extracted Parameters**")
            st.dataframe(feat_df, use_container_width=True, height=420)

            # Debug output (optional)
            with st.expander("PMML debug"):
                st.json(dbg)

# Footer hint
st.markdown("---")
st.caption("Tip: If camera capture doesn’t trigger on iPhone, open this page in Safari and allow camera access.")
