# app.py
# Streamlit app for "Anemia Pen" — REST-based Roboflow detection (no inference-sdk).
# - Upload or camera-capture an eye photo
# - Call Roboflow Hosted REST API to detect conjunctiva and crop it
# - Extract features (same pipeline as training)
# - Load packed model (step5_train_and_pack.py) and estimate Hb (+/- interval)
# - Show crop, numbers, and all extracted parameters
PACK_VERSION = "subset-mlp-1"

import os
import io
import json
import math
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import cv2
import requests
import streamlit as st
from PIL import Image
import joblib

# ------------------------
# CONFIG / PATHS
# ------------------------
PACK_PATH = Path("outputs/models/anemia_pen_pack.joblib")
META_PATH = Path("outputs/models/anemia_pen_pack_meta.json")

# Roboflow (you can override in the sidebar or via env vars)
DEFAULT_ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "jMhyBQxeQvj69nttV0mN")
DEFAULT_ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "eye-conjunctiva-detector/2")
DEFAULT_CONF_THRESH = 0.30
DEFAULT_OVERLAP = 0.30

# ------------------------
# STREAMLIT PAGE SETUP
# ------------------------
st.set_page_config(page_title="Anemia Pen", page_icon="🩸", layout="wide")
st.title("🩸 Anemia Pen — Prototype")
st.caption("DEMO")

# ------------------------
# LITTLE UTILS
# ------------------------
def draw_box_bgr(img_bgr: np.ndarray, x1, y1, x2, y2, color=(0, 255, 0), thickness=2):
    cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return img_bgr

def clamp_box(x1, y1, x2, y2, W, H):
    x1 = int(max(0, min(W - 1, x1)))
    y1 = int(max(0, min(H - 1, y1)))
    x2 = int(max(0, min(W, x2)))
    y2 = int(max(0, min(H, y2)))
    if x2 <= x1: x2 = min(W, x1 + 1)
    if y2 <= y1: y2 = min(H, y1 + 1)
    return x1, y1, x2, y2

def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# ------------------------
# PACK & META LOADERS
# ------------------------
@st.cache_resource(show_spinner=False)
def load_pack():
    if not PACK_PATH.exists():
        raise FileNotFoundError(
            f"Packed model not found at {PACK_PATH}. "
            "Run `python step5_train_and_pack.py` locally and push the files."
        )
    pack = joblib.load(PACK_PATH)
    required = ["vt", "corr_keep_mask", "feat_all", "estimator", "iso", "mask_config"]
    for k in required:
        if k not in pack:
            raise ValueError(f"Model pack missing key: '{k}'. Rebuild with step5_train_and_pack.py.")
    return pack

@st.cache_resource(show_spinner=False)
def load_meta_rmse() -> Optional[float]:
    if META_PATH.exists():
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if "post_metrics" in meta and "RMSE" in meta["post_metrics"]:
                return float(meta["post_metrics"]["RMSE"])
        except Exception:
            return None
    return None

# ------------------------
# FEATURE EXTRACTION (must match training step)
# ------------------------
def pct(x, q):
    return float(np.percentile(x, q)) if x.size else np.nan

def safe_skew(x):
    if x.size < 2 or np.all(x == x.flat[0]):
        return 0.0
    from scipy.stats import skew
    return float(skew(x, bias=False, nan_policy="omit"))

def safe_kurtosis(x):
    if x.size < 2 or np.all(x == x.flat[0]):
        return 0.0
    from scipy.stats import kurtosis
    return float(kurtosis(x, bias=False, fisher=True, nan_policy="omit"))

def basic_stats(x):
    return {
        "mean": float(np.mean(x)) if x.size else np.nan,
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "min": float(np.min(x)) if x.size else np.nan,
        "p10": pct(x, 10),
        "p25": pct(x, 25),
        "p50": pct(x, 50),
        "p75": pct(x, 75),
        "p90": pct(x, 90),
        "max": float(np.max(x)) if x.size else np.nan,
        "skew": safe_skew(x),
        "kurt": safe_kurtosis(x),
    }

def hsv_tissue_mask(bgr, MASK_V_MIN=25, MASK_S_MIN=10, MIN_MASK_FRAC=0.01):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S = hsv[..., 1]
    V = hsv[..., 2]
    mask = (S >= MASK_S_MIN) & (V >= MASK_V_MIN)
    frac = float(np.mean(mask))
    if frac < MIN_MASK_FRAC:
        mask = np.ones(S.shape, dtype=bool)
        frac = 1.0
    return mask, frac, hsv

def circular_mean_hue(h):
    angles = (h.astype(np.float32) * (2*np.pi/180.0))
    s, c = np.sin(angles), np.cos(angles)
    mean_ang = math.atan2(np.mean(s), np.mean(c))
    return float((mean_ang % (2*np.pi)) * (180.0/(2*np.pi)))

def extract_features_for_image_bgr(img_bgr: np.ndarray, mask_cfg: dict):
    h, w = img_bgr.shape[:2]
    area = h * w
    mask, mask_frac, hsv = hsv_tissue_mask(
        img_bgr,
        MASK_V_MIN=mask_cfg.get("MASK_V_MIN", 25),
        MASK_S_MIN=mask_cfg.get("MASK_S_MIN", 10),
        MIN_MASK_FRAC=mask_cfg.get("MIN_MASK_FRAC", 0.01)
    )
    b, g, r = cv2.split(img_bgr)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    Lc, ac, bc = cv2.split(lab)

    r0 = r[mask].astype(np.float32); g0 = g[mask].astype(np.float32); b0 = b[mask].astype(np.float32)
    gray0 = gray[mask].astype(np.float32)
    H = hsv[..., 0][mask].astype(np.float32)
    S = hsv[..., 1][mask].astype(np.float32)
    V = hsv[..., 2][mask].astype(np.float32)
    L0 = Lc[mask].astype(np.float32); a0 = ac[mask].astype(np.float32); bLab0 = bc[mask].astype(np.float32)

    stats = {}
    for name, arr in (("R", r0), ("G", g0), ("B", b0), ("gray", gray0)):
        sdict = basic_stats(arr)
        for k, v in sdict.items():
            stats[f"{name}_{k}"] = v

    denom = r0 + g0 + b0 + 1e-6
    rnorm = r0 / denom
    stats["R_norm_mean"] = float(np.mean(rnorm))
    stats["R_norm_p50"]  = pct(rnorm, 50)
    stats["RG"] = (stats["R_mean"] + 1e-6) / (stats["G_mean"] + 1e-6)
    stats["RB"] = (stats["R_mean"] + 1e-6) / (stats["B_mean"] + 1e-6)

    stats["H_mean"] = circular_mean_hue(H) if H.size else np.nan
    stats["S_mean"] = float(np.mean(S)) if S.size else np.nan
    stats["S_p50"]  = pct(S, 50)
    stats["V_mean"] = float(np.mean(V)) if V.size else np.nan
    stats["V_p50"]  = pct(V, 50)

    stats["L_mean"] = float(np.mean(L0)) if L0.size else np.nan
    stats["a_mean"] = float(np.mean(a0)) if a0.size else np.nan
    stats["b_mean"] = float(np.mean(bLab0)) if bLab0.size else np.nan

    stats.update({
        "width": float(w),
        "height": float(h),
        "aspect_ratio": float(w / h),
        "area_px": float(area),
        "mask_frac": float(mask_frac),
    })
    return stats

# ------------------------
# ROBOFLOW: Hosted REST API (no inference-sdk)
# ------------------------
def rf_infer_image_bgr(image_bgr: np.ndarray, model_id: str, api_key: str,
                       confidence: float = 0.3, overlap: float = 0.3) -> Dict[str, Any]:
    """
    Calls Roboflow 'detect.roboflow.com/{model_id}' REST API by posting a JPEG file.
    Returns JSON with 'predictions': [ {x,y,width,height,confidence,...}, ... ]
    """
    ok, buf = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    url = f"https://detect.roboflow.com/{model_id}"
    params = {
        "api_key": api_key,
        "confidence": confidence,
        "overlap": overlap,
        "format": "json",
    }
    files = {"file": ("image.jpg", buf.tobytes(), "image/jpeg")}
    r = requests.post(url, params=params, files=files, timeout=30)
    r.raise_for_status()
    return r.json()

def choose_and_crop(image_bgr: np.ndarray, rf_json: Dict[str, Any], conf_thresh: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    From Roboflow JSON, pick the best detection (largest area among boxes passing conf_thresh;
    if none pass, pick the highest-confidence box). Return crop (BGR), overlay (RGB), and info dict.
    If no predictions at all: return centered fallback crop.
    """
    H, W = image_bgr.shape[:2]
    preds = rf_json.get("predictions", [])
    if not isinstance(preds, list) or len(preds) == 0:
        # Fallback: central crop (50% of image)
        cx, cy = W // 2, H // 2
        w, h = int(W * 0.5), int(H * 0.5)
        x1 = cx - w // 2; y1 = cy - h // 2; x2 = cx + w // 2; y2 = cy + h // 2
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H)
        crop = image_bgr[y1:y2, x1:x2].copy()
        overlay = draw_box_bgr(image_bgr.copy(), x1, y1, x2, y2)
        return crop, bgr_to_rgb(overlay), {"fallback": True, "reason": "no_predictions"}

    usable = []
    for p in preds:
        try:
            conf = float(p.get("confidence", 0.0))
            x = float(p["x"]); y = float(p["y"])
            w = float(p["width"]); h = float(p["height"])
            x1 = int(x - w / 2); y1 = int(y - h / 2)
            x2 = int(x + w / 2); y2 = int(y + h / 2)
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H)
            area = max(1, (x2 - x1) * (y2 - y1))
            usable.append({"conf": conf, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "area": area, "raw": p})
        except Exception:
            continue

    if not usable:
        cx, cy = W // 2, H // 2
        w, h = int(W * 0.5), int(H * 0.5)
        x1 = cx - w // 2; y1 = cy - h // 2; x2 = cx + w // 2; y2 = cy + h // 2
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H)
        crop = image_bgr[y1:y2, x1:x2].copy()
        overlay = draw_box_bgr(image_bgr.copy(), x1, y1, x2, y2)
        return crop, bgr_to_rgb(overlay), {"fallback": True, "reason": "malformed_predictions"}

    candidates = [u for u in usable if u["conf"] >= conf_thresh] or usable
    best = sorted(candidates, key=lambda d: (d["area"], d["conf"]), reverse=True)[0]
    x1, y1, x2, y2 = best["x1"], best["y1"], best["x2"], best["y2"]
    crop = image_bgr[y1:y2, x1:x2].copy()
    overlay = draw_box_bgr(image_bgr.copy(), x1, y1, x2, y2)
    return crop, bgr_to_rgb(overlay), {"fallback": False, "best": best}

# ------------------------
# PREDICTION PIPE
# ------------------------
def predict_hb_from_bgr_crop(crop_bgr: np.ndarray, pack: dict):
    mask_cfg = pack["mask_config"]
    feats = extract_features_for_image_bgr(crop_bgr, mask_cfg)

    feat_all = pack["feat_all"]
    vt = pack["vt"]
    corr_keep_mask = pack["corr_keep_mask"]
    Xraw = pd.DataFrame([{k: feats.get(k, np.nan) for k in feat_all}], columns=feat_all).astype(float)

    X_vt = vt.transform(Xraw)
    X_sel = X_vt[:, corr_keep_mask]

    model = pack["estimator"]
    iso = pack["iso"]

    hb_pred = float(model.predict(X_sel)[0])
    hb_cal = float(iso.predict([hb_pred])[0])

    return hb_pred, hb_cal, feats

def interval_from_rmse(point_est: float, rmse: Optional[float], z: float = 1.96):
    if rmse is None or not np.isfinite(rmse):
        return None
    half = z * rmse
    return (point_est - half, point_est + half)

# ------------------------
# SIDEBAR
# ------------------------
with st.sidebar:
    st.header("Settings")
    api_key_in = st.text_input("Roboflow API Key", value=DEFAULT_ROBOFLOW_API_KEY, type="password")
    model_id_in = st.text_input("Model ID", value=DEFAULT_ROBOFLOW_MODEL_ID, help="e.g., eye-conjunctiva-detector/2")
    conf_thresh = st.slider("Confidence threshold (select box)", 0.0, 0.9, value=DEFAULT_CONF_THRESH, step=0.05)
    overlap = st.slider("NMS overlap", 0.0, 0.9, value=DEFAULT_OVERLAP, step=0.05)
    st.markdown("---")
    show_uncal = st.checkbox("Show uncalibrated prediction", value=False)
    show_overlay = st.checkbox("Show detection overlay", value=True)

ROBOFLOW_API_KEY = api_key_in.strip() or DEFAULT_ROBOFLOW_API_KEY
ROBOFLOW_MODEL_ID = model_id_in.strip() or DEFAULT_ROBOFLOW_MODEL_ID

# ------------------------
# MAIN UI
# ------------------------
tabs = st.tabs(["Upload", "Camera"])
with tabs[0]:
    up = st.file_uploader("Upload an eye photo", type=["jpg", "jpeg", "png"])
with tabs[1]:
    cam = st.camera_input("Take a picture")

go = st.button("Estimate Hb", type="primary", use_container_width=True)

col1, col2 = st.columns([1, 1])

def process_image_bytes(img_bytes: bytes):
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    image_bgr = pil_to_bgr(pil_img)

    # Roboflow REST
    rf_json = rf_infer_image_bgr(
        image_bgr,
        model_id=ROBOFLOW_MODEL_ID,
        api_key=ROBOFLOW_API_KEY,
        confidence=max(min(conf_thresh, 0.99), 0.0),
        overlap=max(min(overlap, 0.99), 0.0),
    )

    crop_bgr, overlay_rgb, det_info = choose_and_crop(image_bgr, rf_json, conf_thresh=conf_thresh)

    # Model pack + prediction
    pack = load_pack()
    hb_pred, hb_cal, feats = predict_hb_from_bgr_crop(crop_bgr, pack)
    rmse_post = load_meta_rmse()
    interval = interval_from_rmse(hb_cal, rmse_post, z=1.96)

    return pil_img, crop_bgr, overlay_rgb, det_info, hb_pred, hb_cal, feats, rmse_post, interval

if go:
    try:
        uploaded_bytes = None
        original_preview = None

        if up is not None:
            uploaded_bytes = up.read()
            original_preview = Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")
        elif cam is not None:
            uploaded_bytes = cam.getvalue()
            original_preview = Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")
        else:
            st.warning("Please upload or capture an image first.")
            st.stop()

        with st.spinner("Detecting conjunctiva and estimating Hb..."):
            orig_pil, crop_bgr, overlay_rgb, det_info, hb_pred, hb_cal, feats, rmse_post, interval = process_image_bytes(uploaded_bytes)

        with col1:
            st.subheader("Original")
            if show_overlay:
                st.image(overlay_rgb, caption="Detection overlay", use_container_width=True)
            else:
                st.image(orig_pil, caption="Uploaded image", use_container_width=True)

        with col2:
            st.subheader("Conjunctiva Cutout")
            st.image(bgr_to_rgb(crop_bgr), caption="Detected region", use_container_width=True)

            st.subheader("Hb Estimation")
            if show_uncal:
                st.write(f"Uncalibrated: **{hb_pred:.2f} g/dL**")
            st.write(f"**Calibrated Hb:** :red[{hb_cal:.2f} g/dL]")

            if interval is not None:
                lo, hi = interval
                st.caption(f"Approx. 95% interval (±1.96×RMSE): **[{lo:.2f}, {hi:.2f}] g/dL**")
                if rmse_post is not None:
                    st.caption(f"(RMSE used: {rmse_post:.2f} g/dL from validation)")

            if isinstance(det_info, dict) and det_info.get("fallback", False):
                st.info(f"No detection from the model; used a centered fallback crop ({det_info.get('reason','')}).")

        # PARAMETERS
        st.markdown("---")
        st.subheader("Extracted Parameters")
        feat_df = pd.DataFrame([feats]).T.reset_index()
        feat_df.columns = ["parameter", "value"]
        st.dataframe(feat_df, use_container_width=True, height=420)

        csv_bytes = feat_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download parameters as CSV",
            data=csv_bytes,
            file_name="conjunctiva_features.csv",
            mime="text/csv",
            use_container_width=True
        )

    except requests.HTTPError as e:
        try:
            detail = e.response.json()
        except Exception:
            detail = e.response.text if hasattr(e, "response") else str(e)
        st.error(f"Roboflow HTTP error: {e}\nDetails: {detail}")
    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)

# Footer: environment versions (useful for debugging)
import numpy, sklearn
st.markdown("---")
st.caption(f"Env: NumPy {numpy.__version__} | scikit-learn {sklearn.__version__} | OpenCV {cv2.__version__} — Demo only, not for clinical use.")
