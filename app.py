# app.py
import io
import json
import math
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import cv2
import requests
import streamlit as st
from pypmml import Model

# -------------------- Settings -------------------- #
DEFAULT_PMML = "hemo3.xml"   # auto-load this, but allow switching in UI
PMML_DIR = Path(".")
CROPPED_DIR = Path("cropped_images")
CROPPED_DIR.mkdir(parents=True, exist_ok=True)

# Roboflow
ROBOFLOW_MODEL = "eye-conjunctiva-detector/2"
ROBOFLOW_API = st.secrets.get("ROBOFLOW_API_KEY", None) or "jMhyBQxeQvj69nttV0mN"
ROBOFLOW_BASE = "https://detect.roboflow.com"
ROBOFLOW_CONF = 0.25
MAX_RETRIES = 2

# -------------------------------------------------- #
st.set_page_config(page_title="Anemia Pen – Hb Estimator", layout="centered")
st.title("Anemia Pen – Hb Estimator")

# Sidebar: pick PMML (default hemo3.xml)
available_pmml = [p.name for p in PMML_DIR.glob("*.xml")]
if DEFAULT_PMML in available_pmml:
    default_index = available_pmml.index(DEFAULT_PMML)
else:
    default_index = 0 if available_pmml else 0
pmml_file = st.sidebar.selectbox("Model file (PMML)", available_pmml, index=default_index)
st.sidebar.caption("Default is hemo3.xml; you can switch to hemo.xml (old) if needed.")

# Load PMML once
@st.cache_resource(show_spinner=False)
def load_pmml_model(path: str) -> Model:
    return Model.load(path)

if not available_pmml:
    st.error("No PMML file (*.xml) found in the app folder.")
    st.stop()

pmml_path = str(PMML_DIR / pmml_file)
try:
    model = load_pmml_model(pmml_path)
except Exception as e:
    st.error(f"Failed to load PMML: {pmml_path}\n{e}")
    st.stop()

st.write(f"**Loaded PMML:** `{pmml_file}`")

# -------------------- Utils -------------------- #

def imdecode_keep_exif(image_bytes: bytes) -> np.ndarray:
    """Decode JPEG/PNG and rotate to upright based on EXIF if present (OpenCV does not handle EXIF)."""
    try:
        from PIL import Image, ImageOps
        pil = Image.open(io.BytesIO(image_bytes))
        pil = ImageOps.exif_transpose(pil)
        rgb = np.array(pil.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr
    except Exception:
        arr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def roboflow_detect_and_crop(image_bgr: np.ndarray,
                             api_key: str,
                             model_id: str = ROBOFLOW_MODEL,
                             conf: float = ROBOFLOW_CONF,
                             max_retries: int = MAX_RETRIES) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """Send image to Roboflow, get conjunctiva box, crop it, return crop and overlay; retry on 5xx."""
    ok, jpg = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise RuntimeError("Failed to encode image for Roboflow.")
    data = jpg.tobytes()

    url = f"{ROBOFLOW_BASE}/{model_id}"
    params = {"api_key": api_key, "confidence": str(conf)}

    last_err = None
    for attempt in range(1, max_retries + 2):
        try:
            # IMPORTANT: multipart/form-data with field name "image"
            resp = requests.post(
                url,
                params=params,
                files={"image": ("image.jpg", data, "image/jpeg")},
                timeout=25
            )
            if resp.status_code >= 500:
                last_err = f"Roboflow 5xx after retries. Status={resp.status_code}. Body: {resp.text}"
                continue
            resp.raise_for_status()
            det = resp.json()
            preds = det.get("predictions", [])
            # pick highest-confidence conj detection
            conj = None
            best = -1
            for p in preds:
                if p.get("class") == "conjunctiva" and p.get("confidence", 0) > best:
                    best = p["confidence"]
                    conj = p
            if conj is None:
                return None, None, {"rf_status": resp.status_code, "rf_body": det, "note": "No conjunctiva detected."}

            # bbox is center x,y with width,height (in px of input)
            x, y, w, h = conj["x"], conj["y"], conj["width"], conj["height"]
            x0 = max(int(x - w / 2), 0)
            y0 = max(int(y - h / 2), 0)
            x1 = min(int(x + w / 2), image_bgr.shape[1] - 1)
            y1 = min(int(y + h / 2), image_bgr.shape[0] - 1)

            crop = image_bgr[y0:y1, x0:x1].copy()

            overlay = image_bgr.copy()
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)
            info = {
                "rf_status": resp.status_code,
                "rf_box": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                "rf_conf": best,
                "rf_raw_count": len(preds)
            }
            return crop, overlay, info
        except requests.HTTPError as e:
            return None, None, {
                "error": f"HTTP error from Roboflow: {e}",
                "status": getattr(e.response, 'status_code', None),
                "body": getattr(e.response, 'text', None),
            }
        except Exception as e:
            last_err = str(e)
    raise RuntimeError(f"Detection/Cropping failed: {last_err or 'unknown error'}")

def extract_features_from_crop(crop_bgr: np.ndarray) -> Dict[str, float]:
    """Compute the 14 features from cropped conjunctiva."""
    if crop_bgr is None or crop_bgr.size == 0:
        raise ValueError("Empty crop for feature extraction.")
    bgr = crop_bgr
    b = bgr[:, :, 0].astype(np.float32)
    g = bgr[:, :, 1].astype(np.float32)
    r = bgr[:, :, 2].astype(np.float32)

    B_mean = float(np.mean(b))
    G_mean = float(np.mean(g))
    R_mean = float(np.mean(r))
    RG = float(R_mean / (G_mean + 1e-6))

    R_p50 = float(np.percentile(r, 50))
    R_p10 = float(np.percentile(r, 10))
    B_p10 = float(np.percentile(b, 10))
    B_p75 = float(np.percentile(b, 75))

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_mean = float(np.mean(gray))
    gray_std = float(np.std(gray, ddof=0))
    gray_p90 = float(np.percentile(gray, 90))

    def kurtosis_fisher(x: np.ndarray) -> float:
        x = x.ravel().astype(np.float64)
        if x.size < 4:
            return 0.0
        m = np.mean(x)
        s2 = np.mean((x - m) ** 2)
        s4 = np.mean((x - m) ** 4)
        if s2 <= 1e-12:
            return 0.0
        return float(s4 / (s2 ** 2))  # Pearson (Fisher=False)

    gray_kurt = kurtosis_fisher(gray)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    a_mean = float(np.mean(lab[:, :, 1]))

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S_median = float(np.percentile(hsv[:, :, 1], 50))

    R_norm_p50 = float(R_p50 / (R_mean + 1e-6))
    G_kurt = kurtosis_fisher(g)

    feats = {
        "R_norm_p50": R_norm_p50,
        "a_mean": a_mean,
        "R_p50": R_p50,
        "R_p10": R_p10,
        "RG": RG,
        "S_p50": S_median,      # raw (OpenCV 0..255)
        "gray_p90": gray_p90,
        "gray_kurt": gray_kurt,
        "gray_std": gray_std,
        "gray_mean": gray_mean,
        "B_mean": B_mean,
        "B_p10": B_p10,
        "B_p75": B_p75,
        "G_kurt": G_kurt,
    }
    return feats

def adapt_features_for_pmml(feats: Dict[str, float], pmml_name: str) -> Dict[str, float]:
    """
    Map our app features to the units each PMML expects.

    - hemo.xml (old):
        * a_mean -> unoffset Lab: a_mean - 128
        * S_p50  -> 0..1 (if >1.5, divide by 255)
        * others: pass-through
    - hemo3.xml (new):
        * a_mean -> keep offset Lab (0..255)
        * RG     -> gray-world balance to be ~1.0 (tight normalization window)
        * others: pass-through
    """
    f = feats.copy()
    if pmml_name.lower().startswith("hemo.xml") or pmml_name.lower() == "hemo.xml":
        f["a_mean"] = feats["a_mean"] - 128.0
        s = feats["S_p50"]
        f["S_p50"] = s / 255.0 if s > 1.5 else s
        return f
    else:
        # NEW (hemo3.xml) — rebalance RG toward ~1.0 via gray-world
        R_mean_est = feats["R_p50"] / (feats["R_norm_p50"] + 1e-9)
        G_mean_est = R_mean_est / (feats["RG"] + 1e-9)
        gray_mean = feats["gray_mean"]
        scaleR = gray_mean / (R_mean_est + 1e-9)
        scaleG = gray_mean / (G_mean_est + 1e-9)
        RG_bal = (R_mean_est * scaleR) / (G_mean_est * scaleG + 1e-9)
        f["RG"] = float(RG_bal)
        return f

def predict_with_pmml(model: Model, pmml_inputs: Dict[str, float]) -> Tuple[float, Dict[str, Any]]:
    """Call PMML with a single-row DataFrame; return predicted hb and debug dict."""
    df = pd.DataFrame([pmml_inputs])
    out = model.predict(df)

    possible = [c for c in out.columns if "pred" in c.lower() and "hb" in c.lower()]
    if not possible and "hb" in out.columns:
        possible = ["hb"]
    if not possible:
        numcols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
        if numcols:
            used = numcols[0]
        else:
            raise RuntimeError(f"PMML output not found; columns={list(out.columns)}")
    else:
        used = possible[0]

    val = float(out.iloc[0][used])
    dbg = {
        "pmml_input_columns": list(df.columns),
        "pmml_input_row0": {k: float(df.iloc[0][k]) for k in df.columns},
        "pmml_output_columns": list(out.columns),
        "used_column": used,
        "raw_output_row0": {used: val},
    }
    return val, dbg

# -------------------- UI -------------------- #

uploaded = st.file_uploader(
    "Upload an eye photo (or take one)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
    label_visibility="visible",
    help="You can take a photo on iPhone/Android here."
)
if uploaded is None:
    st.info("Upload a photo to start.")
    st.stop()

img_bgr = imdecode_keep_exif(uploaded.read())
if img_bgr is None:
    st.error("Could not read the image.")
    st.stop()

st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Uploaded image", use_container_width=True)

# Crop
try:
    crop_bgr, overlay_bgr, det_info = roboflow_detect_and_crop(img_bgr, ROBOFLOW_API, ROBOFLOW_MODEL, ROBOFLOW_CONF, MAX_RETRIES)
except Exception as e:
    st.error(f"Detection/Cropping failed: {e}")
    st.stop()

if crop_bgr is None:
    st.warning("No conjunctiva detected. Try another photo.")
    st.json(det_info)
    st.stop()

st.image(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB), caption="Detection overlay", use_container_width=True)
st.image(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB), caption="Cropped conjunctiva", use_container_width=True)

# Features
feats = extract_features_from_crop(crop_bgr)

# Adapt per PMML
feats_adapted = adapt_features_for_pmml(feats, pmml_file)

# Ensure the 14 PMML inputs exist and are floats
pmml_inputs = {
    "R_norm_p50": float(feats_adapted["R_norm_p50"]),
    "a_mean": float(feats_adapted["a_mean"]),
    "R_p50": float(feats_adapted["R_p50"]),
    "R_p10": float(feats_adapted["R_p10"]),
    "RG": float(feats_adapted["RG"]),
    "S_p50": float(feats_adapted["S_p50"]),
    "gray_p90": float(feats_adapted["gray_p90"]),
    "gray_kurt": float(feats_adapted["gray_kurt"]),
    "gray_std": float(feats_adapted["gray_std"]),
    "gray_mean": float(feats_adapted["gray_mean"]),
    "B_mean": float(feats_adapted["B_mean"]),
    "B_p10": float(feats_adapted["B_p10"]),
    "B_p75": float(feats_adapted["B_p75"]),
    "G_kurt": float(feats_adapted["G_kurt"]),
}

# Predict
try:
    hb_pred, pmml_dbg = predict_with_pmml(model, pmml_inputs)
except Exception as e:
    st.error(f"PMML prediction failed: {e}")
    st.json({"pmml_inputs": pmml_inputs})
    st.stop()

# A simple uncertainty band (tweak if you like)
RMSE_ASSUMED = 1.7
ci_low = hb_pred - 1.96 * RMSE_ASSUMED
ci_high = hb_pred + 1.96 * RMSE_ASSUMED

st.subheader("Estimated Hemoglobin")
st.metric("Hb (g/dL)", f"{hb_pred:0.2f}", help=f"Approx. 95% band: {ci_low:0.2f}–{ci_high:0.2f} (assuming RMSE≈{RMSE_ASSUMED})")

with st.expander("PMML debug / detection info"):
    st.write("**Roboflow**", det_info)
    st.write("**Raw features (app extraction)**")
    st.json({k: float(v) for k, v in feats.items()})
    st.write("**Adapted features (sent to PMML)**")
    st.json(pmml_inputs)
    st.write("**PMML output**")
    st.json(pmml_dbg)
