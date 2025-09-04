#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import sys
import base64
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageOps, ImageDraw
import cv2
from scipy.stats import kurtosis
import streamlit as st

# Optional imports (gracefully handled)
try:
    import joblib
except Exception:
    joblib = None

# -----------------------------------------------------------------------------
# Compatibility shims for loading .joblib files that contain custom classes
# saved by the converter script (PMMLProxyEstimator / NativeLinearEstimator).
# These classes must be importable under the same module path they were saved
# with (often "main" when the converter was run as a script).
# -----------------------------------------------------------------------------

class PMMLProxyEstimator:
    """
    A scikit-learn-like estimator that delegates prediction to the stored PMML
    via PyPMML. Ensures identical outputs to the original PMML.

    Saved attributes expected in the pickled object:
      - pmml_xml: str (the entire PMML as XML text)
      - feature_names: list[str] (preferred input column order)
      - output_candidates: list[str] (possible output field names, e.g., 'hb')
    """
    def __init__(self, pmml_xml: str, feature_names: List[str], output_candidates: List[str]):
        self.pmml_xml = pmml_xml
        self.feature_names = list(feature_names) if feature_names else []
        self.output_candidates = list(output_candidates) if output_candidates else []
        self._model = None  # lazy-loaded PyPMML model

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from pypmml import Model  # imported here to avoid hard dependency at import time
        except Exception as e:
            raise RuntimeError(f"pypmml is required to use PMMLProxyEstimator: {e}")
        # PyPMML expects a file path; write the XML to a temp file
        with tempfile.NamedTemporaryFile("w", suffix=".pmml", delete=False, encoding="utf-8") as tf:
            tf.write(self.pmml_xml)
            tmp_path = tf.name
        self._model = Model.load(tmp_path)

    def predict(self, X: Any) -> np.ndarray:
        import pandas as pd
        self._ensure_model()
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names if self.feature_names else None)
        # ensure float dtype
        X = X.copy()
        for c in X.columns:
            X[c] = X[c].astype(float)

        df = self._model.predict(X)

        # Prefer known output names
        for key in (self.output_candidates or []):
            if key in df.columns:
                return df[key].astype(float).to_numpy().ravel()

        # Fallback to first numeric column
        for col in df.columns:
            try:
                return df[col].astype(float).to_numpy().ravel()
            except Exception:
                continue

        raise RuntimeError("PMML proxy: couldn't find numeric prediction column in model output.")


class NativeLinearEstimator:
    """
    Minimal linear predictor: y = intercept + Σ coef[i] * x[i]
    Only used when a plain RegressionModel (no transforms) was parsed natively.
    """
    def __init__(self, intercept: float, coefs: Dict[str, float], feature_names: List[str]):
        self.intercept = float(intercept)
        self.coefs = dict(coefs)
        self.feature_names = list(feature_names)

    def predict(self, X: Any) -> np.ndarray:
        import pandas as pd
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names if self.feature_names else None)
        X = X.copy()
        for c in X.columns:
            X[c] = X[c].astype(float)
        y = np.full((len(X),), self.intercept, dtype=float)
        for name, coef in self.coefs.items():
            if name not in X.columns:
                raise ValueError(f"Missing required feature for native linear model: {name}")
            y += coef * X[name].to_numpy(dtype=float)
        return y

# Register this module under names that pickles may reference (e.g., "main", "pmml_to_joblib")
sys.modules.setdefault("main", sys.modules[__name__])
sys.modules.setdefault("pmml_to_joblib", sys.modules[__name__])

# -------------------- Settings -------------------- #
DEFAULT_JOBLIB_PATH = Path("outputs/models/hemo_surrogate.joblib")
DEFAULT_PMML_PATH = Path("outputs/models/hb_modeler.xml")  # change if needed

DEFAULT_MODEL_ID = "eye-conjunctiva-detector/2"   # Roboflow model id
DEFAULT_CLASS_NAME = "conjunctiva"
DEFAULT_CONF_0_100 = 25                           # 0..100 (≈ 0.25)

FEATURE_COLUMNS = [
    "R_norm_p50", "a_mean", "R_p50", "R_p10", "RG", "S_p50",
    "gray_p90", "gray_kurt", "gray_std", "gray_mean",
    "B_mean", "B_p10", "B_p75", "G_kurt",
]

st.set_page_config(page_title="Anemia Pen — PMML & Joblib (Fixed Scaling + Proxy Support)", layout="wide")

# ---------------- Helper functions ---------------- #
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

def compute_features_from_pil(pil_img: Image.Image) -> Dict[str, float]:
    """
    Compute 14 features (non-WB recipe):
      - R_norm_p50 uses R/(R+G+B)
      - gray_kurt & G_kurt: Pearson (fisher=False), bias=False
      - gray_std uses ddof=0
      - S_p50 uses HSV saturation normalized to 0..1
      - a_mean = Lab a* minus 128
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

# --------- PMML helpers --------- #
def pmml_predict(model, X: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
    """
    Score with PyPMML Model. Returns (prediction_value, extra_info).
    Handles different output column names across PMMLs.
    """
    try:
        df = model.predict(X)
        # Prefer realistic target columns
        for key in ["hb", "predicted_hb", "Prediction", "Predicted_hb"]:
            if key in df.columns:
                val = df[key].iloc[0]
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    raise ValueError("PMML returned None/NaN for prediction")
                return float(val), {"raw": df.head(1).to_dict(orient="records")[0]}
        # Fallback: first numeric column
        for col in df.columns:
            try:
                val = float(df[col].iloc[0])
                return val, {"raw": df.head(1).to_dict(orient="records")[0], "picked_col": col}
            except Exception:
                continue
        raise ValueError("Cannot find numeric prediction in PMML result")
    except Exception as e:
        raise RuntimeError(f"PMML scoring error: {e}")

# -------------------- Caching loaders -------------------- #
@st.cache_resource(show_spinner=False)
def load_joblib(path_str: str):
    if joblib is None:
        raise RuntimeError("joblib/scikit-learn not installed. Install scikit-learn and joblib.")
    return joblib.load(path_str)

@st.cache_resource(show_spinner=False)
def load_pmml(path_str: str):
    try:
        from pypmml import Model
    except Exception as e:
        raise RuntimeError(f"PyPMML not available: {e}")
    return Model.load(path_str)

def _extract_estimator(jobj):
    """Handles both raw estimators and bundles {'estimator': est} created by our converter."""
    if isinstance(jobj, dict) and "estimator" in jobj:
        return jobj["estimator"]
    return jobj

def _is_pmml_proxy(est) -> bool:
    """Detect our PMMLProxyEstimator produced by pmml_to_joblib.py."""
    return hasattr(est, "pmml_xml")

# ------------------------- UI ------------------------- #
st.title("Anemia Pen — PMML & Joblib (Fixed Scaling + PMML-Proxy Support)")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "Roboflow API Key",
        value=os.getenv("ROBOFLOW_API_KEY", ""),
        type="password",
        help="Stored only in your session. Or set env var ROBOFLOW_API_KEY.",
    )
    model_id = st.text_input("Roboflow Model ID", value=DEFAULT_MODEL_ID)
    class_name = st.text_input("Target class", value=DEFAULT_CLASS_NAME)
    conf = st.slider("Detection confidence (0–100)", min_value=1, max_value=100, value=DEFAULT_CONF_0_100)
    rmse = st.number_input("Optional RMSE for CI (±1.96×RMSE)", min_value=0.0, value=0.0, step=0.1)

    st.markdown("---")
    st.caption("Choose a scoring backend")
    backend = st.radio("Backend", ["PMML", "Joblib"], horizontal=True)

    # Forced scaling toggles
    force_spss_scaling_pmml = st.checkbox(
        "Force SPSS scaling for PMML (a* + 128, S × 100)", value=True,
        help="Fixes constant predictions from SPSS exports"
    )
    force_spss_scaling_joblib_proxy = st.checkbox(
        "If joblib wraps a PMML, apply the same SPSS scaling", value=True,
        help="Applies (a*+128, S×100) to inputs when the joblib is a PMML proxy"
    )

    model_path_str = ""

    if backend == "PMML":
        src = st.radio("Load PMML from…", ["Path", "Upload"], horizontal=True)
        if src == "Path":
            model_path_str = st.text_input("PMML path", value=str(DEFAULT_PMML_PATH))
        else:
            up = st.file_uploader("Upload .pmml / .xml", type=["pmml", "xml"])
            if up is not None:
                tmp = Path("uploaded_model.pmml")
                with open(tmp, "wb") as f:
                    f.write(up.read())
                model_path_str = str(tmp)
        load_btn = st.button("Load PMML model", use_container_width=True)
        model_loaded = None
        if load_btn and model_path_str:
            try:
                model_loaded = load_pmml(model_path_str)
                st.success(f"Loaded PMML: {model_path_str}")
            except Exception as e:
                st.error(f"Failed to load PMML: {e}")
    else:
        src = st.radio("Load joblib from…", ["Path", "Upload"], horizontal=True)
        if src == "Path":
            model_path_str = st.text_input("Joblib path", value=str(DEFAULT_JOBLIB_PATH))
        else:
            up = st.file_uploader("Upload .joblib", type=["joblib"])
            if up is not None:
                tmp = Path("uploaded_model.joblib")
                with open(tmp, "wb") as f:
                    f.write(up.read())
                model_path_str = str(tmp)
        load_btn = st.button("Load joblib model", use_container_width=True)
        model_loaded = None
        if load_btn and model_path_str:
            try:
                model_loaded = load_joblib(model_path_str)
                est = _extract_estimator(model_loaded)
                if _is_pmml_proxy(est):
                    st.info("Loaded joblib wraps a PMML model (PMML proxy).")
                else:
                    st.success(f"Loaded joblib: {model_path_str}")
            except Exception as e:
                st.error(f"Failed to load joblib: {e}")

# Persist across reruns
if "model_ready" not in st.session_state:
    st.session_state["model_ready"] = False
if "backend" not in st.session_state:
    st.session_state["backend"] = backend
if load_btn:
    st.session_state["model_ready"] = model_loaded is not None
    st.session_state["backend"] = backend
    st.session_state["force_spss_scaling_pmml"] = force_spss_scaling_pmml
    st.session_state["force_spss_scaling_joblib_proxy"] = force_spss_scaling_joblib_proxy
    if backend == "PMML":
        st.session_state["pmml_path"] = model_path_str
    else:
        st.session_state["joblib_path"] = model_path_str

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
    if not st.session_state.get("model_ready", False):
        st.error("Model is not loaded. Load it from the sidebar first.")
        st.stop()

    # Read & upright
    try:
        pil_full = Image.open(io.BytesIO(uploaded.read()))
        pil_full = exif_upright(pil_full)
    except Exception as e:
        st.error(f"Failed to open image: {e}")
        st.stop()

    # Detect
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

    # Overlay & crop
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
        if st.session_state.get("backend") == "PMML":
            from pypmml import Model  # surface import errors clearly
            pmml_model = load_pmml(st.session_state["pmml_path"])

            # ---- PMML expects raw Lab a* and S on 0..100 ----
            feats_pmml = feats.copy()
            if st.session_state.get("force_spss_scaling_pmml", True):
                feats_pmml["a_mean"] = feats_pmml["a_mean"] + 128.0
                feats_pmml["S_p50"]  = feats_pmml["S_p50"]  * 100.0

            X = pd.DataFrame([{k: float(feats_pmml.get(k, np.nan)) for k in FEATURE_COLUMNS}])
            pred_hb, extra = pmml_predict(pmml_model, X)

        else:
            joblib_model = load_joblib(st.session_state["joblib_path"])
            est = _extract_estimator(joblib_model)

            # Build from raw features first
            X = pd.DataFrame([{k: float(feats[k]) for k in FEATURE_COLUMNS}])

            # If this joblib wraps a PMML model, apply the same SPSS scaling
            if _is_pmml_proxy(est) and st.session_state.get("force_spss_scaling_joblib_proxy", True):
                X["a_mean"] = X["a_mean"] + 128.0
                X["S_p50"]  = X["S_p50"]  * 100.0

            pred = est.predict(X)
            pred_hb = float(np.ravel(pred)[0])
            extra = {}
    except Exception as e:
        st.error(f"Model scoring error: {e}")
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

    # Show parameters
    st.markdown("### Extracted Parameters (14)")
    feat_df = pd.DataFrame([feats], columns=FEATURE_COLUMNS)
    st.dataframe(feat_df, use_container_width=True, height=420)

    # Debug
    with st.expander("Advanced / Debug info"):
        st.write("Roboflow best box:", best)
        st.write("Backend:", st.session_state.get("backend"))
        if st.session_state.get("backend") == "PMML":
            st.write("Forced SPSS scaling (PMML):", st.session_state.get("force_spss_scaling_pmml", True))
            if st.session_state.get("force_spss_scaling_pmml", True):
                st.write("Adjusted a_mean & S_p50 sent to PMML:",
                         {"a_mean": feats.get("a_mean", None) + 128.0,
                          "S_p50": feats.get("S_p50", None) * 100.0})
            st.write("PMML raw output (first row):", extra.get("raw"))
        else:
            st.write("Joblib path:", st.session_state.get("joblib_path"))
            est = _extract_estimator(load_joblib(st.session_state["joblib_path"]))
            st.write("Joblib wraps PMML proxy:", _is_pmml_proxy(est))
            if _is_pmml_proxy(est) and st.session_state.get("force_spss_scaling_joblib_proxy", True):
                st.write("Adjusted a_mean & S_p50 sent to Joblib-PMML:",
                         {"a_mean": feats.get("a_mean", None) + 128.0,
                          "S_p50": feats.get("S_p50", None) * 100.0})

# Footer
st.markdown("---")
st.caption("Anemia Pen — PMML/Joblib demo (with fixed scaling & PMML-proxy support)")
