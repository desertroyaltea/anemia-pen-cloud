# app.py
import io
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageOps
import cv2
import streamlit as st

# ---- Try PMML (optional) ----
_PYPMML_AVAILABLE = False
try:
    from pypmml import Model as PMMLModel
    _PYPMML_AVAILABLE = True
except Exception:
    _PYPMML_AVAILABLE = False

# -------------------- Settings -------------------- #
DEFAULT_MODEL_CANDIDATES = [
    "hemo3.joblib", "hemo3.xml", "hemo.joblib", "hemo.xml"
]
ROBOFLOW_MODEL_ID = "eye-conjunctiva-detector/2"
DEFAULT_CONFIDENCE = 0.25

# -------------------- Utils -------------------- #
def pil_to_cv2_bgr(pil_im: Image.Image) -> np.ndarray:
    arr = np.array(pil_im.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv2_bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def _safe_kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x).ravel()
    if x.size == 0:
        return float("nan")
    m = x.mean()
    v = x.var(ddof=0)
    if v == 0:
        return 0.0
    z4 = np.mean(((x - m) ** 4))
    # Pearson (non-Fisher) kurtosis
    return z4 / (v ** 2)

def _percentile(a: np.ndarray, q: float) -> float:
    if a.size == 0:
        return float("nan")
    return float(np.percentile(a, q))

# -------------------- Feature extraction (14 feats) -------------------- #
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

def extract_features_from_crop(crop_bgr: np.ndarray) -> Dict[str, float]:
    # Channels
    b = crop_bgr[:, :, 0].astype(np.float32)
    g = crop_bgr[:, :, 1].astype(np.float32)
    r = crop_bgr[:, :, 2].astype(np.float32)

    # Gray
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # HSV
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    # h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    # v = hsv[:, :, 2]

    # LAB (OpenCV LAB is L in [0..255], a,b in ~[0..255] with 128 center)
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    # L_lab = lab[:, :, 0].astype(np.float32)
    a_lab = lab[:, :, 1].astype(np.float32)
    # b_lab = lab[:, :, 2].astype(np.float32)

    # Core stats
    R_mean = float(r.mean()) if r.size else np.nan
    G_mean = float(g.mean()) if g.size else np.nan
    B_mean = float(b.mean()) if b.size else np.nan

    feats = {
        # red normalized median: median(r) / mean(g)
        "R_norm_p50": (np.median(r) / G_mean) if (G_mean and not np.isnan(G_mean)) else np.nan,
        "a_mean": float(a_lab.mean()) if a_lab.size else np.nan,
        "R_p50": float(np.median(r)) if r.size else np.nan,
        "R_p10": _percentile(r, 10.0),
        "RG": (R_mean / G_mean) if (G_mean and not np.isnan(G_mean)) else np.nan,
        "S_p50": float(np.median(s)) if s.size else np.nan,
        "gray_p90": _percentile(gray, 90.0),
        "gray_kurt": _safe_kurtosis(gray),
        "gray_std": float(gray.std()) if gray.size else np.nan,
        "gray_mean": float(gray.mean()) if gray.size else np.nan,
        "B_mean": B_mean,
        "B_p10": _percentile(b, 10.0),
        "B_p75": _percentile(b, 75.0),
        "G_kurt": _safe_kurtosis(g),
    }
    return {k: float(v) if v is not None else np.nan for k, v in feats.items()}

# -------------------- Roboflow (multipart file) -------------------- #
def roboflow_detect_best_box(image_bytes: bytes, api_key: str, model_id: str, confidence: float = DEFAULT_CONFIDENCE,
                             retries: int = 2, timeout: int = 30) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    url = f"https://detect.roboflow.com/{model_id}"
    params = {
        "api_key": api_key,
        "confidence": str(confidence),
    }
    last_err = None
    body_text = None
    for attempt in range(retries + 1):
        try:
            files = {
                "file": ("image.jpg", image_bytes, "image/jpeg")
            }
            resp = requests.post(url, params=params, files=files, timeout=timeout)
            body_text = resp.text
            if resp.status_code >= 200 and resp.status_code < 300:
                data = resp.json()
                preds = data.get("predictions", []) or []
                if not preds:
                    return None, {"status": resp.status_code, "error": "No detections", "body": data}
                best = max(preds, key=lambda d: d.get("confidence", 0.0))
                return best, {"status": resp.status_code, "best": best, "all": preds}
            else:
                last_err = f"HTTP error {resp.status_code}"
        except requests.RequestException as e:
            last_err = str(e)
        time.sleep(0.6)
    # failed
    return None, {
        "status": 500,
        "error": f"Roboflow 5xx after retries. Last: {last_err}",
        "body": body_text,
    }

def crop_from_box(orig_bgr: np.ndarray, box: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    h, w = orig_bgr.shape[:2]
    cx = float(box["x"])
    cy = float(box["y"])
    bw = float(box["width"])
    bh = float(box["height"])
    x1 = max(0, int(round(cx - bw / 2)))
    y1 = max(0, int(round(cy - bh / 2)))
    x2 = min(w, int(round(cx + bw / 2)))
    y2 = min(h, int(round(cy + bh / 2)))

    crop = orig_bgr[y1:y2, x1:x2].copy()

    overlay = orig_bgr.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 4)
    label = f'{box.get("class","")}: {box.get("confidence",0):.2f}'
    cv2.putText(overlay, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    return crop, overlay

# -------------------- Model loading & prediction -------------------- #
@st.cache_resource
def load_joblib_model(path: Path):
    import joblib
    return joblib.load(str(path))

@st.cache_resource
def load_pmml_model(path: Path):
    if not _PYPMML_AVAILABLE:
        raise RuntimeError("pypmml is not installed in this environment.")
    return PMMLModel.load(str(path))

def pick_default_model_file() -> Optional[Path]:
    for name in DEFAULT_MODEL_CANDIDATES:
        p = Path(name)
        if p.exists():
            return p
    return None

def predict_with_model(model_obj, feat_row: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
    """
    Supports scikit-learn (.joblib) and PMML (.xml).
    Returns (prediction, debug_info)
    """
    debug: Dict[str, Any] = {}

    # --- PMML FIRST ---
    if _PYPMML_AVAILABLE:
        try:
            from pypmml import Model as PMMLModel  # type: ignore
            if isinstance(model_obj, PMMLModel):
                # PMML expects a dict-like input
                inp = {k: float(feat_row.iloc[0][k]) for k in FEATURE_ORDER}
                out = model_obj.predict(inp)

                # pypmml returns a pandas DataFrame in most envs
                if isinstance(out, pd.DataFrame):
                    out_df = out
                else:
                    # fallback: try to coerce
                    try:
                        out_df = pd.DataFrame(out)
                    except Exception:
                        raise RuntimeError(f"Unexpected PMML output type: {type(out)}")

                # choose a predicted column
                used_col = None
                for c in out_df.columns:
                    if str(c).lower().startswith("predicted"):
                        used_col = c
                        break
                if used_col is None:
                    # fallback: first numeric column, else first column
                    num_cols = [c for c in out_df.columns if pd.api.types.is_numeric_dtype(out_df[c])]
                    used_col = num_cols[0] if num_cols else out_df.columns[0]

                y_val = float(out_df.iloc[0][used_col])

                debug["backend"] = "pmml"
                debug["pmml_output_columns"] = list(out_df.columns)
                debug["used_column"] = used_col
                # record numeric outputs for transparency
                debug["raw_output_row0"] = {
                    c: (float(out_df.iloc[0][c]) if pd.api.types.is_numeric_dtype(out_df[c]) else str(out_df.iloc[0][c]))
                    for c in out_df.columns
                }
                return y_val, debug
        except Exception as e:
            raise RuntimeError(f"PMML prediction failed: {e}")

    # --- sklearn / joblib ---
    if hasattr(model_obj, "predict"):
        try:
            X = feat_row[FEATURE_ORDER].astype(float)
            y = model_obj.predict(X)
            y_val = float(y[0])
            debug["backend"] = "joblib_sklearn"
            debug["used_columns"] = list(X.columns)
            return y_val, debug
        except Exception as e:
            raise RuntimeError(f"sklearn prediction failed: {e}")

    raise RuntimeError("Unknown model object type (neither sklearn nor pypmml).")


# -------------------- UI -------------------- #
st.set_page_config(page_title="Anemia Pen (Conjunctiva Hb Estimator)", layout="wide")

st.title("Anemia Pen – Conjunctival Hb Estimator")

with st.sidebar:
    st.subheader("Settings")

    # Model selection (default to hemo3.joblib if present)
    default_model = pick_default_model_file()
    model_files_on_disk = [str(p) for p in Path(".").glob("hemo*.joblib")] + [str(p) for p in Path(".").glob("hemo*.xml")]
    if default_model and str(default_model) not in model_files_on_disk:
        model_files_on_disk.insert(0, str(default_model))

    selected_model_path_str = st.selectbox(
        "Choose model file",
        options=model_files_on_disk or [str(default_model) if default_model else "— no model found —"],
        index=0
    )
    selected_model_path = Path(selected_model_path_str) if selected_model_path_str and selected_model_path_str != "— no model found —" else None
    st.write(f"**Loaded model:** {selected_model_path.name if selected_model_path else 'None'}")

    # RMSE for interval
    rmse = st.number_input("RMSE to build ± interval", value=1.7, min_value=0.0, step=0.1)

    # Roboflow API key
    api_key_default = st.secrets.get("ROBOFLOW_API_KEY", "") if "ROBOFLOW_API_KEY" in st.secrets else ""
    api_key = st.text_input("Roboflow API Key", value=api_key_default or "", type="password")

    show_debug = st.checkbox("Show debug info", value=False)

# Load model object
model_obj = None
model_backend = None
if selected_model_path and selected_model_path.exists():
    if selected_model_path.suffix.lower() == ".joblib":
        model_obj = load_joblib_model(selected_model_path)
        model_backend = "joblib"
    elif selected_model_path.suffix.lower() == ".xml":
        if not _PYPMML_AVAILABLE:
            st.warning("This environment does not have `pypmml` installed. XML/PMML model cannot be loaded.")
        else:
            model_obj = load_pmml_model(selected_model_path)
            model_backend = "pmml"

col_u, col_c = st.columns(2)
with col_u:
    st.subheader("Upload or Capture")
    uploaded = st.file_uploader("Upload eye photo (full eye, conjunctiva visible)", type=["jpg", "jpeg", "png"])
    st.caption("— OR —")
    camera_img = st.camera_input("Take a photo")

    go = st.button("Estimate Hb", type="primary", use_container_width=True)

with col_c:
    st.subheader("Parameters")
    st.write("Will display the **extracted 14 features** and detection info after prediction.")

# Main action
if go:
    if model_obj is None:
        st.error("No model is loaded. Please select a valid model file in the sidebar.")
    elif not api_key:
        st.error("Please enter your Roboflow API key in the sidebar.")
    else:
        # 1) Read image from upload / camera
        raw_bytes = None
        filename = "image.jpg"
        if uploaded is not None:
            raw_bytes = uploaded.read()
            filename = uploaded.name or filename
        elif camera_img is not None:
            raw_bytes = camera_img.getvalue()
            filename = "camera.jpg"
        else:
            st.error("Please upload or capture an image.")
            st.stop()

        # 2) Correct EXIF rotation and convert to bytes (JPEG)
        try:
            pil_im = Image.open(io.BytesIO(raw_bytes))
            pil_im = ImageOps.exif_transpose(pil_im)  # correct rotation
            # Roboflow likes JPG
            bio = io.BytesIO()
            pil_im.save(bio, format="JPEG", quality=95)
            img_bytes_for_rf = bio.getvalue()
            orig_bgr = pil_to_cv2_bgr(pil_im)
        except Exception as e:
            st.error(f"Could not read the image: {e}")
            st.stop()

        # 3) Roboflow detect
        best_box, det_info = roboflow_detect_best_box(
            image_bytes=img_bytes_for_rf,
            api_key=api_key,
            model_id=ROBOFLOW_MODEL_ID,
            confidence=DEFAULT_CONFIDENCE,
            retries=2,
            timeout=30
        )
        if best_box is None:
            st.error("No conjunctiva detected. Try another photo.")
            if det_info:
                st.code(json.dumps(det_info, indent=2), language="json")
            st.stop()

        # 4) Crop & overlay
        crop_bgr, overlay_bgr = crop_from_box(orig_bgr, best_box)

        # 5) Extract features
        feats = extract_features_from_crop(crop_bgr)
        feat_row = pd.DataFrame([[feats[k] for k in FEATURE_ORDER]], columns=FEATURE_ORDER)

        # 6) Predict
        try:
            y_hat, dbg = predict_with_model(model_obj, feat_row)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        # 7) Interval using RMSE
        lo = y_hat - rmse
        hi = y_hat + rmse

        # 8) Show results
        st.success(f"**Estimated Hb:** {y_hat:.2f} g/dL  \n**±RMSE interval:** [{lo:.2f}, {hi:.2f}] g/dL")
        st.caption(f"Model file: **{selected_model_path.name if selected_model_path else '—'}**  | Backend: **{model_backend or '—'}**")

        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.image(cv2_bgr_to_pil(orig_bgr), caption="Original (orientation corrected)", use_container_width=True)
        with img_col2:
            st.image(cv2_bgr_to_pil(overlay_bgr), caption="Detection overlay", use_column_width=True)

        st.subheader("Extracted Features (14)")
        st.dataframe(pd.DataFrame([feats]), use_container_width=True, height=420)

        if show_debug:
            st.subheader("Debug")
            st.write("**Detection (best box):**")
            st.code(json.dumps(best_box, indent=2), language="json")

            st.write("**Roboflow response (summary):**")
            st.code(json.dumps({k: det_info.get(k) for k in ["status", "error"] if k in det_info} | {"best": best_box}, indent=2), language="json")

            st.write("**Model inputs (ordered):**")
            st.code(json.dumps(FEATURE_ORDER, indent=2), language="json")

            st.write("**PMML/Model debug:**")
            try:
                st.code(json.dumps(dbg, indent=2), language="json")
            except Exception:
                st.write(dbg)
