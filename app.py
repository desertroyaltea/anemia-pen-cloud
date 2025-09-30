#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlit app for Patient-Centric Hb Estimation and Tracking.

This redesigned version features a modern health dashboard UI. After login,
the user is taken directly to a dashboard showing their latest reading,
a trend chart, and their full history.

Features:
- Simple user login to persist and view history.
- A central dashboard for all key information.
- Prominent display of the latest Hb reading.
- Integrated camera input within an expander to reduce clutter.
- Visual history with charts and image cards.
- "Doctor View" to show detailed technical features.
- Simple report generation for printing/PDF export.
"""

import os
import io
import json
import base64
from pathlib import Path
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import requests
import joblib
import plotly.express as px

# Imaging / features
import cv2
from scipy.stats import kurtosis
from scipy.ndimage import convolve
from skimage import exposure, filters, morphology, measure
from skimage.morphology import skeletonize

# ---------- CONFIG & CONSTANTS ---------- #

# --- App Behavior ---
DATA_FILE = Path("user_data.json")

# --- Model Artifacts ---
HB_ESTIMATION_RUN_DIR = Path("models") / "run_20250918_192217"
HB_FEATURES = [
    "glare_frac", "R_norm_p50", "a_mean", "R_p50", "R_p10", "RG", "S_p50",
    "gray_p90", "gray_kurt", "gray_std", "gray_mean", "B_mean", "B_p10", "B_p75",
    "G_kurt", "mean_vesselness", "p90_vesselness", "skeleton_len_per_area",
    "branchpoint_density", "tortuosity_mean", "vessel_area_fraction"
]

# --- Roboflow Settings ---
DEFAULT_MODEL_ID  = "eye-conjunctiva-detector/2"
DEFAULT_CLASS     = "conjunctiva"
DEFAULT_CONF      = 25

# ---------- DATA PERSISTENCE HELPERS ---------- #

def load_data():
    """Loads the user data from the JSON file."""
    if DATA_FILE.exists():
        with open(DATA_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_data(data):
    """Saves the user data to the JSON file."""
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def get_user_history(user_id):
    """Retrieves the history for a specific user, sorted by date."""
    data = load_data()
    history = data.get(user_id, [])
    return sorted(history, key=lambda x: x['timestamp'], reverse=True)

def add_reading_to_history(user_id, reading_data):
    """Adds a new reading to a user's history."""
    data = load_data()
    if user_id not in data:
        data[user_id] = []
    data[user_id].append(reading_data)
    save_data(data)

# ---------- CORE ML/IMAGE PROCESSING (from original script) ---------- #

# --- Image Utils ---
def exif_upright(pil_img: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(pil_img).convert("RGB")

def pil_to_jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()

def to_b64_jpeg(img: Image.Image) -> str:
    b64_bytes = base64.b64encode(pil_to_jpeg_bytes(img, quality=90))
    return b64_bytes.decode("utf-8")

def b64_to_pil(b64_str: str) -> Image.Image:
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes))

# --- Roboflow ---
def roboflow_detect_b64(b64_str: str, model_id: str, api_key: str, conf_0_100: int, timeout: float = 60.0) -> dict:
    url = f"https://detect.roboflow.com/{model_id}"
    params = {"api_key": api_key, "confidence": str(conf_0_100), "format": "json"}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    resp = requests.post(url, params=params, data=b64_str, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

def select_best_box(preds, target_class: str):
    if not preds: return None
    target = [p for p in preds if str(p.get("class", "")).lower() == target_class.lower()]
    return max(target, key=lambda p: p.get("confidence", 0.0)) if target else max(preds, key=lambda p: p.get("confidence", 0.0))

def crop_from_box(pil_img: Image.Image, box: dict) -> Image.Image:
    x, y, w, h = float(box["x"]), float(box["y"]), float(box["width"]), float(box["height"])
    left, top = int(round(x - w / 2.0)), int(round(y - h / 2.0))
    right, bottom = int(round(x + w / 2.0)), int(round(y + h / 2.0))
    return pil_img.crop((max(0, left), max(0, top), min(pil_img.width, right), min(pil_img.height, bottom)))

# --- Glare Helpers ---
def detect_glare_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    S, V = hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    mask_hsv = (V > 0.90) & (S < 0.25)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    hi = float(np.quantile(gray, 0.995))
    mask_gray = gray >= hi
    mask = cv2.morphologyEx((mask_hsv | mask_gray).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

def inpaint_glare(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB_BGR)
    out = cv2.inpaint(bgr, (mask.astype(np.uint8) * 255), inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return cv2.cvtColor(out, cv2.COLOR_BGR_RGB)

# --- Feature Extraction ---
def compute_baseline_features(pil_img: Image.Image) -> dict:
    rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    R, G, B = rgb[..., 0].astype(np.float32), rgb[..., 1].astype(np.float32), rgb[..., 2].astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    S = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1].astype(np.float32) / 255.0
    a = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)[..., 1].astype(np.float32) - 128.0
    R_norm = R / (R + G + B + 1e-6)
    return {
        "R_p50": float(np.percentile(R, 50)), "R_norm_p50": float(np.percentile(R_norm, 50)),
        "a_mean": float(np.mean(a)), "R_p10": float(np.percentile(R, 10)),
        "gray_mean": float(np.mean(gray)), "RG": float(np.mean(R) / (np.mean(G) + 1e-6)),
        "gray_kurt": float(kurtosis(gray.ravel(), fisher=False)), "gray_p90": float(np.percentile(gray, 90)),
        "S_p50": float(np.percentile(S, 50)), "B_p10": float(np.percentile(B, 10)),
        "B_mean": float(np.mean(B)), "gray_std": float(np.std(gray)),
        "B_p75": float(np.percentile(B, 75)), "G_kurt": float(kurtosis(G.ravel(), fisher=False)),
    }

def vascularity_features_from_conjunctiva(rgb_u8: np.ndarray, black_ridges: bool = True, min_size: int = 50, area_threshold: int = 50) -> dict:
    g = rgb_u8[..., 1].astype(np.uint8)
    g_eq = exposure.equalize_adapthist(g, clip_limit=0.01)
    vmap = filters.frangi(g_eq, sigmas=np.arange(1, 6, 1), alpha=0.5, beta=0.5, black_ridges=black_ridges)
    vmap = (vmap - vmap.min()) / (np.ptp(vmap) + 1e-8)
    mask = vmap > filters.threshold_otsu(vmap)
    mask = morphology.remove_small_objects(mask, min_size=min_size)
    mask = morphology.remove_small_holes(mask, area_threshold=area_threshold)
    skel = skeletonize(mask)
    area = float(mask.shape[0] * mask.shape[1])
    neigh = convolve(skel.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode='constant', cval=0)
    branches = ((skel) & (neigh >= 4))
    lbl, torts = measure.label(skel, connectivity=2), []
    for region in measure.regionprops(lbl):
        coords = np.array(region.coords)
        if coords.shape[0] < 10: continue
        chord = np.linalg.norm(coords.max(0) - coords.min(0)) + 1e-8
        torts.append(float(coords.shape[0]) / chord)
    return {
        "vessel_area_fraction": float(mask.sum()) / area, "mean_vesselness": float(vmap.mean()),
        "p90_vesselness": float(np.percentile(vmap, 90)), "skeleton_len_per_area": float(skel.sum()) / area,
        "branchpoint_density": float(branches.sum()) / area, "tortuosity_mean": float(np.mean(torts)) if torts else 1.0,
    }

# --- Model Loading (cached) ---
@st.cache_resource(show_spinner="Loading AI Model...")
def load_hb_model():
    path = HB_ESTIMATION_RUN_DIR / "hb_rf.joblib"
    if not path.exists(): raise FileNotFoundError(f"Hb model not found at: {path}")
    return joblib.load(path)

# ---------- UI & APP LOGIC ---------- #

def get_hb_classification(hb_value):
    """Returns a color and text description for an Hb value."""
    if hb_value < 8: return "red", "Severe Anemia Suspected 🔴"
    if hb_value < 10: return "orange", "Moderate Anemia Suspected 🟠"
    if hb_value < 12: return "yellow", "Mild Anemia Suspected 🟡"
    return "green", "Likely Normal 🟢"

def run_analysis_pipeline(pil_image, api_key):
    """Full pipeline from image to Hb prediction."""
    # 1. Roboflow Detection
    b64 = to_b64_jpeg(pil_image)
    rf_json = roboflow_detect_b64(b64, model_id=DEFAULT_MODEL_ID, api_key=api_key, conf_0_100=int(DEFAULT_CONF))
    best = select_best_box(rf_json.get("predictions", []), target_class=DEFAULT_CLASS)
    if best is None:
        st.error("Could not detect the conjunctiva. Please try again with a clearer, brighter photo.")
        return None
    
    crop = crop_from_box(pil_image, best)
    
    # 2. Feature Extraction and Prediction
    rgb = np.array(crop.convert("RGB"), dtype=np.uint8)
    glare_mask = detect_glare_mask(rgb)
    rgb_proc = inpaint_glare(rgb, glare_mask) if glare_mask.sum() > 0 else rgb
    
    feats = {"glare_frac": float(glare_mask.mean())}
    feats.update(compute_baseline_features(Image.fromarray(rgb_proc)))
    feats.update(vascularity_features_from_conjunctiva(rgb_proc))

    rgr = load_hb_model()
    x_vec = np.array([[feats.get(f, 0.0) for f in HB_FEATURES]], dtype=np.float32)
    hb_pred = float(rgr.predict(x_vec)[0])

    return {
        "hb_value": hb_pred,
        "features": {k: round(v, 4) for k, v in feats.items()},
        "crop_b64": to_b64_jpeg(crop),
        "timestamp": datetime.now().isoformat()
    }

def render_login_page():
    st.header("Welcome to your Hb Tracker")
    st.write("A simple way to track your estimated Hemoglobin levels over time.")
    user_id = st.text_input("Please enter your Name or a unique ID to begin:", key="user_id_input", placeholder="e.g., Patient123")
    
    if st.button("Login / Register", use_container_width=True):
        if user_id:
            st.session_state.user_id = user_id
            st.session_state.page = "dashboard" # Go directly to dashboard
            st.rerun()
        else:
            st.warning("Please enter a Name or ID.")

def render_dashboard_page():
    st.title(f"Health Dashboard")
    st.caption(f"Showing results for user: **{st.session_state.user_id}**")
    
    history = get_user_history(st.session_state.user_id)
    
    # --- LATEST READING METRIC ---
    with st.container(border=True):
        if not history:
            st.info("Your latest reading will appear here once you take your first measurement.")
        else:
            latest = history[0]
            hb = latest['hb_value']
            _, text = get_hb_classification(hb)
            st.metric(label="Latest Estimated Hb", value=f"{hb:.2f} g/dL", help=text)

    # --- TAKE NEW READING EXPANDER ---
    with st.expander("📸 Take a New Reading"):
        rf_api_key = st.text_input("Roboflow API Key", value=os.getenv("ROBOFLOW_API_KEY", ""), type="password")
        
        uploaded_photo = st.camera_input(
            "Take a clear, well-lit selfie of your eye.",
            help="Gently pull down your lower eyelid to show the conjunctiva."
        )
        
        if uploaded_photo:
            if not rf_api_key:
                st.warning("Please enter your Roboflow API Key to proceed.")
            else:
                with st.spinner("Analyzing your photo... This may take a moment."):
                    try:
                        pil_full = exif_upright(Image.open(uploaded_photo))
                        analysis_result = run_analysis_pipeline(pil_full, rf_api_key)
                        
                        if analysis_result:
                            add_reading_to_history(st.session_state.user_id, analysis_result)
                            st.success("Analysis complete and result saved!")
                            st.rerun() # Rerun to update the dashboard instantly
                    except Exception as e:
                        st.error(f"An error occurred during analysis: {e}")

    # --- TREND CHART & HISTORY ---
    if not history:
        st.info("Your trend chart and history will be displayed here after you save your first reading.")
    else:
        st.header("📊 Your Hemoglobin Trend")
        with st.container(border=True):
            df = pd.DataFrame(history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            fig = px.line(
                df, x='timestamp', y='hb_value',
                title='Your Hb Trend Over Time', markers=True,
                labels={'timestamp': 'Date', 'hb_value': 'Estimated Hb (g/dL)'}
            )
            fig.update_traces(line=dict(color='royalblue', width=2), marker=dict(size=8))
            st.plotly_chart(fig, use_container_width=True)

        st.header("📋 Your Reading History")
        
        # Controls
        col1, col2 = st.columns(2)
        with col1:
            doctor_view = st.toggle("🔬 Doctor View", help="Show detailed technical data for each reading.")
        with col2:
            if st.button("📄 Export Report", use_container_width=True):
                st.session_state.page = "report"
                st.rerun()

        # History Cards
        for reading in history:
            dt_obj = datetime.fromisoformat(reading['timestamp'])
            date_str = dt_obj.strftime("%B %d, %Y at %I:%M %p")
            hb = reading['hb_value']
            color, text = get_hb_classification(hb)
            
            with st.container(border=True):
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.image(b64_to_pil(reading['crop_b64']), use_column_width='always')
                with c2:
                    st.caption(f"{date_str}")
                    st.markdown(f"#### <font color='{color}'>{hb:.2f} g/dL</font>", unsafe_allow_html=True)
                    st.markdown(f"*{text}*")
                
                if doctor_view:
                    with st.expander("Show Technical Details"):
                        features_df = pd.DataFrame([reading['features']]).T
                        features_df.columns = ["Value"]
                        st.dataframe(features_df, use_container_width=True)

def render_report_page():
    """Generates a clean page for printing to PDF via browser."""
    st.set_page_config(layout="centered")
    history = get_user_history(st.session_state.user_id)
    
    # Hide non-report elements using CSS
    st.markdown("""
        <style>
            #MainMenu, footer, .stHeader, .stButton, .stToggle { display: none !important; }
            .main .block-container { padding-top: 2rem; }
        </style>
    """, unsafe_allow_html=True)
    
    st.title(f"Hemoglobin Trend Report for {st.session_state.user_id}")
    st.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.info("To save this report, use your browser's print function (Ctrl+P or Cmd+P) and select 'Save as PDF'.")
    st.divider()
    
    if not history:
        st.warning("No data to report.")
    else:
        df = pd.DataFrame(history).sort_values('timestamp')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig = px.line(df, x='timestamp', y='hb_value', title='Hb Trend', markers=True, labels={'timestamp': 'Date', 'hb_value': 'Estimated Hb (g/dL)'})
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Readings Data")
        report_df = df[['timestamp', 'hb_value']].rename(columns={'timestamp': 'Date and Time', 'hb_value': 'Estimated Hb (g/dL)'})
        report_df['Date and Time'] = report_df['Date and Time'].dt.strftime('%Y-%m-%d %H:%M')
        report_df['Estimated Hb (g/dL)'] = report_df['Estimated Hb (g/dL)'].round(2)
        st.dataframe(report_df, use_container_width=True, hide_index=True)
        
    if st.button("← Back to App"):
        st.session_state.page = "dashboard"
        st.rerun()

# ---------- MAIN APP ROUTER ---------- #
def main():
    st.set_page_config(page_title="Hb Tracker", layout="wide")
    
    if "page" not in st.session_state:
        st.session_state.page = "login"
    
    if st.session_state.page == "login":
        render_login_page()
    elif st.session_state.page == "dashboard":
        render_dashboard_page()
    elif st.session_state.page == "report":
        render_report_page()

if __name__ == "__main__":
    main()