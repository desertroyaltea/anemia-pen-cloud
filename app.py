import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import knime
from inference_sdk import InferenceHTTPClient
from io import BytesIO

# --- KNIME & ROBOFLOW CONFIGURATION ---
REGRESSION_MODEL_PATH = 'hb_regression_model.zip'
CLASSIFICATION_MODEL_PATH = 'anemia_classification_model.zip'

# Roboflow API configuration
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="jMhyBQxeQvj69nttV0mN" # Your Roboflow API key
)
ROBOFLOW_MODEL_ID = "eye-conjunctiva-detector/2"

# --- SIMPLIFIED FEATURE EXTRACTION FUNCTIONS ---

def calculate_first_order_statistics(image_array):
    """Calculates basic statistics for an image."""
    features = {}
    features['Min'] = np.min(image_array)
    features['Max'] = np.max(image_array)
    features['Mean'] = np.mean(image_array)
    features['Std Dev'] = np.std(image_array)
    features['Variance'] = np.var(image_array)
    features['Skewness'] = pd.Series(image_array.flatten()).skew()
    features['Kurtosis'] = pd.Series(image_array.flatten()).kurtosis()
    return features

def calculate_histogram(image_array, bins=64):
    """Calculates the histogram of an image."""
    hist, _ = np.histogram(image_array.flatten(), bins=bins, range=(0, 255))
    return hist

def extract_all_features(image):
    """Main function to extract all required features from a PIL image."""
    img_array_color = np.array(image.convert('RGB'))
    img_array_gray = np.array(image.convert('L'))

    first_order_stats = calculate_first_order_statistics(img_array_gray)

    hist_features = {}
    channels = ['Red', 'Green', 'Blue']
    for i, channel in enumerate(channels):
        hist = calculate_histogram(img_array_color[:, :, i])
        for j, val in enumerate(hist):
            hist_features[f'Hist_{channel}_bin_{j}'] = val

    all_features = {**first_order_stats, **hist_features}
    
    feature_df = pd.DataFrame([all_features])
    
    # --- THIS IS THE CRITICAL FIX ---
    # Sort the columns alphabetically to match the KNIME model's input order.
    feature_df = feature_df.reindex(sorted(feature_df.columns), axis=1)
    
    return feature_df

# --- STREAMLIT APP LAYOUT ---

st.set_page_config(layout="wide")
st.title("👁️ Anemia Screening via Conjunctiva Image")
st.write("Upload an image of an eye with the lower conjunctiva visible. The app will automatically detect and crop the area, then analyze it using our trained models.")

uploaded_file = st.file_uploader("Choose an eye image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        original_image = Image.open(uploaded_file).convert("RGB")
        st.image(original_image, caption='Original Uploaded Image', width=300)

        st.info("Detecting conjunctiva using Roboflow...")
        result = CLIENT.infer(original_image, model_id=ROBOFLOW_MODEL_ID)
        
        if result['predictions']:
            pred = result['predictions'][0]
            x, y, width, height = pred['x'], pred['y'], pred['width'], pred['height']
            
            x1 = int(x - width / 2)
            y1 = int(y - height / 2)
            x2 = int(x + width / 2)
            y2 = int(y + height / 2)
            
            cropped_image = original_image.crop((x1, y1, x2, y2))
            resized_image = cropped_image.resize((128, 128))
            
            st.success("Conjunctiva detected and cropped successfully!")
            
            st.image(resized_image, caption='Cropped & Resized Conjunctiva (128x128)')
            
            with st.spinner('Extracting image features...'):
                features_df = extract_all_features(resized_image)
            
            st.subheader("Choose an Analysis")
            
            if st.button("Estimate Hb Level"):
                with st.spinner('Running regression model...'):
                    with knime.Workflow(REGRESSION_MODEL_PATH) as wf:
                        wf.data_table_inputs[0] = features_df
                        wf.execute()
                        prediction_table = wf.data_table_outputs[0]
                    
                    hb_prediction = prediction_table.iloc[0]['Prediction (hb)']
                    st.metric(label="Predicted Hemoglobin Level", value=f"{hb_prediction:.2f} g/dL")

            if st.button("Screen for Anemia"):
                with st.spinner('Running classification model...'):
                    with knime.Workflow(CLASSIFICATION_MODEL_PATH) as wf:
                        wf.data_table_inputs[0] = features_df
                        wf.execute()
                        prediction_table = wf.data_table_outputs[0]
                    
                    anemia_probability = prediction_table.iloc[0]['P (Status=1)']
                    final_prediction = 1 if anemia_probability > 0.5 else 0
                    
                    if final_prediction == 1:
                        st.error(f"Anemia Detected (Probability: {anemia_probability:.1%})")
                    else:
                        st.success(f"No Anemia Detected (Probability of Anemia: {anemia_probability:.1%})")

            with st.expander("View Extracted Image Features"):
                st.dataframe(features_df)

        else:
            st.error("No conjunctiva was detected in the image.")

    except Exception as e:
        st.error(f"An error occurred: {e}")