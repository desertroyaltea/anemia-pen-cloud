import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import knime
from inference_sdk import InferenceHTTPClient
import requests
from io import BytesIO
from skimage.feature import graycomatrix, graycoprops
import mahotas

# --- KNIME & ROBOFLOW CONFIGURATION ---
# Make sure these model files are in the same folder as your app.py
REGRESSION_MODEL_PATH = 'hb_regression_model.zip'
CLASSIFICATION_MODEL_PATH = 'anemia_classification_model.zip'

# Roboflow API configuration
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="jMhyBQxeQvj69nttV0mN" # Your Roboflow API key
)
ROBOFLOW_MODEL_ID = "eye-conjunctiva-detector/2"

# --- FEATURE EXTRACTION FUNCTIONS ---
# These functions replicate what the "Image Features" node did in KNIME.

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
    hist, _ = np.histogram(image_array.flatten(), bins=bins, range=(0, 256))
    return hist

def calculate_haralick_features(gray_image):
    """Calculates Haralick texture features."""
    # Convert to 8-bit integer if not already
    if gray_image.dtype != np.uint8:
        gray_image = (gray_image / 256).astype(np.uint8)

    features = mahotas.features.haralick(gray_image).mean(axis=0)
    feature_names = [
        'ASM', 'Contrast', 'Correlation', 'Variance', 'IFDM', 'SumAverage',
        'SumVariance', 'SumEntropy', 'Entropy', 'DifferenceVariance',
        'DifferenceEntropy', 'ICM1', 'ICM2'
    ]
    return {name: val for name, val in zip(feature_names, features)}

def calculate_tamura_features(gray_image):
    """Calculates Tamura texture features."""
    if gray_image.dtype != np.uint8:
        gray_image = (gray_image / 256).astype(np.uint8)

    features = {}
    features['TamuraContrast'] = mahotas.features.tamura(gray_image, 0)
    features['TamuraDirectionality'] = mahotas.features.tamura(gray_image, 1)
    # Add other Tamura features if your KNIME model used them
    return features

def extract_all_features(image):
    """Main function to extract all required features from a PIL image."""
    # Convert image to numpy array
    img_array_color = np.array(image.convert('RGB'))
    img_array_gray = np.array(image.convert('L'))

    # 1. First Order Statistics on grayscale
    first_order_stats = calculate_first_order_statistics(img_array_gray)

    # 2. Histograms for each color channel
    hist_features = {}
    channels = ['Red', 'Green', 'Blue']
    for i, channel in enumerate(channels):
        hist = calculate_histogram(img_array_color[:, :, i])
        for j, val in enumerate(hist):
            hist_features[f'Hist_{channel}_bin_{j}'] = val

    # 3. Haralick texture features on grayscale
    haralick_features = calculate_haralick_features(img_array_gray)

    # 4. Tamura texture features on grayscale
    tamura_features = calculate_tamura_features(img_array_gray)

    # Combine all features into one dictionary
    all_features = {**first_order_stats, **hist_features, **haralick_features, **tamura_features}

    # Convert to a pandas DataFrame for the KNIME model
    feature_df = pd.DataFrame([all_features])
    return feature_df


# --- STREAMLIT APP LAYOUT ---

st.set_page_config(layout="wide")
st.title("👁️ Anemia Screening via Conjunctiva Image")

st.write("Upload an image of an eye with the lower conjunctiva visible. The app will automatically detect and crop the area, then analyze it using our trained models.")

uploaded_file = st.file_uploader("Choose an eye image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        original_image = Image.open(uploaded_file).convert("RGB")
        st.image(original_image, caption='Original Uploaded Image', width=300)

        # --- ROBOFLOW INFERENCE ---
        st.info("Detecting conjunctiva using Roboflow...")

        # Convert PIL image to bytes for the API
        buffered = BytesIO()
        original_image.save(buffered, format="JPEG")
        img_str = buffered.getvalue()

        # Call Roboflow API
        result = CLIENT.infer(img_str, model_id=ROBOFLOW_MODEL_ID)

        if result['predictions']:
            # Get the first prediction's bounding box
            pred = result['predictions'][0]
            x, y, width, height = pred['x'], pred['y'], pred['width'], pred['height']

            # Bounding box coordinates
            x1 = int(x - width / 2)
            y1 = int(y - height / 2)
            x2 = int(x + width / 2)
            y2 = int(y + height / 2)

            # Crop the conjunctiva
            cropped_image = original_image.crop((x1, y1, x2, y2))

            # Resize to 128x128 for our models
            resized_image = cropped_image.resize((128, 128))

            st.success("Conjunctiva detected and cropped successfully!")

            col1, col2 = st.columns(2)
            with col1:
                st.image(resized_image, caption='Cropped & Resized Conjunctiva (128x128)')

            # --- MODEL PREDICTION ---
            with st.spinner('Extracting image features...'):
                features_df = extract_all_features(resized_image)

            st.subheader("Choose an Analysis")

            if st.button("Estimate Hb Level"):
                with st.spinner('Running regression model...'):
                    # Load and execute the KNIME regression model
                    with knime.Workflow(REGRESSION_MODEL_PATH) as wf:
                        wf.data_table_inputs[0] = features_df
                        wf.execute()
                        prediction_table = wf.data_table_outputs[0]

                    hb_prediction = prediction_table.iloc[0]['Prediction (hb)']
                    st.metric(label="Predicted Hemoglobin Level", value=f"{hb_prediction:.2f} g/dL")
                    st.info("This is an estimation based on the visual features of the conjunctiva.")

            if st.button("Screen for Anemia"):
                with st.spinner('Running classification model...'):
                    # Load and execute the KNIME classification model
                    with knime.Workflow(CLASSIFICATION_MODEL_PATH) as wf:
                        wf.data_table_inputs[0] = features_df
                        wf.execute()
                        prediction_table = wf.data_table_outputs[0]

                    # Get the probability and make a final decision
                    anemia_probability = prediction_table.iloc[0]['P (Status=1)']
                    final_prediction = 1 if anemia_probability > 0.5 else 0

                    if final_prediction == 1:
                        st.error(f"Anemia Detected (Probability: {anemia_probability:.1%})")
                        st.warning("This screening suggests a high likelihood of anemia. Please consult a healthcare professional for a formal diagnosis.")
                    else:
                        st.success(f"No Anemia Detected (Probability of Anemia: {anemia_probability:.1%})")
                        st.info("This screening suggests a low likelihood of anemia.")

            # Display extracted features (optional)
            with st.expander("View Extracted Image Features"):
                st.dataframe(features_df)

        else:
            st.error("No conjunctiva was detected in the image. Please try another image with the eye clearly visible.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("Please ensure the uploaded file is a valid image and try again.")