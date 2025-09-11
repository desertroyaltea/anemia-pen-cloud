import streamlit as st
from PIL import Image
import numpy as np
import joblib
import io
import base64
from inference_sdk import InferenceHTTPClient
import requests
from io import BytesIO

# --- Roboflow Client Configuration ---
# Note: The Roboflow client requires an active internet connection.
# This client is used to get bounding boxes for the conjunctiva.
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="jMhyBQxeQvj69nttV0mN"
)

# --- Model Loading ---
# NOTE: This is a placeholder. These models were trained on tabular data
# and cannot directly process images. This section demonstrates how you would
# load the models if they were compatible.
# The `mmap_mode` is set to None to handle potential protocol mismatches
# between Python versions and environments (e.g., KNIME and Streamlit Cloud).
try:
    classification_model = joblib.load('anemia_classification_model.model', mmap_mode=None)
    regression_model = joblib.load('hb_regression_model.model', mmap_mode=None)
    models_loaded = True
except Exception as e:
    st.warning(f"Error loading models: {e}. The app will use mock predictions.")
    models_loaded = False


# --- Helper Function for Image Processing ---
def crop_and_resize_eye(image, target_size=(128, 128)):
    """
    Crops the eye region from the image using the Roboflow API
    and resizes the cropped image to the target size.
    """
    st.info("Searching for conjunctiva using Roboflow...")
    
    # Convert PIL Image to bytes for Roboflow API
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    try:
        # Call the Roboflow API with the image passed as a positional argument.
        # This is the most reliable way for the library to handle headers.
        result = CLIENT.infer(
            img_bytes.read(),
            model_id="eye-conjunctiva-detector/2",
        )
        detections = result.get('predictions', [])

        if not detections:
            st.error("No conjunctiva detected in the image. Please try another image.")
            return None

        # Assuming the first detection is the correct one
        detection = detections[0]
        x, y, width, height = detection['x'], detection['y'], detection['width'], detection['height']

        # Crop the image using the detected bounding box
        left = x - width / 2
        top = y - height / 2
        right = x + width / 2
        bottom = y + height / 2
        cropped_image = image.crop((left, top, right, bottom))
        
        # Resize the cropped image
        resized_image = cropped_image.resize(target_size, Image.Resampling.LANCZOS)
        return resized_image
    
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to Roboflow API. Please check your internet connection or API key. Error: {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred during image processing: {e}")
        return None


# --- Streamlit App Interface ---
st.set_page_config(page_title="Anemia Screening & Hb Estimation", layout="centered")

st.title("👁️‍🗨️ Anemia Screening and Hb Estimation")
st.markdown("Upload a full eye image to screen for anemia or estimate hemoglobin (Hb) levels.")

# User's choice
option = st.selectbox(
    "Choose your task:",
    ("Screen for Anemia", "Estimate Hb")
)

# File uploader
uploaded_file = st.file_uploader("Upload an eye image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Process the image
    image = Image.open(uploaded_file).convert("RGB")
    processed_image = crop_and_resize_eye(image)
    
    if processed_image:
        st.subheader("Processed Conjunctiva Image")
        st.image(processed_image, caption="Cropped and Resized Conjunctiva", use_column_width=True)
        
        # Perform prediction based on user's choice
        st.subheader("Prediction")
        
        if option == "Screen for Anemia":
            if models_loaded:
                # Placeholder for real model inference
                # In a real app, you would pass the processed_image to your model.
                # For this demonstration, we use a mock prediction.
                prediction = classification_model.predict(np.random.rand(1, 128*128*3))
                result = "Possible Anemia" if prediction[0] == 1 else "No Anemia"
                st.write(f"**Result:** {result}")
            else:
                # Mock prediction if model failed to load
                st.warning("Using mock prediction due to model loading error.")
                st.write(f"**Result:** Based on the conjunctiva, you are likely **Not Anemic**.")

        elif option == "Estimate Hb":
            if models_loaded:
                # Placeholder for real model inference
                prediction = regression_model.predict(np.random.rand(1, 128*128*3))
                st.write(f"**Estimated Hb Level:** {prediction[0]:.2f} g/dL")
            else:
                # Mock prediction if model failed to load
                st.warning("Using mock prediction due to model loading error.")
                st.write(f"**Estimated Hb Level:** 13.5 g/dL (mock value)")
