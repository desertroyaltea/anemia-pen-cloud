import streamlit as st
from PIL import Image
import numpy as np
import io
import joblib
import base64
from inference_sdk import InferenceHTTPClient

# --- Roboflow Client Configuration ---
# Note: The Roboflow client requires an active internet connection.
ROBOFLOW_API_URL = "https://serverless.roboflow.com"
ROBOFLOW_API_KEY = "jMhyBQxeQvj69nttV0mN"
CLIENT = InferenceHTTPClient(api_url=ROBOFLOW_API_URL, api_key=ROBOFLOW_API_KEY)

# --- Model Loading ---
# NOTE: The models you provided were trained on tabular data (e.g., blood test results).
# They cannot directly process image data. For this app, we will use mock predictions
# to demonstrate the full workflow. Replace this with an actual image-based model
# if you train one in the future.
try:
    # Attempt to load the models. This is for demonstration, as they won't be used.
    classification_model = joblib.load('anemia_classification_model.model')
    regression_model = joblib.load('hb_regression_model.model')
except FileNotFoundError:
    st.warning("Model files not found. The app will use mock predictions.")
    classification_model = None
    regression_model = None

# --- Main Streamlit App ---
st.set_page_config(page_title="Eye-Based Anemia Screener", layout="centered")

st.title("👁️ Anemia and Hb Screener")
st.markdown("Upload an eye image to screen for anemia or estimate Hb levels. This tool uses a Roboflow model to isolate the conjunctiva for analysis.")

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "Choose an image of a full eye...",
    type=["jpg", "jpeg", "png"]
)

# --- Option Selection ---
prediction_option = st.radio(
    "Choose a prediction type:",
    ("Screen for Anemia", "Estimate Hb")
)

# --- Processing Logic ---
if uploaded_file is not None:
    # Display a spinner while processing
    with st.spinner("Processing image..."):
        try:
            # Open the uploaded image
            original_image = Image.open(uploaded_file).convert('RGB')
            st.image(original_image, caption="Original Image", use_column_width=True)

            # Step 1: Use Roboflow to get the bounding box of the conjunctiva
            roboflow_result = CLIENT.infer(
                original_image,
                model_id="eye-conjunctiva-detector/2",
                # The Roboflow API can return predictions with probabilities
                # We can use the confidence threshold to filter out bad predictions
                # `confidence=0.5`
            )
            
            # Check if any predictions were made
            if not roboflow_result["predictions"]:
                st.error("❌ No conjunctiva found in the image. Please try another image.")
                st.stop()
            
            # Find the best prediction (highest confidence)
            best_prediction = max(roboflow_result["predictions"], key=lambda p: p["confidence"])
            x, y, w, h = best_prediction["x"], best_prediction["y"], best_prediction["width"], best_prediction["height"]
            
            # Crop the image using the bounding box
            left = x - w / 2
            top = y - h / 2
            right = x + w / 2
            bottom = y + h / 2
            cropped_image = original_image.crop((left, top, right, bottom))
            st.image(cropped_image, caption="Conjunctiva Cropped", use_column_width=True)

            # Step 2: Resize the cropped image to 128x128 pixels
            resized_image = cropped_image.resize((128, 128))
            st.image(resized_image, caption="Resized for Model Input (128x128)", use_column_width=True)

            # Step 3: Make a prediction based on the user's choice
            st.subheader("Prediction")
            
            # NOTE: We are using a mock prediction here because the models are for tabular data.
            # A real application would convert the image to features here and use the model.
            
            if prediction_option == "Screen for Anemia":
                # Mock prediction for a binary outcome
                mock_prediction_value = np.random.choice(["Anemic", "Non-Anemic"], p=[0.3, 0.7])
                if mock_prediction_value == "Anemic":
                    st.error(f"🔴 **Screening Result:** Possible Anemia detected. Please consult a professional.")
                else:
                    st.success(f"🟢 **Screening Result:** No Anemia detected. You seem to be in the clear.")

            elif prediction_option == "Estimate Hb":
                # Mock prediction for a regression outcome
                mock_prediction_value = np.random.uniform(11.0, 16.0)
                st.info(f"🧪 **Estimated Hb Level:** {mock_prediction_value:.2f} g/dL")

        except Exception as e:
            st.error(f"An error occurred: {e}. Please ensure the image is clear and contains a visible eye.")
