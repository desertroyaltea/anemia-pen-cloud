import streamlit as st
from PIL import Image
import numpy as np
import joblib
import io
import requests
import base64
from io import BytesIO

# --- Roboflow Client Configuration ---
API_URL = "https://detect.roboflow.com/eye-conjunctiva-detector/2"
API_KEY = "jMhyBQxeQvj69nttV0mN"

# --- Helper Function for Image Processing ---
def crop_eye(image):
    """
    Crops the eye region from the image using the Roboflow API
    and returns the cropped image without resizing.
    """
    st.info("Searching for conjunctiva using Roboflow...")
    
    # Convert PIL Image to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    # Encode the image data to base64
    base64_image = base64.b64encode(img_bytes.read()).decode('utf-8')
    
    try:
        # Make a direct request to the Roboflow API with a JSON payload
        response = requests.post(
            f"{API_URL}?api_key={API_KEY}",
            json={"image": {"type": "base64", "value": base64_image}}
        )
        response.raise_for_status() # Raise an exception for bad status codes
        result = response.json()
        
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
        
        return cropped_image
    
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to Roboflow API. Please check your internet connection or API key. Error: {e}")
        st.error(e)
        return None
    except Exception as e:
        st.error(f"An error occurred during image processing: {e}")
        return None


# --- Streamlit App Interface ---
st.set_page_config(page_title="Image Cropper", layout="centered")

st.title("Image Cropper and Conjunctiva Extractor")
st.markdown("Upload a full eye image to get the cropped conjunctiva region.")

# File uploader
uploaded_file = st.file_uploader("Upload an eye image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Process the image
    image = Image.open(uploaded_file).convert("RGB")
    processed_image = crop_eye(image)
    
    if processed_image:
        st.subheader("Processed Conjunctiva Image")
        st.image(processed_image, caption="Cropped Conjunctiva", use_column_width=True)
