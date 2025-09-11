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
    api_key="jMhyBQxeQvj69nttV0mN" 
)
ROBOFLOW_MODEL_ID = "eye-conjunctiva-detector/2"

# --- FEATURE EXTRACTION FUNCTIONS ---
def calculate_first_order_statistics(image_array):
    """Calculates the simplified set of statistics with KNIME's exact naming."""
    features = {}
    features['Min'] = np.min(image_array)
    features['Max'] = np.max(image_array)
    features['Mean'] = np.mean(image_array)
    features['Std Dev'] = np.std(image_array)
    features['Variance'] = np.var(image_array)
    features['Skewness'] = pd.Series(image_array.flatten()).skew()
    features['Kurtosis'] = pd.Series(image_array.flatten()).kurtosis()
    features['Sum'] = np.sum(image_array)
    features['Squares of Sum'] = np.sum(image_array)**2
    return features

def calculate_grayscale_histogram(image_array, bins=64):
    """Calculates the grayscale histogram with names that match KNIME."""
    hist, _ = np.histogram(image_array.flatten(), bins=bins, range=(0, 255))
    hist_features = {f'h_{i}': val for i, val in enumerate(hist)}
    return hist_features

def extract_all_features(image):
    """Main function to extract the simplified and matched feature set."""
    img_array_gray = np.array(image.convert('L'))

    first_order_stats = calculate_first_order_statistics(img_array_gray)
    histogram_features = calculate_grayscale_histogram(img_array_gray)

    all_features = {**first_order_stats, **histogram_features}
    
    # --- THIS IS THE CRITICAL FIX ---
    # Define the EXACT column order that the KNIME model expects.
    # This order is based on the KNIME Column Resorter's output.
    knime_column_order = [
        'Kurtosis', 'Max', 'Mean', 'Min', 'Skewness', 'Squares of Sum', 
        'Std Dev', 'Sum', 'Variance',
        'h_0', 'h_1', 'h_2', 'h_3', 'h_4', 'h_5', 'h_6', 'h_7', 'h_8', 'h_9',
        'h_10', 'h_11', 'h_12', 'h_13', 'h_14', 'h_15', 'h_16', 'h_17', 'h_18', 'h_19',
        'h_20', 'h_21', 'h_22', 'h_23', 'h_24', 'h_25', 'h_26', 'h_27', 'h_28', 'h_29',
        'h_30', 'h_31', 'h_32', 'h_33', 'h_34', 'h_35', 'h_36', 'h_37', 'h_38', 'h_39',
        'h_40', 'h_41', 'h_42', 'h_43', 'h_44', 'h_45', 'h_46', 'h_47', 'h_48', 'h_49',
        'h_50', 'h_51', 'h_52', 'h_53', 'h_54', 'h_55', 'h_56', 'h_57', 'h_58', 'h_59',
        'h_60', 'h_61', 'h_62', 'h_63'
    ]
    
    feature_df = pd.DataFrame([all_features])
    # Reorder the DataFrame to match KNIME's exact column order
    feature_df = feature_df[knime_column_order]
    
    # Force all data types to float to prevent mismatches
    feature_df = feature_df.astype(np.float64)
    
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
            
            x1, y1 = int(x - width / 2), int(y - height / 2)
            x2, y2 = int(x + width / 2), int(y + height / 2)
            
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
                    
                    prob_col_name = next((col for col in prediction_table.columns if 'P (' in col and '=1' in col), None)
                    if prob_col_name:
                        anemia_probability = prediction_table.iloc[0][prob_col_name]
                        final_prediction = 1 if anemia_probability > 0.5 else 0
                        if final_prediction == 1:
                            st.error(f"Anemia Detected (Probability: {anemia_probability:.1%})")
                        else:
                            st.success(f"No Anemia Detected (Probability of Anemia: {anemia_probability:.1%})")
                    else:
                        st.error("Could not find the probability column in the model output.")

            with st.expander("View Extracted Image Features"):
                st.dataframe(features_df)

        else:
            st.error("No conjunctiva was detected in the image.")

    except Exception as e:
        st.error(f"An error occurred: {e}")