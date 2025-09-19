import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve

# --- CONFIGURATION ---
# Make sure your CSV file name from the best model (run_20250914_224341) matches this.
FILE_PATH = 'predictions.csv' 
# Define the clinical sensitivity targets you want to evaluate
SENSITIVITY_TARGETS = [0.80, 0.85, 0.90, 0.95]
# --- END CONFIGURATION ---

try:
    df = pd.read_csv(FILE_PATH)
    df.columns = df.columns.str.strip()
    
    y_true = df['status_true']
    y_proba = df['status_proba']

    # Calculate ROC curve points
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    
    # Calculate Sensitivity (same as tpr) and Specificity
    sensitivity = tpr
    specificity = 1 - fpr
    
    # --- 1. Find the Best Mathematical Balance (Maximizing G-Mean) ---
    gmean = np.sqrt(sensitivity * specificity)
    best_gmean_idx = np.argmax(gmean)
    best_threshold_gmean = thresholds[best_gmean_idx]
    sensitivity_at_gmean = sensitivity[best_gmean_idx]
    specificity_at_gmean = specificity[best_gmean_idx]

    print("--- Clinical Tuning Dashboard ---")
    print(f"Analyzing model predictions from: {FILE_PATH}\n")
    
    print("## Best Mathematical Balance ##")
    print("This threshold offers the best trade-off between both metrics.")
    print(f"   - Optimal Threshold: {best_threshold_gmean:.2f}")
    print(f"   - Sensitivity: {sensitivity_at_gmean:.1%}")
    print(f"   - Specificity: {specificity_at_gmean:.1%}")
    print("-" * 40)

    # --- 2. Evaluate Performance at Specific Clinical Targets ---
    print("\n## Clinical Scenarios ##")
    print("This table shows the Specificity (false alarm rate) you can expect for different Sensitivity goals.")
    
    results = []
    for target in SENSITIVITY_TARGETS:
        # Find the first index where sensitivity is >= target
        indices = np.where(sensitivity >= target)[0]
        if len(indices) > 0:
            idx = indices[0]
            threshold_at_target = thresholds[idx]
            specificity_at_target = specificity[idx]
            results.append({
                "Sensitivity Target": f">{target:.0%}",
                "Required Threshold": f"<{threshold_at_target:.2f}",
                "Expected Specificity": f"{specificity_at_target:.1%}"
            })
        else:
            results.append({
                "Sensitivity Target": f">{target:.0%}",
                "Required Threshold": "N/A",
                "Expected Specificity": "Not Achievable"
            })

    # Print results in a formatted table
    print(f"{'Sensitivity Target':<20} | {'Required Threshold':<20} | {'Expected Specificity':<20}")
    print("-" * 68)
    for res in results:
        print(f"{res['Sensitivity Target']:<20} | {res['Required Threshold']:<20} | {res['Expected Specificity']:<20}")


except FileNotFoundError:
    print(f"Error: The file '{FILE_PATH}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")