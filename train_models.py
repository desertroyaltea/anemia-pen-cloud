import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib

# Load the complete dataset exported from KNIME
print("Loading data...")
df = pd.read_csv("final_training_data.csv")

# --- Define Features and Targets ---
# The target columns are 'hb' and 'Status'
targets = ['hb', 'Status']
# The features are all columns that are not the targets
features = [col for col in df.columns if col not in targets]

X = df[features]
y_hb = df['hb']
y_status = df['Status']

# --- Train Regression Model for Hb ---
print("Training regression model for Hb...")
reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X, y_hb)

# Save the trained model to a file
joblib.dump(reg_model, 'hb_model.joblib')
print("Regression model saved as hb_model.joblib")

# --- Train Classification Model for Anemia Status ---
print("\nTraining classification model for Anemia...")
clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X, y_status)

# Save the trained model to a file
joblib.dump(clf_model, 'anemia_model.joblib')
print("Classification model saved as anemia_model.joblib")

print("\nModel training complete.")