import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, classification_report, recall_score
import joblib

# --- 1. Load the Data ---
try:
    df = pd.read_csv('final_training_data.csv')
except FileNotFoundError:
    print("Error: 'final_training_data.csv' not found. Please check the file path.")
    exit()

# --- 2. Prepare Data for Modeling ---
target_regression = 'hb'
target_classification = 'Status'

features = [
    col for col in df.columns
    if col not in [target_regression, target_classification, 'Row ID', 'image_filename']
]

X = df[features]
y_reg = df[target_regression]
y_cls = df[target_classification]

# --- 3. Split Data into Training and Testing Sets (One Time) ---
X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls = train_test_split(
    X, y_reg, y_cls, test_size=0.2, random_state=42
)

# --- 4. Train, Tune, and Save the REGRESSION Model ---
# This section remains unchanged as it performed well.
print("--- Building and Tuning Regression Model (Predicting Hb) ---")
reg_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
regressor_grid = GridSearchCV(RandomForestRegressor(random_state=42), reg_param_grid, cv=5, n_jobs=-1, verbose=1)
regressor_grid.fit(X_train, y_train_reg)
regressor = regressor_grid.best_estimator_
y_pred_reg = regressor.predict(X_test)
print("\nBest Regression Model Parameters:")
print(regressor_grid.best_params_)
print(f"R-squared (R²): {r2_score(y_test_reg, y_pred_reg):.4f}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test_reg, y_pred_reg):.4f}\n")
joblib.dump(regressor, 'hb_regressor_tuned.joblib')
print("✅ Tuned Regression model saved to hb_regressor_tuned.joblib\n")

# --- 5. Train, Tune, and Save the CLASSIFICATION Model (Optimizing for Recall) ---
print("--- Building and Tuning Classification Model (Anemia/Not Anemia) for Screening ---")

# Define parameter grid for tuning, including a 'balanced' class weight
cls_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'class_weight': [None, 'balanced']  # Try both balanced and default weights
}

# Perform Grid Search, but this time optimize for 'recall'
classifier_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    cls_param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1,
    scoring='recall'  # ⬅️ Change scoring to optimize for recall
)
classifier_grid.fit(X_train, y_train_cls)

classifier = classifier_grid.best_estimator_
y_pred_cls = classifier.predict(X_test)

print("\nBest Classification Model Parameters (Optimized for Recall):")
print(classifier_grid.best_params_)
print(f"Accuracy: {accuracy_score(y_test_cls, y_pred_cls):.4f}")
print(f"Recall: {recall_score(y_test_cls, y_pred_cls):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_cls, y_pred_cls, target_names=['Non-Anemic (0)', 'Anemic (1)']))

joblib.dump(classifier, 'anemia_classifier_screening.joblib')
print("✅ Screening-optimized Classification model saved to anemia_classifier_screening.joblib\n")