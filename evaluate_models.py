import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, confusion_matrix, recall_score

# Load the complete dataset exported from KNIME
df = pd.read_csv("final_training_data.csv")

# Define features and targets
targets = ['hb', 'Status']
features = [col for col in df.columns if col not in targets]
X = df[features]
y_hb = df['hb']
y_status = df['Status']

# Split the data into an 80% training set and a 20% testing set
# We stratify on 'Status' to ensure the split is balanced, just like in KNIME
X_train, X_test, y_hb_train, y_hb_test, y_status_train, y_status_test = train_test_split(
    X, y_hb, y_status, test_size=0.20, random_state=42, stratify=y_status
)

print("--- Evaluating Regression Model (Hb Estimation) ---")
# Train the model on the training data
reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train, y_hb_train)
# Make predictions on the test data
hb_preds = reg_model.predict(X_test)

# Calculate and print the scores
r2 = r2_score(y_hb_test, hb_preds)
mae = mean_absolute_error(y_hb_test, hb_preds)
print(f"  R-squared: {r2:.3f}")
print(f"  Mean Absolute Error: {mae:.3f}")


print("\n--- Evaluating Classification Model (Anemia Screening) ---")
# Train the model on the training data
clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X_train, y_status_train)
# Make predictions on the test data
status_preds = clf_model.predict(X_test)

# Calculate and print the scores
accuracy = accuracy_score(y_status_test, status_preds)
# pos_label=1 ensures we calculate recall for the "Anemic" class
recall = recall_score(y_status_test, status_preds, pos_label=1) 

print(f"  Overall Accuracy: {accuracy:.1%}")
print(f"  Recall for 'Anemic' class: {recall:.1%}")