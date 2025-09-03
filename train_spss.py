import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load your exported CSV
df = pd.read_csv("spss_predictions.csv")

# Features (the same 14 you used in SPSS)
features = [
    "R_norm_p50","a_mean","R_p50","R_p10","RG","S_p50",
    "gray_p90","gray_kurt","gray_std","gray_mean",
    "B_mean","B_p10","B_p75","G_kurt"
]

X = df[features]
y = df["hb"]

# Train/test split just for checking
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train regression model
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "hemo_surrogate.joblib")

print("Model trained and saved as hemo_surrogate.joblib")
