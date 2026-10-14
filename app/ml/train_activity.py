from pathlib import Path
import pickle

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from app.ml.activity_features import FEATURE_COLUMNS

# Load participant-level features
df = pd.read_csv("data/processed/activity_final.csv")

# Inputs and target
X = df[FEATURE_COLUMNS]
y = df["age"]

# Split participants into training and testing groups
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

# Simple comparison: predict training-set median age
baseline_predictions = [y_train.median()] * len(y_test)
baseline_mae = mean_absolute_error(y_test, baseline_predictions)

print("Training participants:", len(X_train))
print("Testing participants:", len(X_test))
print(f"Activity model MAE: {mae:.2f} years")
print(f"Baseline MAE: {baseline_mae:.2f} years")

# Save trained model
artifact_dir = Path("app/ml/artifacts")
artifact_dir.mkdir(parents=True, exist_ok=True)

with open(artifact_dir / "activity_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save feature list
(artifact_dir / "activity_feature_list.txt").write_text(
    "\n".join(FEATURE_COLUMNS) + "\n"
)

# Record MAE
(artifact_dir / "activity_metrics.txt").write_text(
    f"Activity model MAE: {mae:.2f} years\n"
    f"Baseline MAE: {baseline_mae:.2f} years\n"
    f"Training participants: {len(X_train)}\n"
    f"Testing participants: {len(X_test)}\n"
)

print("\nModel, feature list, and metrics saved!")