import pandas as pd
from app.ml.registry import MODALITIES

OUT_PATH = MODALITIES["activity"]["data"]   # data/processed/activity.csv, set in registry.py

# Load existing features (includes participant age)
activity = pd.read_csv("data/processed/activity_features.csv")

# Load cleaned questionnaire activity minutes
questionnaire = pd.read_csv(
    "data/processed/activity_questionnaire.csv"
)

# Match participants using their IDs
merged = activity.merge(
    questionnaire,
    on="SEQN",
    how="inner",
    validate="one_to_one"
)

# Keep the three features and age
columns = [
    "SEQN",
    "avg_daily_activity",
    "activity_variability",
    "avg_active_minutes",
    "age"
]

merged = merged[columns].copy()

# Remove invalid or incomplete rows
merged = merged.dropna()

merged = merged[
    (merged["avg_daily_activity"] >= 0)
    & (merged["activity_variability"] >= 0)
    & (merged["avg_active_minutes"] >= 0)
]

# Save final model-ready data
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(OUT_PATH, index=False)

print("Participants in final dataset:", len(merged))
print("\nMissing values:")
print(merged.isnull().sum())

print("\nFirst 5 rows:")
print(merged.head())

print(f"\nSaved {OUT_PATH}")