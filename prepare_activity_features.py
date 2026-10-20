import pandas as pd

from app.ml.activity_features import (
    build_activity_features,
    FEATURE_COLUMNS,
)

# Read cleaned activity data
activity = pd.read_csv("data/processed/activity_cleaned.csv")

# Generate participant-level summary features
features = build_activity_features(activity)

# Read demographics containing chronological age
demo = pd.read_sas("data/raw/DEMO_H.xpt", format="xport")

# Only keep participant ID and age
demo = demo[["SEQN", "RIDAGEYR"]].copy()

# Match age to activity features
features = features.merge(demo, on="SEQN", how="inner")

# Keep participants aged 18 and older
features = features[features["RIDAGEYR"] >= 18].copy()

# Rename age for clarity
features = features.rename(columns={"RIDAGEYR": "age"})

# Remove incomplete model records
features = features.dropna(subset=["avg_daily_activity", "activity_variability", "age"])

# Save features
features.to_csv(
    "data/processed/activity_features.csv",
    index=False
)

print("Participants ready for training:", len(features))
print("\nFeature columns:")
print(FEATURE_COLUMNS)
print("\nFirst 5 rows:")
print(features.head())
print("\nFeature CSV saved!")