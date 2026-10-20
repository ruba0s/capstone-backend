import pandas as pd

FEATURE_COLUMNS = [
    "avg_daily_activity",
    "activity_variability",
    "avg_active_minutes",
]

def build_activity_features(df):
    df = df.copy()

    # Average movement per valid recorded minute
    df["daily_activity"] = df["PAXMTSD"] / df["PAXVMD"]

    # One row per participant
    features = df.groupby("SEQN").agg(
        avg_daily_activity=("daily_activity", "mean"),
        activity_variability=("daily_activity", "std"),
        avg_awake_wear_minutes=("PAXWWMD", "mean"),
        awake_wear_variability=("PAXWWMD", "std"),
        valid_days=("PAXDAYD", "nunique"),
    ).reset_index()

    # Require at least 4 usable days per participant
    features = features[features["valid_days"] >= 4].copy()

    return features