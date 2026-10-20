"""
Clean the NHANES 2013-2014 sleep questionnaire -> data/processed/sleep_data.csv

Inputs (put in data/raw/):
    SLQ_H.xpt   sleep questionnaire
    DEMO_H.xpt  demographics (chronological age); same file the activity model uses

Run from the repo root:
    python clean_sleep.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/sleep_data.csv")
FEATURES_PATH = Path("data/processed/sleep_features.md")

# Team decision pending: these two are self-reported doctor history, which a phone
# health app does not measure. False keeps to the ground rule (sleep hours only).
INCLUDE_SELF_REPORT_FLAGS = False


def main() -> None:
    sleep = pd.read_sas(RAW_DIR / "SLQ_H.xpt", format="xport")
    demo = pd.read_sas(RAW_DIR / "DEMO_H.xpt", format="xport")
    print("Sleep rows:", len(sleep), "| columns:", sleep.columns.tolist())

    # 1. Keep only what we need.
    sleep = sleep[["SEQN", "SLD010H", "SLQ050", "SLQ060"]].copy()
    demo = demo[["SEQN", "RIDAGEYR"]].copy()

    # 2. NHANES stores "refused" (77/7) and "don't know" (99/9) as numbers.
    #    Turn them into missing values instead of letting them look like real data.
    sleep["sleep_hours"] = sleep["SLD010H"].where(sleep["SLD010H"].between(1, 12))
    # 12 means "12 or more hours" (top-coded); that is fine for modeling.
    for raw, new in [("SLQ050", "trouble_sleeping"), ("SLQ060", "told_sleep_disorder")]:
        sleep[new] = sleep[raw].map({1.0: 1.0, 2.0: 0.0})  # 1 = yes, 2 = no, rest -> NaN

    # 3. Attach age. Inner merge keeps only people who appear in both files.
    df = sleep.merge(demo, on="SEQN", how="inner")

    # 4. Adults only, same cutoff as the activity data.
    df = df[df["RIDAGEYR"] >= 18]

    # 5. One row per participant, IDs as whole numbers (73557, not 73557.0), so the
    #    shared split and merges with other datasets treat IDs identically.
    df = df.drop_duplicates(subset="SEQN")
    df["SEQN"] = df["SEQN"].astype(int)

    features = ["sleep_hours"]
    if INCLUDE_SELF_REPORT_FLAGS:
        features += ["trouble_sleeping", "told_sleep_disorder"]

    # 6. Drop people missing the age label or any feature (only a few percent).
    n_before = len(df)
    df = df.dropna(subset=["RIDAGEYR", *features])
    print(f"Dropped {n_before - len(df)} rows with missing values")

    df = df[["SEQN", "RIDAGEYR", *features]].reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    FEATURES_PATH.write_text(
        "# Sleep model features\n\n"
        "- ID: `SEQN`\n- Label: `RIDAGEYR` (age in years, adults 18+)\n- Features:\n"
        + "\n".join(f"  - `{c}`" for c in features) + "\n"
    )

    print("\nParticipants:", len(df))
    print("Age range:", df["RIDAGEYR"].min(), "to", df["RIDAGEYR"].max())
    print("Missing values:\n", df.isna().sum().to_string())
    print("\nSummary:\n", df.describe().T.to_string())
    print(f"\nSaved {OUT_PATH} and {FEATURES_PATH}")


if __name__ == "__main__":
    main()
