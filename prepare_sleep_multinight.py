"""
Build multi-night sleep features from the NHANES 2013-2014 wrist accelerometer day file.

Dr. Hao suggested using statistics over several nights (mean, min, max, ...) instead of one
self-reported number. PAXDAY_H.xpt, the same file the activity model uses, has one row per
participant per day, and PAXSWMD is the number of minutes the device classified as sleep that
day. So these features cover the same people as the activity data.

Things to know before trusting the numbers:
  - Sleep is estimated by an algorithm from the wrist accelerometer (CDC says it was checked
    against polysomnography), not self-reported.
  - Bad days are dropped: a day needs real awake wear time (PAXWWMD) and little non-wear
    (PAXNWMD). Without this, days where the device was off overnight (sleep looks like ~1 hour)
    or where the whole day was labelled sleep (1440 minutes) distort the min, max and std.
  - A "day" runs midnight to midnight, so one night's sleep is split across two days. Averaged
    over several days this approximates nightly sleep. Daytime naps are included.

Inputs (put in data/raw/):
    PAXDAY_H.xpt   daily accelerometer summaries (same file as clean_activity.py)
    DEMO_H.xpt     demographics (chronological age)

Output:
    data/processed/sleep_multinight.csv  (+ sleep_multinight_features.md)

Run from the repo root:
    python prepare_sleep_multinight.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/sleep_multinight.csv")
FEATURES_PATH = Path("data/processed/sleep_multinight_features.md")

# Same day rule as clean_activity.py: days 1 and 9 are partial, so use days 2-8.
FIRST_DAY, LAST_DAY = 2, 8
# Tunable: a day needs this many valid minutes (of 1440) or its sleep total would be too low.
MIN_VALID_MINUTES = 1200
# Tunable: the window. A person needs at least this many usable days (same rule as activity).
MIN_DAYS = 4
MIN_AGE = 18
# Tunable: a real day has awake wear time and little non-wear (same wake-wear idea as clean_activity.py).
MIN_WAKE_MINUTES = 600
MAX_NONWEAR_MINUTES = 60

FEATURES = ["mean_sleep_min", "min_sleep_min", "max_sleep_min", "std_sleep_min"]


def build_sleep_features(days: pd.DataFrame) -> pd.DataFrame:
    """One row per participant with sleep statistics over their usable days."""
    df = days[["SEQN", "PAXDAYD", "PAXVMD", "PAXSWMD", "PAXWWMD", "PAXNWMD"]].copy()

    # PAXDAYD can come back as bytes (b"2"), as it does in clean_activity.py
    df["PAXDAYD"] = pd.to_numeric(
        df["PAXDAYD"].map(lambda v: v.decode("utf-8") if isinstance(v, bytes) else v),
        errors="coerce",
    )

    # SAS stores a zero count as a tiny positive number (5.4e-79). These are minute counts, so
    # that means 0 minutes, NOT missing. (Turning them into NaN dropped nearly everyone.)
    for col in ["PAXVMD", "PAXSWMD", "PAXWWMD", "PAXNWMD"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[(df[col] > 0) & (df[col] < 1e-50), col] = 0.0

    df = df[df["PAXDAYD"].between(FIRST_DAY, LAST_DAY)]
    df = df.dropna(subset=["SEQN", "PAXDAYD", "PAXVMD", "PAXSWMD", "PAXWWMD", "PAXNWMD"])
    df = df[df["PAXVMD"] >= MIN_VALID_MINUTES]
    df = df[(df["PAXWWMD"] >= MIN_WAKE_MINUTES) & (df["PAXNWMD"] <= MAX_NONWEAR_MINUTES)]
    df = df.drop_duplicates(subset=["SEQN", "PAXDAYD"])

    per_person = df.groupby("SEQN").agg(
        mean_sleep_min=("PAXSWMD", "mean"),
        min_sleep_min=("PAXSWMD", "min"),
        max_sleep_min=("PAXSWMD", "max"),
        std_sleep_min=("PAXSWMD", "std"),
        usable_days=("PAXDAYD", "nunique"),
    ).reset_index()

    per_person = per_person[per_person["usable_days"] >= MIN_DAYS]
    return per_person.drop(columns="usable_days").reset_index(drop=True)


def main() -> None:
    days = pd.read_sas(RAW_DIR / "PAXDAY_H.xpt", format="xport")
    demo = pd.read_sas(RAW_DIR / "DEMO_H.xpt", format="xport")[["SEQN", "RIDAGEYR"]]
    print("Day rows:", len(days), "| people:", days["SEQN"].nunique())

    feats = build_sleep_features(days)
    print("People with at least", MIN_DAYS, "usable days:", len(feats))

    df = feats.merge(demo, on="SEQN", how="inner")
    df = df[df["RIDAGEYR"] >= MIN_AGE]
    df = df.dropna(subset=["RIDAGEYR", *FEATURES])
    df["SEQN"] = df["SEQN"].astype(int)  # whole-number IDs so merges and the shared split match
    df = df[["SEQN", "RIDAGEYR", *FEATURES]].reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    FEATURES_PATH.write_text(
        "# Multi-night sleep features\n\n"
        "- ID: `SEQN`\n- Label: `RIDAGEYR` (age in years, adults 18+)\n"
        "- Source: PAXSWMD (minutes classified as sleep) per day in PAXDAY_H, days 2-8,\n"
        f"  days with at least {MIN_VALID_MINUTES} valid minutes, {MIN_WAKE_MINUTES}+ awake-wear minutes and at most {MAX_NONWEAR_MINUTES} non-wear minutes, at least {MIN_DAYS} days per person\n"
        "- Features:\n" + "\n".join(f"  - `{c}` (minutes)" for c in FEATURES) + "\n"
    )

    print("\nParticipants:", len(df))
    print("Age range:", df["RIDAGEYR"].min(), "to", df["RIDAGEYR"].max())
    odd = int(((df["mean_sleep_min"] < 180) | (df["mean_sleep_min"] > 720)).sum())
    print(f"People averaging under 3 h or over 12 h of sleep a day (should be about 0): {odd}")
    print("Missing values:\n", df.isna().sum().to_string())
    print("\nSummary:\n", df.describe().T.to_string())
    print(f"\nSaved {OUT_PATH} and {FEATURES_PATH}")


if __name__ == "__main__":
    main()
