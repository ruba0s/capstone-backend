import pandas as pd

# Read questionnaire data
df = pd.read_sas("data/raw/PAQ_H.xpt", format="xport")

# Each activity: yes/no question, days/week, minutes/active day
ACTIVITIES = {
    "vigorous_work": ("PAQ605", "PAQ610", "PAD615"),
    "moderate_work": ("PAQ620", "PAQ625", "PAD630"),
    "transport": ("PAQ635", "PAQ640", "PAD645"),
    "vigorous_recreation": ("PAQ650", "PAQ655", "PAD660"),
    "moderate_recreation": ("PAQ665", "PAQ670", "PAD675"),
}

# Calculate average daily minutes for each activity
for name, (question, days, minutes) in ACTIVITIES.items():

    result = pd.Series(float("nan"), index=df.index)

    # No activity: 0 minutes
    no_activity = df[question] == 2
    result.loc[no_activity] = 0

    # Yes: valid days and duration required
    valid = (
        (df[question] == 1)
        & df[days].between(1, 7)
        & df[minutes].between(10, 900)
    )

    result.loc[valid] = (
        df.loc[valid, days] * df.loc[valid, minutes] / 7
    )

    df[name + "_minutes"] = result

# List the five calculated activity columns
minute_columns = [
    name + "_minutes" for name in ACTIVITIES
]

# Require complete responses for all five categories
df = df.dropna(subset=minute_columns).copy()

# Combine weekly activity into an average per calendar day
df["avg_active_minutes"] = df[minute_columns].sum(axis=1)

# Keep only the participant ID and final feature
result = df[["SEQN", "avg_active_minutes"]].copy()

# Save
result.to_csv(
    "data/processed/activity_questionnaire.csv",
    index=False
)

print("Questionnaire participants:", len(result))
print("\nMissing values:")
print(result.isnull().sum())

print("\nFirst 5 rows:")
print(result.head())

print("\nSaved activity_questionnaire.csv!")