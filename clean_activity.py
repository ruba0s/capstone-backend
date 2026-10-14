import pandas as pd
import numpy as np

# 1. Load the dataset
df = pd.read_sas("data/raw/PAXDAY_H.xpt", format="xport")

print("Original rows:", len(df))

# 2. Keep columns relevant to activity and data quality
columns = [
    "SEQN",
    "PAXDAYD",
    "PAXVMD",
    "PAXMTSD",
    "PAXWWMD",
    "PAXSWMD",
    "PAXNWMD",
    "PAXUMD",
    "PAXQFD"
]

df = df[columns].copy()

# 3. Convert day numbers from bytes to integers
df["PAXDAYD"] = pd.to_numeric(
    df["PAXDAYD"].str.decode("utf-8"),
    errors="coerce"
)

# 4. Convert SAS missing-value placeholders to NaN
numeric_columns = df.select_dtypes(include="number").columns

for col in numeric_columns:
    missing_code = (df[col] > 0) & (df[col] < 1e-50)
    df.loc[missing_code, col] = np.nan

# 5. Remove partial first and last recording days
df = df[df["PAXDAYD"].between(2, 8)]

# 6. Remove records missing essential measurements
df = df.dropna(
    subset=["SEQN", "PAXDAYD", "PAXVMD", "PAXMTSD", "PAXWWMD"]
)

# 7. Keep days with at least 600 valid awake wear minutes
df = df[df["PAXWWMD"] >= 600]

# 8. Remove impossible measurement values
df = df[
    (df["PAXMTSD"] >= 0) &
    (df["PAXVMD"] <= 1440) &
    (df["PAXVMD"] >= df["PAXWWMD"])
]

# 9. Remove duplicate participant-day records
df = df.drop_duplicates(subset=["SEQN", "PAXDAYD"])

# Keep essential columns for the cleaned activity dataset
df = df[
    [
        "SEQN",
        "PAXDAYD",
        "PAXVMD",
        "PAXMTSD",
        "PAXWWMD",
    ]
].copy()

# 10. Save the cleaned dataset
df.to_csv("data/processed/activity_cleaned.csv", index=False)

print("Cleaned rows:", len(df))
print("Unique participants:", df["SEQN"].nunique())
print("Cleaned dataset saved!")

# Verify cleaned dataset
print("\nRemaining missing values:")
print(df.isnull().sum())

print("\nDuplicate participant-day records:")
print(df.duplicated(subset=["SEQN", "PAXDAYD"]).sum())

print("\nFirst 5 cleaned rows:")
print(df.head())