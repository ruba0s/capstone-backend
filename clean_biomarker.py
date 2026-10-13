import pandas as pd

df = pd.read_csv("data/raw/biomarker.csv")

print("Rows before:", len(df))
print("Missing values:\n", df.isna().sum())

# Remove duplicate people and rows with no age
df = df.drop_duplicates(subset="SEQN")
df = df.dropna(subset=["RIDAGEYR"])

# age_group is made from age, so it would leak the answer to the model
df = df.drop(columns=["age_group"])

# Fill any other gaps with the column median
feature_cols = [c for c in df.columns if c not in ["SEQN", "RIDAGEYR"]]
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

print("Rows after:", len(df))
print("Age range:", df["RIDAGEYR"].min(), "to", df["RIDAGEYR"].max())

df.to_csv("data/processed/biomarker_data.csv", index=False)
print("Saved to data/processed/biomarker_data.csv")