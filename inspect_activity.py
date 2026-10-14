import pandas as pd

# Read the activity dataset
df = pd.read_sas("data/raw/PAXDAY_H.xpt", format="xport")

# Check the dataset
print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())