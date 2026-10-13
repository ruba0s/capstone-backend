import pandas as pd

FEATURE_COLS = ["RIAGENDR", "PAQ605", "BMXBMI", "LBXGLU", "DIQ010", "LBXGLT", "LBXIN"]

def clean_biomarker_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset="SEQN")
    df = df.dropna(subset=["RIDAGEYR"])
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURE_COLS]