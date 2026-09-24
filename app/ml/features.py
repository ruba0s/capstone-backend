import pandas as pd

def clean_biomarker_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning for the NHANES biomarker dataset."""
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select/engineer the columns the model will train on.
    Update this once you know your final NHANES column names.
    """
    feature_cols = ["systolic_bp", "diastolic_bp", "cholesterol", "bmi", "glucose"]
    return df[feature_cols]