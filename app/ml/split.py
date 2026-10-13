"""
Shared train/test split (someone in the test set for one model is in the test set for every model). 
Splits based on each person's SEQN.
"""
import hashlib
import pandas as pd
from app.ml.registry import ID_COL

TEST_PERCENT = 20


def is_test(seqn) -> bool:
    digest = hashlib.md5(str(int(seqn)).encode()).hexdigest()
    return int(digest, 16) % 100 < TEST_PERCENT


def split_by_person(df: pd.DataFrame):
    mask = df[ID_COL].map(is_test)
    return df[~mask], df[mask]