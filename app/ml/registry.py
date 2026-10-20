"""
Registry of all the models, each with where its data lives and which columns it uses
"""
from app.config import BASE_DIR
from app.ml.features import FEATURE_COLS

PROCESSED = BASE_DIR / "data" / "processed"
ARTIFACTS = BASE_DIR / "app" / "ml" / "artifacts"
REPORTS = BASE_DIR / "reports"

ID_COL = "SEQN"        # NHANES participant ID, shared across 2013-2014 files
TARGET_COL = "RIDAGEYR"  # chronological age

MODALITIES = {
    "biomarker": {"data": PROCESSED / "biomarker_data.csv", "features": FEATURE_COLS},
    # None = use every column except SEQN and RIDAGEYR
    "activity": {"data": PROCESSED / "activity_data.csv", "features": None},
    "sleep": {"data": PROCESSED / "sleep_data.csv", "features": None},
}

NON_FEATURES = {ID_COL, TARGET_COL, "age_group"}  # age_group is derived from age (leak)


def model_path(name):
    return ARTIFACTS / f"{name}_model.pkl"


def preds_path(name):
    return ARTIFACTS / f"{name}_preds.csv"