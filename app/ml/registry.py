"""
Registry of all the models, each with where its data lives and which columns it uses
"""
from app.config import BASE_DIR
from app.ml.features import FEATURE_COLS
from app.ml.activity_features import FEATURE_COLUMNS as ACTIVITY_FEATURES

PROCESSED = BASE_DIR / "data" / "processed"
ARTIFACTS = BASE_DIR / "app" / "ml" / "artifacts"
REPORTS = BASE_DIR / "reports"

ID_COL = "SEQN"        # NHANES participant ID, shared across 2013-2014 files
TARGET_COL = "RIDAGEYR"  # chronological age

MODALITIES = {
    "biomarker": {"data": PROCESSED / "biomarker_data.csv", "features": FEATURE_COLS},
    # activity_final.csv calls the age column "age" instead of RIDAGEYR
    "activity": {"data": PROCESSED / "activity_final.csv", "features": ACTIVITY_FEATURES, "target": "age"},
    # None = use every column except SEQN and the age column
    "sleep": {"data": PROCESSED / "sleep_data.csv", "features": None},
}

MIN_AGE = 18  # activity and sleep are adults only, so biomarker is filtered to match

NON_FEATURES = {ID_COL, TARGET_COL, "age", "age_group"}  # age_group is derived from age (leak)


def model_path(name):
    return ARTIFACTS / f"{name}_model.pkl"


def preds_path(name):
    return ARTIFACTS / f"{name}_preds.csv"