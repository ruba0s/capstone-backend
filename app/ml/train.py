"""Train any model in the registry with the shared split.

Usage (from the repo root):
    python -m app.ml.train                      # all three
    python -m app.ml.train biomarker activity   # just these
"""
import pickle
import sys

import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.ml.registry import (
    ARTIFACTS, ID_COL, MIN_AGE, MODALITIES, NON_FEATURES, TARGET_COL, model_path, preds_path,
)
from app.ml.split import split_by_person

CANDIDATES = {
    "linear": make_pipeline(StandardScaler(), LinearRegression()),
    "gbr": GradientBoostingRegressor(random_state=42),
}


def load_dataset(name):
    """Reads a cleaned CSV and returns it with standard column names (SEQN, RIDAGEYR)."""
    cfg = MODALITIES[name]
    df = pd.read_csv(cfg["data"])
    df = df.rename(columns={cfg.get("target", TARGET_COL): TARGET_COL})
    missing = {ID_COL, TARGET_COL} - set(df.columns)
    if missing:
        raise ValueError(f"{cfg['data'].name} is missing required columns: {missing}")

    df[ID_COL] = df[ID_COL].astype(int)  # 73557.0 and 73557 must match across datasets
    df = df.drop_duplicates(subset=ID_COL).dropna(subset=[TARGET_COL])
    df = df[df[TARGET_COL] >= MIN_AGE]
    features = cfg["features"] or [c for c in df.columns if c not in NON_FEATURES]
    return df.dropna(subset=features), features


def train(name):
    df, features = load_dataset(name)
    train_df, test_df = split_by_person(df)
    X_train, y_train = train_df[features], train_df[TARGET_COL]
    X_test, y_test = test_df[features], test_df[TARGET_COL]
    print(f"[{name}] {len(train_df)} train / {len(test_df)} test, features: {features}")

    # Out-of-fold predictions: each training person is predicted by a model that never saw them.
    # Used to pick the model type here, and by evaluate.py to learn the ensemble weights.
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    oof, cv_mae = {}, {}
    for kind, est in CANDIDATES.items():
        oof[kind] = cross_val_predict(clone(est), X_train, y_train, cv=cv)
        cv_mae[kind] = mean_absolute_error(y_train, oof[kind])

    best = min(cv_mae, key=lambda k: cv_mae[k])  # chosen on CV, not the test set
    model = clone(CANDIDATES[best]).fit(X_train, y_train)
    test_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_pred)
    dummy_mae = mean_absolute_error(y_test, DummyRegressor().fit(X_train, y_train).predict(X_test))

    for kind in CANDIDATES:
        print(f"  {kind:<7} CV MAE: {cv_mae[kind]:.2f}")
    print(f"  chosen: {best}  test MAE: {test_mae:.2f}  (guess-the-average: {dummy_mae:.2f})")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(model_path(name), "wb") as f:
        pickle.dump({"model": model, "features": features, "kind": best,
                     "test_mae": test_mae, "dummy_mae": dummy_mae}, f)

    pd.concat([
        pd.DataFrame({ID_COL: train_df[ID_COL], "age": y_train, "pred": oof[best], "split": "train"}),
        pd.DataFrame({ID_COL: test_df[ID_COL], "age": y_test, "pred": test_pred, "split": "test"}),
    ]).to_csv(preds_path(name), index=False)
    print(f"  saved {model_path(name).name} and {preds_path(name).name}\n")


if __name__ == "__main__":
    for n in sys.argv[1:] or list(MODALITIES):
        train(n)