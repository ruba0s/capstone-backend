"""
Train the sleep model using the team's shared split and registry.

Run from the repo root:
    python -m app.ml.train_sleep
"""
import pickle

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from app.ml.registry import (
    ARTIFACTS, ID_COL, MODALITIES, NON_FEATURES, TARGET_COL,
    model_path, preds_path,
)
from app.ml.split import split_by_person


def main():
    cfg = MODALITIES["sleep"]
    df = pd.read_csv(cfg["data"])

    # Features = every column except the ID, the age label, and age_group
    features = cfg["features"] or [c for c in df.columns if c not in NON_FEATURES]
    print("Features used:", features)

    # Shared split: the same people are in the test set for every model
    train, test = split_by_person(df)
    print(f"Train people: {len(train)} | Test people: {len(test)}")

    X_train, y_train = train[features], train[TARGET_COL]
    X_test, y_test = test[features], test[TARGET_COL]

    models = {
        "baseline (guess mean age)": DummyRegressor(strategy="mean"),
        "linear regression": LinearRegression(),
        "gradient boosting": GradientBoostingRegressor(
            max_depth=2, n_estimators=100, learning_rate=0.05, random_state=42
        ),
    }

    maes = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        maes[name] = mean_absolute_error(y_test, model.predict(X_test))
        print(f"{name}: MAE = {maes[name]:.2f} years")

    # Save whichever real model (not the baseline) did better
    best = min(["linear regression", "gradient boosting"], key=maes.get)
    best_model = models[best]
    print(f"\nBest model: {best}")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    with open(model_path("sleep"), "wb") as f:
        pickle.dump(best_model, f)

    pd.DataFrame({
        ID_COL: test[ID_COL].values,
        "actual": y_test.values,
        "predicted": best_model.predict(X_test),
    }).to_csv(preds_path("sleep"), index=False)

    lines = [f"{name}: {mae:.2f}" for name, mae in maes.items()]
    lines += [f"best: {best}", f"features: {features}"]
    (ARTIFACTS / "sleep_metrics.txt").write_text("\n".join(lines) + "\n")

    print(f"\nSaved model to {model_path('sleep')}")


if __name__ == "__main__":
    main()