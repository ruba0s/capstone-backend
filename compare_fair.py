"""
Compare every trained model on exactly the same people.

train.py reports each model's test MAE on that model's own test people, and the groups differ
(different sizes, different ages), so the numbers are not directly comparable. This keeps only
people who are in the test split of EVERY trained model and reports, on that identical group:
each model's MAE, a guess-the-average baseline, and the combined (ensemble) result.

Run after training and evaluating (needs app/ml/artifacts/*_preds.csv and ensemble.json):
    python compare_fair.py
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from app.ml.ensemble import weighted_average
from app.ml.registry import ARTIFACTS, ID_COL, MODALITIES, preds_path

MIN_PEOPLE = 100  # below this the comparison is too noisy to trust


def main():
    preds = {n: pd.read_csv(preds_path(n)) for n in MODALITIES if preds_path(n).exists()}
    if len(preds) < 2:
        raise SystemExit("Need at least two trained models. Run python -m app.ml.train first.")

    tests = {n: df[df["split"] == "test"].set_index(ID_COL) for n, df in preds.items()}
    common = sorted(set.intersection(*[set(t.index) for t in tests.values()]))
    print(f"Models: {', '.join(preds)}")
    print(f"People in the test split of every model: {len(common)}")
    if len(common) < MIN_PEOPLE:
        print(f"WARNING: fewer than {MIN_PEOPLE} people, treat these numbers as rough.")

    age = tests[next(iter(tests))].loc[common, "age"]
    table = pd.DataFrame({n: t.loc[common, "pred"] for n, t in tests.items()})

    # One baseline for everyone: the average age of all training people, guessed for every test person
    train_ages = pd.concat([df[df["split"] == "train"][[ID_COL, "age"]] for df in preds.values()])
    baseline_pred = train_ages.drop_duplicates(ID_COL)["age"].mean()
    base_mae = mean_absolute_error(age, np.full(len(age), baseline_pred))

    rows = [("guess the average age", base_mae)]
    rows += [(n, mean_absolute_error(age, table[n])) for n in table]

    ens_path = ARTIFACTS / "ensemble.json"
    if ens_path.exists():
        ens = json.loads(ens_path.read_text())
        weights = {n: w for n, w in ens["weights"].items() if n in table}
        combined = pd.Series(
            [weighted_average(r.to_dict(), weights) for _, r in table.iterrows()], index=table.index)
        rows.append(("combined", mean_absolute_error(age, combined)))
        # Deliberately NOT reporting MAE after the age-bias correction: the correction uses each
        # person's real age, so it makes MAE look much better than the model really is (even a
        # guess-the-average model would score 0). Use it only for the gap, and check the slope.
    else:
        print("(no ensemble.json found, run python -m app.ml.evaluate to add the combined rows)")

    print(f"\n{'model':<30}{'MAE (years)':>12}{'vs guessing':>14}")
    for name, mae in rows:
        gain = "" if name.startswith("guess") else f"{(1 - mae / base_mae) * 100:>12.1f}%"
        print(f"{name:<30}{mae:>12.2f}{gain:>14}")
    print(f"\nAges of these people: {age.min():.0f} to {age.max():.0f}")


if __name__ == "__main__":
    main()
