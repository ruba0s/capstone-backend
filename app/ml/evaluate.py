"""Compare the trained models, ensemble them, and check age bias.

Run after training:  python -m app.ml.evaluate
Writes app/ml/artifacts/ensemble.json and plots in reports/figures/.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from app.ml.ensemble import correct_age_bias, weighted_average
from app.ml.registry import ARTIFACTS, ID_COL, MODALITIES, REPORTS, preds_path

MIN_OVERLAP = 200  # people needed in every dataset before weights are learned from data
FIG_DIR = REPORTS / "figures"


def load_predictions():
    preds = {n: pd.read_csv(preds_path(n)) for n in MODALITIES if preds_path(n).exists()}
    if not preds:
        raise SystemExit("No trained models found. Run python -m app.ml.train first.")
    return preds


def wide_table(preds):
    """One row per person, one prediction column per model (NaN if they're not in that dataset)."""
    long = pd.concat([df.assign(model=n) for n, df in preds.items()])
    wide = long.pivot_table(index=ID_COL, columns="model", values="pred")
    info = long.groupby(ID_COL)[["age", "split"]].first()
    return info.join(wide).reset_index()


def fit_weights(wide, names, cv_mae):
    if len(names) == 1:
        return {names[0]: 1.0}, "single model"
    overlap = wide[wide.split == "train"].dropna(subset=names)
    if len(overlap) >= MIN_OVERLAP:
        reg = LinearRegression(positive=True, fit_intercept=False).fit(overlap[names], overlap["age"])
        w = reg.coef_ / reg.coef_.sum()
        return dict(zip(names, map(float, w))), f"learned from {len(overlap)} people in every dataset"
    inv = {n: 1 / cv_mae[n] for n in names}
    total = sum(inv.values())
    return {n: v / total for n, v in inv.items()}, "inverse CV MAE (too few people in every dataset)"


def plot(df, label, pred_col):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    lims = [df["age"].min(), df["age"].max()]
    a1.scatter(df["age"], df[pred_col], s=8, alpha=0.4)
    a1.plot(lims, lims, "k--", lw=1, label="perfect")
    a1.set(xlabel="Chronological age", ylabel="Predicted age", title=f"{label}: predicted vs real")
    a1.legend()
    gap = df[pred_col] - df["age"]
    slope, intercept = np.polyfit(df["age"], gap, 1)
    a2.scatter(df["age"], gap, s=8, alpha=0.4)
    a2.axhline(0, color="k", lw=1)
    a2.plot(lims, intercept + slope * np.array(lims), "r-", label=f"trend: slope {slope:.2f}")
    a2.set(xlabel="Chronological age", ylabel="Gap (predicted - real)", title=f"{label}: age bias")
    a2.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{label.replace(' ', '_')}.png", dpi=120)
    plt.close(fig)


def main():
    preds = load_predictions()
    names = list(preds)
    wide = wide_table(preds)

    print("=== Per-model results (test set) ===")
    cv_mae, test_mae = {}, {}
    for n, df in preds.items():
        tr, te = df[df.split == "train"], df[df.split == "test"]
        cv_mae[n] = mean_absolute_error(tr["age"], tr["pred"])
        test_mae[n] = mean_absolute_error(te["age"], te["pred"])
        print(f"{n:<10} test MAE {test_mae[n]:6.2f}   n={len(df):5d}   ages {df['age'].min():.0f}-{df['age'].max():.0f}")
        plot(te, n, "pred")

    print("\n=== People in more than one dataset (SEQN overlap) ===")
    ids = {n: set(df[ID_COL]) for n, df in preds.items()}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            print(f"{a} & {b}: {len(ids[a] & ids[b])}")
    if len(names) > 2:
        print(f"all {len(names)}: {len(set.intersection(*ids.values()))}")

    weights, method = fit_weights(wide, names, cv_mae)
    print(f"\n=== Ensemble weights ({method}) ===")
    for n, w in weights.items():
        print(f"{n:<10} {w:.3f}")

    wide["ensemble"] = wide.apply(
        lambda r: weighted_average({n: r[n] for n in names if pd.notna(r[n])}, weights), axis=1)

    # Bias correction: fit on training people (out-of-fold predictions), check on test people
    train_rows, test_rows = wide[wide.split == "train"], wide[wide.split == "test"].copy()
    slope, intercept = np.polyfit(train_rows["age"], train_rows["ensemble"] - train_rows["age"], 1)
    bias = {"intercept": float(intercept), "slope": float(slope)}
    test_rows["corrected"] = [correct_age_bias(p, a, bias) for p, a in zip(test_rows["ensemble"], test_rows["age"])]

    print("\n=== Ensemble (test set) ===")
    ens_mae = mean_absolute_error(test_rows["age"], test_rows["ensemble"])
    print(f"MAE, everyone:                {ens_mae:.2f}  (n={len(test_rows)})")
    full = test_rows.dropna(subset=names)
    if len(names) > 1 and len(full):
        print(f"MAE, people in every dataset: {mean_absolute_error(full['age'], full['ensemble']):.2f}  (n={len(full)})")
    before = np.polyfit(test_rows["age"], test_rows["ensemble"] - test_rows["age"], 1)[0]
    after = np.polyfit(test_rows["age"], test_rows["corrected"] - test_rows["age"], 1)[0]
    print(f"Age-bias slope: {before:.2f} before correction -> {after:.2f} after (0 = no bias)")
    plot(test_rows, "ensemble", "ensemble")
    plot(test_rows, "ensemble corrected", "corrected")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS / "ensemble.json", "w") as f:
        json.dump({"weights": weights, "weight_method": method, "bias": bias,
                   "test_mae": {**test_mae, "ensemble": float(ens_mae)}}, f, indent=2)
    print(f"\nSaved {ARTIFACTS.name}/ensemble.json and plots in reports/figures/")


if __name__ == "__main__":
    main()