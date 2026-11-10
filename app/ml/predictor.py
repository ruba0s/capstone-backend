"""
Runs the full prediction for one person: every model they have data for, the ensemble,
the age-bias correction, and each factor's impact. Kept separate from the API route so it
can be tested and reused without FastAPI.
"""
import json

import pandas as pd

from app.ml.ensemble import correct_age_bias, effective_weights
from app.ml.model import BioAgeModel
from app.ml.registry import ARTIFACTS, MODALITIES

# Display names for every model feature (shown in the app and sent to the LLM)
FRIENDLY_NAMES = {
    # biomarker
    "RIAGENDR": "sex",
    "PAQ605": "vigorous work activity",
    "BMXBMI": "BMI",
    "LBXGLU": "fasting glucose",
    "DIQ010": "diabetes status",
    "LBXGLT": "2-hour glucose",
    "LBXIN": "insulin",
    # activity
    "avg_daily_activity": "average daily movement",
    "activity_variability": "day-to-day activity variation",
    "avg_active_minutes": "active minutes per day",
    # sleep
    "mean_sleep_min": "average sleep",
    "min_sleep_min": "shortest night",
    "max_sleep_min": "longest night",
    "std_sleep_min": "night-to-night sleep variation",
}
FIXED_TRAITS = {"RIAGENDR"}  # can't be changed, so left out of the factor list

_models = {name: BioAgeModel(name) for name in MODALITIES}
_ensemble = None


def ensemble_config() -> dict:
    """Weights and bias correction from app/ml/artifacts/ensemble.json (written by evaluate.py)."""
    global _ensemble
    if _ensemble is None:
        path = ARTIFACTS / "ensemble.json"
        if not path.exists():
            raise FileNotFoundError("ensemble.json not found. Run python -m app.ml.train and python -m app.ml.evaluate.")
        _ensemble = json.loads(path.read_text())
    return _ensemble


def predict_biological_age(age: float, inputs: dict) -> dict:
    """
    age:    chronological age in years
    inputs: {"biomarker": {"BMXBMI": 27.1, ...}, "activity": {...}}  (model column names;
            leave a modality out if the person has no data for it)
    """
    config = ensemble_config()

    raw_preds, contribs = {}, {}
    for name, values in inputs.items():
        features = pd.DataFrame([values])
        raw_preds[name] = _models[name].predict(features)
        contribs[name] = _models[name].contributions(features)

    weights = effective_weights(raw_preds, config["weights"])
    combined = sum(weights[n] * raw_preds[n] for n in weights)
    biological_age = correct_age_bias(combined, age, config["bias"])
    gap = biological_age - age

    # Each factor's impact = its contribution inside its model, scaled by that model's weight.
    # So a model with weight 0 (sleep, when activity is present) shows its factors with 0 impact:
    # they don't move the number, but the values are still there for recommendations.
    factors = []
    for name, model_contribs in contribs.items():
        for col, impact in model_contribs.items():
            factors.append({
                "name": FRIENDLY_NAMES.get(col, col),
                "column": col,
                "source": name,
                "value": float(inputs[name][col]),
                "impact_years": weights[name] * impact,
            })
    shown = [f for f in factors if f["column"] not in FIXED_TRAITS]

    return {
        "chronological_age": age,
        "predicted_biological_age": biological_age,
        "gap": gap,
        "factors": sorted(shown, key=lambda f: -abs(f["impact_years"])),
        # gap = sum of shown impacts + this. It covers the age-bias correction, the average
        # person's baseline, and fixed traits like sex.
        "other_years": gap - sum(f["impact_years"] for f in shown),
        "model_predictions": raw_preds,
        "weights_used": weights,
    }