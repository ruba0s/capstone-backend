from fastapi import APIRouter
import pandas as pd
from app.schemas.biomarker import BiomarkerInput, PredictionResponse
from app.ml.features import FEATURE_COLS, FRIENDLY_NAMES
from app.ml.model import bioage_model
from app.llm.schemas import FactorContribution

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
def predict(data: BiomarkerInput):
    row = {
        "RIAGENDR": data.sex,
        "PAQ605": data.activity,
        "BMXBMI": data.bmi,
        "LBXGLU": data.fasting_glucose,
        "DIQ010": data.diabetes,
        "LBXGLT": data.glucose_2hr,
        "LBXIN": data.insulin,
    }
    features = pd.DataFrame([row])[FEATURE_COLS]  # keeps the column order the model was trained on

    predicted = bioage_model.predict(features)
    contribs = bioage_model.contributions(features)

    factors = [
        FactorContribution(
            name=FRIENDLY_NAMES[col],
            value=float(features[col].iloc[0]),
            impact_years=round(v, 1),
        )
        for col, v in contribs.items()
        if col != "RIAGENDR"  # sex isn't something a person can change
    ]

    return PredictionResponse(
        chronological_age=data.age,
        predicted_biological_age=round(predicted, 1),
        gap=round(predicted - data.age, 1),
        factors=factors,
    )