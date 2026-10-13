from fastapi import APIRouter
import pandas as pd
from app.schemas.biomarker import BiomarkerInput, PredictionResponse
from app.ml.features import FEATURE_COLS
from app.ml.model import bioage_model

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
    return PredictionResponse(
        chronological_age=data.age,
        predicted_biological_age=round(predicted, 1),
        gap=round(predicted - data.age, 1),
    )