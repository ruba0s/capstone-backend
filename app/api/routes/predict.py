from fastapi import APIRouter
from app.schemas.biomarker import BiomarkerInput, PredictionResponse
from app.ml.model import bioage_model
import pandas as pd

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
def predict(input_data: BiomarkerInput):
    features = pd.DataFrame([{
        "systolic_bp": input_data.systolic_bp,
        "diastolic_bp": input_data.diastolic_bp,
        "cholesterol": input_data.cholesterol,
        "bmi": input_data.bmi,
        "glucose": input_data.glucose,
    }])

    predicted_age = bioage_model.predict(features)
    gap = predicted_age - input_data.age

    return PredictionResponse(
        chronological_age=input_data.age,
        predicted_biological_age=round(predicted_age, 1),
        gap=round(gap, 1),
    )