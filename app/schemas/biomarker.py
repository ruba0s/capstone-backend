from pydantic import BaseModel

class BiomarkerInput(BaseModel):
    age: int
    systolic_bp: float
    diastolic_bp: float
    cholesterol: float
    bmi: float
    glucose: float

class PredictionResponse(BaseModel):
    chronological_age: int
    predicted_biological_age: float
    gap: float