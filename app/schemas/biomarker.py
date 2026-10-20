from typing import List
from pydantic import BaseModel
from app.llm.schemas import FactorContribution

class BiomarkerInput(BaseModel):
    age: int
    sex: int          # RIAGENDR: 1 = male, 2 = female
    activity: int     # PAQ605: 1 = yes, 2 = no
    bmi: float        # BMXBMI
    fasting_glucose: float   # LBXGLU
    diabetes: int     # DIQ010: 1 = yes, 2 = no, 3 = borderline
    glucose_2hr: float       # LBXGLT
    insulin: float    # LBXIN

class PredictionResponse(BaseModel):
    chronological_age: int
    predicted_biological_age: float
    gap: float
    factors: List[FactorContribution]