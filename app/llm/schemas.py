from pydantic import BaseModel
from typing import List

class FactorContribution(BaseModel):
    name: str            # e.g. "systolic_bp"
    value: float         # the person's value
    impact_years: float  # + means pushes biological age up, - means pushes it down

class ExplainRequest(BaseModel):
    chronological_age: int
    predicted_biological_age: float
    factors: List[FactorContribution]

class ExplainResponse(BaseModel):
    explanation: str

class Recommendation(BaseModel):
    title: str     # short, widget friendly, e.g. "Drink water"
    detail: str    # one sentence why
    frequency: str # e.g. "3x daily", "daily", "weekly"

class RecommendationsResponse(BaseModel):
    recommendations: List[Recommendation]