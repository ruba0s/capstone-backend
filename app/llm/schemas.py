from pydantic import BaseModel
from typing import List, Optional

class FactorContribution(BaseModel):
    name: str            # display name, e.g. "fasting glucose"
    value: float         # the person's value
    impact_years: float  # + means pushes biological age up, - means pushes it down
    source: Optional[str] = None  # which model it came from: "biomarker", "activity" or "sleep"

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