"""Request and response shapes for POST /predict. Every data section is optional, but at least one is required."""
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from app.llm.schemas import FactorContribution


class BiomarkerInput(BaseModel):
    sex: Literal[1, 2]                                  # RIAGENDR: 1 = male, 2 = female
    vigorous_work: Literal[1, 2]                        # PAQ605: vigorous work activity, 1 = yes, 2 = no
    bmi: float = Field(ge=10, le=80)                    # BMXBMI, kg/m^2
    fasting_glucose: float = Field(ge=30, le=600)       # LBXGLU, mg/dL
    diabetes: Literal[1, 2, 3]                          # DIQ010: 1 = yes, 2 = no, 3 = borderline
    glucose_2hr: float = Field(ge=30, le=700)           # LBXGLT, mg/dL
    insulin: float = Field(ge=0, le=300)                # LBXIN, uU/mL

    def to_model_columns(self) -> dict:
        return {"RIAGENDR": self.sex, "PAQ605": self.vigorous_work, "BMXBMI": self.bmi,
                "LBXGLU": self.fasting_glucose, "DIQ010": self.diabetes,
                "LBXGLT": self.glucose_2hr, "LBXIN": self.insulin}


class ActivityInput(BaseModel):
    avg_daily_activity: float = Field(ge=0)     # MIMS units per valid minute, averaged over days
    activity_variability: float = Field(ge=0)   # std of daily activity across days
    avg_active_minutes: float = Field(ge=0, le=1440)  # minutes per day

    def to_model_columns(self) -> dict:
        return self.model_dump()


class SleepInput(BaseModel):
    mean_sleep_min: float = Field(ge=0, le=1440)  # average minutes of sleep per day, over 4+ days
    min_sleep_min: float = Field(ge=0, le=1440)
    max_sleep_min: float = Field(ge=0, le=1440)
    std_sleep_min: float = Field(ge=0, le=1440)

    def to_model_columns(self) -> dict:
        return self.model_dump()


class PredictRequest(BaseModel):
    age: int = Field(ge=18, le=80)   # the models were trained on adults 18-80
    biomarker: Optional[BiomarkerInput] = None
    activity: Optional[ActivityInput] = None
    sleep: Optional[SleepInput] = None

    @model_validator(mode="after")
    def needs_some_data(self):
        if not (self.biomarker or self.activity or self.sleep):
            raise ValueError("Send at least one of: biomarker, activity, sleep")
        return self

    def model_inputs(self) -> dict:
        """{"biomarker": {column: value}, ...} for the sections that were sent."""
        sections = {"biomarker": self.biomarker, "activity": self.activity, "sleep": self.sleep}
        return {name: s.to_model_columns() for name, s in sections.items() if s is not None}


class PredictionResponse(BaseModel):
    chronological_age: int
    predicted_biological_age: float
    gap: float                          # biological minus chronological; + means older
    factors: List[FactorContribution]   # biggest impact first; sex is left out
    other_years: float                  # gap = sum(factor impacts) + other_years
    model_predictions: Dict[str, float] # each model's raw prediction, before the bias correction
    weights_used: Dict[str, float]      # each model's share of the combined prediction