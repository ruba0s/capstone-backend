from fastapi import APIRouter, HTTPException

from app.llm.schemas import FactorContribution
from app.ml.predictor import predict_biological_age
from app.schemas.prediction import PredictionResponse, PredictRequest

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict(req: PredictRequest):
    try:
        result = predict_biological_age(req.age, req.model_inputs())
    except FileNotFoundError as e:  # models or ensemble.json haven't been trained yet
        raise HTTPException(status_code=503, detail=str(e))

    return PredictionResponse(
        chronological_age=req.age,
        predicted_biological_age=round(result["predicted_biological_age"], 1),
        gap=round(result["gap"], 1),
        factors=[
            FactorContribution(name=f["name"], value=round(f["value"], 2),
                               impact_years=round(f["impact_years"], 2) + 0.0,  # + 0.0 turns -0.0 into 0.0
                               source=f["source"])
            for f in result["factors"]
        ],
        other_years=round(result["other_years"], 2),
        model_predictions={k: round(v, 1) for k, v in result["model_predictions"].items()},
        weights_used={k: round(v, 3) for k, v in result["weights_used"].items()},
    )