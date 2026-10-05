from fastapi import APIRouter
from app.llm.schemas import ExplainRequest, RecommendationsResponse, Recommendation
from app.llm.prompts import SYSTEM_PROMPT, build_recommendations_prompt
from app.llm.client import ask_llm, parse_json_list

router = APIRouter()

FALLBACK = [
    Recommendation(title="Drink water", detail="Staying hydrated supports healthy blood pressure.", frequency="3x daily"),
    Recommendation(title="10 minute walk", detail="Regular movement is linked to slower aging.", frequency="daily"),
    Recommendation(title="Wind down by 10pm", detail="Consistent sleep supports recovery.", frequency="daily"),
]

@router.post("/recommendations", response_model=RecommendationsResponse)
def recommendations(req: ExplainRequest):
    text = ask_llm(SYSTEM_PROMPT, build_recommendations_prompt(req))
    items = parse_json_list(text)
    recs = [Recommendation(**i) for i in items if {"title", "detail", "frequency"} <= i.keys()]
    return RecommendationsResponse(recommendations=recs or FALLBACK)