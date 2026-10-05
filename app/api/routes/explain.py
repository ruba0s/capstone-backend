from fastapi import APIRouter
from app.llm.schemas import ExplainRequest, ExplainResponse
from app.llm.prompts import SYSTEM_PROMPT, build_explain_prompt
from app.llm.client import ask_llm

router = APIRouter()

@router.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    text = ask_llm(SYSTEM_PROMPT, build_explain_prompt(req))
    if not text:
        gap = round(req.predicted_biological_age - req.chronological_age, 1)
        text = f"(LLM key not set) Your biological age is {gap:+} years from your real age."
    return ExplainResponse(explanation=text)