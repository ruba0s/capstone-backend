import json
from anthropic import Anthropic
from app.config import settings

_client = Anthropic(api_key=settings.ANTHROPIC_API_KEY) if settings.ANTHROPIC_API_KEY else None

def ask_llm(system: str, user: str, max_tokens: int = 400) -> str:
    if _client is None:
        return ""  # no key set, caller uses fallback text
    response = _client.messages.create(
        model=settings.LLM_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text

def parse_json_list(text: str):
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []