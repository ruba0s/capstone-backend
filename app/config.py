import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    APP_NAME: str = "BioAge API"
    MODEL_PATH: str = "app/ml/artifacts/bioage_model.pkl"
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "claude-sonnet-5-5")

settings = Settings()