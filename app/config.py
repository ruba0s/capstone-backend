import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent  # the capstone-backend folder

load_dotenv(BASE_DIR / ".env")

class Settings:
    APP_NAME: str = "BioAge API"
    MODEL_PATH: str = str(BASE_DIR / "app" / "ml" / "artifacts" / "bioage_model.pkl")
    DATA_PATH: str = str(BASE_DIR / "data" / "processed" / "biomarker_data.csv")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "claude-sonnet-5-5")

settings = Settings()