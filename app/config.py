import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    APP_NAME: str = "BioAge API"
    MODEL_PATH: str = "app/ml/artifacts/bioage_model.pkl"

settings = Settings()