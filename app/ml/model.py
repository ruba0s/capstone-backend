import pickle
import pandas as pd
from app.config import settings

class BioAgeModel:
    def __init__(self):
        with open(settings.MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, features: pd.DataFrame) -> float:
        return float(self.model.predict(features)[0])

bioage_model = BioAgeModel()