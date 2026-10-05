import pickle
import pandas as pd
from app.config import settings

class BioAgeModel:
    def __init__(self):
        self.model = None

    def _load(self):
        if self.model is None:
            with open(settings.MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)

    def predict(self, features: pd.DataFrame) -> float:
        self._load()
        return float(self.model.predict(features)[0])

bioage_model = BioAgeModel()