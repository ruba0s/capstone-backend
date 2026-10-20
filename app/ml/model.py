import pickle
import pandas as pd
import shap
from app.config import settings

class BioAgeModel:
    def __init__(self):
        self.model = None
        self.explainer = None

    def _load(self):
        if self.model is None:
            with open(settings.MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            self.explainer = shap.TreeExplainer(self.model)

    def predict(self, features: pd.DataFrame) -> float:
        self._load()
        return float(self.model.predict(features)[0])

    def contributions(self, features: pd.DataFrame) -> dict:
        self._load()
        values = self.explainer.shap_values(features)[0]
        return {col: float(v) for col, v in zip(features.columns, values)}

bioage_model = BioAgeModel()