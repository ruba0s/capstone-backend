import pickle
from typing import Any

import pandas as pd
from app.ml.registry import model_path


class BioAgeModel:
    def __init__(self, name="biomarker"):
        self.name = name
        self.model: Any = None          # set in _load(): a sklearn model or pipeline
        self.features: list[str] = []   # set in _load()
        self.explainer: Any = None

    def _load(self):
        if self.model is None:
            with open(model_path(self.name), "rb") as f:
                bundle = pickle.load(f)
            self.model = bundle["model"]
            self.features = bundle["features"]
            if bundle["kind"] == "gbr":
                import shap
                self.explainer = shap.TreeExplainer(self.model)

    def predict(self, features: pd.DataFrame) -> float:
        self._load()
        return float(self.model.predict(features[self.features])[0])

    def contributions(self, features: pd.DataFrame) -> dict:
        self._load()
        x = features[self.features]
        if self.explainer is not None:
            values = self.explainer.shap_values(x)[0]
        else:
            # Linear model: each feature's impact = coefficient * how far it is from the training average
            scaler, linear = self.model[0], self.model[-1]
            values = linear.coef_ * (x.iloc[0].to_numpy() - scaler.mean_) / scaler.scale_
        return {col: float(v) for col, v in zip(self.features, values)}


bioage_model = BioAgeModel("biomarker")