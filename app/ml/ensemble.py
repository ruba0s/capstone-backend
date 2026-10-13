"""
Ensembles the per-modality models into one biological age, then corrects age bias (e.g. if a user only has some 
of the data (for instance missing sleep data), the weights of the remaining models are rescaled to sum to 1).
"""

def weighted_average(preds: dict, weights: dict) -> float:
    """Weighted average of whichever models are available for this person.
    preds: {"biomarker": 47.2, "activity": 51.0}; missing models are skipped and
    the remaining weights are rescaled to sum to 1."""
    available = {k: v for k, v in preds.items() if v is not None and k in weights}
    total = sum(weights[k] for k in available)
    return sum(weights[k] * v for k, v in available.items()) / total


def correct_age_bias(pred: float, age: float, bias: dict) -> float:
    """Removes the 'young people predicted older, old people predicted younger' trend.
    bias = {"intercept": a, "slope": b} from fitting gap = a + b * age."""
    return pred - (bias["intercept"] + bias["slope"] * age)