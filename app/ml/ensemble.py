"""
Ensembles the per-modality models into one biological age, then corrects age bias (e.g. if a user only has some 
of the data (for instance missing sleep data), the weights of the remaining models are rescaled to sum to 1).
"""

def effective_weights(preds: dict, weights: dict) -> dict:
    """The weight each available model actually gets for this person (they sum to 1).
    Missing models are skipped and the rest are rescaled; if every available model has
    weight 0 (e.g. sleep only), they share the weight equally."""
    available = [k for k, v in preds.items() if v is not None and k in weights]
    total = sum(weights[k] for k in available)
    if total == 0:
        return {k: 1 / len(available) for k in available}
    return {k: weights[k] / total for k in available}


def weighted_average(preds: dict, weights: dict) -> float:
    """Weighted average of whichever models are available for this person.
    preds: {"biomarker": 47.2, "activity": 51.0}"""
    w = effective_weights(preds, weights)
    return sum(w[k] * preds[k] for k in w)


def correct_age_bias(pred: float, age: float, bias: dict) -> float:
    """Removes the 'young people predicted older, old people predicted younger' trend.
    bias = {"intercept": a, "slope": b} from fitting gap = a + b * age."""
    return pred - (bias["intercept"] + bias["slope"] * age)