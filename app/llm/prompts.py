SYSTEM_PROMPT = (
    "You are a health insights assistant inside a student capstone app. "
    "You explain biological age results in plain, friendly English at an 8th grade reading level. "
    "You only give general, evidence-based lifestyle suggestions based on patterns in the data. "
    "You never diagnose, never give medical advice, and never mention specific medications. "
    "If a result looks concerning, suggest talking to a doctor. Keep answers short."
)

def build_explain_prompt(req) -> str:
    gap = round(req.predicted_biological_age - req.chronological_age, 1)
    factor_lines = "\n".join(
        f"- {f.name}: value {f.value}, impact {f.impact_years:+.1f} years"
        for f in req.factors
    )
    return (
        f"Chronological age: {req.chronological_age}\n"
        f"Predicted biological age: {req.predicted_biological_age}\n"
        f"Gap (biological minus chronological): {gap} years\n"
        f"Factors (positive impact means older, negative means younger):\n{factor_lines}\n\n"
        "Write 2 to 3 sentences explaining this result and which factors matter most."
    )

def build_recommendations_prompt(req) -> str:
    factor_lines = "\n".join(
        f"- {f.name}: impact {f.impact_years:+.1f} years" for f in req.factors
    )
    return (
        f"Biological age is {req.predicted_biological_age} vs real age {req.chronological_age}.\n"
        f"Factors:\n{factor_lines}\n\n"
        "Give 3 small daily habits that target the factors pushing age up. "
        "Return ONLY a JSON list. Each item has keys: title, detail, frequency. "
        "Titles must be short enough to be a phone widget label."
    )