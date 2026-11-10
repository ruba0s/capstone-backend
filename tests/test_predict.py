"""
Tests for POST /predict. Needs trained models first:
    python -m app.ml.train && python -m app.ml.evaluate
Run from the repo root:  python -m pytest tests -v
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.ml.registry import ARTIFACTS

pytestmark = pytest.mark.skipif(not (ARTIFACTS / "ensemble.json").exists(),
                                reason="train and evaluate the models first")
client = TestClient(app)

BIOMARKER = {"sex": 2, "vigorous_work": 2, "bmi": 27.5, "fasting_glucose": 98,
             "diabetes": 2, "glucose_2hr": 110, "insulin": 9.5}
ACTIVITY = {"avg_daily_activity": 9.5, "activity_variability": 1.6, "avg_active_minutes": 40}
SLEEP = {"mean_sleep_min": 470, "min_sleep_min": 400, "max_sleep_min": 560, "std_sleep_min": 55}


def post(**sections):
    return client.post("/predict", json={"age": 35, **sections})


def test_all_three_sections():
    r = post(biomarker=BIOMARKER, activity=ACTIVITY, sleep=SLEEP)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body["model_predictions"]) == {"biomarker", "activity", "sleep"}
    assert abs(sum(body["weights_used"].values()) - 1) < 0.01
    assert abs(body["predicted_biological_age"] - body["chronological_age"] - body["gap"]) < 0.11


def test_factors_add_up_to_gap():
    body = post(biomarker=BIOMARKER, activity=ACTIVITY).json()
    total = sum(f["impact_years"] for f in body["factors"]) + body["other_years"]
    assert abs(total - body["gap"]) < 0.2  # small rounding differences only


def test_sex_is_not_listed_as_a_factor():
    names = [f["name"] for f in post(biomarker=BIOMARKER).json()["factors"]]
    assert "sex" not in names


def test_single_section_works():
    for section in [{"biomarker": BIOMARKER}, {"activity": ACTIVITY}, {"sleep": SLEEP}]:
        r = post(**section)
        assert r.status_code == 200, r.text
        assert list(r.json()["weights_used"].values()) == [1.0]


def test_no_data_is_rejected():
    assert post().status_code == 422


def test_out_of_range_input_is_rejected():
    assert client.post("/predict", json={"age": 12, "activity": ACTIVITY}).status_code == 422
    assert post(biomarker={**BIOMARKER, "bmi": -5}).status_code == 422