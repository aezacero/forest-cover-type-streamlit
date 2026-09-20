import copy

import pytest

from cover_type.labels import CLASS_NAMES
from cover_type.model import load_model, predict_one


@pytest.fixture(scope="module")
def model():
    return load_model()


def test_load_model_loads_a_fitted_pipeline(model):
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_predict_one_returns_valid_label_and_probs(valid_raw_row, model):
    label, name, probs = predict_one(valid_raw_row, model=model)

    assert label in CLASS_NAMES
    assert name == CLASS_NAMES[label]
    assert set(probs.keys()) == set(CLASS_NAMES.values())
    assert probs[name] == pytest.approx(max(probs.values()))
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)


def test_predict_one_rejects_invalid_row(valid_raw_row, model):
    bad_row = copy.deepcopy(valid_raw_row)
    del bad_row["Elevation"]

    with pytest.raises(ValueError):
        predict_one(bad_row, model=model)
