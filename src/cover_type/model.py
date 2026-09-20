from pathlib import Path

import joblib
import pandas as pd

from .data import validate_raw_row
from .features import engineer_features
from .labels import CLASS_NAMES

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "model.joblib"


def load_model(path=DEFAULT_MODEL_PATH):
    return joblib.load(path)


def predict_one(raw_input: dict, model=None, model_path=DEFAULT_MODEL_PATH):
    validate_raw_row(raw_input)

    if model is None:
        model = load_model(model_path) # loads existing model

    df = pd.DataFrame([raw_input])
    engineered = engineer_features(df)

    label = int(model.predict(engineered)[0])
    proba = model.predict_proba(engineered)[0]
    probs = {CLASS_NAMES[c]: float(p) for c, p in zip(model.classes_, proba)}

    return label, CLASS_NAMES[label], probs
