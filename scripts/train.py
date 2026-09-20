import sys
from pathlib import Path

import joblib
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cover_type.data import load_raw_data
from cover_type.features import engineer_features

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "train.csv"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"

# Winning hyperparameters from RandomizedSearchCV in notebooks/3_Modeling.ipynb (cell 13)
BEST_PARAMS = {
    "colsample_bytree": 0.9312901539863683,
    "learning_rate": 0.10406933945465861,
    "min_child_samples": 98,
    "n_estimators": 214,
    "num_leaves": 120,
    "objective": "multiclass",
    "random_state": 42,
    "reg_alpha": 0.012709563372047594,
    "reg_lambda": 0.10789142699330445,
    "subsample": 0.7094287557060203,
    "verbosity": -1,
}


def main():
    raw = load_raw_data(DATA_PATH)
    y = raw["Cover_Type"]
    X = engineer_features(raw.drop(columns=["Id", "Cover_Type"]))

    pipe = Pipeline([("model", LGBMClassifier(**BEST_PARAMS))])
    pipe.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
