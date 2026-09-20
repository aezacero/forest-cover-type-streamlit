# Forest Cover Type Predictor

A Streamlit app that predicts forest cover type (7 classes) from raw terrain
features, backed by a LightGBM model. Built on top of the
[UCI/Kaggle Forest Cover Type](https://www.kaggle.com/c/forest-cover-type-prediction)
dataset.

## Architecture

- `src/cover_type/` — production code, framework-agnostic:
  - `data.py` — raw column schema, data loading, input validation.
  - `features.py` — the feature engineering pipeline (ported from the
    original feature engineering notebook).
  - `labels.py` — Cover Type class ID → name mapping.
  - `model.py` — model loading and single-row prediction.
- `scripts/train.py` — reproducible training script: raw data → engineered
  features → fitted pipeline → `models/model.joblib`.
- `app/streamlit_app.py` — the Streamlit UI.
- `tests/` — unit tests for `src/cover_type/`.

This repo contains the production app only. It was developed from a set of
exploratory Jupyter notebooks (EDA, feature engineering, modeling, model
selection, evaluation) that are not included here.

## Model

LightGBM classifier, hyperparameters selected via `RandomizedSearchCV`
during exploratory model selection (5-fold cross-validated accuracy:
**87.6%**). The exact winning hyperparameters are hardcoded in
`scripts/train.py`.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group dev
```

## Train the model

```bash
uv run python scripts/train.py
```

Produces `models/model.joblib`, committed to the repo so the app can run
without retraining.

## Run the app

```bash
uv run streamlit run app/streamlit_app.py
```

## Run tests / lint

```bash
uv run pytest
uv run ruff check src app scripts tests
```

## Run with Docker

```bash
docker build -t cover-type-app .
docker run -p 8501:8501 cover-type-app
```

Then open http://localhost:8501.

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs lint, tests, and a Docker
build on every push/PR to `main`.
