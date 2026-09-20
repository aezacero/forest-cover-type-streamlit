import pandas as pd
import pytest

from cover_type.data import RAW_FEATURE_COLUMNS, load_raw_data, validate_raw_row


def test_load_raw_data_returns_expected_columns(tmp_path):
    df = pd.DataFrame([{**{col: 0 for col in RAW_FEATURE_COLUMNS}, "Id": 1, "Cover_Type": 1}])
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)

    loaded = load_raw_data(path)

    assert set(RAW_FEATURE_COLUMNS).issubset(loaded.columns)
    assert len(loaded) == 1


def test_load_raw_data_raises_on_missing_column(tmp_path):
    df = pd.DataFrame([{col: 0 for col in RAW_FEATURE_COLUMNS if col != "Elevation"}])
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Elevation"):
        load_raw_data(path)


def test_validate_raw_row_accepts_valid_row(valid_raw_row):
    validate_raw_row(valid_raw_row)


def test_validate_raw_row_rejects_missing_column(valid_raw_row):
    del valid_raw_row["Elevation"]
    with pytest.raises(ValueError, match="Missing required raw columns"):
        validate_raw_row(valid_raw_row)


def test_validate_raw_row_rejects_nan(valid_raw_row):
    valid_raw_row["Elevation"] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        validate_raw_row(valid_raw_row)


def test_validate_raw_row_rejects_two_wilderness_areas(valid_raw_row):
    valid_raw_row["Wilderness_Area2"] = 1
    with pytest.raises(ValueError, match="Wilderness_Area"):
        validate_raw_row(valid_raw_row)


def test_validate_raw_row_rejects_no_soil_type(valid_raw_row):
    valid_raw_row["Soil_Type10"] = 0
    with pytest.raises(ValueError, match="Soil_Type"):
        validate_raw_row(valid_raw_row)


def test_validate_raw_row_rejects_out_of_range_aspect(valid_raw_row):
    valid_raw_row["Aspect"] = 400
    with pytest.raises(ValueError, match="Aspect"):
        validate_raw_row(valid_raw_row)


def test_validate_raw_row_rejects_out_of_range_hillshade(valid_raw_row):
    valid_raw_row["Hillshade_9am"] = 300
    with pytest.raises(ValueError, match="Hillshade_9am"):
        validate_raw_row(valid_raw_row)
