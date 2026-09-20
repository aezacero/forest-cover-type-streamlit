import math

import pandas as pd
import pytest

from cover_type.features import PRUNED_FEATURES, engineer_features


def test_engineer_features_output_columns(valid_raw_row):
    df = pd.DataFrame([valid_raw_row])
    engineered = engineer_features(df)

    assert list(engineered.columns) == PRUNED_FEATURES


def test_engineer_features_computed_values(valid_raw_row):
    df = pd.DataFrame([valid_raw_row])
    engineered = engineer_features(df).iloc[0]

    expected_euclidean = math.hypot(
        valid_raw_row["Horizontal_Distance_To_Hydrology"],
        valid_raw_row["Vertical_Distance_To_Hydrology"],
    )
    assert engineered["Euclidean_Distance_To_Hydrology"] == pytest.approx(expected_euclidean)

    assert engineered["Near_Water"] == (
        valid_raw_row["Horizontal_Distance_To_Hydrology"] < 100
        and abs(valid_raw_row["Vertical_Distance_To_Hydrology"]) < 20
    )

    assert engineered["Hillshade_Range"] == (
        valid_raw_row["Hillshade_3pm"] - valid_raw_row["Hillshade_9am"]
    )

    expected_northness = math.cos(math.radians(valid_raw_row["Aspect"]))
    expected_slope_x_northness = valid_raw_row["Slope"] * expected_northness
    assert engineered["Slope_x_Northness"] == pytest.approx(expected_slope_x_northness)

    expected_road_to_fire = valid_raw_row["Horizontal_Distance_To_Roadways"] / (
        valid_raw_row["Horizontal_Distance_To_Fire_Points"] + 1
    )
    assert engineered["Road_to_Fire_Ratio"] == pytest.approx(expected_road_to_fire)

    expected_remote_index = (
        valid_raw_row["Horizontal_Distance_To_Roadways"]
        + valid_raw_row["Horizontal_Distance_To_Fire_Points"]
    ) / 2
    assert engineered["Remote_Index"] == pytest.approx(expected_remote_index)

    expected_slope_position = valid_raw_row["Elevation"] * valid_raw_row["Slope"] / 1000
    assert engineered["Slope_Position"] == pytest.approx(expected_slope_position)


def test_engineer_features_fills_missing_elevation_zone_columns(valid_raw_row):
    # Elevation 2000 -> 'low' zone, which is not one of the two pruned zone columns,
    # so both should come back as 0 rather than raising a KeyError
    valid_raw_row["Elevation"] = 2000
    df = pd.DataFrame([valid_raw_row])

    engineered = engineer_features(df).iloc[0]

    assert engineered["Elevation_Zone_high"] == 0
    assert engineered["Elevation_Zone_mid_high"] == 0


def test_engineer_features_sets_high_elevation_zone(valid_raw_row):
    valid_raw_row["Elevation"] = 3500  # > 3250 -> 'high' zone
    df = pd.DataFrame([valid_raw_row])

    engineered = engineer_features(df).iloc[0]

    assert engineered["Elevation_Zone_high"] == 1
    assert engineered["Elevation_Zone_mid_high"] == 0
