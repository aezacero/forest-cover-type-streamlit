import pytest

from cover_type.data import SOIL_COLUMNS, WILDERNESS_COLUMNS


@pytest.fixture
def valid_raw_row():
    row = {
        "Elevation": 2800,
        "Aspect": 120,
        "Slope": 12,
        "Horizontal_Distance_To_Hydrology": 150,
        "Vertical_Distance_To_Hydrology": 20,
        "Horizontal_Distance_To_Roadways": 1000,
        "Hillshade_9am": 210,
        "Hillshade_Noon": 220,
        "Hillshade_3pm": 140,
        "Horizontal_Distance_To_Fire_Points": 900,
    }

    for col in WILDERNESS_COLUMNS:
        row[col] = 0
    row["Wilderness_Area1"] = 1

    for col in SOIL_COLUMNS:
        row[col] = 0
    row["Soil_Type10"] = 1

    return row
