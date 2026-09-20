import pandas as pd

# Column schema
WILDERNESS_COLUMNS = [f"Wilderness_Area{i}" for i in range(1, 5)]
SOIL_COLUMNS = [f"Soil_Type{i}" for i in range(1, 41)]

NUMERIC_COLUMNS = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]

RAW_FEATURE_COLUMNS = NUMERIC_COLUMNS + WILDERNESS_COLUMNS + SOIL_COLUMNS


def load_raw_data(path):
    df = pd.read_csv(path)
    missing = set(RAW_FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required raw columns: {sorted(missing)}")
    return df


def validate_raw_row(row: dict) -> None:
    missing = set(RAW_FEATURE_COLUMNS) - set(row.keys())
    if missing:
        raise ValueError(f"Missing required raw columns: {sorted(missing)}")

    for col in RAW_FEATURE_COLUMNS:
        if pd.isna(row[col]):
            raise ValueError(f"Column '{col}' cannot be missing/NaN")

    if sum(row[c] for c in WILDERNESS_COLUMNS) != 1:
        raise ValueError("Exactly one Wilderness_Area column must be 1")

    if sum(row[c] for c in SOIL_COLUMNS) != 1:
        raise ValueError("Exactly one Soil_Type column must be 1")

    if not (0 <= row["Aspect"] <= 360):
        raise ValueError("Aspect must be between 0 and 360 degrees")

    if not (0 <= row["Slope"] <= 90):
        raise ValueError("Slope must be between 0 and 90 degrees")

    for col in ("Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"):
        if not (0 <= row[col] <= 255):
            raise ValueError(f"'{col}' must be between 0 and 255")
