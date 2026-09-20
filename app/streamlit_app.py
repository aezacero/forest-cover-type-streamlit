import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cover_type.data import SOIL_COLUMNS, WILDERNESS_COLUMNS
from cover_type.model import load_model, predict_one

WILDERNESS_AREAS = {
    "Wilderness_Area1": "Rawah",
    "Wilderness_Area2": "Neota",
    "Wilderness_Area3": "Comanche Peak",
    "Wilderness_Area4": "Cache la Poudre",
}

st.set_page_config(page_title="Forest Cover Type Predictor", layout="centered")
st.title("Forest Cover Type Predictor")
st.caption("Predicts forest cover type from the UCI/Kaggle Forest Cover Type raw terrain features.")


@st.cache_resource
def get_model():
    return load_model()


with st.form("raw_input_form"):
    st.subheader("Terrain")
    elevation = st.slider("Elevation (m)", 1800, 3900, 2754)
    aspect = st.slider("Aspect (degrees azimuth)", 0, 360, 125)
    slope = st.slider("Slope (degrees)", 0, 60, 15)

    st.subheader("Hydrology")
    h_dist_hydro = st.slider("Horizontal Distance To Hydrology (m)", 0, 1400, 180)
    v_dist_hydro = st.slider("Vertical Distance To Hydrology (m)", -150, 600, 32)

    st.subheader("Roads & Fire Points")
    h_dist_roads = st.slider("Horizontal Distance To Roadways (m)", 0, 7000, 1315)
    h_dist_fire = st.slider("Horizontal Distance To Fire Points (m)", 0, 7200, 1266)

    st.subheader("Hillshade")
    hillshade_9am = st.slider("Hillshade at 9am", 0, 255, 220)
    hillshade_noon = st.slider("Hillshade at Noon", 0, 255, 223)
    hillshade_3pm = st.slider("Hillshade at 3pm", 0, 255, 138)

    st.subheader("Wilderness Area")
    wilderness_label = st.radio("Wilderness Area", list(WILDERNESS_AREAS.values()))

    st.subheader("Soil Type")
    soil_label = st.selectbox("Soil Type", [f"Soil Type {i}" for i in range(1, 41)])

    submitted = st.form_submit_button("Predict Cover Type")

if submitted:
    wilderness_col = next(col for col, name in WILDERNESS_AREAS.items() if name == wilderness_label)
    soil_col = f"Soil_Type{soil_label.split()[-1]}"

    raw_input = {
        "Elevation": elevation,
        "Aspect": aspect,
        "Slope": slope,
        "Horizontal_Distance_To_Hydrology": h_dist_hydro,
        "Vertical_Distance_To_Hydrology": v_dist_hydro,
        "Horizontal_Distance_To_Roadways": h_dist_roads,
        "Hillshade_9am": hillshade_9am,
        "Hillshade_Noon": hillshade_noon,
        "Hillshade_3pm": hillshade_3pm,
        "Horizontal_Distance_To_Fire_Points": h_dist_fire,
    }
    for col in WILDERNESS_COLUMNS:
        raw_input[col] = 1 if col == wilderness_col else 0
    for col in SOIL_COLUMNS:
        raw_input[col] = 1 if col == soil_col else 0

    model = get_model()
    label, name, probs = predict_one(raw_input, model=model)

    st.success(f"Predicted Cover Type: **{name}** (class {label})")

    probs_df = (
        pd.DataFrame({"Cover Type": list(probs.keys()), "Probability": list(probs.values())})
        .sort_values("Probability", ascending=False)
        .set_index("Cover Type")
    )
    st.bar_chart(probs_df)
