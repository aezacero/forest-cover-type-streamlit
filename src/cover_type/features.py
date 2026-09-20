import numpy as np
import pandas as pd

# Pruned features based on EDA and engineering 
PRUNED_FEATURES = [
    'Slope',
    'Horizontal_Distance_To_Roadways',
    'Hillshade_9am',
    'Hillshade_Noon',
    'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points',
    'Wilderness_Area1',
    'Wilderness_Area2',
    'Wilderness_Area3',
    'Wilderness_Area4',
    'Soil_Type1',
    'Soil_Type2',
    'Soil_Type3',
    'Soil_Type4',
    'Soil_Type5',
    'Soil_Type6',
    'Soil_Type10',
    'Soil_Type12',
    'Soil_Type13',
    'Soil_Type14',
    'Soil_Type16',
    'Soil_Type17',
    'Soil_Type18',
    'Soil_Type19',
    'Soil_Type20',
    'Soil_Type22',
    'Soil_Type23',
    'Soil_Type24',
    'Soil_Type26',
    'Soil_Type29',
    'Soil_Type30',
    'Soil_Type31',
    'Soil_Type32',
    'Soil_Type33',
    'Soil_Type35',
    'Soil_Type37',
    'Soil_Type38',
    'Soil_Type39',
    'Soil_Type40',
    'Elevation_x_Rawah',
    'Elevation_x_Neota',
    'Elevation_x_Comanche',
    'Elevation_x_Cache',
    'Euclidean_Distance_To_Hydrology',
    'Near_Water',
    'Hillshade_Range',
    'Slope_x_Northness',
    'Road_to_Fire_Ratio',
    'Hydro_to_Roads_Ratio',
    'Elevation_Zone_high',
    'Elevation_Zone_mid_high',
    'High_North',
    'Remote_Index',
    'Slope_Position',
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Create elevation bands based on ecological knowledge
    def elevation_zone(elev):
        if elev < 2750:
            return 'low'  # Ponderosa, Douglas-fir
        elif elev < 3000:
            return 'mid_low'  # Lodgepole
        elif elev < 3250:
            return 'mid_high'  # Spruce/Fir transition
        else:
            return 'high'  # Spruce/Fir, Krummholz

    df['Elevation_Zone'] = df['Elevation'].apply(elevation_zone)
    df = pd.get_dummies(df, columns=['Elevation_Zone'])

    df['Elevation_x_Rawah'] = df['Elevation'] * df['Wilderness_Area1']
    df['Elevation_x_Neota'] = df['Elevation'] * df['Wilderness_Area2']
    df['Elevation_x_Comanche'] = df['Elevation'] * df['Wilderness_Area3']
    df['Elevation_x_Cache'] = df['Elevation'] * df['Wilderness_Area4']

    # Euclidean distance to water (hypotenuse of horizontal and vertical)
    df['Euclidean_Distance_To_Hydrology'] = np.sqrt(
        df['Horizontal_Distance_To_Hydrology']**2 +
        df['Vertical_Distance_To_Hydrology']**2
    )

    # Water presence flag (areas near water)
    df['Near_Water'] = (df['Horizontal_Distance_To_Hydrology'] < 100) & \
                        (abs(df['Vertical_Distance_To_Hydrology']) < 20)

    # Roughness - difference between hillshade times
    df['Hillshade_Range'] = df['Hillshade_3pm'] - df['Hillshade_9am']
    df['Hillshade_Variance'] = df[['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']].var(axis=1)

    # Slope-aspect interaction (northness/eastness)
    df['Northness'] = np.cos(np.radians(df['Aspect']))
    df['Eastness'] = np.sin(np.radians(df['Aspect']))
    df['Slope_x_Northness'] = df['Slope'] * df['Northness']

    # Roads vs Fire points ratio (different access patterns)
    df['Road_to_Fire_Ratio'] = df['Horizontal_Distance_To_Roadways'] / \
                                (df['Horizontal_Distance_To_Fire_Points'] + 1)

    # Hydrology to Roads ratio
    df['Hydro_to_Roads_Ratio'] = df['Horizontal_Distance_To_Hydrology'] / \
                                  (df['Horizontal_Distance_To_Roadways'] + 1)

    # High elevation + specific aspects (Spruce/Fir prefer north aspects)
    df['High_North'] = (df['Elevation'] > 3000) & (df['Northness'] > 0.5)

    # Distance to roads + fire points (managed vs natural areas)
    df['Remote_Index'] = (df['Horizontal_Distance_To_Roadways'] +
                           df['Horizontal_Distance_To_Fire_Points']) / 2

    # Slope position index
    df['Slope_Position'] = df['Elevation'] * df['Slope'] / 1000

    # get_dummies() only creates the Elevation_Zone_* columns present in this input,
    # so a single-row prediction can be missing one of the two pruned zone columns
    for col in PRUNED_FEATURES:
        if col not in df.columns:
            df[col] = 0

    return df[PRUNED_FEATURES]
