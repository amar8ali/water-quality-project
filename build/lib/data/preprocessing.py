import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_path, output_path=None):
    df = pd.read_csv(input_path)
    # Impute missing values with median
    for col in df.columns:
        df[col] = df[col].fillna(df[col].median())
    # Feature scaling
    features = df.drop('Potability', axis=1)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    df_scaled = pd.DataFrame(features_scaled, columns=features.columns)
    df_scaled['Potability'] = df['Potability'].values
    if output_path:
        df_scaled.to_csv(output_path, index=False)
    return df_scaled
