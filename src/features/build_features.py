# build_feature.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

def clean_data(df):
    """
    Cleans the input DataFrame and prepares features for modeling.
    """
    # Handle missing 'area' using 'sub_area'
    df['area'] = df.apply(lambda x: x['sub_area'] if pd.isnull(x['area']) else x['area'], axis=1)
    
    # Remove extreme outliers for price and floor area
    df = df[df['price_per_sqft'] <= 35000]
    df = df[df['floor_area'] <= 10000]

    return df

def preprocess_features(df):
    """
    Preprocess the features including scaling and encoding.
    """
    # Split numerical and categorical data
    num_features = ['bedrooms', 'bathrooms', 'floor_area']
    cat_features = ['area']

    # Standard scaling for numerical features
    scaler = StandardScaler()
    df_num_scaled = pd.DataFrame(scaler.fit_transform(df[num_features]), columns=num_features)

    # Ordinal encoding for categorical features
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df_cat_encoded = pd.DataFrame(encoder.fit_transform(df[cat_features]), columns=cat_features)

    # Combine preprocessed features
    df_preprocessed = pd.concat([df_num_scaled, df_cat_encoded], axis=1)
    return df_preprocessed, scaler, encoder

if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv(r'D:\Work file\bd_real_estate\data\processed\processed_buy_data.csv', index_col=0)

    # Clean and preprocess data
    df = clean_data(df)
    features, scaler, encoder = preprocess_features(df)
    print(features.head())
