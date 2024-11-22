# predict_model.py

import pickle
import pandas as pd
import numpy as np
from src.features.build_features import clean_data, preprocess_features

def load_model(model_path):
    """
    Load a trained model from a file.
    """
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def predict(model, data):
    """
    Predict house prices using the trained model.
    """
    # Preprocess input features
    data_preprocessed, _, _ = preprocess_features(data)
    predictions = model.predict(data_preprocessed)
    # Transform predictions back to original scale
    predictions = np.expm1(predictions)
    return predictions

if __name__ == "__main__":
    # Load the trained model
    model_path = 'final_model.pkl'
    model = load_model(model_path)

    # Example data for prediction
    data = [[ 'mirpur', 2, 2, 1005, 8796]]  # Example input
    columns = ['area', 'bedrooms', 'bathrooms', 'floor_area', 'price_per_sqft']
    data_df = pd.DataFrame(data, columns=columns)

    # Predict house price
    predictions = predict(model, data_df)
    print(f"Predicted Price: {predictions}")
