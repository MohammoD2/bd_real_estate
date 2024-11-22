# %% Import Libraries
import pandas as pd
import numpy as np
import math
import joblib

# %% Load Raw Data
def load_data(path):
    """
    Load raw CSV files into DataFrames and concatenate them into a single DataFrame.
    """
    file_names = [
        "agargaon_properties.csv", "all_properties.csv", "badda_properties.csv",
        "Banani_properties.csv", "banasree_properties.csv", "Baridhara_properties.csv",
        "bashundhara-r-a_properties.csv", "gulshan_properties.csv", "keraniganj_properties.csv",
        "khilgaon_properties.csv", "mirpur_properties.csv", "mohakhali_properties.csv",
        "mohammadpur_properties.csv", "Motijheel_properties.csv", "new-market_properties.csv",
        "rampura_properties.csv", "Tejgaon_properties.csv", "uttora_properties.csv","aftab-nagar.csv",
        "Dakshin_Khan.csv","Dhanmondi_properties.csv","jatrabari_properties.csv"
    ]
    dfs = [pd.read_csv(f"{path}/{file_name}") for file_name in file_names]
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

# %% Predict Bathrooms
def predict_bathrooms(row, model):
    """
    Predict bathrooms using the provided model if the value is missing.
    """
    if pd.isnull(row['bathrooms']):
        new_data = pd.DataFrame({'price': [row['price']], 'floor_area': [row['floor_area']]})
        new_data = new_data.reindex(columns=['price', 'floor_area'], fill_value=0)
        predicted = model.predict(new_data)[0]
        return math.ceil(predicted) if (predicted % 1) >= 0.7 else math.floor(predicted)
    return int(row['bathrooms'])

# %% Predict Bedrooms
def predict_bedrooms(row, model):
    """
    Predict bedrooms using the provided model if the value is missing.
    """
    if pd.isnull(row['bedrooms']):
        new_data = pd.DataFrame({'price': [row['price']], 'floor_area': [row['floor_area']]})
        new_data = new_data.reindex(columns=['price', 'floor_area'], fill_value=0)
        predicted = model.predict(new_data)[0]
        return math.ceil(predicted) if (predicted % 1) >= 0.7 else math.floor(predicted)
    return int(row['bedrooms'])

# %% Clean Data
def clean_data(df, bath_model_path, bed_model_path):
    """
    Clean and process the data: fill missing values, remove duplicates, and calculate new features.
    """
    # Load Models
    bath_model = joblib.load(bath_model_path)
    bed_model = joblib.load(bed_model_path)

    # Drop duplicates
    df = df.drop_duplicates()

    # Remove rows with missing mandatory columns
    df = df.dropna(subset=['floor_area', 'price'])

    # Clean numerical columns
    df['floor_area'] = df['floor_area'].replace('[ sqft]', '', regex=True).astype(float)
    df['price'] = df['price'].replace('[৳,]', '', regex=True).astype(float)

    # Predict missing bathrooms and bedrooms
    df['bathrooms'] = df.apply(lambda row: predict_bathrooms(row, bath_model), axis=1)
    df['bedrooms'] = df.apply(lambda row: predict_bedrooms(row, bed_model), axis=1)

    # Clean text columns
    df['short_description'] = (
        df['short_description']
        .str.lower()
        .str.replace(r'[^A-Za-z0-9 ]', '', regex=True)
        .apply(lambda x: ' '.join(str(x).split()))
    )

    # Handle specific missing values
    missing_addresses = {
        617: 'kalachandpur',
        851: 'Kuril',
        870: 'Kuril'
    }
    for idx, address in missing_addresses.items():
        df.loc[idx, 'address'] = address

    # Drop irrelevant or extreme values
    df.drop(df[df['bedrooms'] >= 10].index, inplace=True)
    df.drop(df[df['bathrooms'] >= 10].index, inplace=True)

    # Calculate price per square foot
    df['price_per_sqft'] = df.apply(lambda x: x['price'] / x['floor_area'], axis=1)

    # Split address into sub-area and area
    df[['sub_area', 'area']] = df['address'].str.split(',', expand=True)

    return df

# %% Save Processed Data
def save_data(df, output_path):
    """
    Save the cleaned data to a CSV file.
    """
    df.to_csv(output_path, index=False)

# %% Main Execution
if __name__ == "__main__":
    RAW_DATA_PATH = r"D:/Work file/bd_real_estate/data/raw"
    OUTPUT_PATH = r"D:/Work file/bd_real_estate/data/processed/processed_buy_data2.csv"
    BATH_MODEL_PATH = r"D:\Work file\bd_real_estate\models\Xgboost_model_bath.joblib"
    BED_MODEL_PATH = r"D:\Work file\bd_real_estate\models\Xgboost_model_bed.joblib"

    # Load and process the data
    raw_data = load_data(RAW_DATA_PATH)
    processed_data = clean_data(raw_data, BATH_MODEL_PATH, BED_MODEL_PATH)

    # Save the cleaned data
    save_data(processed_data, OUTPUT_PATH)


    