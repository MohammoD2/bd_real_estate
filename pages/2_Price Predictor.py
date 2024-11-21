import streamlit as st
import pickle
import pandas as pd
import numpy as np
import gdown
import os

# Set page configuration
st.set_page_config(page_title="Price Prediction Model")

# Retrieve Google Drive file IDs from Streamlit secrets
GDRIVE_FILES = {
    "df": st.secrets["keys"]["DF_KEY"],
    "pipeline": st.secrets["keys"]["PIPELINE_KEY"]
}

# Local file paths
LOCAL_FILES = {
    "df": "df.pkl",
    "pipeline": "pipeline.pkl"
}

# Download a file from Google Drive if it doesn't exist locally
def download_file_from_drive(file_id, destination):
    if not os.path.exists(destination):
        with st.spinner(f"Downloading {destination}..."):
            gdown.download(f"https://drive.google.com/uc?id={file_id}", destination, quiet=False)

# Download required files
for key, file_id in GDRIVE_FILES.items():
    download_file_from_drive(file_id, LOCAL_FILES[key])

# Cache resource loading
@st.cache_resource
def load_resources():
    # Load DataFrame
    with open(LOCAL_FILES["df"], 'rb') as file:
        df = pickle.load(file)
    # Load Pipeline
    with open(LOCAL_FILES["pipeline"], 'rb') as file:
        pipeline = pickle.load(file)
    return df, pipeline

# Load resources
df, pipeline = load_resources()

# Streamlit app header
st.header('Enter Your Inputs')

# User inputs
area = st.selectbox('Area Name', sorted(df['area'].unique().tolist()))
bedrooms = float(st.selectbox('Number of Bedrooms', sorted(df['bedrooms'].unique().tolist())))
bathrooms = float(st.selectbox('Number of Bathrooms', sorted(df['bathrooms'].unique().tolist())))
floor_area = float(st.number_input('Built Up Area (sqft)', min_value=0.0))
price_per_sqft = float(st.number_input('Price per Sqft (Taka)', min_value=0.0))

# Predict button
if st.button('Predict'):
    # Form the input DataFrame
    input_data = pd.DataFrame(
        [[area, bedrooms, bathrooms, floor_area, price_per_sqft]],
        columns=['area', 'bedrooms', 'bathrooms', 'floor_area', 'price_per_sqft']
    )

    # Prediction
    base_price = np.expm1(pipeline.predict(input_data))[0]
    low = base_price - 200000
    high = base_price

    # Display the prediction
    st.success(f"The price of the flat is estimated between **{round(low, 2):,} Taka** and **{round(high, 2):,} Taka**.")
