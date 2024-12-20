import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.base import BaseEstimator, TransformerMixin
import gdown
import os

# Custom CSS for marquee (scrolling text)
st.markdown(
    """
    <style>
    .marquee {
        width: 100%;
        margin: 0 auto;
        white-space: nowrap;
        overflow: hidden;
        box-sizing: border-box;
    }
    .marquee span {
        display: inline-block;
        padding-left: 100%;
        animation: marquee 15s linear infinite;
        font-size: 36px; /* Adjust the font size here to make it bigger */
        font-weight: bold; /* Make the text bold */
    }
    @keyframes marquee {
        0% { transform: translate(0, 0); }
        100% { transform: translate(-100%, 0); }
    }
    </style>
    <div class="marquee">
        <span>Recommendations Are Available in These Areas: Mirpur, Bashundhara R/A, Uttara, Badda, Mohammadpur, Banasree, Aftab Nagar, Dakshin Khan, Dhanmondi, Agargaon, and Rampura</span>
    </div>
    """,
    unsafe_allow_html=True
)

# # Load spaCy model
# nlp = spacy.load('en_core_web_sm')

# # Custom transformer for text preprocessing (with lemmatization)
# class TextPreprocessor(BaseEstimator, TransformerMixin):
#     def __init__(self, nlp):
#         self.nlp = nlp

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         return X.apply(lambda text: ' '.join([token.lemma_ for token in self.nlp(str(text))]))

# File paths and Google Drive IDs from Streamlit secrets
GDRIVE_FILES = {
    "similarity_matrix":  st.secrets["keys"]["SIMILARITY_MATRIX_KEY"],
    "rf": st.secrets["keys"]["RF_KEY"]
}

LOCAL_FILES = {
    "similarity_matrix": "similarity_matrix2.pkl",
    "rf": "rf.pkl"
}

# Download similarity matrix from Google Drive if not exists (optional)
def download_file_from_drive(file_id, destination):
    if not os.path.exists(destination):
        with st.spinner(f"Downloading {destination}..."):
            gdown.download(f"https://drive.google.com/uc?id={file_id}", destination, quiet=False)

# Load the similarity matrix
download_file_from_drive(GDRIVE_FILES["similarity_matrix"], LOCAL_FILES["similarity_matrix"])

with open(LOCAL_FILES["similarity_matrix"], 'rb') as f:
    similarity_matrix = pickle.load(f)
with open(LOCAL_FILES["rf"], 'rb') as file:
    rf = pickle.load(file)

# Load the data directly from the provided path
# rf = pd.read_csv(r"data\processed\Recommendation_data.csv")
# rf.dropna(inplace=True)

# Filter function for filtering the data
def filter_data(data, area=None, min_price=None, max_price=None, min_bedrooms=None, max_bedrooms=None):
    filtered_data = data.copy()
    if area:
        filtered_data = filtered_data[filtered_data['area'].str.contains(area, case=False)]
    if min_price is not None:
        filtered_data = filtered_data[filtered_data['price'] >= min_price]
    if max_price is not None:
        filtered_data = filtered_data[filtered_data['price'] <= max_price]
    if min_bedrooms is not None:
        filtered_data = filtered_data[filtered_data['bedrooms'] >= min_bedrooms]
    if max_bedrooms is not None:
        filtered_data = filtered_data[filtered_data['bedrooms'] <= max_bedrooms]
    return filtered_data

# Recommendation function based on similarity
def get_recommendations(index, top_n=5, area=None, min_price=None, max_price=None, min_bedrooms=None, max_bedrooms=None):
    filtered_data = filter_data(rf, area, min_price, max_price, min_bedrooms, max_bedrooms)
    filtered_indices = filtered_data.index.tolist()
    
    # Calculate similarity only for filtered data
    similarity_scores = [(i, similarity_matrix[index][i]) for i in filtered_indices]
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    top_properties = [i[0] for i in similarity_scores[1:top_n+1]]
    return rf[['property_name', 'price', 'bedrooms', 'bathrooms', 'floor_area', 'area', 'short_description', 'property_url']].iloc[top_properties]

# Streamlit UI
st.title("Real Estate Recommendation System")

# User inputs
area = st.selectbox('Enter Area', sorted(rf['area'].unique().tolist()))
min_price = st.number_input("Minimum Price (in Taka)", min_value=0, step=100000)
max_price = st.number_input("Maximum Price (in Taka)", min_value=0, step=100000)
min_bedrooms = st.number_input("Minimum Bedrooms", min_value=0, step=1)
max_bedrooms = st.number_input("Maximum Bedrooms", min_value=0, step=1)

# Button to get recommendations
if st.button("Get Recommendations"):
    recommendations = get_recommendations(0, top_n=5, area=area, min_price=min_price, max_price=max_price, min_bedrooms=min_bedrooms, max_bedrooms=max_bedrooms)
    
    # Display recommendations
    if not recommendations.empty:
        st.write("### Top Recommendations:")
        for i, row in recommendations.iterrows():
            # Create a visually appealing card for each property
            st.markdown(f"""
            <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 15px; margin-bottom: 20px; background-color: #f9f9f9;">
                <h3 style="color: #4CAF50; margin-bottom: 10px;">🏠 {row['property_name']}</h3>
                <p><strong>📍 Location:</strong> {row['area']}</p>
                <p><strong>💰 Price:</strong> {row['price']:,} Taka</p>
                <p><strong>📏 Floor Area:</strong> {row['floor_area']} sqft</p>
                <p><strong>🛏 Bedrooms:</strong> {row['bedrooms']} | <strong>🛁 Bathrooms:</strong> {row['bathrooms']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add the description inside the container using an expander
            with st.expander("See Full Description"):
                st.write(row['short_description'])
            
            # Add a "View Property" button
            st.markdown(f"""
            <a href="{row['property_url']}" target="_blank" style="color: white; background-color: #4CAF50; padding: 10px 15px; border-radius: 5px; text-decoration: none;">View Property</a>
            """, unsafe_allow_html=True)
            st.write("---")
    else:
        st.warning("No properties found matching your criteria.")
