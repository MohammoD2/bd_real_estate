import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.base import BaseEstimator, TransformerMixin
import spacy

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Custom transformer for text preprocessing (with lemmatization)
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, nlp):
        self.nlp = nlp

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(lambda text: ' '.join([token.lemma_ for token in self.nlp(str(text))]))

# Load similarity matrix and data
with open(r"D:\Work file\bd_real_estate\models\similarity_matrix2.pkl", 'rb') as f:
    similarity_matrix = pickle.load(f)
df = pd.read_csv(r"D:\Work file\bd_real_estate\data\processed\recomendation.csv")
df.dropna(inplace=True)
# Filter function
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

# Recommendation function
def get_recommendations(index, top_n=5, area=None, min_price=None, max_price=None, min_bedrooms=None, max_bedrooms=None):
    filtered_data = filter_data(df, area, min_price, max_price, min_bedrooms, max_bedrooms)
    filtered_indices = filtered_data.index.tolist()
    
    # Calculate similarity only for filtered data
    similarity_scores = [(i, similarity_matrix[index][i]) for i in filtered_indices]
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    top_properties = [i[0] for i in similarity_scores[1:top_n+1]]
    return df[['property_name', 'price', 'bedrooms', 'bathrooms', 'floor_area', 'area', 'short_description', 'property_url']].iloc[top_properties]

# Streamlit UI
st.title("Real Estate Recommendation System")

# User inputs
area = st.selectbox('Enter area',sorted(df['area'].unique().tolist()))
min_price = st.number_input("Minimum Price", min_value=0, step=100000)
max_price = st.number_input("Maximum Price", min_value=0, step=100000)
min_bedrooms = st.number_input("Minimum Bedrooms", min_value=0, step=1)
max_bedrooms = st.number_input("Maximum Bedrooms", min_value=0, step=1)

# Button to get recommendations
if st.button("Get Recommendations"):
    recommendations = get_recommendations(0, top_n=5, area=area, min_price=min_price, max_price=max_price, min_bedrooms=min_bedrooms, max_bedrooms=max_bedrooms)
    
    # Display recommendations
    st.write("Top Recommendations:")
    for i, row in recommendations.iterrows():
        st.write(f"**Property Name**: {row['property_name']}")
        st.write(f"**Location (Area)**: {row['area']}")
        st.write(f"**Price**: {row['price']}")
        st.write(f"**Floor Area**: {row['floor_area']} sqft")
        st.write(f"**Bedrooms**: {row['bedrooms']}, **Bathrooms**: {row['bathrooms']}")
        st.write(f"**Description**: {row['short_description'][:100]}...")  # Display first 100 characters
        st.write(f"[View Property]({row['property_url']})")
        st.write("---")
