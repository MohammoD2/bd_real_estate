import streamlit as st
import os
import pickle
import pandas as pd
import numpy as np
import gdown
# import spacy
import plotly.express as px

# Set page configuration
st.set_page_config(page_title="Integrated Real Estate App")
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
# Set page configuration (this should be the first Streamlit command)
# Retrieve the Google Drive file IDs from Streamlit secrets management
GDRIVE_FILES = {
    "similarity_matrix": st.secrets["keys"]["SIMILARITY_MATRIX_KEY"],
    "pipeline": st.secrets["keys"]["PIPELINE_KEY"],
    "df": st.secrets["keys"]["DF_KEY"],
    "rf": st.secrets["keys"]["RF_KEY"]
}

LOCAL_FILES = {
    "similarity_matrix": "similarity_matrix2.pkl",
    "pipeline": "pipeline.pkl",
    "df": "df.pkl",
    "rf": "rf.pkl"
}

# Download a file from Google Drive if it doesn't exist locally
def download_file_from_drive(file_id, destination):
    if not os.path.exists(destination):
        with st.spinner(f"Downloading {destination}..."):
            gdown.download(f"https://drive.google.com/uc?id={file_id}", destination, quiet=False)

# Download all required files
for key, file_id in GDRIVE_FILES.items():
    download_file_from_drive(file_id, LOCAL_FILES[key])

# Load resources (model, pipeline, and data) once using @st.cache_resource
@st.cache_resource
def load_resources():
    # Load pickled data
    with open(LOCAL_FILES["df"], 'rb') as file:
        df = pickle.load(file)
    with open(LOCAL_FILES["rf"], 'rb') as file:
        rf = pickle.load(file)
    with open(LOCAL_FILES["pipeline"], 'rb') as file:
        pipeline = pickle.load(file)
    with open(LOCAL_FILES["similarity_matrix"], 'rb') as file:
        similarity_matrix = pickle.load(file)

    return df, pipeline, similarity_matrix, rf

# Load resources
df, pipeline, similarity_matrix, rf = load_resources()

# Custom transformer for text preprocessing
# class TextPreprocessor(BaseEstimator, TransformerMixin):
#     def __init__(self, nlp):
#         self.nlp = nlp

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         return X.apply(lambda text: ' '.join([token.lemma_ for token in self.nlp(str(text))]))

# Helper function: filter recommendation data based on rf (recommendation model or dataset)
def filter_data_rf(data, area=None, min_price=None, max_price=None, min_bedrooms=None, max_bedrooms=None):
    # Assuming rf is a DataFrame or something that can be filtered like a DataFrame
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

# Helper function: get recommendations based on rf
def get_recommendations_rf(index, top_n=5, area=None, min_price=None, max_price=None, min_bedrooms=None, max_bedrooms=None):
    # Use the filter function tailored to rf data
    filtered_data = filter_data_rf(rf, area, min_price, max_price, min_bedrooms, max_bedrooms).reset_index()
    if filtered_data.empty:
        st.write("No properties match the filtering criteria.")
        return pd.DataFrame()
    
    if index not in filtered_data.index:
        st.write(f"Warning: The given index {index} is not available in the filtered data.")
        return pd.DataFrame()
    
    similarity_scores = [(i, similarity_matrix[index][i]) for i in filtered_data.index]
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    top_properties = [i[0] for i in similarity_scores[1:top_n+1]]
    recommendations = filtered_data.iloc[top_properties]
    
    return recommendations

# Streamlit App
st.title("Real Estate Prediction, Recommendation, and Analysis")

# Step 1: Price Prediction
st.header("Step 1: Price Prediction")
area = st.selectbox('Area name', sorted(df['area'].unique().tolist()))
bedrooms = float(st.selectbox('Number of Bedrooms', sorted(df['bedrooms'].unique().tolist())))
bathrooms = float(st.selectbox('Number of Bathrooms', sorted(df['bathrooms'].unique().tolist())))
floor_area = float(st.number_input('Built-up Area'))
price_per_sqft = float(st.number_input('Price per Square Foot'))

if st.button("Predict Price"):
    data = [[area, bedrooms, bathrooms, floor_area, price_per_sqft]]
    columns = ['area', 'bedrooms', 'bathrooms', 'floor_area', 'price_per_sqft']
    prediction_df = pd.DataFrame(data, columns=columns)

    base_price = np.expm1(pipeline.predict(prediction_df))[0]
    st.session_state.base_price = base_price
    st.session_state.low_price = base_price - 200000
    st.session_state.lowest_price = base_price - 2000000
    st.session_state.high_price = base_price

    st.write(f"The predicted price is between **{round(st.session_state.low_price, 2)}** Taka and **{round(st.session_state.high_price, 2)}** Taka.")

if 'base_price' in st.session_state:
    # Step 2: Recommendation trigger
    if st.button("Get Recommendations for this Price"):
        st.header("Step 2: Property Recommendations")
        recommendations = get_recommendations_rf(
            0, 
            top_n=5, 
            area=area, 
            min_price=st.session_state.lowest_price, 
            max_price=st.session_state.high_price, 
            min_bedrooms=bedrooms
        )
        
        if recommendations.empty:
            st.write("No recommendations found for the given criteria.")
        else:
            st.session_state.recommended_properties = recommendations  # Save recommended properties for visualization
            for i, row in recommendations.iterrows():
                st.write(f"**Property Name**: {row['property_name']}")
                st.write(f"**Location (Area)**: {row['area']}")
                st.write(f"**Price**: {row['price']}")
                st.write(f"**Floor Area**: {row['floor_area']} sqft")
                st.write(f"**Bedrooms**: {row['bedrooms']}, **Bathrooms**: {row['bathrooms']}")
                st.write(f"**Description**: {row['short_description'][:100]}...")
                st.write(f"[View Property]({row['property_url']})")
                st.write("---")

if 'base_price' in st.session_state and 'recommended_properties' in st.session_state:
    # Step 3: Analysis trigger
    # Step 3: Analysis trigger
    if st.button("Analyze Recommendations"):
        st.header("Step 3: Recommendation Analysis")
        
        # Ensure recommendations exist before copying or using it
        if 'recommended_properties' in st.session_state:
            recommendations = st.session_state.recommended_properties
            
            # Check if recommendations is not empty
            if not recommendations.empty:
                recommended_analysis_data = recommendations.copy()
                st.session_state.recommended_analysis_data = recommended_analysis_data

    if 'recommended_analysis_data' in st.session_state:
        recommended_analysis_data = st.session_state.recommended_analysis_data

        st.subheader("Customize Your Visualization")
        # Select plot type: 2D or 3D
        plot_dimension = st.selectbox("Select Plot Dimension", ["2D Plot", "3D Plot"], key="plot_dimension")

        if plot_dimension == "2D Plot":
            st.subheader("2D Plot Settings")
            # 2D Plot customization
            col1, col2 = st.columns(2)
            with col1:
                column_x = st.selectbox("Select X-axis column", options=recommended_analysis_data.columns, key='2d_x_col')
            with col2:
                column_y = st.selectbox("Select Y-axis column", options=recommended_analysis_data.columns, key='2d_y_col')

            plot_type_2d = st.selectbox(
                "Select 2D Plot Type", 
                [
                    "Scatter Plot", 
                    "Line Plot", 
                    "Bar Plot", 
                    "Histogram", 
                    "Box Plot", 
                    "Violin Plot", 
                    "Density Plot"
                ], 
                key="plot_type_2d"
            )

            st.write(f"**{plot_type_2d} of {column_y} vs {column_x}**")
            if plot_type_2d == "Scatter Plot":
                fig = px.scatter(recommended_analysis_data, x=column_x, y=column_y)
            elif plot_type_2d == "Line Plot":
                fig = px.line(recommended_analysis_data, x=column_x, y=column_y)
            elif plot_type_2d == "Bar Plot":
                fig = px.bar(recommended_analysis_data, x=column_x, y=column_y)
            elif plot_type_2d == "Histogram":
                fig = px.histogram(recommended_analysis_data, x=column_x, y=column_y)
            elif plot_type_2d == "Box Plot":
                fig = px.box(recommended_analysis_data, x=column_x, y=column_y)
            elif plot_type_2d == "Violin Plot":
                fig = px.violin(recommended_analysis_data, x=column_x, y=column_y, box=True, points="all")
            elif plot_type_2d == "Density Plot":
                fig = px.density_contour(recommended_analysis_data, x=column_x, y=column_y)

            st.plotly_chart(fig)

        elif plot_dimension == "3D Plot":
            st.subheader("3D Plot Settings")
            # 3D Plot customization
            col1, col2, col3 = st.columns(3)
            with col1:
                column_x = st.selectbox("Select X-axis column", options=recommended_analysis_data.columns, key='3d_x_col')
            with col2:
                column_y = st.selectbox("Select Y-axis column", options=recommended_analysis_data.columns, key='3d_y_col')
            with col3:
                column_z = st.selectbox("Select Z-axis column", options=recommended_analysis_data.columns, key='3d_z_col')

            plot_type_3d = st.selectbox(
                "Select 3D Plot Type", 
                [
                    "Scatter Plot", 
                    "Line Plot", 
                    "Surface Plot", 
                    "Mesh Plot", 
                    "Bubble Plot"
                ], 
                key="plot_type_3d"
            )

            st.write(f"**{plot_type_3d} of {column_z} vs {column_y} vs {column_x}**")
            if plot_type_3d == "Scatter Plot":
                fig = px.scatter_3d(recommended_analysis_data, x=column_x, y=column_y, z=column_z)
            elif plot_type_3d == "Line Plot":
                fig = px.line_3d(recommended_analysis_data, x=column_x, y=column_y, z=column_z)
            elif plot_type_3d == "Surface Plot":
                fig = px.density_heatmap(recommended_analysis_data, x=column_x, y=column_y, z=column_z)
            elif plot_type_3d == "Mesh Plot":
                fig = px.mesh3d(
                    recommended_analysis_data, 
                    x=column_x, 
                    y=column_y, 
                    z=column_z,
                    opacity=0.5
                )
            elif plot_type_3d == "Bubble Plot":
                fig = px.scatter_3d(
                    recommended_analysis_data, 
                    x=column_x, 
                    y=column_y, 
                    z=column_z, 
                    size="price",  # Example: bubble size based on 'price'
                    color="area"   # Example: color based on 'area'
                )

            st.plotly_chart(fig)
