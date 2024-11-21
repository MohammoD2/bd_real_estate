import pandas as pd
import plotly.express as px
import streamlit as st
import pickle
import gdown
import os
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
    with open(LOCAL_FILES["rf"], 'rb') as file:
        rf = pickle.load(file)

    return  rf

# Load resources
rf = load_resources()
# Load and clean the dataset
data = rf
data = data.drop(columns=['Unnamed: 0'])

# Streamlit UI setup
st.title("Property Data Analytics and Visualization")

# Filter by area
st.subheader("Select Your Area")
available_areas = data['area'].unique().tolist()
selected_area = st.selectbox("Select an Area", options=["All"] + sorted(available_areas))

# Filter the dataset based on the selected area
if selected_area != "All":
    filtered_data = data[data['area'] == selected_area]
else:
    filtered_data = data

# Display the number of records in the filtered dataset
st.write(f"Number of records for selected area: **{len(filtered_data)}**")

# In-page plot selection
st.subheader("Select Plot Type and Axes")
analysis_type = st.selectbox("Choose Analysis Type", ["2D Plot", "3D Plot"])

if analysis_type == "2D Plot":
    st.write("### 2D Visualization")
    col1, col2 = st.columns(2)
    
    with col1:
        column_x = st.selectbox("Select X-axis column", options=filtered_data.columns, key='2d_x')
        column_y = st.selectbox("Select Y-axis column", options=filtered_data.columns, key='2d_y')

    plot_type_2d = st.selectbox(
        "Select 2D Plot Type",
        ["Scatter Plot", "Line Plot", "Bar Plot", "Histogram", "Box Plot", "Violin Plot", "Density Plot"],
        key='2d_plot_type'
    )

    st.write(f"**{plot_type_2d} of {column_y} vs {column_x}**")
    if plot_type_2d == "Scatter Plot":
        fig = px.scatter(filtered_data, x=column_x, y=column_y, title=f"{column_y} vs {column_x} (Scatter Plot)")
    elif plot_type_2d == "Line Plot":
        fig = px.line(filtered_data, x=column_x, y=column_y, title=f"{column_y} vs {column_x} (Line Plot)")
    elif plot_type_2d == "Bar Plot":
        fig = px.bar(filtered_data, x=column_x, y=column_y, title=f"{column_y} vs {column_x} (Bar Plot)")
    elif plot_type_2d == "Histogram":
        fig = px.histogram(filtered_data, x=column_x, y=column_y, title=f"{column_y} vs {column_x} (Histogram)")
    elif plot_type_2d == "Box Plot":
        fig = px.box(filtered_data, x=column_x, y=column_y, title=f"{column_y} vs {column_x} (Box Plot)")
    elif plot_type_2d == "Violin Plot":
        fig = px.violin(filtered_data, x=column_x, y=column_y, box=True, points="all", 
                        title=f"{column_y} vs {column_x} (Violin Plot)")
    elif plot_type_2d == "Density Plot":
        fig = px.density_contour(filtered_data, x=column_x, y=column_y, title=f"{column_y} vs {column_x} (Density Plot)")

    # Display the plot
    st.plotly_chart(fig)

elif analysis_type == "3D Plot":
    st.write("### 3D Visualization")
    col1, col2, col3 = st.columns(3)

    with col1:
        column_x = st.selectbox("Select X-axis column", options=filtered_data.columns, key='3d_x')
    with col2:
        column_y = st.selectbox("Select Y-axis column", options=filtered_data.columns, key='3d_y')
    with col3:
        column_z = st.selectbox("Select Z-axis column", options=filtered_data.columns, key='3d_z')

    plot_type_3d = st.selectbox(
        "Select 3D Plot Type",
        ["3D Scatter Plot", "3D Line Plot", "3D Surface Plot", "3D Bubble Plot"],
        key='3d_plot_type'
    )

    st.write(f"**{plot_type_3d} of {column_z} vs {column_y} vs {column_x}**")
    if plot_type_3d == "3D Scatter Plot":
        fig = px.scatter_3d(filtered_data, x=column_x, y=column_y, z=column_z, 
                            title=f"{column_z} vs {column_y} vs {column_x} (3D Scatter Plot)")
    elif plot_type_3d == "3D Line Plot":
        fig = px.line_3d(filtered_data, x=column_x, y=column_y, z=column_z, 
                         title=f"{column_z} vs {column_y} vs {column_x} (3D Line Plot)")
    elif plot_type_3d == "3D Surface Plot":
        fig = px.density_heatmap(filtered_data, x=column_x, y=column_y, z=column_z, 
                                 title=f"{column_z} vs {column_y} vs {column_x} (3D Surface Plot)")
    elif plot_type_3d == "3D Bubble Plot":
        fig = px.scatter_3d(filtered_data, x=column_x, y=column_y, z=column_z, size="price", color="area", 
                            title=f"{column_z} vs {column_y} vs {column_x} (3D Bubble Plot)")

    # Display the plot
    st.plotly_chart(fig)
