import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="Price prediction model")



with open(r"D:\Work file\bd_real_estate\notebooks\df.pkl",'rb') as file:
    df = pickle.load(file)

with open(r"D:\Work file\bd_real_estate\notebooks\pipeline.pkl",'rb') as file:
    pipeline = pickle.load(file)


st.header('Enter your inputs')
# sector
area = st.selectbox('Area name',sorted(df['area'].unique().tolist()))

bedrooms = float(st.selectbox('Number of Bedroom',sorted(df['bedrooms'].unique().tolist())))

bathrooms = float(st.selectbox('Number of Bathrooms',sorted(df['bathrooms'].unique().tolist())))
floor_area = float(st.number_input('Built Up Area'))
price_per_sqft =float(st.number_input('price per sqft'))
if st.button('Predict'):

    # form a dataframe
    data = [[area, bedrooms,bathrooms, floor_area, price_per_sqft]]
    columns = [ 'area', 'bedrooms', 'bathrooms','floor_area', 'price_per_sqft']
    # Convert to DataFrame
    one_df = pd.DataFrame(data, columns=columns)

    #st.dataframe(one_df)

    # predict
    base_price = np.expm1(pipeline.predict(one_df))[0]
    low = base_price - 200000
    high = base_price 

    # display
    st.text("The price of the flat is between {} taka and {} taka".format(round(low,2),round(high,2)))
