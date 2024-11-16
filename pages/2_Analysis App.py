# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import pickle
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# import seaborn as sns

# st.set_page_config(page_title="Plotting Demo")

# st.title('Analytics')



# new_df = pd.read_csv("E:\Work files\Real_state\Real state\datasets\data_viz1.csv")
# # feature_text = pickle.load(open("E:\\Work files\\Real_state\\Real state\\datasets\\feature_text.pkl", 'rb'))
# with open(r"E:\Work files\Real_state\Real state\datasets\feature_text.pkl",'rb') as file:
#     feature_text = pickle.load(file)



# group_df = new_df.groupby('sector').mean()[['price','price_per_sqft','built_up_area','latitude','longitude']]

# st.header('Sector Price per Sqft Geomap')
# fig = px.scatter_mapbox(group_df, lat="latitude", lon="longitude", color="price_per_sqft", size='built_up_area',
#                   color_continuous_scale=px.colors.cyclical.IceFire, zoom=10,
#                   mapbox_style="open-street-map",width=1200,height=700,hover_name=group_df.index)

# st.plotly_chart(fig,use_container_width=True)

# st.header('Features Wordcloud')

# wordcloud = WordCloud(width = 800, height = 800,
#                       background_color ='black',
#                       stopwords = set(['s']),  # Any stopwords you'd like to exclude
#                       min_font_size = 10).generate(feature_text)

# plt.figure(figsize = (8, 8), facecolor = None)
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.tight_layout(pad = 0)
# st.pyplot()

# st.header('Area Vs Price')

# property_type = st.selectbox('Select Property Type', ['flat','house'])

# if property_type == 'house':
#     fig1 = px.scatter(new_df[new_df['property_type'] == 'house'], x="built_up_area", y="price", color="bedRoom", title="Area Vs Price")

#     st.plotly_chart(fig1, use_container_width=True)
# else:
#     fig1 = px.scatter(new_df[new_df['property_type'] == 'flat'], x="built_up_area", y="price", color="bedRoom",
#                       title="Area Vs Price")

#     st.plotly_chart(fig1, use_container_width=True)

# st.header('BHK Pie Chart')

# sector_options = new_df['sector'].unique().tolist()
# sector_options.insert(0,'overall')

# selected_sector = st.selectbox('Select Sector', sector_options)

# if selected_sector == 'overall':

#     fig2 = px.pie(new_df, names='bedRoom')

#     st.plotly_chart(fig2, use_container_width=True)
# else:

#     fig2 = px.pie(new_df[new_df['sector'] == selected_sector], names='bedRoom')

#     st.plotly_chart(fig2, use_container_width=True)

# st.header('Side by Side BHK price comparison')

# fig3 = px.box(new_df[new_df['bedRoom'] <= 4], x='bedRoom', y='price', title='BHK Price Range')

# st.plotly_chart(fig3, use_container_width=True)


# st.header('Side by Side Distplot for property type')

# fig3 = plt.figure(figsize=(10, 4))
# sns.distplot(new_df[new_df['property_type'] == 'house']['price'],label='house')
# sns.distplot(new_df[new_df['property_type'] == 'flat']['price'], label='flat')
# plt.legend()
# st.pyplot(fig3)










# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import pickle
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# import seaborn as sns

# st.set_page_config(page_title="Plotting Demo")

# st.title('Analytics')

# new_df = pd.read_csv(r"E:\Work files\Real_state\Real state\data\raw\processed_buy_data.csv")

# # Ensure that only numeric columns are selected for aggregation
# numeric_cols = ['price', 'price_per_sqft', 'built_up_area', 'latitude', 'longitude']
# group_df = new_df.groupby('sector')[numeric_cols].mean()

# st.header('Sector Price per Sqft Geomap')
# fig = px.scatter_mapbox(group_df, lat="latitude", lon="longitude", color="price_per_sqft", size='built_up_area',
#                         color_continuous_scale=px.colors.cyclical.IceFire, zoom=10,
#                         mapbox_style="open-street-map", width=1200, height=700, hover_name=group_df.index)

# st.plotly_chart(fig, use_container_width=True)

# st.header('Features Wordcloud')

# wordcloud = WordCloud(width=800, height=800,
#                       background_color='black',
#                       stopwords=set(['s']),  # Any stopwords you'd like to exclude
#                       min_font_size=10).generate(feature_text)

# plt.figure(figsize=(8, 8), facecolor=None)
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.tight_layout(pad=0)
# st.pyplot()

# st.header('Area Vs Price')

# property_type = st.selectbox('Select Property Type', ['flat', 'house'])

# if property_type == 'house':
#     fig1 = px.scatter(new_df[new_df['property_type'] == 'house'], x="built_up_area", y="price", color="bedRoom",
#                       title="Area Vs Price")

#     st.plotly_chart(fig1, use_container_width=True)
# else:
#     fig1 = px.scatter(new_df[new_df['property_type'] == 'flat'], x="built_up_area", y="price", color="bedRoom",
#                       title="Area Vs Price")

#     st.plotly_chart(fig1, use_container_width=True)

# st.header('BHK Pie Chart')

# sector_options = new_df['sector'].unique().tolist()
# sector_options.insert(0, 'overall')

# selected_sector = st.selectbox('Select Sector', sector_options)

# if selected_sector == 'overall':
#     fig2 = px.pie(new_df, names='bedRoom')
#     st.plotly_chart(fig2, use_container_width=True)
# else:
#     fig2 = px.pie(new_df[new_df['sector'] == selected_sector], names='bedRoom')
#     st.plotly_chart(fig2, use_container_width=True)

# st.header('Side by Side BHK price comparison')

# fig3 = px.box(new_df[new_df['bedRoom'] <= 4], x='bedRoom', y='price', title='BHK Price Range')
# st.plotly_chart(fig3, use_container_width=True)

# st.header('Side by Side Distplot for property type')

# fig3 = plt.figure(figsize=(10, 4))
# sns.distplot(new_df[new_df['property_type'] == 'house']['price'], label='house')
# sns.distplot(new_df[new_df['property_type'] == 'flat']['price'], label='flat')
# plt.legend()
# st.pyplot(fig3)



import pandas as pd
import plotly.express as px
import streamlit as st

# Load and clean the dataset
data = pd.read_csv(r'D:\Work file\bd_real_estate\data\processed\recomendation.csv')
data = data.drop(columns=['Unnamed: 0'])

# Streamlit UI setup
st.title("3D Property Data Visualizations")

# Sidebar for user selections
st.sidebar.title("Customize Your 3D Visualization")

# Select X, Y, and Z columns
column_x = st.sidebar.selectbox("Select X-axis column", options=data.columns)
column_y = st.sidebar.selectbox("Select Y-axis column", options=data.columns)
column_z = st.sidebar.selectbox("Select Z-axis column", options=data.columns)

# Select 3D plot type
plot_type = st.sidebar.selectbox(
    "Select 3D Plot Type", 
    ["3D Scatter Plot", "3D Line Plot", "3D Surface Plot"]
)

# Plot generation based on selections
st.header(f"{plot_type} of {column_z} vs {column_y} vs {column_x}")

if plot_type == "3D Scatter Plot":
    fig = px.scatter_3d(data, x=column_x, y=column_y, z=column_z, 
                        title=f"{column_z} vs {column_y} vs {column_x} (3D Scatter Plot)")

elif plot_type == "3D Line Plot":
    fig = px.line_3d(data, x=column_x, y=column_y, z=column_z, 
                     title=f"{column_z} vs {column_y} vs {column_x} (3D Line Plot)")

elif plot_type == "3D Surface Plot":
    # 3D surface plots require grid-like data. This example uses density heatmap as a workaround for 3D surface
    fig = px.density_heatmap(data, x=column_x, y=column_y, z=column_z, 
                             title=f"{column_z} vs {column_y} vs {column_x} (3D Surface Plot)")

# Display plot
st.plotly_chart(fig)
