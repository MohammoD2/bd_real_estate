# data_visualization.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_distributions(df, column, title):
    """
    Plot a distribution plot for a given column.
    """
    sns.distplot(df[column])
    plt.title(title)
    plt.show()

def plot_boxplot(df, column, title):
    """
    Plot a boxplot for a given column.
    """
    sns.boxplot(x=df[column])
    plt.title(title)
    plt.xticks(rotation=90, ha='right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv(r'D:\Work file\bd_real_estate\data\processed\processed_buy_data.csv', index_col=0)
    
    # Cast price to float and transform for visualization
    df['price'] = df['price'].astype(float)
    
    # Visualize price distribution
    plot_distributions(df, 'price', 'Price Distribution')
    
    # Visualize price per sqft distribution
    plot_distributions(df, 'price_per_sqft', 'Price per Sqft Distribution')
    
    # Visualize boxplot of price
    plot_boxplot(df, 'price', 'Price Boxplot')
    
    # Visualize boxplot of price_per_sqft
    plot_boxplot(df, 'price_per_sqft', 'Price per Sqft Boxplot')
