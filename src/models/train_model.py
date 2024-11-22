# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer, r2_score
from src.features.build_features import clean_data, preprocess_features
# build_feature import 
import pickle

def train_model(X, y):
    """
    Train an ExtraTreesRegressor model with GridSearch for hyperparameter optimization.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['bedrooms', 'bathrooms', 'floor_area']),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ['area']),
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', ExtraTreesRegressor(bootstrap=True))
    ])

    param_grid = {
        'regressor__n_estimators': [50, 100, 200, 300],
        'regressor__max_depth': [None, 10, 20, 30],
        'regressor__max_samples': [0.1, 0.25, 0.5, 1.0],
        'regressor__max_features': ['auto', 'sqrt']
    }

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid, scoring=make_scorer(r2_score), cv=kfold, n_jobs=-1)

    # Perform grid search
    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

if __name__ == "__main__":
    # Load and preprocess data
    df = pd.read_csv(r'D:\Work file\bd_real_estate\data\processed\processed_buy_data.csv', index_col=0)
    df = clean_data(df)

    # Split features and target
    X = df.drop(columns=['price', 'property_name', 'address', 'short_description', 'property_url', 'type', 'teg', 'sub_area'])
    y = np.log1p(df['price'])  # Apply log transformation to target variable

    # Train model
    best_model, best_params, best_score = train_model(X, y)
    print(f"Best Score: {best_score}")
    print(f"Best Parameters: {best_params}")

    # Save the model
    with open('final_model.pkl', 'wb') as file:
        pickle.dump(best_model, file)

    print("Model saved to final_model.pkl")
