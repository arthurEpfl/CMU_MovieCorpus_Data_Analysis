import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import random
import sklearn
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import PolynomialFeatures

def enhance_features(X_train, X_test):
    # Extract numerical columns
    numerical_cols = ['movie_release_date', 'budget', 'movie_runtime']
    # 1. Create polynomial features for numerical variables
    poly = PolynomialFeatures(degree=2, include_bias=False)
    numerical_train = poly.fit_transform(X_train[numerical_cols])
    numerical_test = poly.transform(X_test[numerical_cols])
    
    # Add polynomial features back to the dataframe
    poly_features = poly.get_feature_names_out(numerical_cols)
    X_train_poly = pd.DataFrame(numerical_train, columns=poly_features, index=X_train.index)
    X_test_poly = pd.DataFrame(numerical_test, columns=poly_features, index=X_test.index)
    X_train = pd.concat([X_train, X_train_poly], axis=1)
    X_test = pd.concat([X_test, X_test_poly], axis=1)
    
    # 3. Create genre count feature
    genre_cols = [col for col in X_train.columns if col.startswith('movie_genres_')]
    X_train['genre_count'] = X_train[genre_cols].sum(axis=1)
    X_test['genre_count'] = X_test[genre_cols].sum(axis=1)
    
    
    return X_train, X_test


def add_plot_structure_cluster(col):
    # Text Vectorization with TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(col)

    combined_matrix = tfidf_matrix.toarray()

    # Clustering with KMeans
    n_clusters = 15
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    return kmeans.fit_predict(combined_matrix)

def list_to_1_hot(df, column_name):
    mlb = MultiLabelBinarizer()
    one_hot_df = pd.DataFrame(mlb.fit_transform(df[column_name]), columns=mlb.classes_, index=df.index)
    df = pd.concat([df, one_hot_df], axis=1)
    df = df.drop(column_name, axis=1)
    return df

def split_x_y(df, y_column, x_columns_to_drop):
    y = df[y_column]
    x = df.drop(columns=x_columns_to_drop)
    return x, y

def split_train_test(x, y, test_size=0.2):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    return x_train, x_test, y_train, y_test

def scale_data(x_train, x_test):
    scaler = sklearn.preprocessing.StandardScaler()
    train = scaler.fit_transform(x_train)
    test = scaler.transform(x_test)
    x_train_df = pd.DataFrame(train, columns=x_train.columns)
    x_test_df = pd.DataFrame(test, columns=x_test.columns)
    return x_train_df, x_test_df

def preprocess4linreg(df, y_column, x_columns_to_drop, test_size=0.2):
    if 'plot_structure' not in x_columns_to_drop :
        df = pd.get_dummies(df, columns=['plot_structure'], drop_first=True, dtype=int)
    
    if 'movie_genres' not in x_columns_to_drop :
        df = list_to_1_hot(df, 'movie_genres')
    if 'plot_structure_cluster' not in x_columns_to_drop :
        df = pd.get_dummies(df, columns=['plot_structure_cluster'], drop_first=True, dtype=int)
    df = list_to_1_hot(df, 'movie_countries')
    x, y = split_x_y(df, y_column, x_columns_to_drop)
    x_train, x_test, y_train, y_test = split_train_test(x, y, test_size)
    x_train, x_test = scale_data(x_train, x_test)
    
    X_train_scaled_df = sm.add_constant(x_train,has_constant='add')
    X_test_scaled_df = sm.add_constant(x_test,has_constant='add')

    X_train_scaled_df = X_train_scaled_df.fillna(X_train_scaled_df.mean())
    X_test_scaled_df = X_test_scaled_df.fillna(X_train_scaled_df.mean())

    y_train_no_index = y_train.reset_index(drop=True)
    y_test_no_index = y_test.reset_index(drop=True)
    X_train_scaled_df = X_train_scaled_df.reset_index(drop=True)
    X_test_scaled_df = X_test_scaled_df.reset_index(drop=True)
    
    return X_train_scaled_df, X_test_scaled_df, y_train_no_index, y_test_no_index