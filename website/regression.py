import streamlit as st
from streamlit.logger import get_logger
from streamlit_lottie import st_lottie
import requests
import pandas as pd
import ast

import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import intro
import genre
import plot
import conclusion as conc
import format_text as texts
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
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
    """
    Enhance the features of the training and test datasets by adding polynomial features 
    for numerical variables and creating a genre count feature.
    Parameters:
    X_train (pd.DataFrame): The training dataset containing the features.
    X_test (pd.DataFrame): The test dataset containing the features.
    Returns:
    pd.DataFrame: The enhanced training dataset with additional features.
    pd.DataFrame: The enhanced test dataset with additional features.
    """
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

def enhance_features_inflated(X_train, X_test):
    """
    Enhance features by adding polynomial features and genre count.
    Parameters:
    X_train (pd.DataFrame): Training data containing features.
    X_test (pd.DataFrame): Test data containing features.
    Returns:
    pd.DataFrame: Enhanced training data with additional features.
    pd.DataFrame: Enhanced test data with additional features.
    The function performs the following steps:
    1. Creates polynomial features for numerical variables: 'movie_release_date', 'adjusted_budget', and 'movie_runtime'.
    2. Adds the polynomial features back to the original dataframes.
    3. Creates a genre count feature by summing up the genre columns that start with 'movie_genres_'.
    """
    # Extract numerical columns
    numerical_cols = ['movie_release_date', 'adjusted_budget', 'movie_runtime']
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
    """
    Perform text vectorization using TF-IDF and cluster the resulting vectors using KMeans.

    Parameters:
    col (iterable): An iterable containing text data to be vectorized and clustered.

    Returns:
    numpy.ndarray: An array of cluster labels assigned to each input text.
    """
    # Text Vectorization with TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(col)

    combined_matrix = tfidf_matrix.toarray()

    # Clustering with KMeans
    n_clusters = 15
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    return kmeans.fit_predict(combined_matrix)

def list_to_1_hot(df, column_name):
    """
    Converts a column of lists in a DataFrame to one-hot encoded columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the column to be one-hot encoded.
    column_name (str): The name of the column in the DataFrame that contains lists to be one-hot encoded.

    Returns:
    pd.DataFrame: The DataFrame with the specified column replaced by one-hot encoded columns.
    """
    mlb = MultiLabelBinarizer()
    one_hot_df = pd.DataFrame(mlb.fit_transform(df[column_name]), columns=mlb.classes_, index=df.index)
    df = pd.concat([df, one_hot_df], axis=1)
    df = df.drop(column_name, axis=1)
    return df

def split_x_y(df, y_column, x_columns_to_drop):
    """
    Splits a DataFrame into features (X) and target (Y) based on specified columns.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    y_column (str): The name of the column to be used as the target variable (Y).
    x_columns_to_drop (list of str): A list of column names to be dropped from the DataFrame to form the features (X).

    Returns:
    tuple: A tuple containing two elements:
        - x (pandas.DataFrame): The DataFrame containing the features.
        - y (pandas.Series): The Series containing the target variable.
    """
    y = df[y_column]
    x = df.drop(columns=x_columns_to_drop)
    return x, y

def split_train_test(x, y, test_size=0.2):
    """
    Splits the input data into training and testing sets.

    Parameters:
    x (array-like): Features dataset.
    y (array-like): Target dataset.
    test_size (float, optional): Proportion of the dataset to include in the test split. Default is 0.2.

    Returns:
    tuple: A tuple containing four elements:
        - x_train (array-like): Training features.
        - x_test (array-like): Testing features.
        - y_train (array-like): Training targets.
        - y_test (array-like): Testing targets.
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    return x_train, x_test, y_train, y_test

def scale_data(x_train, x_test):
    """
    Scales the training and testing data using StandardScaler.

    Parameters:
    x_train (pd.DataFrame): The training data to be scaled.
    x_test (pd.DataFrame): The testing data to be scaled.

    Returns:
    tuple: A tuple containing two pandas DataFrames:
        - x_train_df (pd.DataFrame): The scaled training data.
        - x_test_df (pd.DataFrame): The scaled testing data.
    """
    scaler = sklearn.preprocessing.StandardScaler()
    train = scaler.fit_transform(x_train)
    test = scaler.transform(x_test)
    x_train_df = pd.DataFrame(train, columns=x_train.columns)
    x_test_df = pd.DataFrame(test, columns=x_test.columns)
    return x_train_df, x_test_df


def preprocess4linreg(df, y_column, x_columns_to_drop, test_size=0.2):
    """
    Preprocesses the input DataFrame for linear regression by performing the following steps:
    1. Converts categorical variables 'plot_structure' and 'plot_structure_cluster' to dummy variables if they are not in x_columns_to_drop.
    2. Converts 'movie_genres' and 'movie_countries' columns to one-hot encoded format.
    3. Splits the DataFrame into features (X) and target (y) based on y_column and x_columns_to_drop.
    4. Splits the data into training and testing sets.
    5. Scales the training and testing feature sets.
    6. Adds a constant term to the feature sets for regression.
    7. Fills any missing values in the feature sets with the mean of the training set.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    y_column (str): The name of the target column.
    x_columns_to_drop (list): List of column names to drop from the features.
    test_size (float): The proportion of the dataset to include in the test split.
    Returns:
    tuple: A tuple containing the following elements:
        - X_train_scaled_df (pd.DataFrame): The preprocessed and scaled training feature set with a constant term.
        - X_test_scaled_df (pd.DataFrame): The preprocessed and scaled testing feature set with a constant term.
        - y_train_no_index (pd.Series): The training target set with reset index.
        - y_test_no_index (pd.Series): The testing target set with reset index.
    """
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

import streamlit as st
import plotly.graph_objects as go
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt  # For using the Viridis color map

import streamlit as st
import plotly.graph_objects as go
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt  # For using the Viridis color map

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
import statsmodels.api as sm

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
import statsmodels.api as sm

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
import statsmodels.api as sm

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
import statsmodels.api as sm

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
import statsmodels.api as sm

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
import statsmodels.api as sm

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
import statsmodels.api as sm

import plotly.graph_objects as go
import pandas as pd
import statsmodels.api as sm
import numpy as np
import streamlit as st

def plot_reg_coeffs(movies_plot):
    # Preprocess data for regression
    movies_plot['plot_structure_cluster'] = add_plot_structure_cluster(movies_plot['plot_summary'])
    X_train_plot, X_test_plot, y_train_plot, y_test_plot = preprocess4linreg(
        movies_plot,
        'adjusted_profit',
        [
            'wikipedia_movie_id', 'freebase_movie_id', 'movie_name', 'movie_box_office_revenue',
            'movie_languages', 'title_year', 'plot_summary', 'opening_weekend', 'year_interval',
            'summarized', 'plot_structure_20', 'producer', 'adjusted_revenue', 'adjusted_profit',
            'budget', 'profit', 'profitability_ratio'
        ]
    )

    # Fit the model
    model = sm.OLS(y_train_plot, X_train_plot)
    results = model.fit()

    # Extract significant coefficients and their confidence intervals
    significant_features = results.pvalues[results.pvalues < 0.05].index
    significant_coefficients = results.params[significant_features]
    conf = results.conf_int().loc[significant_features]
    conf['coef'] = significant_coefficients

    # Sort coefficients by absolute value, but keep track of signs as well
    sorted_conf = conf.assign(
        abs_coef=np.abs(conf['coef'])
    ).sort_values(by='abs_coef', ascending=False)

    # Compute the maximum absolute coefficient across the entire dataset for consistent color scale
    max_color = np.max(np.abs(sorted_conf['coef']))

    # Separate the coefficients into two halves: largest positive and the rest
    half_index = len(sorted_conf) // 2
    positive_half = sorted_conf[sorted_conf['coef'] > 0].head(half_index)  # Top half: largest positive coefficients
    negative_half = sorted_conf[sorted_conf['coef'] <= 0].tail(len(sorted_conf) - half_index)  # Bottom half: others

    # Create an interactive Plotly chart
    fig = go.Figure()

    # Dropdown menu to toggle between top and bottom half
    half_choice = st.selectbox(
        'Select Half of the Plot to Display:',
        ['Top Half (Biggest Positive Norms)', 'Bottom Half (Negative Coefficients)']
    )

    if half_choice == 'Top Half (Biggest Positive Norms)':
        features_to_show = positive_half
    else:
        features_to_show = negative_half

    # Normalize the colors based on the global max absolute coefficient (used for both halves)
    colors = np.abs(features_to_show['coef'])  # Use absolute values for color scaling
    color_scale = np.array([plt.cm.viridis(c / max_color) for c in colors])  # Normalize using the global max

    # Add bar chart for the selected half of the coefficients
    fig.add_trace(go.Bar(
        x=features_to_show['coef'],
        y=features_to_show.index,
        orientation='h',
        name='Coefficients',
        marker=dict(color=colors, colorscale='Viridis', showscale=True),
        width=0.7,  # Reduce bar width to avoid overlap
    ))

    # Add error bars for confidence intervals (black color)
    for i, feature in enumerate(features_to_show.index):
        fig.add_trace(go.Scatter(
            x=[features_to_show.iloc[i, 0], features_to_show.iloc[i, 1]],
            y=[feature, feature],
            mode='lines',
            line=dict(color='black', width=2),
            name=f"Confidence Interval ({feature})",
            showlegend=False
        ))

    # Dynamically adjust height based on the number of coefficients in the top half
    # This height will be used for both halves to maintain a consistent layout
    height = max(600, len(positive_half) * 40)  # Use the height from the top half

    fig.update_layout(
        title={
            'text': "Significant Coefficients with Confidence Intervals",
            'x': 0.5,  # Center the title
            'xanchor': 'center',  # Ensure the title is centered
            'font': {'size': 20}  # Increase the font size
        },
        xaxis_title="Coefficient Value",
        yaxis_title="Features",
        yaxis=dict(categoryorder='total ascending'),
        template='plotly_white',
        height=height,  # Set fixed height
        margin=dict(l=200, r=50, t=50, b=50),  # Ensure labels don't get cut off
        barmode='group',  # Ensure bars do not overlap
        autosize=True,  # Make the plot responsive
    )

    # Display interactive plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)


import streamlit as st
import plotly.graph_objects as go
import statsmodels.api as sm
import numpy as np


import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import statsmodels.api as sm
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import statsmodels.api as sm
import streamlit as st

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import statsmodels.api as sm

def plot_budget_profit(movies_plot):

    # Tabs for switching between plots
    tab1, tab2 = st.tabs(["Budget vs Profit", "Budget vs Profit (Log-Log Scale)"])

    with tab1:
        st.write("### Budget vs Profit")

        # Interactive scatter plot
        fig1 = px.scatter(
            movies_plot,
            x='adjusted_budget',
            y='adjusted_profit',
            labels={'adjusted_budget': 'Budget in $', 'adjusted_profit': 'Profit in $'},
            title='Budget vs Profit',
        )
        fig1.update_traces(marker=dict(size=5))

        # Make title slightly bigger and centered
        fig1.update_layout(
            title=dict(
                text='Budget vs Profit',
                x=0.5,  # Center title
                font=dict(size=20),  # Increase title font size
            ),
            template='plotly_white'
        )
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        st.write("### Budget vs Profit (Log-Log Scale)")

        # Log-transform the data
        log_budget = np.log(movies_plot['adjusted_budget'])
        log_profit = np.log(movies_plot['adjusted_profit'])

        # Prepare the data for regression (including a constant for the intercept)
        X = sm.add_constant(log_budget)  # Add constant for intercept
        y = log_profit

        # Perform linear regression
        model = sm.OLS(y, X).fit()

        # Get regression line data (predicted y values)
        y_pred = model.predict(X)

        # Create the scatter plot in log-log scale
        fig2 = px.scatter(
            movies_plot,
            x='adjusted_budget',
            y='adjusted_profit',
            log_x=True,
            log_y=True,
            labels={'adjusted_budget': 'Budget (Log Scale)', 'adjusted_profit': 'Profit (Log Scale)'},
            title='Budget vs Profit (Log-Log Scale)',
        )
        fig2.update_traces(marker=dict(size=5))

        # Add the regression line in red (log-log regression line)
        # We use np.exp to convert back to the original scale
        fig2.add_trace(go.Scatter(
            x=movies_plot['adjusted_budget'],
            y=np.exp(y_pred),  # Convert predicted values back to original scale
            mode='lines',
            line=dict(color='red', width=2),
            name='Regression Line',
        ))

        # Make title slightly bigger and centered
        fig2.update_layout(
            title=dict(
                text='Budget vs Profit (Log-Log Scale)',
                x=0.5,  # Center title
                font=dict(size=20),  # Increase title font size
            ),
            template='plotly_white'
        )
        st.plotly_chart(fig2, use_container_width=True)


import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def ROI_plot(movies_plot):

    # Tabs for switching between plots
    tab1, tab2 = st.tabs(["ROI vs Budget (Log-Log Scale)", "ROI by Budget Range"])

    with tab1:
        st.write("### ROI vs Budget (Log-Log Scale with Regression Line)")

        # Calculate profitability ratio
        movies_plot['profitability_ratio'] = movies_plot['adjusted_profit'] / movies_plot['adjusted_budget']

        # Log transformation
        log_budget = np.log(movies_plot['adjusted_budget'])
        log_roi = np.log(movies_plot['profitability_ratio'])

        # Prepare the data for regression (including a constant for the intercept)
        X = sm.add_constant(log_budget)  # Add constant for intercept
        y = log_roi

        # Perform linear regression
        model = sm.OLS(y, X).fit()

        # Get regression line data (predicted y values)
        y_pred = model.predict(X)

        # Create the scatter plot in log-log scale using Plotly
        fig1 = px.scatter(
            movies_plot,
            x='adjusted_budget',
            y='profitability_ratio',
            log_x=True,
            log_y=True,
            labels={'adjusted_budget': 'Budget (Log Scale)', 'profitability_ratio': 'ROI (Log Scale)'},
            title='ROI vs Budget (Log-Log Scale)',
        )
        fig1.update_traces(marker=dict(size=5))

        # Add the regression line in red (log-log regression line)
        fig1.add_trace(go.Scatter(
            x=movies_plot['adjusted_budget'],
            y=np.exp(y_pred),  # Convert predicted values back to original scale
            mode='lines',
            line=dict(color='red', width=2),
            name='Regression Line',
        ))

        fig1.update_layout(template='plotly_white')
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        st.write("### ROI by Budget Range")

        # Create budget bins
        bins = [0, 1e6, 1e7, 5e7, 1e8, 5e8]
        labels = ['<1M', '1M-10M', '10M-50M', '50M-100M', '100M+']
        movies_plot['budget_bins'] = pd.cut(movies_plot['adjusted_budget'], bins=bins, labels=labels)

        # Group ROI by budget bins and calculate mean and median
        roi_by_budget_bin = movies_plot.groupby('budget_bins')['profitability_ratio'].agg(['mean', 'median', 'std']).reset_index()

        # Create an interactive bar plot for mean and median ROI by budget range using Plotly
        fig2 = go.Figure()

        # Add bars for mean ROI
        fig2.add_trace(go.Bar(
            x=roi_by_budget_bin['budget_bins'],
            y=roi_by_budget_bin['mean'],
            name='Mean ROI',
            marker=dict(color='skyblue'),
        ))

        # Add bars for median ROI
        fig2.add_trace(go.Bar(
            x=roi_by_budget_bin['budget_bins'],
            y=roi_by_budget_bin['median'],
            name='Median ROI',
            marker=dict(color='orange'),
        ))

        # Update layout for better appearance
        fig2.update_layout(
            title='Mean and Median ROI by Budget Range',
            xaxis_title='Budget Range',
            yaxis_title='ROI',
            barmode='group',
            template='plotly_white'
        )

        # Display interactive Plotly chart
        st.plotly_chart(fig2, use_container_width=True)
