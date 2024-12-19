import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def load_processed_inflation():
    df_inflation = pd.read_csv('data/raw/inflation_data.csv')
    df_inflation['date'] = pd.to_datetime(df_inflation['date'])
    df_inflation = df_inflation.set_index('date')
    df_inflation = df_inflation.reset_index()
    df_inflation['year'] = df_inflation['date'].dt.year
    df_inflation = df_inflation.groupby('year')['value'].mean()
    base_year = 2023
    cpi_base = df_inflation[base_year]
    df_inflation = cpi_base / df_inflation
    return df_inflation

def load_processed_movies_imdb():
    movies_imdb = pd.read_csv("data/processed/movies_summary_BO.csv")
    return movies_imdb

def process_movies_imdb_inflation(movies_imdb, df_inflation):
    movie_years = movies_imdb['movie_release_date'].dropna().unique()
    cpi_years = df_inflation.index.unique()
    missing_years = set(movie_years) - set(cpi_years)
    all_years = pd.RangeIndex(start=min(df_inflation.index.min(), min(movie_years)),
                          stop=max(df_inflation.index.max(), max(movie_years)) + 1)
    df_inflation = df_inflation.reindex(all_years).ffill().bfill()
    movies_imdb = movies_imdb.dropna(subset=['movie_box_office_revenue', 'movie_release_date'])

    movies_imdb.loc[:, 'adjusted_revenue'] = movies_imdb.apply(
        lambda x: x['movie_box_office_revenue'] * df_inflation.get(x['movie_release_date'], 1), axis=1
    )
    movies_by_year = movies_imdb.groupby('movie_release_date').agg({
    'movie_box_office_revenue': ['mean', 'sum', 'max'],
    'adjusted_revenue': ['mean', 'sum', 'max']
        }).reset_index()

    movies_by_year.columns = ['Year', 'Original Mean Revenue', 'Original Sum Revenue', 'Original Max Revenue',
                          'Adjusted Mean Revenue', 'Adjusted Sum Revenue', 'Adjusted Max Revenue']
    return movies_by_year


def return_processed_genre_df(movies, genre_year_range):
    genre_filtered = movies[(movies['movie_release_date'] >= genre_year_range[0]) & (movies['movie_release_date'] <= genre_year_range[1])]
    genre_exploded = genre_filtered.explode('movie_genres')
    genre_counts = genre_exploded['movie_genres'].value_counts()
    genre_percentages = (genre_counts / genre_counts.sum()) * 100
    top_genre = genre_percentages.head(15)
    filtered_df = genre_exploded[genre_exploded['movie_genres'].isin(top_genre.index)]
    mean_revenues = genre_exploded.groupby('movie_genres')['movie_box_office_revenue'].mean().reset_index()
    mean_revenues = mean_revenues.sort_values(by='movie_box_office_revenue', ascending=False)
    return genre_exploded, mean_revenues

def process_inflation_data(movies, df_inflation, year_range=None, selected_genres=None):
    # Filter by year range if provided
    if year_range:
        movies = movies[(movies['movie_release_date'] >= year_range[0]) & 
                       (movies['movie_release_date'] <= year_range[1])]
    
    if selected_genres:
        movies = movies[movies['movie_genres'].apply(lambda x: any(genre in x for genre in selected_genres))]
    movies['adjusted_revenue'] = movies.apply(
        lambda x: x['movie_box_office_revenue'] * df_inflation.get(x['movie_release_date'], 1), axis=1
    )
    
    revenue_by_year = movies.groupby('movie_release_date').agg({
        'movie_box_office_revenue': ['mean', 'sum', 'max'],
        'adjusted_revenue': ['mean', 'sum', 'max']
    }).reset_index()
    revenue_by_year.columns = ['Year', 'Original Mean', 'Original Sum', 'Original Max',
                              'Adjusted Mean', 'Adjusted Sum', 'Adjusted Max']
    return revenue_by_year

def process_inflation_data_genre(movies, df_inflation, year_range=None, selected_genres=None):
    # Filter by year range if provided
    if year_range:
        movies = movies[(movies['movie_release_date'] >= year_range[0]) & 
                       (movies['movie_release_date'] <= year_range[1])]
    
    if selected_genres:
        movies = movies[movies['movie_genres'].apply(lambda x: any(genre in x for genre in selected_genres))]
    
    movies['adjusted_revenue'] = movies.apply(
        lambda x: x['movie_box_office_revenue'] * df_inflation.get(x['movie_release_date'], 1), axis=1
    )
    movies_exploded = movies.explode('movie_genres')
    revenue_by_year_genre = movies_exploded.groupby(['movie_release_date', 'movie_genres']).agg({
        'movie_box_office_revenue': ['mean', 'sum', 'max'],
        'adjusted_revenue': ['mean', 'sum', 'max']
    }).reset_index()
    revenue_by_year_genre.rename(columns={'movie_genres': 'Genre'}, inplace=True)
    revenue_by_year_genre.columns = ['Year', 'Genre', 'Original Mean', 'Original Sum', 'Original Max',
                                    'Adjusted Mean', 'Adjusted Sum', 'Adjusted Max']
    return revenue_by_year_genre

def calculate_profit_metrics(movies_df, df_inflation):
    """Calculate profit and profitability metrics with inflation adjustment"""
    df = movies_df.copy()
    df['adjusted_revenue'] = df.apply(
        lambda x: x['movie_box_office_revenue'] * df_inflation.get(x['movie_release_date'], 1), 
        axis=1
    )
    df['adjusted_budget'] = df.apply(
        lambda x: x['budget'] * df_inflation.get(x['movie_release_date'], 1) if pd.notna(x['budget']) else np.nan,
        axis=1
    )
    
    df['profit'] = df['movie_box_office_revenue'] - df['budget']
    df['adjusted_profit'] = df['adjusted_revenue'] - df['adjusted_budget']
    df['profitability_ratio'] = df['profit'] / df['budget']
    df['adjusted_profitability_ratio'] = df['adjusted_profit'] / df['adjusted_budget']
    
    return df

def get_top_profitable_movies(movies_df, n=10, by='profit'):
    """Get top n profitable movies by profit or profitability ratio"""
    if by not in ['profit', 'profitability_ratio']:
        raise ValueError("'by' must be either 'profit' or 'profitability_ratio'")
    return movies_df.nlargest(n, by)[['movie_name', by]]

def create_budget_bins(movies_df):
    """Create budget bins and calculate ROI statistics"""
    df = movies_df.copy()
    bins = [0, 1e6, 1e7, 5e7, 1e8, 5e8]
    labels = ['<1M', '1M-10M', '10M-50M', '50M-100M', '100M+']
    df['budget_bins'] = pd.cut(df['budget'], bins=bins, labels=labels)
    
    roi_by_budget = df.groupby('budget_bins')['profitability_ratio'].agg([
        'mean', 'median', 'std'
    ]).reset_index()
    
    return df, roi_by_budget, labels

def analyze_commercial_success(movies_df, df_inflation):
    """Analyze commercial success metrics with inflation adjustment"""
    df = movies_df.copy()
    
    # Calculate basic profit metrics
    df['profit'] = df['movie_box_office_revenue'] - df['budget']
    df['adjusted_revenue'] = df.apply(
        lambda x: x['movie_box_office_revenue'] * df_inflation.get(x['movie_release_date'], 1), 
        axis=1
    )
    df['adjusted_budget'] = df.apply(
        lambda x: x['budget'] * df_inflation.get(x['movie_release_date'], 1) 
        if pd.notna(x['budget']) else np.nan, 
        axis=1
    )
    df['adjusted_profit'] = df['adjusted_revenue'] - df['adjusted_budget']
    df['profitability_ratio'] = df['profit'] / df['budget']
    df['adjusted_profitability_ratio'] = df['adjusted_profit'] / df['adjusted_budget']
    
    # Create budget bins
    bins = [0, 1e6, 1e7, 5e7, 1e8, 5e8]
    labels = ['<1M', '1M-10M', '10M-50M', '50M-100M', '100M+']
    df['budget_bins'] = pd.cut(df['budget'], bins=bins, labels=labels)
    
    return df

def get_roi_statistics(movies_df):
    """Calculate ROI statistics by budget bin"""
    roi_stats = movies_df.groupby('budget_bins').agg({
        'profitability_ratio': ['mean', 'median', 'std', 'count'],
        'adjusted_profitability_ratio': ['mean', 'median', 'std', 'count']
    }).round(2)
    
    return roi_stats

def process_plot_structure_data(movies_df, df_inflation):
    """Process data for plot structure analysis"""
    df = movies_df.copy()
    
    # Calculate adjusted values
    df['adjusted_revenue'] = df.apply(
        lambda x: x['movie_box_office_revenue'] * df_inflation.get(x['movie_release_date'], 1), 
        axis=1
    )
    df['adjusted_budget'] = df.apply(
        lambda x: x['budget'] * df_inflation.get(x['movie_release_date'], 1) 
        if pd.notna(x['budget']) else np.nan,
        axis=1
    )
    df['adjusted_profit'] = df['adjusted_revenue'] - df['adjusted_budget']
    return df

def analyze_plot_structure_metrics(df):
    """Calculate various metrics for plot structure analysis"""
    # Revenue metrics by plot structure
    revenue_metrics = df.groupby('plot_summary').agg({
        'adjusted_revenue': ['mean', 'median', 'count'],
        'adjusted_profit': ['mean', 'median'],
        'adjusted_budget': ['mean', 'median']
    }).round(2)
    
    # Success rate (percentage of profitable movies)
    success_rate = df.groupby('plot_structure').apply(
        lambda x: (x['adjusted_profit'] > 0).mean() * 100
    ).round(2)
    
    # Time trends
    yearly_trends = df.groupby(['movie_release_date', 'plot_structure']).size().unstack(fill_value=0)
    
    return revenue_metrics, success_rate, yearly_trends

def get_plot_structure_correlations(df):
    """Analyze correlations between plot structures and other features"""
    # Create dummy variables for plot structures
    plot_dummies = pd.get_dummies(df['plot_structure'])
    
    # Calculate correlations with numerical features
    correlations = pd.DataFrame()
    for col in ['adjusted_revenue', 'adjusted_profit', 'adjusted_budget', 'rating_score']:
        if col in df.columns:
            correlations[col] = plot_dummies.corrwith(df[col])
    
    return correlations

def analyze_genre_plot_combinations(df):
    """Analyze genre and plot structure combinations"""
    # Explode genres if they're in a list
    df_exploded = df.explode('movie_genres')
    
    # Create cross-tabulation
    genre_plot_matrix = pd.crosstab(
        df_exploded['movie_genres'],
        df_exploded['plot_structure']
    )
    
    # Calculate success metrics for each combination
    genre_plot_success = df_exploded.groupby(
        ['movie_genres', 'plot_structure']
    )['adjusted_profit'].agg(['mean', 'median', 'count']).round(2)
    
    return genre_plot_matrix, genre_plot_success

def perform_text_clustering(plot_summaries, n_clusters=3):
    """Perform clustering on movie plot summaries"""
    # Clean and prepare text
    plot_summaries = plot_summaries.fillna('')
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        min_df=2
    )
    
    tfidf_matrix = vectorizer.fit_transform(plot_summaries)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)
    
    # Get top terms per cluster
    top_terms = []
    feature_names = vectorizer.get_feature_names_out()
    
    for i in range(n_clusters):
        center = kmeans.cluster_centers_[i]
        top_indices = center.argsort()[-10:][::-1]  # Get indices of top 10 terms
        top_terms.append([feature_names[idx] for idx in top_indices])
    
    return {
        'matrix': tfidf_matrix,
        'labels': cluster_labels,
        'top_terms': top_terms
    }

def calculate_silhouette_scores(tfidf_matrix, max_clusters=20):
    """Calculate silhouette scores for different numbers of clusters"""
    scores = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(tfidf_matrix)
        score = silhouette_score(tfidf_matrix, labels)
        scores.append(score)
    return scores

def analyze_plot_structure_distribution(df):
    """Analyze the distribution of plot structures"""
    # Handle potential missing values
    df['plot_structure'] = df['plot_structure'].str.split(':').str[0]
    df['plot_structure_20'] = df['plot_structure_20'].str.split(':').str[0]
    df['plot_structure'] = df['plot_structure'].fillna('Unclassified')
    
    # Get the counts
    plot_counts = df['plot_structure'].value_counts()
    
    # Convert to percentage
    total = len(df)
    plot_percentages = (plot_counts / total * 100).round(1)
    df['plot_structure_20'] = df['plot_structure_20'].fillna('Unclassified')
    plot_counts_20 = df['plot_structure_20'].value_counts()
    plot_percentages_20 = (plot_counts_20 / total * 100).round(1)
    return {
        'structures': plot_counts.index.tolist(),
        'plot_counts': plot_counts.values.tolist(),
        'percentages': plot_percentages.values.tolist(),
        'structures_20': plot_counts_20.index.tolist(),
        'plot_counts_20': plot_counts_20.values.tolist(),
        'percentages_20': plot_percentages_20.values.tolist()
    }

def analyze_plot_structure_performance(df):
    """Analyze performance metrics for different plot structures"""
    # Ensure numeric columns are properly formatted
    df['movie_box_office_revenue'] = pd.to_numeric(df['movie_box_office_revenue'], errors='coerce')
    df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
    df['rating_score'] = pd.to_numeric(df['rating_score'], errors='coerce')
    
    # Calculate profit
    df['profit'] = df['movie_box_office_revenue'] - df['budget']
    
    # Group by plot structure and calculate metrics
    metrics = df.groupby('plot_structure').agg({
        'movie_box_office_revenue': ['mean', 'count'],
        'budget': ['mean'],
        'profit': ['mean'],
        'rating_score': ['mean']
    }).round(2)
    
    # Filter out groups with too few movies (optional)
    metrics = metrics[metrics[('movie_box_office_revenue', 'count')] >= 5]
    
    # Sort by average revenue
    metrics = metrics.sort_values(('movie_box_office_revenue', 'mean'), ascending=False)
    
    return metrics

def analyze_plot_structure_profit(df):
    """Analyze profit metrics for different plot structures"""
    # Ensure numeric columns are properly formatted
    df['movie_box_office_revenue'] = pd.to_numeric(df['movie_box_office_revenue'], errors='coerce')
    df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
    
    # Calculate profit
    df['profit'] = df['movie_box_office_revenue'] - df['budget']
    
    # Group by plot structure and calculate metrics
    metrics = df.groupby('plot_structure').agg({
        'profit': ['median', 'mean', 'count'],
        'rating_score': ['mean']
    }).round(2)
    
    # Filter out groups with too few movies
    metrics = metrics[metrics[('profit', 'count')] >= 5]
    
    # Sort by median profit
    metrics = metrics.sort_values(('profit', 'median'), ascending=False)
    
    return metrics

def analyze_genre_profit(df):
    """Analyze profit metrics for different genres"""
    # Ensure numeric columns are properly formatted
    df['movie_box_office_revenue'] = pd.to_numeric(df['movie_box_office_revenue'], errors='coerce')
    df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
    
    df['profit'] = df['movie_box_office_revenue'] - df['budget']
    
    df['movie_genres'] = df['movie_genres'].apply(ast.literal_eval)
    df_exploded = df.explode('movie_genres')
    
    metrics = df_exploded.groupby('movie_genres').agg({
        'profit': ['median', 'mean', 'count']
    }).round(2)
    
    metrics = metrics[metrics[('profit', 'count')] >= 10]
    metrics = metrics.sort_values(('profit', 'median'), ascending=False)
    
    return metrics

