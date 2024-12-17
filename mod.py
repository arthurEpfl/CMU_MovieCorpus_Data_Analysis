import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os  


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

