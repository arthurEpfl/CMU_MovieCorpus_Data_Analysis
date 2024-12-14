import pandas as pd
import plotly.express as px
import ast

def load_data(file_path='data/processed/movies_summary_BO.csv'):
    return pd.read_csv(file_path)

def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val

def plot_movie_release_years(movies):
    movies = movies[movies['movie_release_date'] >= 1920]
    revenue_counts_by_year = movies.groupby('movie_release_date').size().reset_index(name='count')
    fig = px.bar(revenue_counts_by_year, x='movie_release_date', y='count', 
                 title='Count of Movies with Available Box Office Revenue per Year',
                 labels={'movie_release_date': 'Year', 'count': 'Count of Movies'},
                 color_discrete_sequence=['skyblue'])
    fig.update_layout(xaxis_title='Year', yaxis_title='Count of Movies', 
                      xaxis_tickangle=-45, xaxis_tickmode='array', 
                      xaxis_tickvals=revenue_counts_by_year['movie_release_date'][::5])
    return fig

def plot_box_office_revenue_by_year(movies):
    movies = movies[movies['movie_release_date'] >= 1920]
    revenue_by_year = movies.groupby('movie_release_date')['movie_box_office_revenue'].sum().reset_index()
    fig = px.line(revenue_by_year, x='movie_release_date', y='movie_box_office_revenue', 
                  title='Total Box Office Revenue by Year',
                  labels={'movie_release_date': 'Year', 'movie_box_office_revenue': 'Total Revenue'},
                  markers=True)
    fig.update_layout(xaxis_title='Year', yaxis_title='Total Revenue', 
                      xaxis_tickangle=-45, xaxis_tickmode='array', 
                      xaxis_tickvals=revenue_by_year['movie_release_date'][::5])
    return fig

def plot_language_distribution(movies):
    movies = movies[movies['movie_release_date'] >= 1920]
    movies['movie_languages'] = movies['movie_languages'].apply(safe_literal_eval)
    movies['year_interval'] = (movies['movie_release_date'] // 5) * 5  
    movies_languages_exploded = movies.explode('movie_languages')
    language_year_pivot = movies_languages_exploded.pivot_table(index='year_interval', columns='movie_languages', aggfunc='size', fill_value=0)
    language_counts = movies_languages_exploded['movie_languages'].value_counts()
    valid_languages = language_counts[language_counts >= 50].index
    language_year_pivot = language_year_pivot[valid_languages]  
    language_year_pivot = language_year_pivot.reset_index().melt(id_vars='year_interval', var_name='Language', value_name='Count')
    fig = px.bar(language_year_pivot, x='year_interval', y='Count', color='Language', 
                 title='Distribution of Top 30 Movie Languages Across 5-Year Intervals',
                 labels={'year_interval': 'Year Intervals', 'Count': 'Number of Movies'},
                 color_discrete_sequence=px.colors.qualitative.T10)
    fig.update_layout(xaxis_title='Year Intervals', yaxis_title='Number of Movies', 
                      legend_title='Languages', barmode='stack')
    return fig