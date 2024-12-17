import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ast  
import seaborn as sns

def load_data(file_path='data/processed/movies_summary_BO.csv'):
    return pd.read_csv(file_path)

def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val


# Part 4 - Exploratory Data Analysis

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


def plot_top_countries(movies):
    # Ensure that the movie_countries column is parsed as lists
    movies['movie_countries'] = movies['movie_countries'].apply(safe_literal_eval)

    # Top 10 Countries by Number of Movies Produced and filter out rows with empty countries
    movies_countries_exploded = movies[movies['movie_countries'].map(len) > 0].explode('movie_countries')
    top_countries = movies_countries_exploded['movie_countries'].value_counts().head(10)

    # Define specific colors for each country
    country_colors = {
        'United States': '#1f77b4',
        'India': '#ff7f0e',
        'United Kingdom': '#2ca02c',
        'France': '#d62728',
        'Germany': '#9467bd',
        'Japan': '#8c564b',
        'Canada': '#e377c2',
        'Italy': '#7f7f7f',
        'Spain': '#bcbd22',
        'China': '#17becf'
    }

    # Reorder top_countries to have 'United States' first
    if 'United States' in top_countries.index:
        top_countries = top_countries.reindex(['United States'] + [country for country in top_countries.index if country != 'United States'])

    # Create a list of colors for the top countries
    colors = [country_colors.get(country, '#636EFA') for country in top_countries.index]

    fig = px.bar(top_countries, x=top_countries.values, y=top_countries.index, orientation='h',
                 title='Top 10 Countries by Number of Movies Produced',
                 labels={'x': 'Number of Movies', 'y': 'Countries'},
                 color=top_countries.index, color_discrete_sequence=colors)
    fig.update_layout(xaxis_title='Number of Movies', yaxis_title='Countries')
    return fig


def plot_runtime_and_release_year_distributions(movies):
    movies = movies[movies['movie_release_date'] >= 1920]
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        'Boxplot of Movie Runtime', 'Boxplot of Movie Release Year'))

    # Boxplot for movie runtime
    runtime_box = px.box(movies[movies['movie_runtime'] < 300], x='movie_runtime', color_discrete_sequence=['blue'])
    for trace in runtime_box['data']:
        fig.add_trace(trace, row=1, col=1)

    # Boxplot for movie release year
    release_year_box = px.box(movies, x='movie_release_date', color_discrete_sequence=['green'])
    for trace in release_year_box['data']:
        fig.add_trace(trace, row=1, col=2)

    # Update layout for better appearance
    fig.update_layout(height=400, width=1000, title_text='Runtime and Release Year Boxplots', showlegend=False)
    fig.update_xaxes(title_text='Runtime (minutes)', row=1, col=1)
    fig.update_xaxes(title_text='Release Year', row=1, col=2)

    # Set x-axis limits
    fig.update_xaxes(range=[0, 300], row=1, col=1)  # Adjust the range as needed
    fig.update_xaxes(range=[1920, 2020], row=1, col=2)  # Adjust the range as needed

    return fig


# Part 5 - Genre Analysis
def plot_top_genres(movies):
    movies['movie_genres'] = movies['movie_genres'].apply(safe_literal_eval)
    movies_genres_exploded = movies.explode('movie_genres')
    genre_counts = movies_genres_exploded['movie_genres'].value_counts()
    genre_percentages = (genre_counts / genre_counts.sum()) * 100
    top_genre = genre_percentages.head(15)
    
    fig = px.bar(top_genre, x=top_genre.index, y=top_genre.values, 
                 title='Distribution of Top 15 Movie Genres',
                 labels={'x': 'Genres', 'y': 'Frequency (%)'},
                 color=top_genre.index, color_discrete_sequence=px.colors.qualitative.Plotly)
    fig.update_layout(xaxis_title='Genres', yaxis_title='Frequency (%)', 
                      xaxis_tickangle=-45, title_font_size=16)
    return fig


def plot_mean_revenues_by_genre(movies):
    # Ensure that the movie_genres column is parsed as lists
    movies['movie_genres'] = movies['movie_genres'].apply(safe_literal_eval)
    movies_genres_exploded = movies.explode('movie_genres')
    
    # Calculate the top 15 genres
    genre_counts = movies_genres_exploded['movie_genres'].value_counts()
    top_genre = genre_counts.head(15).index
    
    # Filter the dataframe to include only the top 15 genres
    filtered_df = movies_genres_exploded[movies_genres_exploded['movie_genres'].isin(top_genre)]
    
    # Calculate the mean revenues and sort them in descending order
    mean_revenues = filtered_df.groupby('movie_genres')['movie_box_office_revenue'].mean().reset_index()
    mean_revenues = mean_revenues.sort_values(by='movie_box_office_revenue', ascending=False)
    
    # Plot the mean revenues using Plotly
    fig = px.bar(mean_revenues, x='movie_genres', y='movie_box_office_revenue', 
                 title='Mean Box Office Revenues by Top 15 Movie Genres',
                 labels={'movie_genres': 'Movie Genres', 'movie_box_office_revenue': 'Mean Box Office Revenue'},
                 color='movie_genres', color_discrete_sequence=px.colors.qualitative.Plotly)
    fig.update_layout(xaxis_title='Movie Genres', yaxis_title='Mean Box Office Revenue', 
                      xaxis_tickangle=-45, title_font_size=16)
    return fig
