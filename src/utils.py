import ast
import pandas as pd

def extract_dict_to_list(entry):
    """
    Extracts values from a dictionary-like string and returns a list of values.
    """
    try:
        entry_dict = ast.literal_eval(entry) 
        return list(entry_dict.values()) 
    except (ValueError, SyntaxError):
        return []

def extract_release_year(date_str):
    """
    Extracts the year from a date string.
    """
    try:
        return pd.to_datetime(date_str).year
    except (ValueError, TypeError):
        try:
            return int(date_str)
        except ValueError:
            return None

def safe_literal_eval(val):
    """
    Safely evaluates a string containing a Python literal (list, dictionary, etc.).
    If evaluation fails, returns the original value.
    """
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val
    
def parse_genres(genres):
    """
    Parses a string representation of a list of genres and returns a list of genres.
    """
    if isinstance(genres, list):
        return genres
    try:
        return ast.literal_eval(genres)
    except (ValueError, SyntaxError):
        return genres.strip('[]').replace("'", "").split(', ')
    
def add_scraped_features(scraped_data, filtered_movies_summaries_BO):
    """
    Add features obtained from scraping on the movies with summaries and box office available.
    """
    # Merge the two DataFrames on 'wikipedia_movie_id'
    movies_scraped_data = pd.merge(filtered_movies_summaries_BO, scraped_data, on='wikipedia_movie_id', how='left')

    # Ensure there are no duplicates on 'wikipedia_movie_id'
    movies_scraped_data = movies_scraped_data.drop_duplicates(subset=['wikipedia_movie_id'])

    return movies_scraped_data