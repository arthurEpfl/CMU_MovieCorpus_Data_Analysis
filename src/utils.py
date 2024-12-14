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
        
def clean_title(title):
    """
    Cleans movie title by stripping whitespace and removing non-breaking space characters.
    """
    return title.strip().replace(u'\xa0', '')

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