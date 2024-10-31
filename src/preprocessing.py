import kagglehub
import pandas as pd
import os
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def download_imdb_dataset():
    """
    Downloads the IMDB dataset from Kaggle and saves it to the local repository.
    """
    # Download the dataset from Kaggle
    ds_movies = kagglehub.dataset_download("carolzhangdc/imdb-5000-movie-dataset")
    df_movies = pd.read_csv(os.path.join(ds_movies, "movie_metadata.csv"))
    local_path = "../data/raw/imdb_5000_movies.csv"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    df_movies.to_csv(local_path, index=False)
    print(f"Dataset saved to: {local_path}")

def load_movies_from_corpus():
    """
    Loads movies from the corpus into a DataFrame.
    """
    movies = pd.read_csv('../data/MovieSummaries/movie.metadata.tsv', sep='\t', header=None)
    movies.columns = [
        'wikipedia_movie_id', 'freebase_movie_id', 'movie_name', 'movie_release_date',
        'movie_box_office_revenue', 'movie_runtime', 'movie_languages', 'movie_countries',
        'movie_genres'
    ]
    return movies

def load_imdb_data():
    """
    Loads IMDB data and preprocesses it.
    """
    imdb_movies = pd.read_csv('../data/raw/imdb_5000_movies.csv')
    imdb_movies['movie_title'] = imdb_movies['movie_title'].str.strip().str.replace(u'\xa0', '')
    imdb_movies = imdb_movies[['movie_title', 'gross']]

    # Convert 'gross' to numeric to fill empty values later
    imdb_movies['gross'] = pd.to_numeric(imdb_movies['gross'], errors='coerce')
    imdb_movies = imdb_movies.dropna(subset=['gross'])
    return imdb_movies

def merge_movies_data(movies, imdb_movies):
    """
    Merges the movies DataFrame with the IMDB data on movie names and updates box office revenue.
    """
    # Convert 'movie_box_office_revenue' to numeric, handling missing values
    movies['movie_box_office_revenue'] = pd.to_numeric(movies['movie_box_office_revenue'], errors='coerce')

    # Merge the two DataFrames on movie name/title
    merged_movies = pd.merge(
        movies, imdb_movies,
        left_on='movie_name', right_on='movie_title',
        how='left'
    )

    # Update 'movie_box_office_revenue' where it's NaN with 'gross' from IMDB
    merged_movies['movie_box_office_revenue'] = merged_movies['movie_box_office_revenue'].fillna(merged_movies['gross'])
    merged_movies.drop(columns=['movie_title', 'gross'], inplace=True)
    merged_movies = merged_movies.dropna(subset=['movie_box_office_revenue'])
    return merged_movies

def download_nltk_data():
    """
    Downloads necessary NLTK data files.
    """
    nltk.download('punkt')        # For tokenization
    nltk.download('stopwords')    # For stopwords
    nltk.download('wordnet')      # For lemmatization

def initialize_nlp_tools():
    """
    Initializes NLP tools: stopwords set and lemmatizer.
    Returns the stopwords set and lemmatizer object.
    """
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return stop_words, lemmatizer

# Text preprocessing functions from eda_adam.ipynb

def clean_text(text):
    """
    Cleans the input text by converting to lowercase, removing special characters and numbers.
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters and numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

def tokenize_text(text):
    """
    Tokenizes the input text into a list of words.
    """
    return word_tokenize(text)

def remove_stopwords(tokens, stop_words):
    """
    Removes stopwords from the list of tokens.
    """
    return [word for word in tokens if word not in stop_words]

def lemmatize_tokens(tokens, lemmatizer):
    """
    Lemmatizes the list of tokens using WordNet lemmatizer.
    """
    return [lemmatizer.lemmatize(word) for word in tokens]

def load_and_preprocess_summaries(stop_words, lemmatizer):
    """
    Loads plot summaries and applies text preprocessing steps.
    """
    summaries = pd.read_csv('../data/MovieSummaries/plot_summaries.txt', sep='\t', header=None)
    summaries.columns = ['wikipedia_movie_id', 'plot_summary']
    summaries['clean_plot_summary'] = summaries['plot_summary'].apply(clean_text)
    summaries['tokenized_plot_summary'] = summaries['clean_plot_summary'].apply(tokenize_text)
    summaries['filtered_tokens'] = summaries['tokenized_plot_summary'].apply(lambda tokens: remove_stopwords(tokens, stop_words))
    summaries['lemmatized_tokens'] = summaries['filtered_tokens'].apply(lambda tokens: lemmatize_tokens(tokens, lemmatizer))

    return summaries

if __name__ == "__main__":
    # Download necessary NLTK data
    download_nltk_data()

    # Initialize NLP tools
    stop_words, lemmatizer = initialize_nlp_tools()
    download_imdb_dataset()
    movies = load_movies_from_corpus()
    imdb_movies = load_imdb_data()

    # Merge datasets and update box office revenue
    merged_movies = merge_movies_data(movies, imdb_movies)

    # Load and preprocess summaries
    summaries = load_and_preprocess_summaries(stop_words, lemmatizer)

    output_path = "../data/processed/merged_movies.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_movies.to_csv(output_path, index=False)

    summaries_output_path = "../data/processed/summaries_preprocessed.csv"
    os.makedirs(os.path.dirname(summaries_output_path), exist_ok=True)
    summaries.to_csv(summaries_output_path, index=False)
    print(f"Preprocessed summaries saved to: {summaries_output_path}")