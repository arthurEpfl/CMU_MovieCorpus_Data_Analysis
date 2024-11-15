import kagglehub
import pandas as pd
import os
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import ast

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
    Loads movies from the corpus into a DataFrame and preprocesses columns.
    """
    movies = pd.read_csv('../data/MovieSummaries/movie.metadata.tsv', sep='\t', header=None)
    movies.columns = [
        'wikipedia_movie_id', 'freebase_movie_id', 'movie_name', 'movie_release_date',
        'movie_box_office_revenue', 'movie_runtime', 'movie_languages', 'movie_countries',
        'movie_genres'
    ]
    
    # Convert dictionary-like columns to lists
    movies['movie_countries'] = movies['movie_countries'].apply(extract_dict_to_list)
    movies['movie_genres'] = movies['movie_genres'].apply(extract_dict_to_list)
    movies['movie_languages'] = movies['movie_languages'].apply(extract_dict_to_list)

    # Convert 'movie_release_date' to year only
    movies['movie_release_date'] = movies['movie_release_date'].apply(extract_release_year).astype(pd.Int64Dtype())

    # Fill missing values for runtime using the median
    movies['movie_runtime'] = movies['movie_runtime'].fillna(movies['movie_runtime'].median())

    return movies

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

def load_imdb_data():
    """
    Loads IMDB data and preprocesses it, converting title_year to integer format and handling missing values.
    """
    imdb_movies = pd.read_csv('../data/raw/imdb_5000_movies.csv')
    imdb_movies['movie_title'] = imdb_movies['movie_title'].str.strip().str.replace(u'\xa0', '')

    # Convert title_year to integer type, handling missing values as pd.NA
    imdb_movies['title_year'] = imdb_movies['title_year'].fillna(0).astype(int).replace({0: pd.NA})
    imdb_movies = imdb_movies[['movie_title', 'title_year', 'gross']]

    # Convert 'gross' to numeric
    imdb_movies['gross'] = pd.to_numeric(imdb_movies['gross'], errors='coerce')
    imdb_movies = imdb_movies.dropna(subset=['gross'])
    return imdb_movies

def get_matching_duplicates(movies, imdb_movies):
    """
    Get duplicates on the ['movie_name', 'movie_release_date'] subset in the movies dataframe that are also in the imdb_movies dataframe.
    Also return the number of duplicate pairs that have at least one missing value for box office revenue between them,
    the number of duplicate pairs without any missing values for box office,
    and the total number of imdb_films that match a duplicate pair. 
    """
    # Group by on the ['movie_name', 'release_year'] subset in movies and count occurrences
    duplicate_combinations = movies.groupby(['movie_name', 'movie_release_date']).size()
    duplicates = duplicate_combinations[duplicate_combinations > 1].index

    # Get the actual duplicates
    duplicate_rows_in_movies = movies[movies[['movie_name', 'movie_release_date']].apply(tuple, axis=1).isin(duplicates)]

    # Check if these duplicates are also in imdb_movies
    matching_duplicates = duplicate_rows_in_movies[
        duplicate_rows_in_movies[['movie_name', 'movie_release_date']].apply(tuple, axis=1).isin(
            imdb_movies[['movie_title', 'title_year']].apply(tuple, axis=1)
        )
    ]

    count_pairs_having_nan = 0
    count_pairs_not_having_nan = 0
    count_corresponding_imdb_movies = 0

    # Find the cases that will lead to potential issues during merge
    if not matching_duplicates.empty:
        for i, row in matching_duplicates.iterrows():
            sample_combination = row[['movie_name', 'movie_release_date']]

            sample_movies = movies[
                (movies['movie_name'] == sample_combination['movie_name']) &
                (movies['movie_release_date'] == sample_combination['movie_release_date'])
            ]

            if sample_movies['movie_box_office_revenue'].notnull().sum() == len(sample_movies):
                sample_imdb_movies = imdb_movies[
                    (imdb_movies['movie_title'] == sample_combination['movie_name']) &
                    (imdb_movies['title_year'] == sample_combination['movie_release_date'])
                ]
                count_pairs_not_having_nan += 1
            elif sample_movies['movie_box_office_revenue'].notnull().sum() != len(sample_movies):
                sample_imdb_movies = imdb_movies[
                    (imdb_movies['movie_title'] == sample_combination['movie_name']) &
                    (imdb_movies['title_year'] == sample_combination['movie_release_date'])
                ]
                count_pairs_having_nan += 1
                count_corresponding_imdb_movies += sample_imdb_movies.shape[0]
    else:
        print("No duplicates found in movies that also match imdb_movies.")

    # Dataframe to store variables
    counts_data = {
        'count_pairs_having_nan': [count_pairs_having_nan],
        'count_pairs_not_having_nan': [count_pairs_not_having_nan],
        'count_corresponding_imdb_movies': [count_corresponding_imdb_movies]
    }
    counts_nan_df = pd.DataFrame(counts_data)

    return matching_duplicates, counts_nan_df

def merge_movies_data(movies, imdb_movies, matching_duplicates):
    """
    Merges the movies DataFrame with the IMDB data on movie names and release dates to account for different versions.
    Removes movies that could be ambiguous when merging using the matching_duplicates dataframe produced.
    """

    # Remove appropriate movies from movies
    movies_to_remove = matching_duplicates[matching_duplicates['movie_box_office_revenue'].isna()]
    movies = movies[~movies['wikipedia_movie_id'].isin(movies_to_remove['wikipedia_movie_id'])]

    # Convert 'movie_box_office_revenue' to numeric, handling missing values
    movies['movie_box_office_revenue'] = pd.to_numeric(movies['movie_box_office_revenue'], errors='coerce')

    # Merge on both 'movie_name' and 'movie_release_date' to differentiate versions
    merged_movies = pd.merge(
        movies, imdb_movies,
        left_on=['movie_name', 'movie_release_date'],
        right_on=['movie_title', 'title_year'],
        how='left'
    )

    # Update 'movie_box_office_revenue' where it's NaN with 'gross' from IMDB
    merged_movies['movie_box_office_revenue'] = merged_movies['movie_box_office_revenue'].fillna(merged_movies['gross'])
    merged_movies.drop(columns=['movie_title', 'title_year', 'gross'], inplace=True)
    merged_movies = merged_movies.dropna(subset=['movie_box_office_revenue'])
    return merged_movies

def filter_with_summaries(merged_movies, summaries):
    """
    Get only movies with a box office revenue and plot summary available.
    """
    common_index = merged_movies['wikipedia_movie_id'].isin(summaries['wikipedia_movie_id'])
    filtered_movies_summaries_BO = merged_movies[common_index]
    # We remove duplicates
    filtered_movies_summaries_BO = filtered_movies_summaries_BO.drop_duplicates(subset='wikipedia_movie_id', keep='first')

    return filtered_movies_summaries_BO

def add_scraped_features(filtered_movies_summaries_BO):
    """
    Add features obtained from scraping on the movies with summaries and box office available.
    """
    # Load the new imdb_additional_movies_data_left_1.csv
    scraped_data = pd.read_csv('../data/processed/scraped_data_all.csv')

    # Merge the two DataFrames on 'wikipedia_movie_id'
    movies_scraped_data = pd.merge(filtered_movies_summaries_BO, scraped_data, on='wikipedia_movie_id', how='left')

    # Ensure there are no duplicates on 'wikipedia_movie_id'
    movies_scraped_data = movies_scraped_data.drop_duplicates(subset=['wikipedia_movie_id'])

    movies_scraped_data.drop(columns=['Unnamed: 0', 'movie_box_office_revenue_y', 'release_year'], inplace=True)
    movies_scraped_data.rename(columns={'movie_box_office_revenue_x': 'movie_box_office_revenue'}, inplace=True)

    return movies_scraped_data

def download_nltk_data():
    """
    Downloads necessary NLTK data files.
    """
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

def initialize_nlp_tools():
    """
    Initializes NLP tools: stopwords set and lemmatizer.
    Returns the stopwords set and lemmatizer object.
    """
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return stop_words, lemmatizer

def clean_text(text):
    """
    Cleans the input text by converting to lowercase, removing special characters and numbers.
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
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

    # Identify duplicates that could cause inconsistencies with merging on IMDB dataset
    matching_duplicates, counts_nan_df = get_matching_duplicates(movies, imdb_movies)

    # Merge datasets and update box office revenue
    merged_movies = merge_movies_data(movies, imdb_movies, matching_duplicates)

    # Load and preprocess summaries
    summaries = load_and_preprocess_summaries(stop_words, lemmatizer)

    # Get movies with a summary and box office revenue available
    filtered_movies_summaries_BO = filter_with_summaries(merged_movies, summaries)

    # Add the features obtained from the web scraping
    movies_scraped_data = add_scraped_features(filtered_movies_summaries_BO)

    output_path = "../data/processed/merged_movies.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_movies.to_csv(output_path, index=False)
    print(f"Preprocessed merged_movies saved to: {output_path}")

    summaries_output_path = "../data/processed/summaries_preprocessed.csv"
    os.makedirs(os.path.dirname(summaries_output_path), exist_ok=True)
    summaries.to_csv(summaries_output_path, index=False)
    print(f"Preprocessed summaries saved to: {summaries_output_path}")

    counts_nan_df_output_path = "../data/processed/counts_nan_df.csv"
    os.makedirs(os.path.dirname(counts_nan_df_output_path), exist_ok=True)
    counts_nan_df.to_csv(counts_nan_df_output_path, index=False)
    print(f"Counts_nan_df saved to: {counts_nan_df_output_path}")

    filtered_movies_summaries_BO_output_path = "../data/processed/filtered_movies_summaries_BO_output_path.csv"
    os.makedirs(os.path.dirname(filtered_movies_summaries_BO_output_path), exist_ok=True)
    filtered_movies_summaries_BO.to_csv(filtered_movies_summaries_BO_output_path, index=False)
    print(f"Filtered movies with summary and BO saved to: {filtered_movies_summaries_BO_output_path}")

    movies_scraped_data_output_path = "../data/processed/movies_scraped_data.csv"
    os.makedirs(os.path.dirname(movies_scraped_data_output_path), exist_ok=True)
    movies_scraped_data.to_csv(movies_scraped_data_output_path, index=False)
    print(f"Filtered movies scraped saved to: {movies_scraped_data_output_path}")