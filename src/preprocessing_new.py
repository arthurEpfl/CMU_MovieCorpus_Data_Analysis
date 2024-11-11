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
import cpi

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


def merge_movies_data(movies, imdb_movies):
    """
    Merges the movies DataFrame with the IMDB data on movie names and release dates to account for different versions.
    """
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

#############
def clean_characters():
    
    characters = pd.read_csv('../data/MovieSummaries/character.metadata.tsv', sep = '\t', header = None)

    characters.columns = ['wikipedia_movie_id', 'freebase_movie_id', 'movie_release_date', 'character_name', 'actor_birth', 'actor_gender', 'actor_height', 'actor_ethnicity', 'actor_name', 'actor_age', 'freebase_character_map', 'freebase_character_id', 'freebase_actor_id']
    characters = characters.dropna(subset=['wikipedia_movie_id', 'character_name'])

    return characters

#############

def clean_tropes():

    tv_tropes = pd.read_csv('../data/MovieSummaries/tvtropes.clusters.txt', sep = '\t', header = None)

    tv_tropes.columns = ['trope', 'details']
    tv_tropes['details'] = tv_tropes['details'].apply(eval)
    tv_tropes = tv_tropes.join(pd.json_normalize(tv_tropes['details'])).drop(columns=['details'])

    tv_tropes.columns = [
    'trope',
    'character_name',          # Change `char` to `character_name`
    'movie_name',              # Change `movie` to `movie_name`
    'freebase_movie_id',       # Change `id` to `freebase_movie_id`
    'actor_name'               # Change `actor` to `actor_name`
    ]

    return tv_tropes
#############

#############
def load_movies_with_main_characters_and_BO(movies, x):
    'DOESNT FILTER TO MOVIES WITH SUMMARIES LIKE IN OTHER FCT load_movies_with_actors_and_BO_IMDB'

    characters = clean_characters()
    tv_tropes = clean_tropes()

    # Add tropes to characters where it is available
    merged_characters = characters.merge(tv_tropes[['character_name', 'trope', 'actor_name']],
                                      on=['character_name', 'actor_name'],
                                      how='left',
                                      indicator=True)
    characters = merged_characters

    # Choose movies with less than x characters
    sorted_characters = characters.sort_values(by='wikipedia_movie_id', ascending=False)
    char_count_per_movie = sorted_characters.groupby('wikipedia_movie_id')['character_name'].nunique()
    movies_with_at_most_x_actors = char_count_per_movie[char_count_per_movie < x+1].index

    # Keep x principal characters, keep only movies with available box office
    characters_x_df = sorted_characters[sorted_characters['wikipedia_movie_id'].isin(movies_with_at_most_x_actors)]
    with_BO = characters_x_df.merge(movies, on='wikipedia_movie_id', how='inner')

    # movies is the merged_movies file in data/processed/merged_movies.csv
    with_BO_grouped = with_BO.groupby('wikipedia_movie_id').agg({
    'actor_name': lambda x: list(x),  # Collect actor names as list
    'character_name': lambda x: list(x)  # Collect character names as list
    }).reset_index()

    # Function to create separate actor and character columns per movie
    def extract(info_list):
        info_list += [np.nan] * (3 - len(info_list))
        return info_list[:3]

    # Apply function on actor_names
    with_BO_grouped[['actor1_name', 'actor2_name', 'actor3_name']] = pd.DataFrame(
        with_BO_grouped['actor_name'].apply(extract).to_list(),
        index=with_BO_grouped.index
    )

    # Apply function on character_names
    with_BO_grouped[['character1_name', 'character2_name', 'character3_name']] = pd.DataFrame(
        with_BO_grouped['character_name'].apply(extract).to_list(),
        index=with_BO_grouped.index
    )

    # Drop the original actor_name and character_name columns
    with_BO_grouped = with_BO_grouped.drop(columns=['actor_name', 'character_name'])

    # Merge the new columns back to movies with available box office
    movies_x_principal_characters = with_BO.drop_duplicates('wikipedia_movie_id').merge(
        with_BO_grouped, on='wikipedia_movie_id', how='left'
    )

    # Keep only selected columns
    columns_to_keep = ['wikipedia_movie_id', 'freebase_movie_id_x', 'movie_release_date_x', 
                    'actor1_name', 'actor2_name', 'actor3_name',
                    'character1_name', 'character2_name', 'character3_name', 'movie_box_office_revenue']
    movies_x_principal_characters = movies_x_principal_characters[columns_to_keep]

    return movies_x_principal_characters
#############


#############
def load_movies_with_actors_and_BO_IMDB(movies):

    summaries = pd.read_csv('../data/processed/summaries_preprocessed.csv')
    characters = clean_characters()
    tv_tropes = clean_tropes()
    imdb_movies = pd.read_csv('../data/raw/imdb_5000_movies.csv')

    # Get movies with summaries and characters
    common_index = movies['wikipedia_movie_id'].isin(summaries['wikipedia_movie_id']) & movies['wikipedia_movie_id'].isin(characters['wikipedia_movie_id'])
    filtered_movies_summaries_characters = movies[common_index]

    # Add tropes to characters where trope available
    characters = characters.merge(tv_tropes[['character_name', 'trope', 'actor_name']],
                                      on=['character_name', 'actor_name'],
                                      how='left',
                                      indicator=True)

    # Drop duplicates
    filtered_movies = filtered_movies_summaries_characters.drop_duplicates(subset=['wikipedia_movie_id'])

    # Keep characters with either a trope or a summary
    characters_with_tropes = characters[characters['trope'].notna()]
    movie_ids_with_summaries = set(summaries['wikipedia_movie_id'])
    characters_with_summaries = characters[characters['wikipedia_movie_id'].isin(movie_ids_with_summaries)]
    combined_characters = pd.concat([characters_with_tropes, characters_with_summaries]).drop_duplicates()
    # df_characters = combined_characters.drop_duplicates(subset='character_name') (remove this since we keep all movies with same characters?)
    'CHECK THIS COMMENT ABOVE'

    # Get movie titles, release year, and principal actor names from IMDB kaggle dataset
    imdb_movies['movie_title'] = imdb_movies['movie_title'].str.strip().str.replace(u'\xa0', '')
    imdb_selected = imdb_movies[['movie_title', 'title_year', 'actor_1_name', 'actor_2_name', 'actor_3_name']]

    # Inner merge to get all movies that have character-trope/summary and are in the IMDB dataset
    merged_movies = pd.merge(
    filtered_movies, imdb_selected,
    left_on=['movie_name', 'movie_release_date'],
    right_on=['movie_title', 'title_year'],
    how='inner'
    )

    # All movies with three principal actor names
    final_characters = combined_characters.merge(merged_movies[['wikipedia_movie_id', 'actor_1_name', 'actor_2_name', 'actor_3_name']],
                                        on='wikipedia_movie_id',
                                        how='inner')

    # All characters 
    final_characters = final_characters[
    (final_characters['actor_name'].isin(final_characters['actor_1_name'])) |
    (final_characters['actor_name'].isin(final_characters['actor_2_name']))
    ]

    # Clean to keep relevant columns
    columns_to_drop = ['_merge', 'actor_1_name', 'actor_2_name', 'actor_3_name']
    final_characters = final_characters.drop(columns=columns_to_drop)

    # Drop duplicates
    valid_wikipedia_ids = final_characters['wikipedia_movie_id'].unique()
    movies_trope_summary_main_actors_IMDB = merged_movies[merged_movies['wikipedia_movie_id'].isin(valid_wikipedia_ids)]    
    movies_trope_summary_main_actors_IMDB = movies_trope_summary_main_actors_IMDB.drop_duplicates(subset='wikipedia_movie_id', keep='first')

    return movies_trope_summary_main_actors_IMDB
#############

#############



#############
def adjust_for_inflation(df, year):

    def get_year(date):
        if pd.isna(date):  # Check if the date is NaN
            return None
        if isinstance(date, (int, float)):  # If it's already a year
            return int(date)
        elif isinstance(date, str):  # If it's a string, try to parse it as a date
            try:
                return pd.to_datetime(date).year
            except ValueError:
                return None  # Return None if date can't be parsed
        elif hasattr(date, 'year'):  # If it's a datetime object
            return date.year
        return None

    if 'movie_release_date_x' in df.columns:
        df.rename(columns={'movie_release_date_x': 'movie_release_date'}, inplace=True)

    df['years_only'] = df.apply(
        lambda row: get_year(row['movie_release_date']),
        axis=1
    )

    df['adjusted_revenue'] = df.apply(
        lambda row: cpi.inflate(row['movie_box_office_revenue'], int(row['years_only']), to=year) 
        if pd.notna(row['movie_box_office_revenue']) and pd.notna(row['years_only']) else None,
        axis=1
    )
    return df
#############

#############


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

    # Get movies with at most x characters and available box office revenues
    movies_x_principal_characters = load_movies_with_main_characters_and_BO(merged_movies, x=3)
    movies_x_principal_characters = adjust_for_inflation(movies_x_principal_characters)

    movies_x_principal_characters_output_path = "../data/processed/movies_x_principal_characters.csv"
    os.makedirs(os.path.dirname(movies_x_principal_characters_output_path), exist_ok=True)
    movies_x_principal_characters.to_csv(movies_x_principal_characters_output_path, index=False)
    print(f"Preprocessed summaries saved to: {movies_x_principal_characters_output_path}")

    # Get movies with summary/trope and principal actors available on IMDB kaggle dataset
    movies_trope_summary_main_actors_IMDB = load_movies_with_actors_and_BO_IMDB(merged_movies, year=2015)
    movies_trope_summary_main_actors_IMDB = adjust_for_inflation(movies_trope_summary_main_actors_IMDB, year=2015)

    movies_trope_summary_main_actors_IMDB_output_path = "../data/processed/movies_trope_summary_main_actors_IMDB.csv"
    os.makedirs(os.path.dirname(movies_trope_summary_main_actors_IMDB_output_path), exist_ok=True)
    movies_trope_summary_main_actors_IMDB.to_csv(movies_trope_summary_main_actors_IMDB_output_path, index=False)
    print(f"Preprocessed summaries saved to: {movies_trope_summary_main_actors_IMDB_output_path}")