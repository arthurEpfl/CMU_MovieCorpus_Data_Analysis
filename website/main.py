import streamlit as st
from streamlit.logger import get_logger
from streamlit_lottie import st_lottie
import requests
import pandas as pd
import ast

import intro
import genre
import plot
import regression as reg
import conclusion as conc
import format_text as texts


# --- CONFIG --- #

LOGGER = get_logger(__name__)

def set_css():
        css = """
        <style>
            /* Main page layout */
            .main .block-container {
                padding-right: 12rem;   
                padding-left: 12rem;    
            }
            .justified-text {
                text-align: justify;
                text-justify: inter-word;
            }
            .size-text { font-size: 18px; }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

def load_animation(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()


# @st.cache_data
def load_data():
    # Load main datasets after preprocessing and classification
    movies = pd.read_csv('../data/processed/movies_summary_BO.csv', sep=',')
    classified = pd.read_csv('../data/processed/movies_with_classifications.csv')
    movies_for_reg = pd.read_csv('../data/processed/movies_budget_inflation_final.csv')
    return movies, classified, movies_for_reg

movies, classified, movies_for_reg = load_data()

def safe_literal_eval(val):
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


if 'movie_countries' in movies_for_reg.columns:
    movies_for_reg['movie_countries'] = movies_for_reg['movie_countries'].apply(safe_literal_eval)

if 'movie_genres' in movies_for_reg.columns:
    movies_for_reg['movie_genres'] = movies_for_reg['movie_genres'].apply(safe_literal_eval)

if 'movie_countries' in movies.columns:
    movies['movie_countries'] = movies['movie_countries'].apply(safe_literal_eval)

if 'movie_genres' in movies.columns:
    movies['movie_genres'] = movies['movie_genres'].apply(safe_literal_eval)

movies['movie_languages'] = movies['movie_languages'].apply(safe_literal_eval)
movies_for_reg['movie_languages'] = movies_for_reg['movie_languages'].apply(safe_literal_eval)

movies['movie_genres'] = movies['movie_genres'].apply(parse_genres)
movies_for_reg['movie_genres'] = movies_for_reg['movie_genres'].apply(parse_genres)

movies['year_interval'] = (movies['movie_release_date'] // 5) * 5  

movies_budget = movies.dropna(subset=['budget'])

# --- MAIN --- #
def run():
    st.set_page_config(page_title="ADARABLE", page_icon="🎬", layout="wide", initial_sidebar_state="collapsed")

    # if needed
    def upload_css(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    set_css()

    # --- INTRODUCTION --- #
    with st.container():
        st.title("🎬 Decoding the Blueprint of a Blockbuster: Analyzing Plot Structures for Box Office Success :sparkles:")
        st.subheader("Introduction")
        col1, col2 = st.columns(2)
        with col1:
            # call intro.py

            #replace this
            texts.format_text("Here goes the text for the intro.")
        with col2:
            texts.format_text("Here goes some sexy image or animation.")

        

    # --- SIDEBAR --- #
    with st.container():
        with st.sidebar:
            # --- TABLE OF CONTENTS --- #
            st.title("Table of Contents")
            # --- NAVIGATION --- #
            # implement here the navigation through table of contents

    
    # --- DATA STORY --- #
            
    #### PART 1 - Genres ####
    with st.container():
        st.title("Movie genres: an important factor for financial success ?")

        # call functions from genre.py
                    
        texts.format_text("call functions from genre.py")
    
    #### PART 2 - Plot Structures ####       
    with st.container():
        st.title("When story comes into play")
        
        texts.format_text("call functions from plot.py")
        
    #### PART 3 - Linear regression ####  
    with st.container():
        st.title("Overall, what makes a movie financially successful ?")
        st.subheader("Fitting a linear regresion model")
        col1, col2 = st.columns(2)
        with col1:
            # call intro.py

            #replace this
            texts.int()
            texts.regression_interpretation()
        with col2:
            reg.plot_reg_coeffs(movies_for_reg)
    with st.container():
        reg.plot_budget_profit(movies_for_reg)

        reg.ROI_plot(movies_for_reg)

        texts.format_text("call functions from reg.py")

    #### PART 4 - Conclusion ####
    with st.container():
        st.title("Conclusion")

        texts.format_text("call functions from conclusion.py")
        
        
        
     
        
if __name__ == "__main__":
    run()