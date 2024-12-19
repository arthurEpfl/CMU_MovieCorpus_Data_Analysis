import streamlit as st  
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os  

import plot_app
import mod

import plotly.express as px
import plotly.graph_objects as go  

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import statsmodels.api as sm
import sklearn.metrics as metrics

# Set Streamlit page config
st.set_page_config(
    page_title="Cinematic Moral Dilemmas",
    page_icon="🎬",
    layout="wide",
)  

st.markdown("""
    <style>
    .centered-content {
        max-width: 800px;
        margin: auto;
        padding: 200px;
        border: 200px solid #ddd;
        border-radius: 100px;
        background-color: #f9f9f9;
    }
    .container {
        padding-left: 5000px;
        padding-right: 5000px;
    }
    </style>
    <div class="container">
""", unsafe_allow_html=True)

# Your existing Streamlit code here

st.markdown("</div>", unsafe_allow_html=True)

# Set matplotlib and seaborn style for dark background and white text
plt.style.use('dark_background')
sns.set_style("dark")
plt.rcParams['figure.facecolor'] = '#000000'
plt.rcParams['axes.facecolor'] = '#000000'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.titlecolor'] = 'white'  

@st.cache_data
def load_data():
    # Load main datasets after preprocessing and classification
    movies = pd.read_csv('data/processed/movies_summary_BO.csv', sep=',')
    classified = pd.read_csv('data/processed/movies_with_classifications.csv')
    return movies, classified

movies, classified = load_data()

def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val

if 'movie_countries' in movies.columns:
    movies['movie_countries'] = movies['movie_countries'].apply(safe_literal_eval)

if 'movie_genres' in movies.columns:
    movies['movie_genres'] = movies['movie_genres'].apply(safe_literal_eval)


st.markdown("""
    <style>
    .title-viridis-light {
        background: linear-gradient(135deg, #5e4fa2 0%, #3288bd 50%, #66c2a5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
    <div style="text-align:center;">
        <h1 class="title-viridis-light">🎬 Decoding the Blueprint of a Blockbuster: Analyzing Plot Structures for Box Office Success</h1>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="font-size:18px; text-align:center;">
    <div style="text-align:center; font-size:24px; font-weight:bold;">
        What makes a movie both unforgettable and successful?
    </div>
    <br>
    Is it the incredible acting, the clever marketing, or the relatable themes that stick with us? 
    While all of these play a part, history has shown that the real magic lies in the story—the way 
    it draws us in, connects with us, and keeps us hooked. From the magical world of Harry Potter 
    to the mind-bending twists of Inception, blockbuster movies all have something special in their 
    plots that audiences can’t get enough of. But can we measure that? Is there a way to figure out 
    what makes a story truly successful?
</div>  
<br><br>
""", unsafe_allow_html=True)  

st.markdown("""
<div style="text-align:center; font-size:24px; font-family: 'Cursive', sans-serif;">
    Let's analyze this, Action !
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<br><br>", unsafe_allow_html=True)  # Add extra space
    st.image("images_datastory/movie_clap.png", use_container_width=True, width=200)


##### Section 1 : TABLE OF CONTENTS  #####  

st.markdown("""
<div style="text-align:center; font-size:28px; font-family: 'Cursive', sans-serif;">
    Our Story Timeline :
</div>
""", unsafe_allow_html=True)  

# Create the non-clickable list with lighter viridis text, centered, and light boxes with beautiful borders
st.markdown("""
    <style>
    .viridis-list {
        list-style-type: none;
        padding: 0;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    .viridis-list li {
        font-size: 16px; /* Smaller font size */
        font-weight: bold; /* Bold text */
        margin: 4px 0; /* Smaller margin */
        padding: 8px; /* Smaller padding */
        background: #f9f9f9; /* Light box background */
        border-radius: 10px; /* Rounded corners */
        border: 2px solid #2c3e50; /* Darker border */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Light shadow */
        color: transparent;
        background: linear-gradient(135deg, #5e4fa2 0%, #3288bd 50%, #66c2a5 100%); /* Lighter viridis gradient */
        -webkit-background-clip: text;
        display: block; /* Display each item on a new line */
    }
    </style>
    <div style="text-align:center;">
        <ul class="viridis-list">
            <li>First Explorations</li>
            <li>Preprocessing</li>
            <li>Exploratory Data Analysis</li>
            <li>How genres are related to box offices revenues and commercial success</li>  
            <li>Plot Structure Analysis with Clustering & LLM Classification</li>
            <li>Predictive Modeling</li>  
            <li>Conclusion</li>
        </ul>
    </div>
""", unsafe_allow_html=True)


st.markdown("---")


# ===================== SECTION 2: FIRST EXPLORATIONS =====================
st.markdown("""
    <style>
    .title-viridis-light {
        background: linear-gradient(135deg, #3b528b 0%, #21918c 50%, #27ad81 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 44px; /* Adjust the font size as needed */
    }
    </style>
    <div style="text-align:center;">
        <h2 id="first-explorations" class="title-viridis-light">First Explorations</h2>
    </div>
""", unsafe_allow_html=True)


st.markdown("""
<div style="font-size:18px; text-align:center;">
    <div style="text-align:center; font-size:19px; font-weight:bold;">
        Initially, we explored raw data from the <a href="https://www.cs.cmu.edu/~ark/personas/" target="_blank">CMU Movie Corpus dataset</a>.
    </div>
    <br>
    In the latter, missing box office revenues were identified so there is a need for external data sources. Also, budget values are missing,
    we need them if we want to compute profitability of these movies!  
    <br><br>
</div> 
""", unsafe_allow_html=True) 

# Center the image using columns and make the container smaller
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("images_datastory/titanic_image.webp", caption="Titanic Movie", use_container_width=True)

st.markdown("""
<div style="font-size:18px; text-align:center;">
A movie like Titanic is a great example of a blockbuster that captivated audiences worldwide. It's interesting to study this, 
is its success due to its genre, its plot structure, Jack Dawson itself, Rose DeWitt or something else? We will explore this in the following sections. 
<br><br>
""", unsafe_allow_html=True)


st.markdown("""
<div style="font-size:18px; text-align:center;">
Noticing here that there are almost 90% of movies that do not have 
revenue in the CMU movie dataset, this is an issue for our project, 
as we want to investigate how different plot structures and narrative 
formulas affect a movie’s box office success. An other dataset will be 
used to get this information. Further methods during preprocessing will 
be used, such as merging with full IMDb dataset and web scraping on IMDb, 
to complete these missing values. 
<br><br>
""", unsafe_allow_html=True)

st.markdown("---")

# ===================== SECTION 3: PREPROCESSING =====================  

st.markdown("""
    <style>
    .title-viridis-light {
        background: linear-gradient(135deg, #3b528b 0%, #21918c 50%, #27ad81 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 44px; /* Adjust the font size as needed */
    }
    </style>
    <div style="text-align:center;">
        <h2 id="preprocessing" class="title-viridis-light">Preprocessing</h2>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="font-size:18px; text-align:center;">
We made a part of preprocessing in order to get a cleaned dataset and to fill in missing box office revenues. First 
we merged the CMU Movie Corpus dataset with the IMDb dataset to get more information about the movies.  
<br><br>
""", unsafe_allow_html=True)


st.markdown("""
<div style="font-size:18px; text-align:center;">
Kaggle IMDb dataset may be found following this link : <a href="https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset" target="_blank">IMDb Dataset</a>   
<br><br>
</div>
""", unsafe_allow_html=True)  

st.markdown("""
<div style="font-size:18px; text-align:center;">
We keep only movies with an available box office revenue and plot summary in order to make consistent analyses.  
<br><br>
""", unsafe_allow_html=True)  

st.markdown("""
    <style>
    .text-viridis-light {
        background: linear-gradient(135deg, #5e4fa2 0%, #3288bd 50%, #66c2a5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 24px; /* Adjust the font size as needed */
        font-family: 'Cursive', sans-serif;
    }
    </style>
    <div style="text-align:center;">
        <span class="text-viridis-light">Did you think of Web scraping to get budget values?</span>
    </div>
""", unsafe_allow_html=True)  

  

st.markdown("""
<div style="font-size:18px; text-align:center;">
The web scraping is useful to complete our dataset with additional features that can be useful for further analysis, notably film budget.
We use Web Scraping with Selenium that simulates a browser to access IMDb pages dynamically.
We extract structured details like box office revenue, budget, and ratings directly from the website. Wikipedia API helps fetches data 
programmatically without loading full web pages. <br><br>
""", unsafe_allow_html=True)  


st.markdown("""
<div style="font-size:18px; text-align:center;">
This also helps answer the following key questions about movie success, 
does a larger budget always result in higher box office earnings?
 <br><br>
""", unsafe_allow_html=True)

st.markdown("---")






# ===================== SECTION 4: EDA =====================
st.markdown("""
    <style>
    .title-viridis-light {
        background: linear-gradient(135deg, #3b528b 0%, #21918c 50%, #27ad81 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 44px; /* Adjust the font size as needed */
    }
    </style>
    <div style="text-align:center;">
        <h2 id="eda" class="title-viridis-light">Exploratory Data Analysis</h2>
    </div>
""", unsafe_allow_html=True)


st.markdown("""
<div style="font-size:18px; text-align:center;">
Let's analyse the data and see what we can find out about the movies, now that we have more informations. Let's
            first see what are the movie release years !
<br><br>
""", unsafe_allow_html=True)
# Interactive Plot: Movie Release Years  

fig1 = plot_app.plot_movie_release_years(movies)
st.plotly_chart(fig1)  

st.markdown("""
<div style="font-size:18px; text-align:center;">
Before diving into what makes movies successful, we checked how the number of movies with box office data has changed over time. 
In the early years, before the 1930s, there’s barely any data, it was just the start of the industry. 
By the mid-1900s, things picked up as Hollywood grew. In the 2000s, movies exploded, thanks to global markets and big franchises 
like Harry Potter. The dip after 2013 is probably just missing data for newer films. 
Now that we’ve seen this growth, it’s time to figure out what actually makes a movie a hit—plot, genre, or something else?
<br><br>
""", unsafe_allow_html=True)


# Interactive Plot: Total Box Office Revenue by Year  
st.markdown("""
    <style>
    .text-viridis-light {
        background: linear-gradient(135deg, #5e4fa2 0%, #3288bd 50%, #66c2a5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 24px; /* Adjust the font size as needed */
        font-family: 'Cursive', sans-serif;
    }
    </style>
    <div style="text-align:center;">
        <span class="text-viridis-light">Now, what are the box office revenues by years?</span>
    </div>
""", unsafe_allow_html=True)


fig2 = plot_app.plot_box_office_revenue_by_year(movies)
st.plotly_chart(fig2)  

st.markdown("""
<div style="font-size:18px; text-align:center;">
After looking at how the number of movies changed, we now check how much money movies have made over the years. 
Revenues stayed pretty low until the 1990s, but then they shot up, peaking in the 2010s.  
This boom lines up with the rise of huge franchises like Harry Potter and The Avengers and the growth of global 
audiences.
The dip after 2019 is probably just missing data for newer movies. This trend makes us wonder—what’s really driving 
this massive growth? Is it the story, the genre, or something else? Let’s keep digging.
<br><br>
""", unsafe_allow_html=True)


# Interactive Plot : Movies countries  
st.markdown("""
    <style>
    .text-viridis-light {
        background: linear-gradient(135deg, #5e4fa2 0%, #3288bd 50%, #66c2a5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 24px; /* Adjust the font size as needed */
        font-family: 'Cursive', sans-serif;
    }
    </style>
    <div style="text-align:center;">
        <span class="text-viridis-light">Now, what is the distribution of movies by countries ?</span>
    </div>
""", unsafe_allow_html=True)
  

fig3 = plot_app.plot_top_countries(movies)
st.plotly_chart(fig3)  

st.markdown("""
<div style="font-size:18px; text-align:center;">
Next, we looked at where most movies come from, and it’s no surprise, the United States dominates 
by a huge margin, producing more films than the next few countries combined. The UK, France, and 
Germany follow, showing the influence of Europe on global cinema.
It’s interesting to see countries like South Korea and Japan on the list, 
highlighting the rise of Asian cinema, especially with the global success of Korean and Japanese 
films in recent years. This gives us a sense of how different countries contribute to the movie industry 
and sets the stage to explore how these contributions might relate to box office success.
<br><br>
""", unsafe_allow_html=True)


# Interactive Plot: Language Distribution  

st.markdown("""
    <style>
    .text-viridis-light {
        background: linear-gradient(135deg, #5e4fa2 0%, #3288bd 50%, #66c2a5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 24px; /* Adjust the font size as needed */
        font-family: 'Cursive', sans-serif;
    }
    </style>
    <div style="text-align:center;">
        <span class="text-viridis-light">Now, what is the language distribution in movies ?</span>
    </div>
""", unsafe_allow_html=True)  


fig4 = plot_app.plot_language_distribution(movies)
st.plotly_chart(fig4)  

st.markdown("""
<div style="font-size:18px; text-align:center;">
Movies started out mostly in English, but over time, other languages like French, Spanish, 
and German started showing up more. Recently, languages like Korean and Japanese have grown a lot, 
showing how global the movie industry has become.

<br><br>
""", unsafe_allow_html=True)  

st.markdown("""
    <style>
    .text-viridis-light {
        background: linear-gradient(135deg, #5e4fa2 0%, #3288bd 50%, #66c2a5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 24px; /* Adjust the font size as needed */
        font-family: 'Cursive', sans-serif;
    }
    </style>
    <div style="text-align:center;">
        <span class="text-viridis-light">Let's explore the runtime and release year distributions!</span>
    </div>
""", unsafe_allow_html=True)



col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    fig5 = plot_app.plot_runtime_and_release_year_distributions(movies)
    st.plotly_chart(fig5, use_container_width=True) 

st.markdown("""
<div style="font-size:18px; text-align:center;">
Most movies came out after 1980, which fits with what we saw earlier about box office growth. 
On average, movies are about 1 hour and 40 minutes long, though there are a few really short ones 
and some that go way over. Now, let's focus on how genres can be linked to commercial successes.
<br><br>
""", unsafe_allow_html=True)

st.markdown("""
<div style="font-size:18px; text-align:center;">
    <em>Before that, don't forget to grab some popcorn (sweet or salty ?)</em>
</div> 
""", unsafe_allow_html=True)  

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("images_datastory/popcorn_image.jpg", use_container_width=True)  

st.markdown("""
<div style="font-size:18px; text-align:center;">
    <em>Did you know? The ideal temperature for creating popcorn is 180 degrees Celsius.</em>
</div> 
""", unsafe_allow_html=True)


st.markdown("---")    









# ===================== SECTION 5: How genres are related to box offices revenues and commercial success =====================  

st.markdown("""
    <style>
    .title-viridis-light {
        background: linear-gradient(135deg, #3b528b 0%, #21918c 50%, #27ad81 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 44px; /* Adjust the font size as needed */
    }
    </style>
    <div style="text-align:center;">
        <h2 id="genres-revenues" class="title-viridis-light">How genres are related to box offices revenues and commercial success</h2>
    </div>
""", unsafe_allow_html=True)  

st.markdown("""
<div style="font-size:18px; text-align:center;">
You can select here a year range to see what are the top genres across the years :   
 <br><br>
""", unsafe_allow_html=True)

with st.expander("Filters for Genre Analysis"):
    genre_year_range = st.slider("Year Range for Genre Analysis", 1900, 2020, (1980, 2000))

genre_exploded, mean_revenues = mod.return_processed_genre_df(movies, genre_year_range)

fig6 = plot_app.plot_genre_distribution(genre_exploded)
st.plotly_chart(fig6, use_container_width=True)

st.markdown("""
<div style="font-size:18px; text-align:center;">
We notice here, in terms of frequency in percentage, Drama movies are the most distributed ones, 
followed by comedy movies and thrillers. We display here only the 10 most distributed ones in the 
processed dataset. We now have a question, which genres are generating the highest revenues ? 
This may be an excellent question for a filmmakers, we want a movie to generate money, right ? 
<br><br> 
""", unsafe_allow_html=True) 


#Use same filter
fig7 = plot_app.plot_genre_revenue(mean_revenues)
st.plotly_chart(fig7, use_container_width=True)  

st.markdown("""
<div style="font-size:18px; text-align:center;">
In terms of mean revenues, Movies About Gladiators and Humor are the ones generating the highest revenues ! But here, we don't have 
any information about the budget, nor inflation !  
<br><br> 
""", unsafe_allow_html=True) 


st.markdown("""
    <style>
    .title-viridis-light-small {
        background: linear-gradient(135deg, #5e4fa2 0%, #3288bd 50%, #66c2a5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 24px; /* Keep the font size as needed */
    }
    </style>
    <div style="text-align:center;">
        <h4 class="title-viridis-light-small">💰 Adjusting Box Office Revenues for Inflation</h4>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    .centered-text {
        font-size: 18px;
        text-align: center;
    }
    .centered-list {
        list-style-type: none;
        padding: 0;
        text-align: center;
        font-size: 18px;
    }
    .centered-list li {
        margin: 8px 0;
    }
    </style>
    <div class="centered-text">
        To make fair comparisons between movies released in different eras, we need to account for inflation. 
        A movie making $1 million in 1980 is very different from making $1 million today! So we adjusted the revenues : we chose 2023 as our reference point, 
            used Consumer Price Index (CPI) data to track inflation over time, and applied a formula to normalize revenues.
    </div>
""", unsafe_allow_html=True)

st.latex(r"\text{Adjusted Revenue} = \text{Original Revenue} \times \frac{\text{CPI}_{2023}}{\text{CPI}_{\text{Movie Year}}}")

st.markdown("""
<div style="font-size:18px; text-align:center;">
This adjustment helps us to compare movies across different decades fairly, 
understand true financial impact in today's terms and make more accurate assessments of commercial success
</div>
<br>
""", unsafe_allow_html=True)  

 
st.markdown("""
    <style>
    .title-viridis-light-small {
        background: linear-gradient(135deg, #5e4fa2 0%, #3288bd 50%, #66c2a5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 24px; /* Keep the font size as needed */
    }
    </style>
    <div style="text-align:center;">
        <h4 class="title-viridis-light-small">💰 Inflation-Adjusted Revenue Analysis</h4>
    </div>
""", unsafe_allow_html=True)  

st.markdown("""
<div style="font-size:18px; text-align:center;">
You can select here a year range and genres to see how the revenues are evolving across the years.
 <br><br>
""", unsafe_allow_html=True)


# Add filters in an expander
with st.expander("Adjust Analysis Parameters"):
    col1, col2 = st.columns(2)
    
    with col1:
        year_range = st.slider(
            "Select Year Range",
            min_value=1920,
            max_value=2023,
            value=(1980, 2020),
            key="inflation_year_range"
        )
    
    with col2:
        # Get unique genres from the data
        all_genres = sorted(list(set([genre for genres in movies['movie_genres'] for genre in genres])))
        selected_genres = st.multiselect(
            "Filter by Genres",
            options=all_genres,
            default=None,
            key="inflation_genres"
        )
    
    metric = st.radio(
        "Select Revenue Metric",
        options=['Mean', 'Sum', 'Max'],
        horizontal=True
    )

# Process the data with filters
df_inflation = mod.load_processed_inflation()
revenue_data = mod.process_inflation_data(
    movies,
    df_inflation,
    year_range=year_range,
    selected_genres=selected_genres if selected_genres else None
)

# Display the plot
fig = plot_app.plot_inflation_comparison(revenue_data, metric=metric)
st.plotly_chart(fig, use_container_width=True)  

st.markdown("""
    <style>
    .centered-text {
        font-size: 18px;
        text-align: center;
    }
    .centered-list {
        list-style-type: none;
        padding: 0;
        text-align: center;
        font-size: 18px;
    }
    .centered-list li {
        margin: 8px 0;
    }
    </style>
    <div class="centered-text">
        What really makes a movie successful, big numbers overall or a few standout hits? The total revenue from the 1990s and 2000s looks huge, but that’s mostly because so many movies were made during those years. 
        More movies mean higher totals, but does that really show success?
        <br><br>
        To get a better picture, we look at <strong>mean</strong> and <strong>max revenue</strong>:
    </div>
    <ul class="centered-list">
        <li><strong>Mean Revenue</strong> gives us the average, showing how movies performed overall each year without being overwhelmed by how many were released.</li>
        <li><strong>Max Revenue</strong> focuses on the biggest hits—the movies that smashed records and left their mark.</li>
    </ul>
    <div class="centered-text">
        If we stick with total revenue, we miss the real stars and trends. By focusing on <strong>averages and standouts</strong>, we can better figure out what makes a movie truly successful.
        The graphs above show how movie revenues have changed over time, both in original and inflation-adjusted terms. This comparison helps us to understand how the value of movie earnings has changed over time and the true financial impact of movies when accounting for inflation. Which eras were most successful in today's monetary terms.  
        <br><br>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    .title-viridis-light-small {
        background: linear-gradient(135deg, #5e4fa2 0%, #3288bd 50%, #66c2a5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 24px; /* Keep the font size as needed */
    }
    </style>
    <div style="text-align:center;">
        <h4 class="title-viridis-light-small">📊 Revenue Analysis with Genres</h4>
    </div>
""", unsafe_allow_html=True)


st.markdown("""
<div style="font-size:18px; text-align:center;">
Let's analyse how genres are related to box office revenues. You can select genre distribution influence, genre trends,  
revenue heatmap and bubble chart. 
</div>
""", unsafe_allow_html=True)

# Create tabs for different visualizations
tab1, tab2 = st.tabs([
    "Revenue Distribution", 
    "Genre Trends"
])

# Add custom CSS to center the tabs and make them bigger
st.markdown("""
    <style>
    .stTabs [role="tablist"] {
        justify-content: center;
    }
    .stTabs [role="tab"] {
        font-size: 24px; /* Increase font size */
        padding: 15px 30px; /* Increase padding */
    }
    </style>
""", unsafe_allow_html=True)

with tab1:
    st.markdown("### Revenue Distribution by Decade")
    use_adjusted = st.checkbox("Show Inflation Adjusted Values", value=True, key="dist_adjusted")
    dist_fig = plot_app.plot_revenue_distribution(revenue_data, adjusted=use_adjusted)
    st.plotly_chart(dist_fig, use_container_width=True)  

    st.markdown("""
<div style="font-size:18px; text-align:center;">
This visualization shows how movie revenues are distributed across decades, 
    helping identify shifts in the industry's financial landscape.
</div>
""", unsafe_allow_html=True)


with tab2:
    st.markdown("### Revenue Trends by Genre")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_genres = st.multiselect(
            "Select Genres to Compare",
            options=all_genres,
            default=all_genres[:5],
            key="trend_genres"
        )
    
    with col2:
        use_adjusted = st.checkbox("Show Inflation Adjusted Values", value=True, key="trend_adjusted")
    
    if selected_genres:
        revenue_data = mod.process_inflation_data(
            movies,
            df_inflation,
            year_range=year_range,
            selected_genres=selected_genres if selected_genres else None
        )
        trends_fig = plot_app.plot_genre_revenue_trends(
            revenue_data, 
            selected_genres, 
            adjusted=use_adjusted
        )
        st.plotly_chart(trends_fig, use_container_width=True)
    else:
        st.warning("Please select at least one genre to display trends.")
# Add overall insights  

st.markdown("""
    <style>
    .title-viridis-light-small {
        background: linear-gradient(135deg, #5e4fa2 0%, #3288bd 50%, #66c2a5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 24px; /* Keep the font size as needed */
    }
    </style>
    <div style="text-align:center;">
        <h4 class="title-viridis-light-small">🔍 Key Insights</h4>
    </div>
""", unsafe_allow_html=True)   

st.markdown("""
<div style="font-size:18px; text-align:center;">
There are Temporal Patterns, the distribution analysis shows how revenue patterns have evolved over decades. 
In the inflation-adjusted view, Family Films, Fantasy and Dramas rise to the top, showcasing their enduring 
appeal and the lasting power of their stories. On the other hand, the non-adjusted view highlights the dominance 
of blockbuster genres like Adventure, Action, and Fantasy, driven by modern budgets and global excitement.
<br><br>
</div>
""", unsafe_allow_html=True)  

st.markdown("""
<div style="font-size:18px; text-align:center;">
There are also Genre Performance, certain genres consistently outperform others when adjusted for inflation, such as Family Films.  
<br><br> 
</div>
""", unsafe_allow_html=True)  

st.markdown("""
    <style>
    .title-viridis-light-medium {
        background: linear-gradient(135deg, #5e4fa2 0%, #3288bd 50%, #66c2a5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 28px; /* Adjust the font size as needed */
    }
    </style>
    <div style="text-align:center;">
        <h3 class="title-viridis-light-medium">What about the mean and median box office revenues?</h3>
    </div>
""", unsafe_allow_html=True)  

st.markdown("""
<div style="font-size:18px; text-align:center;">
Same pattern is observed if one considers inflation or not, only the scale for revenues in USD 
is changing here. But the mean is really sensible to outliers, some successful movies such as Avatar 
or Titanic may do have a strong impact on the mean. We need an other tool to do this analysis, something more robust than the mean : the median !
</div>
""", unsafe_allow_html=True)  


st.markdown("""
<div style="font-size:18px; text-align:center;">
Analyzing the median, Family Film genre is the one producing the highest median revenues, 
followed by Fantasy and Adventure movies. The previous order is similar to the mean revenues in that case. 
The median gives much more consistent result for Mystery movies, as the latter were negative with mean profits, due to outliers.
<br><br>  
</div>
""", unsafe_allow_html=True)  

st.markdown("""
    <style>
    .title-viridis-light-medium {
        background: linear-gradient(135deg, #5e4fa2 0%, #3288bd 50%, #66c2a5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 28px; /* Adjust the font size as needed */
    }
    </style>
    <div style="text-align:center;">
        <h3 class="title-viridis-light-medium">Commercial success</h3>
    </div>
""", unsafe_allow_html=True)  

st.markdown("""
<div style="font-size:18px; text-align:center;">
Let's define a value that wil be useful. the profitability ratio, defined as the ratio of the profit to the budget.  
<br><br>  
</div>
""", unsafe_allow_html=True)  

st.markdown("""
<div style="font-size:18px; text-align:center;">
Let's take an example: <span style="color:blue;">Avatar</span>.
</div>
""", unsafe_allow_html=True)  

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("images_datastory/avatar.jpeg", caption="Avatar Movie", use_container_width=True, width=50)


st.markdown("""
<div style="font-size:18px; text-align:center;">
The movie Avatar has a budget of 237 million USD and a revenue of 2.8 billion USD.  
<br><br> 
</div>  
""", unsafe_allow_html=True)  

st.markdown("""
<div style="font-size:18px; text-align:center;">
On the other hand, in terms of profitability ratio, Paranormal Activity has a profitability ratio of 12'889 ! As an example 
also, As an example, the movie <a href="https://en.wikipedia.org/wiki/The_Last_Broadcast_(film)">The Last Broadcast</a> is interesting, it only costed 900 dollars to be produced 
and it generated 4 millions of dollars as worldwide revenue. The film was made on a budget of 900 dollars, 
and edited on a desktop computer using Adobe Premiere 4.2. 600 dollars were allocated for 
production, while 240 dollars were utilized for digital video stock, and twenty hours of tape for 
12 dollars each. 
</div>
""", unsafe_allow_html=True)  


st.markdown("---")  










# ===================== SECTION 6: Plot Structure Analysis with Clustering & LLM Classification =====================


st.markdown("""
    <style>
    .title-viridis-light {
        background: linear-gradient(135deg, #3b528b 0%, #21918c 50%, #27ad81 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 44px; /* Adjust the font size as needed */
    }
    </style>
    <div style="text-align:center;">
        <h2 id="plot-structure" class="title-viridis-light">Plot Structure Analysis with Clustering & LLM Classification</h2>
    </div>
""", unsafe_allow_html=True) 

st.markdown("""
<div style="font-size:18px; text-align:center;">
This section analyzes the underlying plot structures of movies using two approaches: Unsupervised clustering to discover emergent patterns and LLM-based classification into predefined categories.  
Firstly, let's have a look to Clustering analysis :  
<br><br>
</div>
""", unsafe_allow_html=True) 

st.markdown("""
    <style>
    .title-viridis-light-medium {
        background: linear-gradient(135deg, #5e4fa2 0%, #3288bd 50%, #66c2a5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 28px; /* Adjust the font size as needed */
    }
    </style>
    <div style="text-align:center;">
        <h3 class="title-viridis-light-medium">Clustering</h3>
    </div>
""", unsafe_allow_html=True)  

# Load both dataframes
movies_bo = pd.read_csv('data/processed/movies_summary_BO.csv')
movies_classified = pd.read_csv('data/processed/movies_with_classifications.csv')

# Merge the dataframes on common identifiers
movies = pd.merge(
    movies_bo,
    movies_classified[['wikipedia_movie_id', 'plot_structure', 'plot_structure_20']],
    on='wikipedia_movie_id',
    how='left'
)  

# Perform clustering
with st.spinner("Performing text clustering..."):
    clustering_results = mod.perform_text_clustering(movies['plot_summary'])

# Calculate silhouette scores
with st.spinner("Calculating silhouette scores..."):
    silhouette_scores = mod.calculate_silhouette_scores(
        clustering_results['matrix']
    )

st.markdown("""
<div style="font-size:18px; text-align:center;">
Plot summaries are transformed into a numerical format for clustering by applying TF-IDF (Term Frequency-Inverse Document Frequency) 
vectorization. TF-IDF highlights important words in each summary by reducing the weight 
of common terms and increasing the importance of unique terms.
<br><br>
</div>
""", unsafe_allow_html=True)



# Plot silhouette analysis  

st.markdown("""
    <style>
    .title-viridis-light-small {
        background: linear-gradient(135deg, #5e4fa2 0%, #3288bd 50%, #66c2a5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 24px; /* Keep the font size as needed */
    }
    </style>
    <div style="text-align:center;">
        <h4 class="title-viridis-light-small">💰 Silhouette Analysis</h4>
    </div>
""", unsafe_allow_html=True)

fig_silhouette = plot_app.plot_silhouette_analysis(silhouette_scores)
st.plotly_chart(fig_silhouette, use_container_width=True)


st.markdown("""
<div style="font-size:18px; text-align:center; line-height:1.6;">
<strong>KMeans Clustering</strong> is employed to group plot summaries based on their TF-IDF representations. This technique helps us uncover distinct plot structure patterns by clustering similar summaries together.

<ul style="text-align:left;">
    <li>The clustering labels are then added to the dataset, enabling a deeper analysis of plot structure patterns within each identified cluster.</li>
    <li>To determine the optimal number of clusters, we plotted the silhouette score for cluster values ranging from 5 to 20.</li>
    <li>Ideally, the optimal number of clusters is indicated by a <strong>peak in the silhouette score</strong>. However, in our plot, the silhouette score continually increases as the number of clusters increases.</li>
    <li>Given these results, we have chosen to proceed with <strong>15 clusters</strong>. This number strikes a balance between interpretability and granularity, allowing us to capture a diverse range of plot structures without creating an excessive number of small, indistinct clusters.</li>
</ul>
</div>
""", unsafe_allow_html=True)  

# Show clustering visualizations
st.markdown("""
    <style>
    .title-viridis-light-medium {
        background: linear-gradient(135deg, #5e4fa2 0%, #3288bd 50%, #66c2a5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 28px; /* Adjust the font size as needed */
    }
    </style>
    <div style="text-align:center;">
        <h3 class="title-viridis-light-medium">Clustering Visualizations</h3>
    </div>
""", unsafe_allow_html=True)  

col1, col2 = st.columns(2)

with col1:
    st.write("T-SNE Visualization")
    fig_tsne = plot_app.plot_clustering_visualization(
        clustering_results['matrix'],
        clustering_results['labels'],
        'tsne'
    )
    st.plotly_chart(fig_tsne)

with col2:
    st.write("PCA Visualization")
    fig_pca = plot_app.plot_clustering_visualization(
        clustering_results['matrix'],
        clustering_results['labels'],
        'pca'
    )
    st.plotly_chart(fig_pca)

# Show top terms per cluster
st.subheader("Top Terms per Cluster")
cols = st.columns(3)
for i, terms in enumerate(clustering_results['top_terms']):
    col_idx = i % 3
    with cols[col_idx]:
        st.write(f"**Cluster {i+1}:**")
        st.write(", ".join(terms))

st.markdown("""
<div style="font-size:18px; text-align:center;">
Each cluster reveals distinct themes and settings. While this analysis helps to identify common elements within each group, we are not fully satisfied with this approach** as it appears to capture genre and themes more than specific plot structures.
Since our goal is to identify different types of plot structures, clustering based solely on keywords may lack the depth needed to capture narrative progression and plot dynamics. Consequently, we explore alternative methods, such as leveraging large 
language models or deeper natural language processing techniques, to classify plot structures more accurately.
<br><br>
</div>
""", unsafe_allow_html=True) 


st.markdown("""
    <style>
    .title-viridis-light-medium {
        background: linear-gradient(135deg, #5e4fa2 0%, #3288bd 50%, #66c2a5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 28px; /* Adjust the font size as needed */
    }
    </style>
    <div style="text-align:center;">
        <h3 class="title-viridis-light-medium">LLM Classification</h3>
    </div>
""", unsafe_allow_html=True)  

st.markdown("""
<div style="font-size:18px; text-align:center;">
To generate the summarized version of the plot summaries, we are using a pre-trained transformer model (`facebook/bart-large-cnn`). Here's an overview of what this script achieves:

<ul style="text-align:left;">
    <li><strong>Text Preprocessing</strong>:
        <ul>
            <li>Cleans and normalizes plot summaries by removing unnecessary characters and whitespace.</li>
            <li>Splits long texts into manageable chunks at sentence boundaries to fit the model's input token limit (1024 tokens).</li>
        </ul>
    </li>
    <li><strong>Summarization</strong>:
        <ul>
            <li>Processes each chunk of text through the model to generate intermediate summaries.</li>
            <li>Combines these intermediate summaries and processes the result to create a final, concise summary for each plot.</li>
            <li>When the summary is short enough, we keep it as it is without using the LLM.</li>
        </ul>
    </li>
    <li><strong>Batch Processing</strong>:
        <ul>
            <li>Summarizes plot summaries in batches.</li>
        </ul>
    </li>
</ul>
</div>
""", unsafe_allow_html=True)  


st.markdown("""
<div style="font-size:18px; text-align:center;">
After having our summarized plot summaries, we create our pipeline for classification, and classify our summarized plot summaries into plot structure categories. We use the pre-trained transformer model (`facebook/bart-large-mnli`) for zero-shot classification of movie plot summaries, since the model fits perfectly our task.

</div>
""", unsafe_allow_html=True)  

st.markdown("""
<div style="font-size:18px; text-align:center;">
To create the list of plot structure categories, we synthesized from several narrative frameworks:
<ul style="text-align:left;">
    <li><em>The Seven Basic Plots</em> by Christopher Booker: <a href="https://www.campfirewriting.com/learn/narrative-structure" target="_blank">campfirewriting.com</a></li>
    <li><em>The Hero's Journey</em> by Joseph Campbell: <a href="https://www.campfirewriting.com/learn/narrative-structure" target="_blank">campfirewriting.com</a></li>
    <li><em>Freytag's Pyramid</em>: <a href="https://blog.reedsy.com/guide/story-structure/" target="_blank">blog.reedsy.com</a></li>
</ul> 
</div>
""", unsafe_allow_html=True)  

st.markdown("""
<div style="font-size:18px; text-align:center;">
Additionally, since we use zero-shot classification, we adapted the categories to be distinct and descriptive enough for the model to differentiate between them.  
It gives use 15 candidate categories. Finally, we tried the same zero-shot classification with different candidate categories.
The goal is to capture a broader range of narrative structures. We tried with 23 different categories,  
but as the results were not satisfying, we decided to keep the 15 categories for this part.   
<br><br>
</div>
""", unsafe_allow_html=True)  

st.markdown("""
    <style>
    .title-viridis-light-medium {
        background: linear-gradient(135deg, #5e4fa2 0%, #3288bd 50%, #66c2a5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 28px; /* Adjust the font size as needed */
    }
    </style>
    <div style="text-align:center;">
        <h3 class="title-viridis-light-medium">Classification and movie revenues and profit based on plot structures</h3>
    </div>
""", unsafe_allow_html=True)   

# Show distribution of plot structures (now using merged data)
distribution_data = mod.analyze_plot_structure_distribution(movies)
fig_distribution = plot_app.plot_plot_structure_distribution(distribution_data)
st.plotly_chart(fig_distribution)

st.markdown("""
<div style="font-size:18px; text-align:center;">

The plot structures "Conflict with Supernatural or Unknown Forces", "Comedy of Errors or Misadventure" 
and "Hero's Journey and "Transformation" are the most represented ones. Drama Comedy and Action are the most represented genres, 
as here, 887 Drama movies are categorized in "Hero's Journey and Transformation". 
</div>
""", unsafe_allow_html=True)  
 

st.markdown("""
    <style>
    .title-viridis-light-medium {
        background: linear-gradient(135deg, #5e4fa2 0%, #3288bd 50%, #66c2a5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 28px; /* Adjust the font size as needed */
    }
    </style>
    <div style="text-align:center;">
        <h3 class="title-viridis-light-medium">Mean Box Office Revenue by Plot Structure</h3>
    </div>
""", unsafe_allow_html=True)  


performance_metrics = mod.analyze_plot_structure_performance(movies)
st.plotly_chart(plot_app.plot_structure_performance(performance_metrics))

st.markdown("""
<div style="font-size:18px; text-align:center;">
The plot structure "Quest for Vengeance or Justice" has the highest mean box office revenues. What about profits ?  
<br><br>    
</div>
""", unsafe_allow_html=True)  

st.markdown("""
    <style>
    .citation {
        font-style: italic;
        text-align: center;
        font-size: 18px;
        margin-top: 20px;
    }
    </style>
    <div class="citation">
        Homer Simpson is rolling in money, a director wants the same things with a movie!
    </div>
""", unsafe_allow_html=True)  

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    components.html("""
        <div class="tenor-gif-embed" data-postid="3555030" data-share-method="host" data-aspect-ratio="1.38889" data-width="100%">
            <a href="https://tenor.com/view/money-rich-cash-rolling-in-the-dough-the-s-impsons-gif-3555030">Money GIF</a> from 
            <a href="https://tenor.com/search/money-gifs">Money GIFs</a>
        </div> 
        <script type="text/javascript" async src="https://tenor.com/embed.js"></script>
    """, height=300)


# Show profit analysis  

st.markdown("""
    <style>
    .title-viridis-light-medium {
        background: linear-gradient(135deg, #5e4fa2 0%, #3288bd 50%, #66c2a5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 28px; /* Adjust the font size as needed */
    }
    </style>
    <div style="text-align:center;">
        <h3 class="title-viridis-light-medium">Profit Analysis by Plot Structure</h3>
    </div>
""", unsafe_allow_html=True)  
profit_metrics = mod.analyze_plot_structure_profit(movies)
st.plotly_chart(plot_app.plot_structure_profit(profit_metrics))

st.markdown("""
<div style="font-size:18px; text-align:center;">
The plot structure "Quest for Vengeance or Justice" has the highest median box office profits ! 
</div>
""", unsafe_allow_html=True)



st.markdown("---")


# SECTION 7 : Predictive Modelling and Commercial Success Analysis
# In your commercial success analysis section
st.markdown("""
    <style>
    .title-viridis-light {
        background: linear-gradient(135deg, #3b528b 0%, #21918c 50%, #27ad81 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 44px; /* Adjust the font size as needed */
    }
    </style>
    <div style="text-align:center;">
        <h2 id="predictive-modeling" class="title-viridis-light">💰 Commercial Success Analysis and Predictive Modelling</h2>
    </div>
""", unsafe_allow_html=True) 

st.markdown("""
<div style="font-size:18px; text-align:center;">

To conclude with the data we have from now, we want to know if we can predict that a movie will be 
profitable based on its plot structure. Two predictions are done, one using only base features and 
an other by adding also the plot structure that we classified, to check if it enhances the probability 
of having a profitable movie !
</div>
""", unsafe_allow_html=True)  

# Process the data with inflation adjustment
df_inflation = mod.load_processed_inflation()
movies_with_profit = mod.calculate_profit_metrics(movies, df_inflation)

# Create tabs for different analyses
profit_tabs = st.tabs([
    "Top Profitable Movies",
    "Budget-Profit Relationship",
    "ROI Analysis"
])

with profit_tabs[0]:
    use_adjusted = True
    col1, col2 = st.columns(2)
    
    profit_col = 'adjusted_profit' if use_adjusted else 'profit'
    ratio_col = 'adjusted_profitability_ratio' if use_adjusted else 'profitability_ratio'
    
    with col1:
        st.markdown("### Top Movies by Profit")
        top_profit = mod.get_top_profitable_movies(movies_with_profit, by="profit")
        fig_profit = plot_app.plot_top_profitable_movies(top_profit, "profit")
        st.plotly_chart(fig_profit, use_container_width=True)
    
    with col2:
        st.markdown("### Top Movies by ROI")
        top_roi = mod.get_top_profitable_movies(movies_with_profit, by='profitability_ratio')
        fig_roi = plot_app.plot_top_profitable_movies(top_roi, 'profitability_ratio')
        st.plotly_chart(fig_roi, use_container_width=True)

with profit_tabs[1]:
    st.markdown("### Budget vs Profit Relationship")
    fig_budget = plot_app.plot_budget_profit_relationship(movies_with_profit)
    st.plotly_chart(fig_budget, use_container_width=True)
    
    # Add correlation statistics
    col1, col2 = st.columns(2)
    with col1:
        pearson = movies_with_profit['budget'].corr(movies_with_profit['profit'])
        st.metric("Pearson Correlation", f"{pearson:.3f}")
    with col2:
        spearman = movies_with_profit['budget'].corr(
            movies_with_profit['profit'], 
            method='spearman'
        )
        st.metric("Spearman Correlation", f"{spearman:.3f}")

with profit_tabs[2]:
    st.markdown("### Return on Investment Analysis")
    
    # Create budget bins and calculate ROI statistics
    movies_binned, roi_stats, bin_labels = mod.create_budget_bins(movies_with_profit)
    
    # Plot ROI by budget range
    fig_roi = plot_app.plot_roi_by_budget(roi_stats)
    st.plotly_chart(fig_roi, use_container_width=True)
    
    # Add statistical test results
    with st.expander("Statistical Test Results"):
        from scipy.stats import f_oneway, kruskal
        
        roi_groups = [
            movies_binned[movies_binned['budget_bins'] == label]['profitability_ratio'] 
            for label in bin_labels
        ]
        
        anova = f_oneway(*roi_groups)
        kruskal_test = kruskal(*roi_groups)
        
        st.write("ANOVA Test Results:")
        st.write(f"- F-statistic: {anova.statistic:.2f}")
        st.write(f"- p-value: {anova.pvalue:.2e}")
        
        st.write("\nKruskal-Wallis Test Results:")
        st.write(f"- H-statistic: {kruskal_test.statistic:.2f}")
        st.write(f"- p-value: {kruskal_test.pvalue:.2e}")
 
 
 ### Using movie plots and genres as predictors for financial success
st.markdown("""
<div style="font-size:11px; text-align:center; line-height:1.6;">
<h2 style="color:#2E86C1;">Observations:</h2>
<ul style="list-style-type:disc; text-align:left; display:inline-block;">
    <li>Significant drop in R-squared value from training to testing data indicates poor generalization.</li>
    <li>Plot structures only model has the lowest training R-squared (0.218) but slightly better predictive performance (0.1908).</li>
    <li>Adding plot clusters provides the highest training fit (0.384) and marginally improves predictive R-squared (0.1803).</li>
</ul>

<h2 style="color:#2E86C1;">P-values Analysis:</h2>
<ul style="list-style-type:disc; text-align:left; display:inline-block;">
    <li><strong>Plot structures:</strong> None of the 15 plot structures have significant p-values (&lt;0.05), indicating they do not significantly explain the variance in adjusted_profit.</li>
    <li><strong>Movie genres:</strong> Some significant p-values (&lt;0.05) are observed, likely due to chance given the large number of genres.</li>
    <li><strong>Plot clusters:</strong> 2 significant p-values (&lt;0.05) suggest certain plot clusters capture meaningful patterns related to financial success, aligning with the slightly better training R-squared value.</li>
</ul>
</div>   
""", unsafe_allow_html=True)  



### Budget as a more straightforward predictor for financial success
st.markdown("""
<div style="font-size:14px; text-align:center;">

Given that the plot structures obtained and movie genres individually act as a poor predictors
for a movie's financial success, we decide to look at budget (p-value = 0.000 in all models) 
as a more straightforward and potentially significant predictor. We apply a feature augmentation 
technique to improve the budget's predictive significance.

</div>
""", unsafe_allow_html=True)  




st.markdown("---")

#######

# SECTION 8 : Conclusion  

st.markdown("""
    <style>
    .title-viridis-light {
        background: linear-gradient(135deg, #3b528b 0%, #21918c 50%, #27ad81 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 44px; /* Adjust the font size as needed */
    }
    </style>
    <div style="text-align:center;">
        <h2 id="conclusion" class="title-viridis-light">Conclusion</h2>
    </div>
""", unsafe_allow_html=True) 

st.markdown("""
    <style>
    .text-viridis-light {
        background: linear-gradient(135deg, #5e4fa2 0%, #3288bd 50%, #66c2a5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 18px;
        text-align: center;
        line-height: 1.6;
    }
    .text-white {
        color: white;
        font-size: 18px;
        text-align: center;
        line-height: 1.6;
    }
    .text-viridis-light h2 {
        font-size: 24px;
        color: #2E86C1;
    }
    .text-viridis-light ul {
        list-style-type: disc;
        text-align: left;
        display: inline-block;
        margin: 0 auto;
    }
    </style>
    <div class="text-white">
        In this project, we are decoding the blueprint of a blockbuster by analyzing plot structures and their impact on box office success. We have done analysis involving data scraping, preprocessing, exploratory data analysis, natural language processing, and predictive modeling.
    </div>
    <div class="text-viridis-light">
        <br><br>
        <h2>What we found</h2>
        <ul>
            <li><strong>Impact of Plot Structures on Box Office Revenue</strong>: Our analysis revealed that certain plot structures, such as the hero's journey, tend to be more successful at the box office. We can suppose such plot structures to resonate well with audiences and people, leading logically to more entries to the cinemas and therefore higher revenues.</li>
            <li><strong>Genre and Plot Structure Correlation</strong>: We found that specific genres are more likely to feature certain plot structures. For example, drama movies often follow hero's journey.</li>
            <li><strong>Temporal Trends in Plot Structures</strong>: The popularity of different plot structures has evolved over time. While classic structures like the hero's journey remain popular, newer structures are emerging and gaining traction in recent years.</li>
            <li><strong>Inflation Adjustment</strong>: Adjusting revenue and budget data for inflation provided a more accurate comparison across different time periods. This adjustment was crucial for understanding the true financial success of movies released in different eras, while giving the money a similar weight for every year, in order to make consistent comparisons between movies budgets, revenues and profits.</li>
            <li><strong>Factors Contributing to Commercial Success</strong>: Besides plot structures, other factors such as budget, release timing, and marketing efforts play a significant role in a movie's commercial success. Our predictive models showed that incorporating plot structure information does not significantly enhance the accuracy of predicting a movie's profitability. In fact, other variables may have an impact, such as the actors, the director, a certain period of time where some movies are more successful, following trends for examples. This might be very interesting to study further, while taking historical events and everyday life, in order to see if these events may also have an impact on a movie profitability.</li>
        </ul>
    </div>
""", unsafe_allow_html=True) 


st.markdown("""
    <style>
    .title-viridis-light {
        background: linear-gradient(135deg, #3b528b 0%, #21918c 50%, #27ad81 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 30px; /* Adjust the font size as needed */
    }
    </style>
    <div style="text-align:center;">
        <h2 id="finally" class="title-viridis-light">Finally</h2>
    </div>
""", unsafe_allow_html=True)

## Conclusion

st.markdown("""
<div style="font-size:18px; text-align:center;">

Our findings highlight the importance of storytelling and plot structures in the film industry. While various factors contribute to a movie's success, the plot structure remains a very important element that can significantly influence audience engagement and box office performance. This project showed that even by leveraging natural language processing and machine learning techniques, it's still a bit difficult to quantify and analyze the impact of plot structures on commercial success. It gives overview of which genres and plot structure are in general the most profitable ones, but it's complicated to really have kind of a "secret recipe" to make a movie really successful in terms of profitability.    

This project provides valuable insights for filmmakers, producers, and marketers, helping them make informed decisions about the types of stories that are likely to resonate with audiences and achieve financial success. Future research could further explore the role of other narrative elements, such as character development and dialogue, actors, directors, historical events influence etc. 

Overall, the project underscores the timeless appeal of well-crafted stories and plot structure and their enduring power to captivate audiences and drive box office revenues (and profits).
</div>
""", unsafe_allow_html=True)  



st.markdown("""
    <style>
    .title-viridis-light {
        background: linear-gradient(135deg, #3b528b 0%, #21918c 50%, #27ad81 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 44px; /* Adjust the font size as needed */
    }
    </style>
    <div style="text-align:center;">
        <h2 id="authors" class="title-viridis-light">Authors</h2>
    </div>
""", unsafe_allow_html=True)  

st.markdown("""
<div style="font-size:18px; text-align:center;">
Sven, Anders, Adam, Malak, Arthur. 
</div>
""", unsafe_allow_html=True)




st.markdown("""21""", unsafe_allow_html=True)









