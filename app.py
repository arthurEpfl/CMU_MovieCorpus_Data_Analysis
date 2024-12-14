import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os

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

st.title("🎬 Cinematic Moral Dilemmas - Dark Mode Edition")
st.subheader("Team Adarable")

st.markdown("""
Welcome to our comprehensive exploration of cinematic moral dilemmas. Here, we present a refined dashboard with smaller, dark-themed graphs (with white text) and additional visualizations.
""")  

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("images_datastory/movie_clap.png", use_container_width=True)
st.markdown("""
Hello you
""")

st.markdown("## Table of Contents")
st.markdown("""
- **2. First Explorations**  
- **3. Preprocessing**  
- **4. Exploratory Data Analysis**  
- **5. Genres, Revenues, and Commercial Success**  
- **6. Plot Structure Analysis with Clustering & LLM Classification**  
- **7. Inflation Adjustments**  
- **8. Predictive Modeling**  
""")

st.markdown("---")

# ===================== SECTION 2: FIRST EXPLORATIONS =====================
st.header("2. First Explorations")
st.markdown("""
Initially, we explored raw data and identified missing box office revenues and the need for external data sources.
""")

with st.expander("No Filters Here"):
    st.info("Initial exploration was static. This section is informational only.")

st.markdown("---")

# ===================== SECTION 3: PREPROCESSING =====================
st.header("3. Preprocessing")
st.markdown("""
We cleaned data, merged with IMDb dataset, and retained movies with both box office and plot summaries.
""")

st.markdown("""
**Resulting dataset:** `movies_summary_BO.csv` with ~7,964 movies.
""")

with st.expander("No Filters"):
    st.info("Preprocessing steps are explained; no filters apply.")

st.markdown("---")

# ===================== SECTION 4: EDA =====================
st.header("4. Exploratory Data Analysis")

with st.expander("Filters for EDA"):
    year_range_eda = st.slider("Select Year Range for EDA", 1900, 2020, (1980, 2000))

eda_filtered = movies[(movies['movie_release_date'] >= year_range_eda[0]) & (movies['movie_release_date'] <= year_range_eda[1])]

st.markdown("### 4.1 Movie Release Years (Filtered)")
yearly_counts = eda_filtered['movie_release_date'].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(8,4))
yearly_counts.plot(kind='bar', color='skyblue', ax=ax)
ax.set_title('Count of Movies Per Year')
ax.set_xlabel('Year')
ax.set_ylabel('Count')
plt.xticks(rotation=45)
st.pyplot(fig)

st.markdown("As we focus on selected years, notice how production volume changes.")

st.markdown("### Additional Graph: Runtime Distribution")
fig, ax = plt.subplots(figsize=(8,4))
sns.histplot(eda_filtered['movie_runtime'].dropna(), kde=True, color='cyan', ax=ax)
ax.set_title('Distribution of Movie Runtime')
ax.set_xlabel('Runtime (minutes)')
st.pyplot(fig)

st.markdown("On average, runtimes cluster around 90-110 minutes.")

st.markdown("### Another Additional Graph: Correlation Heatmap")
# Let's pick some numeric columns
numeric_cols = ['movie_box_office_revenue', 'movie_runtime', 'budget', 'rating_score']
num_data = eda_filtered[numeric_cols].dropna()
if not num_data.empty:
    corr = num_data.corr()
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(corr, annot=True, cmap='magma', ax=ax)
    ax.set_title('Correlation Heatmap of Numeric Features')
    st.pyplot(fig)
else:
    st.info("Not enough numeric data for correlation plot in selected filters.")

st.markdown("---")

# ===================== SECTION 5: Genres and Commercial Success =====================
st.header("5. Genres & Commercial Success")

with st.expander("Filters for Genre Analysis"):
    genre_year_range = st.slider("Year Range for Genre Analysis", 1900, 2020, (1980, 2000))

genre_filtered = movies[(movies['movie_release_date'] >= genre_year_range[0]) & (movies['movie_release_date'] <= genre_year_range[1])]
genre_exploded = genre_filtered.explode('movie_genres')

top_15_genres = genre_exploded['movie_genres'].value_counts().head(15)
fig, ax = plt.subplots(figsize=(8,4))
sns.barplot(x=top_15_genres.index, y=top_15_genres.values, palette='cool', ax=ax)
ax.set_title('Top 15 Genres')
ax.set_xlabel('Genre')
ax.set_ylabel('Count')
plt.xticks(rotation=45)
st.pyplot(fig)

st.markdown("""
Drama, Comedy, and Thriller often dominate. But do they yield the highest revenues?
Let's look at revenue vs. rating as an additional graph:
""")

if 'rating_score' in genre_filtered.columns and 'movie_box_office_revenue' in genre_filtered.columns:
    fig, ax = plt.subplots(figsize=(8,4))
    sns.scatterplot(x='rating_score', y='movie_box_office_revenue', data=genre_filtered, color='yellow', s=10, ax=ax)
    ax.set_title('Revenue vs. Rating Score')
    ax.set_xlabel('Rating Score')
    ax.set_ylabel('Revenue')
    plt.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)
else:
    st.info("No rating or revenue data available for these filters.")

st.markdown("---")

# ===================== SECTION 6: Plot Structure Analysis =====================
st.header("6. Plot Structure Analysis")

with st.expander("Filters for Plot Structure"):
    ps_year_range = st.slider("Year Range for Plot Structure Analysis", 1900, 2020, (1980, 2000))

ps_filtered = classified[(classified['movie_release_date'] >= ps_year_range[0]) & (classified['movie_release_date'] <= ps_year_range[1])]

plot_counts = ps_filtered['plot_structure'].value_counts()
fig, ax = plt.subplots(figsize=(8,4))
plot_counts.plot(kind='bar', color='lime', ax=ax)
ax.set_title('Distribution of Plot Structures')
ax.set_xlabel('Plot Structure')
ax.set_ylabel('Count')
plt.xticks(rotation=90)
st.pyplot(fig)

st.markdown("Some plot structures correlate with higher revenues. For example, 'Quest for Vengeance or Justice' often correlates with higher median revenues.")

# Additional Graph: Boxplot of Profit by Plot Structure (if data available)
if 'profit' in ps_filtered.columns and 'plot_structure' in ps_filtered.columns:
    fig, ax = plt.subplots(figsize=(8,4))
    sns.boxplot(x='plot_structure', y='profit', data=ps_filtered, color='orange', ax=ax)
    ax.set_title('Profit Distribution by Plot Structure')
    plt.xticks(rotation=90)
    st.pyplot(fig)
else:
    st.info("Profit or plot_structure not available in selected subset.")

st.markdown("---")

# ===================== SECTION 7: Inflation Adjustments =====================
st.header("7. Inflation Adjustments")

with st.expander("Filters for Inflation Analysis"):
    inflation_year_range = st.slider("Year Range for Inflation Analysis", 1900, 2020, (1980, 2000))

inf_filtered = classified[(classified['movie_release_date'] >= inflation_year_range[0]) & (classified['movie_release_date'] <= inflation_year_range[1])]

if 'adjusted_profit' in inf_filtered.columns:
    fig, ax = plt.subplots(figsize=(8,4))
    sns.histplot(inf_filtered['adjusted_profit'].dropna(), kde=True, color='pink', ax=ax)
    ax.set_title('Distribution of Adjusted Profit')
    ax.set_xlabel('Adjusted Profit ($)')
    st.pyplot(fig)
else:
    st.info("No adjusted profit data available for these filters.")

st.markdown("---")

# ===================== SECTION 8: Predictive Modeling =====================
st.header("8. Predictive Modeling")

st.markdown("""
We tried linear regression to predict profit using genres, plot structures, and budgets.
Budget correlates strongly with profit, while plot structures and genres add limited predictive power.
""")

# Additional Graph: Budget vs Profit (log scale)
if 'budget' in classified.columns and 'profit' in classified.columns:
    fig, ax = plt.subplots(figsize=(8,4))
    ax.scatter(classified['budget'], classified['profit'], s=10, c='red')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Budget vs Profit (Log-Log)')
    ax.set_xlabel('Budget (log scale)')
    ax.set_ylabel('Profit (log scale)')
    plt.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)
else:
    st.info("No budget or profit data available for visualization.")

st.markdown("""
Notice how increasing budgets often lead to higher profits but may not guarantee a high ROI.
""")

st.markdown("## Conclusion")
st.markdown("""
- **Budget** remains a key factor in determining profit.
- **Plot structures** and **genres**, while interesting thematically, provided limited predictive improvement.
- Adjusting for **inflation** allows fairer comparisons across decades.

This analysis highlights the complexity of film profitability and the potential for more advanced methods or richer data (e.g., marketing spend, star power) to improve predictions.
""")

st.markdown("**Thank you for exploring Cinematic Moral Dilemmas with us!**")  

