import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os  

import plot_app

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
<div style="text-align:center; font-size:24px; font-family: 'Cursive', sans-serif;">
    Adarable
</div>
""", unsafe_allow_html=True)  

st.markdown("""
<div style="text-align:center;">
    <h1>🎬 Decoding the Blueprint of a Blockbuster: Analyzing Plot Structures for Box Office Success</h1>
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


##### TABLE OF CONTENTS  #####

st.markdown("## Our Story Timeline")
st.markdown("""
- **1. First Explorations**  
- **2. Preprocessing**  
- **3. Exploratory Data Analysis**  
- **4. Genres, Revenues, and Commercial Success**  
- **5. How genres are related to box offices revenues and commercial success**  
- **6. Plot Structure Analysis with Clustering & LLM Classification**  
- **7. Predictive Modeling**  
""")

st.markdown("---")


# ===================== SECTION 2: FIRST EXPLORATIONS =====================
st.markdown("""
<div style="text-align:center;">
    <h2>First Explorations</h2>
</div>
""", unsafe_allow_html=True)  


st.markdown("""
<div style="font-size:18px; text-align:center;">
    <div style="text-align:center; font-size:19px; font-weight:bold;">
        Initially, we explored raw data from CMU Movie Corpus dataset.
    </div>
    <br>
    In the latter, missing box office revenues were identified so there is a need for external data sources. Also, budget values are missing,
    we need them if we want to compute profitability of these movies !  
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
is its success due to its genre, its plot structure, Jack Dawson itself or something else? We will explore this in the following sections. 
<br><br>
""", unsafe_allow_html=True)


st.markdown("""
<div style="font-size:18px; text-align:justify;">
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
<div style="text-align:center;">
    <h2>Preprocessing</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("""
We cleaned data, merged with IMDb dataset, and retained movies with both box office and plot summaries.
""")

st.markdown("""
**Resulting dataset:** `movies_summary_BO.csv` with ~7,964 movies.
""")

#with st.expander("No Filters"):
    #st.info("Preprocessing steps are explained; no filters apply.")

st.markdown("""
<div style="font-size:18px; text-align:center;">
Blablabla rapide sur le preprocessing, nettoyage des données, tout clean etc.   
<br><br>
""", unsafe_allow_html=True)

st.markdown("---")

# ===================== SECTION 4: EDA =====================
st.markdown("""
<div style="text-align:center;">
<h2>Exploratory Data Analysis</h2>
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
<div style="text-align:center; font-size:21px;">
    Now, what are the box office revenues by years?
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
<div style="text-align:center; font-size:21px;">
    Now, what is the distribution of movies by countries ?
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
<div style="text-align:center; font-size:21px;">
    Now, what is the language distribution in movies ?
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
<div style="text-align:center; font-size:21px;">
    Let's explore the runtime and release year distributions!
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


#with st.expander("Filters for EDA"):
    #year_range_eda = st.slider("Select Year Range for EDA", 1900, 2020, (1980, 2000))

#eda_filtered = movies[(movies['movie_release_date'] >= year_range_eda[0]) & (movies['movie_release_date'] <= year_range_eda[1])]

# à voir si on garde

st.markdown("""
<div style="text-align:center; font-size:21px;">
    Let's focus now on how genres are related to worldwide box offices revenues and analyses of commercial successes.  
</div>
""", unsafe_allow_html=True)

st.markdown("---")  


# ===================== SECTION 5: Genres and Commercial Success =====================  

st.markdown("""
<div style="text-align:center;">
    <h2>How genres are related to box offices revenues and commercial success  
</h2>
</div>
""", unsafe_allow_html=True)

with st.expander("Filters for Genre Analysis"):
    genre_year_range = st.slider("Year Range for Genre Analysis", 1900, 2020, (1980, 2000))

genre_filtered = movies[(movies['movie_release_date'] >= genre_year_range[0]) & (movies['movie_release_date'] <= genre_year_range[1])]
genre_exploded = genre_filtered.explode('movie_genres')

top_15_genres = genre_exploded['movie_genres'].value_counts().head(15)

# Define a consistent color palette
palette = sns.color_palette("husl", 15)
color_dict = {genre: palette[i] for i, genre in enumerate(top_15_genres.index)}

fig, ax = plt.subplots(figsize=(8, 2))
sns.barplot(x=top_15_genres.index, y=top_15_genres.values, palette=[color_dict[genre] for genre in top_15_genres.index], ax=ax)
ax.set_title('Top 15 Genres')
ax.set_xlabel('Genre')
ax.set_ylabel('Count')
plt.xticks(rotation=45, fontsize=6)  # Adjust the x-tick labels font size
plt.yticks(fontsize=6)  # Adjust the y-tick labels font size
st.pyplot(fig)  

st.markdown("""
<div style="font-size:18px; text-align:center;">
We notice here, in terms of frequency in percentage, Drama movies are the most distributed ones, 
followed by comedy movies and thrillers. We display here only the 15 most distributed ones in the 
processed dataset. We now have a question, which genres are generating the highest revenues ? 
This may be an excellent question for a filmmakers, we want a movie to generate money, right ? 
<br><br> 
""", unsafe_allow_html=True)   


st.markdown("""
<div style="text-align:center; font-size:21px;">
    Which genres generate the highest revenues ?
</div>
""", unsafe_allow_html=True)  

# ADD PLOT HERE  
# 
st.markdown("""
<div style="font-size:18px; text-align:center;">
We notice here that Fantasy, Adventure and Family Film movies are the one that have the highest mean 
box office revenues. Drama movies are the most distributed ones, but do not generated high mean revenues !  
<br><br>
""", unsafe_allow_html=True)  

st.markdown("""
<div style="text-align:center; font-size:24px;">
 There is one thing we forgot...To truly see which genres make the most money, we need to take into account inflation over the years ! 

</div>
<br><br>
""", unsafe_allow_html=True)  


st.markdown("""
### Adjusting Movie Box Office Revenues for Inflation

We want to make movie box office revenues comparable across different release years by adjusting them for inflation. We use the Consumer Price Index (CPI) data to account for inflation and bring all revenues to a common base year. This way, we can fairly compare the earnings of movies released in different years.

### Steps

- We start with CPI data indexed by date, which represents the inflation rate over time.
- We choose a **base year** (2023) to normalize all other CPI values.

- For each year in our CPI dataset, we calculate an **adjustment factor** by dividing the CPI value of the base year by the CPI of that specific year:
""")

st.latex(r'''
\text{Adjustment Factor} = \frac{\text{CPI of Base Year}}{\text{CPI of Movie Year}}
''')

st.markdown("""
- Using the adjustment factors calculated above, we adjust each movie’s box office revenue based on its release year.
""")  


# PLOT 


st.markdown("""
<div style="font-size:18px; text-align:center;">
In the inflation-adjusted view, Family Films and Dramas rise to the top, showcasing their enduring appeal 
and the lasting power of their stories. On the other hand, the non-adjusted view highlights the dominance of 
blockbuster genres like Adventure, Action, and Fantasy, driven by modern budgets and global excitement. It’s like looking at two sides of the same coin: one celebrates the stories that last forever, and the 
other shows off the thrill of today’s biggest hits.  
<br><br>
""", unsafe_allow_html=True)  

st.markdown("""
<div style="font-size:18px; text-align:center;">
Let's analyse evolution over time of revenues and profits, according to the 15 main genres. 
For the rest of the analyses, we'll only use datasets where the inflation is used to adjust the revenues, 
budget and profit values. It's important to compare the films on same values of revenue. 
A dollar forty years ago worths much nowadays.  
<br><br>
""", unsafe_allow_html=True)   



st.markdown("""
### Commercial successes
""")


st.markdown("---")

# ===================== SECTION 6: Plot Structure Analysis =====================
st.markdown("""
<div style="text-align:center;">
    <h2>Plot Structure Analysis with Clustering & LLM Classification**  
</h2>
</div>
""", unsafe_allow_html=True) 


st.markdown("""
<div style="font-size:18px; text-align:center;">
Some explanation on what we done.    
<br><br>
""", unsafe_allow_html=True) 

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


# ===================== SECTION 7: Predictive Modeling =====================
st.markdown("""
<div style="text-align:center;">
    <h2>Predictive Modeling</h2>
</div>
""", unsafe_allow_html=True)  

st.markdown("""
<div style="font-size:18px; text-align:center;">
Some explanation on what we done.    
<br><br>
""", unsafe_allow_html=True) 

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

