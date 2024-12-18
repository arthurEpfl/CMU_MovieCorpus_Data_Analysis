import streamlit as st
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
<div style="text-align:center; font-size:24px; font-family: 'Cursive', sans-serif;">
    Did you think of Web scraping to get budget values ? 
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


# ===================== SECTION 5: Genres and Commercial Success =====================  

st.markdown("""
<div style="text-align:center;">
    <h2>How genres are related to box offices revenues and commercial success  
</h2>
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
<div style="text-align:center;">
    <h3>💰 Adjusting Box Office Revenues for Inflation</h3>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="font-size:18px; text-align:justify;">
To make fair comparisons between movies released in different eras, we need to account for inflation. 
A movie making $1 million in 1980 is very different from making $1 million today!

Here's how we adjusted the revenues:

1. **Base Year Selection**: We chose 2023 as our reference point
2. **CPI Data**: Used Consumer Price Index data to track inflation over time
3. **Adjustment Formula**: Applied this formula to normalize revenues:

</div>
""", unsafe_allow_html=True)

st.latex(r"\text{Adjusted Revenue} = \text{Original Revenue} \times \frac{\text{CPI}_{2023}}{\text{CPI}_{\text{Movie Year}}}")

st.markdown("""
<div style="font-size:18px; text-align:justify;">
This adjustment helps us:
- Compare movies across different decades fairly
- Understand true financial impact in today's terms
- Make more accurate assessments of commercial success
</div>
<br>
""", unsafe_allow_html=True)  

 
st.markdown("""
<div style="text-align:center;">
    <h2>💰 Inflation-Adjusted Revenue Analysis  
</h2>
</div>
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
<div style="font-size:18px; text-align:center;">
What really makes a movie successful, big numbers overall or a few standout hits? The total revenue from the 1990s and 2000s looks huge, but that’s mostly because so many movies were made during those years. More movies mean higher totals, but does that really show success?

To get a better picture, we look at **mean** and **max revenue**:
- **Mean Revenue** gives us the average, showing how movies performed overall each year without being overwhelmed by how many were released.
- **Max Revenue** focuses on the biggest hits—the movies that smashed records and left their mark.

If we stick with total revenue, we miss the real stars and trends. By focusing on **averages and standouts**, we can better figure out what makes a movie truly successful.

</div>
<br>
""", unsafe_allow_html=True)

# Add explanatory text
st.markdown("""
<div style="font-size:18px; text-align:center;">
The graphs above show how movie revenues have changed over time, both in original and 
inflation-adjusted terms. This comparison helps us to understand, how the value of movie earnings has changed over time.
The true financial impact of movies when accounting for inflation. Which eras were most successful in today's monetary terms
</div>
""", unsafe_allow_html=True)  

st.markdown("""
<div style="text-align:center;">
    <h2>📊 Detailed Revenue Analysis with Genres 
</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="font-size:18px; text-align:center;">
Let's analyse how genres are related to box office revenues. You can select genre distribution influence, genre trends,  
revenue heatmap and bubble chart. 
</div>
""", unsafe_allow_html=True)

# Create tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs([
    "Revenue Distribution", 
    "Genre Trends", 
    "Revenue Heatmap",
    "Revenue Bubble Chart"
])

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

with tab3:
    st.markdown("### Revenue Heatmap")
    process_inflation_data = mod.process_inflation_data_genre(
            movies,
            df_inflation,
            year_range=year_range,
            selected_genres=None
    )
    use_adjusted = st.checkbox("Show Inflation Adjusted Values", value=True, key="heatmap_adjusted")
    heatmap_fig = plot_app.plot_revenue_heatmap(process_inflation_data, adjusted=use_adjusted)
    st.plotly_chart(heatmap_fig, use_container_width=True)
    
    st.markdown("""
    The heatmap reveals patterns in revenue across different genres and years, 
    with darker colors indicating higher revenues.
    """)

with tab4:
    st.markdown("### Revenue Bubble Chart")
    process_inflation_data = mod.process_inflation_data_genre(
            movies,
            df_inflation,
            year_range=year_range,
            selected_genres=None
        )
    use_adjusted = st.checkbox("Show Inflation Adjusted Values", value=True, key="bubble_adjusted")
    bubble_fig = plot_app.plot_revenue_bubble(process_inflation_data, adjusted=use_adjusted)
    st.plotly_chart(bubble_fig, use_container_width=True)
    
    st.markdown("""
    This bubble chart combines multiple dimensions:
    - Position shows year and revenue
    - Size represents number of movies
    - Color indicates average rating
    - Hover over bubbles for detailed information
    """)

# Add overall insights  

st.markdown("""
<div style="text-align:center;">
    <h2>🔍 Key Insights 
</h2>
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
There are also Genre Performance, certain genres consistently outperform others when adjusted for inflation. 
In the Industry Growth**, The bubble chart reveals the relationship between movie volume and revenue
</div>
""", unsafe_allow_html=True)

# In your commercial success analysis section
st.markdown("## 💰 Commercial Success Analysis")

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
 
 



