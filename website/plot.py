import pandas as pd
import format_text as texts
import streamlit as st
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ast
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def text_intro():
    texts.format_text("""Alright, so we've talked about genres and their financial impact, but let's be real—genre 
                      alone doesn't tell the whole story. A romantic comedy could be a hit, or it could fall flat. The same goes for a thriller—what's in the story matters just as much as the label. 
                      That's why we took things a step further and dug into the plot structure. After all, the way a story unfolds can make or break a movie's success.
                    """)
    texts.format_text("""So, instead of just putting movies in neat little genre boxes, we went through the plot summaries 
                    and classified them into different narrative structures. You know, the stuff like "The Hero's Journey" 
                      or "The Love Triangle" that really gives a movie its pizzazz. We'll explain the different approaches 
                      we used to do this and why it could be a game-changer for understanding financial success. Ready to dive into the plot twists? Let's go!
                    """)
    st.markdown(f"""<div class='justified-text' style='text-align: justify; font-size: 18px; margin-bottom: 16px;'>
                To achieve this, we experimented with <b>two different approaches</b>:<br>
                1. <b>Clustering</b>: We used unsupervised clustering (KMeans) on plot summaries to explore any emergent plot structure patterns.<br>
                2. <b>Large Language Model (LLM) Classification</b>: Using a predefined set of 15 plot structure categories, we use a LLM to classify each summary. This classification approach uses zero-shot prompting to assign each summary to a category.
                </div>""", unsafe_allow_html=True)
    
def text_clustering():
    st.subheader("Clustering Plot Summaries!")
    texts.format_text("""First, we transform the plot summaries into a numerical format for clustering by applying <strong>TF-IDF (Term Frequency-Inverse Document Frequency)</strong> vectorization. TF-IDF highlights important words in each summary by reducing the weight of common terms and increasing the importance of unique terms.<br>
                      Afterwards, we used <strong>KMeans clustering</strong> to group the plot summaries based on their TF-IDF representations. This step aims to identify distinct plot structure patterns by clustering similar summaries together.<br>
To determine the optimal number of clusters, we used the <strong>silhouette score</strong> for cluster values ranging from 5 to 20.
However, we noticed that the silhouette score continually increased as the number of clusters increased.<br>
Given these results, we proceeded with <strong>15 clusters</strong>. This number provides a balance between interpretability and granularity, allowing us to capture a range of plot structures without creating an excessive number of small, indistinct clusters.
                      """)
    texts.format_text("We finally obtain the following clusters:")


def plot_clusters(movies, combined_matrix):
    X_reduced_tsne = TruncatedSVD(n_components=2, random_state=42).fit_transform(combined_matrix)
    X_reduced_pca = PCA(n_components=2, random_state=42).fit_transform(combined_matrix)

    labels = movies['plot_structure_cluster']

    unique_labels = labels.unique()
    color_palette = px.colors.qualitative.Plotly
    color_map = {label: color_palette[i % len(color_palette)] for i, label in enumerate(unique_labels)}

    col1, col2 = st.columns(2)
    with col1:
        texts.format_text("SVD Visualization")

        fig = go.Figure(data=go.Scatter(
            x=X_reduced_tsne[:, 0],
            y=X_reduced_tsne[:, 1],
            mode='markers',
            marker=dict(
                color=[color_map[label] for label in labels],
                showscale=False,
                size=8
            ),
            text=[f'Cluster {l}' for l in labels]
        ))

        for label in unique_labels:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=color_map[label]),
                legendgroup=str(label),
                showlegend=True,
                name=f'Cluster {label}'
            ))
        
        fig.update_layout(
            title=f'Clustering Visualization using T-SNE',
            template='plotly_white',
            height=500,
            width=600
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        texts.format_text("PCA Visualization")

        fig = go.Figure(data=go.Scatter(
            x=X_reduced_pca[:, 0],
            y=X_reduced_pca[:, 1],
            mode='markers',
            marker=dict(
                color=[color_map[label] for label in labels],
                showscale=False,
                size=8
            ),
            text=[f'Cluster {l}' for l in labels]
        ))

        for label in unique_labels:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=color_map[label]),
                legendgroup=str(label),
                showlegend=True,
                name=f'Cluster {label}'
            ))
        
        fig.update_layout(
            title=f'Clustering Visualization using PCA',
            template='plotly_white',
            height=500,
            width=600
        )
        st.plotly_chart(fig, use_container_width=True)

def text_cluster_distribution():
    texts.format_text("""The distribution of plot summaries across clusters shows that the clustering algorithm has created some clusters with a significantly higher number of summaries than others. The top three clusters (2, 10, and 7) collectively hold a large portion of the summaries, indicating that certain plot structures may be more common. We have to dive more in the clusters.
                      """)
    
    texts.format_text("""Let's visualize the top terms per clusters!""")


def plot_word_clouds(tfidf_vectorizer, kmeans, n_clusters=15):
    # Get the top terms per cluster by averaging the TF-IDF values of the terms in each cluster
    terms = tfidf_vectorizer.get_feature_names_out()
    cluster_centers = kmeans.cluster_centers_
    top_terms_per_cluster = []

    for i in range(n_clusters):
        top_terms_idx = cluster_centers[i].argsort()[-15:]  # Top 15 terms per cluster
        top_terms_per_cluster.append({terms[idx]: cluster_centers[i][idx] for idx in top_terms_idx})

    # Create a dropdown menu for cluster selection
    selected_cluster = st.selectbox('Select Cluster', range(1,n_clusters+1))

    # Generate the word cloud for the selected cluster
    word_freq = top_terms_per_cluster[selected_cluster - 1]
    words = list(word_freq.keys())
    frequencies = list(word_freq.values())

    # Assign a color to each word
    color_palette = px.colors.qualitative.Plotly
    word_colors = [color_palette[i % len(color_palette)] for i in range(len(words))]
    sizes = [np.exp(v)*40 for v in frequencies]  # Log to reduce range of sizes
    x_positions = np.random.normal(0.5, 0.01, len(words))
    y_positions = np.random.normal(0.5, 0.01, len(words))

    # Scatter plot
    trace = go.Scatter(
        x=x_positions,
        y=y_positions,
        text=words,
        mode='text',
        textfont={'size': sizes, 'color': word_colors},
        hoverinfo='text'
    )

    layout = go.Layout(
        title=f"Word Cloud for Cluster {selected_cluster}",
        xaxis={'showgrid': False, 'showticklabels': False, 'zeroline': False},
        yaxis={'showgrid': False, 'showticklabels': False, 'zeroline': False}
    )

    fig = go.Figure(data=[trace], layout=layout)
    st.plotly_chart(fig, use_container_width=True)


def text_cluster_interpretion():
    text = """
    Here’s an interpretation of each cluster based on the top terms:<br><br>
    - <strong>Cluster 1</strong>: Plots focused on competitive themes.<br>
    - <strong>Cluster 2</strong>: Crime or thriller themes, involving murder, gangs, and police confrontations.<br>
    - <strong>Cluster 3</strong>: Domestic and family-centered stories.<br>
    - <strong>Cluster 4</strong>: Sci-fi or adventure narratives set in space or otherworldly environments.<br>
    - <strong>Cluster 5</strong>: War or historical battle narratives, with themes of patriotism, loyalty, and military conflict.<br>
    - <strong>Cluster 6</strong>: Family dynamics involving financial or personal struggles, often with a focus on character growth.<br>
    - <strong>Cluster 7</strong>: Stories focused on love, personal growth, and the journey of family relationships.<br>
    - <strong>Cluster 8</strong>: Character-driven drama with themes of love, relationships, and family life.<br>
    - <strong>Cluster 9</strong>: Domestic dramas with family relationships at the center, often involving parents, spouses, and home life.<br>
    - <strong>Cluster 10</strong>: School or sports settings, focusing on themes of teamwork, mentorship, and competition.<br>
    - <strong>Cluster 11</strong>: Plots involving curses or superstitions, with an emphasis on individual struggles with fate or financial issues.<br>
    - <strong>Cluster 12</strong>: Family and relationship-centered stories, possibly featuring complex dynamics within close-knit communities.<br>
    - <strong>Cluster 13</strong>: Family-focused narratives often with themes of life challenges, father-son relationships, or personal introspection.<br>
    - <strong>Cluster 14</strong>: Stories about family dynamics and personal relationships, with a recurring theme of domestic settings.<br>
    - <strong>Cluster 15</strong>: Family-centered dramas, often highlighting parent-child dynamics and personal development.<br><br>
    """
    texts.format_text(text)

    texts.format_text("""Each cluster reveals distinct themes and settings. While this analysis helps to identify common elements within each group, <strong>we are not fully satisfied with this approach</strong> as it appears to capture <strong>genre and themes more than specific plot structures</strong>.
                      """)
    texts.format_text("""
    Since our goal is to identify different types of plot structures, clustering based solely on keywords may lack the depth needed to capture narrative progression and plot dynamics. Consequently, we explore alternative methods, such as leveraging large language models or deeper natural language processing techniques, to classify plot structures more accurately.
                      """)

def text_llm_classification():
    st.subheader("Classifying Plot Summaries with Large Language Models")

    text1 = """
    As seen before clustering was not enough to extract a plot structure!
    """
    text0= """
    To tackle this, we employed <strong>large language models (LLMs)</strong> for classifying plot summaries into specific plot structure categories. Here's a streamlined view of our process:
    """
    texts.format_text(text1)
    texts.format_text(text0)

    text2 = """
    1. <strong>Summarizing Plot Summaries</strong>:<br>
       Many plot summaries were too long to fit the input token limits of LLMs. To manage this, we summarized them using the <code>facebook/bart-large-cnn</code> model. This step ensured we could process even the most detailed summaries while retaining their narrative essence.
    """
    texts.format_text(text2)

    text3 = """
    2. <strong>Classifying Summaries</strong>:<br>
       Using the summarized versions, we leveraged the <code>facebook/bart-large-mnli</code> model for <strong>zero-shot classification</strong>. This allowed us to categorize plots without additional training, making use of a predefined list of narrative categories.
    """
    texts.format_text(text3)

    text4 = """
    Genres give a general sense of what a movie is about, but plot structures reveal <em>how</em> the story unfolds. For example, two movies labeled "Action" could have vastly different narratives—one following a <em>Hero's Journey</em>, and another centered on a <em>Power Struggle</em>. Understanding these distinctions provides more nuanced insights into what drives audience engagement and profitability.
    """
    texts.format_text(text4)

    text5 = """
    To classify plot structures effectively, we synthesized ideas from classic narrative frameworks, including <em>The Seven Basic Plots</em> by Christopher Booker and <em>The Hero's Journey</em> by Joseph Campbell. Using these, we developed a set of 15 distinct and descriptive categories to ensure clarity for the model:
    """
    texts.format_text(text5)

    text6 = """
    1. <strong>Hero's Journey and Transformation</strong>: Personal growth and transformation through challenges.<br>
    2. <strong>Quest for Vengeance or Justice</strong>: Seeking retribution or justice.<br>
    3. <strong>Coming of Age and Self-Discovery</strong>: Maturation or self-awareness in overcoming obstacles.<br>
    4. <strong>Survival or Escape</strong>: Struggles for survival or freedom.<br>
    5. <strong>Rise and Fall of a Protagonist</strong>: A climb to success followed by a downfall.<br>
    6. <strong>Love and Relationship Dynamics</strong>: Exploring romance and familial bonds.<br>
    7. <strong>Comedy of Errors or Misadventure</strong>: Humorous unintended consequences.<br>
    8. <strong>Crime and Underworld Exploration</strong>: Criminal activities or gang conflicts.<br>
    9. <strong>Power Struggle and Betrayal</strong>: Conflicts for leadership, marked by betrayals.<br>
    10. <strong>Mystery and Conspiracy Unveiling</strong>: Solving mysteries or uncovering conspiracies.<br>
    11. <strong>Tragedy and Inevitability</strong>: Facing unavoidable negative outcomes.<br>
    12. <strong>Conflict with Supernatural or Unknown Forces</strong>: Sci-fi or supernatural challenges.<br>
    13. <strong>Comedy in Domestic Life</strong>: Everyday humor within family life.<br>
    14. <strong>Social Rebellion or Fight Against Oppression</strong>: Challenging societal norms or systems.<br>
    15. <strong>Fantasy or Science Fiction Quest</strong>: Epic quests in fantastical or sci-fi worlds.
    """
    texts.format_text(text6)

    text7 = """
    By transitioning from broad genres to these detailed plot structures, we aim to uncover the <em>storytelling formulas</em> that truly drive financial success. Up next, we’ll explore how these plot structures align with movie profitability and whether certain narratives consistently outperform others.
    """
    texts.format_text(text7)

def text_median_profit_intro():
    st.subheader("Plot-tential Earnings: Which Stories Strike Gold?")
    texts.format_text("""
Before we dive into the numbers, let's talk about why this is exciting. Plot structures are like the secret sauce of storytelling—they're the frameworks that hold our favorite movies together. But not all sauces are created equal. Some are rich and flavorful, while others... well, let's just say they leave a lot to be desired. Now, it's time to find out which narrative recipes rake in the big bucks and which ones just simmer on the back burner. Let’s take a look at the median profit per plot structure to see which stories truly pay off at the box office.
                      """)
    
def plot_median_profit(movies, adjusted=True):
    median_profit_plot = movies.groupby('plot_structure')['adjusted_profit'].median().reset_index()
    mean_profit_plot = movies.groupby('plot_structure')['adjusted_profit'].mean().reset_index()
    rating_score_plot = movies.groupby('plot_structure')['rating_score'].mean().reset_index()

    # Merge median, mean, and rating score dataframes
    profit_plot = median_profit_plot.merge(mean_profit_plot, on='plot_structure', suffixes=('_median', '_mean'))
    profit_plot = profit_plot.merge(rating_score_plot, on='plot_structure')
    profit_plot = profit_plot.sort_values(by='adjusted_profit_median', ascending=False)

    fig = go.Figure()

    # Add median profit bar
    fig.add_trace(go.Bar(
        x=profit_plot['plot_structure'],
        y=profit_plot['adjusted_profit_median'],
        marker_color=px.colors.sequential.Viridis,
        name='Median Revenue',
        hovertemplate="Plot Structure: %{x}<br>Median Revenue ($): %{y:$,.0f}<extra></extra>"
    ))

    # Add mean profit line
    fig.add_trace(go.Scatter(
        x=profit_plot['plot_structure'],
        y=profit_plot['adjusted_profit_mean'],
        mode='lines+markers',
        line=dict(color='red', width=2),
        name='Mean Revenue',
        hovertemplate="Plot Structure: %{x}<br>Mean Revenue ($): %{y:$,.0f}<extra></extra>"
    ))

    # Add rating score line
    fig.add_trace(go.Scatter(
        x=profit_plot['plot_structure'],
        y=profit_plot['rating_score'],
        mode='lines+markers',
        line=dict(color='blue', width=2, dash='dash'),
        name='Rating Score',
        yaxis='y2',
        hovertemplate="Plot Structure: %{x}<br>Rating Score: %{y:.2f}<extra></extra>"
    ))

    # Update layout for better appearance
    fig.update_layout(
        title={
            'text': f"Median Revenue by Plot Structure {'Adjusted with Inflation' if adjusted else '' }",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Plot Structure",
        yaxis_title=f"Revenue {'Adjusted with Inflation' if adjusted else ''} (in $)",
        yaxis2=dict(
            title="Rating Score",
            overlaying='y',
            side='right'
        ),
        xaxis_tickangle=-45,
        template="plotly_white",
        height=500,  # Increased height for better visibility
        showlegend=True,
        margin=dict(t=50, l=50, r=50, b=100)  # Adjusted margins for better label visibility
    )
    
    st.plotly_chart(fig, use_container_width=True)

def text_conclusion_median_profit():
    texts.format_text(""" FAIRE INTERPRETATION DE CE GRAPHE BEAUCOUP de BLABLA + TANSITION POUR HEATMAP
                      A COMPLETER !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                      """)

def plot_genre_plot_structure_heatmap(movies, top_genre):
    df_exploded = movies.explode('movie_genres')
    df_2 = df_exploded[df_exploded['movie_genres'].isin(top_genre.index)]

    # Create a pivot table to count the occurrences of each genre-plot structure combination
    pivot_table = df_2.pivot_table(index='movie_genres', columns='plot_structure', aggfunc='size', fill_value=0)

    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        colorscale='Viridis',
        showscale=True,
        hoverongaps=False,
        text=pivot_table.values,
        texttemplate="%{text}",
        textfont={"size": 12},
        zmin=0,
        zmax=pivot_table.values.max(),
        hovertemplate="Genre: %{y}<br>Plot Structure: %{x}<br>Count: %{z}<extra></extra>"
    ))

    # Update layout for better appearance
    fig.update_layout(
        title='Relation between Genres and Plot Structure',
        xaxis_title='Plot Structure',
        yaxis_title='Movie Genres',
        xaxis=dict(tickangle=90),
        template='plotly_white',
        height=600,
        width=800,
        margin=dict(t=50, l=50, r=50, b=100),
        legend=dict(
            title="Count",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Add separation between cells
    fig.update_traces(
        xgap=3,  # gap between columns
        ygap=3   # gap between rows
    )

    st.plotly_chart(fig, use_container_width=True)

def text_conclusion():
    texts.format_text("""FAIRE INTERPRETATION DE CE GRAPHE BEAUCOUP de BLABLA + TANSITION POUR LA SUITE
                      A COMPLETER !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                      """)


# --- UTILS --- #

def get_clusters(col):
    # Text Vectorization with TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(col)

    combined_matrix = tfidf_matrix.toarray()

    # Clustering with KMeans
    n_clusters = 15
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    return kmeans.fit_predict(combined_matrix), combined_matrix, tfidf_vectorizer, kmeans