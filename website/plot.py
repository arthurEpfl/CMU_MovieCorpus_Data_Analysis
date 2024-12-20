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
import networkx as nx
from pyvis.network import Network  
from format_text import apply_gradient_color, apply_gradient_color_small

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
    st.markdown(f"""<div class='justified-text' style='text-align: center; font-size: 18px; margin-bottom: 16px;'>
                To achieve this, we experimented with <b>two different approaches</b>:<br>
                1. <b>Clustering</b>: We used unsupervised clustering (KMeans) on plot summaries to explore any emergent plot structure patterns.<br>
                2. <b>Large Language Model (LLM) Classification</b>: Using a predefined set of 15 plot structure categories, we use a LLM to classify each summary. This classification approach uses zero-shot prompting to assign each summary to a category.
                </div>""", unsafe_allow_html=True)
    
def text_clustering():
    apply_gradient_color_small("Clustering Plot Summaries!")
    texts.format_text("""First, we transform the plot summaries into a numerical format for clustering by applying <strong>TF-IDF (Term Frequency-Inverse Document Frequency)</strong> vectorization. TF-IDF highlights important words in each summary by reducing the weight of common terms and increasing the importance of unique terms.<br>
                      Afterwards, we used <strong>KMeans clustering</strong> to group the plot summaries based on their TF-IDF representations. This step aims to identify distinct plot structure patterns by clustering similar summaries together.<br>
To determine the optimal number of clusters, we used the <strong>silhouette score</strong> for cluster values ranging from 5 to 20.
However, we noticed that the silhouette score continually increased as the number of clusters increased. Given these results, we proceeded with <strong>15 clusters</strong>. This number provides a balance between interpretability and granularity, allowing us to capture a range of plot structures without creating an excessive number of small, indistinct clusters.
                      """)
    texts.format_text("""
    <div style="text-align:center;">
        We finally obtain the following clusters:
    </div>
    """)

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
    
    texts.format_text("""
    <div style="text-align:center;">
        Let's visualize the top terms per clusters!
    </div>
    """)


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
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 16px;  /* Reduced font size */
            text-align: left;
        }
        th, td {
            padding: 10px;  /* Reduced padding */
            border-bottom: 1px solid #ddd;
            background-color: #001f3f;  /* Even darker blue background for all cells */
            color: white;  /* White text color for better readability */
        }
        th {
            background-color: #001a33;  /* Slightly darker blue for header */
        }
        tr:nth-child(even) {
            background-color: #00264d;  /* Slightly lighter blue for even rows */
        }
        tr:hover {
            background-color: #003366;  /* Highlight color on hover */
        }
        .viridis-light {
            background: linear-gradient(135deg, #a6bddb 0%, #67a9cf 50%, #3690c0 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-decoration: none; /* Prevent text from being clickable */
        }
    </style>
    <table>
        <thead>
            <tr>
                <th>Cluster</th>
                <th>Interpretation</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong class="viridis-light">Cluster 1</strong></td>
                <td>Plots focused on competitive themes.</td>
            </tr>
            <tr>
                <td><strong class="viridis-light">Cluster 2</strong></td>
                <td>Crime or thriller themes, involving murder, gangs, and police confrontations.</td>
            </tr>
            <tr>
                <td><strong class="viridis-light">Cluster 3</strong></td>
                <td>Domestic and family-centered stories.</td>
            </tr>
            <tr>
                <td><strong class="viridis-light">Cluster 4</strong></td>
                <td>Sci-fi or adventure narratives set in space or otherworldly environments.</td>
            </tr>
            <tr>
                <td><strong class="viridis-light">Cluster 5</strong></td>
                <td>War or historical battle narratives, with themes of patriotism, loyalty, and military conflict.</td>
            </tr>
            <tr>
                <td><strong class="viridis-light">Cluster 6</strong></td>
                <td>Family dynamics involving financial or personal struggles, often with a focus on character growth.</td>
            </tr>
            <tr>
                <td><strong class="viridis-light">Cluster 7</strong></td>
                <td>Stories focused on love, personal growth, and the journey of family relationships.</td>
            </tr>
            <tr>
                <td><strong class="viridis-light">Cluster 8</strong></td>
                <td>Character-driven drama with themes of love, relationships, and family life.</td>
            </tr>
            <tr>
                <td><strong class="viridis-light">Cluster 9</strong></td>
                <td>Domestic dramas with family relationships at the center, often involving parents, spouses, and home life.</td>
            </tr>
            <tr>
                <td><strong class="viridis-light">Cluster 10</strong></td>
                <td>School or sports settings, focusing on themes of teamwork, mentorship, and competition.</td>
            </tr>
            <tr>
                <td><strong class="viridis-light">Cluster 11</strong></td>
                <td>Plots involving curses or superstitions, with an emphasis on individual struggles with fate or financial issues.</td>
            </tr>
            <tr>
                <td><strong class="viridis-light">Cluster 12</strong></td>
                <td>Family and relationship-centered stories, possibly featuring complex dynamics within close-knit communities.</td>
            </tr>
            <tr>
                <td><strong class="viridis-light">Cluster 13</strong></td>
                <td>Family-focused narratives often with themes of life challenges, father-son relationships, or personal introspection.</td>
            </tr>
            <tr>
                <td><strong class="viridis-light">Cluster 14</strong></td>
                <td>Stories about family dynamics and personal relationships, with a recurring theme of domestic settings.</td>
            </tr>
            <tr>
                <td><strong class="viridis-light">Cluster 15</strong></td>
                <td>Family-centered dramas, often highlighting parent-child dynamics and personal development.</td>
            </tr>
        </tbody>
    </table>
    """
    texts.format_text(text)

    texts.format_text("""
    <div style="text-align:center;">
        Each cluster reveals distinct themes and settings. While this analysis helps to identify common elements within each group, <strong>we are not fully satisfied with this approach</strong> as it appears to capture <strong>genre and themes more than specific plot structures</strong>.
    </div>
    """)

    texts.format_text("""
    <div style="text-align:center;">
        Since our goal is to identify different types of plot structures, clustering based solely on keywords may lack the depth needed to capture narrative progression and plot dynamics. Consequently, we explore alternative methods, such as leveraging large language models or deeper natural language processing techniques, to classify plot structures more accurately.
    </div>
    """)

def text_llm_classification():
    apply_gradient_color_small("Classifying Plot Summaries with Large Language Models")

    text1 = """
    As seen before clustering was not enough to extract a plot structure!
    """
    text0= """
    To tackle this, we employed <strong>large language models (LLMs)</strong> for classifying plot summaries into specific plot structure categories. Here's a streamlined view of our process:
    """
    texts.format_text("""
    <div style="text-align:center;">
        As seen before clustering was not enough to extract a plot structure!
    </div>
""")
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
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 16px;  /* Reduced font size */
            text-align: left;
        }
        th, td {
            padding: 10px;  /* Reduced padding */
            border-bottom: 1px solid #ddd;
            background-color: #001f3f;  /* Even darker blue background for all cells */
            color: white;  /* White text color for better readability */
        }
        th {
            background-color: #001a33;  /* Slightly darker blue for header */
        }
        tr:nth-child(even) {
            background-color: #00264d;  /* Slightly lighter blue for even rows */
        }
        tr:hover {
            background-color: #003366;  /* Highlight color on hover */
        }
        .viridis-light {
            background: linear-gradient(135deg, #a6bddb 0%, #67a9cf 50%, #3690c0 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-decoration: none; /* Prevent text from being clickable */
        }
    </style>
    <table>
        <thead>
            <tr>
                <th>Plot Structure</th>
                <th>Description</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong class="viridis-light">Hero's Journey and Transformation</strong></td>
                <td>Personal growth and transformation through challenges.</td>
            </tr>
            <tr>
                <td><strong class="viridis-light">Quest for Vengeance or Justice</strong></td>
                <td>Seeking retribution or justice.</td>
            </tr>
            <tr>
                <td><strong class="viridis-light">Coming of Age and Self-Discovery</strong></td>
                <td>Maturation or self-awareness in overcoming obstacles.</td>
            </tr>
            <tr>
                <td><strong class="viridis-light">Survival or Escape</strong></td>
                <td>Struggles for survival or freedom.</td>
            </tr>
            <tr>
                <td><strong class="viridis-light">Rise and Fall of a Protagonist</strong></td>
                <td>A climb to success followed by a downfall.</td>
            </tr>
            <tr>
                <td><strong class="viridis-light">Love and Relationship Dynamics</strong></td>
                <td>Exploring romance and familial bonds.</td>
            </tr>
            <tr>
                <td><strong class="viridis-light">Comedy of Errors or Misadventure</strong></td>
                <td>Humorous unintended consequences.</td>
            </tr>
            <tr>
                <td><strong class="viridis-light">Crime and Underworld Exploration</strong></td>
                <td>Criminal activities or gang conflicts.</td>
            </tr>
            <tr>
                <td><strong class="viridis-light">Power Struggle and Betrayal</strong></td>
                <td>Conflicts for leadership, marked by betrayals.</td>
            </tr>
            <tr>
                <td><strong class="viridis-light">Mystery and Conspiracy Unveiling</strong></td>
                <td>Solving mysteries or uncovering conspiracies.</td>
            </tr>
            <tr>
                <td><strong class="viridis-light">Tragedy and Inevitability</strong></td>
                <td>Facing unavoidable negative outcomes.</td>
            </tr>
            <tr>
                <td><strong class="viridis-light">Conflict with Supernatural or Unknown Forces</strong></td>
                <td>Sci-fi or supernatural challenges.</td>
            </tr>
            <tr>
                <td><strong class="viridis-light">Comedy in Domestic Life</strong></td>
                <td>Everyday humor within family life.</td>
            </tr>
            <tr>
                <td><strong class="viridis-light">Social Rebellion or Fight Against Oppression</strong></td>
                <td>Challenging societal norms or systems.</td>
            </tr>
            <tr>
                <td><strong class="viridis-light">Fantasy or Science Fiction Quest</strong></td>
                <td>Epic quests in fantastical or sci-fi worlds.</td>
            </tr>
        </tbody>
    </table>
    """
    texts.format_text(text6)

    text7 = """
    By transitioning from broad genres to these detailed plot structures, we aim to uncover the <em>storytelling formulas</em> that truly drive financial success. Up next, we’ll explore how these plot structures align with movie profitability and whether certain narratives consistently outperform others.
    """
    texts.format_text(text7)

def text_median_profit_intro():
    apply_gradient_color_small("Plot-tential Earnings: Which Stories Strike Gold?")
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
    texts.format_text(""" In terms of median profits, Love and Relation dynamics are the most profitable plot structure, followed by Quest for Vengeance or Justice and Mystery and Conspiracy Unveiling.  
                      By looking at the mean profits, Quest for Vengence or Justice is the most profitable one ! This is due to some outliers movies that generates very high revenues, while Love and Relation dynamics
                      plot structure do not have such outliers, and more "stable" and important profits.
                      """)  
    
    texts.format_text(""" Overall, mean and median profits according to plot structure decrease in a similar way. In terms of rating score, 
                      and based on our data, Love and Relationship Dynamics and Crime and Underworld Exploration are the most appreciated plot structure by the audience.
                      """)  

    texts.format_text("""
    <div style="text-align:center;">
        Now that we analyzed the profits according to plot structures, it may be interesting to study
        if plot structures and the genres of the movies are related in a certain way. Let's dive into it, shall we?
    </div>
""") 
    
    apply_gradient_color_small("Correlation between Plot Structure and Genres")  


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
    texts.format_text("""The plot structures "Conflict with Supernatural or Unknown Forces", "Comedy of Errors or Misadventure" and "Hero's Journey 
                      and "Transformation" are the most represented ones. Drama Comedy and Action are the most represented genres, as here, 
                      889 Drama movies are categorized in "Hero's Journey and Transformation"
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


# --- NETWORK GRAPH --- #
def process_data_for_directors(movies):
    data_filtered = movies.dropna(subset=['producer', 'plot_structure', 'adjusted_profit'])
    director_revenue = data_filtered.groupby('producer')['adjusted_profit'].sum()
    top_directors = director_revenue.nlargest(5).index
    return data_filtered[data_filtered['producer'].isin(top_directors)]

def create_graph(data):
    """
    Creates a NetworkX graph from a DataFrame, ensuring node attributes such as 'size' aggregate properly.
    """
    G = nx.Graph() 
    node_attributes = {}
    
    size_factor = 1e50
    offset = 5
    for _, row in data.iterrows():
        source = row['producer']
        target = row['plot_structure']
        weight = row['adjusted_profit'] / size_factor + offset

        if source not in node_attributes:
            node_attributes[source] = {'weight': weight, 'type': 'director'}
        else:
            node_attributes[source]['weight'] += weight

        # Initialize or update target node attributes
        if target not in node_attributes:
            node_attributes[target] = {'weight': weight, 'type': 'plot'}
        else:
            node_attributes[target]['weight'] += weight


    # Add nodes with aggregated attributes
    for node, attrs in node_attributes.items():
        color = '#72A0C1' if attrs['type'] == 'director' else '#90EE90'
        G.add_node(node, type=attrs['type'], size=attrs['weight'], color=color)

    # Add edges between nodes
    for _, row in data.iterrows():
        G.add_edge(row['producer'], row['plot_structure'], weight=6)

    return G

def create_network(data, selected_directors):
    # Create network graph based on selected directors
    df_select = data[data['producer'].isin(selected_directors)]
    G = create_graph(df_select)

    pos = nx.spring_layout(G)  # Positions for the nodes in G

    # Preparing to collect edge data
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        mode='lines')

    # Preparing to collect node data
    node_x = []
    node_y = []
    node_text = []
    node_sizes = []
    node_colors = []
    for node in G.nodes(data=True):
        x, y = pos[node[0]]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node[0]))
        # Assume 'size' and 'color' are attributes; scale size appropriately
        node_sizes.append(node[1]['size'])  # Adjust scaling factor as needed
        node_colors.append(node[1]['color'])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        marker=dict(
            showscale=True,  # Enable color scale if needed
            colorscale='YlGnBu',
            size=node_sizes,
            color=node_colors,
            line_width=2),
        text=node_text,
        hoverinfo='text')

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        template='plotly_white'  # Apply plotly_white theme
                        ))

    return fig

def plot_network(data):
    df_top_5 = process_data_for_directors(data)
    director_list = sorted(df_top_5['producer'].unique())
    fig = create_network(df_top_5, director_list)
    st.plotly_chart(fig, use_container_width=True)
        

def text_network_intro():
    st.markdown("""
        <style>
        .title-viridis-light {
            background: linear-gradient(135deg, #3b528b 0%, #21918c 50%, #27ad81 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 44px; 
        }
        .text-content {
            font-size: 18px;
            text-align: center;
            margin-top: 20px;
        }
        </style>
        <div class="text-content">
            <h2 id="spotlight-directors" class="title-viridis-light">Spotlight on Top Directors and Their Go-To Storylines</h2>
            We want to explore the connection between plot types and commercial success. For this reason, we looked at who brings these plots to life—the directors who craft stories that resonate with audiences.
            In our network graph, directors are represented by blue nodes and plot structures by green. Each node's size reflects the total adjusted profit associated with that director or plot structure, providing a bit more insight into their commercial impact.
        </div>
    """, unsafe_allow_html=True)
    
def text_network_conclusion():
    st.markdown("""
    <style>
    .title-viridis-light {
        background: linear-gradient(135deg, #3b528b 0%, #21918c 50%, #27ad81 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 44px; 
        text-align: center; 
    }
    .text-content {
        font-size: 18px;
        text-align: left; 
        margin: 20px;
    }
    </style>
    <div class="text-content">
        Every blockbuster shares a plot that connects with us on a deeper level. Some plots spark inspiration, others thrill us, and a few make us dream of escape.
        <ol>
            <li><strong>Hero’s Journey and Transformation</strong>
                <ul>
                    <li>Movies like <em>The Lord of the Rings</em> and <em>Star Wars</em> follow a hero’s growth, struggles, and triumph.</li>
                    <li>This timeless plot resonates because everyone loves a journey of growth.</li>
                </ul>
            </li>
            <li><strong>Survival or Escape</strong>
                <ul>
                    <li>Whether it’s escaping a dinosaur-infested island (<em>Jurassic Park</em>) or a killer shark (<em>Jaws</em>), survival stories keep audiences on the edge of their seats.</li>
                    <li>These adrenaline-filled plots deliver the excitement people crave.</li>
                </ul>
            </li>
            <li><strong>Conflict and Betrayal</strong>
                <ul>
                    <li>Betrayal stories add drama and depth, from kingdoms falling apart (<em>Game of Thrones</em>) to friendships tested (<em>The Dark Knight</em>).</li>
                </ul>
            </li>
            <li><strong>Coming-of-Age and Self-Discovery</strong>
                <ul>
                    <li>These themes connect deeply with audiences, especially younger generations (<em>Harry Potter</em> and <em>Stand by Me</em>).</li>
                </ul>
            </li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    .text-content {
        font-size: 18px;
        text-align: left; 
        margin: 20px; 
    }
    </style>
    <div class="text-content">
        Once we understood the plots, we looked at <strong>who tells these stories</strong>. Directors shape the narratives we love, and each has their own sweet spot when it comes to plot structures.
        <ul>
            <li><strong>George Lucas</strong> loves a <em>Hero’s Journey</em>. From <em>Star Wars</em>, he showed how powerful transformation stories could be when mixed with fantasy and epic conflicts.
                <ul>
                    <li><em>The rise of Luke Skywalker</em> isn’t just a movie—it’s a story of courage, destiny, and hope.</li>
                </ul>
            </li>
            <li><strong>Chris Columbus</strong> thrives on <em>Coming-of-Age and Self-Discovery</em>. His success with <em>Harry Potter and the Sorcerer's Stone</em> speaks to the magic of relatable childhood journeys.</li>
            <li><strong>Steven Spielberg</strong> takes us on adventures of <em>Survival and Escape</em>. From <em>Jaws</em> to <em>E.T.</em>, he’s the master of blending thrills with heartfelt moments.</li>
            <li><strong>James Cameron</strong> brings drama with <em>Conflict and Power Struggles</em>. Think of <em>Titanic</em> and <em>Avatar</em>—both feature conflict at every level, from personal relationships to epic battles.</li>
            <li><strong>Robert Zemeckis</strong> delivers stories of <em>Transformation and Growth</em>. Movies like <em>Back to the Future</em> mix adventure and self-discovery, adding a nostalgic charm.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    .text-insights {
        font-size: 18px;
        text-align: left;
        margin: 20px; 
    }
    </style>
    <div class="text-insights">
        Our network graph uncovered some fascinating insights:
        <ul>
            <li><strong>Relatable Plots = Big Success</strong>: Themes like transformation, escape, and betrayal appear repeatedly in top-performing movies.</li>
            <li><strong>Directors Have Their Comfort Zones</strong>: The most successful directors stick to plot structures they excel at.</li>
            <li><strong>Genres Enhance the Plot</strong>: Adventure, Fantasy, and Drama amplify these plots, making them even more impactful.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    .text-closing {
        font-size: 18px;
        text-align: left;
        margin: 20px;
    }
    .highlight {
        font-weight: bold;
        color: #31708f;
    }
    </style>
    <div class="text-closing">
        The real magic of cinema starts with a powerful story. Directors like Lucas, Spielberg, and Columbus proved that the right plot structure, combined with strong execution, can create unforgettable experiences that audiences love—and box offices celebrate.
        <p class="highlight">
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")