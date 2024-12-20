import pandas as pd
import format_text as texts
import streamlit as st
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ast
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import numpy as np  

from format_text import apply_gradient_color, apply_gradient_color_small


def intro_text():  
    apply_gradient_color_small('First of all, what is the genre of a movie?')
    texts.format_text(""" The genre of a movie defines its category or type, characterized by shared themes, 
                      storytelling elements, and emotional tone. It helps audiences identify what kind of experience to expect, 
                      such as humor in comedies, suspense in thrillers, or emotional depth in dramas.
                      """)
    texts.format_text("""
    <div style="text-align:center;">
        Therefore, we look into the distribution of genres in our dataset !  
    </div>
""")
    
def plot_genre_distribution(top_genre, color_dict):
    # Create Plotly pie chart
    fig = go.Figure(data=[
        go.Pie(
            labels=top_genre.index,
            values=top_genre.values,
            marker=dict(colors=[color_dict[genre] for genre in top_genre.index]),
            hovertemplate="Genre: %{label}<br>Frequency (%): %{value}<extra></extra>",
            pull=[0.05] * len(top_genre)  # Separate each slice a bit
        )
    ])

    # Update layout for better appearance
    fig.update_layout(
        title={
            'text': 'Top 15 Movie Genres',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        template="plotly_white",  # Matches your dark theme
        height=500,  # Increased height for better visibility
        showlegend=True,
        margin=dict(t=50, l=50, r=50, b=100)  # Adjusted margins for better label visibility
    )

    st.plotly_chart(fig, use_container_width=True)
    texts.format_text(""" We observe that Drama movies dominate in terms of frequency, followed by comedy
                       and thriller movies. Here, we display only the top 15 most frequent genres in the processed dataset. 
                      This raises the question, which genres generate the highest revenues? For filmmakers, this is a critical question, after all, 
                      the goal is to create a movie that generates money, right?
                      """)
    
def text_genre_revenue():
    apply_gradient_color_small('Which genres generate the highest revenues?')
    texts.format_text("""To answer this question, we look at the average revenues for each genre by only keeping the 15 top genres. 
                      The average revenue is the total revenue divided by the number of movies in this genre.
                      """)

def plot_genre__mean_revenue(filtered_df, color_dict, adjusted=False) :
    revenue_column = 'adjusted_revenue' if adjusted else 'movie_box_office_revenue'
    mean_revenues = filtered_df.groupby('movie_genres')[revenue_column].mean().reset_index()
    mean_revenues = mean_revenues.sort_values(by=revenue_column, ascending=False)

    # Create Plotly bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=mean_revenues['movie_genres'],
            y=mean_revenues[revenue_column],
            marker_color=[color_dict[genre] for genre in mean_revenues['movie_genres']],
            hovertemplate="Genre: %{x}<br>Average Revenue ($): %{y:$,.0f}<extra></extra>"
        )
    ])

    # Update layout for better appearance
    fig.update_layout(
        title={
            'text': f"Average Revenue by Genre {'Adjusted with Inflation' if adjusted else '' }",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Genre",
        yaxis_title= f"Average Revenue {'Adjusted with Inflation' if adjusted else ''} (in $)",
        xaxis_tickangle=-45,
        template="plotly_white",  # Matches your dark theme
        height=500,  # Increased height for better visibility
        showlegend=False,
        margin=dict(t=50, l=50, r=50, b=100)  # Adjusted margins for better label visibility
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
def text_mean_revenue_comment():
    texts.format_text(""" We notice here that Fantasy, Adventure and Family Film movies are the one that have the highest mean box office revenues. 
                      Drama movies are the most distributed ones, but do not generated high mean revenues !
                      However, this is still consistent with the idea that these genres are popular among audiences and have a high potential for financial success.
                      """)
    
def text_transition_inflation():
    apply_gradient_color_small('What about the value of money over time?')
    texts.format_text("""Inflation is the rate at which the general level of prices for goods and services rises, 
                      leading to a decrease in the purchasing power of a nation's currency. 
                      In other words, the same amount of money will buy fewer goods and services over time.
                      """)
    texts.format_text("""Therefore, to truly see which genres make the most money, we need to take into account inflation over the years ! 
                        """)
    
def text_explanation_inflation():
    texts.format_text("""A movie making $1 million in 1980 is very different from making $1 million today! 
                      So we adjusted the revenues : we chose 2023 as our reference point, used Consumer Price Index (CPI) data 
                      to track inflation over time, and applied a formula to normalize revenues:
                      """)
    st.latex(r"\text{Adjusted Revenue} = \text{Original Revenue} \times \frac{\text{CPI}_{2023}}{\text{CPI}_{\text{Movie Year}}}")
    texts.format_text(""" This adjustment helps us to compare movies across different decades fairly, 
                    understand true financial impact in today's terms and make more accurate assessments of commercial success.
                      """)
    texts.format_text("""Now, let's see the impact of the inflation by inspecting the distribution of mean box office revenue by decade!""")
    
def plot_distribution_revenue_by_decade(revenue_by_year):
    fig = go.Figure()

    # Add original mean revenue line
    fig.add_trace(go.Scatter(
        x=revenue_by_year['Year'],
        y=revenue_by_year['Original Mean'],
        mode='lines',
        name='Original Mean',
        line=dict(color='green')
    ))

    # Add adjusted mean revenue line
    fig.add_trace(go.Scatter(
        x=revenue_by_year['Year'],
        y=revenue_by_year['Adjusted Mean'],
        mode='lines',
        name='Adjusted Mean',
        line=dict(color='blue')
    ))

    # Add animation frame
    fig.update_layout(
        title="Mean Revenue by Year",
        xaxis_title="Year",
        yaxis_title="Revenue ($)",
        template="plotly_white",
        height=500,
        updatemenus=[dict(type="buttons", showactive=False,
                        buttons=[dict(label="Play",
                                        method="animate",
                                        args=[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]),
                                dict(label="Pause",
                                        method="animate",
                                        args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate",
                                                    "transition": {"duration": 0}}])])],
        showlegend=True
    )

    frames = [go.Frame(data=[
        go.Scatter(x=revenue_by_year['Year'][:k+1], y=revenue_by_year['Original Mean'][:k+1], mode='lines', line=dict(color='green')),
        go.Scatter(x=revenue_by_year['Year'][:k+1], y=revenue_by_year['Adjusted Mean'][:k+1], mode='lines', line=dict(color='blue'))
    ], name=str(year), layout=go.Layout(yaxis=dict(autorange=True))) for k, year in enumerate(revenue_by_year['Year'])]

    fig.frames = frames

    st.plotly_chart(fig, use_container_width=True)
    texts.format_text(""" We can see that without adjusting for inflation, the mean revenue increases over time.
                        However, when we adjust for inflation, we see that the mean revenue is is not increasing over time.
                      In fact, we notice that beetween 1958 and 1980, the average mean revenue is pretty high! 
                      This could be due to several reasons—and no, it's not just because people in the '60s and '70s had nothing 
                      better to do on a Saturday night. During this golden era, movies were the entertainment event, with no Netflix,
                       TikTok, or video games competing for attention. Blockbusters like Jaws and Star Wars drew crowds like popcorn to butter, 
                      and with no home video options, theaters were the only place to catch these cultural phenomena. Plus, post-WWII baby boomers 
                      were prime movie-goers, and urbanization made cinemas the perfect weekend escape. 
                      Hypothetically, this era also saw studios pouring money into epic productions and massive marketing campaigns that practically screamed, "You HAVE to see this in theaters!"
                      """)
    texts.format_text("""
    <div style="text-align:center;">
        This is why we need to adjust for inflation to truly understand the financial success of a movie.
        Now, let's see the impact of inflation on our mean revenue by genre!
    </div>
""")
    
def plot_comparison_genre_mean_revenue(filtered_df, classified_summaries_inflation_BO, color_dict):
    # Calculate the mean revenues and sort them in descending order for the first dataset
    mean_revenues_1 = filtered_df.groupby('movie_genres')['movie_box_office_revenue'].mean().reset_index()
    mean_revenues_1 = mean_revenues_1.sort_values(by='movie_box_office_revenue', ascending=False)

    # Calculate the mean revenues and sort them in descending order for the second dataset
    mean_revenues_2 = classified_summaries_inflation_BO.groupby('movie_genres')['adjusted_revenue'].mean().reset_index()
    mean_revenues_2 = mean_revenues_2.sort_values(by='adjusted_revenue', ascending=False)

    # Create subplots with shared y-axis
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=(
        'Mean Box Office Revenues by Top 15 Movie Genres',
        'Mean Box Office Revenues by Top 15 Movie Genres (adjusted with inflation)'
    ))

    # Plot the first mean revenues using a bar plot
    fig.add_trace(go.Bar(
        x=mean_revenues_1['movie_genres'],
        y=mean_revenues_1['movie_box_office_revenue'],
        marker_color=[color_dict.get(genre, '#333333') for genre in mean_revenues_1['movie_genres']],
        name='Original Mean'
    ), row=1, col=1)

    # Plot the second mean revenues using a bar plot
    fig.add_trace(go.Bar(
        x=mean_revenues_2['movie_genres'],
        y=mean_revenues_2['adjusted_revenue'],
        marker_color=[color_dict.get(genre, '#333333') for genre in mean_revenues_2['movie_genres']],
        name='Adjusted Mean'
    ), row=1, col=2)

    # Update layout for better appearance
    fig.update_layout(
        title_text='Comparison of Mean Box Office Revenues by Genre',
        title={
            'text': 'Comparison of Mean Box Office Revenues by Genre',
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        xaxis_title='Movie Genres',
        yaxis_title='Mean Box Office Revenue',
        template='plotly_white',
        height=600,
        showlegend=False,
        margin=dict(t=80, l=50, r=50, b=100)  # Adjusted margins for better label visibility
    )

    # Update x-axis labels for both subplots
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=12), title_text='Movie Genres', row=1, col=1)
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=12), title_text='Movie Genres', row=1, col=2)

    # Update y-axis labels
    fig.update_yaxes(tickfont=dict(size=12), title_text='Mean Box Office Revenue', row=1, col=1)

    st.plotly_chart(fig, use_container_width=True)

def text_transition_to_distribution():
    texts.format_text("""We can see that the mean revenues of the genres have changed after adjusting for inflation. 
                      Indeed, even some genres switched places in the ranking such as Comedy and Crime Fiction or Horror and Romance Film. The order of the end of the top 15 is even more impacted!
                      This is particularly important for filmmakers to understand the true financial success of their movies.
                      """)
    
    texts.format_text(""" Therefore, from now on, we will use the adjusted revenues to analyze the financial success of movies.
                      """)
    
    apply_gradient_color_small('Now, let\'s see go back to the distribution of revenues by genre!')
    texts.format_text("""
    <div style="text-align:center;">
        Consider that you are in a mystery movie, we want to find the outliers, the easter eggs that may explain some revenues distributions!
    </div>
""")
    

def plot_genre_distribution_revenue(classified_summaries_inflation_BO, color_dict):
    # Create the violin plot
    fig = go.Figure()

    # Add violin traces for each genre
    for genre in classified_summaries_inflation_BO['movie_genres'].unique():
        genre_data = classified_summaries_inflation_BO[classified_summaries_inflation_BO['movie_genres'] == genre]
        fig.add_trace(go.Violin(
            x=genre_data['movie_genres'],
            y=genre_data['adjusted_revenue'],
            name=genre,
            box_visible=True,
            meanline_visible=True,
            points="all",
            line_color=color_dict.get(genre, '#333333'),
            fillcolor=color_dict.get(genre, '#333333'),
            opacity=0.6
        ))

    # Update layout for better appearance
    fig.update_layout(
        title='Distribution of Box Office Revenues by Top 15 Movie Genres',
        xaxis_title='Movie Genres',
        yaxis_title='Box Office Revenue',
        xaxis_tickangle=-45,
        template='plotly_white',
        height=600,
        showlegend=False,
        margin=dict(t=80, l=50, r=50, b=100)  # Adjusted margins for better label visibility
    )

    st.plotly_chart(fig, use_container_width=True)


def text_transition_to_median():
    texts.format_text(""" We can see that each genre has a lot (and we mean a lot) of variability
                       and outliers when it comes to revenues. It's like comparing apples, oranges, and the occasional
                       flying watermelon. So, relying on the mean revenue for our analysis might not be the brightest idea. 
                      Let's be smarter about this and take a look at the median revenue instead, it's much more robust!
                      """)

def plot_genre_median_revenue(filtered_df, color_dict, adjusted=False, key=0):
    revenue_column = 'adjusted_revenue' if adjusted else 'movie_box_office_revenue'
    median_revenues = filtered_df.groupby('movie_genres')[revenue_column].median().reset_index()
    median_revenues = median_revenues.sort_values(by=revenue_column, ascending=False)
    # Create Plotly bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=median_revenues['movie_genres'],
            y=median_revenues[revenue_column],
            marker_color=[color_dict[genre] for genre in median_revenues['movie_genres']],
            hovertemplate="Genre: %{x}<br>Median Revenue ($): %{y:$,.0f}<extra></extra>"
        )
    ])

    # Update layout for better appearance
    fig.update_layout(
        title={
            'text': f"Median Revenue by Genre {'Adjusted with Inflation' if adjusted else '' }",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Genre",
        yaxis_title= f"Median Revenue {'Adjusted with Inflation' if adjusted else ''} (in $)",
        xaxis_tickangle=-45,
        template="plotly_white",
        height=500,  # Increased height for better visibility
        showlegend=False,
        margin=dict(t=50, l=50, r=50, b=100)  # Adjusted margins for better label visibility
    )
    
    st.plotly_chart(fig, use_container_width=True, key=key)

def text_comment_median_revenue():
    texts.format_text("""Looking at the genres by mean revenue vs. median revenue is like seeing two different movies!
                       The mean ranking puts Fantasy at the top, but when we adjust for the outliers and skewed data, 
                      the Family Film genre takes the lead in terms of median revenue. It seems that while fantasy films 
                      might get a few huge blockbusters (hello, Lord of the Rings), family-friendly movies have a more consistent
                       box office performance. Meanwhile, Indie films—well, they're still the underdog, no matter how you slice it! 
                      Looks like we've got a classic case of “big hits vs. steady earners.”
                      """)

def text_transition_to_profit():
    texts.format_text("""Now that we've seen how the genres perform in terms of revenue, let's take a step further and look at profit.
                       Revenue is great, but it doesn't tell the whole story. A film might gross a lot, but if it cost an arm and a leg to make, 
                      it's not exactly a financial triumph. Profit, on the other hand, takes both the production and marketing costs into account, 
                      giving us a much clearer picture of a film's actual financial success. So, instead of just celebrating big numbers at the box office, 
                      let's see which genres are truly making the most money after all expenses are factored in!
                        """)

def text_intro_profit():
    apply_gradient_color_small('Show Me the Money: A Deep Dive into Movie Profits')
    texts.format_text("""Now that we've seen which genres rake in the big bucks in terms of revenue, it's time to get real about what the 
                      studios actually take home. Profit is the true measure of financial success, so let's break down the numbers and see which genres are
                       not just making noise at the box office, but also filling up the bank accounts. Ready to find out who's really cashing in? Let's dive in!
                      """)
    
def plot_median_and_mean_profit_adjusted(filtered_df, color_dict):
    # Calculate median and mean profit
    median_profit = filtered_df.groupby('movie_genres')['adjusted_profit'].median().reset_index()
    mean_profit = filtered_df.groupby('movie_genres')['adjusted_profit'].mean().reset_index()
    
    # Merge median and mean dataframes
    profit_data = median_profit.merge(mean_profit, on='movie_genres', suffixes=('_median', '_mean'))
    profit_data = profit_data.sort_values(by='adjusted_profit_median', ascending=False)

    # Create Plotly bar chart
    fig = go.Figure()

    # Add median profit bar
    fig.add_trace(go.Bar(
        x=profit_data['movie_genres'],
        y=profit_data['adjusted_profit_median'],
        marker_color=[color_dict[genre] for genre in profit_data['movie_genres']],
        name='Median Profit',
        hovertemplate="Genre: %{x}<br>Median Profit ($): %{y:$,.0f}<extra></extra>"
    ))

    # Add mean profit line
    fig.add_trace(go.Scatter(
        x=profit_data['movie_genres'],
        y=profit_data['adjusted_profit_mean'],
        mode='lines+markers',
        line=dict(color='red', width=2),
        name='Mean Profit',
        hovertemplate="Genre: %{x}<br>Mean Profit ($): %{y:$,.0f}<extra></extra>"
    ))

    # Update layout for better appearance
    fig.update_layout(
        title={
            'text': f"Median and Mean Profit by Genre",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Genre",
        yaxis_title="Profit (in $)",
        xaxis_tickangle=-45,
        template="plotly_white",
        height=500,
        showlegend=True,
        margin=dict(t=50, l=50, r=50, b=100)  # Adjusted margins for better label visibility
    )

    st.plotly_chart(fig, use_container_width=True)


def text_conclusion_profit():
    texts.format_text(""" Looking at the genres by median profit, we see some interesting shifts compared to median revenue. Family Film still holds strong at the top, 
                      proving that consistent, broad appeal translates into profit, not just revenue. However, Fantasy, which ranked high for revenue, drops a bit in 
                      the profit rankings—perhaps those big-budget effects and extensive marketing eat into its earnings more than we thought! Genres like Horror and 
                      Romantic comedy also make a noticeable jump, suggesting that they manage to keep production costs in check while still attracting solid audiences. 
                      On the other hand, Indie films remain at the bottom, which is no surprise given their often limited budgets and niche markets. 
                      Overall, it looks like some genres know how to make their money work harder!
                      """)
    
    texts.format_text("""
        <div style="text-align:center;">
     Overall, it looks like some genres know how to make their money work harder!
        </div>
    """)
def text_intro_time_series():
    apply_gradient_color_small("Profit Through the Ages")
    texts.format_text(""" Now that we've got a snapshot of the genres by median profit, let's turn our attention to how these profits have evolved over time. By examining 
                      the year-by-year progression of mean and median profit across genres, we can uncover trends, shifts in audience preferences, and how the industry's financial
                       landscape has changed. It's time to see which genres have been consistently profitable and which ones have had their moments of glory (or perhaps a few rough patches)
                       as the years have gone by. Let's take a closer look!
                      """)

def plot_genre_profit_evolution(classified_summaries_inflation_BO, top_genre, color_dict):
    # Create a new column for 5-year bins
    classified_summaries_inflation_BO['5_year_bin'] = (classified_summaries_inflation_BO['movie_release_date'] // 5) * 5

    # Create a multiselect menu for genre selection
    selected_genres = st.multiselect('Select Genres', top_genre.index, default=top_genre.index[0])

    # Initialize the figure
    fig = go.Figure()

    # Add traces for each selected genre
    for genre in selected_genres:
        genre_df = classified_summaries_inflation_BO[classified_summaries_inflation_BO['movie_genres'] == genre]
        median_profit_by_bin = genre_df.groupby('5_year_bin')['adjusted_profit'].median().reset_index()
        mean_profit_by_bin = genre_df.groupby('5_year_bin')['adjusted_profit'].mean().reset_index()

        fig.add_trace(go.Scatter(
            x=median_profit_by_bin['5_year_bin'],
            y=median_profit_by_bin['adjusted_profit'],
            mode='lines+markers',
            name=f'{genre} Median Profit',
            line=dict(color=color_dict.get(genre, '#333333')),
            visible=True
        ))

    # Update layout for better appearance
    fig.update_layout(
        title='Evolution of Median Profit by Genre',
        xaxis_title='Year',
        yaxis_title='Adjusted Profit ($)',
        template='plotly_white',
        height=600,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

def text_conclusion_time_series():
    texts.format_text("""
    <div style="text-align:center;">
        It's noticeable here, that in general, and for all movie genres, 
        same tendency is observed in terms of median profit with inflation adjustment, 
        they are overall all most profitable between the 60s and 80s. This is really interesting,
        It's during this period that lots of masterpieces came out in cinemas, such as the Godfather or Star Wars and many more!
    </div>
    """)  

    texts.format_text("""
    <div style="text-align:center;">
        Now that we have analyzed genres and their related revenues and profits, we ask ourselves, what really makes a movie profitable? 
        Is there another criterion or recipe that we may have missed? 
        Could it be the star power of the actors, the director's vision, or perhaps the marketing strategy? 
        Or is there something more intrinsic to the movie itself, like its plot structure, that holds the key to financial success? 
        Let's dive deeper and analyze the plot structures of movies to uncover the secrets behind their success!
    </div>
""")  




#### UTILS ####
def return_processed_genre_df(movies):
    genre_exploded = movies.explode('movie_genres')
    genre_counts = genre_exploded['movie_genres'].value_counts()
    genre_percentages = (genre_counts / genre_counts.sum()) * 100
    # top 15 genres
    top_genre = genre_percentages.head(15)
    # Define a consistent color palette
    palette = sns.color_palette("husl", 15)
    #color_dict = {genre: palette[i] for i, genre in enumerate(top_genre.index)}
    color_dict = {genre: f'rgb({int(p[0]*255)}, {int(p[1]*255)}, {int(p[2]*255)})' for genre, p in zip(top_genre.index, palette)}
    filtered_df = genre_exploded[genre_exploded['movie_genres'].isin(top_genre.index)]
    mean_revenues = genre_exploded.groupby('movie_genres')['movie_box_office_revenue'].mean().reset_index()
    mean_revenues = mean_revenues.sort_values(by='movie_box_office_revenue', ascending=False)
    median_revenues = genre_exploded.groupby('movie_genres')['movie_box_office_revenue'].median().reset_index()
    median_revenues = median_revenues.sort_values(by='movie_box_office_revenue', ascending=False)
    return genre_exploded, mean_revenues, median_revenues, color_dict, top_genre, filtered_df
    
def load_processed_inflation():
    df_inflation = pd.read_csv('data/raw/inflation_data.csv')
    df_inflation['date'] = pd.to_datetime(df_inflation['date'])
    df_inflation = df_inflation.set_index('date')
    df_inflation = df_inflation.reset_index()
    df_inflation['year'] = df_inflation['date'].dt.year
    df_inflation = df_inflation.groupby('year')['value'].mean()
    base_year = 2023
    cpi_base = df_inflation[base_year]
    df_inflation = cpi_base / df_inflation
    return df_inflation

def process_inflation_data(movies, df_inflation, year_range=None, selected_genres=None):
    # Filter by year range if provided
    if year_range:
        movies = movies[(movies['movie_release_date'] >= year_range[0]) & 
                       (movies['movie_release_date'] <= year_range[1])]
    
    if selected_genres:
        movies = movies[movies['movie_genres'].apply(lambda x: any(genre in x for genre in selected_genres))]
    movies['adjusted_revenue'] = movies.apply(
        lambda x: x['movie_box_office_revenue'] * df_inflation.get(x['movie_release_date'], 1), axis=1
    )

    # Group by decade and calculate the required statistics
    revenue_by_year = movies.groupby('movie_release_date').agg({
        'movie_box_office_revenue': ['mean', 'sum', 'max'],
        'adjusted_revenue': ['mean', 'sum', 'max']
    }).reset_index()
    revenue_by_year.columns = ['Year', 'Original Mean', 'Original Sum', 'Original Max',
                              'Adjusted Mean', 'Adjusted Sum', 'Adjusted Max']
    return revenue_by_year

def add_inflation(filtered_df, df_inflation):
    copy = filtered_df.copy()
    copy['adjusted_revenue'] = copy.apply(
        lambda x: x['movie_box_office_revenue'] * df_inflation.get(x['movie_release_date'], 1), axis=1
    )

    copy['adjusted_budget'] = copy.apply(
        lambda x: x['budget'] * df_inflation.get(x['movie_release_date'], 1)if pd.notna(x['budget']) else np.nan, axis=1
    )

    copy['adjusted_profit'] = copy['adjusted_revenue'] - copy['adjusted_budget']
    return copy