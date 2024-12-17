import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ast

# Helper functions
def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val

# Plot functions
def plot_movie_release_years(movies):
    movies = movies[movies['movie_release_date'] >= 1920]
    revenue_counts_by_year = movies.groupby('movie_release_date').size().reset_index(name='count')
    fig = px.bar(revenue_counts_by_year, x='movie_release_date', y='count', 
                 title='Count of Movies with Available Box Office Revenue per Year',
                 labels={'movie_release_date': 'Year', 'count': 'Count of Movies'},
                 color_discrete_sequence=['skyblue'])
    fig.update_layout(xaxis_title='Year', yaxis_title='Count of Movies', 
                      xaxis_tickangle=-45, xaxis_tickmode='array', 
                      xaxis_tickvals=revenue_counts_by_year['movie_release_date'][::5])
    return fig

def plot_box_office_revenue_by_year(movies):
    movies = movies[movies['movie_release_date'] >= 1920]
    revenue_by_year = movies.groupby('movie_release_date')['movie_box_office_revenue'].sum().reset_index()
    fig = px.line(revenue_by_year, x='movie_release_date', y='movie_box_office_revenue', 
                  title='Total Box Office Revenue by Year',
                  labels={'movie_release_date': 'Year', 'movie_box_office_revenue': 'Total Revenue'},
                  markers=True)
    fig.update_layout(xaxis_title='Year', yaxis_title='Total Revenue', 
                      xaxis_tickangle=-45, xaxis_tickmode='array', 
                      xaxis_tickvals=revenue_by_year['movie_release_date'][::5])
    return fig

def plot_language_distribution(movies):
    movies = movies[movies['movie_release_date'] >= 1920]
    movies['movie_languages'] = movies['movie_languages'].apply(safe_literal_eval)
    movies['year_interval'] = (movies['movie_release_date'] // 5) * 5
    movies_languages_exploded = movies.explode('movie_languages')
    language_year_pivot = movies_languages_exploded.pivot_table(index='year_interval', columns='movie_languages', aggfunc='size', fill_value=0)
    language_counts = movies_languages_exploded['movie_languages'].value_counts()
    valid_languages = language_counts[language_counts >= 50].index
    language_year_pivot = language_year_pivot[valid_languages]
    language_year_pivot = language_year_pivot.reset_index().melt(id_vars='year_interval', var_name='Language', value_name='Count')
    fig = px.bar(language_year_pivot, x='year_interval', y='Count', color='Language', 
                 title='Distribution of Top 30 Movie Languages Across 5-Year Intervals',
                 labels={'year_interval': 'Year Intervals', 'Count': 'Number of Movies'},
                 color_discrete_sequence=px.colors.qualitative.T10)
    fig.update_layout(xaxis_title='Year Intervals', yaxis_title='Number of Movies', 
                      legend_title='Languages', barmode='stack')
    return fig

def plot_top_countries(movies):
    movies['movie_countries'] = movies['movie_countries'].apply(safe_literal_eval)
    movies_countries_exploded = movies[movies['movie_countries'].map(len) > 0].explode('movie_countries')
    top_countries = movies_countries_exploded['movie_countries'].value_counts().head(10)
    fig = px.bar(top_countries, x=top_countries.values, y=top_countries.index, orientation='h',
                 title='Top 10 Countries by Number of Movies Produced',
                 labels={'x': 'Number of Movies', 'y': 'Countries'})
    fig.update_layout(xaxis_title='Number of Movies', yaxis_title='Countries')
    return fig

def plot_runtime_and_release_year_distributions(movies):
    movies = movies[movies['movie_release_date'] >= 1920]
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        'Boxplot of Movie Runtime', 'Boxplot of Movie Release Year'))
    runtime_box = px.box(movies[movies['movie_runtime'] < 300], x='movie_runtime', color_discrete_sequence=['blue'])
    for trace in runtime_box['data']:
        fig.add_trace(trace, row=1, col=1)
    release_year_box = px.box(movies, x='movie_release_date', color_discrete_sequence=['green'])
    for trace in release_year_box['data']:
        fig.add_trace(trace, row=1, col=2)
    fig.update_layout(height=400, width=1000, title_text='Runtime and Release Year Boxplots', showlegend=False)
    fig.update_xaxes(title_text='Runtime (minutes)', row=1, col=1)
    fig.update_xaxes(title_text='Release Year', row=1, col=2)
    fig.update_xaxes(range=[0, 300], row=1, col=1)
    fig.update_xaxes(range=[1920, 2020], row=1, col=2)
    return fig

def plot_genre_distribution(genre_exploded):
    top_10_genres = genre_exploded['movie_genres'].value_counts().head(10)

    # Create Plotly bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=top_10_genres.index,
            y=top_10_genres.values,
            marker_color=px.colors.qualitative.Set3,  # Using Plotly's built-in color sequence
            hovertemplate="Genre: %{x}<br>Count: %{y}<extra></extra>"
        )
    ])

    # Update layout for better appearance
    fig.update_layout(
        title={
            'text': 'Top 10 Movie Genres',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Genre",
        yaxis_title="Count",
        xaxis_tickangle=-45,
        template="plotly_dark",  # Matches your dark theme
        height=500,  # Increased height for better visibility
        showlegend=False,
        margin=dict(t=50, l=50, r=50, b=100)  # Adjusted margins for better label visibility
    )
    return fig

def plot_genre_revenue(mean_revenues):
    # Create interactive bar plot with Plotly
    fig = px.bar(mean_revenues,
                 x='movie_genres',
                 y='movie_box_office_revenue',
                 title='Mean Box Office Revenues by Top 15 Movie Genres',
                 labels={
                     'movie_genres': 'Movie Genres',
                     'movie_box_office_revenue': 'Mean Box Office Revenue'
                 },
                 color='movie_box_office_revenue',
                 color_continuous_scale='Viridis')
    
    # Update layout with dark theme
    fig.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        xaxis_tickangle=-45,
        template="plotly_dark",  # Use dark template as base
        height=600,
        showlegend=False,
        hoverlabel=dict(
            bgcolor="black",
            font_color="white",
            bordercolor="white"
        ),
        hovermode='x unified',
        margin=dict(t=50, l=50, r=50, b=100),
        title=dict(
            font=dict(color='white')
        ),
        xaxis=dict(
            gridcolor='rgba(128,128,128,0.2)',
            tickfont=dict(color='white'),
            title_font=dict(color='white')
        ),
        yaxis=dict(
            gridcolor='rgba(128,128,128,0.2)',
            tickfont=dict(color='white'),
            title_font=dict(color='white'),
            tickformat="$,.0f",
            title="Mean Box Office Revenue ($)"
        )
    )
    
    # Update hover template
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>" +
                      "Mean Revenue: $%{y:,.0f}<br>" +
                      "<extra></extra>",
        marker_line_color='white',
        marker_line_width=1
    )
    
    return fig

def plot_revenue_distribution(data, adjusted=True):
    """Plot violin plots showing revenue distribution by year ranges"""
    # Create decade bins
    data['decade'] = (data['Year'] // 10) * 10
    column = 'Adjusted Mean' if adjusted else 'Original Mean'
    
    fig = go.Figure()
    
    fig.add_trace(go.Violin(
        x=data['decade'],
        y=data[column],
        box_visible=True,
        meanline_visible=True,
        points="all"
    ))
    
    fig.update_layout(
        title=f"Distribution of {'Adjusted' if adjusted else 'Original'} Revenue by Decade",
        xaxis_title="Decade",
        yaxis_title="Revenue ($)",
        template="plotly_dark",
        height=500
    )
    
    return fig

def plot_genre_revenue_trends(data, genres, adjusted=True):
    """Plot revenue trends by genre over time"""
    fig = go.Figure()
    
    for genre in genres:
        genre_data = data
        column = 'Adjusted Mean' if adjusted else 'Original Mean'
        
        fig.add_trace(go.Scatter(
            x=genre_data['Year'],
            y=genre_data[column],
            name=genre,
            mode='lines+markers'
        ))
    
    fig.update_layout(
        title=f"{'Adjusted' if adjusted else 'Original'} Revenue Trends by Genre",
        xaxis_title="Year",
        yaxis_title="Revenue ($)",
        template="plotly_dark",
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def plot_revenue_heatmap(data, adjusted=True):
    """Create a heatmap showing revenue patterns across years and genres"""
    column = 'Adjusted Mean' if adjusted else 'Original Mean'
    
    # Pivot data for heatmap
    print(data.columns)
    heatmap_data = data.pivot(
        index='Genre',
        columns='Year',
        values=column
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Viridis'
    ))
    
    fig.update_layout(
        title=f"{'Adjusted' if adjusted else 'Original'} Revenue Heatmap",
        xaxis_title="Year",
        yaxis_title="Genre",
        template="plotly_dark",
        height=600
    )
    
    return fig

def plot_revenue_bubble(data, adjusted=True):
    """Create a bubble chart showing revenue, count, and average rating"""
    column = 'Adjusted Mean' if adjusted else 'Original Mean'
    data['Count'] = data.groupby('Year')['Genre'].transform('count')
    fig = go.Figure(data=go.Scatter(
        x=data['Year'],
        y=data[column],
        mode='markers',
        marker=dict(
            size=data['Count'],
            sizemode='area',
            sizeref=2.*max(data['Count'])/(40.**2),
            sizemin=4,
            colorscale='Viridis',
            showscale=True
        ),
        text=data['Genre'],
        hovertemplate=
        "<b>%{text}</b><br>" +
        "Year: %{x}<br>" +
        "Revenue: $%{y:,.0f}<br>" +
        "Count: %{marker.size:,}<br>" +
        "<extra></extra>"
    ))
    
    fig.update_layout(
        title=f"{'Adjusted' if adjusted else 'Original'} Revenue Bubble Chart",
        xaxis_title="Year",
        yaxis_title="Revenue ($)",
        template="plotly_dark",
        height=600
    )
    
    return fig

def plot_inflation_comparison(revenue_data, metric='mean'):
    """
    Plot comparison of original vs inflation-adjusted revenues
    metric: 'mean', 'sum', or 'max'
    """
    metric = metric.lower()
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        f'Original {metric.title()} Revenue', 
        f'Adjusted {metric.title()} Revenue'
    ))
    
    # Original Revenue
    fig.add_trace(
        go.Scatter(
            x=revenue_data['Year'],
            y=revenue_data[f'Original {metric.title()}'],
            mode='lines+markers',
            name='Original',
            line=dict(color='skyblue')
        ),
        row=1, col=1
    )
    
    # Adjusted Revenue
    fig.add_trace(
        go.Scatter(
            x=revenue_data['Year'],
            y=revenue_data[f'Adjusted {metric.title()}'],
            mode='lines+markers',
            name='Adjusted',
            line=dict(color='salmon')
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        template="plotly_dark",
        title_text=f"Revenue Comparison ({metric.title()})",
        yaxis_title="Revenue ($)",
        yaxis2_title="Adjusted Revenue ($)"
    )
    
    return fig

def plot_silhouette_scores(k_range, scores):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(k_range),
        y=scores,
        mode='lines+markers',
        name='Silhouette Score',
        line=dict(color='skyblue', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='Silhouette Score for Different Numbers of Clusters',
        xaxis_title='Number of Clusters',
        yaxis_title='Silhouette Score',
        template='plotly_dark',
        height=500,
        showlegend=False
    )
    
    return fig

def plot_top_profitable_movies(data, metric='profit'):
    """Plot top profitable movies by profit or profitability ratio"""
    fig = go.Figure(go.Bar(
        x=data[metric],
        y=data['movie_name'],
        orientation='h',
        marker_color=px.colors.sequential.Viridis[3]  # Use a specific color from the Viridis sequence
    ))
    
    title_text = "Top 10 Movies by " + metric.replace("_", " ").title()
    if "adjusted" in metric.lower():
        title_text += " (Inflation Adjusted)"
    
    fig.update_layout(
        title=title_text,
        xaxis_title=metric.replace("_", " ").title(),
        yaxis_title='Movie Name',
        template='plotly_dark',
        height=500,
        margin=dict(t=50, l=50, r=50, b=50)
    )
    
    return fig

def plot_budget_profit_relationship(data, adjusted=True):
    """Plot budget vs profit relationship with inflation adjustment option"""
    budget_col = 'adjusted_budget' if adjusted else 'budget'
    profit_col = 'adjusted_profit' if adjusted else 'profit'
    
    fig = make_subplots(rows=1, cols=2, 
                       subplot_titles=(
                           f'Budget vs Profit ({["Original", "Adjusted"][adjusted]})',
                           f'Budget vs Profit Log Scale ({["Original", "Adjusted"][adjusted]})'
                       ))
    
    # Linear scale plot
    fig.add_trace(
        go.Scatter(
            x=data[budget_col],
            y=data[profit_col],
            mode='markers',
            marker=dict(
                size=5,
                color=data['rating_score'],
                colorscale='Viridis',
                showscale=True
            ),
            name='Movies'
        ),
        row=1, col=1
    )
    
    fig.update_xaxes(type='log', row=1, col=2)
    fig.update_yaxes(type='log', row=1, col=2)
    
    fig.update_layout(
        height=500,
        template='plotly_dark',
        showlegend=False
    )
    
    return fig

def plot_roi_by_budget(roi_data):
    """Plot ROI statistics by budget range"""
    fig = go.Figure()
    
    # Add mean ROI bars
    fig.add_trace(go.Bar(
        name='Mean ROI',
        x=roi_data['budget_bins'],
        y=roi_data['mean'],
        marker_color='skyblue'
    ))
    
    # Add median ROI bars
    fig.add_trace(go.Bar(
        name='Median ROI',
        x=roi_data['budget_bins'],
        y=roi_data['median'],
        marker_color='salmon'
    ))
    
    fig.update_layout(
        title='ROI Statistics by Budget Range',
        xaxis_title='Budget Range',
        yaxis_title='Return on Investment (ROI)',
        template='plotly_dark',
        height=500,
        barmode='group',
        showlegend=True
    )
    
    return fig



