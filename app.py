import streamlit as st
import pandas as pd
import ast
import plotly.express as px

st.set_page_config(
    page_title="Cinematic Moral Dilemmas",
    page_icon="🎬",
    layout="wide",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main, .reportview-container {
        background: linear-gradient(135deg, #f0f0f0, #fafafa);
        padding-left: 50px;
        padding-right: 50px;
    }
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: "Helvetica Neue", sans-serif;
        font-weight: 600;
    }
    p, li, div, input, select, label, button {
        font-family: "Helvetica", sans-serif;
        font-size: 16px;
        color: #FFF
    }
    .reportview-container .markdown-text-container {
        line-height: 1.6;
    }
    .stPlotlyChart {
        margin-top: 30px;
        margin-bottom: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/movies_with_classifications.csv')
    return df

movies = load_data()

def parse_list_column(df, col):
    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df

for col in ['movie_genres', 'movie_countries', 'movie_languages']:
    if col in movies.columns:
        movies = parse_list_column(movies, col)

if 'budget' in movies.columns:
    movies['profit'] = movies['movie_box_office_revenue'] - movies['budget']

# Plot functions
def plot_yearly_revenue(df):
    if df.empty:
        return None
    revenue_by_year = df.groupby('movie_release_date')['movie_box_office_revenue'].sum().reset_index()
    fig = px.line(revenue_by_year, x='movie_release_date', y='movie_box_office_revenue', 
                  title='Total Box Office Revenue by Year', markers=True,
                  template='simple_white')
    fig.update_traces(line_color='#D62728')
    return fig

def plot_genre_distribution(df, top_n=15):
    if df.empty:
        return None
    exploded = df.explode('movie_genres')
    top_genres = exploded['movie_genres'].value_counts().head(top_n)
    if len(top_genres) == 0:
        return None
    fig = px.bar(
        x=top_genres.index, 
        y=top_genres.values, 
        title='Top Genres Distribution',
        labels={'x':'Genre', 'y':'Count'},
        template='simple_white',
        color=top_genres.index,
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig.update_layout(xaxis_tickangle=45, showlegend=False)
    return fig

def plot_plot_structure_distribution(df):
    if df.empty:
        return None
    ps_counts = df['plot_structure'].value_counts()
    if ps_counts.empty:
        return None
    fig = px.bar(
        ps_counts, 
        x=ps_counts.index, 
        y=ps_counts.values,
        title='Plot Structure Distribution',
        labels={'x':'Plot Structure', 'y':'Count'},
        template='simple_white',
        color=ps_counts.index,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_layout(xaxis_tickangle=45, showlegend=False)
    return fig

def plot_revenue_by_plot_structure(df):
    if df.empty:
        return None
    median_rev = df.groupby('plot_structure')['movie_box_office_revenue'].median().sort_values(ascending=False)
    if median_rev.empty:
        return None
    fig = px.bar(
        x=median_rev.index,
        y=median_rev.values,
        title='Median Box Office Revenue by Plot Structure',
        labels={'x':'Plot Structure', 'y':'Median Revenue'},
        template='simple_white',
        color=median_rev.index,
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig.update_layout(xaxis_tickangle=45, showlegend=False)
    return fig

def plot_profit_by_genre(df):
    if df.empty or 'profit' not in df.columns:
        return None
    exploded = df.explode('movie_genres')
    exploded = exploded.dropna(subset=['movie_genres'])
    if exploded.empty:
        return None
    median_profit = exploded.groupby('movie_genres')['profit'].median().sort_values(ascending=False).head(15)
    if median_profit.empty:
        return None
    fig = px.bar(
        x=median_profit.index, 
        y=median_profit.values,
        title='Median Profit by Genre (Top 15)',
        labels={'x':'Genre', 'y':'Median Profit'},
        template='simple_white',
        color=median_profit.index,
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig.update_layout(xaxis_tickangle=45, showlegend=False)
    return fig

# Header Section with Image
st.title("🎬 Cinematic Moral Dilemmas")
st.subheader("A Data-Driven Exploration of Movie Narratives, Genres, and Success")

st.markdown("""
Welcome to our in-depth exploration of cinematic narratives. Adjust filters for each section to explore different aspects of the data independently.
""")

st.markdown("---")

# ===================== YEARLY REVENUE SECTION =====================
st.header("Evolution of Box Office Revenue Over Time")
st.markdown("Customize the filters below to explore revenue trends across different time periods.")

with st.expander("Filters"):
    all_years = sorted(movies['movie_release_date'].dropna().unique())
    if all_years:
        year_min, year_max = int(min(all_years)), int(max(all_years))
    else:
        year_min, year_max = (1900, 2020)
    selected_year_range = st.slider("Select Year Range", min_value=year_min, max_value=year_max, value=(1980, 2000))

yearly_filtered = movies[
    (movies['movie_release_date'] >= selected_year_range[0]) &
    (movies['movie_release_date'] <= selected_year_range[1])
]

fig_yearly = plot_yearly_revenue(yearly_filtered)
if fig_yearly:
    st.plotly_chart(fig_yearly, use_container_width=True)
else:
    st.info("No data available for the selected year range.")

st.markdown("---")

# ===================== GENRE ANALYSIS SECTION =====================
st.header("Genre Landscape")
st.markdown("Explore which genres dominate certain periods or filter by a specific set of genres.")

with st.expander("Filters"):
    # Filter by genres and year again for genre analysis
    selected_genres = st.multiselect(
        "Select Genres",
        sorted(list({g for lst in movies['movie_genres'].dropna() for g in lst}))
    )
    # Also allow year filtering here for genre distribution
    genre_year_range = st.slider("Select Year Range for Genre Analysis",
                                 min_value=year_min, max_value=year_max, 
                                 value=(1980, 2000))

genre_filtered = movies[
    (movies['movie_release_date'] >= genre_year_range[0]) &
    (movies['movie_release_date'] <= genre_year_range[1])
]

if selected_genres:
    genre_filtered = genre_filtered[genre_filtered['movie_genres'].apply(
        lambda gs: any(g in gs for g in selected_genres) if isinstance(gs, list) else False
    )]

fig_genre = plot_genre_distribution(genre_filtered)
if fig_genre:
    st.plotly_chart(fig_genre, use_container_width=True)
else:
    st.info("No data available for the selected genres or year range.")

st.markdown("---")

# ===================== PLOT STRUCTURE ANALYSIS SECTION =====================
st.header("Plot Structures & Narratives")

st.markdown("""
Beyond genres, narrative structures (Hero’s Journey, Quest for Vengeance, etc.) can shape audience reception.
Filter by specific plot structures or time periods to explore their frequency and median revenue.
""")

with st.expander("Filters"):
    available_structures = movies['plot_structure'].dropna().unique().tolist()
    selected_structures = st.multiselect("Select Plot Structures", available_structures)
    structure_year_range = st.slider("Select Year Range for Plot Structures",
                                     min_value=year_min, max_value=year_max,
                                     value=(1980, 2000))

structure_filtered = movies[
    (movies['movie_release_date'] >= structure_year_range[0]) &
    (movies['movie_release_date'] <= structure_year_range[1])
]

if selected_structures:
    structure_filtered = structure_filtered[structure_filtered['plot_structure'].isin(selected_structures)]

fig_structure_dist = plot_plot_structure_distribution(structure_filtered)
fig_structure_rev = plot_revenue_by_plot_structure(structure_filtered)

if fig_structure_dist:
    st.plotly_chart(fig_structure_dist, use_container_width=True)
else:
    st.info("No plot structure data available for the selected filters.")

if fig_structure_rev:
    st.plotly_chart(fig_structure_rev, use_container_width=True)
else:
    st.info("No revenue data available for the selected plot structures and year range.")

st.markdown("---")

# ===================== PROFIT ANALYSIS SECTION =====================
st.header("Profitability: Does Genre or Story Pattern Influence Success?")

with st.expander("Filters"):
    # Filter by year and optionally by genre for the profit analysis
    profit_year_range = st.slider("Select Year Range for Profit Analysis",
                                  min_value=year_min, max_value=year_max,
                                  value=(1980, 2000))
    profit_selected_genres = st.multiselect(
        "Filter Genres for Profit Analysis",
        sorted(list({g for lst in movies['movie_genres'].dropna() for g in lst}))
    )

profit_filtered = movies[
    (movies['movie_release_date'] >= profit_year_range[0]) &
    (movies['movie_release_date'] <= profit_year_range[1])
]

if profit_selected_genres:
    profit_filtered = profit_filtered[profit_filtered['movie_genres'].apply(
        lambda gs: any(g in gs for g in profit_selected_genres) if isinstance(gs, list) else False
    )]

fig_profit = plot_profit_by_genre(profit_filtered)
if fig_profit:
    st.plotly_chart(fig_profit, use_container_width=True)
else:
    st.info("No profit data available for the selected filters.")

st.markdown("---")

# ===================== DATA SAMPLE SECTION =====================
st.header("A Peek at the Data")

with st.expander("Filters"):
    # Let the user filter by a simple condition or search
    sample_year_range = st.slider("Select Year Range for Data Sample",
                                  min_value=year_min, max_value=year_max,
                                  value=(1980, 2000))
    search_genre = st.selectbox(
        "Filter Data Sample by One Genre",
        ["(None)"] + sorted(list({g for lst in movies['movie_genres'].dropna() for g in lst}))
    )
    search_plot_struct = st.selectbox(
        "Filter Data Sample by Plot Structure",
        ["(None)"] + sorted(movies['plot_structure'].dropna().unique().tolist())
    )

data_filtered = movies[
    (movies['movie_release_date'] >= sample_year_range[0]) &
    (movies['movie_release_date'] <= sample_year_range[1])
]

if search_genre != "(None)":
    data_filtered = data_filtered[data_filtered['movie_genres'].apply(
        lambda gs: search_genre in gs if isinstance(gs, list) else False
    )]

if search_plot_struct != "(None)":
    data_filtered = data_filtered[data_filtered['plot_structure'] == search_plot_struct]

st.write("Below is a sample of the filtered dataset:")
st.dataframe(data_filtered.head(50))

st.markdown("---")

st.markdown("""
**Note:** Each section above has its own independent filter controls, allowing you to explore the dataset from multiple angles without affecting the other sections.
""")
