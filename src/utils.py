import ast
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def extract_dict_to_list(entry):
    """
    Extracts values from a dictionary-like string and returns a list of values.
    """
    try:
        entry_dict = ast.literal_eval(entry) 
        return list(entry_dict.values()) 
    except (ValueError, SyntaxError):
        return []

def extract_release_year(date_str):
    """
    Extracts the year from a date string.
    """
    try:
        return pd.to_datetime(date_str).year
    except (ValueError, TypeError):
        try:
            return int(date_str)
        except ValueError:
            return None

def safe_literal_eval(val):
    """
    Safely evaluates a string containing a Python literal (list, dictionary, etc.).
    If evaluation fails, returns the original value.
    """
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
    
def add_scraped_features(scraped_data, filtered_movies_summaries_BO):
    """
    Add features obtained from scraping on the movies with summaries and box office available.
    """
    # Merge the two DataFrames on 'wikipedia_movie_id'
    movies_scraped_data = pd.merge(filtered_movies_summaries_BO, scraped_data, on='wikipedia_movie_id', how='left')

    # Ensure there are no duplicates on 'wikipedia_movie_id'
    movies_scraped_data = movies_scraped_data.drop_duplicates(subset=['wikipedia_movie_id'])

    return movies_scraped_data

exchange_rates = {
    '$': 1.0,
    '€': 1.0970,  # Example rate: 1 EUR = 1.1 USD
    '£': 1.2770,
    'CA$' : 0.7569,
    'A$' : 0.6842,
    '¥' : 0.0071,
    '₩' : 0.0078,
    '₹' : 0.0120,
    'HK$' : 0.1280   # Example rate: 1 GBP = 1.3 USD
}

def convert_currency_to_dollar(df, exchange_rates):
    """
    Convert the budget currency to dollars using the exchange rates provided.
    """
    df['currency_budget_dollar'] = df['budget']*df['currency_budget'].map(exchange_rates)
    return df

def create_graph(data, source_col, target_col, weight_col, source_type='source', target_type='target'):
    """
    Creates a NetworkX graph from a DataFrame.

    Parameters:
        data (DataFrame): Input data containing nodes and edges.
        source_col (str): Column name for source nodes.
        target_col (str): Column name for target nodes.
        weight_col (str): Column name for edge weights.
        source_type (str): Type label for source nodes.
        target_type (str): Type label for target nodes.

    Returns:
        G (nx.Graph): A NetworkX graph.
    """
    G = nx.Graph()
    for index, row in data.iterrows():
        source = row[source_col]
        target = row[target_col]
        weight = row[weight_col]

        # Add source and target nodes with type and weight attributes
        G.add_node(source, type=source_type, weight=weight)
        G.add_node(target, type=target_type, weight=weight)
        G.add_edge(source, target, weight=weight)
    return G


def assign_node_attributes(G, size_factor=3e6, offset=1e3, color_mapping=None):
    """
    Assigns node sizes and colors based on attributes.

    Parameters:
        G (nx.Graph): Input graph.
        size_factor (float): Scaling factor for node size.
        offset (float): Offset for node size visibility.
        color_mapping (dict): Mapping of node types to colors.

    Returns:
        node_colors (list): List of colors for nodes.
        node_sizes (list): List of sizes for nodes.
    """
    if color_mapping is None:
        color_mapping = {'director': '#72A0C1', 'plot': '#90EE90'}

    node_colors = [color_mapping.get(G.nodes[node]['type'], '#D3D3D3') for node in G.nodes]
    node_sizes = [G.nodes[node]['weight'] * 3 / size_factor + offset for node in G.nodes]
    return node_colors, node_sizes

def draw_graph(G, title="Graph Visualization", k=0.5, seed=42, iterations=100):
    """
    Draws a NetworkX graph with attributes.

    Parameters:
        G (nx.Graph): Input graph.
        title (str): Title of the graph.
        k (float): Optimal distance between nodes in spring layout.
        seed (int): Seed for consistent layout.
        iterations (int): Number of iterations for spring layout.
    """
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=k, seed=seed, iterations=iterations)

    # Assign node attributes
    node_colors = [G.nodes[node]['color'] for node in G]
    node_sizes = [G.nodes[node]['size'] for node in G]  # Scale sizes up for visibility

    # Draw nodes, edges, and labels
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.4, edgecolors='k')
    nx.draw_networkx_edges(G, pos, width=1, alpha=0.5, edge_color="gray")
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif', font_color='black')

    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def create_graph2(data):
    """
    Creates a NetworkX graph from a DataFrame, ensuring node attributes such as 'size' aggregate properly.
    """
    G = nx.Graph() 
    node_attributes = {}
    
    size_factor = 1e6
    offset = 10
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
        G.add_edge(row['producer'], row['plot_structure'])

    return G