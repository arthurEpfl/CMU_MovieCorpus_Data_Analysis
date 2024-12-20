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

def conclusion():
    texts.format_text("""In this project, we are decoding the blueprint of a blockbuster by analyzing plot structures and their impact on box office success. Our analysis involved data scraping, preprocessing, exploratory data analysis, natural language processing, and predictive modeling.""")
    texts.format_text("""<li><strong>Impact of Plot Structures on Box Office Revenue</strong Our regression analysis revealed that plot structures were relatively insignificant as a factor contributing to financial success. While certain plot clusters and genres showed some level of significance, this suggests that with higher-quality and more precise plot structure classifications, their contribution to financial success might become more evident. The current application of LLM models for classifying plot structures resulted in highly skewed classifications, indicating room for improvement in both methodology and model accuracy. Enhanced classification techniques could potentially uncover deeper insights into the role of plot structures in determining profitability. </li>
            <li><strong>Genre and Plot Structure Correlation</strong>: We found that specific genres are more likely to feature certain plot structures. For example, drama movies often follow hero's journey.</li>
            <li><strong>Temporal Trends in Plot Structures</strong>: The popularity of different plot structures has evolved over time. While classic structures like the hero's journey remain popular, newer structures are emerging and gaining traction in recent years.</li>
            <li><strong>Inflation Adjustment</strong>: Adjusting revenue and budget data for inflation provided a more accurate comparison across different time periods, allowing for consistent comparisons between budgets, revenues and profits. This adjustment was crucial for understanding the true financial success of movies from different eras. </li>
<li><strong>Key Factors Contributing to Commercial Success</strong>: Budget and rating score emerged as the most significant factors influencing commercial success. Budget, in particular, demonstrated a clear relationship with profit, showing a consistent reduction in risk on average as budget increases. However, lower-budget films exhibited significantly higher variability, underscoring the heightened financial risk and potential reward associated with smaller investments. </li>
""")
    texts.format_text("""Limitations and potential improvements include:""")
    texts.format_text("""<li><strong>Unbiased Dataset for Lower-Budget Films</strong>: The dataset may disproportionately represent successful low-budget films, making them appear more profitable than they typically are. While this skew still captures their variance and risk, it potentially does not accurately reflect the average performance of all low-budget films. </li>
<li><strong>Skewed Classification in LLM Application</strong>: The classification of plot structures using the current LLM model is highly skewed, which could contribute to the lack of significance observed in the relationship between plot structures and financial success.</li>
<li><strong>Simplistic Linear Regression Model</strong>: The use of a simple linear regression model may fail to capture the nuanced and complex relationships within the data, particularly those linked to plot structures and their potential impact on profitability. More advanced modeling techniques could provide deeper insights.</li>
""")