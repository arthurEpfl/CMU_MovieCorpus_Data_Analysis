import pandas as pd
import format_text as texts
import streamlit as st

def intro_text():
    st.subheader('First of all, what is the genre of a movie?')
    texts.format_text(""" The genre of a movie defines its category or type, characterized by shared themes, 
                      storytelling elements, and emotional tone. It helps audiences identify what kind of experience to expect, 
                      such as humor in comedies, suspense in thrillers, or emotional depth in dramas.
                      """)
    texts.format_text(""" Therefore, we look into the distribution of genres in our dataset...
                    """)
    
    
