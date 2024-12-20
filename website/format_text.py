import streamlit as st


def format_text(text, size="18px", bottom_margin="16px"):
    # Use Markdown directly without embedding HTML in the format text
    st.markdown(f"""<div class='justified-text' style='text-align: justify; font-size: {size}; margin-bottom: {bottom_margin};'>{text}</div>""", unsafe_allow_html=True)

def regression_interpretation():
    format_text("""
        The model achieves an R-squared value of 0.384 for the training set and 0.1803 for the test set, indicating poor generalization. 
                However, we can still derive valuable insights from the significant features of the model. 
                Here are the key takeaways based on the features with a p-value less than 0.05:
    """)
    format_text("- **Movie Runtime:** Longer movies tend to have higher adjusted profits.")
    format_text("- **Rating Score:** Higher-rated movies are associated with higher adjusted profits.")
    format_text("- **Adjusted Budget:** Movies with higher budgets tend to have higher adjusted profits.")
    format_text("- **Genres:** Certain genres such as Action/Adventure, Adventure, Children's, Family Film, and Space Opera are positively associated with higher adjusted profits, while genres like Crime Thriller, Drama, and Western are negatively associated.")
    format_text("- **Plot Clusters:** Specific plot clusters, such as Cluster 4 (Sci-fi or adventure narratives set in space or otherworldly environments) and Cluster 9 (Domestic dramas with family relationships at the center, often involving parents, spouses, and home life), show significant positive associations with adjusted profits, indicating that the plot clusters may capture some of the complexity in the data and help determine whether a movie will be a financial success. ")

    format_text("It is important to note that the significance of some genres and plot clusters may be influenced by the fact that only a few movies belong to these categories. This limited representation can lead to higher variability and potentially significant results due to chance.")
    format_text("""Given that plot structures and genres alone don’t reliably predict financial success, we focus more on budget (with a p-value of 0.000) as a more direct and significant predictor. 
                To improve its predictive power, we apply a feature augmentation technique. 
                While movie ratings show a higher coefficient, budget is a more straightforward measure for investors and can be predicted before production, unlike ratings.""")
    format_text("""By augmenting the budget feature, we see an improvement in the model’s performance, with the training R-squared increasing to 0.397. 
                This suggests that budget explains more of the variance in the data, though the model still struggles with generalization.""")
    
    format_text("""We now shift focus to a more direct analysis of the relationship between budget and profit, which provides clearer insights for investors.""")

def int():
    format_text("We fit a linear regression model using the following:")
    format_text("- **Features**: movie_release_date, budget, rating_score, producer, movie_genres, movie_countries, plot structures, and plot clusters")
    format_text("- **Predicted variable**: adjusted_profit")
    st.markdown('''<style>
        [data-testid="stMarkdownContainer"] ul{
            padding-left:40px;
        }
    </style>''', unsafe_allow_html=True)

def int2():
    st.markdown("- Features: movie_release_date, budget, rating_score, producer, movie_genres, movie_countries, plot structures and plot clusters")
    st.markdown("- Predicted variable: budget")

    st.markdown('''
    <style>
    [data-testid="stMarkdownContainer"] ul{
        padding-left:40px;
    }
    </style>
    ''', unsafe_allow_html=True)