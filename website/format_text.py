import streamlit as st


def format_text(text, size="18px", bottom_margin="16px"):
    # Use Markdown directly without embedding HTML in the format text
    st.markdown(f"""<div class='justified-text' style='text-align: justify; font-size: {size}; margin-bottom: {bottom_margin};'>{text}</div>""", unsafe_allow_html=True)

def intro():
    format_text("""Is it the incredible acting, the clever marketing, or the relatable themes that stick with us? 
    While all of these play a part, history has shown that the real magic lies in the story—the way 
    it draws us in, connects with us, and keeps us hooked. From the magical world of Harry Potter 
    to the mind-bending twists of Inception, blockbuster movies all have something special in their 
    plots that audiences can’t get enough of. But can we measure that? Is there a way to figure out 
    what makes a story truly successful?""")
    format_text("""The answer lies at the intersection of creativity and data. In a world increasingly dominated by streaming platforms and ever-growing libraries of content, understanding what makes a movie resonate with audiences is more important than ever. Plot success isn’t just about creativity; it’s also about patterns, structures, and underlying formulas that can be identified, analyzed, and even predicted.
    """)
    format_text("""Consider this: while Harry Potter may have captured imaginations with its magical world, what about its storytelling resonates across cultures and generations? And for Inception, is it the characters, the pacing, or the multilayered narrative that drives its massive appeal? By quantifying key storytelling features—like pacing, themes, character arcs, and conflict resolution—we can begin to unravel these mysteries.""")
    format_text("""But this isn’t just about understanding stories; it’s about what we can do with that knowledge. How can filmmakers optimize their scripts to increase box office success? Can studios predict a movie's profitability based on its plot structure? Could aspiring screenwriters use data-driven insights to craft the next big hit?""")


def regression_interpretation():
    format_text("""
        Let’s take a systematic approach to answering this question. By fitting a linear regression model, we explored various factors to understand their influence on a movie's adjusted profit.
    """)
    format_text("Here’s how we structured the model:")
    format_text("<li> <strong>Features considered</strong>: movie release date, budget, rating score, producer, genres, countries of production, plot structures, and plot clusters.</li>")  
    format_text("<li> <strong>Target variable</strong>: adjusted profit.</li>")
    format_text("Although the model achieved an R-squared value of 0.384 on the training set and 0.1803 on the test set—indicating limited generalizability—it still highlighted valuable insights. Below are some of the key findings based on features with statistically significant p-values (less than 0.05):")
    format_text("<li> <strong>Movie Runtime:</strong> Longer movies tend to yield higher adjusted profits. </li>")
    format_text("<li> <strong>Rating Score:</strong> Higher ratings correlate positively with adjusted profits. </li>")   
    format_text("<li> <strong>Adjusted Budget:</strong> Larger budgets are strongly associated with greater financial success. </li>")
    format_text("<li><strong>Genres:</strong> Certain genres such as Action/Adventure, Family Film, and Space Opera are positively linked to profitability. In contrast, genres like Drama, Crime Thriller, and Western tend to show negative associations.</li>")
    format_text("<li><strong>Plot Clusters:</strong> Specific clusters, such as sci-fi adventures (Cluster 4) and domestic dramas centered on family relationships (Cluster 9), exhibit a significant positive impact on adjusted profits.</li>")
    format_text("However, some caveats must be noted. The small sample sizes for certain genres and plot clusters may amplify their variability, making them appear more significant than they might be in a larger dataset.")
    format_text("Interestingly, while genres and plot structures contribute meaningfully, they alone do not reliably predict financial success. Instead, the budget emerged as the most direct and impactful predictor, with a highly significant p-value (0.000). This makes intuitive sense—budgeting decisions are typically made early in a movie's lifecycle and often set the stage for its eventual scale and reach.")
    format_text("To refine the model, we applied feature augmentation techniques, particularly focusing on the budget variable. This boosted the training R-squared to 0.397, further underscoring the centrality of budget as a predictor of financial success.")
    format_text("Ultimately, while other factors like ratings and runtime add nuance, budget remains the clearest guide for investors aiming to make informed decisions about where to allocate their resources. Next, we’ll delve deeper into this relationship to uncover actionable insights for stakeholders.")

def budget_interpretation1():
    format_text("""
        Understanding the relationship between budget and profitability offers key insights into the financial dynamics of the film industry. Our analysis reveals that budget is not only a determinant of profitability but also a strong indicator of financial risk.
    """)

def budget_interpretation2():
    format_text("""An initial visualization of the relationship between budget and profit shows that higher-budget films generally have more consistent profits with less variation. Conversely, lower-budget films exhibit a broader range of outcomes, including both exceptional successes and significant failures.
""")
    format_text("""
        However, the data suggests that while higher budgets reduce risk on average, they do not eliminate it entirely. Among the highest-budget films, there are some notable outliers—big-budget productions that turned into major flops at the box office. This highlights that even substantial investments are not entirely free from financial risk. 
                Some of these movies include: Mars Needs Moms, The 13th Warrior, The Adventures of Pluto Nash and Cloud Atlas all registering losses upwards of 90M$.""")
def ROI_interpretation():
    format_text("""
        When we assess the return on investment (ROI), a clearer pattern emerges. Variance in ROI decreases as budgets increase, suggesting that lower-budget films are inherently riskier ventures. This pattern is evident in the graph, where films with budgets exceeding 100M$ display a distribution similar to those between 10M$ and 100M$, indicating similar risk profiles. Nevertheless, very high-budget films tend to deliver higher absolute profits compared to their moderately high-budget counterparts.""")
    format_text("""To further confirm the observed trends, we performed both ANOVA and Kruskal-Wallis tests, which revealed statistically significant differences in ROI across budget bins. These findings corroborate our hypothesis: budget is a critical factor influencing financial risk and return.
""")

def key_concl():
    format_text("Our analysis suggests the following:")
    format_text("<li> <strong>Risk vs. Reward</strong>: Lower-budget films carry higher financial risk but offer the potential for greater profitability, as evidenced by their higher mean ROI.</li>")
    format_text("<li> <strong>Stability of High Budgets</strong>: Higher-budget films, while not immune to failure, tend to offer more stable returns.</li>") 
    format_text("However, it is important to consider bias: the findings indicate that lower-budget films appear to be more profitable on average. However, this is likely influenced by the skewness of the original dataset. Since we only include movies with available budget and revenue data and rely on a predefined dataset, it is probable that the sample of low-budget films is disproportionately successful compared to what would be observed in a truly random sample. An unbiased random sample would likely reveal a higher risk profile for low-budget films and demonstrate that they are significantly less profitable on average.")
    format_text("")



def apply_gradient_color(text):
    st.markdown(f"""
        <style>
        .title-viridis-light {{
            background: linear-gradient(135deg, #5e4fa2 0%, #3288bd 50%, #66c2a5 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        </style>
        <div style="text-align:center;">
            <h1 class="title-viridis-light">{text}</h1>
        </div>
    """, unsafe_allow_html=True)  


def apply_gradient_color_small(text):
    st.markdown(f"""
        <style>
        .title-viridis-light-small {{
            background: linear-gradient(135deg, #a6bddb 0%, #67a9cf 50%, #3690c0 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 28px;  /* Adjust the font size as needed */
        }}
        </style>
        <div style="text-align:center;">
            <h3 class="title-viridis-light-small">{text}</h3>
        </div>
    """, unsafe_allow_html=True) 

