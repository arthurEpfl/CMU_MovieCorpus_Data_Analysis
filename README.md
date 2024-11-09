# Cinematic Moral Dilemmas

# Team Adarable 

## Abstract
## Abstract
The project investigates how different plot structures and narrative formulas affect a movie’s box office success. It explores how these structures evolve over time and across genres. By applying natural language processing (NLP) techniques to classify plot summaries into various narrative archetypes, the project analyzes trends over different periods and examines whether certain plot structures tend to be more financially successful.

The project involves scraping and merging IMDb data with the original dataset. The data is then preprocessed to handle missing values and convert data types.

Additionally, the project explores the idea that certain plot structures are frequently recycled across decades and genres. It investigates how often narrative formulas such as the hero's journey, love triangles, or rags-to-riches stories are reused and whether this repetition correlates with box office success. Using the provided plot summaries and metadata, the project applies NLP to identify recurring narrative structures and tracks how the popularity of these plots has shifted over time and across genres. The goal is to understand how audiences respond to such plot structures and whether this appeal varies across demographic segments. This approach offers insights into societal preferences and how narrative structures impact a film’s commercial success.


## Research Questions
1. How do different plot structures and narrative formulas impact a movie’s box office revenue?
2. What are the most common genres associated with various narrative archetypes?
3. How has the popularity of different plot structures evolved over different time periods?
4. What are the key factors that contribute to the commercial success of movies with different narrative structures?
5. How do recurring narrative structures correlate with box office success across different genres and time periods?

## Proposed Additional Dataset
-[IMDb 5000](https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset) (to complete the revenue where it was missing in the original dataset from CMU Movie Summary Corpus)


## Methods
1. **Data Collection and Cleaning**: Gather additional datasets from sources like Kaggle and clean the data to ensure consistency and accuracy. This includes handling missing values and converting data types.
2. **Data Scraping**: Use the ImdbScraper class to scrape additional data from IMDB, extracting useful information such as revenue, budget, and ratings.
3. **Data Merging**: Merge the collected datasets with the IMDB data to create a comprehensive dataset for analysis, ensuring that different versions of movies are accounted for.
4. **Exploratory Data Analysis (EDA)**: Perform EDA to understand the distribution and characteristics of the data, including checking for duplicates and missing values.
5. **Text Preprocessing**: Apply natural language processing (NLP) techniques to clean and preprocess plot summaries, including tokenization, stopword removal, and lemmatization.
6. **Narrative Archetype Classification**: Use NLP techniques to classify plot summaries into various narrative archetypes.
7. **Statistical Analysis**: Use statistical methods to compare the box office revenue of movies with different narrative structures and to identify trends over different time periods.  
8. **Visualization**: Create visualizations to illustrate the findings and provide insights into the data, such as bar plots, line plots, and stacked bar plots for genres and narrative archetypes.
9. **Machine Learning**: Build predictive models to identify key factors contributing to the success of movies with different narrative structures, using features engineered from the data.
10. **Temporal Analysis**: Analyze how the popularity of different plot structures and narrative formulas has evolved over different time periods and across genres.

## Proposed Timeline
- **Week 1 (16.11.2024-22.11.2024)**: Finalize data collection, preprocessing, cleaning and finalize the first visualisations.    
- **Week 2 (23.11.2024-29.11.2024)**: Narrative Archetype Classification, using NLP techniques to classify plot summaries into various narrative archetypes.  
- **29.11.2024** : Homework 2 to be submitted.
- **Week 3 (30.11.2024-06.12.2024)** Statistical Analysis, 
using statistical methods to compare the box office revenue of movies with different narrative structures and to identify trends over different time periods.  
- **Week 4 (07.12.2024-13.12.2024)** Visualization, 
creating visualizations to illustrate the findings and provide insights into the data, such as bar plots, line plots, and stacked bar plots for genres and narrative archetypes.  
- **Week 5 (14.12.2024-20.12.2024)** Temporal Analysis, 
analyzing how the popularity of different plot structures and narrative formulas has evolved over different time periods and across genres. Finalisation of project structure and design.  
- **20.12.2024** : Project to be submitted.

## Organization Within the Team  

-Finalize data collection, preprocessing, cleaning, scraping and merging : Anders & Malak

-Scrape additional data from IMDB using the ImdbScraper class : Malak 

-Classify plot summaries into various narrative archetypes using NLP techniques : Adam & Arthur  

-Perform statistical analysis to compare the box office revenue of movies with different narrative structures and identify trends over different time periods : Adam & Arthur  

-Create visualizations to illustrate the findings and provide insights into the data, such as bar plots, line plots, and stacked bar plots for genres and narrative archetypes : Sven

-Analyze how the popularity of different plot structures and narrative formulas has evolved over different time periods and across genres, finalize the project structure and design : Anders & Sven

