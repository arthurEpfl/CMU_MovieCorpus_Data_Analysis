# Cinematic Moral Dilemmas

# Team Adarable 

## Abstract
The project investigates how different genres and plot structures affect a movie’s box office success and their evolution over time and genres. By applying NLP techniques to classify plot summaries into narrative archetypes, it analyzes trends and examines the financial success of certain structures.

The project involves scraping and merging IMDb data with the original dataset, preprocessing it to handle missing values and convert data types. It explores extracting plot structures from the plot summaries like the hero's journey, love triangles, and rags-to-riches stories, investigating their relation with box office success.

Using plot summaries and metadata, the project applies NLP to identify narrative structures and tracks their popularity shifts over time and genres. The goal is to understand audience responses to these structures and their appeal across demographic segments, offering insights into societal preferences and the impact of narrative structures on a film’s commercial success.

## Research Questions:
1. How do different genre and plot structures impact a movie’s box office revenue?
2. What are the most common genres associated with various plot structures?
3. How has the popularity of different plot structures evolved over different time periods?
4. What are the key factors that contribute to the commercial success of movies with different plot structures?
5. How do plot structures correlate with box office success across different genres and time periods?

## Proposed Additional Dataset
To enrich our analysis and fill gaps in the original dataset (CMU Movie Summary Corpus), we incorporated the following additional datasets:

1. **[IMDb 5000](https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset)**  
   - **Purpose**: Used to supplement missing revenue information in the original dataset.  
   - **Why**: The CMU Movie Summary Corpus lacked comprehensive financial data for several movies. This dataset provides detailed box office revenue for a broader range of movies.

2. **[US Inflation Dataset](https://www.kaggle.com/datasets/pavankrishnanarne/us-inflation-dataset-1947-present)**  
   - **Purpose**: Used to adjust revenue and budget data for inflation to ensure consistency across movies released in different time periods.  
   - **Why**: Revenue and budget figures need to be standardized to account for changes in the value of money over time, making comparisons between older and newer movies more meaningful.

3. **Custom Web Scraping (IMDb and Wikipedia)**  
   - **Purpose**: Used to gather additional metrics like IMDb ratings, producers, opening weekend earnings, and missing revenue or budget data.  
   - **Why**: Some key information, such as audience ratings and financial details, was unavailable in the existing datasets. Scraping ensured that our analysis included these critical features.  


## Methods
1. **First exploration of the data**
   - Loading the data
   - Checking samples of the different tables
   - Apply basic preprocessing such as adding column names
   - First explorations

2. **Preprocessing**
   2.1 Data Cleaning:
      - Gather additional dataset from Kaggle (IMDb 5000) to complete missing values.
      - Clean the data to ensure consistency.
      - Handle missing values using imputation techniques.
      - Convert data types to appropriate formats for analysis (dictionaries nd lists)
   
   2.2 Data Scraping:
      - Use the `ImdbScraper` class to scrape additional data from IMDB.
      - Extract useful information such as revenue, budget, and ratings.
      - Store the scraped data in a structured format.

   2.3 Data Merging:
      - Merge the collected datasets with the IMDB data.
      - Ensure that different versions of movies are accounted for.
      - Resolve any conflicts or duplicates during the merging process.
      - Merge with the plot summaries

3. **Exploratory Data Analysis (EDA)**
   - First visualizations and quick analysis after preprocessing.
   - Perform EDA to understand the distribution and characteristics of the data.
   - Check for duplicates and missing values.

4. **Natural Language Processing**  
   4.1 Clustering:  
      - Apply clustering algorithms to group movies based on narrative structures and other features.
      - Use techniques like K-means, hierarchical clustering, or DBSCAN.
      - Analyze the clusters to identify common patterns and trends.  
   4.2 Summarization:  
      - Apply natural language processing (NLP) techniques to clean and preprocess plot summaries to make them shorter.
      - Using a summarization pipeline from LLM models from TheHuggingFace: `facebook/bart-large-cnn`  
   4.3 Plot Structure Classification:  
      - Use LLM (`facebook/bart-large-mnli`) and zero-shot classification to classify plot summaries into various plot structures.  
   4.4 Analysis of Plot Structure:  
      - Distribution of plot structure across the dataset and the years.
      - Relation between plot structure and box office revenues/profit. 

8. **Statistical Analysis**
   - Use statistical methods to compare the box office revenue of movies with different narrative structures.
   - Conduct hypothesis testing to identify significant differences.
   - Analyze trends over different time periods using time series analysis.

9. **Visualization**
   - Create visualizations to illustrate the findings and provide insights into the data.
   - Use bar plots, line plots, and stacked bar plots for genres and narrative archetypes.
   - Employ tools like Matplotlib, Seaborn, or Plotly for visualization.

10. **Temporal Analysis**
    - Analyze how the popularity of different plot structures and narrative formulas has evolved over different time periods and across genres.
    - Use time series analysis techniques to identify trends and patterns.
    - Visualize the temporal changes using line plots or heatmaps.


## Proposed Timeline and Organisation within the team
- **Week 1 (16.11.2024-22.11.2024)**:   
  - Enhance our preprocessing pipeline.   
    
- **Week 2 (23.11.2024-29.11.2024)**: 
  - Scrape additional data from IMDB using the ImdbScraper class. Correct the values to only have budget in dollars for the scraping.
  - Enhance our plot structure pipeline.

- **29.11.2024** : Homework 2 to be submitted.  
  
- **Week 3 (30.11.2024-06.12.2024)**   
  - Perform statistical analysis to compare the box office revenue of movies with different narrative structures and identify trends over different time periods, taking also into account inflation.
  
- **Week 4 (07.12.2024-13.12.2024)**        
  - Create visualizations to illustrate the findings and provide insights into the data, such as bar plots, line plots, and stacked bar plots for genres, characters, and narrative archetypes.  
   
- **Week 5 (14.12.2024-20.12.2024)**       
  - Analyze how the popularity of different plot structures and narrative formulas has evolved over different time periods and across genres.
  - Finalize the project structure, overall comments, interpretation and design.
  
- **20.12.2024** : Project to be submitted.




