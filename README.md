# Cinematic Moral Dilemmas

# Team Adarable <3

## Abstract (à modifier plus tard)
The project investigates how morally ambiguous characters, such as anti-heroes or flawed protagonists, affect a movie’s box office success compared to traditional roles like heroes and villains. It explores how character complexity evolves over time and across genres. By applying persona clustering to classify characters into moral archetypes, the project analyzes trends over different periods and examines whether morally ambiguous characters tend to be more financially successful. Additionally, the project explores the idea that certain plot structures are frequently recycled across decades and genres. It investigates how often narrative formulas such as the hero's journey, love triangles, or rags-to-riches stories are reused and whether this repetition correlates with box office success. Using the provided plot summaries and metadata, the project applies natural language processing (NLP) to identify recurring narrative structures and tracks how the popularity of these plots has shifted over time and across genres. The goal is to understand how audiences respond to such characters and plot structures, and whether this appeal varies across demographic segments. This approach offers insights into societal preferences and how character morality and narrative structures impact a film’s commercial success.


## Research Questions
1. How do movies featuring anti-heroes compare to those featuring heroes in terms of box office revenue?
2. What are the most common genres associated with anti-hero and hero movies?
3. What are the key factors that contribute to the commercial success of movies with anti-heroes and heroes?

## Proposed Additional Datasets
- à voir


## Methods
1. **Data Collection and Cleaning**: Gather additional datasets and clean the data to ensure consistency and accuracy.
2. **Exploratory Data Analysis (EDA)**: Perform EDA to understand the distribution and characteristics of the data.
3. **Feature Engineering**: Create new features based on the research questions, such as genre counts, actor characteristics, and release date differences.
4. **Statistical Analysis**: Use statistical methods to compare the box office revenue of movies with anti-heroes and heroes.
5. **Visualization**: Create visualizations to illustrate the findings and provide insights into the data.
6. **Machine Learning**: Build predictive models to identify key factors contributing to the success of movies with different character archetypes.

## Proposed Timeline
- **Week 1**: Data collection and cleaning
- **Week 2**: Exploratory Data Analysis (EDA) and structure of the code

## Organization Within the Team
- **Data Collection and Cleaning**: [Team Member 1]
- **Exploratory Data Analysis (EDA)**: [Team Member 2]
- **Statistical Analysis**: [Team Member 3]
- **Visualization**: [Team Member 4]
- **Machine Learning (Optional)**: [Team Member 5]

### Internal Milestones
- **Milestone X (Week X)**: Complete data collection and cleaning


## Questions for TAs (Optional)
- XXX ?

## Template de base (à retirer plus tard)
This is a template repo for your project to help you organise and document your code better. 
Please use this structure for your project and document the installation, usage and structure as below.

## Quickstart

```bash
# clone project
git clone <project link>
cd <project repo>

# [OPTIONAL] create conda environment
conda create -n <env_name> python=3.11 or ...
conda activate <env_name>


# install requirements
pip install -r pip_requirements.txt
```



### How to use the library
Tell us how the code is arranged, any explanations goes here.



## Project Structure

The directory structure of new project looks like this:

```
├── data                        <- Project data files
│
├── src                         <- Source code
│   ├── data                            <- Data directory
│   ├── models                          <- Model directory
│   ├── utils                           <- Utility directory
│   ├── scripts                         <- Shell scripts
│
├── tests                       <- Tests of any kind
│
├── results.ipynb               <- a well-structured notebook showing the results
│
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing python dependencies
└── README.md
```