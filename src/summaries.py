# %%
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from joblib import Parallel, delayed
import torch
import math

# %% [markdown]
# # Introduction: Plot Structure Analysis for Movie Summaries
# 
# The goal of this notebook is to **process movie plot summaries to identify their underlying plot structures**. By categorizing each summary according to distinct narrative patterns, we aim to gain insights into common plot structures and explore potential correlations with financial success.
# 
# To achieve this, we experimented with **two different approaches**:
# 
# 1. **Clustering**: We used unsupervised clustering (KMeans) on plot summaries to explore any emergent plot structure patterns.
# 
# 2. **Large Language Model (LLM) Classification**: Using a predefined set of 15 plot structure categories, we use a LLM to classify each summary. This classification approach uses zero-shot prompting to assign each summary to a category.

# %% [markdown]
# # Importing the data

# %%
# Use GPU if available
device = 0 if torch.cuda.is_available() else -1  # -1 for CPU, 0 for GPU
movies_summary = pd.read_csv('../data/processed/movies_summary_BO.csv')

# %%
# We create the pipeline for summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

# %%
def summarize_text(text):
    """
    Summarizes a plot summary if it exceeds a specified word count limit.

    This function leverages `facebook/bart-large-cnn` for summarization, handling lengthy plot summaries by dividing
    them into chunks when they exceed a set word count threshold. Each chunk is summarized separately, and the resulting
    summaries are concatenated to create a final, shorter version of the plot summary.

    Args:
        text (str): The input plot summary text to be summarized.

    Returns:
        str: The summarized text if the original text length exceeds the limit; otherwise, returns the original text.

    Logic:
        1. **Limit Check**: If the text's word count is below `limit` (600 words), the function returns the original text.
        2. **Chunking and Summarization**:
            - If the text exceeds `chunk_size` (700 words), it is divided into chunks of `chunk_size` words.
            - A 20-word overlap is added to the start of each subsequent chunk (except the first) to preserve context.
            - Each chunk is summarized individually.
            - The summaries of all chunks are concatenated to produce the final summary.
        3. **Single-Summary for Moderate-Length Texts**: If the text is above `limit` but under `chunk_size`, it is summarized
           directly without chunking.
    """

    limit=600
    chunk_size=700
    max_len=500
    min_len=300
    word_count = len(text.split())
    print("len text", len(text))
    print("word_count ", word_count)
    #print(text.split())
    if word_count < limit:
        return text
    if word_count > chunk_size :
      summary = ''
      for i in range(0, word_count, chunk_size):
        start = 20 if i == 0 else i
        end = min(i + chunk_size, word_count)
        print(i, start - 20, end)
        chunk = " ".join(text.split()[start-20:end])
        print("chunk :", chunk)
        chunk_len = len(chunk.split())
        if chunk_len < 250:
          summary += chunk
        else:
          summary += summarizer(chunk, max_length=450, min_length=min_len, do_sample=False)[0]['summary_text']
        print(summary)
      return summary

    summary = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']
    return summary

# %%
movies_summary['summarized'] = movies_summary['plot_summary'].apply(summarize_text)

# %% [markdown]
# ## 2.2 Zero-shot Classification
# 
# Here we create our pipeline for classification, and classify our summarized plot summaries into plot structure categories. The plot structure categories were found using GPT4o.

# %%
# We create the classification pipeline
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli", device=device)


# %%
# Function to classify each plot summary
def classify_summary(summary):
    """
    Classifies a plot summary into a predefined set of plot structure categories using a zero-shot classification model.

    This function leverages a zero-shot classifier to categorize each plot summary into one of 15 distinct narrative
    structures.

    Args:
        summary (str): The plot summary to classify.

    Returns:
        str: The most likely plot structure category based on the model's classification.

    Notes:
        - The zero-shot classification model is pre-trained to handle tasks without additional training, leveraging its
          existing knowledge to map text to candidate categories.

    """
    candidate_labels = [
    "Hero’s Journey and Transformation: The protagonist undergoes personal growth, often starting as an ordinary individual who faces challenges, gains allies, overcomes obstacles, and returns transformed.",
    "Quest for Vengeance or Justice: A revenge-driven plot where the protagonist seeks retribution or justice for a past wrong or injustice.",
    "Coming of Age and Self-Discovery: The protagonist matures or gains self-awareness, often overcoming personal or societal obstacles.",
    "Survival or Escape: The story revolves around characters trying to survive dangerous situations or escape captivity.",
    "Rise and Fall of a Protagonist: The protagonist experiences a rise to power or success, followed by a tragic or inevitable downfall.",
    "Love and Relationship Dynamics: Focuses on romantic or family relationships, often dealing with misunderstandings, unions, reconciliations, or unfulfilled love.",
    "Comedy of Errors or Misadventure: Characters experience humorous, unintended consequences or misadventures while pursuing a goal.",
    "Crime and Underworld Exploration: The story explores criminal activities or the underworld, often involving heists, gang conflicts, or undercover missions.",
    "Power Struggle and Betrayal: Focuses on conflicts for power or leadership, with betrayal as a central theme, often involving shifting alliances.",
    "Mystery and Conspiracy Unveiling: The protagonist uncovers a hidden conspiracy, solves puzzles, or discovers hidden truths.",
    "Tragedy and Inevitability: A character-driven plot where the protagonist faces an inevitable negative outcome, often due to a flaw or external betrayal.",
    "Conflict with Supernatural or Unknown Forces: The protagonist encounters supernatural entities, unknown forces, or sci-fi elements that pose existential challenges.",
    "Comedy in Domestic Life: Focuses on the humor and challenges of family life, with everyday misunderstandings and domestic issues driving the plot.",
    "Social Rebellion or Fight Against Oppression: The protagonist challenges societal norms or oppressive systems, leading to personal or collective change.",
    "Fantasy or Science Fiction Quest: Centers on a journey or quest in a fantastical or sci-fi setting, involving world-building, encounters with non-human entities, and mythical or technological challenges."
  ]

    input_prompt = summary
    response = classifier(input_prompt, candidate_labels)
    classification = response['labels'][0]
    return classification


# %% [markdown]
# We then classify each plot summary's structure in batches, allowing us to handle large datasets more efficiently and prevent potential memory issues. We set a `batch_size` of 500 and use parallel processing to classify summaries within each batch concurrently. After processing each batch, we save the results to a CSV file as a checkpoint.

# %%
movies_summary['summarized'] = movies_summary['summarized'].apply(classify_summary)

batch_size = 500
output_dir = '../data/classified_summaries_batches'
os.makedirs(output_dir, exist_ok=True)

for i in range(0, len(movies_summary), batch_size):
    batch = movies_summary.iloc[i:i + batch_size].copy()  # Work with a copy to avoid modifying the original DataFrame

    # Parallel processing of classification within the batch
    batch['plot_structure'] = Parallel(n_jobs=4)(delayed(classify_summary)(summary) for summary in batch['summarized'])

    # Save the processed batch to a CSV file as a checkpoint
    batch.to_csv(f"{output_dir}/classified_summaries_batch_{i}.csv", index=False)

# %%
# We combine all batch files into a single file
all_batches = [pd.read_csv(f"{output_dir}/classified_summaries_batch_{i}.csv") for i in range(0, len(movies_summary), batch_size)]
classified_summaries = pd.concat(all_batches, ignore_index=True)
dir = '../data/processed'
os.makedirs(dir, exist_ok=True)
classified_summaries.to_csv('../data/processed/classified_summaries_with_plot_structures.csv', index=False)

# %% [markdown]
# ## 2.3 Try classification with different categories
# 

# %%
# Function to classify each plot summary
def classify_summary_different_labels(summary):
    """
    Classifies a plot summary into a predefined set of plot structure categories using a zero-shot classification model.

    This function leverages a zero-shot classifier to categorize each plot summary into one of 15 distinct narrative
    structures.

    Args:
        summary (str): The plot summary to classify.

    Returns:
        str: The most likely plot structure category based on the model's classification.

    Notes:
        - The zero-shot classification model is pre-trained to handle tasks without additional training, leveraging its
          existing knowledge to map text to candidate categories.

    """
    candidate_labels = ['Relentless Pursuit: The protagonist is continuously pursued by a formidable opponent or authority, ending in a final showdown or escape.',
 'Memory Recovery: The protagonist suffers from memory loss and gradually uncovers their past identity and purpose, often leading to a significant revelation or reunion.',
 'Inheritance of Duty: A character inherits an extraordinary responsibility, skill, or artifact, which they must protect or learn to wield, typically involving rigorous challenges.',
 'Ordinary to Extraordinary Journey: An unsuspecting character is drawn into an epic journey, evolving through obstacles to fulfill a unique role they initially resisted or doubted.',
 'Countdown Crisis: Characters face a strict deadline, solving complex problems or overcoming obstacles before time runs out to avert disaster.',
 'High-Stakes Rescue: The plot centers on rescuing a person or group from peril, often requiring the protagonist to confront significant physical or moral dilemmas.',
 'Breakout Plot: The protagonist begins in captivity and must plan and execute an escape, often by navigating complex social or environmental challenges.',
 'Amnesiac Reunion: Following separation and loss of memory, characters are reunited after a series of incidental encounters gradually restore memories.',
 'Redemption Arc: A flawed or once-villainous character seeks redemption, confronting their past through sacrifice or atonement.',
 'Hidden Power Awakening: A character discovers hidden abilities and must undergo training, typically to prepare for a unique challenge or mission.',
 'Unlikely Alliance: Disparate characters join forces to achieve a shared goal, learning to overcome differences and trust each other to succeed.',
 'Revolutionary Uprising: The protagonist and allies challenge an oppressive authority, with the story leading to a climactic confrontation or overthrow.',
 'Guardian Mission: The protagonist is tasked with safeguarding a vulnerable individual, group, or object, navigating multiple threats to fulfill their protective role.',
 'Duel for Justice: The storyline builds toward a climactic one-on-one duel, often against a personal nemesis, to settle a longstanding score or defend a cause.',
 'Treasure Hunt: Characters compete to uncover a coveted object or location, leading to rivalries, alliances, and betrayals along the way.',
 'Mistaken Identity Spiral: A character is misidentified or assumed to be someone else, resulting in escalating misunderstandings they must unravel.',
 'Rise to Downfall: The protagonist rises to prominence but faces a dramatic fall, often due to personal flaws or betrayal, leading to a period of reckoning.',
 'Parallel Journeys: Two main characters embark on distinct but intersecting journeys, with their paths influencing each other toward a shared conclusion.',
 'Mission to Save a Community: The protagonist works to save or uplift their community from a specific threat, often gaining respect or unity along the way.',
 'Discovery Expedition: A journey or exploration driven by curiosity or necessity, uncovering significant discoveries or secrets that impact the characters or world.',
 'Comedy of Errors or Misadventure: Characters experience humorous, unintended consequences or misadventures while pursuing a goal.',
  'Fantasy or Science Fiction Quest: Journey in a fantastical or sci-fi setting.',
  'Survival or Escape: The story revolves around characters trying to survive dangerous situations or escape captivity.']

    input_prompt = summary
    response = classifier(input_prompt, candidate_labels)
    classification = response['labels'][0]
    return classification


# %%
movies_summary['summarized'] = movies_summary['summarized'].apply(classify_summary_different_labels)

batch_size = 500
output_dir = '../data/classified_summaries_batches_second_labels'
os.makedirs(output_dir, exist_ok=True)

for i in range(0, len(movies_summary), batch_size):
    batch = movies_summary.iloc[i:i + batch_size].copy()  # Work with a copy to avoid modifying the original DataFrame

    # Parallel processing of classification within the batch
    batch['plot_structure'] = Parallel(n_jobs=4)(delayed(classify_summary)(summary) for summary in batch['summarized'])

    # Save the processed batch to a CSV file as a checkpoint
    batch.to_csv(f"{output_dir}/classified_summaries_batch_{i}.csv", index=False)

# %%
# We combine all batch files into a single file
all_batches = [pd.read_csv(f"{output_dir}/classified_summaries_batch_{i}.csv") for i in range(0, len(movies_summary), batch_size)]
classified_summaries = pd.concat(all_batches, ignore_index=True)
classified_summaries.to_csv('../data/processed/classified_summaries_with_plot_structures_second_labels.csv', index=False)


