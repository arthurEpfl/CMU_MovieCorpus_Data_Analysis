import logging
import torch
from transformers import pipeline
from tqdm import tqdm
import pandas as pd
import os
import gc

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_classifier(device):
    try:
        logger.info("Initializing classifier...")
        model_name = "facebook/bart-large-mnli"
        classifier = pipeline("zero-shot-classification",
                            model=model_name,
                            device=device)
        return classifier
    except Exception as e:
        logger.error(f"Error initializing classifier: {str(e)}")
        raise

def get_candidate_labels():
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
    return candidate_labels

def process_single_text(text, classifier, candidate_labels):
    try:
        if pd.isna(text) or not text.strip():
            return ""
        
        response = classifier(text, candidate_labels)
        return response['labels'][0]
        
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        return ""

def process_classifications_in_batches(df, batch_size=4, output_dir='../data/processed'):
    device = 0 if torch.cuda.is_available() else -1
    classifier = initialize_classifier(device)
    candidate_labels = get_candidate_labels()
    
    os.makedirs(output_dir, exist_ok=True)
    processed_batches = []
    
    for i in tqdm(range(0, len(df), batch_size)):
        try:
            batch = df.iloc[i:i + batch_size].copy()
            classifications = []
            
            for text in batch['summarized']:
                classification = process_single_text(text, classifier, candidate_labels)
                classifications.append(classification)
                gc.collect()
            
            batch['plot_structure'] = classifications
            processed_batches.append(batch)
            
            if len(processed_batches) % 5 == 0:
                checkpoint_df = pd.concat(processed_batches)
                checkpoint_df.to_csv(f"{output_dir}/classifications_checkpoint.csv", index=False)
                
        except Exception as e:
            logger.error(f"Error processing batch {i}: {str(e)}")
            continue
    
    try:
        final_df = pd.concat(processed_batches)
        final_df.to_csv(f"{output_dir}/movies_with_classifications.csv", index=False)
        return final_df
    except Exception as e:
        logger.error(f"Error saving final results: {str(e)}")
        return pd.concat(processed_batches)

if __name__ == "__main__":
    try:
        # Set CUDA options
        torch.cuda.empty_cache()
        torch.backends.cuda.max_split_size_mb = 512
        
        # Load data
        summaries_df = pd.read_csv('../data/processed/summaries_checkpoint.csv')
        logger.info(f"Loaded {len(summaries_df)} summaries")
        
        # Process classifications
        processed_df = process_classifications_in_batches(summaries_df, batch_size=1)
        logger.info("Classification complete!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")