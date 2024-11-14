import logging
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import pandas as pd
import os
import gc

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_summarizer(device):
    try:
        logger.info("Initializing summarizer...")
        model_name = "facebook/bart-large-cnn"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        if torch.cuda.is_available():
            model = model.to(device)
            model.eval()
            
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error initializing summarizer: {str(e)}")
        raise

def clean_text(text):
    """Clean and normalize text."""
    if pd.isna(text):
        return ""
    # Clean and normalize text
    text = str(text).strip()
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = ' '.join(text.split())  # Normalize whitespace
    return text

def chunk_text(text, max_chunk_length=1024):
    """Split text into chunks at sentence boundaries."""
    sentences = text.replace('\n', ' ').split('. ')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip() + '. '
        sentence_length = len(sentence)
        
        if current_length + sentence_length > max_chunk_length and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def process_single_text(text, model, tokenizer, device):
    try:
        if pd.isna(text) or not text.strip():
            return ""
            
        # Clean the text
        clean_input = clean_text(text)
        word_count = len(clean_input.split())
        
        # If text is short enough, return it as is
        if word_count <= 600:
            return clean_input
            
        # For longer texts, process in chunks
        chunks = chunk_text(clean_input)
        chunk_summaries = []
        
        # Process each chunk
        for chunk in chunks:
            inputs = tokenizer(chunk, 
                             max_length=1024,
                             truncation=True, 
                             return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                summary_ids = model.generate(
                    inputs["input_ids"],
                    max_length=150,
                    min_length=30,
                    num_beams=4,
                    length_penalty=1.5,
                    early_stopping=True,
                    no_repeat_ngram_size=2
                )
            
            chunk_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            chunk_summaries.append(chunk_summary)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Combine chunk summaries and create final summary
        combined_summary = ' '.join(chunk_summaries)
        inputs = tokenizer(combined_summary, 
                         max_length=1024,
                         truncation=True, 
                         return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            final_summary_ids = model.generate(
                inputs["input_ids"],
                max_length=200,
                min_length=50,
                num_beams=4,
                length_penalty=1.5,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        final_summary = tokenizer.decode(final_summary_ids[0], skip_special_tokens=True)
        return final_summary
        
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        return ""

def process_summaries_in_batches(df, batch_size=4, output_dir='../data/processed'):
    device = 0 if torch.cuda.is_available() else -1
    model, tokenizer = initialize_summarizer(device)
    
    os.makedirs(output_dir, exist_ok=True)
    processed_batches = []
    
    for i in tqdm(range(0, len(df), batch_size)):
        try:
            batch = df.iloc[i:i + batch_size].copy()
            summaries = []
            
            # Process each text individually
            for text in batch['plot_summary']:
                summary = process_single_text(text, model, tokenizer, device)
                summaries.append(summary)
                gc.collect()
            
            batch['summarized'] = summaries
            processed_batches.append(batch)
            
            # Save checkpoint every 5 batches
            if len(processed_batches) % 5 == 0:
                checkpoint_df = pd.concat(processed_batches)
                checkpoint_df.to_csv(f"{output_dir}/summaries_checkpoint.csv", index=False)
                
        except Exception as e:
            logger.error(f"Error processing batch {i}: {str(e)}")
            continue
    
    try:
        final_df = pd.concat(processed_batches)
        final_df.to_csv(f"{output_dir}/movies_with_summaries.csv", index=False)
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
        movies_df = pd.read_csv('../data/processed/movies_summary_BO.csv')
        logger.info(f"Loaded {len(movies_df)} movies")
        
        # Process summaries
        processed_df = process_summaries_in_batches(movies_df, batch_size=1)  # Process one at a time
        logger.info("Processing complete!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")