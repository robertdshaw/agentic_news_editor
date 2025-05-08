# Agentic AI News Editor: Comprehensive Preprocessing Pipeline
# -----------------------------------------------
# This pipeline processes the Microsoft MIND dataset (train, validation, and test splits)
# to prepare data for an agentic AI system that selects, ranks, and rewrites news headlines

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from datetime import datetime
from collections import Counter
import re
import json
from textstat import flesch_reading_ease
import warnings
warnings.filterwarnings('ignore')

# Set visualization defaults
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Create output directory for results
output_dir = 'agentic_news_editor'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    os.makedirs(f'{output_dir}/plots')
    os.makedirs(f'{output_dir}/processed_data')

print("# Agentic AI News Editor - Preprocessing Pipeline")
print("Starting preprocessing pipeline... Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("-" * 80)

# Define helper functions
def calculate_reading_scores(df):
    """Calculate reading ease scores for titles and abstracts"""
    if 'title_length' not in df.columns:
        df['title_length'] = df['title'].str.len()
    if 'abstract_length' not in df.columns:
        df['abstract_length'] = df['abstract'].str.len()
    
    # Calculate Flesch reading ease score
    df['title_reading_ease'] = df['title'].apply(lambda x: flesch_reading_ease(x) if isinstance(x, str) else 0)
    df['abstract_reading_ease'] = df['abstract'].apply(lambda x: flesch_reading_ease(x) if isinstance(x, str) else 0)
    
    return df

def process_impressions(behaviors_df):
    """Process the impressions data to extract clicks and impressions with chunking and progress tracking"""
    # Create an empty list to hold the impression records
    all_impression_records = []
    total_rows = len(behaviors_df)
    
    # Process in chunks to manage memory usage
    chunk_size = 5000  # Adjust based on your system's memory
    num_chunks = (total_rows + chunk_size - 1) // chunk_size  # Ceiling division
    
    print(f"Processing {total_rows} behavior records in {num_chunks} chunks...")
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_rows)
        
        print(f"Processing chunk {chunk_idx+1}/{num_chunks} (records {start_idx}-{end_idx})...")
        
        # Get the current chunk
        behaviors_chunk = behaviors_df.iloc[start_idx:end_idx]
        chunk_records = []
        
        # Process each row in the chunk
        for i, (_, row) in enumerate(behaviors_chunk.iterrows()):
            if i % 1000 == 0 and i > 0:
                print(f"  Progress: {i}/{len(behaviors_chunk)} records in current chunk")
                
            impression_id = row['impression_id']
            user_id = row['user_id']
            time = row['time']
            impressions = row['impressions']
            
            # Split the impressions string and process each impression
            if isinstance(impressions, str):
                for impression in impressions.split():
                    parts = impression.split('-')
                    if len(parts) == 2:
                        news_id, clicked = parts
                        clicked = 1 if clicked == '1' else 0
                        
                        chunk_records.append({
                            'impression_id': impression_id,
                            'user_id': user_id,
                            'time': time,
                            'news_id': news_id,
                            'clicked': clicked
                        })
        
        # Add chunk records to the main list
        all_impression_records.extend(chunk_records)
        print(f"  Processed {len(chunk_records)} impressions in this chunk")
        print(f"  Total impressions so far: {len(all_impression_records)}")
    
    print(f"Finished processing all {len(all_impression_records)} impressions")
    
    # Create a dataframe from all impression records
    return pd.DataFrame(all_impression_records)

def process_dataset(data_type='train'):
    """Process a single dataset split (train, val, or test)"""
    print(f"\n## Processing {data_type.upper()} Dataset")
    
    # Define file paths
    news_path = f"{data_type}_data/news.tsv"
    behaviors_path = f"{data_type}_data/behaviors.tsv"
    
    # Define output file path - create early to check if already processed
    output_file = f'{output_dir}/processed_data/{data_type}_headline_ctr.csv'
    
    # Check if output file already exists to allow resuming interrupted processing
    if os.path.exists(output_file):
        print(f"{data_type} data already processed. Loading from {output_file}")
        return pd.read_csv(output_file)
    
    # Check if files exist
    if not os.path.exists(news_path) or not os.path.exists(behaviors_path):
        print(f"Warning: {data_type} data files not found. Skipping this split.")
        return None
    
    # 1. Load data
    print(f"Loading {data_type} data...")
    news_cols = ["newsID", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
    news_df = pd.read_csv(news_path, sep="\t", header=None, names=news_cols)
    print(f"{data_type} news data loaded: {news_df.shape[0]} rows, {news_df.shape[1]} columns")
    
    behaviors_cols = ["impression_id", "user_id", "time", "history", "impressions"]
    behaviors_df = pd.read_csv(behaviors_path, sep="\t", header=None, names=behaviors_cols)
    print(f"{data_type} behaviors data loaded: {behaviors_df.shape[0]} rows, {behaviors_df.shape[1]} columns")
    
    # 2. Clean and process news data
    print(f"Cleaning {data_type} news data...")
    news_df_cleaned = news_df.copy()
    
    # Add length columns
    news_df_cleaned['title_length'] = news_df_cleaned['title'].str.len()
    news_df_cleaned['abstract_length'] = news_df_cleaned['abstract'].str.len()
    
    # Filter out articles with very short titles or abstracts
    news_df_cleaned = news_df_cleaned[news_df_cleaned['title_length'] >= 10]
    news_df_cleaned = news_df_cleaned[news_df_cleaned['abstract_length'] >= 20]
    print(f"After filtering short content: {len(news_df_cleaned)} articles remaining")
    
    # Add reading ease score
    news_df_cleaned = calculate_reading_scores(news_df_cleaned)
    print(f"Added reading ease scores")
    
    # Save intermediate data to allow resuming if process fails
    intermediate_file = f'{output_dir}/processed_data/{data_type}_news_cleaned.csv'
    news_df_cleaned.to_csv(intermediate_file, index=False)
    print(f"Saved intermediate cleaned news data to {intermediate_file}")
    
    # 3. Process impressions data
    print(f"Processing {data_type} impressions data...")
    impressions_df = process_impressions(behaviors_df)
    
    # Save intermediate impressions data
    intermediate_impressions = f'{output_dir}/processed_data/{data_type}_impressions.csv'
    impressions_df.to_csv(intermediate_impressions, index=False)
    print(f"Saved intermediate impressions data to {intermediate_impressions}")
    
    # For test data, if clicks aren't available, we'll create a placeholder
    if data_type == 'test' and 'clicked' not in impressions_df.columns:
        print("Test data doesn't have click information; using placeholder values.")
        impressions_df['clicked'] = 0
    
    # 4. Add CTR data to news articles
    print(f"Calculating CTR for {data_type} data...")
    article_ctr_data = impressions_df.groupby('news_id').agg({
        'clicked': ['sum', 'count']
    })
    article_ctr_data.columns = ['total_clicks', 'total_impressions']
    article_ctr_data['ctr'] = article_ctr_data['total_clicks'] / article_ctr_data['total_impressions']
    
    print(f"Calculated CTR for {len(article_ctr_data)} articles")
    print(f"  Average CTR: {article_ctr_data['ctr'].mean():.4f}")
    print(f"  Max impressions: {article_ctr_data['total_impressions'].max()}")
    
    # 5. Merge CTR data with news data
    print(f"Merging news and engagement data...")
    news_with_engagement = news_df_cleaned.merge(
        article_ctr_data.reset_index(),
        left_on='newsID',
        right_on='news_id',
        how='left'
    )
    
    # Fill missing engagement data with zeros
    engagement_columns = ['total_clicks', 'total_impressions', 'ctr']
    news_with_engagement[engagement_columns] = news_with_engagement[engagement_columns].fillna(0)
    
    # 6. Apply readability binning
    try:
        news_with_engagement['reading_ease_bin'] = pd.qcut(
            news_with_engagement['title_reading_ease'], 
            q=5, 
            labels=['Very Hard', 'Hard', 'Medium', 'Easy', 'Very Easy'],
            duplicates='drop'  # Handle potential duplicate bin edges
        )
    except Exception as e:
        print(f"Warning: Could not create reading ease bins due to: {e}")
        # Create a simple binning as fallback
        bins = [-float('inf'), 30, 50, 70, 90, float('inf')]
        labels = ['Very Hard', 'Hard', 'Medium', 'Easy', 'Very Easy']
        news_with_engagement['reading_ease_bin'] = pd.cut(
            news_with_engagement['title_reading_ease'],
            bins=bins,
            labels=labels
        )
    
    # 7. Save processed data
    news_with_engagement.to_csv(output_file, index=False)
    print(f"Processed {data_type} data saved to {output_file}")
    
    # Remove intermediate files to save space (comment out to keep them)
    # os.remove(intermediate_file)
    # os.remove(intermediate_impressions)
    
    return news_with_engagement

# Process all three dataset splits with timing info
import time

# Process training data
print("\n" + "="*50)
print("PROCESSING TRAINING DATASET")
print("="*50)
train_start = time.time()
train_data = process_dataset('train')
train_end = time.time()
print(f"Training data processing took {(train_end - train_start)/60:.2f} minutes")

# Process validation data
print("\n" + "="*50)
print("PROCESSING VALIDATION DATASET")
print("="*50)
val_start = time.time()
val_data = process_dataset('val')
val_end = time.time()
print(f"Validation data processing took {(val_end - val_start)/60:.2f} minutes")

# Process test data
print("\n" + "="*50)
print("PROCESSING TEST DATASET")
print("="*50)
test_start = time.time()
test_data = process_dataset('test')
test_end = time.time()
print(f"Test data processing took {(test_end - test_start)/60:.2f} minutes")

# Overall timing
total_time = (test_end - train_start)/60
print(f"\nTotal processing time: {total_time:.2f} minutes")

# Print summary statistics for each split
def print_split_summary(data, split_name):
    if data is not None:
        print(f"\n## {split_name} Split Summary:")
        print(f"Total headlines: {len(data)}")
        print(f"Headlines with impressions: {len(data[data['total_impressions'] > 0])}")
        print(f"Average CTR: {data['ctr'].mean():.4f}")
        print(f"Average title length: {data['title_length'].mean():.2f} characters")
        print(f"Average reading ease: {data['title_reading_ease'].mean():.2f}")
        print(f"Category distribution: {data['category'].value_counts().head(3).to_dict()}")

print_split_summary(train_data, "Training")
print_split_summary(val_data, "Validation")
print_split_summary(test_data, "Test")

# Create a json summary of the dataset statistics for reference
dataset_stats = {
    "train": {
        "total_headlines": len(train_data) if train_data is not None else 0,
        "with_impressions": len(train_data[train_data['total_impressions'] > 0]) if train_data is not None else 0,
        "avg_ctr": float(train_data['ctr'].mean()) if train_data is not None else 0,
        "avg_reading_ease": float(train_data['title_reading_ease'].mean()) if train_data is not None else 0
    },
    "val": {
        "total_headlines": len(val_data) if val_data is not None else 0,
        "with_impressions": len(val_data[val_data['total_impressions'] > 0]) if val_data is not None else 0,
        "avg_ctr": float(val_data['ctr'].mean()) if val_data is not None else 0,
        "avg_reading_ease": float(val_data['title_reading_ease'].mean()) if val_data is not None else 0
    },
    "test": {
        "total_headlines": len(test_data) if test_data is not None else 0,
        "with_impressions": len(test_data[test_data['total_impressions'] > 0]) if test_data is not None else 0,
        "avg_ctr": float(test_data['ctr'].mean()) if test_data is not None else 0, 
        "avg_reading_ease": float(test_data['title_reading_ease'].mean()) if test_data is not None else 0
    }
}

with open(f'{output_dir}/processed_data/dataset_statistics.json', 'w') as f:
    json.dump(dataset_stats, f, indent=2)

print("-" * 80)
print(f"Preprocessing Pipeline completed! Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Processed data saved to '{output_dir}/processed_data/'")
print("Files created:")
print(f"  - train_headline_ctr.csv")
print(f"  - val_headline_ctr.csv")
print(f"  - test_headline_ctr.csv")
print(f"  - dataset_statistics.json")
print(f"Ready for Agentic AI News Editor development!")