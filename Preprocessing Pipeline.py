# Memory-Efficient MIND Dataset Preprocessing Pipeline
# -----------------------------------------------------
# Processes train, validation, and test splits with sampling to reduce memory usage
# Includes checks to verify data quality and correctness

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import time
from datetime import datetime
from collections import Counter
import re
import json
from textstat import flesch_reading_ease
import warnings
import random
warnings.filterwarnings('ignore')

# Set visualization defaults
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# ================== CONFIGURATION ==================
# Directory settings
output_dir = 'agentic_news_editor'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    os.makedirs(f'{output_dir}/plots')
    os.makedirs(f'{output_dir}/processed_data')

# Sampling configuration - adjust these to control memory usage
BEHAVIOR_SAMPLE_RATE = 0.2  # Process 20% of behavior records
NEWS_SAMPLE_RATE = 1.0  # Process all news articles
MAX_IMPRESSIONS_PER_BEHAVIOR = 20  # Limit impressions per behavior record
MIN_IMPRESSIONS_FOR_CTR = 5  # Min impressions needed for reliable CTR

# ================== UTILITY FUNCTIONS ==================
def memory_usage():
    """Get current memory usage in MB"""
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)  # Convert to MB

def print_with_timestamp(message):
    """Print message with timestamp and memory usage info"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mem = memory_usage()
    print(f"[{timestamp}] {message} (Memory: {mem:.1f} MB)")

def validate_dataframe(df, name, required_columns=None):
    """Validate a dataframe's structure and contents"""
    print_with_timestamp(f"Validating {name} dataframe...")
    if df is None:
        print(f"  ERROR: {name} dataframe is None!")
        return False
    
    print(f"  Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Check for required columns
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            print(f"  WARNING: Missing required columns: {', '.join(missing)}")
            return False
        print(f"  All required columns present")
    
    # Check for NaN values in key columns
    if required_columns:
        for col in required_columns:
            if col in df.columns and df[col].isna().any():
                print(f"  WARNING: NaN values found in column '{col}'")
    
    # Check min/max for numeric columns
    for col in df.select_dtypes(include=['number']).columns:
        print(f"  '{col}': min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.4f}")
    
    return True

def calculate_reading_scores(df):
    """Calculate reading ease scores for titles and abstracts"""
    if 'title_length' not in df.columns:
        df['title_length'] = df['title'].str.len()
    if 'abstract_length' not in df.columns:
        df['abstract_length'] = df['abstract'].str.len()
    
    # Calculate Flesch reading ease score
    # Use a try/except to handle potential errors
    df['title_reading_ease'] = df['title'].apply(
        lambda x: flesch_reading_ease(x) if isinstance(x, str) else 0
    )
    df['abstract_reading_ease'] = df['abstract'].apply(
        lambda x: flesch_reading_ease(x) if isinstance(x, str) else 0
    )
    
    return df

def process_impressions_to_file(behaviors_df, output_path, sample_rate=1.0, max_per_behavior=None):
    """
    Process impressions from behaviors and write directly to a CSV file
    to avoid memory issues. Allows sampling to reduce size.
    """
    import csv
    
    # Create CSV file and write header
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['impression_id', 'user_id', 'time', 'news_id', 'clicked'])
    
    # Sample the behaviors if requested
    if sample_rate < 1.0:
        behaviors_df = behaviors_df.sample(frac=sample_rate, random_state=42)
        print_with_timestamp(f"Sampling behaviors at rate {sample_rate}, using {len(behaviors_df)} records")
    
    total_impressions = 0
    total_sampled_impressions = 0
    total_rows = len(behaviors_df)
    
    # Process in chunks to manage memory usage
    chunk_size = 5000
    num_chunks = (total_rows + chunk_size - 1) // chunk_size
    
    print_with_timestamp(f"Processing {total_rows} behavior records in {num_chunks} chunks...")
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_rows)
        
        print_with_timestamp(f"Processing chunk {chunk_idx+1}/{num_chunks} (records {start_idx}-{end_idx})...")
        
        # Get the current chunk
        behaviors_chunk = behaviors_df.iloc[start_idx:end_idx]
        chunk_impressions = 0
        chunk_sampled = 0
        
        # Open file in append mode
        with open(output_path, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Process each row in the chunk
            for i, (_, row) in enumerate(behaviors_chunk.iterrows()):
                if i % 1000 == 0 and i > 0:
                    print(f"  Progress: {i}/{len(behaviors_chunk)} records in current chunk")
                    
                impression_id = row['impression_id']
                user_id = row['user_id']
                time = row['time']
                impressions = row['impressions']
                
                # Process impressions for this behavior
                if isinstance(impressions, str):
                    all_impressions = impressions.split()
                    # Apply max per behavior if specified
                    if max_per_behavior and len(all_impressions) > max_per_behavior:
                        # Keep a random subset to maintain distribution
                        all_impressions = random.sample(all_impressions, max_per_behavior)
                    
                    for impression in all_impressions:
                        parts = impression.split('-')
                        if len(parts) == 2:
                            news_id, clicked = parts
                            clicked = 1 if clicked == '1' else 0
                            
                            # Write directly to CSV
                            writer.writerow([impression_id, user_id, time, news_id, clicked])
                            chunk_impressions += 1
                            chunk_sampled += 1
        
        total_impressions += chunk_impressions
        total_sampled_impressions += chunk_sampled
        print_with_timestamp(f"  Processed {chunk_impressions} impressions in this chunk")
        print_with_timestamp(f"  Total impressions so far: {total_impressions}")
    
    print_with_timestamp(f"Finished processing all {total_impressions} impressions")
    print_with_timestamp(f"Impressions data saved to: {output_path}")
    
    return output_path, total_impressions

def calculate_ctr_from_impressions_file(impressions_path):
    """
    Calculate CTR metrics from impressions file in a memory-efficient way
    by processing it in chunks
    """
    print_with_timestamp(f"Calculating CTR from impressions file: {impressions_path}")
    
    # Check if file exists
    if not os.path.exists(impressions_path):
        print_with_timestamp(f"ERROR: Impressions file not found: {impressions_path}")
        return None
    
    # Create temporary files for intermediate aggregations to save memory
    temp_clicks_file = impressions_path + ".clicks.tmp"
    temp_impressions_file = impressions_path + ".impressions.tmp"
    
    # To prevent memory issues, we'll use a two-step approach:
    # 1. First, count rows per news_id and write to a file
    # 2. Then aggregate from that file
    
    # Count clicks and impressions for each news_id
    news_id_counter = {}  # {news_id: [clicks, impressions]}
    
    chunk_size = 1000000  # 1 million rows at a time
    total_rows_processed = 0
    
    # Process file in chunks
    for chunk_num, chunk in enumerate(pd.read_csv(impressions_path, chunksize=chunk_size)):
        print_with_timestamp(f"Processing chunk {chunk_num+1} for CTR calculation...")
        
        # Process each row in the chunk
        for _, row in chunk.iterrows():
            news_id = row['news_id']
            clicked = row['clicked']
            
            if news_id not in news_id_counter:
                news_id_counter[news_id] = [0, 0]
            
            news_id_counter[news_id][0] += clicked  # Add to clicks
            news_id_counter[news_id][1] += 1  # Add to impressions
        
        total_rows_processed += len(chunk)
        print_with_timestamp(f"Processed {total_rows_processed} rows so far...")
        
        # To prevent memory issues with extremely large datasets, periodically write to disk
        if len(news_id_counter) > 1000000:  # If tracking over 1M news IDs
            # Create dataframe and write to disk
            ctr_df = pd.DataFrame([
                {'news_id': nid, 'total_clicks': stats[0], 'total_impressions': stats[1]}
                for nid, stats in news_id_counter.items()
            ])
            
            # Append to temp file or create it
            if os.path.exists(temp_clicks_file):
                ctr_df.to_csv(temp_clicks_file, mode='a', header=False, index=False)
            else:
                ctr_df.to_csv(temp_clicks_file, index=False)
            
            # Clear the counter to free memory
            news_id_counter = {}
    
    # Write any remaining data to disk
    if news_id_counter:
        ctr_df = pd.DataFrame([
            {'news_id': nid, 'total_clicks': stats[0], 'total_impressions': stats[1]}
            for nid, stats in news_id_counter.items()
        ])
        
        # Append to temp file or create it
        if os.path.exists(temp_clicks_file):
            ctr_df.to_csv(temp_clicks_file, mode='a', header=False, index=False)
        else:
            ctr_df.to_csv(temp_clicks_file, index=False)
    
    # If we used temp files, aggregate the results
    if os.path.exists(temp_clicks_file):
        print_with_timestamp("Aggregating final CTR data from temporary files...")
        
        # Again process in chunks to avoid memory issues
        final_results = {}  # {news_id: [total_clicks, total_impressions]}
        
        for chunk in pd.read_csv(temp_clicks_file, chunksize=100000):
            for _, row in chunk.iterrows():
                news_id = row['news_id']
                if news_id not in final_results:
                    final_results[news_id] = [0, 0]
                
                final_results[news_id][0] += row['total_clicks']
                final_results[news_id][1] += row['total_impressions']
        
        # Create final dataframe
        final_df = pd.DataFrame([
            {'news_id': nid, 'total_clicks': stats[0], 'total_impressions': stats[1]}
            for nid, stats in final_results.items()
        ])
        
        # Calculate CTR
        final_df['ctr'] = final_df['total_clicks'] / final_df['total_impressions']
        
        # Clean up temp files
        if os.path.exists(temp_clicks_file):
            os.remove(temp_clicks_file)
        if os.path.exists(temp_impressions_file):
            os.remove(temp_impressions_file)
        
        print_with_timestamp(f"Calculated CTR for {len(final_df)} news items")
        return final_df
    else:
        # If we didn't need temp files, just convert the dictionary to a dataframe
        ctr_df = pd.DataFrame([
            {'news_id': nid, 'total_clicks': stats[0], 'total_impressions': stats[1]}
            for nid, stats in news_id_counter.items()
        ])
        
        # Calculate CTR
        ctr_df['ctr'] = ctr_df['total_clicks'] / ctr_df['total_impressions']
        
        print_with_timestamp(f"Calculated CTR for {len(ctr_df)} news items")
        return ctr_df

def plot_ctr_distribution(df, title, output_path):
    """Create and save a plot of CTR distribution"""
    try:
        # Filter to items with sufficient impressions for reliable CTR
        df_filtered = df[df['total_impressions'] >= MIN_IMPRESSIONS_FOR_CTR]
        
        plt.figure(figsize=(10, 6))
        sns.histplot(df_filtered['ctr'], bins=30, kde=True)
        plt.title(f"CTR Distribution - {title}")
        plt.xlabel("Click-Through Rate")
        plt.ylabel("Count")
        plt.savefig(output_path)
        plt.close()
        
        print_with_timestamp(f"CTR distribution plot saved to: {output_path}")
        
        # Return some statistics
        stats = {
            'mean': float(df_filtered['ctr'].mean()),
            'median': float(df_filtered['ctr'].median()),
            'std': float(df_filtered['ctr'].std()),
            'min': float(df_filtered['ctr'].min()),
            'max': float(df_filtered['ctr'].max()),
            '25th': float(df_filtered['ctr'].quantile(0.25)),
            '75th': float(df_filtered['ctr'].quantile(0.75)),
            'items_with_sufficient_impressions': int(len(df_filtered)),
            'total_items': int(len(df))
        }
        
        return stats
    except Exception as e:
        print_with_timestamp(f"Error creating CTR distribution plot: {e}")
        return None

# ================== MAIN PROCESSING FUNCTION ==================
def process_dataset(data_type='train', behavior_sample_rate=BEHAVIOR_SAMPLE_RATE, 
                   max_impressions=MAX_IMPRESSIONS_PER_BEHAVIOR):
    """Process a single dataset split (train, val, or test) with sampling options"""
    print_with_timestamp(f"\n{'='*50}")
    print_with_timestamp(f"PROCESSING {data_type.upper()} DATASET")
    print_with_timestamp(f"{'='*50}")
    
    # Define file paths
    news_path = f"{data_type}_data/news.tsv"
    behaviors_path = f"{data_type}_data/behaviors.tsv"
    
    # Define output file path - create early to check if already processed
    output_file = f'{output_dir}/processed_data/{data_type}_headline_ctr.csv'
    
    # Check if files exist
    if not os.path.exists(news_path) or not os.path.exists(behaviors_path):
        print_with_timestamp(f"WARNING: {data_type} data files not found. Skipping this split.")
        return None
    
    # 1. Load news data
    print_with_timestamp(f"Loading {data_type} news data...")
    news_cols = ["newsID", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
    news_df = pd.read_csv(news_path, sep="\t", header=None, names=news_cols)
    print_with_timestamp(f"{data_type} news data loaded: {news_df.shape[0]} rows, {news_df.shape[1]} columns")
    
    # Sample news data if needed
    if NEWS_SAMPLE_RATE < 1.0:
        original_size = len(news_df)
        news_df = news_df.sample(frac=NEWS_SAMPLE_RATE, random_state=42)
        print_with_timestamp(f"Sampled news data from {original_size} to {len(news_df)} articles")
    
    # 2. Load behaviors data
    print_with_timestamp(f"Loading {data_type} behaviors data...")
    behaviors_cols = ["impression_id", "user_id", "time", "history", "impressions"]
    behaviors_df = pd.read_csv(behaviors_path, sep="\t", header=None, names=behaviors_cols)
    print_with_timestamp(f"{data_type} behaviors data loaded: {behaviors_df.shape[0]} rows, {behaviors_df.shape[1]} columns")
    
    # Validate loaded data
    validate_dataframe(news_df, f"{data_type} news", ['newsID', 'title', 'abstract', 'category'])
    validate_dataframe(behaviors_df, f"{data_type} behaviors", ['impression_id', 'impressions'])
    
    # 3. Clean and process news data
    print_with_timestamp(f"Cleaning {data_type} news data...")
    news_df_cleaned = news_df.copy()
    
    # Add length columns
    news_df_cleaned['title_length'] = news_df_cleaned['title'].str.len()
    news_df_cleaned['abstract_length'] = news_df_cleaned['abstract'].str.len()
    
    # Filter out articles with very short titles or abstracts
    orig_len = len(news_df_cleaned)
    news_df_cleaned = news_df_cleaned[news_df_cleaned['title_length'] >= 10]
    news_df_cleaned = news_df_cleaned[news_df_cleaned['abstract_length'] >= 20]
    print_with_timestamp(f"After filtering short content: {len(news_df_cleaned)} articles remaining (removed {orig_len - len(news_df_cleaned)})")
    
    # Add reading ease score
    news_df_cleaned = calculate_reading_scores(news_df_cleaned)
    print_with_timestamp(f"Added reading ease scores")
    
    # Save news data for verification
    news_csv = f'{output_dir}/processed_data/{data_type}_news_cleaned.csv'
    news_df_cleaned.to_csv(news_csv, index=False)
    print_with_timestamp(f"Saved cleaned news data to {news_csv}")
    
    # Check if we should re-process impressions or use existing file
    impressions_file = f'{output_dir}/processed_data/{data_type}_impressions.csv'
    if os.path.exists(impressions_file) and behavior_sample_rate == 1.0:
        print_with_timestamp(f"Using existing impressions file: {impressions_file}")
        total_impressions = sum(1 for _ in open(impressions_file)) - 1  # Subtract header
        print_with_timestamp(f"Impression file contains {total_impressions} rows")
    else:
        # 4. Process impressions data with sampling
        print_with_timestamp(f"Processing {data_type} impressions data with sample rate {behavior_sample_rate}...")
        impressions_file, total_impressions = process_impressions_to_file(
            behaviors_df, 
            impressions_file, 
            sample_rate=behavior_sample_rate,
            max_per_behavior=max_impressions
        )
    
    # 5. Calculate CTR from impressions file
    print_with_timestamp(f"Calculating CTR for {data_type} data...")
    article_ctr_data = calculate_ctr_from_impressions_file(impressions_file)
    
    if article_ctr_data is None or len(article_ctr_data) == 0:
        print_with_timestamp(f"ERROR: Failed to calculate CTR for {data_type} data")
        return None
    
    # Create a CTR distribution plot
    ctr_plot_path = f'{output_dir}/plots/{data_type}_ctr_distribution.png'
    ctr_stats = plot_ctr_distribution(article_ctr_data, f"{data_type.capitalize()} Data", ctr_plot_path)
    
    # 6. Merge CTR data with news data
    print_with_timestamp(f"Merging news and engagement data...")
    news_with_engagement = news_df_cleaned.merge(
        article_ctr_data,
        left_on='newsID',
        right_on='news_id',
        how='left'
    )
    
    # Fill missing engagement data with zeros
    engagement_columns = ['total_clicks', 'total_impressions', 'ctr']
    news_with_engagement[engagement_columns] = news_with_engagement[engagement_columns].fillna(0)
    
    # 7. Apply readability binning
    print_with_timestamp("Creating readability bins...")
    try:
        news_with_engagement['reading_ease_bin'] = pd.qcut(
            news_with_engagement['title_reading_ease'], 
            q=5, 
            labels=['Very Hard', 'Hard', 'Medium', 'Easy', 'Very Easy'],
            duplicates='drop'  # Handle potential duplicate bin edges
        )
    except Exception as e:
        print_with_timestamp(f"Warning: Could not create reading ease bins using qcut: {e}")
        # Create a simple binning as fallback
        bins = [-float('inf'), 30, 50, 70, 90, float('inf')]
        labels = ['Very Hard', 'Hard', 'Medium', 'Easy', 'Very Easy']
        news_with_engagement['reading_ease_bin'] = pd.cut(
            news_with_engagement['title_reading_ease'],
            bins=bins,
            labels=labels
        )
    
    # 8. Validate the resulting dataframe
    validate_dataframe(
        news_with_engagement, 
        f"{data_type} processed data", 
        ['newsID', 'title', 'ctr', 'total_impressions', 'title_reading_ease']
    )
    
    # 9. Save processed data
    news_with_engagement.to_csv(output_file, index=False)
    print_with_timestamp(f"Processed {data_type} data saved to {output_file}")
    
    # 10. Create a quick summary report
    summary = {
        'data_type': data_type,
        'news_articles': len(news_with_engagement),
        'total_impressions': total_impressions,
        'total_clicks': int(article_ctr_data['total_clicks'].sum()),
        'articles_with_impressions': int((news_with_engagement['total_impressions'] > 0).sum()),
        'avg_ctr': float(news_with_engagement[news_with_engagement['total_impressions'] > 0]['ctr'].mean()),
        'ctr_stats': ctr_stats,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'sampling': {
            'behavior_sample_rate': behavior_sample_rate,
            'max_impressions_per_behavior': max_impressions
        }
    }
    
    # Save summary to JSON
    with open(f'{output_dir}/processed_data/{data_type}_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print_with_timestamp(f"Processing of {data_type} dataset complete!")
    
    return news_with_engagement, summary

# ================== MAIN SCRIPT ==================
if __name__ == "__main__":
    print_with_timestamp("Starting memory-efficient preprocessing pipeline...")
    print_with_timestamp(f"Sampling configuration: {BEHAVIOR_SAMPLE_RATE*100}% of behaviors, "
                 f"max {MAX_IMPRESSIONS_PER_BEHAVIOR} impressions per behavior")
    
    # Process all dataset splits
    start_time = time.time()
    
    # Process training data
    train_data, train_summary = process_dataset('train', BEHAVIOR_SAMPLE_RATE, MAX_IMPRESSIONS_PER_BEHAVIOR)
    train_elapsed = (time.time() - start_time) / 60
    print_with_timestamp(f"Training data processing took {train_elapsed:.2f} minutes")
    
    # Process validation data - can use a higher sample rate since it's smaller
    val_start = time.time()
    val_data, val_summary = process_dataset('val', min(BEHAVIOR_SAMPLE_RATE * 2, 1.0), MAX_IMPRESSIONS_PER_BEHAVIOR)
    val_elapsed = (time.time() - val_start) / 60
    print_with_timestamp(f"Validation data processing took {val_elapsed:.2f} minutes")
    
    # Process test data if available - can use a higher sample rate since it's smaller
    test_start = time.time()
    test_data, test_summary = process_dataset('test', min(BEHAVIOR_SAMPLE_RATE * 2, 1.0), MAX_IMPRESSIONS_PER_BEHAVIOR)
    test_elapsed = (time.time() - test_start) / 60
    print_with_timestamp(f"Test data processing took {test_elapsed:.2f} minutes")
    
    # Overall timing
    total_elapsed = (time.time() - start_time) / 60
    
    # Create a final report with combined statistics
    full_summary = {
        'overall': {
            'total_processing_time_minutes': total_elapsed,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'sampling_configuration': {
                'behavior_sample_rate': BEHAVIOR_SAMPLE_RATE,
                'max_impressions_per_behavior': MAX_IMPRESSIONS_PER_BEHAVIOR
            }
        },
        'train': train_summary if train_data is not None else None,
        'val': val_summary if val_data is not None else None,
        'test': test_summary if test_data is not None else None
    }
    
    # Save full summary
    with open(f'{output_dir}/processed_data/dataset_statistics.json', 'w') as f:
        json.dump(full_summary, f, indent=2)
    
    print_with_timestamp(f"\n{'='*50}")
    print_with_timestamp(f"PREPROCESSING COMPLETE")
    print_with_timestamp(f"{'='*50}")
    print_with_timestamp(f"Total processing time: {total_elapsed:.2f} minutes")
    print_with_timestamp(f"Processed data saved to '{output_dir}/processed_data/'")
    print_with_timestamp(f"Files created:")
    print_with_timestamp(f"  - train_headline_ctr.csv")
    print_with_timestamp(f"  - val_headline_ctr.csv")
    if test_data is not None:
        print_with_timestamp(f"  - test_headline_ctr.csv")
    print_with_timestamp(f"  - dataset_statistics.json")
    print_with_timestamp(f"Ready for model training!")