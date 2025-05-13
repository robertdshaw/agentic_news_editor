# Complete MIND Dataset Preprocessing Pipeline with Timestamps
# Memory-optimized version to prevent system crashes
# ---------------------------------------------------------------
# Processes splits (train, validation, test) and preserves per-article timestamps

import pandas as pd
import numpy as np
import os
import time
import json
import csv
import random
import logging
import gc  # Garbage collection
from datetime import datetime
from textstat import flesch_reading_ease
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ================== CONFIGURATION (MEMORY OPTIMIZED) ==================
output_dir = 'agentic_news_editor'
plots_dir = f'{output_dir}/plots'
processed_dir = f'{output_dir}/processed_data'
for d in (output_dir, plots_dir, processed_dir):
    os.makedirs(d, exist_ok=True)

BEHAVIOR_SAMPLE_RATE = 0.05
TRAIN_NEWS_SAMPLE_RATE = 0.1
VAL_NEWS_SAMPLE_RATE   = 0.1
TEST_NEWS_SAMPLE_RATE  = 0.1
MAX_IMPRESSIONS_PER_BEHAVIOR = 20
MIN_IMPRESSIONS_FOR_CTR      = 5

# Memory optimization settings
CHUNK_SIZE = 10000  # Reduced from 50000
BEHAVIOR_CHUNK_SIZE = 5000  # New: Process behaviors in smaller chunks
CTR_CHUNK_SIZE = 100000  # For CTR calculation

# ================== LOGGING SETUP ==================
def setup_logging():
    """Setup logging configuration"""
    log_file = f'{output_dir}/preprocessing.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ================== MEMORY MANAGEMENT UTILS ==================
def monitor_memory():
    """Monitor memory usage"""
    import psutil
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    logger.info(f"Current memory usage: {memory_mb:.2f} MB")
    return memory_mb

def cleanup_memory():
    """Force garbage collection"""
    gc.collect()
    logger.info("Memory cleanup performed")

# ================== UTILS ==================
def print_with_timestamp(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}")
    logger.info(msg)

def safe_flesch_score(text):
    """Calculate Flesch reading ease score with error handling"""
    try:
        if isinstance(text, str) and text.strip():
            return flesch_reading_ease(text)
        return 0
    except Exception as e:
        return 0  # Don't log every error

def calculate_reading_scores(df):
    """Calculate reading ease scores with memory optimization"""
    print_with_timestamp("Calculating reading ease scores...")
    
    # Process in chunks to avoid memory issues
    chunk_size = 1000
    df['title_length'] = df['title'].str.len()
    df['abstract_length'] = df['abstract'].str.len()
    
    # Initialize columns
    df['title_reading_ease'] = 0.0
    df['abstract_reading_ease'] = 0.0
    
    # Process in smaller chunks
    for i in range(0, len(df), chunk_size):
        end_idx = min(i + chunk_size, len(df))
        chunk = df.iloc[i:end_idx]
        
        # Process titles
        df.loc[chunk.index, 'title_reading_ease'] = chunk['title'].apply(safe_flesch_score)
        
        # Process abstracts
        df.loc[chunk.index, 'abstract_reading_ease'] = chunk['abstract'].apply(safe_flesch_score)
        
        # Clean up after each chunk
        if i % (chunk_size * 10) == 0:
            cleanup_memory()
    
    logger.info(f"Calculated reading scores for {len(df)} articles")
    return df

def validate_data_files(data_type):
    """Validate that required data files exist"""
    news_path = f"{data_type}_data/news.tsv"
    behaviors_path = f"{data_type}_data/behaviors.tsv"
    
    if not os.path.exists(news_path):
        raise FileNotFoundError(f"News file not found: {news_path}")
    if not os.path.exists(behaviors_path):
        raise FileNotFoundError(f"Behaviors file not found: {behaviors_path}")
    
    logger.info(f"Validated data files for {data_type}")
    return news_path, behaviors_path

def validate_dataframe(df, name, required_columns):
    """Validate DataFrame structure and contents"""
    if df.empty:
        raise ValueError(f"{name} DataFrame is empty")
    
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"{name} missing columns: {missing_cols}")
    
    logger.info(f"Validated {name} DataFrame with {len(df)} rows")

# ================== MEMORY-OPTIMIZED IMPRESSION PROCESSING ==================

def detect_impression_format(behaviors_df, sample_size=100):
    """Detect if impressions include click information with sampling"""
    # Only check a sample to avoid memory issues
    sample_df = behaviors_df.head(sample_size)
    
    for _, row in sample_df.iterrows():
        impressions = row.get('impressions', '')
        if isinstance(impressions, str) and impressions.strip():
            items = impressions.split()
            if items:
                has_clicks = any('-' in item for item in items[:5])
                logger.info(f"Detected impression format: {'WITH' if has_clicks else 'WITHOUT'} click information")
                return has_clicks
    
    logger.warning("Could not detect impression format, assuming no click information")
    return False

def process_impressions_to_file(behaviors_df, output_path, sampled_article_ids=None, sample_rate=1.0, max_per_behavior=None):
    """
    Process impressions from behaviors and write directly to a CSV file
    MEMORY OPTIMIZED VERSION
    """
    # Create CSV file and write header
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['impression_id', 'user_id', 'time', 'news_id', 'clicked'])

    # NOTE: Don't apply additional sampling here if already sampled
    if sampled_article_ids is None and sample_rate < 1.0:
        behaviors_df = behaviors_df.sample(frac=sample_rate, random_state=42)
        print_with_timestamp(f"Sampling behaviors at rate {sample_rate}, using {len(behaviors_df)} records")

    # Detect format with smaller sample
    format_with_clicks = detect_impression_format(behaviors_df, sample_size=100)

    total_impressions = 0
    chunk_size = BEHAVIOR_CHUNK_SIZE  # Use smaller chunk size
    num_chunks = (len(behaviors_df) + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(behaviors_df))
        chunk = behaviors_df.iloc[start:end]
        
        # Process chunk
        impressions_to_write = []
        
        for _, row in chunk.iterrows():
            impression_id = row['impression_id']
            user_id = row['user_id']
            time_val = row['time']
            
            impressions = row.get('impressions', '')
            if not isinstance(impressions, str) or not impressions.strip():
                continue
            
            items = impressions.split()
            
            if max_per_behavior and len(items) > max_per_behavior:
                items = random.sample(items, max_per_behavior)
            
            for item in items:
                if not item:
                    continue
                
                if format_with_clicks:
                    parts = item.split('-')
                    if len(parts) == 2:
                        news_id, clicked = parts
                        try:
                            clicked = int(clicked)
                        except ValueError:
                            continue
                    else:
                        continue
                else:
                    news_id = item
                    clicked = 0
                
                if sampled_article_ids is None or news_id in sampled_article_ids:
                    impressions_to_write.append([impression_id, user_id, time_val, news_id, clicked])
        
        # Write chunk to file
        with open(output_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(impressions_to_write)
            total_impressions += len(impressions_to_write)
        
        # Memory cleanup every 10 chunks
        if chunk_idx % 10 == 0:
            cleanup_memory()
            print_with_timestamp(f"Processed {chunk_idx + 1}/{num_chunks} chunks, {total_impressions} impressions so far")
    
    print_with_timestamp(f"Total impressions written: {total_impressions}")
    return output_path, total_impressions

def calculate_ctr_from_impressions_file(impressions_path):
    """
    Calculate CTR metrics from impressions file with memory optimization
    """
    if not os.path.exists(impressions_path):
        raise FileNotFoundError(f"Impressions file not found: {impressions_path}")
    
    clicks = Counter()
    imps = Counter()
    
    # Use smaller chunk size for reading
    chunk_num = 0
    try:
        for chunk in pd.read_csv(impressions_path, chunksize=CTR_CHUNK_SIZE):
            for _, row in chunk.iterrows():
                nid = row['news_id']
                imps[nid] += 1
                clicks[nid] += row['clicked']
            
            chunk_num += 1
            if chunk_num % 5 == 0:  # Clean up every 5 chunks
                cleanup_memory()
                print_with_timestamp(f"Processed {chunk_num} chunks for CTR calculation")
            
    except Exception as e:
        logger.error(f"Error processing impressions file: {e}")
        raise
    
    data = [
        {
            'news_id': nid, 
            'total_clicks': clicks[nid], 
            'total_impressions': imps[nid], 
            'ctr': (clicks[nid]/imps[nid] if imps[nid] > 0 else 0)
        }
        for nid in imps
    ]
    
    logger.info(f"Calculated CTR for {len(data)} articles")
    return pd.DataFrame(data)

# ================== MEMORY-OPTIMIZED TIMESTAMP EXTRACTION ==================

def extract_timestamps_optimized(behaviors_df):
    """Extract timestamps in a memory-efficient way"""
    print_with_timestamp("Extracting timestamps (memory-optimized)...")
    
    # Process in chunks to avoid memory issues
    chunk_size = 5000
    timestamp_data = []
    
    for i in range(0, len(behaviors_df), chunk_size):
        end_idx = min(i + chunk_size, len(behaviors_df))
        chunk = behaviors_df.iloc[i:end_idx][['time', 'impressions']].copy()
        
        # Clean up impressions
        chunk['impressions'] = chunk['impressions'].fillna('')
        chunk = chunk[chunk['impressions'] != '']
        
        # Process impressions
        for _, row in chunk.iterrows():
            time_val = row['time']
            impressions = row['impressions']
            
            if not impressions:
                continue
            
            items = impressions.split()
            for item in items:
                if not item:
                    continue
                news_id = item.split('-')[0]
                timestamp_data.append({'news_id': news_id, 'time': time_val})
        
        # Cleanup after each chunk
        del chunk
        cleanup_memory()
        
        if i % (chunk_size * 10) == 0:
            print_with_timestamp(f"Processed timestamps for {i}/{len(behaviors_df)} behaviors")
    
    # Create DataFrame and aggregate
    ts_df = pd.DataFrame(timestamp_data)
    ts = ts_df.groupby('news_id')['time'].agg(['min', 'max']).rename(columns={'min': 'first_seen', 'max': 'last_seen'})
    ts.reset_index(inplace=True)
    
    print_with_timestamp(f"Aggregated timestamps for {len(ts)} articles")
    return ts

# ================== MAIN PROCESSING (MEMORY OPTIMIZED) ==================

def process_dataset(data_type='train'):
    """Process dataset with memory optimization"""
    print_with_timestamp(f"\n{'='*40}\nPROCESSING {data_type.upper()}\n{'='*40}")
    
    try:
        # Monitor memory at start
        monitor_memory()
        
        # Validate configuration
        sample_rate = {'train': TRAIN_NEWS_SAMPLE_RATE, 'val': VAL_NEWS_SAMPLE_RATE, 'test': TEST_NEWS_SAMPLE_RATE}[data_type]
        
        # Validate data files exist
        news_path, behaviors_path = validate_data_files(data_type)
        out_csv = f"{processed_dir}/{data_type}_headline_ctr.csv"

        # 1. Load news with memory optimization
        news_cols = ['newsID','category','subcategory','title','abstract','url','title_entities','abstract_entities']
        print_with_timestamp("Loading news data...")
        news_df = pd.read_csv(news_path, sep='\t', names=news_cols, header=None, low_memory=False)
        validate_dataframe(news_df, "News", news_cols)
        print_with_timestamp(f"Loaded {len(news_df)} news rows")
        monitor_memory()

        # 2. Sample news articles
        if sample_rate < 1.0:
            news_df = news_df.sample(frac=sample_rate, random_state=42)
            print_with_timestamp(f"Sampled news to {len(news_df)} rows (rate: {sample_rate})")
        sampled_ids = set(news_df['newsID'])

        # 3. Load behaviors in chunks WITH SAMPLING
        beh_cols = ['impression_id','user_id','time','history','impressions']
        print_with_timestamp(f"Loading behaviors data (with {BEHAVIOR_SAMPLE_RATE} sampling)...")
        
        # Read behaviors in chunks to avoid memory issues
        behaviors_chunks = []
        chunk_reader = pd.read_csv(behaviors_path, sep='\t', names=beh_cols, header=None, chunksize=10000)
        
        for chunk in chunk_reader:
            # APPLY BEHAVIOR SAMPLING HERE - before processing timestamps
            if BEHAVIOR_SAMPLE_RATE < 1.0:
                chunk = chunk.sample(frac=BEHAVIOR_SAMPLE_RATE, random_state=42)
            
            # Convert time immediately
            chunk['time'] = pd.to_datetime(chunk['time'], errors='coerce')
            behaviors_chunks.append(chunk)
            
            if len(behaviors_chunks) % 10 == 0:
                total_so_far = sum(len(c) for c in behaviors_chunks)
                print_with_timestamp(f"Loaded {len(behaviors_chunks)} behavior chunks, {total_so_far} rows after sampling...")
                cleanup_memory()
        
        # Combine chunks
        beh = pd.concat(behaviors_chunks, ignore_index=True)
        del behaviors_chunks  # Free memory
        cleanup_memory()
        
        validate_dataframe(beh, "Behaviors", beh_cols)
        print_with_timestamp(f"Final behavior count after sampling: {len(beh)} rows (from ~{len(beh)/BEHAVIOR_SAMPLE_RATE:,.0f} original)")
        monitor_memory()

        # 4. Extract per-article timestamps (now using sampled data)
        ts = extract_timestamps_optimized(beh)
        monitor_memory()

        # 5. Process impressions file & compute CTR
        print_with_timestamp("Processing impressions...")
        imp_file, total_imp = process_impressions_to_file(
            beh, f'{processed_dir}/{data_type}_impressions.csv',
            sampled_article_ids=sampled_ids, 
            sample_rate=1.0,  # Don't sample again, already sampled
            max_per_behavior=MAX_IMPRESSIONS_PER_BEHAVIOR
        )
        
        # Free behaviors DataFrame memory
        del beh
        cleanup_memory()
        monitor_memory()
        
        print_with_timestamp("Calculating CTR...")
        ctr_df = calculate_ctr_from_impressions_file(imp_file)
        monitor_memory()

        # 6. Merge news, CTR and timestamps
        print_with_timestamp("Merging data...")
        news_df.rename(columns={'newsID':'news_id'}, inplace=True)
        
        # Merge in steps to control memory
        print_with_timestamp("Merging news with CTR...")
        merged = news_df.merge(ctr_df, on='news_id', how='left')
        del ctr_df
        cleanup_memory()
        
        print_with_timestamp("Merging with timestamps...")
        merged = merged.merge(ts, on='news_id', how='left')
        del ts
        cleanup_memory()
        
        merged[['total_clicks','total_impressions','ctr']] = merged[['total_clicks','total_impressions','ctr']].fillna(0)
        monitor_memory()

        # 7. Add reading ease & bins (optimized)
        merged = calculate_reading_scores(merged)
        
        # Create reading ease bins with error handling
        try:
            merged['reading_ease_bin'] = pd.qcut(
                merged['title_reading_ease'], 5, 
                labels=['Very Hard','Hard','Medium','Easy','Very Easy'], 
                duplicates='drop'
            )
        except Exception as e:
            logger.warning(f"Error creating reading ease bins: {e}")
            merged['reading_ease_bin'] = pd.cut(
                merged['title_reading_ease'], 
                bins=5, 
                labels=['Very Hard','Hard','Medium','Easy','Very Easy']
            )

        # 8. Save
        print_with_timestamp("Saving results...")
        merged.to_csv(out_csv, index=False)
        print_with_timestamp(f"Saved {data_type} to {out_csv}")
        logger.info(f"Successfully processed {data_type} dataset with {len(merged)} articles")
        
        # Final cleanup
        cleanup_memory()
        monitor_memory()
        
        return merged
        
    except Exception as e:
        logger.error(f"Failed to process {data_type} dataset: {e}")
        cleanup_memory()
        raise

def main():
    """Main execution function with memory monitoring"""
    try:
        print_with_timestamp("Starting MIND dataset preprocessing...")
        logger.info("Starting MIND dataset preprocessing")
        monitor_memory()
        
        for split in ['train', 'val', 'test']:
            try:
                print_with_timestamp(f"\n=== Starting {split} split ===")
                result = process_dataset(split)
                
                # Clean up after each split
                del result
                cleanup_memory()
                
                print_with_timestamp(f"=== Completed {split} split ===")
                monitor_memory()
                
            except Exception as e:
                logger.error(f"Failed to process {split} split: {e}")
                print_with_timestamp(f"Warning: Failed to process {split} split. Continuing with next...")
                cleanup_memory()
        
        print_with_timestamp("Preprocessing complete")
        logger.info("Preprocessing complete")
        
    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}")
        cleanup_memory()
        raise

if __name__ == '__main__':
    main()