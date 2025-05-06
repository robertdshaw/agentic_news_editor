import pandas as pd
import numpy as np
import logging
import pickle
import re
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Import your HeadlineMetrics class but only use it for feature extraction
from headline_metrics import HeadlineMetrics

def prepare_mind_dataset(behaviors_path, news_path):
    """Prepare training data from MIND dataset"""
    print("Loading MIND dataset...")
    
    # Load MIND behavior data
    behaviors = pd.read_csv(behaviors_path, sep='\t', 
                           header=None, 
                           names=['impression_id', 'user_id', 'time', 'history', 'impressions'])
    
    # Load news data
    news = pd.read_csv(news_path, sep='\t',
                      header=None, 
                      names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])
    
    print(f"Loaded {len(behaviors)} behavior records and {len(news)} news items")
    
    # Process behavior data to get click-through rates for headlines
    print("Processing behavior data...")
    impressions_expanded = []
    for _, row in behaviors.iterrows():
        impression_items = row['impressions'].split()
        for item in impression_items:
            parts = item.split('-')
            if len(parts) == 2:
                news_id, click = parts
                impressions_expanded.append({
                    'impression_id': row['impression_id'],
                    'user_id': row['user_id'],
                    'news_id': news_id,
                    'clicked': int(click)
                })
    
    impressions_df = pd.DataFrame(impressions_expanded)
    print(f"Processed {len(impressions_df)} impression records")
    
    # Merge with news data to get headlines
    clicks_with_headlines = impressions_df.merge(news[['news_id', 'title']], on='news_id')
    
    # Calculate CTR for each headline
    print("Calculating CTR for each headline...")
    headline_ctr = clicks_with_headlines.groupby('title').agg({'clicked': ['sum', 'count']})
    headline_ctr.columns = ['click_sum', 'impression_count']
    headline_ctr['ctr'] = headline_ctr['click_sum'] / headline_ctr['impression_count']
    
    # Filter out headlines with very few impressions to ensure statistical significance
    headline_ctr = headline_ctr[headline_ctr['impression_count'] >= 10]
    
    # Prepare final dataset
    training_data = headline_ctr.reset_index()
    print(f"Created training dataset with {len(training_data)} headlines")
    
    return training_data

def generate_synthetic_data(real_data, multiplier=2):
    """Generate synthetic headline variations to augment training data"""
    print("Generating synthetic data...")
    
    synthetic_data = []
    
    # Define common headline transformations
    transformations = [
        # Add a number
        lambda h: f"5 {h}" if not any(c.isdigit() for c in h) else h,
        # Add a question mark
        lambda h: f"{h}?" if not h.endswith('?') else h,
        # Add "How to"
        lambda h: f"How to {h.lower()}" if not h.lower().startswith('how to') else h,
        # Shorten if too long
        lambda h: ' '.join(h.split()[:6]) if len(h.split()) > 10 else h,
        # Lengthen if too short
        lambda h: f"{h} you need to know about" if len(h.split()) < 5 else h,
    ]
    
    for _, row in real_data.iterrows():
        headline = row['title']
        ctr = row['ctr']
        
        # Apply random transformations
        for _ in range(multiplier):
            # Apply 1-2 random transformations
            new_headline = headline
            for transform in np.random.choice(transformations, size=np.random.randint(1, 3), replace=False):
                new_headline = transform(new_headline)
                
            if new_headline != headline:  # Only add if it's different
                # Modify CTR slightly (within ±20%)
                modified_ctr = ctr * (0.8 + 0.4 * np.random.random())
                
                synthetic_data.append({
                    'title': new_headline,
                    'click_sum': 0,  # Placeholder
                    'impression_count': 0,  # Placeholder
                    'ctr': modified_ctr
                })
    
    synthetic_df = pd.DataFrame(synthetic_data)
    print(f"Generated {len(synthetic_df)} synthetic headlines")
    
    return synthetic_df

def extract_features_batch(headlines, metrics):
    """Extract features for a batch of headlines"""
    features = []
    for headline in headlines:
        try:
            features.append(metrics.extract_features(headline))
        except Exception as e:
            print(f"Error extracting features for headline '{headline}': {e}")
            # Add empty features
            features.append({})
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features)
    
    # Drop rows with missing features
    valid_rows = ~features_df.isnull().any(axis=1)
    return features_df[valid_rows], valid_rows

def train_headline_model(real_data, use_synthetic=None, output_path='headline_ctr_model.pkl'):
    """
    Train the ML model using real data and optionally synthetic data
    
    Args:
        real_data: DataFrame with real headline data
        use_synthetic: None to auto-determine, True to force use, False to disable
        output_path: Where to save the trained model
        
    Returns:
        Trained model
    """
    print(f"Training headline CTR prediction model...")
    
    # Split real data into train and validation sets
    train_data, val_data = train_test_split(real_data, test_size=0.2, random_state=42)
    print(f"Split data into {len(train_data)} training and {len(val_data)} validation examples")
    
    # Create a temporary HeadlineMetrics instance for feature extraction
    metrics = HeadlineMetrics(model_path=None)  # Pass None to avoid loading a model
    
    # Extract features from training data
    print("Extracting features from training data...")
    X_train, valid_train_rows = extract_features_batch(train_data['title'], metrics)
    y_train = train_data['ctr'].iloc[valid_train_rows.values]
    
    # Extract features from validation data
    print("Extracting features from validation data...")
    X_val, valid_val_rows = extract_features_batch(val_data['title'], metrics)
    y_val = val_data['ctr'].iloc[valid_val_rows.values]
    
    print(f"Training base model with {len(X_train)} examples...")
    
    # Train base model on real data only
    base_model = RandomForestRegressor(n_estimators=100, random_state=42)
    base_model.fit(X_train, y_train)
    
    # Evaluate base model
    base_val_preds = base_model.predict(X_val)
    base_val_mse = mean_squared_error(y_val, base_val_preds)
    base_val_r2 = r2_score(y_val, base_val_preds)
    
    print(f"Base model validation MSE: {base_val_mse:.6f}, R²: {base_val_r2:.4f}")
    
    # Decide whether to use synthetic data
    final_model = base_model
    
    if use_synthetic is None or use_synthetic:
        # Generate synthetic data
        synthetic_data = generate_synthetic_data(train_data, multiplier=2)
        
        # Extract features from synthetic data
        print("Extracting features from synthetic data...")
        X_synthetic, valid_synthetic_rows = extract_features_batch(synthetic_data['title'], metrics)
        y_synthetic = synthetic_data['ctr'].iloc[valid_synthetic_rows.values]
        
        # Combine with real training data
        X_combined = pd.concat([X_train, X_synthetic], ignore_index=True)
        y_combined = pd.concat([y_train, y_synthetic], ignore_index=True)
        
        print(f"Training combined model with {len(X_combined)} examples ({len(X_train)} real + {len(X_synthetic)} synthetic)...")
        
        # Train model on combined data
        combined_model = RandomForestRegressor(n_estimators=100, random_state=42)
        combined_model.fit(X_combined, y_combined)
        
        # Evaluate combined model
        combined_val_preds = combined_model.predict(X_val)
        combined_val_mse = mean_squared_error(y_val, combined_val_preds)
        combined_val_r2 = r2_score(y_val, combined_val_preds)
        
        print(f"Combined model validation MSE: {combined_val_mse:.6f}, R²: {combined_val_r2:.4f}")
        
        # Compare models and decide which to use
        if use_synthetic is True or combined_val_mse < base_val_mse:
            print("Using model trained with synthetic data")
            final_model = combined_model
        else:
            print("Synthetic data did not improve performance; using base model")
    else:
        print("Synthetic data generation disabled; using base model")
    
    # Identify feature importance for interpretability
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    # Save model
    with open(output_path, 'wb') as f:
        pickle.dump(final_model, f)
    
    print(f"Model trained and saved to {output_path}")
    return final_model

if __name__ == "__main__":
    # Set paths to MIND dataset files
    behaviors_path = "path/to/mind/behaviors.tsv"
    news_path = "path/to/mind/news.tsv"
    
    # Prepare dataset
    real_data = prepare_mind_dataset(behaviors_path, news_path)
    
    # Train and save model with auto-determination of synthetic data usage
    model = train_headline_model(real_data, use_synthetic=None)