import pandas as pd
import numpy as np
import logging
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Import your HeadlineMetrics class but only use it for feature extraction
from headline_metrics import HeadlineMetrics

def prepare_mind_dataset(behaviors_path, news_path):
    """Prepare training data from MIND dataset"""
    print(f"Loading MIND dataset from {behaviors_path} and {news_path}")
    
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
    print(f"Created dataset with {len(training_data)} headlines")
    
    return training_data

def train_model_with_mind_data(train_dir, val_dir, test_dir=None, use_synthetic=None, output_path='headline_ctr_model.pkl'):
    """Train model using MIND dataset with proper train/validation/test splits"""
    
    # Create HeadlineMetrics instance for feature extraction
    metrics = HeadlineMetrics(model_path=None)  # Pass None to avoid loading a model
    
    # Load training data
    print("Loading training data...")
    train_data = prepare_mind_dataset(
        os.path.join(train_dir, 'behaviors.tsv'),
        os.path.join(train_dir, 'news.tsv')
    )
    
    # Load validation data
    print("Loading validation data...")
    val_data = prepare_mind_dataset(
        os.path.join(val_dir, 'behaviors.tsv'),
        os.path.join(val_dir, 'news.tsv')
    )
    
    # Extract features from training data
    print("Extracting features from training data...")
    train_features = []
    train_valid_indices = []
    
    for i, headline in enumerate(train_data['title']):
        try:
            features = metrics.extract_features(headline)
            train_features.append(features)
            train_valid_indices.append(i)
        except Exception as e:
            print(f"Error extracting features for headline: {headline}")
    
    X_train = pd.DataFrame(train_features)
    y_train = train_data.iloc[train_valid_indices]['ctr'].values
    
    # Extract features from validation data
    print("Extracting features from validation data...")
    val_features = []
    val_valid_indices = []
    
    for i, headline in enumerate(val_data['title']):
        try:
            features = metrics.extract_features(headline)
            val_features.append(features)
            val_valid_indices.append(i)
        except Exception as e:
            print(f"Error extracting features for headline: {headline}")
    
    X_val = pd.DataFrame(val_features)
    y_val = val_data.iloc[val_valid_indices]['ctr'].values
    
    # Train base model (without synthetic data)
    print(f"Training base model with {len(X_train)} examples...")
    base_model = RandomForestRegressor(n_estimators=100, random_state=42)
    base_model.fit(X_train, y_train)
    
    # Evaluate on validation set
    base_val_preds = base_model.predict(X_val)
    base_val_mse = mean_squared_error(y_val, base_val_preds)
    base_val_r2 = r2_score(y_val, base_val_preds)
    
    print(f"Base model validation MSE: {base_val_mse:.6f}, R²: {base_val_r2:.4f}")
    
    # Decide whether to use synthetic data
    final_model = base_model
    
    if use_synthetic is None or use_synthetic is True:
        # Generate synthetic data
        train_headlines_df = pd.DataFrame({
            'title': train_data.iloc[train_valid_indices]['title'].values,
            'ctr': y_train
        })
        
        # Create synthetic variations
        print("Generating synthetic variations of headlines...")
        synthetic_headlines = []
        
        # Define headline transformations
        transformations = [
            lambda h: f"5 {h}" if not any(c.isdigit() for c in h) else h,  # Add number
            lambda h: f"{h}?" if not h.endswith('?') else h,  # Add question mark
            lambda h: f"How to {h.lower()}" if not h.lower().startswith('how to') else h,  # Add "How to"
            lambda h: ' '.join(h.split()[:6]) if len(h.split()) > 10 else h,  # Shorten
            lambda h: f"{h} you need to know" if len(h.split()) < 5 else h,  # Lengthen
        ]
        
        multiplier = 2  # How many synthetic headlines per real headline
        for _, row in train_headlines_df.iterrows():
            headline = row['title']
            ctr = row['ctr']
            
            for _ in range(multiplier):
                # Apply 1-2 random transformations
                new_headline = headline
                for transform in np.random.choice(transformations, size=np.random.randint(1, 3), replace=False):
                    new_headline = transform(new_headline)
                
                if new_headline != headline:
                    # Modify CTR slightly (within ±20%)
                    modified_ctr = ctr * (0.8 + 0.4 * np.random.random())
                    synthetic_headlines.append((new_headline, modified_ctr))
        
        # Extract features from synthetic headlines
        print(f"Extracting features from {len(synthetic_headlines)} synthetic headlines...")
        synthetic_features = []
        synthetic_ctrs = []
        
        for headline, ctr in synthetic_headlines:
            try:
                features = metrics.extract_features(headline)
                synthetic_features.append(features)
                synthetic_ctrs.append(ctr)
            except Exception as e:
                print(f"Error extracting features for synthetic headline: {headline}")
        
        X_synthetic = pd.DataFrame(synthetic_features)
        y_synthetic = np.array(synthetic_ctrs)
        
        # Combine with original training data
        X_combined = pd.concat([X_train, X_synthetic], ignore_index=True)
        y_combined = np.concatenate([y_train, y_synthetic])
        
        # Train model with combined data
        print(f"Training model with combined data ({len(X_combined)} examples)...")
        combined_model = RandomForestRegressor(n_estimators=100, random_state=42)
        combined_model.fit(X_combined, y_combined)
        
        # Evaluate on validation set
        combined_val_preds = combined_model.predict(X_val)
        combined_val_mse = mean_squared_error(y_val, combined_val_preds)
        combined_val_r2 = r2_score(y_val, combined_val_preds)
        
        print(f"Combined model validation MSE: {combined_val_mse:.6f}, R²: {combined_val_r2:.4f}")
        
        # Compare and select the better model
        if use_synthetic is True or combined_val_mse < base_val_mse:
            print("Using model trained with synthetic data")
            final_model = combined_model
        else:
            print("Synthetic data did not improve performance; using base model")
    
    # Evaluate on test set if provided
    if test_dir is not None:
        print("Loading test data for final evaluation...")
        test_data = prepare_mind_dataset(
            os.path.join(test_dir, 'behaviors.tsv'),
            os.path.join(test_dir, 'news.tsv')
        )
        
        # Extract features from test data
        test_features = []
        test_valid_indices = []
        
        for i, headline in enumerate(test_data['title']):
            try:
                features = metrics.extract_features(headline)
                test_features.append(features)
                test_valid_indices.append(i)
            except Exception as e:
                print(f"Error extracting features for test headline: {headline}")
        
        X_test = pd.DataFrame(test_features)
        y_test = test_data.iloc[test_valid_indices]['ctr'].values
        
        # Final evaluation
        test_preds = final_model.predict(X_test)
        test_mse = mean_squared_error(y_test, test_preds)
        test_r2 = r2_score(y_test, test_preds)
        
        print(f"Final model test MSE: {test_mse:.6f}, R²: {test_r2:.4f}")
    
    # Show feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    # Save the final model
    with open(output_path, 'wb') as f:
        pickle.dump(final_model, f)
    
    print(f"Model trained and saved to {output_path}")
    return final_model

if __name__ == "__main__":
    # Set paths to MIND dataset directories
    train_dir = "path/to/mind/train"
    val_dir = "path/to/mind/validation"
    test_dir = "path/to/mind/test"  # Optional, set to None to skip test evaluation
    
    # Train model with proper dataset splits
    model = train_model_with_mind_data(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        use_synthetic=None  # Auto-determine based on validation performance
    )