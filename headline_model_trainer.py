# Accuracy is high (86.54% on training, 96.25% on validation): This suggests that the model is correctly predicting the majority class.
# AUC is moderate (0.6623 on training, 0.5993 on validation): The AUC score indicates that the model has some ability to distinguish between classes, but it's not strong.
# Zero Precision/Recall/F1: This typically happens when the model is not predicting any positive instances (i.e., it's predicting all samples as the negative class).

# Imbalanced Dataset Problem
# This pattern strongly suggests you have a highly imbalanced dataset. In headline click prediction:

# Most headlines probably have CTR = 0 (no clicks) → the majority class
# Few headlines have CTR > 0 (got clicks) → the minority class

# The model is taking the "easy way out" by predicting everything as the majority class (no clicks), which gives it a high accuracy but makes it useless for identifying which headlines will get clicks.

import os
import pandas as pd
import numpy as np
import pickle
import logging
import re
from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import time
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            roc_auc_score, confusion_matrix, mean_squared_error, 
                            r2_score, mean_absolute_error)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HeadlineModelTrainer:
    """
    Training function to load data, extract headline features, add semantic features through embedding.
    Use XGBoost regression model to select important features, train, validate and test.
    """
    
    def __init__(self, processed_data_dir='agentic_news_editor/processed_data', use_log_transform=True):
        """Initialize the headline model trainer"""
        self.processed_data_dir = processed_data_dir
        self.embedding_dims = 20
        self.use_log_transform = use_log_transform
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        logging.info(f"Log transform for CTR: {self.use_log_transform}")
        
        # Create output directory for results
        self.output_dir = 'model_output'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Create mapping for common first/last words in headlines
        self.word_to_index = {}
        common_words = ['the', 'a', 'to', 'how', 'why', 'what', 'when', 'is', 'are', 
                        'says', 'report', 'announces', 'reveals', 'show', 'study',
                        'new', 'top', 'best', 'worst', 'first', 'last', 'latest']
        for i, word in enumerate(common_words, 1):  # Start from 1, 0 for unknown
            self.word_to_index[word] = i
            
        # Load and prepare models
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.bert_model = AutoModel.from_pretrained("distilbert-base-uncased")
            self.bert_model = self.bert_model.to(self.device)
            logging.info("Loaded DistilBERT model successfully")
        except Exception as e:
            logging.error(f"Failed to load DistilBERT model: {e}")
            raise ValueError(f"Could not load embedding model: {e}")
        
    def extract_features_cached(self, headlines, cache_name):
        """
        Extract features with caching to disk
        
        Args:
            headlines: List of headlines
            cache_name: Name prefix for the cache file
            
        Returns:
            DataFrame of features
        """
        # Create cache directory if it doesn't exist
        cache_dir = os.path.join(self.output_dir, 'feature_cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        # Define cache file path
        cache_file = os.path.join(cache_dir, f'{cache_name}_features.pkl')
        
        # Check if cache exists
        if os.path.exists(cache_file):
            logging.info(f"Loading cached features from {cache_file}")
            try:
                return pd.read_pickle(cache_file)
            except Exception as e:
                logging.warning(f"Failed to load cached features: {e}")
        
        # Extract features if cache doesn't exist or couldn't be loaded
        features = self.extract_features(headlines)
        
        # Save to cache
        try:
            logging.info(f"Saving features to cache: {cache_file}")
            features.to_pickle(cache_file)
        except Exception as e:
            logging.warning(f"Failed to save features to cache: {e}")
        
        return features
    
    def load_data(self, data_type='train'):
        """Load processed headlines for specified split (train, val, or test)"""
        try:
            file_path = os.path.join(self.processed_data_dir, f'{data_type}_headline_ctr.csv')
            if not os.path.exists(file_path):
                logging.error(f"{data_type.capitalize()} data not found at {file_path}")
                return None
                    
            data = pd.read_csv(file_path)
            logging.info(f"Loaded {len(data)} headlines from {data_type} set")
            
            # Quick data validation
            required_cols = ['title', 'newsID']
            if not all(col in data.columns for col in required_cols):
                missing = [col for col in required_cols if col not in data.columns]
                logging.error(f"Missing required columns in {data_type} data: {missing}")
                return None
                
            return data
        except Exception as e:
            logging.error(f"Error loading {data_type} data: {e}")
            return None
    
    def extract_features(self, headlines):
        """Extract features from headlines for model training"""
        logging.info(f"Extracting features from {len(headlines)} headlines")
        features_list = []
        
        # Process in smaller batches to avoid memory issues
        batch_size = 500
        for i in range(0, len(headlines), batch_size):
            batch = headlines[i:i+batch_size]
            logging.info(f"Processing batch {i//batch_size + 1}/{(len(headlines)-1)//batch_size + 1}")
            
            for headline in batch:
                features = {}
                
                # Basic features based on EDA findings
                features['length'] = len(headline)
                features['word_count'] = len(headline.split())
                features['has_number'] = int(bool(re.search(r'\d', headline)))
                features['num_count'] = len(re.findall(r'\d+', headline))
                features['is_question'] = int(headline.endswith('?') or headline.lower().startswith('how') or 
                                           headline.lower().startswith('what') or headline.lower().startswith('why') or 
                                           headline.lower().startswith('where') or headline.lower().startswith('when') or
                                           headline.lower().startswith('is '))
                features['has_colon'] = int(':' in headline)
                features['has_quote'] = int('"' in headline or "'" in headline)
                features['has_how_to'] = int('how to' in headline.lower())
                
                # Additional features
                features['capital_ratio'] = sum(1 for c in headline if c.isupper()) / len(headline) if len(headline) > 0 else 0
                features['first_word_length'] = len(headline.split()[0]) if len(headline.split()) > 0 else 0
                features['last_word_length'] = len(headline.split()[-1]) if len(headline.split()) > 0 else 0
                features['avg_word_length'] = sum(len(word) for word in headline.split()) / len(headline.split()) if len(headline.split()) > 0 else 0
    
                # 1. Headline structure features
                features['has_number_at_start'] = int(bool(re.match(r'^\d+', headline)))
                features['starts_with_digit'] = int(headline[0].isdigit() if len(headline) > 0 else 0)
                features['has_date'] = int(bool(re.search(r'\b(20\d\d|19\d\d|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b', headline.lower())))
                
                # 2. Clickbait pattern detection
                features['has_question_words'] = int(bool(re.search(r'\b(how|what|why|when|where|who|which)\b', headline.lower())))
                features['has_suspense'] = int(bool(re.search(r'\b(secret|reveal|shock|stun|surprise|you won\'t believe)\b', headline.lower())))
                features['has_urgency'] = int(bool(re.search(r'\b(breaking|urgent|just in|now|today)\b', headline.lower())))
                features['has_list'] = int(bool(re.search(r'\b(\d+ (?:things|ways|tips|reasons|facts))\b', headline.lower())))
                
                # 3. Emotional content
                features['has_positive'] = int(bool(re.search(r'\b(best|top|good|great|amazing|awesome|success)\b', headline.lower())))
                features['has_negative'] = int(bool(re.search(r'\b(worst|bad|terrible|fail|problem|crisis|disaster)\b', headline.lower())))
                features['has_controversy'] = int(bool(re.search(r'\b(vs|versus|against|fight|battle|war|clash)\b', headline.lower())))
                
                # 4. Structure analysis  
                words = headline.split()
                if len(words) > 0:
                    features['first_word'] = self.word_to_index.get(words[0].lower(), 0)
                    features['last_word'] = self.word_to_index.get(words[-1].lower(), 0)
                    features['has_quotes'] = int("\"" in headline or "'" in headline)
                    features['title_case_words'] = sum(1 for word in words if word and word[0].isupper())
                    features['title_case_ratio'] = features['title_case_words'] / len(words) if len(words) > 0 else 0
                
                # 5. Feature interactions (very powerful!)
                features['length_question_interaction'] = features['length'] * features['is_question']
                features['word_count_list_interaction'] = features['word_count'] * features['has_list']
                
                # Get embedding for semantic features
                try:
                    inputs = self.tokenizer(headline, return_tensors="pt", padding=True, truncation=True, max_length=128)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.bert_model(**inputs)
                    
                    # Use the [CLS] token embedding
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                    
                    # Add first embedding dimensions as features
                    for j in range(self.embedding_dims):
                        features[f'emb_{j}'] = embedding[j]
                except Exception as e:
                    logging.error(f"Error extracting embedding for '{headline}': {e}")
                    # Add zero embeddings if failed
                    for j in range(self.embedding_dims):
                        features[f'emb_{j}'] = 0.0
                
                features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def manual_feature_selection(self, features, target, threshold=0.2):
        """
        Manual feature selection based on feature importance.
        This replaces the SelectFromModel approach that causes compatibility issues.
        
        Args:
            features: DataFrame of features
            target: Target values
            threshold: Importance threshold for keeping features (0-1)
            
        Returns:
            list: Selected feature names
        """
        logging.info("Performing manual feature selection...")
        
        # Create a simple model for getting feature importances
        model = XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=5, random_state=42)
        
        # Fit the model
        model.fit(features, target)
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Create a DataFrame with features and their importances
        importance_df = pd.DataFrame({
            'feature': features.columns,
            'importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Select features based on threshold
        if 0 < threshold < 1:
            # Use relative threshold (keep features with importance >= threshold * max_importance)
            max_importance = importance_df['importance'].max()
            selected = importance_df[importance_df['importance'] >= threshold * max_importance]
        else:
            # Keep top N features
            n_features = int(max(1, min(50, len(features.columns) * 0.5)))  # Default: keep 50% of features, max 50
            selected = importance_df.head(n_features)
            
        selected_features = selected['feature'].tolist()
        logging.info(f"Selected {len(selected_features)} features with threshold {threshold}")
        
        return selected_features
    
    # Add this method to your HeadlineModelTrainer class
    def evaluate_model_ranking(self, model_data, val_data):
        """
        Evaluate model's ability to rank headlines by their likelihood of getting clicks
        
        Args:
            model_data: Dictionary containing model, feature_names and other model info
            val_data: Validation data with 'title' and 'ctr' columns
        
        Returns:
            Dictionary with ranking evaluation metrics
        """
        logging.info("Evaluating model's headline ranking ability...")
        
        # Get validation headlines and CTRs
        headlines = val_data['title'].values
        actual_ctrs = val_data['ctr'].values
        
        # Extract features for validation data
        val_features = self.extract_features(headlines)
        
        # Get selected features - FIX: Changed 'result' to 'model_data'
        feature_names = model_data['feature_names']
        
        # Filter features
        val_features_sel = val_features[feature_names]
        
        # Get model predictions
        if model_data.get('model_type', 'regression') == 'classification':
            # For classification model, get probability of click
            predicted_probs = model_data['model'].predict_proba(val_features_sel)[:, 1]
        else:
            # For regression model, get predicted CTR
            predicted_probs = model_data['model'].predict(val_features_sel)
            if model_data.get('use_log_transform', False):
                predicted_probs = np.expm1(predicted_probs)
        
        # Create DataFrame with results
        results_df = pd.DataFrame({
            'headline': headlines,
            'actual_ctr': actual_ctrs,
            'predicted_prob': predicted_probs,
            'has_clicks': (actual_ctrs > 0).astype(int)
        })
        
        # Calculate Spearman rank correlation
        rank_correlation = results_df['predicted_prob'].corr(results_df['actual_ctr'], method='spearman')
        logging.info(f"Spearman rank correlation: {rank_correlation:.4f}")
        
        # Sort by predicted probability
        sorted_results = results_df.sort_values('predicted_prob', ascending=False).reset_index(drop=True)
        
        # Create buckets for analysis
        num_headlines = len(sorted_results)
        num_buckets = 10  # Aim for 10 buckets
        bucket_size = max(10, num_headlines // num_buckets)  # At least 10 headlines per bucket
        
        buckets = []
        for i in range(0, num_headlines, bucket_size):
            end_idx = min(i + bucket_size, num_headlines)
            bucket = sorted_results.iloc[i:end_idx]
            
            # Calculate metrics for this bucket
            bucket_index = i // bucket_size + 1
            avg_predicted = bucket['predicted_prob'].mean()
            avg_ctr = bucket['actual_ctr'].mean()
            click_rate = bucket['has_clicks'].mean() * 100
            
            buckets.append({
                'bucket': bucket_index,
                'size': len(bucket),
                'avg_predicted': avg_predicted,
                'avg_actual_ctr': avg_ctr,
                'click_rate': click_rate
            })
        
        # Create DataFrame of bucket results
        buckets_df = pd.DataFrame(buckets)
        
        # Calculate lift metrics
        overall_avg_ctr = results_df['actual_ctr'].mean()
        overall_click_rate = results_df['has_clicks'].mean() * 100
        
        # Get metrics for top buckets
        top_bucket = buckets_df.iloc[0]
        top_3_buckets = buckets_df.iloc[:3]
        
        top_bucket_lift = (top_bucket['avg_actual_ctr'] / overall_avg_ctr - 1) * 100
        top_3_buckets_lift = (top_3_buckets['avg_actual_ctr'].mean() / overall_avg_ctr - 1) * 100
        
        logging.info(f"Overall average CTR: {overall_avg_ctr:.6f}")
        logging.info(f"Overall click rate: {overall_click_rate:.2f}%")
        logging.info(f"Top bucket average CTR: {top_bucket['avg_actual_ctr']:.6f} (lift: {top_bucket_lift:.1f}%)")
        logging.info(f"Top 3 buckets average CTR: {top_3_buckets['avg_actual_ctr'].mean():.6f} (lift: {top_3_buckets_lift:.1f}%)")
        
        # Log bucket results
        logging.info("CTR and click rate by bucket (higher bucket = higher predicted probability):")
        for _, row in buckets_df.iterrows():
            logging.info(f"Bucket {row['bucket']}: Avg CTR={row['avg_actual_ctr']:.6f}, Click Rate={row['click_rate']:.2f}%, Pred={row['avg_predicted']:.4f}")
        
        # Create visualization
        plt.figure(figsize=(14, 10))
        
        # Plot 1: Average CTR by bucket
        plt.subplot(2, 2, 1)
        plt.bar(buckets_df['bucket'], buckets_df['avg_actual_ctr'])
        plt.axhline(y=overall_avg_ctr, color='r', linestyle='--', label=f'Overall avg: {overall_avg_ctr:.6f}')
        plt.xlabel('Bucket (1=lowest predicted, 10=highest predicted)')
        plt.ylabel('Average CTR')
        plt.title('Average CTR by Predicted Probability Bucket')
        plt.legend()
        
        # Plot 2: Click rate by bucket
        plt.subplot(2, 2, 2)
        plt.bar(buckets_df['bucket'], buckets_df['click_rate'])
        plt.axhline(y=overall_click_rate, color='r', linestyle='--', label=f'Overall: {overall_click_rate:.2f}%')
        plt.xlabel('Bucket (1=lowest predicted, 10=highest predicted)')
        plt.ylabel('Headlines with Clicks (%)')
        plt.title('% Headlines with Clicks by Predicted Probability Bucket')
        plt.legend()
        
        # Plot 3: CTR Lift by bucket
        plt.subplot(2, 2, 3)
        ctr_lift = (buckets_df['avg_actual_ctr'] / overall_avg_ctr - 1) * 100
        plt.bar(buckets_df['bucket'], ctr_lift)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Bucket (1=lowest predicted, 10=highest predicted)')
        plt.ylabel('CTR Lift (%)')
        plt.title('CTR Lift Compared to Average')
        
        # Plot 4: Scatterplot of predicted vs actual
        plt.subplot(2, 2, 4)
        plt.scatter(results_df['predicted_prob'], results_df['actual_ctr'], alpha=0.3)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Actual CTR')
        plt.title(f'Actual vs Predicted (Spearman corr: {rank_correlation:.4f})')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'ranking_performance.png'), dpi=300)
        plt.close()
        
        # Also save data for further analysis
        sorted_results.to_csv(os.path.join(self.output_dir, 'headline_ranking_results.csv'), index=False)
        buckets_df.to_csv(os.path.join(self.output_dir, 'headline_buckets.csv'), index=False)
        
        # Return evaluation results
        return {
            'rank_correlation': rank_correlation,
            'buckets': buckets_df,
            'overall_avg_ctr': overall_avg_ctr,
            'overall_click_rate': overall_click_rate,
            'top_bucket_lift': top_bucket_lift,
            'top_3_buckets_lift': top_3_buckets_lift
        }

    def train_model(self, train_features, train_ctr, val_features=None, val_ctr=None,
           output_file='headline_ctr_model.pkl', mode='classification', class_weight=None):
        """
        Train model
        
        Parameters:
        -----------
        train_features : pandas DataFrame
            Features for training the model
        train_ctr : numpy array or pandas Series
            Target CTR values for training
        val_features : pandas DataFrame, optional
            Features for validation
        val_ctr : numpy array or pandas Series, optional
            Target CTR values for validation
        output_file : str, optional
            Filename to save the trained model
        mode : str, optional
            'classification' or 'regression'
         class_weight : dict, optional
        Class weights for handling imbalanced data
            
        Returns:
        --------
        dict
            Dictionary containing model, selected features, and performance metrics
        """
        if mode == 'classification':
            logging.info("Training headline CTR prediction model (Classification mode)")
            
            # Convert CTR to binary targets (1 if CTR > 0, 0 if CTR = 0)
            train_y_binary = (train_ctr > 0).astype(int)
            val_y_binary = (val_ctr > 0).astype(int) if val_ctr is not None else None
            
            # Get feature selection
            selected_features = self.manual_feature_selection(train_features, train_y_binary, threshold=0.2)
            
            # Filter features
            train_features_sel = train_features[selected_features]
            val_features_sel = val_features[selected_features] if val_features is not None else None
            
            # Define model parameters for classification
            params = {
                'n_estimators': 100, 
                'learning_rate': 0.1,
                'max_depth': 5, 
                'min_child_weight': 3,
                'subsample': 0.8, 
                'colsample_bytree': 0.8,
                'reg_alpha': 0.5,
                'reg_lambda': 1.0,
                'objective': 'binary:logistic',
                'random_state': 42,
                'n_jobs': -1
            }
            
            # Add scale_pos_weight parameter if class_weight provided
            if class_weight is not None:
                # XGBoost uses scale_pos_weight parameter
                # This is the ratio of negative to positive samples
                scale_pos_weight = class_weight[1] / class_weight[0]
                params['scale_pos_weight'] = scale_pos_weight
                logging.info(f"Using scale_pos_weight={scale_pos_weight} for imbalanced classes")
            
            logging.info(f"Training classification model with params: {params}")
            
            # Create model
            final = XGBClassifier(**params)
            
            # Train model
            start = time.time()
            if val_features_sel is not None and val_y_binary is not None:
                final.fit(
                    train_features_sel, train_y_binary,
                    eval_set=[(val_features_sel, val_y_binary)],
                    eval_metric='auc',
                    early_stopping_rounds=10,
                    verbose=0
                )
            else:
                final.fit(train_features_sel, train_y_binary, verbose=0)
            
            ttime = time.time() - start
            logging.info(f"Model trained in {ttime:.2f}s")
            
            # Training metrics
            train_pred_proba = final.predict_proba(train_features_sel)[:, 1]
            train_pred = (train_pred_proba > 0.5).astype(int)
            
            train_metrics = {
                'accuracy': accuracy_score(train_y_binary, train_pred),
                'precision': precision_score(train_y_binary, train_pred, zero_division=0),
                'recall': recall_score(train_y_binary, train_pred, zero_division=0),
                'f1': f1_score(train_y_binary, train_pred, zero_division=0),
                'auc': roc_auc_score(train_y_binary, train_pred_proba) if len(np.unique(train_y_binary)) > 1 else 0.5
            }
            
            logging.info(f"Train metrics: {train_metrics}")
            
            # Validation metrics
            val_metrics = {}
            if val_features_sel is not None and val_y_binary is not None:
                val_pred_proba = final.predict_proba(val_features_sel)[:, 1]
                val_pred = (val_pred_proba > 0.5).astype(int)
                
                val_metrics = {
                    'accuracy': accuracy_score(val_y_binary, val_pred),
                    'precision': precision_score(val_y_binary, val_pred, zero_division=0),
                    'recall': recall_score(val_y_binary, val_pred, zero_division=0),
                    'f1': f1_score(val_y_binary, val_pred, zero_division=0),
                    'auc': roc_auc_score(val_y_binary, val_pred_proba) if len(np.unique(val_y_binary)) > 1 else 0.5
                }
                
                logging.info(f"Val metrics: {val_metrics}")
                
                # Confusion matrix
                cm = confusion_matrix(val_y_binary, val_pred)
                logging.info(f"Validation confusion matrix:\n{cm}")
                
                # Create visualization of classifier performance
                self.visualize_classifier_performance(val_y_binary, val_pred_proba, 'validation_classifier_performance.png')
            
            # Get feature importances
            importances = final.feature_importances_
            fi = pd.DataFrame({
                'feature': selected_features,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            logging.info(f"Top features: {fi.head(10).to_string()}")
            
            # Save model
            model_data = {
                'model': final,
                'model_type': 'classification',
                'feature_names': selected_features,
                'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'metrics': {'train': train_metrics, 'val': val_metrics}
            }
            
            with open(os.path.join(self.output_dir, output_file), 'wb') as f:
                pickle.dump(model_data, f)
            logging.info(f"Model saved to {os.path.join(self.output_dir, output_file)}")
            
            # Visualize feature importance
            self.visualize_feature_importance(fi)
            fi.to_csv(os.path.join(self.output_dir, 'feature_importance.csv'), index=False)
            logging.info("Feature importance saved.")
            
            return {
                'model': final, 
                'selected_features': selected_features, 
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'feature_importances': fi, 
                'training_time': ttime,
                'model_type': 'classification'
            }
        else:
            # Original regression code
            logging.info("Training headline CTR prediction model (Regression mode)")
            
            # Apply log transform if specified
            if self.use_log_transform:
                train_y = np.log1p(train_ctr)
                val_y = np.log1p(val_ctr) if val_ctr is not None else None
            else:
                train_y = train_ctr
                val_y = val_ctr
            
            # Manual feature selection (doesn't depend on scikit-learn)
            selected_features = self.manual_feature_selection(train_features, train_y, threshold=0.2)
            
            # Filter features
            train_features_sel = train_features[selected_features]
            val_features_sel = val_features[selected_features] if val_features is not None else None
            
            # Define final model parameters
            final_params = {
                'n_estimators': 200, 
                'learning_rate': 0.01,
                'max_depth': 4, 
                'min_child_weight': 5,
                'subsample': 0.7, 
                'colsample_bytree': 0.7,
                'reg_alpha': 1.0,
                'reg_lambda': 2.0,
                'objective': 'reg:squarederror',
                'random_state': 42,
                'n_jobs': -1
            }
            
            logging.info(f"Training final model with params: {final_params}")
            
            # Create final model
            final = XGBRegressor(**final_params)
            
            # Train final model
            start = time.time()
            if val_features_sel is not None and val_y is not None:
                final.fit(
                    train_features_sel, train_y,
                    eval_set=[(val_features_sel, val_y)],
                    eval_metric='rmse',
                    early_stopping_rounds=50,
                    verbose=0
                )
            else:
                final.fit(train_features_sel, train_y, verbose=0)
            ttime = time.time() - start
            logging.info(f"Final model trained in {ttime:.2f}s")
            
            # Evaluate on train/val
            train_pred_t = final.predict(train_features_sel)
            train_pred = np.expm1(train_pred_t) if self.use_log_transform else train_pred_t
            train_metrics = {
                'mse': mean_squared_error(train_ctr, train_pred),
                'rmse': np.sqrt(mean_squared_error(train_ctr, train_pred)),
                'mae': mean_absolute_error(train_ctr, train_pred),
                'r2': r2_score(train_ctr, train_pred)
            }
            logging.info(f"Train metrics: {train_metrics}")
            
            val_metrics = {}
            if val_features_sel is not None and val_ctr is not None:
                val_pred_t = final.predict(val_features_sel)
                val_pred = np.expm1(val_pred_t) if self.use_log_transform else val_pred_t
                val_metrics = {
                    'mse': mean_squared_error(val_ctr, val_pred),
                    'rmse': np.sqrt(mean_squared_error(val_ctr, val_pred)),
                    'mae': mean_absolute_error(val_ctr, val_pred),
                    'r2': r2_score(val_ctr, val_pred)
                }
                logging.info(f"Val metrics: {val_metrics}")
                self.visualize_predictions(val_ctr, val_pred, 'validation_predictions.png')
            
            # Feature importances
            importances = final.feature_importances_
            fi = pd.DataFrame({
                'feature': selected_features,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            logging.info(f"Top features: {fi.head(10).to_string()}")
            
            # Save model and metrics
            model_data = {
                'model': final,
                'model_type': 'regression',
                'use_log_transform': self.use_log_transform,
                'feature_names': selected_features,
                'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'metrics': {'train': train_metrics, 'val': val_metrics}
            }
            
            with open(os.path.join(self.output_dir, output_file), 'wb') as f:
                pickle.dump(model_data, f)
            logging.info(f"Model saved to {os.path.join(self.output_dir, output_file)}")
            
            # Visualize feature importance
            self.visualize_feature_importance(fi)
            fi.to_csv(os.path.join(self.output_dir, 'feature_importance.csv'), index=False)
            logging.info("Feature importance saved.")
            
            return {
                'model': final, 
                'selected_features': selected_features, 
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'feature_importances': fi, 
                'training_time': ttime,
                'model_type': 'regression'
            }
    
    def visualize_ctr_distribution(self, train_ctr, val_ctr=None):
        """Visualize the distribution of CTR values in train and val sets"""
        try:
            plt.figure(figsize=(12, 6))
            
            # Train distribution
            plt.subplot(1, 2, 1)
            sns.histplot(train_ctr, kde=True, bins=50)
            plt.title('Train CTR Distribution')
            plt.xlabel('CTR')
            plt.ylabel('Count')
            plt.grid(alpha=0.3)
            
            # Log distribution
            plt.subplot(1, 2, 2)
            non_zero_ctr = train_ctr[train_ctr > 0]
            if len(non_zero_ctr) > 0:
                sns.histplot(np.log1p(non_zero_ctr), kde=True, bins=50)
                plt.title('Train log(CTR+1) Distribution (non-zero values)')
                plt.xlabel('log(CTR+1)')
                plt.ylabel('Count')
                plt.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'ctr_distribution.png'), dpi=300)
            plt.close()
            
            # Compare train vs val if validation data provided
            if val_ctr is not None:
                plt.figure(figsize=(12, 5))
                
                # Plot distributions
                plt.subplot(1, 2, 1)
                sns.kdeplot(train_ctr, label='Train', fill=True, alpha=0.3)
                sns.kdeplot(val_ctr, label='Validation', fill=True, alpha=0.3)
                plt.title('CTR Distribution Comparison')
                plt.xlabel('CTR')
                plt.ylabel('Density')
                plt.legend()
                plt.grid(alpha=0.3)
                
                # Plot log distributions for non-zero values
                plt.subplot(1, 2, 2)
                train_non_zero = train_ctr[train_ctr > 0]
                val_non_zero = val_ctr[val_ctr > 0]
                
                if len(train_non_zero) > 0 and len(val_non_zero) > 0:
                    sns.kdeplot(np.log1p(train_non_zero), label='Train', fill=True, alpha=0.3)
                    sns.kdeplot(np.log1p(val_non_zero), label='Validation', fill=True, alpha=0.3)
                    plt.title('log(CTR+1) Distribution Comparison (non-zero)')
                    plt.xlabel('log(CTR+1)')
                    plt.ylabel('Density')
                    plt.legend()
                    plt.grid(alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'ctr_distribution_comparison.png'), dpi=300)
                plt.close()
            
            logging.info(f"CTR distribution visualizations saved to {self.output_dir}")
        except Exception as e:
            logging.error(f"Error creating CTR distribution visualization: {e}")
    
    def visualize_feature_importance(self, feature_importance_df, output_file='feature_importance.png'):
        """Visualize feature importance from model"""
        try:
            # Get top features
            top_n = min(20, len(feature_importance_df))
            top_features = feature_importance_df.head(top_n)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', data=top_features)
            plt.title(f'Top {top_n} Feature Importance')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, output_file), dpi=300)
            plt.close()
            
            logging.info(f"Feature importance visualization saved to {os.path.join(self.output_dir, output_file)}")
        except Exception as e:
            logging.error(f"Error creating feature importance visualization: {e}")
    
    def visualize_predictions(self, y_true, y_pred, output_file='predictions.png'):
        """Create scatter plot of actual vs predicted values"""
        try:
            plt.figure(figsize=(10, 8))
            
            # Calculate metrics for title
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            
            # Create scatter plot
            plt.scatter(y_true, y_pred, alpha=0.5)
            
            # Add perfect prediction line
            max_val = max(np.max(y_true), np.max(y_pred))
            min_val = min(np.min(y_true), np.min(y_pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            plt.title(f'Actual vs Predicted CTR (RMSE: {rmse:.6f}, R²: {r2:.4f})')
            plt.xlabel('Actual CTR')
            plt.ylabel('Predicted CTR')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, output_file), dpi=300)
            plt.close()
            
            # Create additional plot with log scale if values are very small
            if np.min(y_true) > 0 and np.min(y_pred) > 0:
                plt.figure(figsize=(10, 8))
                plt.scatter(y_true, y_pred, alpha=0.5)
                plt.plot([min_val, max_val], [min_val, max_val], 'r--')
                plt.xscale('log')
                plt.yscale('log')
                plt.title(f'Actual vs Predicted CTR (Log Scale)')
                plt.xlabel('Actual CTR (log scale)')
                plt.ylabel('Predicted CTR (log scale)')
                plt.grid(alpha=0.3, which='both')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'log_{output_file}'), dpi=300)
                plt.close()
            
            logging.info(f"Prediction visualization saved to {os.path.join(self.output_dir, output_file)}")
        except Exception as e:
            logging.error(f"Error creating prediction visualization: {e}")
          
    def visualize_classifier_performance(self, y_true, y_proba, output_file='classifier_performance.png'):
        """Create visualization of classifier performance with ROC curve and PR curve"""
        try:
            from sklearn.metrics import roc_curve, precision_recall_curve, auc
            
            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            
            ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax1.set_xlim([0.0, 1.0])
            ax1.set_ylim([0.0, 1.05])
            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate')
            ax1.set_title('Receiver Operating Characteristic')
            ax1.legend(loc="lower right")
            
            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            pr_auc = auc(recall, precision)
            
            ax2.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
            ax2.set_xlim([0.0, 1.0])
            ax2.set_ylim([0.0, 1.05])
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.set_title('Precision-Recall Curve')
            ax2.axhline(y=sum(y_true)/len(y_true), color='red', linestyle='--', label=f'Baseline ({sum(y_true)/len(y_true):.3f})')
            ax2.legend(loc="lower left")
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, output_file), dpi=300)
            plt.close()
            
            logging.info(f"Classifier performance visualization saved to {os.path.join(self.output_dir, output_file)}")
        except Exception as e:
            logging.error(f"Error creating classifier performance visualization: {e}")       
        
    def run_training_pipeline(self, use_cached_features=True):
        """Run the complete model training pipeline with train/val/test splits"""
        try:
            # Load data for all splits
            train_data = self.load_data('train')
            val_data = self.load_data('val')
            test_data = self.load_data('test')
            
            if train_data is None or val_data is None:
                logging.error("Could not load required data. Aborting.")
                return None
            
            # Handle NaN values
            train_data = train_data.dropna(subset=['title', 'ctr'])
            val_data = val_data.dropna(subset=['title', 'ctr'])
            if test_data is not None:
                test_data = test_data.dropna(subset=['title'])
            
            
            # Print class distribution
            train_clicks = (train_data['ctr'] > 0).mean() * 100
            val_clicks = (val_data['ctr'] > 0).mean() * 100
            logging.info(f"Train dataset: {train_clicks:.2f}% headlines with clicks")
            logging.info(f"Validation dataset: {val_clicks:.2f}% headlines with clicks")
            
            # Visualize CTR distribution
            self.visualize_ctr_distribution(train_data['ctr'].values, val_data['ctr'].values)
            
             # Extract features for all splits (with caching)
            logging.info("Extracting features for training data...")
            if use_cached_features:
                train_features = self.extract_features_cached(train_data['title'].values, 'train')
            else:
                train_features = self.extract_features(train_data['title'].values)
            
            logging.info("Extracting features for validation data...")
            if use_cached_features:
                val_features = self.extract_features_cached(val_data['title'].values, 'val')
            else:
                val_features = self.extract_features(val_data['title'].values)
                
            test_features = None
            if test_data is not None:
                logging.info("Extracting features for test data...")
                if use_cached_features:
                    test_features = self.extract_features_cached(test_data['title'].values, 'test')
                else:
                    test_features = self.extract_features(test_data['title'].values)
                
            # Compute class weights for handling imbalanced data
            from sklearn.utils.class_weight import compute_class_weight
            import numpy as np
            
            # Convert to binary targets
            train_y_binary = (train_data['ctr'].values > 0).astype(int)
            
            # Compute class weights - higher weight for minority class
            class_weight = {}
            class_weights = compute_class_weight('balanced', classes=np.unique(train_y_binary), y=train_y_binary)
            for i, weight in enumerate(class_weights):
                class_weight[i] = weight
            
            logging.info(f"Using class weights: {class_weight} to handle imbalanced data")
        
            # Train model
            result = self.train_model(
                train_features, train_data['ctr'].values,
                val_features, val_data['ctr'].values,
                output_file='headline_classifier_model.pkl',
                mode='classification',
                class_weight=class_weight
            )
            
            if result is None:
                logging.error("Model training failed. Aborting pipeline.")
                return None
                
            # If test data is available, generate predictions
            if test_data is not None and test_features is not None and 'selected_features' in result:
                logging.info("Generating predictions for test set...")
                
                # Ensure we're using the selected features only
                test_features_selected = test_features[result['selected_features']]
                
                if self.use_log_transform:
                    test_pred_transformed = result['model'].predict(test_features_selected)
                    test_predictions = np.expm1(test_pred_transformed)
                else:
                    test_predictions = result['model'].predict(test_features_selected)
                
                # Save test predictions
                test_results = test_data[['newsID', 'title']].copy()
                test_results['predicted_ctr'] = test_predictions
                test_results.to_csv(os.path.join(self.output_dir, 'test_predictions.csv'), index=False)
                logging.info(f"Test predictions saved to {os.path.join(self.output_dir, 'test_predictions.csv')}")
            
            # Create a report
            self.create_model_report(result, train_data, val_data, test_data)
            
                    # Add this line to evaluate ranking performance
            ranking_eval = self.evaluate_model_ranking(result, val_data)
            
            # You might want to add the ranking evaluation results to the result dictionary
            if ranking_eval is not None:
                result['ranking_evaluation'] = ranking_eval
        
            return result
        except Exception as e:
            logging.error(f"Error in training pipeline: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return None
                
            return result
        except Exception as e:
            logging.error(f"Error in training pipeline: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return None

    def create_model_report(self, result, train_data, val_data, test_data=None):
        """Create a markdown report about the model performance"""
        if result is None:
            return
            
        # Access nested metrics correctly
        train_metrics = result['train_metrics']
        val_metrics = result['val_metrics']
        
        # Check model type
        is_classifier = result.get('model_type', 'regression') == 'classification'
        
        if is_classifier:
            report = f"""# Headline Click Prediction Model Report
    Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

    ## Model Configuration
    - Model Type: XGBClassifier (Binary Classification)
    - Task: Predict if a headline will be clicked
    - Training Time: {result['training_time']:.2f} seconds

    ## Model Performance
    - Training Accuracy: {train_metrics['accuracy']:.4f}
    - Training Precision: {train_metrics['precision']:.4f}
    - Training Recall: {train_metrics['recall']:.4f}
    - Training F1 Score: {train_metrics['f1']:.4f}
    - Training AUC: {train_metrics['auc']:.4f}
    """

            if val_metrics:
                report += f"""
    - Validation Accuracy: {val_metrics['accuracy']:.4f}
    - Validation Precision: {val_metrics['precision']:.4f}
    - Validation Recall: {val_metrics['recall']:.4f}
    - Validation F1 Score: {val_metrics['f1']:.4f}
    - Validation AUC: {val_metrics['auc']:.4f}
    """
        else:
            # Original regression report
            report = f"""# Headline CTR Prediction Model Report
    Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

    ## Model Configuration
    - Model Type: XGBRegressor
    - Log Transform CTR: {self.use_log_transform}
    - Training Time: {result['training_time']:.2f} seconds

    ## Model Performance
    - Training MSE: {train_metrics['mse']:.6f}
    - Training RMSE: {train_metrics['rmse']:.6f}
    - Training MAE: {train_metrics['mae']:.6f}
    - Training R-squared: {train_metrics['r2']:.4f}
    """

            if val_metrics:
                report += f"""
    - Validation MSE: {val_metrics['mse']:.6f}
    - Validation RMSE: {val_metrics['rmse']:.6f}
    - Validation MAE: {val_metrics['mae']:.6f}
    - Validation R-squared: {val_metrics['r2']:.4f}
    """

        report += f"""
    ## Dataset Summary
    - Training headlines: {len(train_data)}
    - Validation headlines: {len(val_data)}
    - Test headlines: {len(test_data) if test_data is not None else 'N/A'}
    """

        if not is_classifier:
            report += f"""
    - Training CTR range: {train_data['ctr'].min():.4f} to {train_data['ctr'].max():.4f}
    - Training Mean CTR: {train_data['ctr'].mean():.4f}
    - Validation Mean CTR: {val_data['ctr'].mean():.4f}
    """
        else:
            report += f"""
    - Training Click Rate: {(train_data['ctr'] > 0).mean():.4f}
    - Validation Click Rate: {(val_data['ctr'] > 0).mean():.4f}
    """

        report += """
    ## Key Feature Importances
    """
        
        for i, row in result['feature_importances'].head(15).iterrows():
            report += f"- {row['feature']}: {row['importance']:.4f}\n"
        
        report += """
    ## Usage Guidelines
    """

        if is_classifier:
            report += """
    This model can be used to predict whether headlines will be clicked.
    It outputs a probability score (0-1) representing the likelihood of a click.
    It can be integrated into a headline optimization workflow for automated
    headline suggestions or ranking.
    """
        else:
            report += """
    This model can be used to predict the expected CTR of news headlines.
    It can be integrated into a headline optimization workflow for automated
    headline suggestions or ranking.
    """

        report += """
    ## Features Used
    The model uses both basic text features and semantic embeddings:
    - Basic features: length, word count, question marks, numbers, etc.
    - Semantic features: BERT embeddings to capture meaning

    ## Visualizations
    The following visualizations have been generated:
    - feature_importance.png: Importance of different features
    """

        if is_classifier:
            report += """
    - validation_classifier_performance.png: ROC and PR curves for classification performance
    """
        else:
            report += """
    - ctr_distribution.png: Distribution of CTR values
    - validation_predictions.png: True vs predicted CTR values
    """
        
        with open(os.path.join(self.output_dir, 'headline_model_report.md'), 'w') as f:
            f.write(report)
        
        logging.info(f"Model report saved to {os.path.join(self.output_dir, 'headline_model_report.md')}")
        
        print(f"Clicks percentage in training: {(train_data['ctr'] > 0).mean() * 100:.2f}%")
        print(f"Clicks percentage in validation: {(val_data['ctr'] > 0).mean() * 100:.2f}%")

    def predict_ctr(self, headlines, model_data=None, model_file='headline_ctr_model.pkl'):
        """
        Predict CTR for headlines
        
        Args:
            headlines: List of headlines
            model_data: Model data dictionary
            model_file: File to load model from if model_data not provided
            
        Returns:
            array: Predicted CTR values or click probabilities
        """
        if model_data is None:
            # Load the model
            try:
                model_path = os.path.join(self.output_dir, model_file)
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
            except Exception as e:
                logging.error(f"Error loading model: {e}")
                return None
        
        # Extract features
        features = self.extract_features(headlines)
        
        # Get selected features
        feature_names = model_data['feature_names']
        
        # Add missing features with zeros
        for f in feature_names:
            if f not in features.columns:
                features[f] = 0.0
        
        # Get filtered features
        features_filtered = features[feature_names]
        
        # Check model type
        is_classifier = model_data.get('model_type', 'regression') == 'classification'
        
        # Make predictions
        if is_classifier:
            # For classifier, return probability of click
            predictions = model_data['model'].predict_proba(features_filtered)[:, 1]
        else:
            # For regressor
            predictions = model_data['model'].predict(features_filtered)
            # Apply inverse transform if needed
            if model_data.get('use_log_transform', False):
                predictions = np.expm1(predictions)
                
        return predictions

    def headline_analysis(self, headline, model_data=None, model_file='headline_ctr_model.pkl'):
        """
        Analyze what makes a headline perform well or poorly
        
        Args:
            headline: Headline string
            model_data: Model data dictionary (optional)
            model_file: File to load model from if model_data not provided
            
        Returns:
            dict: Analysis information
        """
        if model_data is None:
            # Load the model
            try:
                model_path = os.path.join(self.output_dir, model_file)
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
            except Exception as e:
                logging.error(f"Error loading model: {e}")
                return None
        
        # Extract features
        features = self.extract_features([headline])
        
        # Get selected features
        feature_names = model_data['feature_names']
        
        # Add missing features
        for f in feature_names:
            if f not in features.columns:
                features[f] = 0.0
        
        # Get filtered features
        features_filtered = features[feature_names]
        
        # Check model type
        is_classifier = model_data.get('model_type', 'regression') == 'classification'
        
        # Make prediction
        if is_classifier:
            # For classifier, get probability
            prediction = model_data['model'].predict_proba(features_filtered)[0, 1]
            prediction_label = "Click probability"
        else:
            # For regressor, get CTR
            prediction = model_data['model'].predict(features_filtered)[0]
            # Apply inverse transform if needed
            if model_data.get('use_log_transform', False):
                prediction = np.expm1(prediction)
            prediction_label = "Predicted CTR"
        
        # Get feature importances
        importances = model_data['model'].feature_importances_
        
        # Calculate feature contributions
        contributions = []
        for i, feat in enumerate(feature_names):
            value = features_filtered[feat].values[0]
            importance = importances[i]
            contribution = value * importance
            contributions.append({
                'feature': feat,
                'value': value,
                'importance': importance,
                'contribution': abs(contribution),
                'raw_contribution': contribution
            })
        
        # Sort by absolute contribution
        contributions = sorted(contributions, key=lambda x: x['contribution'], reverse=True)
        
        # Basic headline stats
        headline_stats = {
            'length': len(headline),
            'word_count': len(headline.split()),
            'is_question': '?' in headline,
            'has_colon': ':' in headline,
            'has_number': bool(re.search(r'\d', headline)),
            'uppercase_ratio': sum(1 for c in headline if c.isupper()) / max(1, len(headline)),
            'clickbait_indicators': self._detect_clickbait_patterns(headline)
        }
        
        # Create analysis result
        analysis = {
            'headline': headline,
            'prediction': prediction,
            'prediction_type': 'click_probability' if is_classifier else 'ctr',
            'headline_stats': headline_stats,
            'top_contributions': contributions[:10],
            'negative_contributions': [c for c in contributions if c['raw_contribution'] < 0][:5],
            'positive_contributions': [c for c in contributions if c['raw_contribution'] > 0][:5]
        }
        
        # Print analysis
        print(f"\nHeadline: '{headline}'")
        print(f"{prediction_label}: {prediction:.6f}")
        
        if is_classifier:
            print(f"Interpretation: {'Likely to be clicked' if prediction > 0.5 else 'Unlikely to be clicked'}")
        
        print("\nPositive contributing factors:")
        for c in analysis['positive_contributions']:
            print(f"- {c['feature']}: {c['raw_contribution']:.6f}")
            
        print("\nNegative contributing factors:")
        for c in analysis['negative_contributions']:
            print(f"- {c['feature']}: {c['raw_contribution']:.6f}")
        
        return analysis

    def _detect_clickbait_patterns(self, headline):
        """Detect common clickbait patterns in a headline"""
        patterns = []
        
        headline_lower = headline.lower()
        
        if headline.endswith('?'):
            patterns.append('question_ending')
            
        if headline.endswith('!'):
            patterns.append('exclamation_ending')
            
        if ':' in headline:
            patterns.append('colon')
            
        if re.search(r'\b(how|what|why|when|where|who|which)\b', headline_lower):
            patterns.append('question_words')
            
        if re.search(r'\b(secret|reveal|shock|stun|surprise|you won\'t believe)\b', headline_lower):
            patterns.append('suspense')
            
        if re.search(r'\b(breaking|urgent|just in|now|today)\b', headline_lower):
            patterns.append('urgency')
            
        if re.search(r'\b(\d+ (?:things|ways|tips|reasons|facts))\b', headline_lower):
            patterns.append('list')
            
        return patterns

    def optimize_headline(self, headline, n_variations=10, model_data=None, model_file='headline_ctr_model.pkl'):
        """
        Generate optimized variations of a headline
        
        Args:
            headline: Original headline
            n_variations: Number of variations to generate
            model_data: Model data dictionary (optional)
            model_file: File to load model from if model_data not provided
            
        Returns:
            DataFrame: Original and optimized headlines with predicted CTR
        """
        if model_data is None:
            # Load the model
            try:
                model_path = os.path.join(self.output_dir, model_file)
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
            except Exception as e:
                logging.error(f"Error loading model: {e}")
                return None
        
        # Define common headline optimizations
        optimizations = [
            lambda h: f"How {h}",  # Add "How" at the beginning
            lambda h: f"Why {h}",  # Add "Why" at the beginning
            lambda h: f"{h}?",  # Add question mark
            lambda h: f"{h}!",  # Add exclamation mark
            lambda h: f"Top 10 {h}",  # Add list prefix
            lambda h: f"{h}: What You Need to Know",  # Add clarifying suffix
            lambda h: f"Breaking: {h}",  # Add urgency
            lambda h: f"Expert Reveals: {h}",  # Add authority
            lambda h: f"The Truth About {h}",  # Add intrigue
            lambda h: f"You Won't Believe {h}",  # Add surprise
            lambda h: h.title(),  # Title case
            lambda h: h.upper(),  # All caps for emphasis
            lambda h: f"New Study Shows {h}",  # Add credibility
            lambda h: f"{h} [PHOTOS]",  # Add media indicator
            lambda h: f"{h} - Here's Why",  # Add explanation indicator
        ]
        
        # Generate variations (unique ones only)
        variations = [headline]  # Include original
        for optimize_func in optimizations:
            try:
                variation = optimize_func(headline)
                if variation not in variations:
                    variations.append(variation)
                    
                # Stop if we have enough variations
                if len(variations) >= n_variations + 1:  # +1 for original
                    break
            except Exception as e:
                logging.warning(f"Error generating headline variation: {e}")
        
        # Get predictions for all variations
        predictions = self.predict_ctr(variations, model_data)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'headline': variations,
            'predicted_ctr': predictions,
            'is_original': [i == 0 for i in range(len(variations))]
        })
        
        # Sort by predicted CTR
        results = results.sort_values('predicted_ctr', ascending=False).reset_index(drop=True)
        
        return results
    
def main():
    """Main function to run the headline model training pipeline"""
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train headline CTR prediction model')
    parser.add_argument('--data_dir', type=str, default='agentic_news_editor/processed_data', 
                    help='Directory containing processed data files')
    parser.add_argument('--log_transform', action='store_true',
                    help='Apply log transform to CTR values (for regression mode)')
    parser.add_argument('--mode', type=str, choices=['classification', 'regression'], default='classification',
                    help='Training mode: classification or regression')
    parser.add_argument('--predict', type=str,
                    help='Enter a headline to predict CTR or click probability')
    parser.add_argument('--analyze', type=str,
                    help='Analyze what makes a headline perform well')
    parser.add_argument('--optimize', type=str,
                    help='Generate optimized versions of a headline')
    parser.add_argument('--n_variations', type=int, default=10,
                    help='Number of headline variations to generate')
    parser.add_argument('--model_file', type=str, default=None,
                    help='Model file to use for prediction/analysis (auto-detected if None)')
    parser.add_argument('--use_cached_features', action='store_true',
                       help='Use cached features if available')
    args = parser.parse_args()
    
    # Set default model file based on mode
    if args.model_file is None:
        if args.mode == 'classification':
            args.model_file = 'headline_classifier_model.pkl'
        else:
            args.model_file = 'headline_ctr_model.pkl'
            
    # Otherwise, run the training pipeline
    else:
        # Select the appropriate mode
        mode = args.mode
        output_file = 'headline_classifier_model.pkl' if mode == 'classification' else 'headline_ctr_model.pkl'
        
        print(f"Running training pipeline in {mode.upper()} mode...")
        result = trainer.run_training_pipeline(use_cached_features=args.use_cached_features)
    
    # Create trainer
    trainer = HeadlineModelTrainer(
        processed_data_dir=args.data_dir,
        use_log_transform=args.log_transform
    )
    
    # Check if we're predicting a headline
    if args.predict:
        try:
            # Find the model file
            if not os.path.exists(os.path.join(trainer.output_dir, args.model_file)):
                # Try to find any model file
                model_files = [f for f in os.listdir(trainer.output_dir) if f.endswith('.pkl')]
                if model_files:
                    args.model_file = model_files[0]
                    print(f"Using model file: {args.model_file}")
                else:
                    print("No model file found. Please train a model first.")
                    return
            
            # Load model
            model_path = os.path.join(trainer.output_dir, args.model_file)
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            # Get prediction
            prediction = trainer.predict_ctr([args.predict], model_data)[0]
            
            # Check model type
            is_classifier = model_data.get('model_type', 'regression') == 'classification'
            
            print(f"\nHeadline: '{args.predict}'")
            if is_classifier:
                print(f"Click probability: {prediction:.4f} ({prediction*100:.1f}%)")
                print(f"Interpretation: {'Likely to be clicked' if prediction > 0.5 else 'Unlikely to be clicked'}")
            else:
                print(f"Predicted CTR: {prediction:.6f}")
            
        except Exception as e:
            logging.error(f"Error predicting headline CTR: {e}")
            import traceback
            logging.error(traceback.format_exc())
            
    # Check if we're analyzing a headline
    elif args.analyze:
        try:
            # Find the model file
            if not os.path.exists(os.path.join(trainer.output_dir, args.model_file)):
                # Try to find any model file
                model_files = [f for f in os.listdir(trainer.output_dir) if f.endswith('.pkl')]
                if model_files:
                    args.model_file = model_files[0]
                    print(f"Using model file: {args.model_file}")
                else:
                    print("No model file found. Please train a model first.")
                    return
            
            # Load model
            model_path = os.path.join(trainer.output_dir, args.model_file)
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            # Analyze headline
            trainer.headline_analysis(args.analyze, model_data)
            
        except Exception as e:
            logging.error(f"Error analyzing headline: {e}")
            import traceback
            logging.error(traceback.format_exc())
            
    # Check if we're optimizing a headline
    elif args.optimize:
        try:
            # Find the model file
            if not os.path.exists(os.path.join(trainer.output_dir, args.model_file)):
                # Try to find any model file
                model_files = [f for f in os.listdir(trainer.output_dir) if f.endswith('.pkl')]
                if model_files:
                    args.model_file = model_files[0]
                    print(f"Using model file: {args.model_file}")
                else:
                    print("No model file found. Please train a model first.")
                    return
            
            # Load model
            model_path = os.path.join(trainer.output_dir, args.model_file)
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            # Optimize headline
            results = trainer.optimize_headline(
                args.optimize,
                n_variations=args.n_variations,
                model_data=model_data
            )
            
            # Display results
            original = results[results['is_original']]
            original_ctr = original['predicted_ctr'].iloc[0]
            
            # Check model type
            is_classifier = model_data.get('model_type', 'regression') == 'classification'
            prediction_label = "Click probability" if is_classifier else "Predicted CTR"
            
            print(f"\nOriginal headline: '{args.optimize}'")
            print(f"{prediction_label}: {original_ctr:.6f}")
            
            if is_classifier and original_ctr > 0.5:
                print("Interpretation: Likely to be clicked")
            elif is_classifier:
                print("Interpretation: Unlikely to be clicked")
            
            print("\nOptimized headlines:")
            for i, row in results[~results['is_original']].head(5).iterrows():
                improvement = (row['predicted_ctr'] / original_ctr - 1) * 100
                print(f"{i+1}. '{row['headline']}'")
                print(f"   {prediction_label}: {row['predicted_ctr']:.6f} ({improvement:.1f}% improvement)")
                
        except Exception as e:
            logging.error(f"Error optimizing headline: {e}")
            import traceback
            logging.error(traceback.format_exc())
            
    # Otherwise, run the training pipeline
    else:
        # Select the appropriate mode
        mode = args.mode
        output_file = 'headline_classifier_model.pkl' if mode == 'classification' else 'headline_ctr_model.pkl'
        
        print(f"Running training pipeline in {mode.upper()} mode...")
        result = trainer.run_training_pipeline()
        
        if result is not None:
            print(f"Model training complete.")
            
            # Print appropriate metrics based on model type
            if result.get('model_type', 'regression') == 'classification':
                print(f"Training metrics:")
                print(f"  Accuracy: {result['train_metrics']['accuracy']:.4f}")
                print(f"  Precision: {result['train_metrics']['precision']:.4f}")
                print(f"  Recall: {result['train_metrics']['recall']:.4f}")
                print(f"  F1 Score: {result['train_metrics']['f1']:.4f}")
                print(f"  AUC: {result['train_metrics']['auc']:.4f}")
                
                if 'val_metrics' in result and result['val_metrics']:
                    print(f"Validation metrics:")
                    print(f"  Accuracy: {result['val_metrics']['accuracy']:.4f}")
                    print(f"  Precision: {result['val_metrics']['precision']:.4f}")
                    print(f"  Recall: {result['val_metrics']['recall']:.4f}")
                    print(f"  F1 Score: {result['val_metrics']['f1']:.4f}")
                    print(f"  AUC: {result['val_metrics']['auc']:.4f}")
            else:
                print(f"Training R-squared: {result['train_metrics']['r2']:.4f}")
                if 'val_metrics' in result and result['val_metrics']:
                    print(f"Validation R-squared: {result['val_metrics']['r2']:.4f}")
            
            print(f"Results saved to {trainer.output_dir}")
        else:
            print("Model training failed.")


if __name__ == "__main__":
    main()