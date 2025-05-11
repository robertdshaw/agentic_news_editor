import os
import pandas as pd
import numpy as np
import pickle
import logging
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import re
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectFromModel
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import time
from xgboost import XGBRegressor
from datetime import datetime
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HeadlineModelTrainer:
    """
    Trains and evaluates a model for predicting headline CTR based on the MIND dataset
    and the preprocessed train/val/test splits.
    """
    
    def __init__(self, processed_data_dir='agentic_news_editor/processed_data', use_log_transform=True, 
                 cache_dir='model_cache', batch_size=500):
        """Initialize the headline model trainer
        
        Args:
            processed_data_dir (str): Directory containing processed data files
            use_log_transform (bool): Whether to apply log transform to CTR values
            cache_dir (str): Directory to cache intermediate results
            batch_size (int): Batch size for processing headlines
        """
        self.processed_data_dir = processed_data_dir
        self.embedding_dims = 20
        self.use_log_transform = use_log_transform
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        
        logging.info(f"Using device: {self.device}")
        logging.info(f"Log transform for CTR: {self.use_log_transform}")
        logging.info(f"Batch size: {self.batch_size}")
        
        # Create output directories
        self.output_dir = 'model_output'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
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
    
    def debug_state(self, stage=""):
        """Log debug information about the current state"""
        memory_usage = None
        try:
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # in MB
        except ImportError:
            pass
            
        logging.info(f"DEBUG [{stage}] - Memory usage: {memory_usage} MB" if memory_usage else f"DEBUG [{stage}]")
        
        # Report GPU memory if using CUDA
        if torch.cuda.is_available():
            try:
                logging.info(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
                logging.info(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB")
            except Exception as e:
                logging.warning(f"Could not report GPU memory: {e}")
    
    def _get_xgboost_params(self, n_estimators=100, learning_rate=0.01, max_depth=4):
        """Get consistent XGBoost parameters"""
        return {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'min_child_weight': 5,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 1.0,
            'reg_lambda': 2.0,
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_jobs': -1
        }
    
    def _filter_features(self, features, selected_features):
        """Helper method to filter features by selected columns"""
        if isinstance(features, pd.DataFrame) and isinstance(selected_features, pd.Index):
            return features[selected_features]
        elif isinstance(features, pd.DataFrame) and isinstance(selected_features, list):
            return features[selected_features]
        else:
            logging.warning(f"Could not filter features. Types: features={type(features)}, selected={type(selected_features)}")
            return features
    
    def _generate_cache_key(self, data_type, suffix=""):
        """Generate a cache key for the given data type and suffix"""
        key = f"{data_type}_{suffix}" if suffix else data_type
        return os.path.join(self.cache_dir, f"{key}.pkl")
    
    def _save_to_cache(self, data, data_type, suffix=""):
        """Save data to cache"""
        cache_path = self._generate_cache_key(data_type, suffix)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logging.info(f"Saved {data_type} data to cache: {cache_path}")
            return True
        except Exception as e:
            logging.error(f"Error saving {data_type} to cache: {e}")
            return False
    
    def _load_from_cache(self, data_type, suffix=""):
        """Load data from cache"""
        cache_path = self._generate_cache_key(data_type, suffix)
        if not os.path.exists(cache_path):
            return None
            
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            logging.info(f"Loaded {data_type} data from cache: {cache_path}")
            return data
        except Exception as e:
            logging.error(f"Error loading {data_type} from cache: {e}")
            return None
    
    def load_data(self, data_type='train'):
        """Load processed headlines for specified split (train, val, or test)"""
        # Try to load from cache first
        cached_data = self._load_from_cache(data_type, "headline_data")
        if cached_data is not None:
            return cached_data
            
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
            
            # Save to cache for future use
            self._save_to_cache(data, data_type, "headline_data")
                
            return data
        except Exception as e:
            logging.error(f"Error loading {data_type} data: {e}")
            return None
    
    def extract_features(self, headlines, data_type=None):
        """
        Extract features from headlines for model training
        
        Args:
            headlines: Array-like of headline strings
            data_type: Optional string ('train', 'val', 'test') for caching
        
        Returns:
            pandas DataFrame of extracted features
        """
        # Try to load from cache if data_type is provided
        if data_type:
            cached_features = self._load_from_cache(data_type, "features")
            if cached_features is not None:
                return cached_features
        
        logging.info(f"Extracting features from {len(headlines)} headlines")
        features_list = []
        
        # Process in smaller batches to avoid memory issues
        for i in range(0, len(headlines), self.batch_size):
            batch = headlines[i:i+self.batch_size]
            logging.info(f"Processing batch {i//self.batch_size + 1}/{(len(headlines)-1)//self.batch_size + 1}")
            batch_features = self._extract_batch_features(batch)
            features_list.extend(batch_features)
            
            # Optionally save intermediate batches for very large datasets
            if data_type and (i + self.batch_size) % (self.batch_size * 10) == 0:
                self._save_to_cache(pd.DataFrame(features_list), data_type, f"features_partial_{i}")
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Save to cache if data_type is provided
        if data_type:
            self._save_to_cache(features_df, data_type, "features")
        
        return features_df
    
    def _extract_batch_features(self, headlines):
        """
        Extract features from a batch of headlines
        
        Args:
            headlines: List of headline strings
            
        Returns:
            List of feature dictionaries
        """
        batch_features = []
        
        # Process headlines in batch
        for headline in headlines:
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
            
            # 5. Feature interactions
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
                
                # Add embedding dimensions as features
                for j in range(self.embedding_dims):
                    features[f'emb_{j}'] = embedding[j]
            except Exception as e:
                logging.error(f"Error extracting embedding for '{headline}': {e}")
                # Add zero embeddings if failed
                for j in range(self.embedding_dims):
                    features[f'emb_{j}'] = 0.0
            
            batch_features.append(features)
        
        return batch_features
    
    def train_model(self, train_features, train_ctr, val_features=None, val_ctr=None,
               output_file='headline_ctr_model.pkl'):
        """
        Train model with simplified approach
        
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
            
        Returns:
        --------
        dict
            Dictionary containing model, selected features, and performance metrics
        """
        logging.info("Training headline CTR prediction model")
        
        # Apply log transform if specified
        if self.use_log_transform:
            train_y = np.log1p(train_ctr)
            val_y = np.log1p(val_ctr) if val_ctr is not None else None
        else:
            train_y = train_ctr
            val_y = val_ctr
        
        # Create base model
        logging.info("Creating XGBoost model...")
        base_params = self._get_xgboost_params(n_estimators=50, learning_rate=0.05, max_depth=5)
        base = XGBRegressor(**base_params)
        
        # Feature selection
        logging.info("Performing feature selection...")
        fs_model = XGBRegressor(n_estimators=50, learning_rate=0.05, max_depth=5, random_state=42)
        fs_model.fit(train_features, train_y)
        selector = SelectFromModel(fs_model, threshold='median', prefit=True)
        mask = selector.get_support()
        sel_feats = train_features.columns[mask]
        logging.info(f"Selected {len(sel_feats)} features.")
        tf_sel = train_features[sel_feats]
        vf_sel = val_features[sel_feats] if val_features is not None else None
        
        # Define final hyperparameters
        final_params = self._get_xgboost_params(n_estimators=200, learning_rate=0.01, max_depth=4)
        logging.info(f"Training final model with params: {final_params}")
        
        # Create final model
        final = XGBRegressor(**final_params)
        
        # Train final model
        start = time.time()
        if vf_sel is not None and val_y is not None:
            final.fit(
                tf_sel, train_y,
                eval_set=[(vf_sel, val_y)],
                eval_metric='rmse',
                early_stopping_rounds=50,
                verbose=0
            )
        else:
            final.fit(tf_sel, train_y, verbose=0)
        ttime = time.time() - start
        logging.info(f"Final model trained in {ttime:.2f}s")
        
        # Evaluate on train/val
        train_pred_t = final.predict(tf_sel)
        train_pred = np.expm1(train_pred_t) if self.use_log_transform else train_pred_t
        train_metrics = {
            'mse': mean_squared_error(train_ctr, train_pred),
            'rmse': np.sqrt(mean_squared_error(train_ctr, train_pred)),
            'mae': mean_absolute_error(train_ctr, train_pred),
            'r2': r2_score(train_ctr, train_pred)
        }
        logging.info(f"Train metrics: {train_metrics}")
        
        val_metrics = {}
        if vf_sel is not None and val_ctr is not None:
            val_pred_t = final.predict(vf_sel)
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
        fi = pd.DataFrame({'feature': sel_feats, 'importance': final.feature_importances_})
        fi.sort_values('importance', ascending=False, inplace=True)
        logging.info(f"Top features: {fi.head(10)}")
        
        # Save model and metrics
        model_data = {
            'model': final,
            'use_log_transform': self.use_log_transform,
            'feature_names': sel_feats.tolist(),
            'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': {'train': train_metrics, 'val': val_metrics}
        }
        
        model_path = os.path.join(self.output_dir, output_file)
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        logging.info(f"Model saved to {model_path}")
        
        # Save feature importance
        self.visualize_feature_importance(fi)
        fi.to_csv(os.path.join(self.output_dir, 'feature_importance.csv'), index=False)
        logging.info("Feature importance saved.")
        
        return {
            'model': final, 
            'selected_features': sel_feats, 
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'feature_importances': fi, 
            'training_time': ttime
        }
    
    def visualize_feature_importance(self, feature_importances, output_file='feature_importance.png'):
        """Create and save feature importance visualization"""
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importances.head(15), palette='viridis')
        plt.title('Top 15 Feature Importances for CTR Prediction', fontsize=14)
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_file), dpi=300)
        plt.close()
        logging.info(f"Feature importance visualization saved to {os.path.join(self.output_dir, output_file)}")
    
    def visualize_ctr_distribution(self, train_ctr, val_ctr=None, output_file='ctr_distribution.png'):
        """Create and save CTR distribution visualization"""
        plt.figure(figsize=(14, 7))
        
        # Plot histogram and KDE for training CTR
        plt.subplot(1, 2, 1)
        sns.histplot(train_ctr, kde=True, bins=50, color='blue', alpha=0.7)
        plt.title('Training CTR Distribution', fontsize=14)
        plt.xlabel('Click-Through Rate', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        
        # Plot transformed CTR if log transform is used
        if self.use_log_transform:
            plt.subplot(1, 2, 2)
            log_ctr = np.log1p(train_ctr)
            sns.histplot(log_ctr, kde=True, bins=50, color='green', alpha=0.7)
            plt.title('Log-Transformed CTR Distribution', fontsize=14)
            plt.xlabel('Log(CTR + 1)', fontsize=12)
            plt.ylabel('Count', fontsize=12)
        elif val_ctr is not None:
            # If no log transform but we have validation data, plot validation CTR
            plt.subplot(1, 2, 2)
            sns.histplot(val_ctr, kde=True, bins=50, color='orange', alpha=0.7)
            plt.title('Validation CTR Distribution', fontsize=14)
            plt.xlabel('Click-Through Rate', fontsize=12)
            plt.ylabel('Count', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_file), dpi=300)
        plt.close()
        logging.info(f"CTR distribution visualization saved to {os.path.join(self.output_dir, output_file)}")
    
    def visualize_predictions(self, true_values, predicted_values, output_file='prediction_scatter.png'):
        """Create and save scatter plot of true vs predicted values"""
        plt.figure(figsize=(10, 8))
        
        # Plot scatter plot with alpha for density visualization
        plt.scatter(true_values, predicted_values, alpha=0.5, color='blue')
        
        # Add diagonal line (perfect predictions)
        max_val = max(np.max(true_values), np.max(predicted_values))
        min_val = min(np.min(true_values), np.min(predicted_values))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title('True vs Predicted CTR Values', fontsize=14)
        plt.xlabel('True CTR', fontsize=12)
        plt.ylabel('Predicted CTR', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_file), dpi=300)
        plt.close()
        logging.info(f"Prediction visualization saved to {os.path.join(self.output_dir, output_file)}")
    
    def run_training_pipeline(self):
        """Run the complete model training pipeline with train/val/test splits"""
        try:
            self.debug_state("Start of training pipeline")
            
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
            
            self.debug_state("After loading data")
            
            # Visualize CTR distribution
            self.visualize_ctr_distribution(train_data['ctr'].values, val_data['ctr'].values)
            
            # Extract features for all splits
            logging.info("Extracting features for training data...")
            train_features = self.extract_features(train_data['title'].values, 'train')
            self.debug_state("After extracting training features")
            
            logging.info("Extracting features for validation data...")
            val_features = self.extract_features(val_data['title'].values, 'val')
            self.debug_state("After extracting validation features")
            
            test_features = None
            if test_data is not None:
                logging.info("Extracting features for test data...")
                test_features = self.extract_features(test_data['title'].values, 'test')
                self.debug_state("After extracting test features")
            
            # Train model using proper splits
            self.debug_state("Before model training")
            result = self.train_model(
                train_features, train_data['ctr'].values,
                val_features, val_data['ctr'].values,
                output_file='headline_ctr_model.pkl'
            )
            self.debug_state("After model training")
            
            if result is None:
                logging.error("Model training failed. Aborting pipeline.")
                return None
            
            # Visualize predictions for validation set
            val_features_selected = val_features[result['selected_features']]
            
            if self.use_log_transform:
                val_pred = np.expm1(result['model'].predict(val_features_selected))
            else:
                val_pred = result['model'].predict(val_features_selected)
                
            self.visualize_predictions(val_data['ctr'].values, val_pred, 'validation_predictions.png')
            
            # If test data is available, generate predictions
            if test_data is not None and test_features is not None:
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
            self.debug_state("End of training pipeline")
            
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
- Training CTR range: {train_data['ctr'].min():.4f} to {train_data['ctr'].max():.4f}
- Training Mean CTR: {train_data['ctr'].mean():.4f}
- Validation Mean CTR: {val_data['ctr'].mean():.4f}

## Key Feature Importances
"""
        
        for i, row in result['feature_importances'].head(15).iterrows():
            report += f"- {row['feature']}: {row['importance']:.4f}\n"
        
        report += """
## Usage Guidelines
This model can be used to predict the expected CTR of news headlines.
It can be integrated into a headline optimization workflow for automated
headline suggestions or ranking.

## Features Used
The model uses both basic text features and semantic embeddings:
- Basic features: length, word count, question marks, numbers, etc.
- Semantic features: BERT embeddings to capture meaning

## Visualizations
The following visualizations have been generated:
- feature_importance.png: Importance of different features
- ctr_distribution.png: Distribution of CTR values
- validation_predictions.png: True vs predicted CTR values
"""
        
        with open(os.path.join(self.output_dir, 'headline_model_report.md'), 'w') as f:
            f.write(report)
        
        logging.info(f"Model report saved to {os.path.join(self.output_dir, 'headline_model_report.md')}")
    
    def save_model(self, model, selected_features, metrics, output_file='headline_ctr_model.pkl'):
        """
        Save the trained model to disk
        
        Args:
            model: Trained model object
            selected_features: List or Index of feature names
            metrics: Dictionary of evaluation metrics
            output_file: File path to save the model
        """
        model_data = {
            'model': model,
            'use_log_transform': self.use_log_transform,
            'feature_names': selected_features.tolist() if hasattr(selected_features, 'tolist') else selected_features,
            'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': metrics
        }
        
        model_path = os.path.join(self.output_dir, output_file)
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Also save a metadata file in JSON for easier inspection
        meta_data = {
            'model_type': str(type(model).__name__),
            'use_log_transform': self.use_log_transform,
            'feature_count': len(model_data['feature_names']),
            'training_date': model_data['training_date'],
            'metrics': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in metrics.items()}
        }
        
        meta_path = os.path.join(self.output_dir, output_file.replace('.pkl', '_meta.json'))
        with open(meta_path, 'w') as f:
            json.dump(meta_data, f, indent=2)
            
        logging.info(f"Model saved to {model_path}")
        logging.info(f"Model metadata saved to {meta_path}")
        
        return model_path
    
    def load_model(self, model_file='headline_ctr_model.pkl'):
        """
        Load a trained model from disk
        
        Args:
            model_file: File path of the saved model
            
        Returns:
            dict: Dictionary containing model and metadata
        """
        model_path = os.path.join(self.output_dir, model_file)
        if not os.path.exists(model_path):
            logging.error(f"Model file not found: {model_path}")
            return None
            
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            logging.info(f"Loaded model from {model_path}")
            logging.info(f"Model was trained on: {model_data['training_date']}")
            
            # Verify model data
            required_keys = ['model', 'feature_names', 'use_log_transform']
            if not all(key in model_data for key in required_keys):
                missing = [key for key in required_keys if key not in model_data]
                logging.error(f"Missing required keys in model data: {missing}")
                return None
                
            return model_data
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return None
    
    def predict_ctr(self, headlines, model_data=None, model_file='headline_ctr_model.pkl'):
        """
        Predict CTR for a list of headlines
        
        Args:
            headlines: List of headline strings
            model_data: Optional preloaded model data
            model_file: Path to the model file if model_data is not provided
            
        Returns:
            numpy array: Predicted CTR values
        """
        if model_data is None:
            model_data = self.load_model(model_file)
            if model_data is None:
                return None
        
        try:
            # Extract features
            features = self.extract_features(headlines)
            
            # Filter to selected features
            feature_names = model_data['feature_names']
            missing_features = [f for f in feature_names if f not in features.columns]
            
            if missing_features:
                logging.warning(f"Missing features in prediction data: {missing_features}")
                # Add missing features with zeros
                for feat in missing_features:
                    features[feat] = 0.0
            
            # Select only the features used by the model
            features_filtered = features[feature_names]
            
            # Make predictions
            model = model_data['model']
            predictions = model.predict(features_filtered)
            
            # Apply inverse transform if log transform was used
            if model_data['use_log_transform']:
                predictions = np.expm1(predictions)
            
            return predictions
        except Exception as e:
            logging.error(f"Error predicting CTR: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return None
    
    def batch_predict_ctr(self, headlines, batch_size=None, model_data=None, model_file='headline_ctr_model.pkl'):
        """
        Predict CTR for a large list of headlines by batching
        
        Args:
            headlines: List of headline strings
            batch_size: Batch size for processing
            model_data: Optional preloaded model data
            model_file: Path to the model file if model_data is not provided
            
        Returns:
            numpy array: Predicted CTR values
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        if model_data is None:
            model_data = self.load_model(model_file)
            if model_data is None:
                return None
                
        all_predictions = []
        
        try:
            for i in range(0, len(headlines), batch_size):
                batch = headlines[i:i+batch_size]
                logging.info(f"Processing prediction batch {i//batch_size + 1}/{(len(headlines)-1)//batch_size + 1}")
                
                batch_predictions = self.predict_ctr(batch, model_data)
                if batch_predictions is not None:
                    all_predictions.extend(batch_predictions)
                else:
                    logging.error(f"Failed to get predictions for batch {i//batch_size + 1}")
                    return None
            
            return np.array(all_predictions)
        except Exception as e:
            logging.error(f"Error in batch prediction: {e}")
            return None
    
    def optimize_headline(self, headline, n_variations=10, model_data=None, model_file='headline_ctr_model.pkl'):
        """
        Generate optimized variations of a headline for better CTR
        
        Args:
            headline: Original headline string
            n_variations: Number of variations to generate
            model_data: Optional preloaded model data
            model_file: Path to the model file if model_data is not provided
            
        Returns:
            pandas DataFrame: Original and optimized headlines with predicted CTR
        """
        if model_data is None:
            model_data = self.load_model(model_file)
            if model_data is None:
                logging.error("Could not load model for headline optimization")
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
        
        # Predict CTR for all variations
        predictions = self.predict_ctr(variations, model_data)
        if predictions is None:
            return None
            
        # Create results dataframe
        results = pd.DataFrame({
            'headline': variations,
            'predicted_ctr': predictions,
            'is_original': [i == 0 for i in range(len(variations))]
        })
        
        # Sort by predicted CTR (descending)
        results = results.sort_values('predicted_ctr', ascending=False).reset_index(drop=True)
        
        return results
    
    def headline_analysis(self, headline, model_data=None, model_file='headline_ctr_model.pkl'):
        """
        Analyze a headline and explain its predicted CTR
        
        Args:
            headline: Headline string to analyze
            model_data: Optional preloaded model data
            model_file: Path to the model file if model_data is not provided
            
        Returns:
            dict: Analysis results including predicted CTR and feature breakdown
        """
        if model_data is None:
            model_data = self.load_model(model_file)
            if model_data is None:
                return None
                
        # Extract features for the headline
        features_df = self.extract_features([headline])
        
        # Get selected features and ensure they exist
        feature_names = model_data['feature_names']
        for feat in feature_names:
            if feat not in features_df.columns:
                features_df[feat] = 0.0
                
        # Select only model features
        features_filtered = features_df[feature_names]
        
        # Get model prediction
        model = model_data['model']
        prediction = model.predict(features_filtered)[0]
        
        # Apply inverse transform if needed
        if model_data['use_log_transform']:
            prediction = np.expm1(prediction)
            
        # Get feature importances from the model
        try:
            importances = model.feature_importances_
            # Calculate feature contributions
            contributions = []
            for i, feat in enumerate(feature_names):
                value = features_filtered[feat].values[0]
                importance = importances[i]
                contributions.append({
                    'feature': feat,
                    'value': value,
                    'importance': importance,
                    'contribution': abs(value * importance)
                })
                
            # Sort by contribution
            contributions = sorted(contributions, key=lambda x: x['contribution'], reverse=True)
        except Exception as e:
            logging.warning(f"Could not calculate feature contributions: {e}")
            contributions = []
            
        # Prepare analysis results
        analysis = {
            'headline': headline,
            'predicted_ctr': prediction,
            'features': features_filtered.to_dict(orient='records')[0],
            'top_contributing_features': contributions[:10],
            'headline_stats': {
                'length': len(headline),
                'word_count': len(headline.split()),
                'is_question': '?' in headline,
                'has_numbers': bool(re.search(r'\d', headline)),
                'clickbait_score': self._calculate_clickbait_score(headline)
            }
        }
        
        return analysis
        
    def _calculate_clickbait_score(self, headline):
        """Calculate a simple clickbait score for a headline"""
        score = 0
        
        # Check for common clickbait patterns
        if any(word in headline.lower() for word in ['shocking', 'amazing', 'unbelievable', 'surprise']):
            score += 1
        if bool(re.search(r'\b(how|what|why|when|where|who|which)\b', headline.lower())):
            score += 1
        if bool(re.search(r'\b(you won\'t believe|mind blown|jaw-dropping)\b', headline.lower())):
            score += 2
        if bool(re.search(r'\b(secret|reveal|shock|stun|surprise)\b', headline.lower())):
            score += 1
        if bool(re.search(r'\b(breaking|urgent|just in|now|today)\b', headline.lower())):
            score += 1
        if bool(re.search(r'\b(\d+ (?:things|ways|tips|reasons|facts))\b', headline.lower())):
            score += 2
        if headline.endswith('?'):
            score += 1
        if headline.endswith('!'):
            score += 1
        if ':' in headline:
            score += 0.5
            
        # Normalize to 0-10 scale
        return min(score * 1.0, 10.0)

def main():
    """Main function to run the headline model training pipeline"""
    
    # Set up argument parser
    import argparse
    parser = argparse.ArgumentParser(description='Train headline CTR prediction model')
    parser.add_argument('--data_dir', type=str, default='agentic_news_editor/processed_data', 
                        help='Directory containing processed data files')
    parser.add_argument('--output_dir', type=str, default='model_output',
                        help='Directory to save model output')
    parser.add_argument('--cache_dir', type=str, default='model_cache',
                        help='Directory to cache intermediate results')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Batch size for processing headlines')
    parser.add_argument('--log_transform', action='store_true',
                        help='Apply log transform to CTR values')
    parser.add_argument('--predict', type=str, default=None,
                        help='Path to a CSV file containing headlines for prediction')
    parser.add_argument('--analyze', type=str, default=None,
                        help='Analyze a specific headline')
    parser.add_argument('--optimize', type=str, default=None,
                        help='Optimize a specific headline')
    parser.add_argument('--n_variations', type=int, default=10,
                        help='Number of headline variations to generate when optimizing')
    parser.add_argument('--model_file', type=str, default='headline_ctr_model.pkl',
                        help='Model file to use for prediction/analysis/optimization')
    
    args = parser.parse_args()
    
    # Create trainer instance
    trainer = HeadlineModelTrainer(
        processed_data_dir=args.data_dir,
        use_log_transform=args.log_transform,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size
    )
    
    # Check if we're running prediction
    if args.predict:
        if not os.path.exists(args.predict):
            logging.error(f"Prediction file not found: {args.predict}")
            return
            
        logging.info(f"Loading headlines from {args.predict} for prediction")
        try:
            data = pd.read_csv(args.predict)
            if 'title' not in data.columns and 'headline' not in data.columns:
                logging.error("Prediction file must contain 'title' or 'headline' column")
                return
                
            headline_col = 'title' if 'title' in data.columns else 'headline'
            headlines = data[headline_col].values
            
            logging.info(f"Making predictions for {len(headlines)} headlines")
            model_data = trainer.load_model(args.model_file)
            if model_data is None:
                return
                
            predictions = trainer.batch_predict_ctr(headlines, model_data=model_data)
            
            # Save predictions
            data['predicted_ctr'] = predictions
            output_file = os.path.join(args.output_dir, 'headline_predictions.csv')
            data.to_csv(output_file, index=False)
            logging.info(f"Predictions saved to {output_file}")
            
        except Exception as e:
            logging.error(f"Error making predictions: {e}")
            import traceback
            logging.error(traceback.format_exc())
            
    # Check if we're analyzing a headline
    elif args.analyze:
        logging.info(f"Analyzing headline: {args.analyze}")
        analysis = trainer.headline_analysis(args.analyze, model_file=args.model_file)
        if analysis:
            print(f"\nHeadline Analysis for: '{args.analyze}'")
            print(f"Predicted CTR: {analysis['predicted_ctr']:.6f}")
            print(f"Headline Stats: {json.dumps(analysis['headline_stats'], indent=2)}")
            print("\nTop Contributing Features:")
            for feature in analysis['top_contributing_features'][:5]:
                print(f"- {feature['feature']}: {feature['contribution']:.4f}")
    
    # Check if we're optimizing a headline
    elif args.optimize:
        logging.info(f"Optimizing headline: {args.optimize}")
        results = trainer.optimize_headline(
            args.optimize, 
            n_variations=args.n_variations,
            model_file=args.model_file
        )
        if results is not None:
            print(f"\nHeadline Optimization Results for: '{args.optimize}'")
            print(f"Original CTR: {results[results['is_original']]['predicted_ctr'].iloc[0]:.6f}")
            print("\nTop 5 Optimized Headlines:")
            for i, row in results.head(5).iterrows():
                print(f"{i+1}. '{row['headline']}' (CTR: {row['predicted_ctr']:.6f})")
    
    # Otherwise, run the full training pipeline
    else:
        logging.info("Running full training pipeline")
        result = trainer.run_training_pipeline()
        
        if result is not None:
            print(f"Model training complete.")
            print(f"Training R-squared: {result['train_metrics']['r2']:.4f}")
            print(f"Validation R-squared: {result['val_metrics']['r2']:.4f}")
            print(f"Results saved to {trainer.output_dir}")
        else:
            print("Model training failed.")


if __name__ == "__main__":
    main()

# # Train a new model
# python headline_model.py --data_dir path/to/data --log_transform

# # Make predictions with an existing model
# python headline_model.py --predict headlines.csv --model_file my_model.pkl

# # Analyze a specific headline
# python headline_model.py --analyze "This is a headline to analyze"

# # Optimize a headline
# python headline_model.py --optimize "Original headline to improve" --n_variations 15