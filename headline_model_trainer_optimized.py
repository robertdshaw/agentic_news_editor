# CTR Prediction Model for MIND Dataset

#Libraries
import pandas as pd
import numpy as np
import os
import logging
import json
import pickle
from datetime import datetime
from pathlib import Path

# ML Imports
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost.callback import EarlyStopping
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (roc_auc_score, precision_recall_curve, roc_curve, 
                           classification_report, confusion_matrix, precision_score, 
                           recall_score, f1_score)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer

# NLP Imports  
from sentence_transformers import SentenceTransformer
from textstat import flesch_reading_ease

# Viz Imports
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CTRPredictor:
    """Production CTR prediction model for news headlines"""
    
    # Default thresholds for evaluation
    DEFAULT_THRESHOLDS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
   
    def __init__(self,
                 processed_data_dir='agentic_news_editor/processed_data',
                 output_dir='model_output',
                 thresholds=None):
        base = Path(__file__).parent.resolve()
        self.processed_data_dir = (base / processed_data_dir).resolve()
        self.output_dir         = (base / output_dir).resolve()
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Model components
        self.models = {}
        self.encoders = {}
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.feature_names = None
        self.best_model = None
        
        # Configuration
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS
        
        # Initialize embeddings model
        self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Performance tracking
        self.results = {}
    
    def load_processed_data(self):
        """Load data processed by your preprocessing pipeline"""
        # Load train/val/test splits
        files = {
            'train': self.processed_data_dir / 'train_headline_ctr.csv',
            'val': self.processed_data_dir / 'val_headline_ctr.csv', 
            'test': self.processed_data_dir / 'test_headline_ctr.csv'
        }
        
        data = {}
        for split, file_path in files.items():
            if file_path.exists():
                df = pd.read_csv(file_path)
                # Create binary click column if needed
                if 'clicked' not in df.columns and 'ctr' in df.columns:
                    df['clicked'] = (df['ctr'] > 0).astype(int)
                data[split] = df
                logger.info(f"Loaded {split}: {len(df)} rows")
            else:
                logger.warning(f"File not found: {file_path}")
        
        return data.get('train'), data.get('val'), data.get('test')
    
    def check_class_distribution(self):
        """Check class distribution in train/val/test splits"""
        files = {
            'train': self.processed_data_dir / 'train_headline_ctr.csv',
            'val': self.processed_data_dir / 'val_headline_ctr.csv', 
            'test': self.processed_data_dir / 'test_headline_ctr.csv'
        }
        
        for split, file_path in files.items():
            if file_path.exists():
                df = pd.read_csv(file_path)
                
                # Create binary click column if needed (same logic as in load_processed_data)
                if 'clicked' not in df.columns and 'ctr' in df.columns:
                    df['clicked'] = (df['ctr'] > 0).astype(int)
                
                if 'clicked' in df.columns:
                    class_dist = df['clicked'].value_counts()
                    class_pct = df['clicked'].value_counts(normalize=True) * 100
                    print(f"{split} class distribution:")
                    print(f"  Class 0: {class_dist.get(0, 0)} ({class_pct.get(0, 0):.1f}%)")
                    print(f"  Class 1: {class_dist.get(1, 0)} ({class_pct.get(1, 0):.1f}%)")
                else:
                    print(f"{split}: No 'clicked' or 'ctr' column found")
    
    def temporal_stratified_split(self, df, target_col='clicked', split_percentages=(0.6, 0.2, 0.2)):
        """Create temporal stratified split ensuring chronological order and balanced classes"""
        train_pct, val_pct, test_pct = split_percentages
        
        # Create clicked column if needed
        if 'clicked' not in df.columns and 'ctr' in df.columns:
            df['clicked'] = (df['ctr'] > 0).astype(int)
        
        # Check if we have positive samples
        if df['clicked'].sum() == 0:
            print("WARNING: No positive samples found in the data!")
            print("Creating temporal split without stratification...")
        
        # Sort by time if available
        time_col = None
        for col in ['time', 'first_seen', 'timestamp']:
            if col in df.columns:
                time_col = col
                break
        
        if time_col:
            df = df.sort_values(time_col)
            logger.info(f"Sorted data by {time_col}")
        else:
            logger.warning("No time column found, using index order")
        
        # Calculate split indices for temporal split
        n_total = len(df)
        n_train = int(n_total * train_pct)
        n_val = int(n_total * val_pct)
        
        # Create initial temporal splits
        train_data = df.iloc[:n_train].copy()
        val_data = df.iloc[n_train:n_train + n_val].copy()
        test_data = df.iloc[n_train + n_val:].copy()
        
        # Check if each split has positive samples
        train_pos = (train_data['clicked'] == 1).sum()
        val_pos = (val_data['clicked'] == 1).sum()
        test_pos = (test_data['clicked'] == 1).sum()
        
        print(f"Initial temporal split click distribution:")
        print(f"  Train: {train_pos} clicks (out of {len(train_data)})")
        print(f"  Val: {val_pos} clicks (out of {len(val_data)})")
        print(f"  Test: {test_pos} clicks (out of {len(test_data)})")
        
        # If validation or test has no positive samples, use stratified approach instead
        if val_pos == 0 or test_pos == 0:
            print("\nWARNING: Temporal split resulted in zero positive samples in val or test!")
            print("Falling back to stratified split to ensure balanced class distribution...")
            return self.random_stratified_split(df, target_col, split_percentages)
        
        # Reset indices
        train_data.reset_index(drop=True, inplace=True)
        val_data.reset_index(drop=True, inplace=True)
        test_data.reset_index(drop=True, inplace=True)
        
        print(f"\nFinal temporal split:")
        print(f"Train: {len(train_data)} ({len(train_data)/len(df):.1%})")
        print(f"Val: {len(val_data)} ({len(val_data)/len(df):.1%})") 
        print(f"Test: {len(test_data)} ({len(test_data)/len(df):.1%})")
        
        # Report class distribution
        for split_name, split_data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
            if target_col in split_data.columns:
                pos_count = split_data[target_col].sum()
                neg_count = len(split_data) - pos_count
                pos_rate = pos_count / len(split_data) * 100
                print(f"  {split_name}: {pos_count} positive ({pos_rate:.1f}%), {neg_count} negative")
        
        return train_data, val_data, test_data
    
    def random_stratified_split(self, df, target_col='clicked', split_percentages=(0.6, 0.2, 0.2)):
        """Random stratified split ensuring each split has positive and negative samples"""
        train_pct, val_pct, test_pct = split_percentages
        
        # Create clicked column if needed
        if 'clicked' not in df.columns and 'ctr' in df.columns:
            df['clicked'] = (df['ctr'] > 0).astype(int)
        
        # Separate positive and negative samples
        positive_samples = df[df[target_col] == 1]
        negative_samples = df[df[target_col] == 0]
        
        print(f"Total positive samples: {len(positive_samples)}")
        print(f"Total negative samples: {len(negative_samples)}")
        
        # Check if we have enough positive samples
        min_val_positive = max(1, int(len(positive_samples) * val_pct))
        min_test_positive = max(1, int(len(positive_samples) * test_pct))
        
        if len(positive_samples) < 3:
            print(f"WARNING: Only {len(positive_samples)} positive samples available!")
            print("This may not be enough for proper train/val/test split.")
            if len(positive_samples) == 2:
                print("Using 1 positive sample each for train and val, leaving test without positive samples.")
                min_test_positive = 0
            elif len(positive_samples) == 1:
                print("Using 1 positive sample for train, leaving val and test without positive samples.")
                min_val_positive = 0
                min_test_positive = 0
        
        # Split positive samples with minimum guarantees
        pos_train_size = max(len(positive_samples) - min_val_positive - min_test_positive, 1)
        pos_val_size = min_val_positive
        
        pos_train = positive_samples.sample(n=pos_train_size, random_state=42)
        remaining_pos = positive_samples.drop(pos_train.index)
        
        if len(remaining_pos) >= pos_val_size and pos_val_size > 0:
            pos_val = remaining_pos.sample(n=pos_val_size, random_state=42)
            pos_test = remaining_pos.drop(pos_val.index)
        else:
            pos_val = remaining_pos.copy() if pos_val_size > 0 else pd.DataFrame()
            pos_test = pd.DataFrame()
        
        # Split negative samples proportionally
        neg_train_size = int(len(negative_samples) * train_pct)
        neg_val_size = int(len(negative_samples) * val_pct)
        
        neg_train = negative_samples.sample(n=neg_train_size, random_state=42)
        remaining_neg = negative_samples.drop(neg_train.index)
        neg_val = remaining_neg.sample(n=min(neg_val_size, len(remaining_neg)), random_state=42)
        neg_test = remaining_neg.drop(neg_val.index)
        
        # Combine positive and negative samples
        train_data = pd.concat([pos_train, neg_train]).sample(frac=1, random_state=42).reset_index(drop=True)
        val_data = pd.concat([pos_val, neg_val]).sample(frac=1, random_state=42).reset_index(drop=True)
        test_data = pd.concat([pos_test, neg_test]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\nStratified random split:")
        print(f"Train: {len(train_data)} ({len(train_data)/len(df):.1%})")
        print(f"Val: {len(val_data)} ({len(val_data)/len(df):.1%})") 
        print(f"Test: {len(test_data)} ({len(test_data)/len(df):.1%})")
        
        # Report class distribution
        for split_name, split_data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
            pos_count = split_data[target_col].sum()
            neg_count = len(split_data) - pos_count
            pos_rate = pos_count / len(split_data) * 100 if len(split_data) > 0 else 0
            print(f"  {split_name}: {pos_count} positive ({pos_rate:.1f}%), {neg_count} negative")
        
        return train_data, val_data, test_data
    
    def load_data_with_temporal_split(self, split_percentages=(0.6, 0.2, 0.2)):
        """Load data and create temporal splits with custom percentages"""
        # Check if splits already exist
        train_path = self.processed_data_dir / 'train_headline_ctr.csv'
        
        if train_path.exists():
            # Data already split, load normally
            return self.load_processed_data()
        else:
            # Need to create temporal splits
            combined_path = self.processed_data_dir / 'combined_headline_ctr.csv'
            if combined_path.exists():
                combined_data = pd.read_csv(combined_path)
                return self.temporal_stratified_split(combined_data, 
                                                split_percentages=split_percentages)
            else:
                logger.error("No data files found")
                return None, None, None
    
    def extract_features(self, df):
        """Extract features from headlines and metadata"""
        features = pd.DataFrame(index=df.index)
        
        # Text features from title
        features['title_length'] = df['title'].str.len()
        features['title_word_count'] = df['title'].str.split().str.len()
        
        # Pattern features based on EDA findings
        features['has_question'] = df['title'].str.contains(r'\?').astype(int)
        features['has_number'] = df['title'].str.contains(r'\d').astype(int)
        features['has_colon'] = df['title'].str.contains(':').astype(int)
        features['has_quotes'] = df['title'].str.contains(r'["\']').astype(int)
        features['has_ellipsis'] = df['title'].str.contains(r'\.\.\.').astype(int)
        
        # Problematic patterns (from EDA)
        features['starts_with_number'] = df['title'].str.match(r'^\d').astype(int)
        features['starts_with_question'] = df['title'].str.match(r'^(Is|What|How|Why|When|Where)').astype(int)
        features['has_superlative'] = df['title'].str.contains(r'\b(?:best|worst|most|least|biggest|smallest)\b', case=False).astype(int)
        features['title_uppercase_ratio'] = df['title'].apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0)
        
        # Reading ease with proper error handling
        if 'title_reading_ease' not in df.columns:
            def safe_flesch_reading_ease(text):
                try:
                    if pd.isna(text) or not isinstance(text, str) or len(text) == 0:
                        return 0
                    return flesch_reading_ease(text)
                except:
                    return 0
            
            features['title_reading_ease'] = df['title'].apply(safe_flesch_reading_ease)
        else:
            features['title_reading_ease'] = df['title_reading_ease'].fillna(0)
        
        # Category encoding with proper NaN handling
        if 'category' in df.columns:
            if 'category' not in self.encoders:
                self.encoders['category'] = LabelEncoder()
                features['category_encoded'] = self.encoders['category'].fit_transform(df['category'].fillna('unknown'))
            else:
                # Handle unseen categories
                try:
                    features['category_encoded'] = self.encoders['category'].transform(df['category'].fillna('unknown'))
                except ValueError:
                    # If new categories exist, fit_transform again
                    all_categories = list(self.encoders['category'].classes_) + list(df['category'].fillna('unknown').unique())
                    self.encoders['category'].fit(all_categories)
                    features['category_encoded'] = self.encoders['category'].transform(df['category'].fillna('unknown'))
        
        # Time features with proper error handling
        if 'time' in df.columns or 'first_seen' in df.columns:
            time_col = 'time' if 'time' in df.columns else 'first_seen'
            try:
                df[time_col] = pd.to_datetime(df[time_col])
                features['hour'] = df[time_col].dt.hour
                features['day_of_week'] = df[time_col].dt.dayofweek
                features['is_weekend'] = (df[time_col].dt.dayofweek >= 5).astype(int)
            except:
                # If time parsing fails, use default values
                features['hour'] = 12  # Default to noon
                features['day_of_week'] = 1  # Default to Monday
                features['is_weekend'] = 0  # Default to weekday
        
        # Embeddings (using cache) with better error handling
        cache_file = self.output_dir / f'embeddings_{len(df)}_{hash(str(df.index.tolist()))}.pkl'
        if cache_file.exists():
            logger.info("Loading cached embeddings")
            try:
                embeddings = pickle.load(open(cache_file, 'rb'))
            except:
                logger.warning("Failed to load cached embeddings, creating new ones")
                embeddings = self._create_embeddings(df['title'])
                pickle.dump(embeddings, open(cache_file, 'wb'))
        else:
            logger.info("Creating embeddings...")
            embeddings = self._create_embeddings(df['title'])
            pickle.dump(embeddings, open(cache_file, 'wb'))
        
        # Add first 50 embedding dimensions as features
        for i in range(min(50, embeddings.shape[1])):
            features[f'emb_{i}'] = embeddings[:, i]
        
        # Fill any remaining NaN values
        features = features.fillna(0)
        
        return features
    
    def _create_embeddings(self, titles):
        """Create embeddings with proper error handling"""
        try:
            # Clean titles before embedding
            titles_clean = titles.fillna('').astype(str).tolist()
            embeddings = self.embeddings_model.encode(titles_clean, batch_size=32)
            return embeddings
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            # Return zero embeddings if failed
            return np.zeros((len(titles), 384))
    
    def prepare_data(self, X, y, fit=True):
        """Prepare features and handle class imbalance with NaN handling"""
        # Check for and handle NaN values before scaling
        logger.info(f"Checking for NaN values. NaN count: {X.isna().sum().sum()}")
        
        # Fill any remaining NaN values
        X = X.fillna(0)
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            # Also fit imputer as backup
            self.imputer.fit(X)
            self.feature_names = X.columns.tolist()
        else:
            X_scaled = self.scaler.transform(X)
            # Apply imputer if still NaN values exist
            if np.isnan(X_scaled).any():
                X_scaled = self.imputer.transform(X_scaled)
        
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Final check for NaN values
        if X_scaled.isna().any().any():
            logger.warning("NaN values detected after scaling, filling with 0")
            X_scaled = X_scaled.fillna(0)
        
        # Handle class imbalance with SMOTE
        if fit and y.mean() < 0.1:  # If very imbalanced
            logger.info(f"Applying SMOTE. Original distribution: {y.value_counts().to_dict()}")
            
            # Double-check for NaN values before SMOTE
            if X_scaled.isna().any().any() or pd.isna(y).any():
                logger.error("NaN values detected before SMOTE")
                X_scaled = X_scaled.fillna(0)
                y = y.fillna(0)
            
            try:
                # Apply SMOTE followed by undersampling
                smote = SMOTE(sampling_strategy=0.2, random_state=42)
                X_smote, y_smote = smote.fit_resample(X_scaled, y)
                
                under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
                X_final, y_final = under.fit_resample(X_smote, y_smote)
                
                logger.info(f"After SMOTE+Undersampling: {pd.Series(y_final).value_counts().to_dict()}")
                return X_final, y_final
            except Exception as e:
                logger.error(f"SMOTE failed: {e}")
                logger.info("Proceeding without SMOTE")
                return X_scaled, y
        
        return X_scaled, y

    def train_models(self, X_train, y_train, X_val=None, y_val=None):
        """Train multiple models and select the best"""
        # Model configurations
        model_configs = {
            'xgboost': {
                'model': xgb.XGBClassifier,
                'params': {
                    'n_estimators': 1000,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'n_jobs': -1,
                    'eval_metric': 'auc'
                }
            },
            'lightgbm': {
                'model': lgb.LGBMClassifier,
                'params': {
                    'n_estimators': 1000,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbose': -1
                }
            },
            'catboost': {
                'model': CatBoostClassifier,
                'params': {
                    'iterations': 1000,
                    'depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'verbose': False
                }
            }
        }
        
        # Handle class imbalance in model parameters
        if len(y_train[y_train == 1]) > 0:  # Ensure positive class exists
            imbalance_ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
            model_configs['xgboost']['params']['scale_pos_weight'] = imbalance_ratio
            model_configs['lightgbm']['params']['class_weight'] = 'balanced'
        
        # Train models
        best_auc = 0
        for name, config in model_configs.items():
            logger.info(f"Training {name}...")
            
            # Create eval sets for early stopping
            if X_val is not None and y_val is not None:
                eval_set = [(X_train, y_train), (X_val, y_val)]
                early_stopping_rounds = 50
            else:
                eval_set = [(X_train, y_train)]
                early_stopping_rounds = None
            
            if name == 'xgboost':
                xgb_params = config['params'].copy()
                xgb_params['verbosity'] = 0
                
                if early_stopping_rounds:
                    xgb_params['callbacks'] = [
                        EarlyStopping(rounds=early_stopping_rounds, save_best=True, maximize=True)
                    ]
                model = config['model'](**xgb_params)
            else:
                model = config['model'](**config['params'])
            
            # Fit model
            try:
                if name == 'xgboost':
                    model.fit(
                        X_train, 
                        y_train, 
                        eval_set=eval_set,
                        verbose=False
                    )
                elif name == 'lightgbm':
                    if early_stopping_rounds:
                        model.fit(X_train, y_train,
                                 eval_set=eval_set,
                                 callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
                    else:
                        model.fit(X_train, y_train, eval_set=eval_set)
                elif name == 'catboost':
                    if early_stopping_rounds:
                        model.fit(X_train, y_train,
                                 eval_set=eval_set[1],  # CatBoost wants single eval set
                                 early_stopping_rounds=early_stopping_rounds,
                                 verbose=False)
                    else:
                        model.fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train)
                    
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue
            
            # Store model
            self.models[name] = model
            
            # Evaluate on validation (dynamic threshold via PR curve)
            if X_val is not None:
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_pred_proba)

                # Get precision & recall at every possible threshold
                precisions, recalls, pr_thresholds = precision_recall_curve(y_val, y_pred_proba)

                # Compute F1 at each threshold
                f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)

                # Find the best threshold
                best_idx = np.argmax(f1_scores)
                best_threshold = pr_thresholds[best_idx]
                best_f1 = f1_scores[best_idx]
                best_prec = precisions[best_idx]
                best_rec = recalls[best_idx]

                # Store results
                self.results[name] = {
                    'model': model,
                    'auc': auc,
                    'predictions': y_pred_proba,
                    'best_threshold': best_threshold,
                    'best_f1': best_f1,
                    'best_precision': best_prec,
                    'best_recall': best_rec,
                    'all_thresholds': list(zip(pr_thresholds,
                                            precisions[:-1],
                                            recalls[:-1],
                                            f1_scores))
                }

                logger.info(f"{name} – val_auc: {auc:.4f}; "
                            f"best_thresh: {best_threshold:.4f}, F1: {best_f1:.4f}")

                # Update overall best by AUC
                if auc > best_auc:
                    best_auc = auc
                    self.best_model = name
            else:
                # No validation => only AUC info
                self.results[name] = {'model': model, 'auc': None}

        # After looping over all models
        if not self.best_model and self.results:
            self.best_model = next(iter(self.results))
            logger.info(f"No val set: default best_model = {self.best_model}")

        if self.best_model:
            logger.info(f"Selected best_model = {self.best_model} (AUC: {best_auc:.4f})")
        
        return self.results

    def evaluate_on_test(self, X_test, y_test):
        """Evaluate best model on test set"""
        if not self.best_model:
            logger.error("No trained models available")
            return None
        
        model = self.results[self.best_model]['model']
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate AUC once (fixes repeated calculation bug)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Evaluate different thresholds
        thresholds = self.thresholds
        results = {}
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # Calculate metrics
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            results[threshold] = {
                'auc': auc,  # AUC is the same for all thresholds
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predictions': y_pred,
                'probabilities': y_pred_proba.tolist()
            }
            
            logger.info(f"Threshold {threshold}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        
        # Generate detailed report for best threshold (based on F1)
        best_threshold = max(results.keys(), key=lambda x: results[x]['f1'])
        best_results = results[best_threshold]
        
        detailed_report = {
            'auc': auc,
            'best_threshold': best_threshold,
            'classification_report': classification_report(y_test, best_results['predictions'], output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, best_results['predictions']).tolist(),
            'all_thresholds': results
        }
        
        logger.info(f"Best threshold: {best_threshold}")
        logger.info(f"Test AUC: {auc:.4f}")
        logger.info(f"Best F1: {best_results['f1']:.4f}")
        
        cm = confusion_matrix(y_test, best_results['predictions'])
        print(f"\nConfusion Matrix (threshold={best_threshold}):")
        print(f"True Negative: {cm[0,0]}, False Positive: {cm[0,1]}")
        print(f"False Negative: {cm[1,0]}, True Positive: {cm[1,1]}")
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Click', 'Click'], 
                    yticklabels=['No Click', 'Click'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix (Threshold = {best_threshold})')
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        detailed_report_json = {}
        for key, value in detailed_report.items():
            if key == 'confusion_matrix':
                detailed_report_json[key] = [[int(x) for x in row] for row in value]
            elif key == 'all_thresholds':
                detailed_report_json[key] = {}
                for thresh, thresh_results in value.items():
                    detailed_report_json[key][str(thresh)] = {
                        k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in thresh_results.items()
                    }
            else:
                detailed_report_json[key] = value
        
        with open(self.output_dir / 'test_results.json', 'w') as f:
            json.dump(detailed_report_json, f, indent=2)
        
        return detailed_report
    
    def save_model(self):
        """Save the trained model and components"""
        if not self.best_model:
            logger.error("No trained models to save")
            return
        
        model_data = {
            'model': self.results[self.best_model]['model'],
            'model_name': self.best_model,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'encoders': self.encoders,
            'feature_names': self.feature_names,
            'training_date': datetime.now().isoformat(),
            'thresholds': self.thresholds,
            'best_threshold': self.results[self.best_model].get('best_threshold')
        }
        
        with open(self.output_dir / 'ctr_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {self.output_dir / 'ctr_model.pkl'}")
    
    def load_model(self, model_path=None):
        """Load a previously trained model"""
        if model_path is None:
            model_path = self.output_dir / 'ctr_model.pkl'
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.best_model = model_data['model_name']
        self.results[self.best_model] = {'model': model_data['model']}
        self.scaler = model_data['scaler']
        self.imputer = model_data.get('imputer', self.imputer)
        self.encoders = model_data['encoders']
        self.feature_names = model_data['feature_names']
        self.thresholds = model_data.get('thresholds', self.DEFAULT_THRESHOLDS)
        
        # Load best threshold if available
        if 'best_threshold' in model_data and model_data['best_threshold']:
            self.results[self.best_model]['best_threshold'] = model_data['best_threshold']
        
        logger.info(f"Model loaded from {model_path}")
    
    def predict_click_probability(self, headlines, use_best_threshold=True):
        """Predict click probability for new headlines"""
        if not self.best_model:
            logger.error("No trained model available")
            return None
        
        # Create DataFrame with headlines
        df = pd.DataFrame({'title': headlines})
        
        # Extract features
        features = self.extract_features(df)
        
        # Ensure all features are present
        for feat in self.feature_names:
            if feat not in features.columns:
                features[feat] = 0
        
        # Prepare data
        X = features[self.feature_names]
        X_scaled = self.scaler.transform(X)
        
        # Handle any NaN values
        if np.isnan(X_scaled).any():
            X_scaled = self.imputer.transform(X_scaled)
        
        # Predict
        model = self.results[self.best_model]['model']
        probabilities = model.predict_proba(X_scaled)[:, 1]
        
        # Calculate quartiles for relative ranking
        q25 = np.percentile(probabilities, 25)
        q50 = np.percentile(probabilities, 50)
        q75 = np.percentile(probabilities, 75)
        
        results = []
        for headline, prob in zip(headlines, probabilities):
            # Better relative ranking based on quartiles
            if prob >= q75:
                relative_rank = "Top 25%"
            elif prob >= q50:
                relative_rank = "Above Median"
            elif prob >= q25:
                relative_rank = "Below Median"
            else:
                relative_rank = "Bottom 25%"

            results.append({
                'headline': headline,
                'click_probability': float(prob),
                'relative_rank': relative_rank,
                'percentile': int(100 - (np.searchsorted(np.sort(probabilities), prob) / len(probabilities) * 100)),
            })
        if use_best_threshold and self.best_model in self.results and 'best_threshold' in self.results[self.best_model]:
            best_val_thresh = self.results[self.best_model]['best_threshold']
            results = [r for r in results if r['click_probability'] >= best_val_thresh]
        results.sort(key=lambda x: x['click_probability'], reverse=True)
        for i, result in enumerate(results):
            result['rank'] = i + 1
        return results
    
    def train_pipeline(self):
        """Run the complete training pipeline"""
        logger.info("Starting CTR prediction training pipeline...")
        
        # Load data
        train_data, val_data, test_data = self.load_processed_data()
        
        if train_data is None:
            logger.error("No training data found")
            return
        
        # Extract features
        logger.info("Extracting features...")
        X_train = self.extract_features(train_data)
        y_train = train_data['clicked']
        
        X_val, y_val = None, None
        if val_data is not None:
            X_val = self.extract_features(val_data)
            y_val = val_data['clicked']
        
        # Prepare data
        logger.info("Preparing data...")
        X_train_prep, y_train_prep = self.prepare_data(X_train, y_train, fit=True)
        if X_val is not None:
            X_val_prep, _ = self.prepare_data(X_val, y_val, fit=False)
        else:
            X_val_prep = None
        
        # Train models
        logger.info("Training models...")
        self.train_models(X_train_prep, y_train_prep, X_val_prep, y_val)
        
        # Evaluate on test data
        if test_data is not None:
            logger.info("Evaluating on test set...")
            X_test = self.extract_features(test_data)
            X_test_prep, _ = self.prepare_data(X_test, test_data['clicked'], fit=False)
            self.evaluate_on_test(X_test_prep, test_data['clicked'])
        
        self.save_model()
        
        logger.info(f"Saving visualizations into {self.output_dir/'visualizations'} …")
        self.create_visualizations()
        logger.info("Training pipeline completed!")
        return self.results
    
    def recreate_splits_with_proportions(self, split_percentages=(0.6, 0.2, 0.2)):
        """Recreate train/val/test splits with new proportions"""
        
        # Load all existing data
        train_path = self.processed_data_dir / 'train_headline_ctr.csv'
        val_path = self.processed_data_dir / 'val_headline_ctr.csv'
        test_path = self.processed_data_dir / 'test_headline_ctr.csv'
        combined_path = self.processed_data_dir / 'combined_headline_ctr.csv'
        combined_data = []
        if combined_path.exists():
            combined_data = pd.read_csv(combined_path)
            logger.info(f"Loaded combined data: {len(combined_data)} rows")
        else:
            # Combine all split files
            for path in [train_path, val_path, test_path]:
                if path.exists():
                    df = pd.read_csv(path)
                    combined_data.append(df)
            
            if not combined_data:
                logger.error("No existing split files found")
                return None, None, None
            
            # Combine all data
            combined_data = pd.concat(combined_data, ignore_index=True)
            logger.info(f"Combined existing splits: {len(combined_data)} rows")
        
        # Create new temporal splits
        train_data, val_data, test_data = self.temporal_stratified_split(
            combined_data, split_percentages=split_percentages
        )
        train_data.to_csv(self.processed_data_dir / 'train_headline_ctr.csv', index=False)
        val_data.to_csv(self.processed_data_dir / 'val_headline_ctr.csv', index=False)
        test_data.to_csv(self.processed_data_dir / 'test_headline_ctr.csv', index=False)
        logger.info("New splits saved successfully")
        return train_data, val_data, test_data
    
    def create_visualizations(self):
        """Create comprehensive visualizations for model analysis"""
        if not self.best_model:
            logger.error("No trained model available for visualization")
            return
        viz_dir = self.output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        _, val_data, _ = self.load_processed_data()
        if val_data is None:
            logger.warning("No validation data available for visualization")
            return
        try:
            model = self.results[self.best_model]['model']
            X_val = self.extract_features(val_data)
            X_val_prep, _ = self.prepare_data(X_val, val_data['clicked'], fit=False)
            y_pred_proba = model.predict_proba(X_val_prep)[:, 1]
            y_true = val_data['clicked']
            logger.info("Creating visualizations...")
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            auc = roc_auc_score(y_true, y_pred_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve - Validation Set')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(viz_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Saved ROC curve")
            
            # Feature Importance 
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': feature_importance
                }).sort_values('importance', ascending=False).head(20)
                plt.figure(figsize=(10, 8))
                plt.barh(range(len(importance_df)), importance_df['importance'])
                plt.yticks(range(len(importance_df)), importance_df['feature'])
                plt.xlabel('Feature Importance')
                plt.title('Top 20 Feature Importances')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.savefig(viz_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
                plt.close()
                logger.info("Saved feature importance plot")
            
            # 3. Prediction Distribution
            plt.figure(figsize=(10, 6))
            # Use log scale for better visualization
            plt.hist(y_pred_proba[y_true == 0], bins=50, alpha=0.7, label='No Click', density=True)
            plt.hist(y_pred_proba[y_true == 1], bins=50, alpha=0.7, label='Click', density=True)
            plt.xlabel('Predicted Probability')
            plt.ylabel('Density')
            plt.title('Distribution of Predicted Probabilities')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(viz_dir / 'prediction_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Saved prediction distribution")
            
            # 4. Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, label=f'PR Curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(viz_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Saved precision-recall curve")
            
            # 5. Threshold Analysis
            if self.best_model in self.results and 'all_thresholds' in self.results[self.best_model]:
                thresholds_data = self.results[self.best_model]['all_thresholds']
                thresholds, precisions, recalls, f1s = zip(*thresholds_data)
                
                plt.figure(figsize=(10, 6))
                plt.plot(thresholds, precisions, 'b-', label='Precision', marker='o')
                plt.plot(thresholds, recalls, 'r-', label='Recall', marker='s')
                plt.plot(thresholds, f1s, 'g-', label='F1 Score', marker='^')
                plt.xlabel('Threshold')
                plt.ylabel('Score')
                plt.title('Metrics vs Threshold')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xscale('log')
                plt.savefig(viz_dir / 'threshold_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()
                logger.info("Saved threshold analysis")
            
            logger.info(f"All visualizations saved to {viz_dir}")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            import traceback
            traceback.print_exc()
    
    def check_data_columns(self):
        """Check what columns exist in your data"""
        files = {
            'train': self.processed_data_dir / 'train_headline_ctr.csv',
            'val': self.processed_data_dir / 'val_headline_ctr.csv', 
            'test': self.processed_data_dir / 'test_headline_ctr.csv',
            'combined': self.processed_data_dir / 'combined_headline_ctr.csv'
        }
        
        for split, file_path in files.items():
            if file_path.exists():
                df = pd.read_csv(file_path, nrows=1)  # Read just one row to see columns
                print(f"{split} columns: {list(df.columns)}")
                print(f"{split} shape: {pd.read_csv(file_path).shape}")
    
    def analyze_ctr_distribution(self):
        """Analyze CTR distribution across splits"""
        files = {
            'train': self.processed_data_dir / 'train_headline_ctr.csv',
            'val': self.processed_data_dir / 'val_headline_ctr.csv', 
            'test': self.processed_data_dir / 'test_headline_ctr.csv'
        }
        
        for split, file_path in files.items():
            if file_path.exists():
                df = pd.read_csv(file_path)
                if 'ctr' in df.columns:
                    # Basic statistics
                    print(f"\n{split} CTR Statistics:")
                    print(f"  Mean CTR: {df['ctr'].mean():.6f}")
                    print(f"  Median CTR: {df['ctr'].median():.6f}")
                    print(f"  Zero CTR count: {(df['ctr'] == 0).sum()} ({(df['ctr'] == 0).mean()*100:.1f}%)")
                    print(f"  Non-zero CTR count: {(df['ctr'] > 0).sum()} ({(df['ctr'] > 0).mean()*100:.1f}%)")
                    print(f"  Max CTR: {df['ctr'].max():.6f}")
                    print(f"  CTR > 0.1 count: {(df['ctr'] > 0.1).sum()}")
                    print(f"  CTR > 0.01 count: {(df['ctr'] > 0.01).sum()}")
                else:
                    print(f"{split}: No 'ctr' column found")
    
    def monitor_threshold_performance(self, threshold=0.05, time_period_days=7):
        """Monitor threshold performance over time"""
        if not self.best_model:
            logger.error("No trained model available for monitoring")
            return None
        
        # This is a placeholder for production monitoring
        # In production, you would load actual performance data
        logger.info(f"Monitoring threshold {threshold} performance over last {time_period_days} days")
        
        # Simulate monitoring results (replace with actual data)
        monitoring_report = {
            'threshold': threshold,
            'time_period_days': time_period_days,
            'recommendations': {
                'current_status': 'Not implemented - requires production deployment',
                'next_steps': [
                    'Deploy model to production',
                    'Track actual click rates',
                    'Compare predicted vs actual performance',
                    'Adjust threshold based on business metrics'
                ]
            }
        }
        
        # Save monitoring report
        with open(self.output_dir / 'monitoring_report.json', 'w') as f:
            json.dump(monitoring_report, f, indent=2)
        
        return monitoring_report
    
    def get_threshold_recommendation(self, probability):
        """Get human-readable recommendation based on probability"""
        if probability >= 0.005:
            return {"action": "Must Use", "color": "green"}
        elif probability >= 0.002:
            return {"action": "Strongly Recommend", "color": "blue"}
        elif probability >= 0.001:
            return {"action": "Consider - If Space Available", "color": "yellow"}
        else:
            return {"action": "Avoid - Better Options Available", "color": "red"}

if __name__ == "__main__":
    ctr_predictor = CTRPredictor()
    
    print("=== Checking data columns and shapes ===")
    ctr_predictor.check_data_columns()
    
    print("\n=== Analyzing CTR Distribution ===")
    ctr_predictor.analyze_ctr_distribution()
    print("\n=== Current class distribution (clicked vs non-clicked) ===")
    ctr_predictor.check_class_distribution()
    
    # Check if validation and test sets have positive examples
    _, val_data, test_data = ctr_predictor.load_processed_data()
    if val_data is not None:
        val_clicks = (val_data['ctr'] > 0).sum()
        test_clicks = (test_data['ctr'] > 0).sum() if test_data is not None else 0
        
        if val_clicks == 0 or test_clicks == 0:
            train, val, test = ctr_predictor.recreate_splits_with_proportions(split_percentages=(0.6, 0.2, 0.2))
            if train is not None:
                print("\n=== New class distribution after recreating splits ===")
                ctr_predictor.check_class_distribution()
                
                # Train with new splits
                print("\n=== Training with balanced splits ===")
                try:
                    results = ctr_predictor.train_pipeline()
                    
                    if results:
                        print("\n=== Training Results ===")
                        print(f"Best model: {ctr_predictor.best_model}")
                        print("\nAll model results:")
                        for model_name, model_result in results.items():
                            print(f"  {model_name}:")
                            print(f"    AUC: {model_result['auc']:.4f}")
                            if 'best_threshold' in model_result:
                                print(f"    Best threshold: {model_result['best_threshold']}")
                                print(f"    F1: {model_result['best_f1']:.4f}")
                                print(f"    Precision: {model_result['best_precision']:.4f}")
                                print(f"    Recall: {model_result['best_recall']:.4f}")
                        
                        # Additional metrics
                        if ctr_predictor.best_model:
                            best_result = results[ctr_predictor.best_model]
                            print(f"\n=== Best Model ({ctr_predictor.best_model}) Summary ===")
                            print(f"AUC: {best_result['auc']:.4f}")
                            print(f"Best threshold: {best_result.get('best_threshold', 'N/A')}")
                            print(f"F1: {best_result.get('best_f1', 'N/A'):.4f}")
                            print(f"Precision: {best_result.get('best_precision', 'N/A'):.4f}")
                            print(f"Recall: {best_result.get('best_recall', 'N/A'):.4f}")
                            print("\nInterpretation:")
                            print("- AUC > 0.5: Better than random")
                            print("- AUC > 0.7: Good performance")
                            print("- AUC > 0.9: Excellent performance")
                            print("- Precision: Of predicted clicks, what % were actual clicks")
                            print("- Recall: Of actual clicks, what % were predicted correctly")
                except Exception as e:
                    print(f"Error in training: {e}")
                    import traceback
                    traceback.print_exc()
                                        
                    # Check if any models were trained before the error
                    if ctr_predictor.results:
                        print("\n=== Partially Trained Models ===")
                        for model_name, result in ctr_predictor.results.items():
                            if result.get('auc') is not None:
                                print(f"{model_name}:")
                                print(f"  AUC: {result['auc']:.4f}")
                                if 'best_threshold' in result:
                                    print(f"  Best threshold: {result['best_threshold']}")
                                    print(f"  F1: {result['best_f1']:.4f}")
                                    print(f"  Precision: {result['best_precision']:.4f}")
                                    print(f"  Recall: {result['best_recall']:.4f}")
                        
                        # Use the best partially trained model
                        ctr_predictor.best_model = max(
                            [k for k, v in ctr_predictor.results.items() if v.get('auc') is not None], 
                            key=lambda x: ctr_predictor.results[x]['auc']
                        )
                        print(f"\nUsing best partial model: {ctr_predictor.best_model}")
                        
                        # Continue with the best model for predictions
                        results = ctr_predictor.results
        else:
            # Original splits are fine, proceed with training
            print("\n=== Training with existing splits ===")
            try:
                results = ctr_predictor.train_pipeline()
                
                if results:
                    print("\n=== Training Results ===")
                    print(f"Best model: {ctr_predictor.best_model}")
                    print("\nAll model results:")
                    for model_name, model_result in results.items():
                        print(f"  {model_name}:")
                        print(f"    AUC: {model_result['auc']:.4f}")
                        if 'best_threshold' in model_result:
                            print(f"    Best threshold: {model_result['best_threshold']}")
                            print(f"    F1: {model_result['best_f1']:.4f}")
                            print(f"    Precision: {model_result['best_precision']:.4f}")
                            print(f"    Recall: {model_result['best_recall']:.4f}")
            except Exception as e:
                print(f"Error in training: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("Could not load validation data. Attempting to recreate splits...")
        # Recreate splits with stratified approach
        train, val, test = ctr_predictor.recreate_splits_with_proportions(split_percentages=(0.6, 0.2, 0.2))
        
        if train is not None:
            # Check new class distribution
            print("\n=== New class distribution after recreating splits ===")
            ctr_predictor.check_class_distribution()
            
            # Train with new splits
            print("\n=== Training with new splits ===")
            try:
                results = ctr_predictor.train_pipeline()
                
                if results:
                    print("\n=== Training Results ===")
                    print(f"Best model: {ctr_predictor.best_model}")
                    print("\nAll model results:")
                    for model_name, model_result in results.items():
                        print(f"  {model_name}:")
                        print(f"    AUC: {model_result['auc']:.4f}")
                        if 'best_threshold' in model_result:
                            print(f"    Best threshold: {model_result['best_threshold']}")
                            print(f"    F1: {model_result['best_f1']:.4f}")
                            print(f"    Precision: {model_result['best_precision']:.4f}")
                            print(f"    Recall: {model_result['best_recall']:.4f}")
            except Exception as e:
                print(f"Error in training: {e}")
                import traceback
                traceback.print_exc()
    
    # Test predictions if any model was trained successfully
    if ctr_predictor.best_model and ctr_predictor.results:
        print("\n=== Testing Predictions ===")
        print(f"Using model: {ctr_predictor.best_model}")
        
        test_headlines = [
            "Biden Announces New Economic Policy",
            "Tech Giants Report Record Earnings",
            "Climate Summit Reaches Historic Agreement",
            "How to Make Money Online in 2024",
            "Is This the End of Remote Work?",
            "Why You Should Avoid This Common Mistake",
            "Expert Tips: How to Save Money Fast"
        ]
        
        # Test with different filtering options
        predictions_all = ctr_predictor.predict_click_probability(test_headlines, use_best_threshold=False)
        predictions_filtered = ctr_predictor.predict_click_probability(test_headlines, use_best_threshold=True)
        best_threshold = ctr_predictor.results[ctr_predictor.best_model].get('best_threshold', None)
        
        if predictions_all:
            print("\nPrediction Results (all headlines, ranked by probability):")
            print("-" * 80)
            for pred in predictions_all:
                print(f"Rank {pred['rank']:2d}: {pred['headline']}")
                print(f"         Click Probability: {pred['click_probability']:.6f}")
                print(f"         Relative Rank: {pred['relative_rank']}")
                print(f"         Percentile: {pred['percentile']}th percentile")
                print()
            
            # Show filtered results if different
            if len(predictions_filtered) == 0:
                print(f"\n  No headlines above f1 threshold ({best_threshold:.6f})")
                print("This is expected with low CTR data.")
                print("\nTop 3 recommendations (regardless of threshold):")
                print("-" * 50)
                for pred in predictions_all[:3]:
                    print(f"{pred['rank']}. {pred['headline']}")
                    print(f"   Probability: {pred['click_probability']:.6f} ({pred['percentile']}th percentile)")
                    print()
                    
            elif len(predictions_filtered) < len(predictions_all):
                print(f"\nFiltered Results (above F1 threshold, {len(predictions_filtered)} out of {len(predictions_all)}):")
                print("-" * 80)
                for pred in predictions_filtered:
                    print(f"Rank {pred['rank']:2d}: {pred['headline']}")
                    print(f"         Click Probability: {pred['click_probability']:.6f}")
                    print(f"         Relative Rank: {pred['relative_rank']}")
                    print()
            else:
                print("\nAll headlines are above the F1 threshold (unusual for low-CTR data)")
            
            print("\n=== Dataset Insights ===")
            print(f"- Training data: ~{len(ctr_predictor.results[ctr_predictor.best_model].get('predictions', [])):,} samples")
            print(f"- Positive rate: Very low (~1-2%)")
            print(f"- Model achieved: {ctr_predictor.results[ctr_predictor.best_model]['auc']:.4f} AUC")
            if 'best_threshold' in ctr_predictor.results[ctr_predictor.best_model]:
                print(f"- Best threshold: {ctr_predictor.results[ctr_predictor.best_model]['best_threshold']:.6f}")
            print("\nNote: Low probabilities (0.001-0.01) are expected given the low CTR in the dataset.")
            print("Focus on relative ranking rather than absolute probabilities.")
    else:
        print("\n=== No Successfully Trained Models ===")
        print("Unable to make predictions. Please check the error messages above.")


# # Additional utility functions for production use

# def load_and_predict(model_path, headlines):
#     """Convenience function to load model and make predictions"""
#     predictor = CTRPredictor()
#     predictor.load_model(model_path)
#     return predictor.predict_click_probability(headlines)

# def batch_predict_from_csv(model_path, input_csv, output_csv, headline_column='title'):
#     """Predict CTR for headlines in a CSV file"""
#     predictor = CTRPredictor()
#     predictor.load_model(model_path)
    
#     df = pd.read_csv(input_csv)
#     headlines = df[headline_column].tolist()
    
#     predictions = predictor.predict_click_probability(headlines, use_best_threshold=False)
    
#     # Create results DataFrame
#     results_df = pd.DataFrame(predictions)
#     results_df.to_csv(output_csv, index=False)
    
#     logger.info(f"Predictions saved to {output_csv}")
#     return results_df

# def evaluate_model_drift(model_path, new_test_data_path):
#     """Evaluate if the model is experiencing drift on new data"""
#     predictor = CTRPredictor()
#     predictor.load_model(model_path)
    
#     # Load new test data
#     new_data = pd.read_csv(new_test_data_path)
    
#     # Make predictions
#     features = predictor.extract_features(new_data)
#     X_new = features[predictor.feature_names]
#     X_new_scaled = predictor.scaler.transform(X_new)
    
#     # Handle NaN values
#     if np.isnan(X_new_scaled).any():
#         X_new_scaled = predictor.imputer.transform(X_new_scaled)
    
#     model = predictor.results[predictor.best_model]['model']
#     predictions = model.predict_proba(X_new_scaled)[:, 1]
    
#     # Compare with original model performance
#     if 'clicked' in new_data.columns:
#         y_true = new_data['clicked']
#         new_auc = roc_auc_score(y_true, predictions)
#         original_auc = predictor.results[predictor.best_model]['auc']
        
#         drift_report = {
#             'original_auc': original_auc,
#             'new_auc': new_auc,
#             'auc_difference': abs(original_auc - new_auc),
#             'performance_degraded': new_auc < original_auc - 0.05,  # Alert if AUC drops by more than 0.05
#             'recommendation': 'Retrain model' if new_auc < original_auc - 0.05 else 'Model still performing well'
#         }
        
#         return drift_report
#     else:
#         logger.warning("No ground truth available in new data")
#         return None