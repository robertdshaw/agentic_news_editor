import os
import pandas as pd
import numpy as np
import pickle
import logging
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HeadlineModelTrainer:
    """
    Trains and evaluates a model for predicting headline CTR based on the MIND dataset
    and the preprocessed train/val/test splits.
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
                    
                    # Add first 10 embedding dimensions as features
                    for j in range(20):
                        features[f'emb_{j}'] = embedding[j]
                except Exception as e:
                    logging.error(f"Error extracting embedding for '{headline}': {e}")
                    # Add zero embeddings if failed
                    for j in range(10):
                        features[f'emb_{j}'] = 0.0
                
                features_list.append(features)
        
        return pd.DataFrame(features_list)
    

    def train_model(self, train_features, train_ctr, val_features=None, val_ctr=None,
               output_file='headline_ctr_model.pkl', batch_info=None):
        """
        Train model with simplified approach (no cross-validation)
        
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
        base = XGBRegressor(
            n_estimators=100, learning_rate=0.01, max_depth=4,
            min_child_weight=5, subsample=0.7, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=2.0, objective='reg:squarederror',
            random_state=42, n_jobs=-1
        )
        
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
        
        # Define simpler hyperparameters
        final_params = {
            'n_estimators': 200, 
            'learning_rate': 0.01,
            'max_depth': 4, 
            'min_child_weight': 5,
            'subsample': 0.7, 
            'colsample_bytree': 0.7
        }
        
        # Create final model
        logging.info(f"Training final model with params: {final_params}")
        final = XGBRegressor(
            objective='reg:squarederror',
            reg_alpha=1.0, reg_lambda=2.0, 
            random_state=42, n_jobs=-1,
            **final_params
        )
        
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
            if hasattr(self, 'visualize_predictions'):
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
        with open(os.path.join(self.output_dir, output_file), 'wb') as f:
            pickle.dump(model_data, f)
        logging.info(f"Model saved to {os.path.join(self.output_dir, output_file)}")
        
        if hasattr(self, 'visualize_feature_importance'):
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
    
    def run_training_pipeline(self, force_reprocess=False):
        if not force_reprocess:
            tf, ty, vf, vy, testf, testy = load_processed_data()
            if tf is not None:
                logging.info("Using cached data.")
                # Check if we have a batch checkpoint
                batch_checkpoint_path = os.path.join(self.output_dir, 'batch_checkpoint.pkl')
                batch_info = None
                if os.path.exists(batch_checkpoint_path):
                    try:
                        with open(batch_checkpoint_path, 'rb') as f:
                            batch_info = pickle.load(f)
                        if batch_info.get('stage') == 'complete':
                            logging.info("Training already completed. Loading saved model.")
                        else:
                            logging.info(f"Resuming training from stage '{batch_info.get('stage')}'")
                    except Exception:
                        pass
                return self.train_model(tf, ty, vf, vy, batch_info=batch_info)
        """Run the complete model training pipeline with train/val/test splits"""
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
        
        # Visualize CTR distribution
        self.visualize_ctr_distribution(train_data['ctr'].values, val_data['ctr'].values)
        
        # Extract features for all splits
        logging.info("Extracting features for training data...")
        train_features = self.extract_features(train_data['title'].values)
        
        logging.info("Extracting features for validation data...")
        val_features = self.extract_features(val_data['title'].values)
        
        test_features = None
        if test_data is not None:
            logging.info("Extracting features for test data...")
            test_features = self.extract_features(test_data['title'].values)
        
        # Train model using proper splits
        result = self.train_model(
            train_features, train_data['ctr'].values,
            val_features, val_data['ctr'].values,
            output_file='headline_ctr_model.pkl'
        )
        
        # Visualize predictions for validation set
        if self.use_log_transform:
            val_pred = np.expm1(result['model'].predict(val_features))
        else:
            val_pred = result['model'].predict(val_features)
            
        self.visualize_predictions(val_data['ctr'].values, val_pred, 'validation_predictions.png')
        
        # If test data is available, generate predictions
        if test_data is not None and test_features is not None:
            logging.info("Generating predictions for test set...")
            
            if self.use_log_transform:
                test_pred_transformed = result['model'].predict(test_features)
                test_predictions = np.expm1(test_pred_transformed)
            else:
                test_predictions = result['model'].predict(test_features)
            
            # Save test predictions
            test_results = test_data[['newsID', 'title']].copy()
            test_results['predicted_ctr'] = test_predictions
            test_results.to_csv(os.path.join(self.output_dir, 'test_predictions.csv'), index=False)
            logging.info(f"Test predictions saved to {os.path.join(self.output_dir, 'test_predictions.csv')}")
        
        # Create a report
        self.create_model_report(result, train_data, val_data, test_data)
        
        return result
    
    def create_model_report(self, result, train_data, val_data, test_data=None):
        """Create a markdown report about the model performance"""
        if result is None:
            return
        
        report = f"""# Headline CTR Prediction Model Report
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

## Model Configuration
- Model Type: RandomForestRegressor
- Log Transform CTR: {self.use_log_transform}
- Training Time: {result['training_time']:.2f} seconds

## Model Performance
- Training MSE: {result['train_mse']:.6f}
- Training RMSE: {result['train_rmse']:.6f}
- Training MAE: {result['train_mae']:.6f}
- Training R-squared: {result['train_r2']:.4f}
"""

        if result['val_mse'] is not None:
            report += f"""
- Validation MSE: {result['val_mse']:.6f}
- Validation RMSE: {result['val_rmse']:.6f}
- Validation MAE: {result['val_mae']:.6f}
- Validation R-squared: {result['val_r2']:.4f}
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


if __name__ == "__main__":
    # Create trainer (set use_log_transform=False if you don't want log transformation)
    trainer = HeadlineModelTrainer(use_log_transform=False)
    result = trainer.run_training_pipeline()
    
    if result is not None:
        print(f"Model training complete. Training R-squared: {result['train_r2']:.4f}, Validation R-squared: {result['val_r2']:.4f}")
    else:
        print("Model training failed.")