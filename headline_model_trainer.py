import os
import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import re
from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import time

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
        self.use_log_transform = use_log_transform
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        logging.info(f"Log transform for CTR: {self.use_log_transform}")
        
        # Create output directory for results
        self.output_dir = 'model_output'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
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
        batch_size = 100
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
                
                # Get embedding for semantic features
                try:
                    inputs = self.tokenizer(headline, return_tensors="pt", padding=True, truncation=True, max_length=128)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.bert_model(**inputs)
                    
                    # Use the [CLS] token embedding
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                    
                    # Add first 10 embedding dimensions as features
                    for j in range(10):
                        features[f'emb_{j}'] = embedding[j]
                except Exception as e:
                    logging.error(f"Error extracting embedding for '{headline}': {e}")
                    # Add zero embeddings if failed
                    for j in range(10):
                        features[f'emb_{j}'] = 0.0
                
                features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def train_model(self, train_features, train_ctr, val_features=None, val_ctr=None, 
                    output_file='headline_ctr_model.pkl'):
        """Train a model to predict headline CTR using proper train/val splits"""
        logging.info(f"Training headline CTR prediction model on {len(train_features)} examples")
        
        # Apply log transformation to handle skewed CTR distribution
        if self.use_log_transform:
            # Add a small constant to avoid log(0)
            train_ctr_transformed = np.log1p(train_ctr)
            logging.info(f"Applied log transformation to CTR values")
        else:
            train_ctr_transformed = train_ctr
        
        # Define and train model on full training set
        model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        
        # Record training time
        start_time = time.time()
        model.fit(train_features, train_ctr_transformed)
        training_time = time.time() - start_time
        logging.info(f"Model training completed in {training_time:.2f} seconds")
        
        # Evaluate on training set
        train_pred_transformed = model.predict(train_features)
        
        # Convert predictions back to original scale if log transform was used
        if self.use_log_transform:
            train_pred = np.expm1(train_pred_transformed)
        else:
            train_pred = train_pred_transformed
            
        train_mse = mean_squared_error(train_ctr, train_pred)
        train_mae = mean_absolute_error(train_ctr, train_pred)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(train_ctr, train_pred)
        
        logging.info(f"Training metrics - MSE: {train_mse:.6f}, RMSE: {train_rmse:.6f}, MAE: {train_mae:.6f}, R²: {train_r2:.4f}")
        
        # Evaluate on validation set if provided
        val_mse, val_mae, val_rmse, val_r2 = None, None, None, None
        if val_features is not None and val_ctr is not None:
            val_pred_transformed = model.predict(val_features)
            
            # Convert predictions back to original scale if log transform was used
            if self.use_log_transform:
                val_pred = np.expm1(val_pred_transformed)
            else:
                val_pred = val_pred_transformed
                
            val_mse = mean_squared_error(val_ctr, val_pred)
            val_mae = mean_absolute_error(val_ctr, val_pred)
            val_rmse = np.sqrt(val_mse)
            val_r2 = r2_score(val_ctr, val_pred)
            
            logging.info(f"Validation metrics - MSE: {val_mse:.6f}, RMSE: {val_rmse:.6f}, MAE: {val_mae:.6f}, R²: {val_r2:.4f}")
        
        # Feature importance
        feature_importances = pd.DataFrame({
            'feature': train_features.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logging.info("Top 10 important features:")
        for i, row in feature_importances.head(10).iterrows():
            logging.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Plot feature importance
        self.visualize_feature_importance(feature_importances)
        
        # Save model
        with open(os.path.join(self.output_dir, output_file), 'wb') as f:
            pickle.dump({
                'model': model,
                'use_log_transform': self.use_log_transform,
                'feature_names': train_features.columns.tolist()
            }, f)
        
        logging.info(f"Model saved to {os.path.join(self.output_dir, output_file)}")
        
        # Return results
        return {
            'model': model,
            'train_mse': train_mse,
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'val_mse': val_mse,
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'val_r2': val_r2,
            'feature_importances': feature_importances,
            'training_time': training_time
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
    trainer = HeadlineModelTrainer(use_log_transform=True)
    result = trainer.run_training_pipeline()
    
    if result is not None:
        print(f"Model training complete. Training R-squared: {result['train_r2']:.4f}, Validation R-squared: {result['val_r2']:.4f}")
    else:
        print("Model training failed.")