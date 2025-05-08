import os
import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HeadlineModelTrainer:
    """
    Trains and evaluates a model for predicting headline CTR based on the MIND dataset
    and your EDA insights.
    """
    
    def __init__(self, processed_data_dir='agentic_news_editor/processed_data'):
        """Initialize the headline model trainer"""
        self.processed_data_dir = processed_data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        
        # Load and prepare models
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.bert_model = AutoModel.from_pretrained("distilbert-base-uncased")
            self.bert_model = self.bert_model.to(self.device)
            logging.info("Loaded DistilBERT model successfully")
        except Exception as e:
            logging.error(f"Failed to load DistilBERT model: {e}")
            raise ValueError(f"Could not load embedding model: {e}")
    
    def load_training_data(self):
        """Load processed headlines with CTR data"""
        try:
            headlines_path = os.path.join(self.processed_data_dir, 'news_with_engagement.csv') 
            if not os.path.exists(headlines_path):
                logging.error(f"Training data not found at {headlines_path}")
                return None
                
            headline_data = pd.read_csv(headlines_path)
            logging.info(f"Loaded {len(headline_data)} headlines with CTR data")
            return headline_data
        except Exception as e:
            logging.error(f"Error loading training data: {e}")
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
                
                # Get embedding for semantic features
                try:
                    inputs = self.tokenizer(headline, return_tensors="pt", padding=True, truncation=True, max_length=128)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.bert_model(**inputs)
                    
                    # Use the [CLS] token embedding
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                    
                    # Add first 10 embedding dimensions as features
                    for i in range(10):
                        features[f'emb_{i}'] = embedding[i]
                except Exception as e:
                    logging.error(f"Error extracting embedding for '{headline}': {e}")
                    # Add zero embeddings if failed
                    for i in range(10):
                        features[f'emb_{i}'] = 0.0
                
                features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def train_model(self, features_df, ctr_values, output_file='headline_ctr_model.pkl'):
        """Train a model to predict headline CTR"""
        logging.info("Training headline CTR prediction model")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, ctr_values, test_size=0.2, random_state=42
        )
        
        # Define and train model
        model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10,
            min_samples_split=10,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logging.info(f"Model evaluation - MSE: {mse:.4f}, R²: {r2:.4f}")
        
        # Feature importance
        feature_importances = pd.DataFrame({
            'feature': features_df.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logging.info("Top 10 important features:")
        for i, row in feature_importances.head(10).iterrows():
            logging.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importances.head(10))
        plt.title('Top 10 Feature Importances for CTR Prediction')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        
        # Save model
        with open(output_file, 'wb') as f:
            pickle.dump(model, f)
        
        logging.info(f"Model saved to {output_file}")
        
        return {
            'model': model,
            'mse': mse,
            'r2': r2,
            'feature_importances': feature_importances
        }
    
    def run_training_pipeline(self):
        """Run the complete model training pipeline"""
        # Load data
        headline_data = self.load_training_data()
        if headline_data is None:
            logging.error("Could not load training data. Aborting.")
            return None
        
        # Handle NaN values
        headline_data = headline_data.dropna(subset=['title', 'ctr'])
        
        # Extract features
        features_df = self.extract_features(headline_data['title'].values)
        
        # Train model
        result = self.train_model(features_df, headline_data['ctr'].values)
        
        # Create a report
        self.create_model_report(result, headline_data)
        
        return result
    
    def create_model_report(self, result, headline_data):
        """Create a markdown report about the model performance"""
        if result is None:
            return
        
        report = f"""# Headline CTR Prediction Model Report
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

## Model Performance
- Mean Squared Error: {result['mse']:.4f}
- R-squared: {result['r2']:.4f}

## Dataset Summary
- Total headlines analyzed: {len(headline_data)}
- CTR range: {headline_data['ctr'].min():.2f} to {headline_data['ctr'].max():.2f}
- Mean CTR: {headline_data['ctr'].mean():.2f}

## Key Feature Importances
"""
        
        for i, row in result['feature_importances'].head(10).iterrows():
            report += f"- {row['feature']}: {row['importance']:.4f}\n"
        
        report += """
## Usage Guidelines
This model can be used to predict the expected CTR of news headlines.
It's integrated with the HeadlineMetrics class for headline evaluation
and the HeadlineLearningLoop for continuous improvement.

## Features Based on EDA
The model uses features derived from EDA findings:
- Questions in headlines significantly reduce CTR
- Numbers in headlines can reduce CTR if used inappropriately
- Headline length and structure matter for engagement
- Category-specific patterns influence performance
"""
        
        with open('headline_model_report.md', 'w') as f:
            f.write(report)
        
        logging.info("Model report saved to headline_model_report.md")


if __name__ == "__main__":
    trainer = HeadlineModelTrainer()
    result = trainer.run_training_pipeline()
    
    if result is not None:
        print(f"Model training complete. R-squared: {result['r2']:.4f}")
    else:
        print("Model training failed.")