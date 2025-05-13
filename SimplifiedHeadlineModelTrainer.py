import pandas as pd
import numpy as np
import os
import logging
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, f1_score, precision_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImprovedHeadlineFeatures:
    """
    Improved feature extraction based on EDA insights
    """
    
    def __init__(self):
        # High-performing categories from EDA
        self.high_ctr_categories = {'kids', 'music', 'tv'}
        self.low_ctr_categories = {'autos', 'travel', 'northamerica'}
        
        # Pattern definitions
        self.positive_patterns = {
            'authority': [r'\b(expert|scientist|study|research|reveals|finds|shows)\b'],
            'urgency': [r'\b(breaking|urgent|just|now|today|latest)\b'],
            'list_numbers': [r'\b\d+\s+(ways|tips|secrets|tricks|facts|things)\b'],
            'exclusive': [r'\b(exclusive|first|only|never\s*before)\b'],
            'emotional_positive': [r'\b(amazing|incredible|stunning|shocking|mind\s*blowing)\b']
        }
        
        self.negative_patterns = {
            'generic_questions': [r'\b(how|what|why|when|where|who|which)\b'],
            'weak_language': [r'\b(maybe|perhaps|might|could|possibly)\b'],
            'overused_clickbait': [r'\b(one\s*weird\s*trick|doctors\s*hate|you\s*won\'t\s*believe)\b']
        }
    
    def extract_features(self, headlines, categories=None, abstracts=None):
        """Extract features with proper handling of all cases"""
        features_list = []
        
        for i, headline in enumerate(headlines):
            if pd.isna(headline):
                headline = ""
            
            features = {}
            headline_lower = headline.lower()
            
            # 1. QUESTION-RELATED FEATURES
            features['is_question'] = int(headline.endswith('?'))
            features['has_question_words'] = int(bool(re.search(r'\b(how|what|why|when|where|who|which)\b', headline_lower)))
            features['starts_with_question'] = int(bool(re.match(r'^(how|what|why|when|where|who|which)\b', headline_lower)))
            
            # 2. NUMBER-RELATED FEATURES
            features['has_numbers'] = int(bool(re.search(r'\d', headline)))
            features['num_count'] = len(re.findall(r'\d+', headline))
            features['starts_with_number'] = int(bool(re.match(r'^\d+', headline)))
            features['has_list_number'] = int(bool(re.search(r'\b\d+\s+(ways|tips|secrets|tricks|facts|things|reasons)\b', headline_lower)))
            
            # 3. STRUCTURE FEATURES
            features['has_colon'] = int(':' in headline)
            features['has_quotes'] = int(bool(re.search(r'["\']', headline)))
            features['has_exclamation'] = int('!' in headline)
            features['has_parentheses'] = int(bool(re.search(r'[()]', headline)))
            
            # 4. LENGTH FEATURES
            features['char_length'] = len(headline)
            features['word_count'] = len(headline.split())
            features['avg_word_length'] = np.mean([len(word) for word in headline.split()]) if headline.split() else 0
            
            # 5. CATEGORY FEATURES - FIXED TO ALWAYS INCLUDE
            if categories is not None and i < len(categories) and pd.notna(categories[i]):
                category = str(categories[i]).lower()
            else:
                category = 'unknown'
            
            # Always set all category features (prevent KeyError)
            features['category_high_ctr'] = int(category in self.high_ctr_categories)
            features['category_low_ctr'] = int(category in self.low_ctr_categories)
            features['category_kids'] = int(category == 'kids')
            features['category_music'] = int(category == 'music')
            features['category_tv'] = int(category == 'tv')
            features['category_autos'] = int(category == 'autos')
            features['category_travel'] = int(category == 'travel')
            features['category_northamerica'] = int(category == 'northamerica')
            
            # 6. POSITIVE PATTERNS
            for pattern_type, patterns in self.positive_patterns.items():
                features[f'has_{pattern_type}'] = int(any(bool(re.search(p, headline_lower)) for p in patterns))
            
            # 7. NEGATIVE PATTERNS
            for pattern_type, patterns in self.negative_patterns.items():
                features[f'has_{pattern_type}'] = int(any(bool(re.search(p, headline_lower)) for p in patterns))
            
            # 8. WORD POSITION FEATURES
            words = headline.split()
            if words:
                features['first_word_length'] = len(words[0])
                features['last_word_length'] = len(words[-1])
                features['first_word_caps'] = int(words[0][0].isupper()) if words[0] else 0
                features['last_word_caps'] = int(words[-1][0].isupper()) if words[-1] else 0
            else:
                features['first_word_length'] = 0
                features['last_word_length'] = 0
                features['first_word_caps'] = 0
                features['last_word_caps'] = 0
            
            # 9. CAPITALIZATION
            features['all_caps_words'] = sum(1 for word in words if word.isupper() and len(word) > 1) if words else 0
            features['cap_ratio'] = sum(1 for c in headline if c.isupper()) / len(headline) if headline else 0
            
            # 10. ADVANCED FEATURES
            features['has_action_words'] = int(bool(re.search(r'\b(get|find|learn|discover|unlock|master)\b', headline_lower)))
            features['has_numbers_and_action'] = features['has_numbers'] * features['has_action_words']
            features['question_length_penalty'] = features['is_question'] * features['char_length']
            features['positive_number_indicator'] = features['has_list_number'] * (1 - features['is_question'])
            features['has_breaking_news'] = int(bool(re.search(r'\b(breaking|urgent|alert)\b', headline_lower)))
            features['has_time_reference'] = int(bool(re.search(r'\b(today|now|just|latest|new|recent)\b', headline_lower)))
            features['has_personal_pronoun'] = int(bool(re.search(r'\b(you|your|we|our|i|my)\b', headline_lower)))
            features['has_celebrity_pattern'] = int(bool(re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', headline)))
            features['has_price'] = int(bool(re.search(r'[\$£€¥]|\b(cost|price|expensive|cheap|free)\b', headline_lower)))
            features['has_comparison'] = int(bool(re.search(r'\b(vs|versus|better|best|worst|than|compared)\b', headline_lower)))
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)

class SimplifiedHeadlineTrainer:
    """
    Simplified trainer focused on achieving high AUC and F1 scores
    """
    
    def __init__(self, output_dir='output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.feature_extractor = ImprovedHeadlineFeatures()
        self.scaler = StandardScaler()
        
        # Store best model info
        self.best_model = None
        self.best_model_name = None
        self.best_features = None
        self.best_metrics = None
        
    def load_and_prepare_data(self, data_dir='agentic_news_editor/processed_data'):
        """Load and prepare training and validation data with imbalance handling"""
        logging.info("Loading training and validation data...")
        
        # Load data
        train_path = os.path.join(data_dir, 'train_headline_ctr.csv')
        val_path = os.path.join(data_dir, 'val_headline_ctr.csv')
        
        train_data = pd.read_csv(train_path)
        val_data = pd.read_csv(val_path)
        
        # Clean data
        train_data = train_data.dropna(subset=['title', 'ctr'])
        val_data = val_data.dropna(subset=['title', 'ctr'])
        
        # Convert to binary classification
        train_data['clicked'] = (train_data['ctr'] > 0).astype(int)
        val_data['clicked'] = (val_data['ctr'] > 0).astype(int)
        
        logging.info(f"Training data: {len(train_data)} samples")
        logging.info(f"Validation data: {len(val_data)} samples")
        logging.info(f"Training click rate: {train_data['clicked'].mean():.4f}")
        logging.info(f"Validation click rate: {val_data['clicked'].mean():.4f}")
        
        # Handle severe imbalance
        if train_data['clicked'].mean() < 0.05:
            logging.info("Addressing severe class imbalance with undersampling...")
            positive_samples = train_data[train_data['clicked'] == 1]
            negative_samples = train_data[train_data['clicked'] == 0]
            
            # Keep all positive but limit negative to 5:1 ratio
            n_positive = len(positive_samples)
            n_negative_keep = min(n_positive * 5, len(negative_samples))
            
            negative_samples_reduced = negative_samples.sample(n_negative_keep, random_state=42)
            train_data = pd.concat([positive_samples, negative_samples_reduced]).sample(frac=1, random_state=42)
            
            logging.info(f"Reduced training data to {len(train_data)} samples")
            logging.info(f"New training click rate: {train_data['clicked'].mean():.4f}")
        
        return train_data, val_data
    
    def extract_features(self, data):
        """Extract features using improved feature extractor"""
        logging.info("Extracting features...")
        
        headlines = data['title'].tolist()
        categories = data['category'].tolist() if 'category' in data.columns else None
        
        # Extract features
        features = self.feature_extractor.extract_features(headlines, categories)
        
        # Handle any remaining NaN values
        features = features.fillna(0)
        
        # Log correlations with target if available
        if 'clicked' in data.columns:
            correlations = {}
            for col in features.columns:
                corr_val = features[col].corr(data['clicked'])
                if not pd.isna(corr_val):
                    correlations[col] = abs(corr_val)
            
            # Show top correlated features
            top_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:10]
            logging.info("Top 10 correlated features:")
            for feat, corr in top_corr:
                if not pd.isna(corr):
                    logging.info(f"  {feat}: {corr:.4f}")
        
        logging.info(f"Extracted {features.shape[1]} features")
        return features
    
    def feature_selection(self, X_train, y_train, method='correlation', top_k=40):
        """Select top features with focus on EDA insights"""
        logging.info(f"Selecting top {top_k} features using {method}...")
        
        # Calculate correlations
        correlations = {}
        for col in X_train.columns:
            corr_val = X_train[col].corr(y_train)
            if not pd.isna(corr_val):
                correlations[col] = abs(corr_val)
            else:
                correlations[col] = 0
        
        # Always include critical EDA features
        critical_features = [
            'is_question', 'has_question_words', 'starts_with_question',
            'has_numbers', 'has_list_number',
            'category_high_ctr', 'category_low_ctr',
            'has_authority', 'has_urgency', 'has_quotes'
        ]
        
        # Get available critical features
        selected_features = [f for f in critical_features if f in X_train.columns]
        
        # Add top correlated features
        sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        for feat, _ in sorted_corr:
            if feat not in selected_features and len(selected_features) < top_k:
                selected_features.append(feat)
        
        logging.info(f"Selected {len(selected_features)} features")
        return selected_features
    
    def train_and_evaluate_models(self, X_train, y_train, X_val, y_val):
        """Train and evaluate models optimized for imbalanced data"""
        logging.info("Training and evaluating models...")
        
        # Calculate class weights
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        models = {
            'XGBoost': XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                min_child_weight=5,
                scale_pos_weight=pos_weight,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.5,
                random_state=42,
                eval_metric='auc'
            ),
            'LogisticRegression': LogisticRegression(
                C=0.1,
                penalty='l2',
                solver='liblinear',
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                class_weight='balanced'
            )
        }
        
        results = {}
        
        for model_name, model in models.items():
            logging.info(f"\nTraining {model_name}...")
            
            # Handle scaling for LogisticRegression
            if model_name == 'LogisticRegression':
                X_train_processed = self.scaler.fit_transform(X_train)
                X_val_processed = self.scaler.transform(X_val)
            else:
                X_train_processed = X_train
                X_val_processed = X_val
            
            # Train model
            model.fit(X_train_processed, y_train)
            
            # Make predictions
            y_pred = model.predict(X_val_processed)
            y_pred_proba = model.predict_proba(X_val_processed)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': model.score(X_val_processed, y_val),
                'precision': precision_score(y_val, y_pred, zero_division=0),
                'recall': recall_score(y_val, y_pred, zero_division=0),
                'f1': f1_score(y_val, y_pred, zero_division=0),
                'auc': roc_auc_score(y_val, y_pred_proba)
            }
            
            results[model_name] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred_proba,
                'scaler': self.scaler if model_name == 'LogisticRegression' else None
            }
            
            # Log metrics
            logging.info(f"{model_name} metrics:")
            for metric, value in metrics.items():
                logging.info(f"  {metric}: {value:.4f}")
        
        return results
    
    def select_best_model(self, results, metric='f1'):
        """Select best model based on F1 score (better for imbalanced data)"""
        logging.info(f"\nSelecting best model based on {metric}...")
        
        best_score = 0
        best_model_name = None
        
        for model_name, result in results.items():
            score = result['metrics'][metric]
            if score > best_score:
                best_score = score
                best_model_name = model_name
        
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        self.best_metrics = results[best_model_name]['metrics']
        
        if results[best_model_name]['scaler']:
            self.scaler = results[best_model_name]['scaler']
        
        logging.info(f"Best model: {best_model_name} with {metric} = {best_score:.4f}")
        return best_model_name, results[best_model_name]
    
    def optimize_threshold(self, X_val, y_val):
        """Find optimal threshold for F1 score"""
        if self.best_model is None:
            return 0.5
        
        # Prepare validation data
        if self.best_model_name == 'LogisticRegression':
            X_val_processed = self.scaler.transform(X_val)
        else:
            X_val_processed = X_val
        
        # Get prediction probabilities
        y_proba = self.best_model.predict_proba(X_val_processed)[:, 1]
        
        # Try different thresholds
        thresholds = np.arange(0.1, 0.9, 0.02)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            f1 = f1_score(y_val, y_pred)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        logging.info(f"Optimal threshold: {best_threshold:.3f} (F1 = {best_f1:.4f})")
        return best_threshold
    
    def create_visualizations(self, X_val, y_val, results):
        """Create performance visualizations"""
        logging.info("Creating visualizations...")
        
        # 1. Model comparison plot
        plt.figure(figsize=(12, 8))
        metrics_df = pd.DataFrame({
            model_name: result['metrics'] 
            for model_name, result in results.items()
        }).T
        
        ax = metrics_df[['auc', 'f1', 'precision', 'recall']].plot(kind='bar')
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.legend(loc='upper right')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_comparison.png'), dpi=300)
        plt.close()
        
        # 2. Confusion matrix for best model
        if self.best_model_name == 'LogisticRegression':
            X_val_processed = self.scaler.transform(X_val)
        else:
            X_val_processed = X_val
        
        y_pred = self.best_model.predict(X_val_processed)
        
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_val, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()
        
        # 3. Feature importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.best_features,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False).head(15)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(data=feature_importance, x='importance', y='feature')
            plt.title(f'Top 15 Feature Importances - {self.best_model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'), dpi=300)
            plt.close()
    
    def save_model(self):
        """Save the best model and metadata"""
        if self.best_model is None:
            logging.error("No model to save!")
            return
        
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler if self.best_model_name == 'LogisticRegression' else None,
            'features': self.best_features,
            'feature_extractor': self.feature_extractor,
            'metrics': self.best_metrics,
            'model_name': self.best_model_name
        }
        
        model_path = os.path.join(self.output_dir, 'simplified_headline_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logging.info(f"Model saved to {model_path}")
        
        # Save summary
        summary = f"""# Headline CTR Prediction Model Summary

## Best Model: {self.best_model_name}

## Performance Metrics:
- AUC: {self.best_metrics['auc']:.4f}
- F1 Score: {self.best_metrics['f1']:.4f}
- Precision: {self.best_metrics['precision']:.4f}
- Recall: {self.best_metrics['recall']:.4f}
- Accuracy: {self.best_metrics['accuracy']:.4f}

## Features Used: {len(self.best_features)} features

## Key EDA Insights Implemented:
- Questions reduce CTR (detected and penalized)
- Numbers have slight negative effect (handled appropriately)
- Category effects incorporated
- Authority and urgency patterns detected

## Usage:
```python
import pickle
with open('simplified_headline_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Use the model...
```
"""
        
        summary_path = os.path.join(self.output_dir, 'model_summary.md')
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        logging.info(f"Summary saved to {summary_path}")
    
    def run_training_pipeline(self):
        """Run the complete simplified training pipeline"""
        logging.info("Starting simplified headline CTR prediction pipeline...")
        
        try:
            # 1. Load data
            train_data, val_data = self.load_and_prepare_data()
            
            # 2. Extract features
            X_train = self.extract_features(train_data)
            X_val = self.extract_features(val_data)
            
            y_train = train_data['clicked']
            y_val = val_data['clicked']
            
            # 3. Feature selection
            selected_features = self.feature_selection(X_train, y_train)
            X_train_selected = X_train[selected_features]
            X_val_selected = X_val[selected_features]
            self.best_features = selected_features
            
            # 4. Train and evaluate models
            results = self.train_and_evaluate_models(X_train_selected, y_train, X_val_selected, y_val)
            
            # 5. Select best model (using F1 score for imbalanced data)
            best_model_name, best_result = self.select_best_model(results, metric='f1')
            
            # 6. Optimize threshold
            best_threshold = self.optimize_threshold(X_val_selected, y_val)
            
            # 7. Create visualizations
            self.create_visualizations(X_val_selected, y_val, results)
            
            # 8. Save model
            self.save_model()
            
            # 9. Print final results
            logging.info("\n" + "="*50)
            logging.info("PIPELINE COMPLETE!")
            logging.info("="*50)
            logging.info(f"Best Model: {self.best_model_name}")
            logging.info(f"AUC: {self.best_metrics['auc']:.4f}")
            logging.info(f"F1 Score: {self.best_metrics['f1']:.4f}")
            logging.info(f"Precision: {self.best_metrics['precision']:.4f}")
            logging.info(f"Recall: {self.best_metrics['recall']:.4f}")
            logging.info(f"Optimal Threshold: {best_threshold:.3f}")
            logging.info(f"Results saved to: {self.output_dir}")
            
            return {
                'best_model': self.best_model,
                'best_model_name': self.best_model_name,
                'metrics': self.best_metrics,
                'threshold': best_threshold,
                'features': self.best_features
            }
            
        except Exception as e:
            logging.error(f"Error in pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Example usage of the simplified trainer"""
    
    # Initialize trainer
    trainer = SimplifiedHeadlineTrainer(output_dir='output')
    
    # Run the pipeline
    result = trainer.run_training_pipeline()
    
    if result:
        # Example prediction with proper feature handling
        test_headlines = [
            "How to Make Money Online?",  # Question (should have lower probability)
            "Expert Study Reveals Success Secrets",  # Authority words (should be higher)
            "Breaking: 5 Tips for Success",  # Urgency + list number
        ]
        
        print("\nPrediction Examples:")
        
        try:
            # Load the saved model for prediction example
            with open('output/simplified_headline_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            
            # Extract features for test headlines
            test_features = trainer.feature_extractor.extract_features(test_headlines)
            
            # FIXED: Ensure all required features are present
            for feature in model_data['features']:
                if feature not in test_features.columns:
                    test_features[feature] = 0
                    
            test_features_selected = test_features[model_data['features']]
            
            # Make predictions
            if model_data['scaler']:
                test_features_scaled = model_data['scaler'].transform(test_features_selected)
                predictions = model_data['model'].predict_proba(test_features_scaled)[:, 1]
            else:
                predictions = model_data['model'].predict_proba(test_features_selected)[:, 1]
            
            for headline, prob in zip(test_headlines, predictions):
                print(f"'{headline}' -> Click Probability: {prob:.4f}")
                
        except Exception as e:
            logging.error(f"Error in prediction example: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()