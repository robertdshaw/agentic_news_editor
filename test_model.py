import os
import pandas as pd
import numpy as np
import pickle
import logging
import argparse
import re
import time
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
from sklearn.base import RegressorMixin

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_processed_data(train_features, train_ctr, val_features, val_ctr, test_features, test_ctr, base_path='./processed_data'):
    """Save processed feature data to pickle files to avoid reprocessing."""
    # Create directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Save training data
    with open(os.path.join(base_path, 'train_features.pkl'), 'wb') as f:
        pickle.dump(train_features, f)
    with open(os.path.join(base_path, 'train_ctr.pkl'), 'wb') as f:
        pickle.dump(train_ctr, f)
    
    # Save validation data
    with open(os.path.join(base_path, 'val_features.pkl'), 'wb') as f:
        pickle.dump(val_features, f)
    with open(os.path.join(base_path, 'val_ctr.pkl'), 'wb') as f:
        pickle.dump(val_ctr, f)
    
    # Save test data
    with open(os.path.join(base_path, 'test_features.pkl'), 'wb') as f:
        pickle.dump(test_features, f)
    with open(os.path.join(base_path, 'test_ctr.pkl'), 'wb') as f:
        pickle.dump(test_ctr, f)
    
    logging.info(f"Saved processed data to {base_path}")

def load_processed_data(base_path='./processed_data'):
    """Load processed feature data from pickle files if they exist."""
    try:
        # Load training data
        with open(os.path.join(base_path, 'train_features.pkl'), 'rb') as f:
            train_features = pickle.load(f)
        with open(os.path.join(base_path, 'train_ctr.pkl'), 'rb') as f:
            train_ctr = pickle.load(f)
        
        # Load validation data
        with open(os.path.join(base_path, 'val_features.pkl'), 'rb') as f:
            val_features = pickle.load(f)
        with open(os.path.join(base_path, 'val_ctr.pkl'), 'rb') as f:
            val_ctr = pickle.load(f)
        
        # Load test data
        with open(os.path.join(base_path, 'test_features.pkl'), 'rb') as f:
            test_features = pickle.load(f)
        with open(os.path.join(base_path, 'test_ctr.pkl'), 'rb') as f:
            test_ctr = pickle.load(f)
        
        logging.info(f"Loaded processed data from {base_path}")
        return train_features, train_ctr, val_features, val_ctr, test_features, test_ctr
    
    except (FileNotFoundError, EOFError) as e:
        logging.info(f"Could not load processed data: {e}")
        return None, None, None, None, None, None
    
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Headline CTR Prediction Model Training')
    parser.add_argument('--reprocess', action='store_true', help='Force reprocessing of data')
    return parser.parse_args()

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
        """
        Load processed headlines for specified split (train, val, or test)
        
        Parameters:
        -----------
        data_type : str
            Type of data to load ('train', 'val', or 'test')
            
        Returns:
        --------
        pandas.DataFrame or None
            Loaded data or None if loading failed
        """
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
        """
        Extract features from headlines for model training
        
        Parameters:
        -----------
        headlines : list or numpy.ndarray
            List of headline strings
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing extracted features
        """
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
                    
                    # Add embedding dimensions as features
                    for j in range(self.embedding_dims):
                        features[f'emb_{j}'] = embedding[j]
                except Exception as e:
                    logging.error(f"Error extracting embedding for '{headline}': {e}")
                    # Add zero embeddings if failed
                    for j in range(self.embedding_dims):
                        features[f'emb_{j}'] = 0.0
                
                features_list.append(features)
        
        return pd.DataFrame(features_list)
class SklearnCompatibleXGBRegressor(XGBRegressor, RegressorMixin):
    """Wrapper to make XGBoost compatible with scikit-learn's cross-validation"""
    
    @classmethod
    def __sklearn_tags__(cls):
        return {"estimator_type": "regressor"}
    
    def train_model(self, train_features, train_ctr, val_features=None, val_ctr=None, 
                output_file='headline_ctr_model.pkl'):
        """
        Train model with cross-validation and feature selection
        
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
        logging.info(f"Training headline CTR prediction model on {len(train_features)} examples")
        
        # STEP 1: Prepare target variable
        if self.use_log_transform:
            train_ctr_transformed = np.log1p(train_ctr)
            val_ctr_transformed = np.log1p(val_ctr) if val_ctr is not None else None
            logging.info(f"Applied log transformation to CTR values")
        else:
            train_ctr_transformed = train_ctr
            val_ctr_transformed = val_ctr
        
        # STEP 2: Define base model
        base_model = SklearnCompatibleXGBRegressor(
            n_estimators=100,
            learning_rate=0.01,
            max_depth=4,
            min_child_weight=5,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=2.0,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        
        # STEP 3: Cross-validation to evaluate base model
        logging.info("Performing 5-fold cross-validation on base model...")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            base_model, 
            train_features, 
            train_ctr_transformed, 
            cv=kf, 
            scoring='neg_mean_squared_error'
        )
        
        # Convert negative MSE to RMSE
        cv_rmse_scores = np.sqrt(-cv_scores)
        avg_cv_rmse = cv_rmse_scores.mean()
        std_cv_rmse = cv_rmse_scores.std()
        
        logging.info(f"Base model 5-fold CV RMSE: {avg_cv_rmse:.6f} ± {std_cv_rmse:.6f}")
        
        # STEP 4: Feature selection
        logging.info("Performing feature selection...")
        
        # Train a model for feature selection
        selection_model = SklearnCompatibleXGBRegressor(
            n_estimators=50,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
        selection_model.fit(train_features, train_ctr_transformed)
        
        # Select features using the model
        selector = SelectFromModel(selection_model, threshold='median', prefit=True)
        selected_features_mask = selector.get_support()
        selected_features = train_features.columns[selected_features_mask]
        
        logging.info(f"Selected {len(selected_features)} out of {len(train_features.columns)} features")
        logging.info(f"Top selected features: {', '.join(selected_features[:10])}")
        
        # Extract selected features
        train_features_selected = train_features[selected_features]
        val_features_selected = val_features[selected_features] if val_features is not None else None
        
        # STEP 5: Cross-validation with selected features
        logging.info("Performing cross-validation with selected features...")
        cv_scores_selected = cross_val_score(
            base_model, 
            train_features_selected, 
            train_ctr_transformed, 
            cv=kf, 
            scoring='neg_mean_squared_error'
        )
        
        # Convert negative MSE to RMSE
        cv_rmse_scores_selected = np.sqrt(-cv_scores_selected)
        avg_cv_rmse_selected = cv_rmse_scores_selected.mean()
        std_cv_rmse_selected = cv_rmse_scores_selected.std()
        
        logging.info(f"Selected features 5-fold CV RMSE: {avg_cv_rmse_selected:.6f} ± {std_cv_rmse_selected:.6f}")
        
        # STEP 6: Hyperparameter tuning with Grid Search
        logging.info("Performing hyperparameter tuning with grid search...")
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05],
            'max_depth': [3, 4, 5],
            'min_child_weight': [3, 5, 7],
            'subsample': [0.7, 0.8],
            'colsample_bytree': [0.7, 0.8]
        }
        
        # Use smaller param grid if dataset is large (more than 10k samples)
        if len(train_features) > 10000:
            logging.info("Large dataset detected, using reduced parameter grid for faster tuning")
            param_grid = {
                'n_estimators': [100],
                'learning_rate': [0.01, 0.05],
                'max_depth': [4, 5],
                'min_child_weight': [5]
            }
        
        grid_search = GridSearchCV(
            estimator=SklearnCompatibleXGBRegressor(
                objective='reg:squarederror',
                reg_alpha=1.0,
                reg_lambda=2.0,
                random_state=42,
                n_jobs=-1
            ),
            param_grid=param_grid,
            cv=3,  # Use 3-fold CV for faster tuning
            scoring='neg_mean_squared_error',
            verbose=0,
            n_jobs=-1
        )
        
        try:
            grid_search.fit(train_features_selected, train_ctr_transformed)
            best_params = grid_search.best_params_
            logging.info(f"Best parameters from grid search: {best_params}")
            
            # Create final model with best parameters
            final_model = SklearnCompatibleXGBRegressor(
                objective='reg:squarederror',
                reg_alpha=1.0,
                reg_lambda=2.0,
                random_state=42,
                n_jobs=-1,
                **best_params
            )
        except Exception as e:
            logging.warning(f"Grid search failed with error: {str(e)}. Using base model parameters.")
            final_model = base_model
        
        # STEP 7: Train final model on full training set with selected features
        logging.info("Training final model with selected features...")
        start_time = time.time()

        # If we have validation data, use it for early stopping
        if val_features_selected is not None and val_ctr_transformed is not None:
            # For XGBoost 1.7.x, use early_stopping_rounds directly
            final_model.fit(
                train_features_selected,
                train_ctr_transformed,
                eval_set=[(val_features_selected, val_ctr_transformed)],
                eval_metric="rmse",
                early_stopping_rounds=50,
                verbose=0  # Use 0 instead of False for older XGBoost versions
            )
        else:
            # If no validation data, fit without early stopping
            final_model.fit(
                train_features_selected,
                train_ctr_transformed,
                verbose=0  # Use 0 instead of False for older XGBoost versions
            )

        training_time = time.time() - start_time
        logging.info(f"Final model training completed in {training_time:.2f} seconds")
        
        # STEP 8: Evaluate on training set
        train_pred_transformed = final_model.predict(train_features_selected)
        
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
        
        # STEP 9: Evaluate on validation set
        val_mse, val_mae, val_rmse, val_r2 = None, None, None, None
        val_pred = None  # Initialize for visualization later
        
        if val_features_selected is not None and val_ctr is not None:
            val_pred_transformed = final_model.predict(val_features_selected)
            
            # Convert predictions back to original scale for log transformation 
            if self.use_log_transform:
                val_pred = np.expm1(val_pred_transformed)
            else:
                val_pred = val_pred_transformed
                
            val_mse = mean_squared_error(val_ctr, val_pred)
            val_mae = mean_absolute_error(val_ctr, val_pred)
            val_rmse = np.sqrt(val_mse)
            val_r2 = r2_score(val_ctr, val_pred)
            
            logging.info(f"Validation metrics - MSE: {val_mse:.6f}, RMSE: {val_rmse:.6f}, MAE: {val_mae:.6f}, R²: {val_r2:.4f}")
            
            # Visualize predictions
            if val_pred is not None and hasattr(self, 'visualize_predictions'):
                self.visualize_predictions(val_ctr, val_pred, 'validation_predictions.png')
        
        # STEP 10: Feature importance of final model
        feature_importances = pd.DataFrame({
            'feature': selected_features,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logging.info("Top 10 important features in final model:")
        for i, row in feature_importances.head(10).iterrows():
            logging.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # STEP 11: Save model with selected features list and metadata
        model_data = {
            'model': final_model,
            'use_log_transform': self.use_log_transform,
            'feature_names': selected_features.tolist(),
            'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': {
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'cv_rmse': avg_cv_rmse_selected
            }
        }
        
        os.makedirs(os.path.dirname(os.path.join(self.output_dir, output_file)), exist_ok=True)
        with open(os.path.join(self.output_dir, output_file), 'wb') as f:
            pickle.dump(model_data, f)
        
        logging.info(f"Model saved to {os.path.join(self.output_dir, output_file)}")
        
        # Visualize feature importance
        if hasattr(self, 'visualize_feature_importance'):
            self.visualize_feature_importance(feature_importances)
        
        # STEP 12: Save feature importance separately for easy access
        feature_importance_file = os.path.join(self.output_dir, 'feature_importance.csv')
        feature_importances.to_csv(feature_importance_file, index=False)
        logging.info(f"Feature importance saved to {feature_importance_file}")
        
        # Return results dictionary with additional info for test predictions
        return {
            'model': final_model,
            'selected_features': selected_features,
            'train_mse': train_mse,
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'val_mse': val_mse,
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'val_r2': val_r2,
            'cv_rmse': avg_cv_rmse_selected,
            'cv_rmse_std': std_cv_rmse_selected,
            'feature_importances': feature_importances,
            'training_time': training_time
        }
    
    def predict(self, features, model_file='headline_ctr_model.pkl'):
        """
        Make predictions using a trained model
        
        Parameters:
        -----------
        features : pandas DataFrame
            Features to predict on
        model_file : str, optional
            Filename of the saved model
            
        Returns:
        --------
        numpy array
            Predicted CTR values
        """
        # Load model
        model_path = os.path.join(self.output_dir, model_file)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        feature_names = model_data['feature_names']
        use_log_transform = model_data['use_log_transform']
        
        # Check if all required features are present
        missing_features = set(feature_names) - set(features.columns)
        if missing_features:
            logging.warning(f"Missing features in input data: {missing_features}")
            # Create missing features with zeros
            for feat in missing_features:
                features[feat] = 0
        
        # Select only the features used during training
        features_selected = features[feature_names]
        
        # Make predictions
        predictions_transformed = model.predict(features_selected)
        
        # Convert predictions back to original scale if log transform was used
        if use_log_transform:
            predictions = np.expm1(predictions_transformed)
        else:
            predictions = predictions_transformed
        
        return predictions
    
    def visualize_feature_importance(self, feature_importances, top_n=15, output_file='feature_importance.png'):
        """
        Create and save feature importance visualization
        
        Parameters:
        -----------
        feature_importances : pandas DataFrame
            DataFrame with 'feature' and 'importance' columns
        top_n : int, optional
            Number of top features to visualize
        output_file : str, optional
            Filename to save the visualization
        """
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importances.head(top_n), palette='viridis')
        plt.title(f'Top {top_n} Feature Importances for CTR Prediction', fontsize=14)
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_file), dpi=300)
        plt.close()
        logging.info(f"Feature importance visualization saved to {os.path.join(self.output_dir, output_file)}")
    
    def visualize_predictions(self, true_values, predicted_values, output_file='prediction_scatter.png'):
        """
        Create and save scatter plot of true vs predicted values
        
        Parameters:
        -----------
        true_values : numpy array or pandas Series
            Actual values
        predicted_values : numpy array or pandas Series
            Predicted values
        output_file : str, optional
            Filename to save the visualization
        """
        plt.figure(figsize=(10, 8))
        
        # Plot scatter plot with alpha for density visualization
        plt.scatter(true_values, predicted_values, alpha=0.5, color='blue')
        
        # Add diagonal line (perfect predictions)
        max_val = max(np.max(true_values), np.max(predicted_values))
        min_val = min(np.min(true_values), np.min(predicted_values))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # Calculate R-squared
        r2 = r2_score(true_values, predicted_values)
        
        plt.title(f'True vs Predicted CTR Values (R² = {r2:.4f})', fontsize=14)
        plt.xlabel('True CTR', fontsize=12)
        plt.ylabel('Predicted CTR', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Try to add density information if scipy is available
        try:
            from scipy.stats import gaussian_kde
            
            # Calculate point density
            xy = np.vstack([true_values, predicted_values])
            density = gaussian_kde(xy)(xy)
            
            # Sort points by density
            idx = density.argsort()
            x, y, z = np.array(true_values)[idx], np.array(predicted_values)[idx], density[idx]
            
            plt.figure(figsize=(10, 8))
            plt.scatter(x, y, c=z, s=30, alpha=0.6, cmap='viridis')
            plt.colorbar(label='Density')
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            plt.title(f'True vs Predicted CTR Values with Density (R² = {r2:.4f})', fontsize=14)
            plt.xlabel('True CTR', fontsize=12)
            plt.ylabel('Predicted CTR', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'density_{output_file}'), dpi=300)
            plt.close()
        except Exception:
            # Continue if density visualization fails
            pass
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_file), dpi=300)
        plt.close()
        logging.info(f"Prediction visualization saved to {os.path.join(self.output_dir, output_file)}")
    
    def visualize_ctr_distribution(self, train_ctr, val_ctr=None, output_file='ctr_distribution.png'):
        """
        Create and save CTR distribution visualization
        
        Parameters:
        -----------
        train_ctr : numpy array or pandas Series
            Training CTR values
        val_ctr : numpy array or pandas Series, optional
            Validation CTR values
        output_file : str, optional
            Filename to save the visualization
        """
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
    
    def run_training_pipeline(self, force_reprocess=False):
        """
        Run the complete model training pipeline with train/val/test splits
        
        Parameters:
        -----------
        force_reprocess : bool, optional (default=False)
            If True, reprocess data from scratch even if cached data exists
        
        Returns:
        --------
        dict or None
            Results dictionary or None if training failed
        """
        # Try to load processed features from pickle files if not forcing reprocessing
        if not force_reprocess:
            train_features, train_ctr, val_features, val_ctr, test_features, test_ctr = load_processed_data()
            
            # If data was successfully loaded
            if train_features is not None:
                logging.info("Using cached processed feature data")
                
                # Train model using loaded features
                result = self.train_model(
                    train_features, train_ctr,
                    val_features, val_ctr,
                    output_file='headline_ctr_model.pkl'
                )
                
                return result
        
        # If we need to process from scratch (either force_reprocess=True or no cached data)
        logging.info("Processing data from scratch...")
        
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
        train_ctr = train_data['ctr'].values
        
        logging.info("Extracting features for validation data...")
        val_features = self.extract_features(val_data['title'].values)
        val_ctr = val_data['ctr'].values
        
        test_features = None
        test_ctr = None
        if test_data is not None:
            logging.info("Extracting features for test data...")
            test_features = self.extract_features(test_data['title'].values)
            # If test data has 'ctr' column
            if 'ctr' in test_data.columns:
                test_ctr = test_data['ctr'].values
        
        # Save the processed features for future runs
        save_processed_data(train_features, train_ctr, val_features, val_ctr, test_features, test_ctr)
        
        # Train model using proper splits
        result = self.train_model(
            train_features, train_ctr,
            val_features, val_ctr,
            output_file='headline_ctr_model.pkl'
        )
        
        # Visualize predictions for validation set
        if result is not None:  # Check if training was successful
            if self.use_log_transform:
                val_pred = np.expm1(result['model'].predict(val_features[result['selected_features']]))
            else:
                val_pred = result['model'].predict(val_features[result['selected_features']])
                
            self.visualize_predictions(val_ctr, val_pred, 'validation_predictions.png')
            
            # If test data is available, generate predictions
            if test_data is not None and test_features is not None:
                logging.info("Generating predictions for test set...")
                
                if self.use_log_transform:
                    test_pred_transformed = result['model'].predict(test_features[result['selected_features']])
                    test_predictions = np.expm1(test_pred_transformed)
                else:
                    test_predictions = result['model'].predict(test_features[result['selected_features']])
                
                # Save test predictions
                test_results = test_data[['newsID', 'title']].copy()
                test_results['predicted_ctr'] = test_predictions
                test_results.to_csv(os.path.join(self.output_dir, 'test_predictions.csv'), index=False)
                logging.info(f"Test predictions saved to {os.path.join(self.output_dir, 'test_predictions.csv')}")
            
            # Create a report
            self.create_model_report(result, train_data, val_data, test_data)
        
        return result
    
    def create_model_report(self, result, train_data, val_data, test_data=None):
        """
        Create a markdown report about the model performance
        
        Parameters:
        -----------
        result : dict
            Dictionary containing model results
        train_data : pandas DataFrame
            Training data
        val_data : pandas DataFrame
            Validation data
        test_data : pandas DataFrame, optional
            Test data
        """
        if result is None:
            return
        
        report = f"""# Headline CTR Prediction Model Report
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

## Model Configuration
- Model Type: XGBoost Regressor
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
## Cross-Validation Results
- 5-fold CV RMSE: {result['cv_rmse']:.6f} ± {result['cv_rmse_std']:.6f}

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
## Selected Features
The model used a subset of features after performing feature selection to improve performance.

## Model Hyperparameters
The final model was trained with optimal hyperparameters found through grid search.

## Usage Guidelines
This model can be used to predict the expected CTR of news headlines.
It can be integrated into a headline optimization workflow for automated
headline suggestions or ranking.

## Features Used
The model uses both basic text features and semantic embeddings:
- Basic features: length, word count, question marks, numbers, etc.
- Semantic features: DistilBERT embeddings to capture meaning

## Visualizations
The following visualizations have been generated:
- feature_importance.png: Importance of different features
- ctr_distribution.png: Distribution of CTR values
- validation_predictions.png: True vs predicted CTR values
"""
        
        # Create report directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        with open(os.path.join(self.output_dir, 'headline_model_report.md'), 'w') as f:
            f.write(report)
        
        logging.info(f"Model report saved to {os.path.join(self.output_dir, 'headline_model_report.md')}")


if __name__ == "__main__":
    """
    Main entry point for running the headline CTR model training pipeline
    """
    
    # Debug print statements
    print(f"Methods available in HeadlineModelTrainer: {[method for method in dir(HeadlineModelTrainer) if not method.startswith('_')]}")
    print(f"Methods available in SklearnCompatibleXGBRegressor: {[method for method in dir(SklearnCompatibleXGBRegressor) if not method.startswith('_')]}")
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Create trainer (set use_log_transform=True for log transformation of CTR values)
    trainer = HeadlineModelTrainer(use_log_transform=True)
    
    # More debug information
    print(f"Methods available in trainer instance: {[method for method in dir(trainer) if not method.startswith('_')]}")
    
    # Run the complete training pipeline
    result = trainer.run_training_pipeline(force_reprocess=args.reprocess)
    
    if result is not None:
        print(f"Model training complete. Training R-squared: {result['train_r2']:.4f}, Validation R-squared: {result['val_r2']:.4f}")
        print(f"Model saved to {os.path.join(trainer.output_dir, 'headline_ctr_model.pkl')}")
    else:
        print("Model training failed.")