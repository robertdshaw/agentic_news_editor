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
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            roc_auc_score, confusion_matrix)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HeadlineModelTrainer:
    """
    Training function to load data, extract headline features, add semantic features through embedding.
    Use XGBoost regression model to select important features, train, validate and test.
    """
    
    def __init__(self, processed_data_dir='agentic_news_editor/processed_data'):
        """Initialize the headline model trainer"""
        self.processed_data_dir = processed_data_dir
        self.embedding_dims = 20
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        
        # Create output directory for results
        self.output_dir = 'model_output'
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
        Feature selection based on feature importance.
        
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
    
    def evaluate_model_ranking(self, model_data, val_data):
        """
        Evaluate the classifier’s ability to rank headlines by click likelihood.

        Args:
            model_data (dict): Contains 'model' and 'feature_names'.
            val_data (DataFrame): Must have columns ['title', 'ctr'].

        Returns:
            dict: Spearman correlation, per-bucket click rates, and overall metrics.
        """
        logging.info("Evaluating model's headline ranking ability...")

        # 1) Prepare true binary clicks and features
        headlines     = val_data['title'].values
        actual_clicks = (val_data['ctr'].values > 0).astype(int)

        features = self.extract_features(headlines)
        feats_sel = features[model_data['feature_names']]

        # 2) Get predicted click probabilities
        probs = model_data['model'].predict_proba(feats_sel)[:, 1]

        # 3) Build results DataFrame
        df = pd.DataFrame({
            'headline'      : headlines,
            'actual_click'  : actual_clicks,
            'predicted_prob': probs
        })

        # 4) Spearman rank correlation between score and true click flag
        rank_corr = df['predicted_prob'].corr(df['actual_click'], method='spearman')
        logging.info(f"Spearman rank correlation: {rank_corr:.4f}")

        # 5) Bucket into deciles by predicted probability
        df = df.sort_values('predicted_prob', ascending=False).reset_index(drop=True)
        n = len(df)
        bucket_size = max(1, n // 10)
        buckets = []
        for i in range(0, n, bucket_size):
            block = df.iloc[i : i + bucket_size]
            buckets.append({
                'bucket'       : i // bucket_size + 1,
                'size'         : len(block),
                'click_rate'   : block['actual_click'].mean() * 100,
                'mean_prob'    : block['predicted_prob'].mean()
            })
        buckets_df = pd.DataFrame(buckets)

        overall_click_rate = df['actual_click'].mean() * 100
        logging.info(f"Overall click rate: {overall_click_rate:.2f}%")
        for _, row in buckets_df.iterrows():
            logging.info(f"Bucket {int(row['bucket'])}: Click rate = {row['click_rate']:.2f}%")

        # 6) Plot click rate by bucket
        plt.figure()
        plt.bar(buckets_df['bucket'], buckets_df['click_rate'])
        plt.axhline(overall_click_rate, linestyle='--', label=f'Overall: {overall_click_rate:.2f}%')
        plt.xlabel('Bucket (1 = highest predicted probability)')
        plt.ylabel('Click Rate (%)')
        plt.title('Click Rate by Predicted‐Probability Bucket')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'ranking_click_rate_by_bucket.png'), dpi=300)
        plt.close()
        logging.info("Saved ranking plot to ranking_click_rate_by_bucket.png")

        return {
            'spearman_corr'    : rank_corr,
            'overall_click_rate': overall_click_rate,
            'buckets'          : buckets_df
        }

    def train_model(self,
                    train_features: pd.DataFrame,
                    train_ctr: np.ndarray,
                    val_features: pd.DataFrame = None,
                    val_ctr: np.ndarray = None,
                    output_file: str = 'headline_classifier_model.pkl',
                    class_weight: dict = None) -> dict:
        """
        Train a binary click-probability classifier for headlines.

        Returns a dict with the trained model, feature names, metrics, and metadata.
        """
        logging.info("Training headline click-probability classification model")

        # 1) Binary targets
        y_train = (train_ctr > 0).astype(int)
        y_val = (val_ctr > 0).astype(int) if val_ctr is not None else None

        # 2) Feature selection
        selected = self.manual_feature_selection(train_features, y_train, threshold=0.2)
        X_tr = train_features[selected]
        X_val = val_features[selected] if val_features is not None else None

        # 3) Set up classifier params
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
        if class_weight:
            scale_pos = class_weight.get(0,1)/class_weight.get(1,1)
            params['scale_pos_weight'] = scale_pos
            logging.info(f"Using scale_pos_weight={scale_pos}")

        logging.info(f"Classifier params: {params}")
        model = XGBClassifier(**params)

        # 4) Train with early stopping
        start = time.time()
        if X_val is not None and y_val is not None:
            model.fit(
                X_tr, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='auc',
                early_stopping_rounds=10,
                verbose=False
            )
        else:
            model.fit(X_tr, y_train, verbose=False)
        elapsed = time.time() - start
        logging.info(f"Training completed in {elapsed:.2f}s")

        # 5) Compute metrics
        p_tr = model.predict_proba(X_tr)[:, 1]
        y_tr_pred = (p_tr > 0.5).astype(int)
        train_metrics = {
            'accuracy': accuracy_score(y_train, y_tr_pred),
            'precision': precision_score(y_train, y_tr_pred, zero_division=0),
            'recall': recall_score(y_train, y_tr_pred, zero_division=0),
            'f1': f1_score(y_train, y_tr_pred, zero_division=0),
            'auc': roc_auc_score(y_train, p_tr)
        }
        logging.info(f"Train metrics: {train_metrics}")

        val_metrics = {}
        if X_val is not None and y_val is not None:
            p_val = model.predict_proba(X_val)[:, 1]
            y_val_pred = (p_val > 0.5).astype(int)
            val_metrics = {
                'accuracy': accuracy_score(y_val, y_val_pred),
                'precision': precision_score(y_val, y_val_pred, zero_division=0),
                'recall': recall_score(y_val, y_val_pred, zero_division=0),
                'f1': f1_score(y_val, y_val_pred, zero_division=0),
                'auc': roc_auc_score(y_val, p_val)
            }
            logging.info(f"Validation metrics: {val_metrics}")
            self.visualize_classifier_performance(y_val, p_val, output_prefix='validation_performance')

        # 6) Feature importances
        fi = pd.DataFrame({
            'feature': selected,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        logging.info(f"Top features: {fi.head(10).to_dict('records')}")

        # 7) Save model + metadata
        model_data = {
            'model': model,
            'feature_names': selected,
            'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'training_time': elapsed
        }
        with open(os.path.join(self.output_dir, output_file), 'wb') as f:
            pickle.dump(model_data, f)
        logging.info(f"Model saved to {os.path.join(self.output_dir, output_file)}")

        # 8) Save and plot importances
        self.visualize_feature_importance(fi)
        fi.to_csv(os.path.join(self.output_dir, 'feature_importance.csv'), index=False)
        logging.info("Feature importances saved.")

        return model_data
    
    def visualize_click_distribution(self, train_y, val_y=None):
        """
        Bar‐plot of click vs no‐click counts for train (and optional validation).
        """
        # Convert to binary: 0 = no click, 1 = click
        train_counts = np.bincount((train_y > 0).astype(int))
        labels = ['No Click', 'Click']

        # Training
        plt.figure()
        plt.bar(labels, train_counts)
        plt.title('Training Click Distribution')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'click_distribution_train.png'), dpi=300)
        plt.close()

        # Validation (if provided)
        if val_y is not None:
            val_counts = np.bincount((val_y > 0).astype(int))
            plt.figure()
            plt.bar(labels, val_counts)
            plt.title('Validation Click Distribution')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'click_distribution_val.png'), dpi=300)
            plt.close()

        logging.info(f"Click distribution plots saved to {self.output_dir}")

    
    def visualize_feature_importance(self, df, output_file='feature_importance.png'):
        """
        Horizontal bar‐chart of top N feature importances.
        """
        top_n = min(15, len(df))
        top = df.head(top_n)

        plt.figure()
        plt.barh(top['feature'], top['importance'])
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_file), dpi=300)
        plt.close()
        logging.info(f"Feature importance visualization saved to {self.output_dir}/{output_file}")
          
    def visualize_classifier_performance(self, y_true, y_proba, output_prefix='classifier_performance'):
        """
        Create separate visualizations of classifier performance:
        - ROC Curve → {output_prefix}_roc.png
        - Precision-Recall Curve → {output_prefix}_pr.png
        """
        from sklearn.metrics import roc_curve, precision_recall_curve, auc

        # 1) ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{output_prefix}_roc.png'), dpi=300)
        plt.close()

        # 2) Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        baseline = (y_true.sum() / len(y_true))
        plt.figure()
        plt.plot(recall, precision, label=f'AUC = {pr_auc:.3f}')
        plt.plot([0, 1], [baseline, baseline], linestyle='--', label=f'Baseline = {baseline:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision–Recall Curve')
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{output_prefix}_pr.png'), dpi=300)
        plt.close()

        logging.info(f"Classifier performance visuals saved to {self.output_dir}")
        
        
    def run_training_pipeline(self, use_cached_features: bool = True):
        """Run the complete classification pipeline with train/val/test splits."""
        try:
            # 1) Load splits
            train_data = self.load_data('train')
            val_data   = self.load_data('val')
            test_data  = self.load_data('test')

            if train_data is None or val_data is None:
                logging.error("Missing train/val splits. Aborting.")
                return None

            # 2) Drop any rows with missing titles or CTR
            train_data.dropna(subset=['title', 'ctr'], inplace=True)
            val_data.dropna(subset=['title', 'ctr'], inplace=True)
            if test_data is not None:
                test_data.dropna(subset=['title'], inplace=True)

            # 3) Log class distribution
            train_click_rate = (train_data['ctr'] > 0).mean() * 100
            val_click_rate   = (val_data['ctr'] > 0).mean() * 100
            logging.info(f"Train click rate: {train_click_rate:.2f}%")
            logging.info(f"Val   click rate: {val_click_rate:.2f}%")

            # 4) Visualize CTR distribution (optional)
            self.visualize_ctr_distribution(train_data['ctr'].values,
                                            val_data['ctr'].values)

            # 5) Feature extraction (with optional caching)
            logging.info("Extracting features for training set...")
            train_feats = (self.extract_features_cached(train_data['title'].values, 'train')
                        if use_cached_features else
                        self.extract_features(train_data['title'].values))

            logging.info("Extracting features for validation set...")
            val_feats = (self.extract_features_cached(val_data['title'].values, 'val')
                        if use_cached_features else
                        self.extract_features(val_data['title'].values))

            test_feats = None
            if test_data is not None:
                logging.info("Extracting features for test set...")
                test_feats = (self.extract_features_cached(test_data['title'].values, 'test')
                            if use_cached_features else
                            self.extract_features(test_data['title'].values))

            # 6) Compute class weights for imbalanced binary classes
            from sklearn.utils.class_weight import compute_class_weight
            y_train_binary = (train_data['ctr'] > 0).astype(int)
            weights = compute_class_weight('balanced',
                                        classes=np.unique(y_train_binary),
                                        y=y_train_binary)
            class_weight = {cls: w for cls, w in zip(np.unique(y_train_binary), weights)}
            logging.info(f"Using class weights: {class_weight}")

            # 7) Train the classifier
            result = self.train_model(
                train_feats,
                train_data['ctr'].values,
                val_feats,
                val_data['ctr'].values,
                output_file='headline_classifier_model.pkl',
                class_weight=class_weight
            )
            if result is None:
                logging.error("Model training failed.")
                return None

            # 8) Test‐set predictions (click probabilities)
            if test_data is not None and test_feats is not None and 'selected_features' in result:
                logging.info("Generating test‐set click probabilities...")
                feats_sel = test_feats[result['selected_features']]
                probs = result['model'].predict_proba(feats_sel)[:, 1]

                test_out = test_data[['newsID', 'title']].copy()
                test_out['click_probability'] = probs
                out_path = os.path.join(self.output_dir, 'test_predictions.csv')
                test_out.to_csv(out_path, index=False)
                logging.info(f"Test predictions saved to {out_path}")

            # 9) Reports and ranking evaluation
            self.create_model_report(result, train_data, val_data, test_data)
            ranking = self.evaluate_model_ranking(result, val_data)
            if ranking:
                result['ranking_evaluation'] = ranking

            return result

        except Exception as e:
            logging.error(f"Error in training pipeline: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return None

    def create_model_report(self, result, train_data, val_data, test_data=None):
        """
        Create a markdown report about the binary classification model performance.
        """
        if result is None:
            return

        # Pull out metrics
        train_metrics = result['train_metrics']
        val_metrics   = result.get('val_metrics', {})

        # Header and configuration
        report = f"""# Headline Click Prediction Model Report
    Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

    ## Model Configuration
    - Model Type: XGBClassifier (Binary Classification)
    - Task: Predict whether a headline will be clicked
    - Training Time: {result['training_time']:.2f} seconds

    ## Model Performance
    - Training Accuracy : {train_metrics['accuracy']:.4f}
    - Training Precision: {train_metrics['precision']:.4f}
    - Training Recall   : {train_metrics['recall']:.4f}
    - Training F1 Score : {train_metrics['f1']:.4f}
    - Training AUC      : {train_metrics['auc']:.4f}
    """

        if val_metrics:
            report += f"""
    - Validation Accuracy : {val_metrics['accuracy']:.4f}
    - Validation Precision: {val_metrics['precision']:.4f}
    - Validation Recall   : {val_metrics['recall']:.4f}
    - Validation F1 Score : {val_metrics['f1']:.4f}
    - Validation AUC      : {val_metrics['auc']:.4f}
    """

        # Dataset summary
        report += f"""
    ## Dataset Summary
    - Training headlines  : {len(train_data)}
    - Validation headlines: {len(val_data)}
    - Test headlines      : {len(test_data) if test_data is not None else 'N/A'}
    - Training click rate : {(train_data['ctr'] > 0).mean():.4f}
    - Validation click rate: {(val_data['ctr'] > 0).mean():.4f}

    ## Key Feature Importances
    """
        for feat, imp in result['feature_importances'].head(15).itertuples(index=False):
            report += f"- {feat}: {imp:.4f}\n"

        report += """
    ## Usage Guidelines
    This model predicts the probability that a headline will be clicked.
    It returns a score between 0 and 1 indicating click likelihood,
    and can be integrated into automated headline optimization workflows.

    ## Features Used
    - Basic text features: length, word count, punctuation, etc.
    - Semantic features: BERT embeddings to capture meaning

    ## Visualizations
    - feature_importance.png: Top feature importances
    - validation_classifier_performance.png: ROC and precision–recall curves
    """

        # Write the markdown file
        report_path = os.path.join(self.output_dir, 'headline_model_report.md')
        with open(report_path, 'w') as f:
            f.write(report)
        logging.info(f"Model report saved to {report_path}")

        # Print click‐rate summaries to console
        print(f"Clicks percentage in training  : {(train_data['ctr'] > 0).mean() * 100:.2f}%")
        print(f"Clicks percentage in validation: {(val_data['ctr'] > 0).mean() * 100:.2f}%")


    def predict_click_proba(
        self,
        headlines: list[str],
        model_data: dict | None = None,
        model_file: str = 'headline_classifier_model.pkl'
    ) -> np.ndarray:
        """
        Predict click probability for each headline.

        Args:
            headlines (list of str): Headlines to score.
            model_data (dict, optional): Loaded model dict containing
                'model' and 'feature_names'. If None, it will be loaded
                from self.output_dir/model_file.
            model_file (str): Filename (in self.output_dir) of the
                classifier pickle to load if model_data is None.

        Returns:
            np.ndarray: Array of click probabilities in [0,1].
        """
        # 1) Load model_data from disk if not provided
        if model_data is None:
            model_path = os.path.join(self.output_dir, model_file)
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
            except Exception as e:
                logging.error(f"Error loading classifier model: {e}")
                raise

        # 2) Extract features for the input headlines
        features = self.extract_features(headlines)

        # 3) Align to the training feature set
        feature_names = model_data['feature_names']
        for feat in feature_names:
            if feat not in features.columns:
                features[feat] = 0.0
        features = features[feature_names]

        # 4) Return the probability of the positive (clicked) class
        return model_data['model'].predict_proba(features)[:, 1]


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
        
        features_filtered = features[feature_names]
        
        prediction = model_data['model'].predict_proba(features_filtered)[0, 1]
        prediction_label = "Click probability"
        
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
            'prediction_type': 'click_probability',
            'headline_stats': headline_stats,
            'top_contributions': contributions[:10],
            'negative_contributions': [c for c in contributions if c['raw_contribution'] < 0][:5],
            'positive_contributions': [c for c in contributions if c['raw_contribution'] > 0][:5]
        }
        
        print(f"\nHeadline: '{headline}'")
        print(f"{prediction_label}: {prediction:.6f}")
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
            model_data: Model data dictionary
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
             
    # Create trainer
    trainer = HeadlineModelTrainer(
        processed_data_dir=args.data_dir,
    )
    
    print(f"Running training pipeline in classification mode...")
    result = trainer.run_training_pipeline(use_cached_features=args.use_cached_features)
    
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
                
            prediction = trainer.predict_ctr([args.predict], model_data)[0]
            print(f"\nHeadline: '{args.predict}'")
            print(f"Click probability: {prediction:.4f} ({prediction*100:.1f}%)")
            print(f"Interpretation: {'Likely to be clicked' if prediction > 0.5 else 'Unlikely to be clicked'}")   
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
            print(f"\nOriginal headline: '{args.optimize}'")
            print(f"{original_ctr:.6f}")
            
            if original_ctr > 0.5:
                print("Interpretation: Likely to be clicked")
            else:
                print("Interpretation: Unlikely to be clicked")
            
            print("\nOptimized headlines:")
            for i, row in results[~results['is_original']].head(5).iterrows():
                improvement = (row['predicted_ctr'] / original_ctr - 1) * 100
                print(f"{i+1}. '{row['headline']}'")
                print(f"   {row['predicted_ctr']:.6f} ({improvement:.1f}% improvement)")
                
        except Exception as e:
            logging.error(f"Error optimizing headline: {e}")
            import traceback
            logging.error(traceback.format_exc())
            
    # Otherwise, run the training pipeline
    else:
        trainer = HeadlineModelTrainer()
        trainer.run_training_pipeline(output_file='headline_classifier_model.pkl')
        result = trainer.run_training_pipeline()
        
        if result is not None:
            print(f"Model training complete.")
            
            if result:
                print("Training metrics:")
                print(f"  Accuracy : {result['train_metrics']['accuracy']:.4f}")
                print(f"  Precision: {result['train_metrics']['precision']:.4f}")
                print(f"  Recall   : {result['train_metrics']['recall']:.4f}")
                print(f"  F1 Score : {result['train_metrics']['f1']:.4f}")
                print(f"  AUC      : {result['train_metrics']['auc']:.4f}")
                
                if result.get('val_metrics'):
                    print("Validation metrics:")
                    print(f"  Accuracy : {result['val_metrics']['accuracy']:.4f}")
                    print(f"  Precision: {result['val_metrics']['precision']:.4f}")
                    print(f"  Recall   : {result['val_metrics']['recall']:.4f}")
                    print(f"  F1 Score : {result['val_metrics']['f1']:.4f}")
                    print(f"  AUC      : {result['val_metrics']['auc']:.4f}")
                
                print(f"Results saved to {trainer.output_dir}")
            else:
                print("Model training failed.")

if __name__ == "__main__":
    main()