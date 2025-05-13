import os
import re
import time
import pickle
import logging

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

from xgboost import XGBClassifier, XGBRegressor
from imblearn.combine import SMOTETomek

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, make_scorer
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import precision_recall_curve, auc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class HeadlineModelTrainer:
    def __init__(self,
                 data_path='agentic_news_editor/processed_data/train_headline_ctr.csv',
                 processed_data_dir='agentic_news_editor/processed_data',
                 embedding_dims=20):
        self.data_path = data_path
        self.processed_data_dir = processed_data_dir  # Added this attribute
        self.embedding_dims = embedding_dims
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = 'model_output'
        os.makedirs(self.output_dir, exist_ok=True)

        # common words mapping
        common_words = ['the','a','to','how','why','what','when','is','are',
                        'says','report','announces','reveals','show','study',
                        'new','top','best','worst','first','last','latest']
        self.word_to_index = {w:i+1 for i,w in enumerate(common_words)}

        # Load BERT
        logging.info(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.bert_model = AutoModel.from_pretrained("distilbert-base-uncased").to(self.device)
        
        # Initialize ImprovedHeadlineFeatures - you may need to adjust this based on your implementation
        try:
            from improved_features import ImprovedHeadlineFeatures
            self.improved = ImprovedHeadlineFeatures()
        except ImportError:
            logging.warning("ImprovedHeadlineFeatures not found. Creating a dummy implementation.")
            self.improved = DummyImprovedFeatures()

    def run_training_pipeline(self,
                              test_size: float = 0.3,
                              random_state: int = 42,
                              resample: bool = True,
                              use_separate_validation: bool = False):
        """
        Run the complete training pipeline.
        
        Args:
            test_size: Fraction of data to use for validation if splitting from training data
            random_state: Random seed for reproducibility
            resample: Whether to apply SMOTE to balance training data
            use_separate_validation: If True, loads val_headline_ctr.csv as validation set
        """
        # 1) Load training data
        train_file = os.path.join(self.processed_data_dir, 'train_headline_ctr.csv')
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Training file not found: {train_file}")
        
        train_df = pd.read_csv(train_file)
        train_df = train_df.dropna(subset=['title','ctr'])
        
        # 2) Load validation data or split from training data
        if use_separate_validation:
            val_file = os.path.join(self.processed_data_dir, 'val_headline_ctr.csv')
            if os.path.exists(val_file):
                val_df = pd.read_csv(val_file)
                val_df = val_df.dropna(subset=['title','ctr'])
                
                # Extract features for both sets
                logging.info("Extracting features from training set...")
                X_train = self.extract_features_cached(train_df['title'].tolist(), 'train_features')
                y_train = (train_df['ctr'] > 0).astype(int)
                
                logging.info("Extracting features from validation set...")
                X_val = self.extract_features_cached(val_df['title'].tolist(), 'val_features')
                y_val = (val_df['ctr'] > 0).astype(int)
                
                feat_names = X_train.columns.tolist()
                X_train, X_val = X_train.values, X_val.values
                
                logging.info(f"Train set: {len(train_df)} samples, CTR: {y_train.mean():.2%}")
                logging.info(f"Val set:   {len(val_df)} samples, CTR: {y_val.mean():.2%}")
            else:
                logging.warning(f"Validation file not found: {val_file}. Using train/test split instead.")
                use_separate_validation = False
        
        if not use_separate_validation:
            # Original approach: split from training data
            logging.info("Extracting features from training data...")
            X = self.extract_features_cached(train_df['title'].tolist(), 'train_features')
            y = (train_df['ctr'] > 0).astype(int)
            feat_names = X.columns.tolist()
            X = X.values
            
            logging.info(f"Overall click rate: {y.mean():.2%}")
            
            # Split into train/val
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=test_size,
                stratify=y,
                random_state=random_state
            )
            logging.info(f"Train CTR: {y_train.mean():.2%}, Val CTR: {y_val.mean():.2%}")

        # 3) Balance TRAIN only if requested
        if resample:
            smk = SMOTETomek(random_state=random_state)
            X_train, y_train = smk.fit_resample(X_train, y_train)
            logging.info(f"After SMOTE: Train CTR = {y_train.mean():.2%}")

        # 4) Back to DataFrame
        X_tr_df  = pd.DataFrame(X_train, columns=feat_names)
        X_val_df = pd.DataFrame(X_val, columns=feat_names)

        # 5) Compute class weights
        weights = compute_class_weight('balanced',
                                       classes=np.unique(y_train),
                                       y=y_train)
        class_weight = {cls: w for cls,w in zip(np.unique(y_train), weights)}
        logging.info(f"Class weights: {class_weight}")

        # 6) Train & evaluate
        model_data = self.train_model(
            train_features = X_tr_df,
            train_ctr      = y_train,
            val_features   = X_val_df,
            val_ctr        = y_val,
            class_weight   = class_weight
        )
        return model_data

    def extract_features_cached(self, headlines, cache_name):
        cache_dir = os.path.join(self.output_dir, 'feature_cache')
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f'{cache_name}_features.pkl')

        if os.path.exists(path):
            logging.info(f"Loading cached features: {path}")
            return pd.read_pickle(path)

        feats = self.extract_features(headlines)
        feats.to_pickle(path)
        return feats

    def extract_features(self, headlines):
        """
        Build a combined feature set, handling duplicate columns properly.
        """
        original_feats = []
        for h in headlines:
            f = {}
            # — your existing basic & structural features —
            f['length']          = len(h)
            f['word_count']      = len(h.split())
            f['has_number']      = int(bool(re.search(r'\d', h)))
            f['num_count']       = len(re.findall(r'\d+', h))
            f['is_question']     = int(h.strip().endswith('?'))
            f['has_colon']       = int(':' in h)
            f['has_quote']       = int('"' in h or "'" in h)
            f['capital_ratio']   = sum(1 for c in h if c.isupper()) / max(1, len(h))

            # first/last word index features
            words = h.split()
            f['first_word']      = self.word_to_index.get(words[0].lower(), 0) if words else 0
            f['last_word']       = self.word_to_index.get(words[-1].lower(), 0) if words else 0

            # interaction
            f['len_question_inter'] = f['length'] * f['is_question']

            # — BERT embeddings (first self.embedding_dims dims) —
            try:
                inp = self.tokenizer(h, return_tensors='pt', truncation=True,
                                        padding=True, max_length=128)
                inp = {k: v.to(self.device) for k, v in inp.items()}
                with torch.no_grad():  # Add this for efficiency
                    out = self.bert_model(**inp).last_hidden_state[:, 0, :].cpu().numpy()[0]
                for i in range(self.embedding_dims):
                    f[f'bert_emb_{i}'] = float(out[i])  # Prefix BERT embeddings to avoid conflicts
            except Exception as e:
                logging.warning(f"Error processing headline: {e}")
                for i in range(self.embedding_dims):
                    f[f'bert_emb_{i}'] = 0.0

            original_feats.append(f)

        original_df = pd.DataFrame(original_feats)

        # — now the EDA‐driven features —
        if hasattr(self.improved, 'extract_features'):
            try:
                eda_df = self.improved.extract_features(
                    headlines,
                    categories=None,   # pass a list if you have one
                    abstracts=None     # pass a list if you have one
                )
                
                # Handle duplicate columns by prefixing them
                eda_df = eda_df.add_prefix('eda_')  # Prefix all EDA features
                
                # Check for any remaining duplicates and handle them
                combined = pd.concat(
                    [original_df.reset_index(drop=True),
                     eda_df.reset_index(drop=True)],
                    axis=1
                )
                
                # Final check: if there are still duplicates, handle them
                if combined.columns.duplicated().any():
                    logging.warning("Found duplicate columns after prefixing. Handling...")
                    # Keep only the first occurrence of each column
                    combined = combined.loc[:, ~combined.columns.duplicated()]
                    # Or alternatively, you could rename duplicates:
                    # cols = combined.columns.tolist()
                    # for i, col in enumerate(cols):
                    #     if cols.count(col) > 1:
                    #         suffix = cols[:i].count(col)
                    #         if suffix > 0:
                    #             cols[i] = f"{col}_{suffix}"
                    # combined.columns = cols
                
                return combined
            except Exception as e:
                logging.warning(f"Error with ImprovedHeadlineFeatures: {e}")
                return original_df
        else:
            return original_df

    def manual_feature_selection(self, features, target, threshold=0.2):
        logging.info("Performing manual feature selection…")
        
        # Check for duplicate columns before fitting
        if features.columns.duplicated().any():
            logging.warning("Duplicate columns found in features. Removing duplicates...")
            # Keep only the first occurrence of each column
            features = features.loc[:, ~features.columns.duplicated()]
        
        # Also check for any NaN or inf values that might cause issues
        if features.isnull().any().any():
            logging.warning("NaN values found in features. Filling with 0...")
            features = features.fillna(0)
        
        if np.isinf(features.values).any():
            logging.warning("Infinite values found in features. Replacing with large finite values...")
            features = features.replace([np.inf, -np.inf], [1e10, -1e10])
        
        m = XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=5, random_state=42)
        m.fit(features, target)
        imp = pd.Series(m.feature_importances_, index=features.columns).sort_values(ascending=False)
        cutoff = imp.max() * threshold
        sel = imp[imp >= cutoff].index.tolist()
        logging.info(f"Selected {len(sel)} features from {len(imp)} total")
        return sel

    def train_model(self,
                    train_features: pd.DataFrame,
                    train_ctr: np.ndarray,
                    val_features: pd.DataFrame,
                    val_ctr: np.ndarray,
                    class_weight: dict,
                    output_file: str = 'headline_classifier_model.pkl'
                    ) -> dict:
        """
        Train an XGBClassifier with early stopping, compute metrics,
        and return a model_data dict with everything the pipeline needs.
        """
        # 1) Bin target and select features
        y_train = (train_ctr > 0).astype(int)
        y_val   = (val_ctr   > 0).astype(int)
        selected_features = self.manual_feature_selection(train_features, y_train)
        X_tr = train_features[selected_features]
        X_val= val_features[selected_features]

        # 2) Build classifier
        params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'objective': 'binary:logistic',
            'random_state': 42,
            'n_jobs': -1,
            'scale_pos_weight': class_weight.get(0,1) / class_weight.get(1,1)
        }
        model = XGBClassifier(**params)

        # 3) Train with early stopping
        start = time.time()
        model.fit(
            X_tr, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            early_stopping_rounds=10,
            verbose=False
        )
        training_time = time.time() - start
        logging.info(f"Training completed in {training_time:.2f}s")

        # 4) Train metrics
        p_tr     = model.predict_proba(X_tr)[:,1]
        y_tr_pred= (p_tr > 0.5).astype(int)
        train_metrics = {
            'accuracy':  accuracy_score(y_train, y_tr_pred),
            'precision': precision_score(y_train, y_tr_pred, zero_division=0),
            'recall':    recall_score(y_train, y_tr_pred, zero_division=0),
            'f1':        f1_score(y_train, y_tr_pred, zero_division=0),
            'auc':       roc_auc_score(y_train, p_tr)
        }
        logging.info(f"Train metrics: {train_metrics}")

        # 5) Validation metrics (with best‐F1 threshold)
        p_val     = model.predict_proba(X_val)[:,1]
        prec, rec, thresh = precision_recall_curve(y_val, p_val)
        f1s       = 2 * prec * rec / (prec + rec + 1e-8)
        best_idx  = np.argmax(f1s)
        best_thr  = float(thresh[best_idx]) if best_idx < len(thresh) else 0.5
        y_val_pred= (p_val >= best_thr).astype(int)

        val_metrics = {
            'accuracy':  accuracy_score(y_val, y_val_pred),
            'precision': precision_score(y_val, y_val_pred, zero_division=0),
            'recall':    recall_score(y_val, y_val_pred, zero_division=0),
            'f1':        f1_score(y_val, y_val_pred, zero_division=0),
            'auc':       roc_auc_score(y_val, p_val),
            'best_threshold': best_thr
        }
        logging.info(f"Validation metrics: {val_metrics}")

        # 6) Feature importances
        fi = pd.DataFrame({
            'feature': selected_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        # 7) Pack into model_data and save
        model_data = {
            'model': model,
            'feature_names': selected_features,
            'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'training_time': training_time,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'feature_importances': fi
        }
        with open(os.path.join(self.output_dir, output_file), 'wb') as f:
            pickle.dump(model_data, f)
        logging.info("Model and metadata saved to disk.")

        return model_data


# Dummy class in case ImprovedHeadlineFeatures is not available
class DummyImprovedFeatures:
    def extract_features(self, headlines, categories=None, abstracts=None):
        # Return empty DataFrame with same number of rows
        return pd.DataFrame(index=range(len(headlines)))


def main():
    # You can choose between two approaches:
    
    # Option 1: Use separate validation file (recommended if you have val_headline_ctr.csv)
    trainer = HeadlineModelTrainer()
    result = trainer.run_training_pipeline(
        test_size=0.3,      # This parameter is ignored if use_separate_validation=True
        random_state=42,
        resample=True,
        use_separate_validation=True  # Set to True to use val_headline_ctr.csv
    )
    
    # Option 2: Split training data into train/val (uncomment to use this instead)
    # trainer = HeadlineModelTrainer()
    # result = trainer.run_training_pipeline(
    #     test_size=0.3,      # 30% of training data will be used for validation
    #     random_state=42,
    #     resample=True,
    #     use_separate_validation=False  # Split from training data
    # )
    
    logging.info("Pipeline complete.")
    logging.info(f"Best validation F1 score: {result['val_metrics']['f1']:.3f}")
    logging.info(f"Best validation AUC: {result['val_metrics']['auc']:.3f}")


if __name__ == "__main__":
    main()