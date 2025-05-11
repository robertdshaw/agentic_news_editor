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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor

# Constants
PROCESSED_DATA_DIR = './processed_data'
EMBEDDING_DIMS = 20

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def save_processed_data(train_features, train_ctr, val_features, val_ctr, test_features, test_ctr,
                        base_path=PROCESSED_DATA_DIR):
    os.makedirs(base_path, exist_ok=True)
    with open(os.path.join(base_path, 'train_features.pkl'), 'wb') as f:
        pickle.dump(train_features, f)
    with open(os.path.join(base_path, 'train_ctr.pkl'), 'wb') as f:
        pickle.dump(train_ctr, f)
    with open(os.path.join(base_path, 'val_features.pkl'), 'wb') as f:
        pickle.dump(val_features, f)
    with open(os.path.join(base_path, 'val_ctr.pkl'), 'wb') as f:
        pickle.dump(val_ctr, f)
    with open(os.path.join(base_path, 'test_features.pkl'), 'wb') as f:
        pickle.dump(test_features, f)
    with open(os.path.join(base_path, 'test_ctr.pkl'), 'wb') as f:
        pickle.dump(test_ctr, f)
    logging.info(f"Saved processed data to {base_path}")


def load_processed_data(base_path=PROCESSED_DATA_DIR):
    try:
        with open(os.path.join(base_path, 'train_features.pkl'), 'rb') as f:
            train_features = pickle.load(f)
        with open(os.path.join(base_path, 'train_ctr.pkl'), 'rb') as f:
            train_ctr = pickle.load(f)
        with open(os.path.join(base_path, 'val_features.pkl'), 'rb') as f:
            val_features = pickle.load(f)
        with open(os.path.join(base_path, 'val_ctr.pkl'), 'rb') as f:
            val_ctr = pickle.load(f)
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
    parser = argparse.ArgumentParser(description='Headline CTR Prediction Model Training')
    parser.add_argument('--reprocess', action='store_true', help='Force reprocessing of data')
    return parser.parse_args()


class SklearnCompatibleXGBRegressor(XGBRegressor):
    # Instruct sklearn that this is a regressor
    _more_tags = {"estimator_type": "regressor"}

    def __sklearn_tags__(self):
        # Bypass BaseEstimator.__sklearn_tags__ entirely
        return {"estimator_type": "regressor"}


class HeadlineModelTrainer:
    """
    Trains and evaluates a model for predicting headline CTR based on processed splits.
    """

    def __init__(self, processed_data_dir=PROCESSED_DATA_DIR, use_log_transform=True):
        self.processed_data_dir = processed_data_dir
        self.embedding_dims = EMBEDDING_DIMS
        self.use_log_transform = use_log_transform
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        logging.info(f"Log transform for CTR: {self.use_log_transform}")

        self.output_dir = 'model_output'
        os.makedirs(self.output_dir, exist_ok=True)

        # Common word mapping
        common_words = ['the', 'a', 'to', 'how', 'why', 'what', 'when', 'is', 'are',
                        'says', 'report', 'announces', 'reveals', 'show', 'study',
                        'new', 'top', 'best', 'worst', 'first', 'last', 'latest']
        self.word_to_index = {w: i+1 for i, w in enumerate(common_words)}

        # Load BERT
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.bert_model = AutoModel.from_pretrained("distilbert-base-uncased").to(self.device)
            logging.info("Loaded DistilBERT model successfully")
        except Exception as e:
            logging.error(f"Failed to load DistilBERT model: {e}")
            raise

    def load_data(self, data_type='train'):
        file_path = os.path.join(self.processed_data_dir, f'{data_type}_headline_ctr.csv')
        if not os.path.exists(file_path):
            logging.error(f"{data_type.capitalize()} data not found at {file_path}")
            return None
        data = pd.read_csv(file_path)
        logging.info(f"Loaded {len(data)} headlines from {data_type} set")

        # Ensure CTR is present for train/val
        required = ['title', 'newsID'] + (['ctr'] if data_type in ('train', 'val') else [])
        missing = [c for c in required if c not in data.columns]
        if missing:
            logging.error(f"Missing columns in {data_type} data: {missing}")
            return None
        return data

    def extract_features(self, headlines):
        logging.info(f"Extracting features from {len(headlines)} headlines")
        features = []
        batch_size = 500
        for i in range(0, len(headlines), batch_size):
            batch = headlines[i:i+batch_size]
            for h in batch:
                f = {
                    'length': len(h),
                    'word_count': len(h.split()),
                    'has_number': int(bool(re.search(r'\d', h))),
                    'num_count': len(re.findall(r'\d+', h)),
                    'is_question': int(h.endswith('?') or bool(re.match(r'^(how|what|why|where|when|is) ', h.lower()))),
                    'has_colon': int(':' in h),
                    'has_quote': int('"' in h or "'" in h),
                    'has_how_to': int('how to' in h.lower()),
                    'capital_ratio': sum(c.isupper() for c in h) / len(h) if h else 0
                }
                words = h.split()
                f.update({
                    'first_word_length': len(words[0]) if words else 0,
                    'last_word_length': len(words[-1]) if words else 0,
                    'avg_word_length': sum(len(w) for w in words)/len(words) if words else 0,
                    'has_date': int(bool(re.search(r'\b(20\d\d|19\d\d|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b', h.lower()))),
                    'has_list': int(bool(re.search(r'\b(\d+ (?:things|ways|tips|reasons|facts))\b', h.lower())))
                })
                if words:
                    f['first_word'] = self.word_to_index.get(words[0].lower(), 0)
                    f['last_word']  = self.word_to_index.get(words[-1].lower(), 0)
                # BERT embedding
                try:
                    tokens = self.tokenizer(h, return_tensors="pt", padding=True, truncation=True, max_length=128)
                    tokens = {k: v.to(self.device) for k, v in tokens.items()}
                    with torch.no_grad():
                        out = self.bert_model(**tokens)
                    emb = out.last_hidden_state[0,0].cpu().numpy()
                    for j in range(self.embedding_dims):
                        f[f'emb_{j}'] = emb[j]
                except Exception:
                    for j in range(self.embedding_dims):
                        f[f'emb_{j}'] = 0.0
                features.append(f)
        return pd.DataFrame(features)

    def train_model(self, X_train, y_train, X_val=None, y_val=None):
        if self.use_log_transform:
            y_train_t = np.log1p(y_train)
            y_val_t   = np.log1p(y_val) if y_val is not None else None
        else:
            y_train_t, y_val_t = y_train, y_val

        base = SklearnCompatibleXGBRegressor(
            n_estimators=100, learning_rate=0.01, max_depth=4,
            min_child_weight=5, subsample=0.7, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=2.0,
            objective='reg:squarederror', random_state=42, n_jobs=-1
        )

        logging.info("Performing 5-fold CV on base model...")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(base, X_train, y_train_t, cv=kf, scoring='neg_mean_squared_error')
        rmse = np.sqrt(-scores)
        logging.info(f"Base CV RMSE: {rmse.mean():.4f} ± {rmse.std():.4f}")

        # ... rest of training pipeline (feature selection, grid search, final fit) ...

        return {
            'base_cv_rmse_mean': rmse.mean(),
            'base_cv_rmse_std':  rmse.std(),
            # ... other metrics ...
        }

    def run_training_pipeline(self, force_reprocess=False):
        if not force_reprocess:
            Xtr, ytr, Xv, yv, Xt, yt = load_processed_data()
            if Xtr is not None:
                logging.info("Using cached data.")
                return self.train_model(Xtr, ytr, Xv, yv)

        logging.info("Processing data from scratch...")
        train = self.load_data('train')
        val   = self.load_data('val')
        # ... load test if needed ...

        # Drop NA, extract features, save, then:
        Xtr = self.extract_features(train['title'].values)
        ytr = train['ctr'].values
        Xv  = self.extract_features(val['title'].values)
        yv  = val['ctr'].values

        save_processed_data(Xtr, ytr, Xv, yv, None, None)
        return self.train_model(Xtr, ytr, Xv, yv)


if __name__ == "__main__":
    args    = parse_arguments()
    trainer = HeadlineModelTrainer(use_log_transform=True)
    result  = trainer.run_training_pipeline(force_reprocess=args.reprocess)
    if result:
        print(f"5-fold CV RMSE: {result['base_cv_rmse_mean']:.4f} ± {result['base_cv_rmse_std']:.4f}")
    else:
        print("Training failed.")
