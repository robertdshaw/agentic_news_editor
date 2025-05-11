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

# Constants
PROCESSED_DATA_DIR = './processed_data'
EMBEDDING_DIMS = 20

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def save_processed_data(train_features, train_ctr, val_features, val_ctr, test_features, test_ctr,
                        base_path=PROCESSED_DATA_DIR):
    """Save processed feature data to pickle files to avoid reprocessing."""
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
    """Load processed feature data from pickle files if they exist."""
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
    _more_tags = {"estimator_type": "regressor"}


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

        self.word_to_index = {}
        common_words = ['the', 'a', 'to', 'how', 'why', 'what', 'when', 'is', 'are',
                        'says', 'report', 'announces', 'reveals', 'show', 'study',
                        'new', 'top', 'best', 'worst', 'first', 'last', 'latest']
        for i, word in enumerate(common_words, 1):
            self.word_to_index[word] = i

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

        if data_type in ['train', 'val']:
            required_cols = ['title', 'newsID', 'ctr']
        else:
            required_cols = ['title', 'newsID']
        missing = [col for col in required_cols if col not in data.columns]
        if missing:
            logging.error(f"Missing required columns in {data_type} data: {missing}")
            return None

        return data

    def extract_features(self, headlines):
        logging.info(f"Extracting features from {len(headlines)} headlines")
        features_list = []
        batch_size = 500
        for i in range(0, len(headlines), batch_size):
            batch = headlines[i:i+batch_size]
            logging.info(f"Processing batch {i//batch_size + 1}/{(len(headlines)-1)//batch_size + 1}")
            for headline in batch:
                f = {}
                f['length'] = len(headline)
                f['word_count'] = len(headline.split())
                f['has_number'] = int(bool(re.search(r'\d', headline)))
                f['num_count'] = len(re.findall(r'\d+', headline))
                f['is_question'] = int(headline.endswith('?') or bool(re.match(r'^(how|what|why|where|when|is) ', headline.lower())))
                f['has_colon'] = int(':' in headline)
                f['has_quote'] = int('"' in headline or "'" in headline)
                f['has_how_to'] = int('how to' in headline.lower())
                f['capital_ratio'] = sum(1 for c in headline if c.isupper()) / len(headline) if headline else 0
                words = headline.split()
                f['first_word_length'] = len(words[0]) if words else 0
                f['last_word_length'] = len(words[-1]) if words else 0
                f['avg_word_length'] = sum(len(w) for w in words) / len(words) if words else 0
                f['has_number_at_start'] = int(bool(re.match(r'^\d+', headline)))
                f['starts_with_digit'] = int(headline[0].isdigit() if headline else 0)
                f['has_date'] = int(bool(re.search(r'\b(20\d\d|19\d\d|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b', headline.lower())))
                f['has_question_words'] = int(bool(re.search(r'\b(how|what|why|when|where|who|which)\b', headline.lower())))
                f['has_suspense'] = int(bool(re.search(r'\b(secret|reveal|shock|stun|surprise|you won\'t believe)\b', headline.lower())))
                f['has_urgency'] = int(bool(re.search(r'\b(breaking|urgent|just in|now|today)\b', headline.lower())))
                f['has_list'] = int(bool(re.search(r'\b(\d+ (?:things|ways|tips|reasons|facts))\b', headline.lower())))
                f['has_positive'] = int(bool(re.search(r'\b(best|top|good|great|amazing|awesome|success)\b', headline.lower())))
                f['has_negative'] = int(bool(re.search(r'\b(worst|bad|terrible|fail|problem|crisis|disaster)\b', headline.lower())))
                f['has_controversy'] = int(bool(re.search(r'\b(vs|versus|against|fight|battle|war|clash)\b', headline.lower())))
                if words:
                    f['first_word'] = self.word_to_index.get(words[0].lower(), 0)
                    f['last_word'] = self.word_to_index.get(words[-1].lower(), 0)
                    f['title_case_words'] = sum(1 for w in words if w and w[0].isupper())
                    f['title_case_ratio'] = f['title_case_words'] / len(words)
                f['length_question_interaction'] = f['length'] * f['is_question']
                f['word_count_list_interaction'] = f['word_count'] * f['has_list']
                try:
                    tokens = self.tokenizer(headline, return_tensors="pt", padding=True, truncation=True, max_length=128)
                    inputs = {k: v.to(self.device) for k, v in tokens.items()}
                    with torch.no_grad():
                        outputs = self.bert_model(**inputs)
                    emb = outputs.last_hidden_state[0, 0, :].cpu().numpy()
                    for j in range(self.embedding_dims):
                        f[f'emb_{j}'] = emb[j]
                except Exception as e:
                    logging.error(f"Embedding error for '{headline}': {e}")
                    for j in range(self.embedding_dims):
                        f[f'emb_{j}'] = 0.0
                features_list.append(f)
        return pd.DataFrame(features_list)

    def train_model(self, train_features, train_ctr, val_features=None, val_ctr=None,
                    output_file='headline_ctr_model.pkl'):
        logging.info("Training headline CTR prediction model")
        if self.use_log_transform:
            train_y = np.log1p(train_ctr)
            val_y = np.log1p(val_ctr) if val_ctr is not None else None
        else:
            train_y = train_ctr
            val_y = val_ctr
        base = SklearnCompatibleXGBRegressor(
            n_estimators=100, learning_rate=0.01, max_depth=4,
            min_child_weight=5, subsample=0.7, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=2.0, objective='reg:squarederror',
            random_state=42, n_jobs=-1
        )
        logging.info("Performing 5-fold CV on base model...")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(base, train_features, train_y, cv=kf, scoring='neg_mean_squared_error')
        rmse = np.sqrt(-scores)
        logging.info(f"Base CV RMSE: {rmse.mean():.4f} ± {rmse.std():.4f}")

        # Feature selection
        fs_model = SklearnCompatibleXGBRegressor(n_estimators=50, learning_rate=0.05, max_depth=5, random_state=42)
        fs_model.fit(train_features, train_y)
        selector = SelectFromModel(fs_model, threshold='median', prefit=True)
        mask = selector.get_support()
        sel_feats = train_features.columns[mask]
        logging.info(f"Selected {len(sel_feats)} features.")
        tf_sel = train_features[sel_feats]
        vf_sel = val_features[sel_feats] if val_features is not None else None

        logging.info("CV on selected features...")
        sel_scores = cross_val_score(base, tf_sel, train_y, cv=kf, scoring='neg_mean_squared_error')
        sel_rmse = np.sqrt(-sel_scores)
        logging.info(f"Selected CV RMSE: {sel_rmse.mean():.4f} ± {sel_rmse.std():.4f}")

        # Hyperparameter tuning
        logging.info("Starting grid search...")
        param_grid = {
            'n_estimators': [100, 200], 'learning_rate': [0.01, 0.05],
            'max_depth': [3, 4, 5], 'min_child_weight': [3, 5, 7],
            'subsample': [0.7, 0.8], 'colsample_bytree': [0.7, 0.8]
        }
        if len(train_features) > 10000:
            param_grid = {'n_estimators': [100], 'learning_rate': [0.01, 0.05], 'max_depth': [4, 5], 'min_child_weight': [5]}
        grid = GridSearchCV(
            estimator=SklearnCompatibleXGBRegressor(objective='reg:squarederror', reg_alpha=1.0, reg_lambda=2.0,
                                                    random_state=42, n_jobs=-1),
            param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1
        )
        try:
            grid.fit(tf_sel, train_y)
            best = grid.best_params_
            logging.info(f"Best params: {best}")
            final = SklearnCompatibleXGBRegressor(objective='reg:squarederror', reg_alpha=1.0,
                                                reg_lambda=2.0, random_state=42, n_jobs=-1, **best)
        except Exception as e:
            logging.warning(f"Grid search failed: {e}")
            final = base

        logging.info("Training final model...")
        start = time.time()
        if vf_sel is not None:
            final.fit(tf_sel, train_y, eval_set=[(vf_sel, val_y)], eval_metric='rmse', early_stopping_rounds=50, verbose=0)
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

        model_data = {
            'model': final,
            'use_log_transform': self.use_log_transform,
            'feature_names': sel_feats.tolist(),
            'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': {'train': train_metrics, 'val': val_metrics, 'cv': (sel_rmse.mean(), sel_rmse.std())}
        }
        with open(os.path.join(self.output_dir, output_file), 'wb') as f:
            pickle.dump(model_data, f)
        logging.info(f"Model saved to {os.path.join(self.output_dir, output_file)}")

        if hasattr(self, 'visualize_feature_importance'):
            self.visualize_feature_importance(fi)
        fi.to_csv(os.path.join(self.output_dir, 'feature_importance.csv'), index=False)
        logging.info("Feature importance saved.")

        return {'model': final, 'selected_features': sel_feats, 'train_metrics': train_metrics,
                'val_metrics': val_metrics, 'cv_rmse': sel_rmse.mean(), 'cv_rmse_std': sel_rmse.std(),
                'feature_importances': fi, 'training_time': ttime}

    def predict(self, features, model_file='headline_ctr_model.pkl'):
        path = os.path.join(self.output_dir, model_file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)
        feat_names = data['feature_names']
        missing = set(feat_names) - set(features.columns)
        if missing:
            logging.warning(f"Missing features: {missing}")
            for m in missing:
                features[m] = 0
        feats = features[feat_names]
        preds_t = data['model'].predict(feats)
        return np.expm1(preds_t) if data['use_log_transform'] else preds_t

    # Visualization methods unchanged...
    
    def run_training_pipeline(self, force_reprocess=False):
        if not force_reprocess:
            tf, ty, vf, vy, testf, testy = load_processed_data()
            if tf is not None:
                logging.info("Using cached data.")
                return self.train_model(tf, ty, vf, vy)
        logging.info("Processing from scratch.")
        train_data = self.load_data('train')
        val_data = self.load_data('val')
        test_data = self.load_data('test')
        if train_data is None or val_data is None:
            logging.error("Missing train/val splits; aborting.")
            return None
        if test_data is None:
            logging.warning("No test data; skipping test predictions.")
        train_data.dropna(subset=['title', 'ctr'], inplace=True)
        val_data.dropna(subset=['title', 'ctr'], inplace=True)
        if test_data is not None:
            test_data.dropna(subset=['title', 'newsID'], inplace=True)

        self.visualize_ctr_distribution(train_data['ctr'].values, val_data['ctr'].values)
        train_feats = self.extract_features(train_data['title'].values)
        val_feats = self.extract_features(val_data['title'].values)
        test_feats = self.extract_features(test_data['title'].values) if test_data is not None else None

        save_processed_data(train_feats, train_data['ctr'].values,
                            val_feats, val_data['ctr'].values,
                            test_feats, test_data['ctr'].values if test_data is not None and 'ctr' in test_data.columns else None)

        result = self.train_model(train_feats, train_data['ctr'].values,
                                   val_feats, val_data['ctr'].values)
        if result and test_data is not None and test_feats is not None:
            logging.info("Generating test predictions...")
            preds = self.predict(test_feats)
            out = test_data[['newsID', 'title']].copy()
            out['predicted_ctr'] = preds
            out.to_csv(os.path.join(self.output_dir, 'test_predictions.csv'), index=False)
            logging.info("Test predictions saved.")
        return result


if __name__ == "__main__":
    args = parse_arguments()
    trainer = HeadlineModelTrainer(use_log_transform=True)
    res = trainer.run_training_pipeline(force_reprocess=args.reprocess)
    if res:
        tm = res['train_metrics']['r2']
        vm = res['val_metrics'].get('r2', None)
        print(f"Training complete. R²: train={tm:.4f}, val={vm:.4f if vm is not None else 'N/A'}")
    else:
        print("Pipeline failed.")
