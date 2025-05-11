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

class SklearnCompatibleXGBRegressor(XGBRegressor, RegressorMixin):
    """Wrapper to make XGBoost compatible with scikit-learn's cross-validation"""
    
    @classmethod
    def __sklearn_tags__(cls):
        return {"estimator_type": "regressor"}