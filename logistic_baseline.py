# logistic_baseline.py
import os
import pandas as pd
import numpy as np
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # 1) Load cached features & raw CTRs
    FEATURE_DIR = 'model_output/feature_cache'
    train_feats = pd.read_pickle(os.path.join(FEATURE_DIR, 'train_features.pkl'))
    val_feats   = pd.read_pickle(os.path.join(FEATURE_DIR, 'val_features.pkl'))
    test_feats  = pd.read_pickle(os.path.join(FEATURE_DIR, 'test_features.pkl'))

    train_data = pd.read_csv('agentic_news_editor/processed_data/train_headline_ctr.csv')
    val_data   = pd.read_csv('agentic_news_editor/processed_data/val_headline_ctr.csv')
    test_data  = pd.read_csv('agentic_news_editor/processed_data/test_headline_ctr.csv')

    y_train = (train_data['ctr'] > 0).astype(int)
    y_val   = (val_data['ctr'] > 0).astype(int)
    has_test_ctr = 'ctr' in test_data.columns
    if has_test_ctr:
        y_test = (test_data['ctr'] > 0).astype(int)

    # 2) Fit L1‐penalized logistic to select top 10 features
    selector_lr = LogisticRegression(
        penalty='l1',
        solver='liblinear',
        C=0.1,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    selector_lr.fit(train_feats, y_train)

    selector = SelectFromModel(
        selector_lr,
        prefit=True,
        max_features=10,
        threshold=-np.inf
    )
    keep_feats = train_feats.columns[selector.get_support()].tolist()
    logging.info(f"Selected {len(keep_feats)} features: {keep_feats}")

    # 3) Retrain a plain logistic on those features and eval AUC on validation
    X_tr = train_feats[keep_feats]
    X_va = val_feats[keep_feats]
    X_te = test_feats[keep_feats]

    baseline = LogisticRegression(
        class_weight='balanced',
        solver='liblinear',
        max_iter=1000,
        random_state=42
    )
    baseline.fit(X_tr, y_train)

    p_val = baseline.predict_proba(X_va)[:, 1]
    auc_val = roc_auc_score(y_val, p_val)
    logging.info(f"Logistic baseline VALIDATION AUC = {auc_val:.4f}")

    # 4) (Optional) evaluate on test if both classes present
    if has_test_ctr and len(np.unique(y_test)) == 2:
        p_test = baseline.predict_proba(X_te)[:, 1]
        auc_test = roc_auc_score(y_test, p_test)
        logging.info(f"Logistic baseline TEST AUC       = {auc_test:.4f}")
    else:
        logging.info("Skipping test AUC (need both classes present).")

    # 5) Save your feature list for copy-paste
    with open('selected_logistic_feats.txt', 'w') as f:
        for feat in keep_feats:
            f.write(feat + '\n')
    logging.info("Wrote selected feature names to selected_logistic_feats.txt")

if __name__ == "__main__":
    main()
