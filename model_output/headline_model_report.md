# Headline Click Prediction Model Report
    Generated: 2025-05-12 21:44

    ## Model Configuration
    - Model Type: XGBClassifier (Binary Classification)
    - Task: Predict whether a headline will be clicked
    - Training Time: 0.20 seconds

    ## Model Performance
    - Training Accuracy : 0.8654
    - Training Precision: 0.0000
    - Training Recall   : 0.0000
    - Training F1 Score : 0.0000
    - Training AUC      : 0.5000
    
    - Validation Accuracy : 0.9625
    - Validation Precision: 0.0000
    - Validation Recall   : 0.0000
    - Validation F1 Score : 0.0000
    - Validation AUC      : 0.5000
    
    ## Dataset Summary
    - Training headlines  : 28648
    - Validation headlines: 13558
    - Test headlines      : 11358
    - Training click rate : 0.1346
    - Validation click rate: 0.0375

    ## Key Feature Importances
    - has_number_at_start: 0.0000
- emb_13: 0.0000
- last_word: 0.0000
- word_count_list_interaction: 0.0000
- emb_3: 0.0000
- emb_7: 0.0000
- emb_6: 0.0000
- title_case_ratio: 0.0000
- emb_1: 0.0000
- emb_8: 0.0000
- emb_18: 0.0000
- has_suspense: 0.0000
- emb_2: 0.0000
- emb_15: 0.0000
- emb_9: 0.0000

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
    