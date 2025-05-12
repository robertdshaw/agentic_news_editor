# Headline Click Prediction Model Report
    Generated: 2025-05-12 23:25

    ## Model Configuration
    - Model Type: XGBClassifier (Binary Classification)
    - Task: Predict whether a headline will be clicked
    - Training Time: 0.11 seconds

    ## Model Performance
    - Training Accuracy : 0.5000
    - Training Precision: 0.0000
    - Training Recall   : 0.0000
    - Training F1 Score : 0.0000
    - Training AUC      : 0.6425
    
    - Validation Accuracy : 0.2794
    - Validation Precision: 0.0390
    - Validation Recall   : 0.7717
    - Validation F1 Score : 0.0743
    - Validation AUC      : 0.5048
    
    ## Dataset Summary
    - Training headlines  : 28648
    - Validation headlines: 13558
    - Test headlines      : 11358
    - Training click rate : 0.1346
    - Validation click rate: 0.0375

    ## Key Feature Importances
    - has_colon: 0.3026
- has_quote: 0.2220
- has_controversy: 0.1493
- num_count: 0.1216
- title_case_ratio: 0.1029
- has_number: 0.0903
- has_date: 0.0111
- has_urgency: 0.0000
- has_positive: 0.0000
- has_question_words: 0.0000

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
    