# Headline Click Prediction Model Report
    Generated: 2025-05-12 22:49

    ## Model Configuration
    - Model Type: XGBClassifier (Binary Classification)
    - Task: Predict whether a headline will be clicked
    - Training Time: 0.20 seconds

    ## Model Performance
    - Training Accuracy : 0.5171
    - Training Precision: 0.9895
    - Training Recall   : 0.0345
    - Training F1 Score : 0.0666
    - Training AUC      : 0.6656
    
    - Validation Accuracy : 0.9622
    - Validation Precision: 0.1667
    - Validation Recall   : 0.0020
    - Validation F1 Score : 0.0039
    - Validation AUC      : 0.5085
    
    ## Dataset Summary
    - Training headlines  : 28648
    - Validation headlines: 13558
    - Test headlines      : 11358
    - Training click rate : 0.1346
    - Validation click rate: 0.0375

    ## Key Feature Importances
    - has_colon: 0.3025
- has_quote: 0.2143
- title_case_ratio: 0.1171
- num_count: 0.1036
- has_controversy: 0.0733
- has_number: 0.0719
- has_date: 0.0395
- has_urgency: 0.0368
- has_positive: 0.0246
- has_question_words: 0.0164

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
    