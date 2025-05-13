# Headline Click Prediction Model Report
    Generated: 2025-05-13 08:55

    ## Model Configuration
    - Model Type: XGBClassifier (Binary Classification)
    - Task: Predict whether a headline will be clicked
    - Training Time: 0.06 seconds

    ## Model Performance
    - Training Accuracy : 0.5000
    - Training Precision: 0.0000
    - Training Recall   : 0.0000
    - Training F1 Score : 0.0000
    - Training AUC      : 0.5604
    
    - Validation Accuracy : 0.2311
    - Validation Precision: 0.0390
    - Validation Recall   : 0.8268
    - Validation F1 Score : 0.0746
    - Validation AUC      : 0.5173
    
    ## Dataset Summary
    - Training headlines  : 28648
    - Validation headlines: 13558
    - Test headlines      : 11358
    - Training click rate : 0.1346
    - Validation click rate: 0.0375

    ## Key Feature Importances
    - has_colon: 0.5450
- has_quote: 0.4550

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
    