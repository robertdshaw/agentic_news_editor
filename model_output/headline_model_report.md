# Headline Click Prediction Model Report
    Generated: 2025-05-12 22:16

    ## Model Configuration
    - Model Type: XGBClassifier (Binary Classification)
    - Task: Predict whether a headline will be clicked
    - Training Time: 1.78 seconds

    ## Model Performance
    - Training Accuracy : 0.5127
    - Training Precision: 1.0000
    - Training Recall   : 0.0254
    - Training F1 Score : 0.0496
    - Training AUC      : 0.7356
    
    - Validation Accuracy : 0.9625
    - Validation Precision: 0.0000
    - Validation Recall   : 0.0000
    - Validation F1 Score : 0.0000
    - Validation AUC      : 0.5333
    
    ## Dataset Summary
    - Training headlines  : 28648
    - Validation headlines: 13558
    - Test headlines      : 11358
    - Training click rate : 0.1346
    - Validation click rate: 0.0375

    ## Key Feature Importances
    - title_case_ratio: 0.1284
- has_quote: 0.1240
- has_colon: 0.0952
- num_count: 0.0683
- has_controversy: 0.0605
- has_date: 0.0473
- emb_7: 0.0348
- has_number: 0.0334
- emb_15: 0.0309
- emb_17: 0.0284
- first_word_length: 0.0266
- emb_4: 0.0253
- has_question_words: 0.0237
- emb_19: 0.0200
- emb_13: 0.0199

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
    