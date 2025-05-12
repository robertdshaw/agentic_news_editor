# Headline Click Prediction Model Report
    Generated: 2025-05-12 07:55

    ## Model Configuration
    - Model Type: XGBClassifier (Binary Classification)
    - Task: Predict if a headline will be clicked
    - Training Time: 1.14 seconds

    ## Model Performance
    - Training Accuracy: 0.6259
    - Training Precision: 0.2113
    - Training Recall: 0.6512
    - Training F1 Score: 0.3191
    - Training AUC: 0.6930
    
    - Validation Accuracy: 0.5895
    - Validation Precision: 0.0481
    - Validation Recall: 0.5295
    - Validation F1 Score: 0.0882
    - Validation AUC: 0.5796
    
    ## Dataset Summary
    - Training headlines: 28648
    - Validation headlines: 13558
    - Test headlines: 11358
    
    - Training Click Rate: 0.1346
    - Validation Click Rate: 0.0375
    
    ## Key Feature Importances
    - num_count: 0.0440
- has_number_at_start: 0.0377
- first_word_length: 0.0344
- emb_0: 0.0333
- last_word_length: 0.0331
- has_quote: 0.0323
- emb_7: 0.0319
- emb_12: 0.0319
- title_case_ratio: 0.0312
- length_question_interaction: 0.0306
- emb_1: 0.0306
- emb_3: 0.0305
- last_word: 0.0303
- emb_2: 0.0301
- has_question_words: 0.0299

    ## Usage Guidelines
    
    This model can be used to predict whether headlines will be clicked.
    It outputs a probability score (0-1) representing the likelihood of a click.
    It can be integrated into a headline optimization workflow for automated
    headline suggestions or ranking.
    
    ## Features Used
    The model uses both basic text features and semantic embeddings:
    - Basic features: length, word count, question marks, numbers, etc.
    - Semantic features: BERT embeddings to capture meaning

    ## Visualizations
    The following visualizations have been generated:
    - feature_importance.png: Importance of different features
    
    - validation_classifier_performance.png: ROC and PR curves for classification performance
    