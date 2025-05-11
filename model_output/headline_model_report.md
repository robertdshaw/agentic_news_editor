# Headline Click Prediction Model Report
    Generated: 2025-05-11 20:55

    ## Model Configuration
    - Model Type: XGBClassifier (Binary Classification)
    - Task: Predict if a headline will be clicked
    - Training Time: 1.25 seconds

    ## Model Performance
    - Training Accuracy: 0.8654
    - Training Precision: 0.0000
    - Training Recall: 0.0000
    - Training F1 Score: 0.0000
    - Training AUC: 0.6623
    
    - Validation Accuracy: 0.9625
    - Validation Precision: 0.0000
    - Validation Recall: 0.0000
    - Validation F1 Score: 0.0000
    - Validation AUC: 0.5993
    
    ## Dataset Summary
    - Training headlines: 28648
    - Validation headlines: 13558
    - Test headlines: 11358
    
    - Training Click Rate: 0.1346
    - Validation Click Rate: 0.0375
    
    ## Key Feature Importances
    - title_case_words: 0.0451
- has_number_at_start: 0.0450
- first_word_length: 0.0382
- has_controversy: 0.0331
- last_word_length: 0.0324
- emb_1: 0.0317
- emb_17: 0.0312
- num_count: 0.0311
- emb_3: 0.0307
- emb_12: 0.0298
- emb_8: 0.0298
- emb_0: 0.0297
- title_case_ratio: 0.0294
- emb_18: 0.0293
- emb_4: 0.0291

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
    