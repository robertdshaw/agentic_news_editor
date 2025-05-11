# Headline CTR Prediction Model Report
Generated: 2025-05-09 16:59

## Model Configuration
- Model Type: RandomForestRegressor
- Log Transform CTR: False
- Training Time: 0.00 seconds

## Model Performance
- Training MSE: 0.002064
- Training RMSE: 0.045430
- Training MAE: 0.019428
- Training R-squared: 0.3272

- Validation MSE: 0.000981
- Validation RMSE: 0.031319
- Validation MAE: 0.014999
- Validation R-squared: -0.2870

## Dataset Summary
- Training headlines: 28648
- Validation headlines: 13558
- Test headlines: 11358
- Training CTR range: 0.0000 to 1.0000
- Training Mean CTR: 0.0128
- Validation Mean CTR: 0.0032

## Key Feature Importances
- has_how_to: 0.0379
- emb_11: 0.0337
- emb_2: 0.0333
- has_quote: 0.0316
- has_controversy: 0.0298
- has_colon: 0.0277
- emb_7: 0.0272
- has_number_at_start: 0.0271
- title_case_words: 0.0257
- emb_6: 0.0250
- has_positive: 0.0244
- first_word_length: 0.0243
- length: 0.0241
- emb_14: 0.0240
- emb_12: 0.0234

## Usage Guidelines
This model can be used to predict the expected CTR of news headlines.
It can be integrated into a headline optimization workflow for automated
headline suggestions or ranking.

## Features Used
The model uses both basic text features and semantic embeddings:
- Basic features: length, word count, question marks, numbers, etc.
- Semantic features: BERT embeddings to capture meaning

## Visualizations
The following visualizations have been generated:
- feature_importance.png: Importance of different features
- ctr_distribution.png: Distribution of CTR values
- validation_predictions.png: True vs predicted CTR values
