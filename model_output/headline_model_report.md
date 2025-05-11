# Headline CTR Prediction Model Report
Generated: 2025-05-11 17:53

## Model Configuration
- Model Type: XGBRegressor
- Log Transform CTR: False
- Training Time: 1.95 seconds

## Model Performance
- Training MSE: 0.007324
- Training RMSE: 0.085580
- Training MAE: 0.076589
- Training R-squared: -1.3876

- Validation MSE: 0.006394
- Validation RMSE: 0.079965
- Validation MAE: 0.077603
- Validation R-squared: -7.3902

## Dataset Summary
- Training headlines: 28648
- Validation headlines: 13558
- Test headlines: 11358
- Training CTR range: 0.0000 to 1.0000
- Training Mean CTR: 0.0128
- Validation Mean CTR: 0.0032

## Key Feature Importances
- emb_3: 0.3762
- emb_11: 0.2677
- emb_10: 0.1306
- emb_7: 0.0804
- word_count: 0.0713
- emb_16: 0.0380
- emb_15: 0.0298
- length: 0.0059
- has_urgency: 0.0000
- emb_8: 0.0000
- emb_19: 0.0000
- has_controversy: 0.0000
- emb_0: 0.0000
- first_word_length: 0.0000
- emb_4: 0.0000

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
