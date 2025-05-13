# Headline CTR Prediction Model Summary

## Best Model: RandomForest

## Performance Metrics:
- AUC: 0.5727
- F1 Score: 0.0936
- Precision: 0.0541
- Recall: 0.3445
- Accuracy: 0.7500

## Features Used: 40 features

## Key EDA Insights Implemented:
- Questions reduce CTR (detected and penalized)
- Numbers have slight negative effect (handled appropriately)
- Category effects incorporated
- Authority and urgency patterns detected

## Usage:
```python
import pickle
with open('simplified_headline_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Use the model...
```
