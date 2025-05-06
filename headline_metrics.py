import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor
import pickle
import re
from transformers import AutoTokenizer, AutoModel
import torch

class HeadlineMetrics:
    def __init__(self, model_path='headline_ctr_model.pkl'):
        """Initialize with a trained model for CTR prediction"""
        self.model = None
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logging.info(f"Loaded headline CTR model from {model_path}")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise ValueError(f"Could not load model from {model_path}: {e}")
            
        # Load embedding model for feature extraction
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.bert_model = AutoModel.from_pretrained("distilbert-base-uncased")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.bert_model = self.bert_model.to(self.device)
        except Exception as e:
            logging.error(f"Failed to load embedding model: {e}")
            raise ValueError(f"Could not load embedding model: {e}")
    
    def extract_features(self, headline):
        """Extract features from headline for model input"""
        features = {}
        
        # Basic features
        features['length'] = len(headline)
        features['word_count'] = len(headline.split())
        features['has_number'] = int(bool(re.search(r'\d', headline)))
        features['num_count'] = len(re.findall(r'\d+', headline))
        features['is_question'] = int(headline.endswith('?'))
        features['has_how_to'] = int('how to' in headline.lower())
        
        # Embedding features
        inputs = self.tokenizer(headline, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        
        # Use the [CLS] token embedding
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        # Add first 10 embedding dimensions as features
        for i in range(10):
            features[f'emb_{i}'] = embedding[i]
        
        return features
    
    def calculate_ctr_score(self, headline):
        """Calculate predicted CTR using ML model"""
        try:
            # Extract features
            features = self.extract_features(headline)
            
            # Convert features to dataframe
            features_df = pd.DataFrame([features])
            
            # Make prediction
            score = float(self.model.predict(features_df)[0])
            
            # Convert to percentage (0-100)
            ctr_percentage = min(max(score * 100, 10), 90)
            
            logging.debug(f"Headline: {headline}")
            logging.debug(f"Model score: {score}, CTR: {ctr_percentage}%")
            
            return score, ctr_percentage / 100.0  # Return as decimal for consistency
        except Exception as e:
            logging.error(f"Error calculating CTR score: {e}")
            return 50.0, 0.5  # Return baseline on error
    
    def compare_headlines(self, original, rewritten):
        """Compare original and rewritten headlines"""
        try:
            original_score, original_ctr = self.calculate_ctr_score(original)
            rewritten_score, rewritten_ctr = self.calculate_ctr_score(rewritten)
            
            improvement = rewritten_ctr - original_ctr
            score_percent_change = (improvement / original_ctr) * 100 if original_ctr > 0 else 0
            
            # Identify key improvements
            key_improvements = self._explain_improvements(original, rewritten)
            
            return {
                'original_score': original_score,
                'rewritten_score': rewritten_score,
                'original_ctr': original_ctr,
                'rewritten_ctr': rewritten_ctr,
                'score_percent_change': score_percent_change,
                'key_improvements': key_improvements,
                'headline_improvement': improvement * 100  # Convert to percentage points
            }
        except Exception as e:
            logging.error(f"Error comparing headlines: {e}")
            return {
                'original_score': 50.0,
                'rewritten_score': 50.0,
                'original_ctr': 0.5,
                'rewritten_ctr': 0.5,
                'score_percent_change': 0,
                'key_improvements': ["Error in comparison"],
                'headline_improvement': 0
            }
    
    def _explain_improvements(self, original, rewritten):
        """Explain what improved in the headline based on features"""
        orig_features = self.extract_features(original)
        new_features = self.extract_features(rewritten)
        
        improvements = []
        
        # Check for specific improvements
        if 40 <= new_features['length'] <= 60 and (orig_features['length'] < 40 or orig_features['length'] > 60):
            improvements.append("Optimized headline length")
            
        if new_features['has_number'] > orig_features['has_number']:
            improvements.append("Added specific numbers")
            
        if new_features['is_question'] > orig_features['is_question']:
            improvements.append("Added question format")
            
        if new_features['has_how_to'] > orig_features['has_how_to']:
            improvements.append("Added 'how to' format")
            
        # If no specific improvements found
        if not improvements and (new_features['rewritten_score'] > orig_features['original_score']):
            improvements.append("General semantic improvement")
            
        return improvements
    
    def get_headline_feedback(self, original, rewritten):
        """Generate detailed feedback on the headline rewrite"""
        comparison = self.compare_headlines(original, rewritten)
        
        feedback = []
        
        if comparison['score_percent_change'] > 0:
            feedback.append(f"✅ Improved CTR by {comparison['score_percent_change']:.1f}%")
        else:
            feedback.append(f"⚠️ Decreased CTR by {abs(comparison['score_percent_change']):.1f}%")
        
        for improvement in comparison['key_improvements']:
            feedback.append(f"• {improvement}")
        
        return "\n".join(feedback)