import pandas as pd
import numpy as np
import logging
import pickle
import re
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from pathlib import Path

class HeadlineMetrics:
    def __init__(self, model_path='model_output/ctr_model.pkl', client=None):
        """Initialize with the CTR prediction model"""
        self.openai_client = client
        
        # Load the CTR model and its components
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.imputer = model_data['imputer']
            self.encoders = model_data['encoders']
            self.feature_names = model_data['feature_names']
            self.best_model = model_data['model_name']
            self.thresholds = model_data.get('thresholds', [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3])
            
            logging.info(f"Loaded CTR model ({self.best_model}) from {model_path}")
        except Exception as e:
            logging.error(f"Failed to load CTR model: {e}")
            raise ValueError(f"Could not load model from {model_path}: {e}")
        
        # Load embedding model (same as used in training)
        try:
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            logging.info("Loaded sentence transformer model")
        except Exception as e:
            logging.error(f"Failed to load sentence transformer: {e}")
            raise ValueError(f"Could not load sentence transformer: {e}")
    
    def extract_features(self, df_or_headline):
        """Extract features using the same method as the CTR model training"""
        # Handle single headline or dataframe
        if isinstance(df_or_headline, str):
            df = pd.DataFrame({'title': [df_or_headline]})
        else:
            df = df_or_headline.copy()
        
        features = pd.DataFrame(index=df.index)
        
        # Text features from title
        features['title_length'] = df['title'].str.len()
        features['title_word_count'] = df['title'].str.split().str.len()
        
        # Pattern features
        features['has_question'] = df['title'].str.contains(r'\?').astype(int)
        features['has_number'] = df['title'].str.contains(r'\d').astype(int)
        features['has_colon'] = df['title'].str.contains(':').astype(int)
        features['has_quotes'] = df['title'].str.contains(r'["\']').astype(int)
        features['has_ellipsis'] = df['title'].str.contains(r'\.\.\.').astype(int)
        
        # Problematic patterns
        features['starts_with_number'] = df['title'].str.match(r'^\d').astype(int)
        features['starts_with_question'] = df['title'].str.match(r'^(Is|What|How|Why|When|Where)').astype(int)
        features['has_superlative'] = df['title'].str.contains(r'\b(?:best|worst|most|least|biggest|smallest)\b', case=False).astype(int)
        features['title_uppercase_ratio'] = df['title'].apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0)
        
        # Reading ease (simplified version)
        def safe_flesch_reading_ease(text):
            try:
                if pd.isna(text) or not isinstance(text, str) or len(text) == 0:
                    return 50.0  # Default reading ease
                # Simplified calculation
                words = len(text.split())
                sentences = max(1, text.count('.') + text.count('!') + text.count('?'))
                return max(0, min(100, 206.835 - 1.015 * (words / sentences) - 84.6 * (words / words)))
            except:
                return 50.0
        
        features['title_reading_ease'] = df['title'].apply(safe_flesch_reading_ease)
        
        # Category encoding (use 'unknown' as default)
        features['category_encoded'] = 0  # Default category
        
        # Time features (use defaults)
        features['hour'] = 12
        features['day_of_week'] = 1
        features['is_weekend'] = 0
        
        # Embeddings
        try:
            titles_clean = df['title'].fillna('').astype(str).tolist()
            embeddings = self.embeddings_model.encode(titles_clean, batch_size=32)
            
            # Add first 50 embedding dimensions as features
            for i in range(min(50, embeddings.shape[1])):
                features[f'emb_{i}'] = embeddings[:, i]
        except Exception as e:
            logging.error(f"Error creating embeddings: {e}")
            # Add zero embeddings if failed
            for i in range(50):
                features[f'emb_{i}'] = 0.0
        
        # Fill any remaining NaN values
        features = features.fillna(0)
        
        return features
    
    def predict_ctr(self, headline):
        """Predict CTR for a single headline"""
        try:
            # Extract features
            features = self.extract_features(headline)
            
            # Ensure all required features are present
            for feat in self.feature_names:
                if feat not in features.columns:
                    features[feat] = 0
            
            # Select and order features as expected by the model
            X = features[self.feature_names]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Handle any NaN values
            if np.isnan(X_scaled).any():
                X_scaled = self.imputer.transform(X_scaled)
            
            # Predict
            probability = self.model.predict_proba(X_scaled)[0, 1]  # Get probability of class 1
            
            return float(probability)
        except Exception as e:
            logging.error(f"Error predicting CTR for headline '{headline}': {e}")
            return 0.001  # Return a low default CTR on error
    
    def calculate_ctr_score(self, headline):
        """Calculate CTR score (compatibility method)"""
        ctr = self.predict_ctr(headline)
        # Return both raw score and percentage for compatibility
        return ctr, ctr
    
    def compare_headlines(self, original, rewritten):
        """Compare original and rewritten headlines"""
        try:
            original_ctr = self.predict_ctr(original)
            rewritten_ctr = self.predict_ctr(rewritten)
            
            # Calculate improvement
            if original_ctr > 0:
                improvement = rewritten_ctr - original_ctr
                score_percent_change = (improvement / original_ctr) * 100
            else:
                improvement = rewritten_ctr
                score_percent_change = 100 if rewritten_ctr > 0 else 0
            
            # Identify key improvements
            key_improvements = self._explain_improvements(original, rewritten)
            
            return {
                'original_score': original_ctr,
                'rewritten_score': rewritten_ctr,
                'original_ctr': original_ctr,
                'rewritten_ctr': rewritten_ctr,
                'score_percent_change': score_percent_change,
                'key_improvements': key_improvements,
                'headline_improvement': improvement * 100  # Convert to percentage points
            }
        except Exception as e:
            logging.error(f"Error comparing headlines: {e}")
            return {
                'original_score': 0.001,
                'rewritten_score': 0.001,
                'original_ctr': 0.001,
                'rewritten_ctr': 0.001,
                'score_percent_change': 0,
                'key_improvements': ["Error in comparison"],
                'headline_improvement': 0
            }
    
    def _explain_improvements(self, original, rewritten):
        """Explain what improved in the headline"""
        orig_features = self.extract_features(original)
        new_features = self.extract_features(rewritten)
        
        improvements = []
        
        # Check for length optimization
        orig_len = orig_features['title_length'].iloc[0]
        new_len = new_features['title_length'].iloc[0]
        if 30 <= new_len <= 60 and (orig_len < 30 or orig_len > 60):
            improvements.append("Optimized headline length")
        
        # Check for added numbers
        if new_features['has_number'].iloc[0] > orig_features['has_number'].iloc[0]:
            improvements.append("Added specific numbers")
        
        # Check for question format
        if new_features['has_question'].iloc[0] > orig_features['has_question'].iloc[0]:
            improvements.append("Added question format")
        
        # Check for how-to format
        if 'how to' in rewritten.lower() and 'how to' not in original.lower():
            improvements.append("Added 'how to' format")
        
        # Check for superlatives
        if new_features['has_superlative'].iloc[0] > orig_features['has_superlative'].iloc[0]:
            improvements.append("Added superlative words")
        
        # Check readability improvement
        orig_readability = orig_features['title_reading_ease'].iloc[0]
        new_readability = new_features['title_reading_ease'].iloc[0]
        if new_readability > orig_readability + 5:
            improvements.append("Improved readability")
        
        # If no specific improvements found but CTR improved
        if not improvements:
            orig_ctr = self.predict_ctr(original)
            new_ctr = self.predict_ctr(rewritten)
            if new_ctr > orig_ctr:
                improvements.append("General semantic improvement")
        
        return improvements
    
    def get_headline_feedback(self, original, rewritten):
        """Generate detailed feedback on the headline rewrite"""
        comparison = self.compare_headlines(original, rewritten)
        
        feedback = []
        
        if comparison['score_percent_change'] > 0:
            feedback.append(f"✅ Improved CTR by {comparison['score_percent_change']:.1f}%")
            feedback.append(f"📈 CTR: {comparison['original_ctr']:.4f} → {comparison['rewritten_ctr']:.4f}")
        else:
            feedback.append(f"⚠️ Decreased CTR by {abs(comparison['score_percent_change']):.1f}%")
            feedback.append(f"📉 CTR: {comparison['original_ctr']:.4f} → {comparison['rewritten_ctr']:.4f}")
        
        if comparison['key_improvements']:
            feedback.append("Key improvements:")
            for improvement in comparison['key_improvements']:
                feedback.append(f"• {improvement}")
        
        return "\n".join(feedback)
    
    def get_recommendation(self, ctr_probability):
        """Get recommendation based on CTR probability"""
        if ctr_probability >= 0.02:
            return {"action": "Must Use - Exceptional", "color": "green"}
        elif ctr_probability >= 0.01:
            return {"action": "Strongly Recommend", "color": "blue"}
        elif ctr_probability >= 0.005:
            return {"action": "Recommend - Good Choice", "color": "lime"}
        elif ctr_probability >= 0.001:
            return {"action": "Consider - If Space Available", "color": "yellow"}
        else:
            return {"action": "Avoid - Better Options Available", "color": "red"}