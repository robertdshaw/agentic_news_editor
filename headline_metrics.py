import numpy as np
import logging
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from headline_features import HeadlineFeatureExtractor
from headline_utils import load_ctr_model


class HeadlineMetrics:
    """Evaluates headline effectiveness using a trained CTR prediction model"""

    def __init__(self, model_path="model_output/ctr_model.pkl"):
        """Load the model and initialize embedding tools"""
        # Load the CTR model
        model_data, error = load_ctr_model(model_path)
        if model_data is None:
            raise ValueError(f"Failed to load CTR model: {error}")

        # Set up model components
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.imputer = model_data.get("imputer")
        self.feature_names = model_data["feature_names"]
        self.best_model = model_data["model_name"]
        logging.info(f"Loaded CTR model ({self.best_model}) from {model_path}")

        # Set up embedding model
        try:
            self.embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.feature_extractor = HeadlineFeatureExtractor(self.embeddings_model)
            logging.info("Loaded sentence transformer model")
        except Exception as e:
            logging.error(f"Failed to load embedding model: {e}")
            raise

    def predict_ctr(self, headline):
        """Predict CTR for a headline"""
        try:
            # Extract features
            features = self.feature_extractor.extract_features(headline)

            # Ensure all required features exist
            missing_features = set(self.feature_names) - set(features.columns)
            for feat in missing_features:
                features[feat] = 0

            # Use only the features the model expects
            X = features[self.feature_names]

            # Scale and predict
            X_scaled = self.scaler.transform(X)
            if self.imputer and np.isnan(X_scaled).any():
                X_scaled = self.imputer.transform(X_scaled)

            # Get the click probability (class 1)
            probability = self.model.predict_proba(X_scaled)[0, 1]
            return float(probability)
        except Exception as e:
            logging.error(f"Error predicting CTR for '{headline[:30]}...': {e}")
            return 0.001  # Safe default

    def compare_headlines(self, original, rewritten):
        """Compare original and rewritten headlines"""
        try:
            # Get CTR predictions
            original_ctr = self.predict_ctr(original)
            rewritten_ctr = self.predict_ctr(rewritten)

            # Calculate improvement percentages
            if original_ctr > 0:
                pct_change = ((rewritten_ctr - original_ctr) / original_ctr) * 100
            else:
                pct_change = 100 if rewritten_ctr > 0 else 0

            # Identify what improved
            improvements = self._identify_improvements(original, rewritten)

            return {
                "original_ctr": original_ctr,
                "rewritten_ctr": rewritten_ctr,
                "score_percent_change": pct_change,
                "key_improvements": improvements,
                # For backward compatibility
                "original_score": original_ctr,
                "rewritten_score": rewritten_ctr,
                "headline_improvement": pct_change,
            }
        except Exception as e:
            logging.error(f"Error comparing headlines: {e}")
            return {
                "original_ctr": 0.001,
                "rewritten_ctr": 0.001,
                "score_percent_change": 0,
                "key_improvements": ["Error in comparison"],
                "original_score": 0.001,
                "rewritten_score": 0.001,
                "headline_improvement": 0,
            }

    def _identify_improvements(self, original, rewritten):
        """Identify what improved between headlines"""
        improvements = []

        # Extract features for comparison
        orig_features = self.feature_extractor.extract_features(original)
        new_features = self.feature_extractor.extract_features(rewritten)

        # Check length (30-60 chars is optimal)
        orig_len = orig_features["title_length"].iloc[0]
        new_len = new_features["title_length"].iloc[0]
        if 30 <= new_len <= 60 and (orig_len < 30 or orig_len > 60):
            improvements.append("Optimized headline length")

        # Check for numbers
        if new_features["has_number"].iloc[0] > orig_features["has_number"].iloc[0]:
            improvements.append("Added specific numbers")

        # Check for questions
        if new_features["has_question"].iloc[0] > orig_features["has_question"].iloc[0]:
            improvements.append("Added question format")

        # Check for how-to format
        if "how to" in rewritten.lower() and "how to" not in original.lower():
            improvements.append("Added 'how to' format")

        # Check for superlatives
        if (
            new_features["has_superlative"].iloc[0]
            > orig_features["has_superlative"].iloc[0]
        ):
            improvements.append("Added superlative words")

        # If nothing specific improved but CTR is better
        if not improvements and self.predict_ctr(rewritten) > self.predict_ctr(
            original
        ):
            improvements.append("General semantic improvement")

        return improvements

    def get_headline_category(self, ctr):
        """Get headline category based on CTR"""
        if ctr >= 0.02:
            return "Excellent"
        elif ctr >= 0.01:
            return "Strong"
        elif ctr >= 0.005:
            return "Good"
        elif ctr >= 0.001:
            return "Average"
        else:
            return "Poor"
