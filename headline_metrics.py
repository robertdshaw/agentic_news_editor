import pandas as pd
import numpy as np
import logging
import pickle
import re
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from pathlib import Path
from headline_features import HeadlineFeatureExtractor
from headline_utils import load_ctr_model
from sklearn.impute import SimpleImputer


class HeadlineMetrics:
    def __init__(self, model_path="model_output/ctr_model.pkl", client=None):
        """Initialize with the CTR prediction model"""
        self.openai_client = client

        # Load the CTR model and its components
        # Load the CTR model and its components
        try:
            model_data, error = load_ctr_model(model_path)

            if model_data is None:
                logging.error(f"Failed to load CTR model: {error}")
                raise ValueError(f"Could not load model from {model_path}: {error}")

            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.imputer = model_data.get("imputer", SimpleImputer(strategy="mean"))
            self.encoders = model_data["encoders"]
            self.feature_names = model_data["feature_names"]
            self.best_model = model_data["model_name"]
            self.thresholds = model_data.get(
                "thresholds", [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
            )

            logging.info(f"Loaded CTR model ({self.best_model}) from {model_path}")
        except Exception as e:
            logging.error(f"Failed to load CTR model: {e}")
            raise ValueError(f"Could not load model from {model_path}: {e}")

    def extract_features(self, df_or_headline):
        """Extract features using the shared utility"""
        # Initialize the feature extractor if not already done
        if not hasattr(self, "feature_extractor"):
            self.feature_extractor = HeadlineFeatureExtractor(self.embeddings_model)

        # Get features from the shared extractor
        features = self.feature_extractor.extract_features(df_or_headline)

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
            probability = self.model.predict_proba(X_scaled)[
                0, 1
            ]  # Get probability of class 1

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
                "original_score": original_ctr,
                "rewritten_score": rewritten_ctr,
                "original_ctr": original_ctr,
                "rewritten_ctr": rewritten_ctr,
                "score_percent_change": score_percent_change,
                "key_improvements": key_improvements,
                "headline_improvement": improvement
                * 100,  # Convert to percentage points
            }
        except Exception as e:
            logging.error(f"Error comparing headlines: {e}")
            return {
                "original_score": 0.001,
                "rewritten_score": 0.001,
                "original_ctr": 0.001,
                "rewritten_ctr": 0.001,
                "score_percent_change": 0,
                "key_improvements": ["Error in comparison"],
                "headline_improvement": 0,
            }

    def _explain_improvements(self, original, rewritten):
        """Explain what improved in the headline"""
        orig_features = self.extract_features(original)
        new_features = self.extract_features(rewritten)

        improvements = []

        # Check for length optimization
        orig_len = orig_features["title_length"].iloc[0]
        new_len = new_features["title_length"].iloc[0]
        if 30 <= new_len <= 60 and (orig_len < 30 or orig_len > 60):
            improvements.append("Optimized headline length")

        # Check for added numbers
        if new_features["has_number"].iloc[0] > orig_features["has_number"].iloc[0]:
            improvements.append("Added specific numbers")

        # Check for question format
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

        # Check readability improvement
        orig_readability = orig_features["title_reading_ease"].iloc[0]
        new_readability = new_features["title_reading_ease"].iloc[0]
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

        if comparison["score_percent_change"] > 0:
            feedback.append(
                f" Improved CTR by {comparison['score_percent_change']:.1f}%"
            )
            feedback.append(
                f" CTR: {comparison['original_ctr']:.4f} → {comparison['rewritten_ctr']:.4f}"
            )
        else:
            feedback.append(
                f" Decreased CTR by {abs(comparison['score_percent_change']):.1f}%"
            )
            feedback.append(
                f" CTR: {comparison['original_ctr']:.4f} → {comparison['rewritten_ctr']:.4f}"
            )

        if comparison["key_improvements"]:
            feedback.append("Key improvements:")
            for improvement in comparison["key_improvements"]:
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
