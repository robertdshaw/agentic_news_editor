import pandas as pd
import numpy as np
import logging
from textstat import flesch_reading_ease


class HeadlineFeatureExtractor:
    def __init__(self, embeddings_model=None):
        """Initialize with an optional pre-loaded embeddings model"""
        self.embeddings_model = embeddings_model

    def extract_features(self, df_or_headline):
        """Extract features from headlines and metadata"""
        # Handle single headline or dataframe
        if isinstance(df_or_headline, str):
            df = pd.DataFrame({"title": [df_or_headline]})
        else:
            df = df_or_headline.copy()

        features = pd.DataFrame(index=df.index)

        # Text features from title
        features["title_length"] = df["title"].str.len()
        features["title_word_count"] = df["title"].str.split().str.len()

        # Pattern features
        features["has_question"] = df["title"].str.contains(r"\?").astype(int)
        features["has_number"] = df["title"].str.contains(r"\d").astype(int)
        features["has_colon"] = df["title"].str.contains(":").astype(int)
        features["has_quotes"] = df["title"].str.contains(r'["\']').astype(int)
        features["has_ellipsis"] = df["title"].str.contains(r"\.\.\.").astype(int)

        # Problematic patterns
        features["starts_with_number"] = df["title"].str.match(r"^\d").astype(int)
        features["starts_with_question"] = (
            df["title"].str.match(r"^(Is|What|How|Why|When|Where)").astype(int)
        )
        features["has_superlative"] = (
            df["title"]
            .str.contains(r"\b(?:best|worst|most|least|biggest|smallest)\b", case=False)
            .astype(int)
        )
        features["title_uppercase_ratio"] = df["title"].apply(
            lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
        )

        # Reading ease with proper error handling
        if "title_reading_ease" not in df.columns:

            def safe_flesch_reading_ease(text):
                try:
                    if pd.isna(text) or not isinstance(text, str) or len(text) == 0:
                        return 0
                    return flesch_reading_ease(text)
                except:
                    return 0

            features["title_reading_ease"] = df["title"].apply(safe_flesch_reading_ease)
        else:
            features["title_reading_ease"] = df["title_reading_ease"].fillna(0)

        # Default values for other features
        features["category_encoded"] = 0  # Default category
        features["hour"] = 12  # Default to noon
        features["day_of_week"] = 1  # Default to Monday
        features["is_weekend"] = 0  # Default to weekday

        # Embeddings - only if model is provided
        if self.embeddings_model is not None:
            try:
                titles_clean = df["title"].fillna("").astype(str).tolist()
                embeddings = self.embeddings_model.encode(titles_clean, batch_size=32)

                # Add first 50 embedding dimensions as features
                for i in range(min(50, embeddings.shape[1])):
                    features[f"emb_{i}"] = embeddings[:, i]
            except Exception as e:
                logging.error(f"Error creating embeddings: {e}")
                # Add zero embeddings if failed
                for i in range(50):
                    features[f"emb_{i}"] = 0.0
        else:
            # Add zero embeddings if no model
            for i in range(50):
                features[f"emb_{i}"] = 0.0

        # Fill any remaining NaN values
        features = features.fillna(0)

        return features
