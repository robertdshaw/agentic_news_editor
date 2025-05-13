import pandas as pd
import numpy as np
import re
import string
from textstat import flesch_reading_ease
from collections import Counter
import logging

class ImprovedHeadlineFeatures:
    """
    Improved feature extraction based on EDA insights:
    - Questions reduce CTR significantly (-27.9%)
    - Numbers have slight negative effect (-9.8%)
    - Reading ease score doesn't correlate with CTR
    - Categories matter significantly
    - Quotes and numbers in high-CTR headlines
    """
    
    def __init__(self):
        # High-performing categories from EDA
        self.high_ctr_categories = {'kids', 'music', 'tv'}
        self.low_ctr_categories = {'autos', 'travel', 'northamerica'}
        
        # Common clickbait patterns (some positive, some negative)
        self.positive_patterns = {
            'authority': [r'\b(expert|scientist|study|research|reveals|finds|shows)\b'],
            'urgency': [r'\b(breaking|urgent|just|now|today|latest)\b'],
            'list_numbers': [r'\b\d+\s+(ways|tips|secrets|tricks|facts|things)\b'],
            'exclusive': [r'\b(exclusive|first|only|never\s+before)\b'],
            'emotional_positive': [r'\b(amazing|incredible|stunning|shocking|mind\s*blowing)\b']
        }
        
        self.negative_patterns = {
            'generic_questions': [r'\b(how|what|why|when|where|who|which)\b'],
            'weak_language': [r'\b(maybe|perhaps|might|could|possibly)\b'],
            'overused_clickbait': [r'\b(one\s+weird\s+trick|doctors\s+hate|you\s+won\'t\s+believe)\b']
        }
    
    def extract_features(self, headlines, categories=None, abstracts=None):
        """
        Extract features based on EDA insights
        
        Args:
            headlines: List of headlines
            categories: List of categories (optional)
            abstracts: List of abstracts (optional)
            
        Returns:
            DataFrame with features
        """
        features_list = []
        
        for i, headline in enumerate(headlines):
            if pd.isna(headline):
                headline = ""
            
            features = {}
            headline_lower = headline.lower()
            
            # 1. QUESTION-RELATED FEATURES (Major negative impact)
            features['is_question'] = int(headline.endswith('?'))
            features['has_question_words'] = int(bool(re.search(r'\b(how|what|why|when|where|who|which)\b', headline_lower)))
            features['starts_with_question'] = int(bool(re.match(r'^(how|what|why|when|where|who|which)\b', headline_lower)))
            
            # 2. NUMBER-RELATED FEATURES (Slight negative impact)
            features['has_numbers'] = int(bool(re.search(r'\d', headline)))
            features['num_count'] = len(re.findall(r'\d+', headline))
            features['starts_with_number'] = int(bool(re.match(r'^\d+', headline)))
            
            # Special case: List numbers (can be positive in right context)
            features['has_list_number'] = int(bool(re.search(r'\b\d+\s+(ways|tips|secrets|tricks|facts|things|reasons)\b', headline_lower)))
            
            # 3. PUNCTUATION AND STRUCTURE
            features['has_colon'] = int(':' in headline)
            features['has_quotes'] = int(bool(re.search(r'["\']', headline)))
            features['has_exclamation'] = int('!' in headline)
            features['has_parentheses'] = int(bool(re.search(r'[()]', headline)))
            
            # 4. LENGTH AND READABILITY
            features['char_length'] = len(headline)
            features['word_count'] = len(headline.split())
            features['avg_word_length'] = np.mean([len(word) for word in headline.split()]) if headline.split() else 0
            
            # 5. CATEGORY FEATURES (if available)
            if categories and i < len(categories):
                category = categories[i].lower() if pd.notna(categories[i]) else 'unknown'
                features['category_high_ctr'] = int(category in self.high_ctr_categories)
                features['category_low_ctr'] = int(category in self.low_ctr_categories)
                features['category_kids'] = int(category == 'kids')
                features['category_music'] = int(category == 'music')
                features['category_tv'] = int(category == 'tv')
                features['category_autos'] = int(category == 'autos')
                features['category_travel'] = int(category == 'travel')
                features['category_northamerica'] = int(category == 'northamerica')
            
            # 6. POSITIVE CLICKBAIT PATTERNS
            for pattern_type, patterns in self.positive_patterns.items():
                features[f'has_{pattern_type}'] = int(any(bool(re.search(p, headline_lower)) for p in patterns))
            
            # 7. NEGATIVE PATTERNS
            for pattern_type, patterns in self.negative_patterns.items():
                features[f'has_{pattern_type}'] = int(any(bool(re.search(p, headline_lower)) for p in patterns))
            
            # 8. WORD POSITION FEATURES
            words = headline.split()
            if words:
                features['first_word_length'] = len(words[0])
                features['last_word_length'] = len(words[-1])
                features['first_word_caps'] = int(words[0][0].isupper()) if words[0] else 0
                features['last_word_caps'] = int(words[-1][0].isupper()) if words[-1] else 0
            
            # 9. CAPITALIZATION
            features['all_caps_words'] = sum(1 for word in words if word.isupper() and len(word) > 1) if words else 0
            features['cap_ratio'] = sum(1 for c in headline if c.isupper()) / len(headline) if headline else 0
            
            # 10. SEMANTIC INDICATORS
            features['has_action_words'] = int(bool(re.search(r'\b(get|find|learn|discover|unlock|master)\b', headline_lower)))
            features['has_numbers_and_action'] = features['has_numbers'] * features['has_action_words']
            
            # 11. INTERACTION FEATURES (Based on EDA insights)
            # Questions are more harmful when longer
            features['question_length_penalty'] = features['is_question'] * features['char_length']
            # Numbers in high-CTR headlines might be list-type
            features['positive_number_indicator'] = features['has_list_number'] * (1 - features['is_question'])
            
            # 12. DOMAIN-SPECIFIC FEATURES
            features['has_breaking_news'] = int(bool(re.search(r'\b(breaking|urgent|alert)\b', headline_lower)))
            features['has_time_reference'] = int(bool(re.search(r'\b(today|now|just|latest|new|recent)\b', headline_lower)))
            features['has_personal_pronoun'] = int(bool(re.search(r'\b(you|your|we|our|i|my)\b', headline_lower)))
            
            # 13. ADVANCED PATTERN DETECTION
            # Celebrity/name patterns
            features['has_celebrity_pattern'] = int(bool(re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', headline)))
            
            # Price/money references
            features['has_price'] = int(bool(re.search(r'[\$£€¥]|\b(cost|price|expensive|cheap|free)\b', headline_lower)))
            
            # Comparison indicators
            features['has_comparison'] = int(bool(re.search(r'\b(vs|versus|better|best|worst|than|compared)\b', headline_lower)))
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def get_feature_importance_groups(self):
        """Return feature groups by expected importance based on EDA"""
        return {
            'critical_negative': ['is_question', 'has_question_words', 'starts_with_question'],
            'slight_negative': ['has_numbers', 'num_count', 'starts_with_number'],
            'category_positive': ['category_high_ctr', 'category_kids', 'category_music', 'category_tv'],
            'category_negative': ['category_low_ctr', 'category_autos', 'category_travel'],
            'positive_patterns': ['has_quotes', 'has_authority', 'has_urgency', 'has_list_number'],
            'structure': ['has_colon', 'has_exclamation', 'word_count', 'char_length'],
            'interactions': ['question_length_penalty', 'positive_number_indicator']
        }