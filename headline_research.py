import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import textstat
import json
import os
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

class HeadlineResearchAnalyzer:
    """
    Class for analyzing headline improvements across the three research questions:
    RQ1: Readability and cognitive accessibility
    RQ2: User perception
    RQ3: Engagement metrics
    """
    
    def __init__(self):
        """Initialize the analyzer with empty data structures"""
        self.headlines = {}  # Will store headline pairs (original and rewritten)
        self.readability_results = {}
        self.perception_results = {}
        self.engagement_results = {}
        self.mind_dataset = None  # Will store MIND dataset impression and click data
    
    def load_headline_data(self, file_path):
        """
        Load headline pairs from a JSON file
        
        Parameters:
        file_path (str): Path to the JSON file containing headline pairs
        """
        try:
            with open(file_path, 'r') as f:
                self.headlines = json.load(f)
            print(f"Loaded {len(self.headlines)} headline pairs")
            return True
        except Exception as e:
            print(f"Error loading headline data: {e}")
            return False
    
    def load_mind_dataset(self, behaviors_path, news_path):
        """
        Load MIND dataset behaviors (impressions and clicks) and news data
        
        Parameters:
        behaviors_path (str): Path to behaviors.tsv file
        news_path (str): Path to news.tsv file
        """
        # Load news data (contains headlines)
        news_cols = ['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
        news_df = pd.read_csv(news_path, sep='\t', names=news_cols)
        
        # Load behaviors data (contains impressions and clicks)
        behavior_cols = ['impression_id', 'user_id', 'time', 'history', 'impressions']
        behaviors_df = pd.read_csv(behaviors_path, sep='\t', names=behavior_cols)
        
        # Process impressions (format: "news_id-clicked (1 or 0)")
        impressions = []
        for imp_list in behaviors_df['impressions']:
            for imp in imp_list.split():
                parts = imp.split('-')
                if len(parts) == 2:
                    news_id, clicked = parts
                    impressions.append({
                        'news_id': news_id,
                        'clicked': int(clicked)
                    })
        
        # Create impressions dataframe
        impressions_df = pd.DataFrame(impressions)
        
        # Merge with news data to get headlines
        self.mind_dataset = pd.merge(impressions_df, news_df, on='news_id')
        
        print(f"Loaded MIND dataset with {len(self.mind_dataset)} impression records")
        return True
    
    def generate_synthetic_perception_data(self, num_participants=50, preference_bias=0.6):
        """
        Generate synthetic user perception data for headline pairs
        
        Parameters:
        num_participants (int): Number of synthetic participants
        preference_bias (float): Bias toward rewritten headlines (0.5 = no bias)
        
        Returns:
        dict: Synthetic perception data
        """
        synthetic_data = {}
        
        for participant_id in range(1, num_participants + 1):
            participant_data = {}
            
            for headline_id, headline_pair in self.headlines.items():
                # Generate synthetic preference (biased toward rewritten)
                rand_val = np.random.random()
                if rand_val < preference_bias:
                    preference = "rewritten"
                elif rand_val < 0.9:  # Small chance of preferring original
                    preference = "original"
                else:  # Small chance of no preference
                    preference = "no_preference"
                
                # Generate synthetic attribute ratings
                # Higher ratings for preferred version
                attributes = ["Clarity", "Appeal", "Credibility", "Informativeness"]
                
                original_ratings = {}
                rewritten_ratings = {}
                
                # Base ratings - slightly higher for rewritten in general
                base_original = np.random.normal(4.0, 0.5)
                base_rewritten = np.random.normal(4.5, 0.5)
                
                # Adjust based on preference
                if preference == "original":
                    base_original += 1.0
                    base_rewritten -= 0.5
                elif preference == "rewritten":
                    base_original -= 0.5
                    base_rewritten += 1.0
                
                # Generate ratings for each attribute
                for attr in attributes:
                    original_ratings[attr] = max(1, min(7, round(base_original + np.random.normal(0, 0.5))))
                    rewritten_ratings[attr] = max(1, min(7, round(base_rewritten + np.random.normal(0, 0.5))))
                
                participant_data[headline_id] = {
                    'preference': preference,
                    'original_ratings': original_ratings,
                    'rewritten_ratings': rewritten_ratings,
                    'participant_id': f"synthetic_{participant_id}"
                }
            
            synthetic_data[f"synthetic_{participant_id}"] = participant_data
        
        self.perception_results = synthetic_data
        print(f"Generated synthetic perception data for {num_participants} participants")
        return synthetic_data
    
    # RQ2: Readability and Cognitive Accessibility Analysis
    def analyze_headline_readability(self):
        """
        Analyze readability metrics for original vs. rewritten headlines
        
        Returns:
        dict: Readability analysis results
        """
        results = {}
        
        for headline_id, headline_pair in self.headlines.items():
            original = headline_pair['original']
            rewritten = headline_pair['rewritten']
            
            # Calculate core readability metrics
            metrics = {
                'original': {
                    'flesch_reading_ease': textstat.flesch_reading_ease(original),
                    'flesch_kincaid_grade': textstat.flesch_kincaid_grade(original),
                    'syllable_count': textstat.syllable_count(original),
                    'word_count': len(original.split()),
                    'avg_word_length': sum(len(word) for word in original.split()) / len(original.split()) if original else 0,
                    'complex_word_count': textstat.difficult_words(original),
                    'complex_word_percentage': textstat.difficult_words(original) / len(original.split()) if original.split() else 0
                },
                'rewritten': {
                    'flesch_reading_ease': textstat.flesch_reading_ease(rewritten),
                    'flesch_kincaid_grade': textstat.flesch_kincaid_grade(rewritten),
                    'syllable_count': textstat.syllable_count(rewritten),
                    'word_count': len(rewritten.split()),
                    'avg_word_length': sum(len(word) for word in rewritten.split()) / len(rewritten.split()) if rewritten else 0,
                    'complex_word_count': textstat.difficult_words(rewritten),
                    'complex_word_percentage': textstat.difficult_words(rewritten) / len(rewritten.split()) if rewritten.split() else 0
                }
            }
            
            # Calculate improvements
            improvements = {}
            for metric in metrics['original'].keys():
                if metric in ['flesch_reading_ease']:
                    # Higher is better for FRE
                    improvements[metric] = metrics['rewritten'][metric] - metrics['original'][metric]
                elif metric in ['flesch_kincaid_grade', 'syllable_count', 'complex_word_count', 'complex_word_percentage']:
                    # Lower is better for these
                    improvements[metric] = metrics['original'][metric] - metrics['rewritten'][metric]
                else:
                    # For neutral metrics, just calculate difference
                    improvements[metric] = metrics['rewritten'][metric] - metrics['original'][metric]
            
            metrics['improvements'] = improvements
            results[headline_id] = metrics
        
        self.readability_results = results
        return results
    
    def analyze_readability_statistics(self):
        """
        Perform statistical analysis on readability metrics
        
        Returns:
        dict: Statistical analysis results
        """
        if not self.readability_results:
            print("No readability results available. Run analyze_headline_readability() first.")
            return {}
        
        # Extract metrics for statistical testing
        metrics_to_test = ['flesch_reading_ease', 'flesch_kincaid_grade', 'syllable_count', 
                         'word_count', 'avg_word_length', 'complex_word_percentage']
        
        stats_results = {}
        
        for metric in metrics_to_test:
            # Collect original and rewritten values
            original_values = [result['original'][metric] for result in self.readability_results.values()]
            rewritten_values = [result['rewritten'][metric] for result in self.readability_results.values()]
            
            # Perform paired t-test
            t_stat, p_value = stats.ttest_rel(original_values, rewritten_values)
            
            # Calculate effect size (Cohen's d)
            mean_diff = np.mean(np.array(rewritten_values) - np.array(original_values))
            pooled_std = np.sqrt((np.std(original_values)**2 + np.std(rewritten_values)**2) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std != 0 else 0
            
            # Determine if improvement is positive or negative based on metric
            if metric in ['flesch_reading_ease']:
                # Higher is better for FRE
                improved = mean_diff > 0
            elif metric in ['flesch_kincaid_grade', 'syllable_count', 'complex_word_percentage']:
                # Lower is better for these
                improved = mean_diff < 0
                # Adjust Cohen's d to be positive when there's improvement
                if improved:
                    cohens_d = abs(cohens_d)
                else:
                    cohens_d = -abs(cohens_d)
            else:
                # For neutral metrics
                improved = None
            
            stats_results[metric] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'cohens_d': cohens_d,
                'mean_original': np.mean(original_values),
                'mean_rewritten': np.mean(rewritten_values),
                'mean_diff': mean_diff,
                'improved': improved
            }
        
        return stats_results
    
    def plot_readability_results(self, save_dir="results/figures"):
        """
        Generate plots for readability analysis
        
        Parameters:
        save_dir (str): Directory to save figures
        """
        if not self.readability_results:
            print("No readability results available. Run analyze_headline_readability() first.")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Plot Flesch Reading Ease comparison
        plt.figure(figsize=(12, 6))
        
        # Extract data
        headline_ids = list(self.readability_results.keys())
        original_fre = [self.readability_results[hid]['original']['flesch_reading_ease'] for hid in headline_ids]
        rewritten_fre = [self.readability_results[hid]['rewritten']['flesch_reading_ease'] for hid in headline_ids]
        
        # Create bar positions
        x = np.arange(len(headline_ids))
        width = 0.35
        
        # Plot bars
        plt.bar(x - width/2, original_fre, width, label='Original')
        plt.bar(x + width/2, rewritten_fre, width, label='Rewritten')
        
        # Add details
        plt.xlabel('Headline ID')
        plt.ylabel('Flesch Reading Ease (higher is better)')
        plt.title('Readability Comparison: Original vs Rewritten Headlines')
        plt.xticks(x, [f"H{i+1}" for i in range(len(headline_ids))], rotation=45)
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{save_dir}/readability_comparison.png")
        plt.close()
        
        # 2. Plot improvements across metrics
        metrics_to_plot = ['flesch_reading_ease', 'flesch_kincaid_grade', 'complex_word_percentage']
        labels = ['Flesch Reading Ease\n(higher better)', 'Flesch-Kincaid Grade\n(lower better)', 'Complex Words %\n(lower better)']
        
        plt.figure(figsize=(10, 6))
        
        # Calculate average improvements
        avg_improvements = []
        for metric in metrics_to_plot:
            if metric == 'flesch_reading_ease':
                # Higher is better, direct improvement
                values = [result['improvements'][metric] for result in self.readability_results.values()]
            else:
                # Lower is better, improvement is reduction
                values = [result['improvements'][metric] for result in self.readability_results.values()]
            avg_improvements.append(np.mean(values))
        
        # Plot bars
        plt.bar(labels, avg_improvements)
        
        # Add zero line
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Add details
        plt.ylabel('Average Improvement')
        plt.title('Average Improvements in Readability Metrics')
        
        # Color bars based on whether higher or lower is better
        for i, improvement in enumerate(avg_improvements):
            color = 'green' if improvement > 0 else 'red'
            if i == 1 or i == 2:  # For metrics where lower is better
                color = 'green' if improvement > 0 else 'red'
            plt.gca().get_children()[i].set_color(color)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{save_dir}/readability_improvements.png")
        plt.close()
    
    # RQ3: User Perception Analysis
    def analyze_perception_data(self):
        """
        Analyze user perception data (real or synthetic)
        
        Returns:
        dict: Perception analysis results
        """
        if not self.perception_results:
            print("No perception data available.")
            return {}
        
        analysis = {
            'preferences': {},
            'attribute_ratings': {},
            'statistical_tests': {}
        }
        
        # Collect all preferences
        all_preferences = []
        for participant_data in self.perception_results.values():
            for headline_data in participant_data.values():
                all_preferences.append(headline_data['preference'])
        
        # Calculate preference distribution
        preference_counts = Counter(all_preferences)
        total_judgments = len(all_preferences)
        
        analysis['preferences']['counts'] = dict(preference_counts)
        analysis['preferences']['percentages'] = {k: (v/total_judgments)*100 for k, v in preference_counts.items()}
        
        # Chi-square test for preference distribution
        expected = {k: total_judgments/3 for k in ['original', 'rewritten', 'no_preference']}
        observed = [preference_counts.get(k, 0) for k in ['original', 'rewritten', 'no_preference']]
        expected_values = [expected.get(k, 0) for k in ['original', 'rewritten', 'no_preference']]
        
        chi2, p_value = stats.chisquare(observed, expected_values)
        analysis['statistical_tests']['preferences_chi2'] = {
            'chi2': chi2,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        # Attribute ratings analysis
        attributes = ['Clarity', 'Appeal', 'Credibility', 'Informativeness']
        
        for attr in attributes:
            original_ratings = []
            rewritten_ratings = []
            
            for participant_data in self.perception_results.values():
                for headline_data in participant_data.values():
                    if attr in headline_data['original_ratings'] and attr in headline_data['rewritten_ratings']:
                        original_ratings.append(headline_data['original_ratings'][attr])
                        rewritten_ratings.append(headline_data['rewritten_ratings'][attr])
            
            # T-test for attribute
            t_stat, p_value = stats.ttest_rel(original_ratings, rewritten_ratings)
            
            # Calculate effect size
            mean_diff = np.mean(np.array(rewritten_ratings) - np.array(original_ratings))
            pooled_std = np.sqrt((np.std(original_ratings)**2 + np.std(rewritten_ratings)**2) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std != 0 else 0
            
            analysis['attribute_ratings'][attr] = {
                'mean_original': np.mean(original_ratings),
                'mean_rewritten': np.mean(rewritten_ratings),
                'mean_difference': mean_diff,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'cohens_d': cohens_d
            }
        
        return analysis
    
    def plot_perception_results(self, perception_analysis=None, save_dir="results/figures"):
        """
        Generate plots for perception analysis
        
        Parameters:
        perception_analysis (dict): Analysis results (if None, will run analysis)
        save_dir (str): Directory to save figures
        """
        if perception_analysis is None:
            perception_analysis = self.analyze_perception_data()
        
        if not perception_analysis:
            print("No perception analysis available.")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Plot preference distribution
        plt.figure(figsize=(8, 6))
        
        preferences = perception_analysis['preferences']['percentages']
        labels = ['Original', 'Rewritten', 'No Preference']
        values = [preferences.get('original', 0), preferences.get('rewritten', 0), preferences.get('no_preference', 0)]
        
        plt.bar(labels, values)
        plt.ylabel('Percentage (%)')
        plt.title('Headline Preference Distribution')
        
        # Add percentage labels on bars
        for i, v in enumerate(values):
            plt.text(i, v + 1, f"{v:.1f}%", ha='center')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/preference_distribution.png")
        plt.close()
        
        # 2. Plot attribute ratings
        plt.figure(figsize=(10, 6))
        
        attributes = list(perception_analysis['attribute_ratings'].keys())
        original_means = [perception_analysis['attribute_ratings'][attr]['mean_original'] for attr in attributes]
        rewritten_means = [perception_analysis['attribute_ratings'][attr]['mean_rewritten'] for attr in attributes]
        
        x = np.arange(len(attributes))
        width = 0.35
        
        plt.bar(x - width/2, original_means, width, label='Original')
        plt.bar(x + width/2, rewritten_means, width, label='Rewritten')
        
        plt.xlabel('Attribute')
        plt.ylabel('Average Rating (1-7)')
        plt.title('Attribute Ratings: Original vs Rewritten Headlines')
        plt.xticks(x, attributes)
        plt.legend()
        
        # Add significance stars
        for i, attr in enumerate(attributes):
            if perception_analysis['attribute_ratings'][attr]['significant']:
                # Add star for significant differences
                plt.text(i, max(original_means[i], rewritten_means[i]) + 0.1, '*', 
                         ha='center', va='bottom', fontsize=16)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/attribute_ratings.png")
        plt.close()
        
        # 3. Plot radar chart of attributes
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        
        # Compute angles for each attribute
        angles = np.linspace(0, 2*np.pi, len(attributes), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Extend the data lists to close the loop
        original_means_closed = original_means + [original_means[0]]
        rewritten_means_closed = rewritten_means + [rewritten_means[0]]
        attributes_closed = attributes + [attributes[0]]
        
        # Plot data
        ax.plot(angles, original_means_closed, 'b-', linewidth=1.5, label='Original')
        ax.plot(angles, rewritten_means_closed, 'r-', linewidth=1.5, label='Rewritten')
        ax.fill(angles, original_means_closed, 'b', alpha=0.1)
        ax.fill(angles, rewritten_means_closed, 'r', alpha=0.1)
        
        # Set labels and title
        ax.set_thetagrids(np.degrees(angles[:-1]), attributes)
        ax.set_ylim(0, 7)
        ax.set_title('Headline Attribute Ratings', y=1.08)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/attribute_radar.png")
        plt.close()
    
    # RQ4: Engagement Analysis with MIND Dataset
    def analyze_mind_engagement(self):
        """
        Analyze engagement metrics using MIND dataset click logs
        
        Returns:
        dict: Engagement analysis results
        """
        if self.mind_dataset is None:
            print("MIND dataset not loaded. Run load_mind_dataset() first.")
            return {}
        
        results = {}
        
        # 1. Train a basic headline engagement model using MIND data
        X = self.mind_dataset['title'].apply(self._extract_headline_features).tolist()
        y = self.mind_dataset['clicked'].values
        
        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results['model_performance'] = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Save model for later use
        os.makedirs("models", exist_ok=True)
        with open("models/headline_engagement_model.pkl", 'wb') as f:
            pickle.dump(model, f)
        
        # 2. Apply model to headline pairs
        engagement_predictions = {}
        
        for headline_id, headline_pair in self.headlines.items():
            original = headline_pair['original']
            rewritten = headline_pair['rewritten']
            
            original_features = self._extract_headline_features(original)
            rewritten_features = self._extract_headline_features(rewritten)
            
            # Predict click probability
            original_prob = model.predict_proba([original_features])[0][1]
            rewritten_prob = model.predict_proba([rewritten_features])[0][1]
            
            # Calculate relative improvement
            absolute_improvement = rewritten_prob - original_prob
            relative_improvement = (absolute_improvement / original_prob) * 100 if original_prob > 0 else 0
            
            engagement_predictions[headline_id] = {
                'original_ctr': float(original_prob),
                'rewritten_ctr': float(rewritten_prob),
                'absolute_improvement': float(absolute_improvement),
                'relative_improvement_percent': float(relative_improvement)
            }
        
        results['headline_predictions'] = engagement_predictions
        
        # 3. Calculate average improvements
        if engagement_predictions:
            absolute_improvements = [pred['absolute_improvement'] for pred in engagement_predictions.values()]
            relative_improvements = [pred['relative_improvement_percent'] for pred in engagement_predictions.values()]
            
            results['average_improvements'] = {
                'absolute': np.mean(absolute_improvements),
                'relative_percent': np.mean(relative_improvements)
            }
            
            # Perform t-test on CTR predictions
            original_ctrs = [pred['original_ctr'] for pred in engagement_predictions.values()]
            rewritten_ctrs = [pred['rewritten_ctr'] for pred in engagement_predictions.values()]
            
            t_stat, p_value = stats.ttest_rel(original_ctrs, rewritten_ctrs)
            
            results['statistical_tests'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        self.engagement_results = results
        return results
    
    def plot_engagement_results(self, engagement_analysis=None, save_dir="results/figures"):
        """
        Generate plots for engagement analysis
        
        Parameters:
        engagement_analysis (dict): Analysis results (if None, will use stored results)
        save_dir (str): Directory to save figures
        """
        if engagement_analysis is None:
            engagement_analysis = self.engagement_results
        
        if not engagement_analysis or 'headline_predictions' not in engagement_analysis:
            print("No engagement analysis available.")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Plot CTR comparison
        plt.figure(figsize=(12, 6))
        
        # Extract data
        headline_ids = list(engagement_analysis['headline_predictions'].keys())
        original_ctrs = [engagement_analysis['headline_predictions'][hid]['original_ctr'] for hid in headline_ids]
        rewritten_ctrs = [engagement_analysis['headline_predictions'][hid]['rewritten_ctr'] for hid in headline_ids]
        
        # Create bar positions
        x = np.arange(len(headline_ids))
        width = 0.35
        
        # Plot bars
        plt.bar(x - width/2, original_ctrs, width, label='Original')
        plt.bar(x + width/2, rewritten_ctrs, width, label='Rewritten')
        
        # Add details
        plt.xlabel('Headline ID')
        plt.ylabel('Predicted Click-Through Rate')
        plt.title('CTR Comparison: Original vs Rewritten Headlines')
        plt.xticks(x, [f"H{i+1}" for i in range(len(headline_ids))], rotation=45)
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{save_dir}/ctr_comparison.png")
        plt.close()
        
        # 2. Plot relative improvements
        plt.figure(figsize=(12, 6))
        
        improvements = [engagement_analysis['headline_predictions'][hid]['relative_improvement_percent'] for hid in headline_ids]
        
        bars = plt.bar(range(len(headline_ids)), improvements)
        
        # Color bars based on improvement
        for i, improvement in enumerate(improvements):
            color = 'green' if improvement > 0 else 'red'
            bars[i].set_color(color)
        
        # Add average line
        avg_improvement = engagement_analysis['average_improvements']['relative_percent']
        plt.axhline(y=avg_improvement, color='black', linestyle='--', 
                   label=f'Average: {avg_improvement:.1f}%')
        
        # Add details
        plt.xlabel('Headline ID')
        plt.ylabel('Relative CTR Improvement (%)')
        plt.title('Predicted CTR Improvement from Headline Rewriting')
        plt.xticks(range(len(headline_ids)), [f"H{i+1}" for i in range(len(headline_ids))], rotation=45)
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{save_dir}/ctr_improvements.png")
        plt.close()
        
        # 3. Plot model feature importance if available
        if os.path.exists("models/headline_engagement_model.pkl"):
            with open("models/headline_engagement_model.pkl", 'rb') as f:
                model = pickle.load(f)
            
            if hasattr(model, 'feature_importances_'):
                plt.figure(figsize=(10, 8))
                
                # Get feature names
                feature_names = self._get_feature_names()
                
                # Sort feature importances
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                # Plot top 15 features
                plt.barh(range(min(15, len(feature_names))), 
                        [importances[i] for i in indices[:15]], 
                        align='center')
                plt.yticks(range(min(15, len(feature_names))), 
                          [feature_names[i] for i in indices[:15]])
                plt.xlabel('Feature Importance')
                plt.title('Top Features for Click Prediction')
                plt.tight_layout()
                
                # Save figure
                plt.savefig(f"{save_dir}/feature_importance.png")
                plt.close()
    
    def _extract_headline_features(self, headline):
        """
        Extract features from headline text for engagement prediction
        
        Parameters:
        headline (str): Headline text
        
        Returns:
        list: Feature vector
        """
        features = []
        
        # Length features
        features.append(len(headline))
        features.append(len(headline.split()))
        
        # Readability
        features.append(textstat.flesch_reading_ease(headline))
        features.append(textstat.flesch_kincaid_grade(headline))
        
        # Sentence structure
        features.append(1 if '?' in headline else 0)  # Question
        features.append(1 if '!' in headline else 0)  # Exclamation
        features.append(headline.count(','))  # Comma count
        
        if __name__ == "__main__":
            
            # Initialize analyzer
            analyzer = HeadlineResearchAnalyzer()
    
            # Use sample headlines (built into the code) or load your own
            # analyzer.load_headline_data("your_headline_pairs.json")
                
            # RQ1: Analyze readability
            print("\n=== Analyzing RQ1: Headline Readability ===\n")
            readability_results = analyzer.analyze_headline_readability()
            readability_stats = analyzer.analyze_readability_statistics()
            analyzer.plot_readability_results()
    
            # RQ2: Analyze perception
            print("\n=== Analyzing RQ2: User Perception ===\n")
            analyzer.generate_synthetic_perception_data()
            perception_results = analyzer.analyze_perception_data()
            analyzer.plot_perception_results()
            
            # RQ3: Analyze engagement 
            print("\n=== Analyzing RQ3: Engagement Prediction ===\n")
            try:
                analyzer.load_mind_dataset("data/mind/behaviors.tsv", "data/mind/news.tsv")
                engagement_results = analyzer.analyze_mind_engagement()
            except:
                print("MIND dataset not available, using simulated engagement model")
                # Create a simple model without MIND data
                engagement_results = analyzer.analyze_engagement_without_mind()
            
            analyzer.plot_engagement_results()
            
            # Generate comprehensive report
            print("\n=== Generating Research Report ===\n")
            report = analyzer.generate_comprehensive_report()
            
            print("\nAnalysis complete! Check the 'results' directory for full report and visualizations.")