import pandas as pd
import numpy as np
import os
import logging
import pickle
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import json
from headline_metrics import HeadlineMetrics

class HeadlineLearningLoop:
    """
    A learning system that continuously improves headline rewriting by:
    1. Collecting pairs of original and rewritten headlines
    2. Analyzing their performance metrics
    3. Periodically retraining the underlying CTR prediction model
    4. Generating insights about what makes headlines effective
    """
    
    def __init__(self, data_file="headline_learning_data.csv", model_file="headline_ctr_model.pkl"):
        """Initialize the headline learning system"""
        self.data_file = data_file
        self.model_file = model_file
        self.metrics_analyzer = HeadlineMetrics()
        
        # Initialize or load the dataset
        if os.path.exists(data_file):
            self.data = pd.read_csv(data_file)
            logging.info(f"Loaded {len(self.data)} headline pairs from {data_file}")
        else:
            self.data = pd.DataFrame({
                'original_title': [],
                'rewritten_title': [],
                'headline_score_original': [],
                'headline_score_rewritten': [],
                'headline_ctr_original': [],
                'headline_ctr_rewritten': [],
                'headline_improvement': [],
                'headline_key_factors': [],
                'topic': [],
                'timestamp': []
            })
            logging.info(f"Created new headline learning dataset")
    
    def add_headline_pair(self, original, rewritten, topic=None):
        """Add a single pair of headlines to the learning system"""
        try:
            # Calculate metrics
            comparison = self.metrics_analyzer.compare_headlines(original, rewritten)
            
            # Create new entry
            new_entry = pd.DataFrame({
                'original_title': [original],
                'rewritten_title': [rewritten],
                'headline_score_original': [comparison['original_score']],
                'headline_score_rewritten': [comparison['rewritten_score']],
                'headline_ctr_original': [comparison['original_ctr']],
                'headline_ctr_rewritten': [comparison['rewritten_ctr']],
                'headline_improvement': [comparison['score_percent_change']],
                'headline_key_factors': [', '.join(comparison['key_improvements'])],
                'topic': [topic if topic else 'General'],
                'timestamp': [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            })
            
            # Append to dataset
            self.data = pd.concat([self.data, new_entry], ignore_index=True)
            
            # Save the updated dataset
            self.data.to_csv(self.data_file, index=False)
            
            return True
        except Exception as e:
            logging.error(f"Error adding headline pair: {e}")
            return False
    
    def add_headlines_from_dataframe(self, df):
        """Add multiple headline pairs from a dataframe"""
        count = 0
        
        for _, row in df.iterrows():
            if 'original_title' in row and 'rewritten_title' in row:
                topic = row.get('topic', 'General')
                
                # Check if we already have metrics
                if ('headline_score_original' in row and 
                    'headline_score_rewritten' in row and
                    'headline_ctr_original' in row and
                    'headline_ctr_rewritten' in row and
                    'headline_improvement' in row and
                    'headline_key_factors' in row):
                    
                    # Create new entry with existing metrics
                    new_entry = pd.DataFrame({
                        'original_title': [row['original_title']],
                        'rewritten_title': [row['rewritten_title']],
                        'headline_score_original': [row['headline_score_original']],
                        'headline_score_rewritten': [row['headline_score_rewritten']],
                        'headline_ctr_original': [row['headline_ctr_original']],
                        'headline_ctr_rewritten': [row['headline_ctr_rewritten']],
                        'headline_improvement': [row['headline_improvement']],
                        'headline_key_factors': [row['headline_key_factors']],
                        'topic': [topic],
                        'timestamp': [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
                    })
                    
                    # Append to dataset
                    self.data = pd.concat([self.data, new_entry], ignore_index=True)
                    count += 1
                else:
                    # Calculate new metrics
                    if self.add_headline_pair(row['original_title'], row['rewritten_title'], topic):
                        count += 1
        
        # Save the updated dataset
        if count > 0:
            self.data.to_csv(self.data_file, index=False)
        
        return count
    
    def retrain_model(self, force=False):
        """Retrain the CTR prediction model using collected data"""
        # Check if we have enough data
        if len(self.data) < 50 and not force:
            logging.info("Not enough headline data to retrain model (min 50 required)")
            return False
        
        try:
            # Prepare features from original headlines
            orig_features = []
            for title in self.data['original_title']:
                features = self.metrics_analyzer.extract_features(title)
                orig_features.append(features)
            
            orig_df = pd.DataFrame(orig_features)
            orig_df['ctr'] = self.data['headline_ctr_original']
            
            # Prepare features from rewritten headlines
            rewritten_features = []
            for title in self.data['rewritten_title']:
                features = self.metrics_analyzer.extract_features(title)
                rewritten_features.append(features)
            
            rewritten_df = pd.DataFrame(rewritten_features)
            rewritten_df['ctr'] = self.data['headline_ctr_rewritten']
            
            # Combine datasets
            combined_df = pd.concat([orig_df, rewritten_df], ignore_index=True)
            
            # Split features and target
            X = combined_df.drop('ctr', axis=1)
            y = combined_df['ctr']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logging.info(f"Model MSE: {mse}, R²: {r2}")
            
            # Save model if performance is good enough
            if r2 > 0.3 or force:
                with open(self.model_file, 'wb') as f:
                    pickle.dump(model, f)
                
                # Also save model metrics
                with open('headline_model_metrics.json', 'w') as f:
                    json.dump({
                        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'samples': len(combined_df),
                        'mse': mse,
                        'r2': r2
                    }, f)
                
                logging.info(f"Model saved to {self.model_file}")
                return True
            else:
                logging.warning(f"Model performance too low (R²={r2}), not saving")
                return False
                
        except Exception as e:
            logging.error(f"Error retraining model: {e}")
            return False
    
    def analyze_headline_patterns(self):
        """Analyze headline patterns to extract insights"""
        if len(self.data) < 20:
            return {
                'status': 'insufficient_data',
                'message': 'Need at least 20 headline pairs for analysis'
            }
        
        try:
            # Calculate improvement stats
            improvements = self.data['headline_improvement']
            avg_improvement = improvements.mean()
            median_improvement = improvements.median()
            improvement_rate = (improvements > 0).mean()
            
            # Find common improvement factors
            if 'headline_key_factors' in self.data.columns:
                all_factors = []
                for factors in self.data['headline_key_factors']:
                    if isinstance(factors, str):
                        all_factors.extend([f.strip() for f in factors.split(',')])
                
                factor_counts = pd.Series(all_factors).value_counts()
                top_factors = factor_counts.head(5).to_dict()
            else:
                top_factors = {}
            
            # Look at topic performance
            topic_performance = {}
            if 'topic' in self.data.columns:
                for topic in self.data['topic'].unique():
                    topic_data = self.data[self.data['topic'] == topic]
                    topic_performance[topic] = {
                        'count': len(topic_data),
                        'avg_improvement': topic_data['headline_improvement'].mean(),
                        'improvement_rate': (topic_data['headline_improvement'] > 0).mean()
                    }
            
            return {
                'status': 'success',
                'sample_size': len(self.data),
                'overall_stats': {
                    'avg_improvement': avg_improvement,
                    'median_improvement': median_improvement,
                    'improvement_rate': improvement_rate
                },
                'top_improvement_factors': top_factors,
                'topic_performance': topic_performance
            }
            
        except Exception as e:
            logging.error(f"Error analyzing headline patterns: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def prompt_improvement_report(self):
        """Generate a markdown report with insights for headline improvement"""
        analysis = self.analyze_headline_patterns()
        
        if analysis['status'] != 'success':
            return False
        
        try:
            report = f"""# Headline Improvement Analysis
Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}

## Overview
- Dataset contains {analysis['sample_size']} headline pairs
- Average improvement: {analysis['overall_stats']['avg_improvement']:.1f}%
- Improvement rate: {analysis['overall_stats']['improvement_rate']:.1%}

## Top Factors for Headline Improvement
"""
            
            for factor, count in analysis['top_improvement_factors'].items():
                report += f"- {factor}: {count} occurrences\n"
            
            report += """
## Performance by Topic
"""
            
            for topic, stats in analysis['topic_performance'].items():
                if stats['count'] > 5:  # Only include topics with sufficient data
                    report += f"### {topic}\n"
                    report += f"- Sample size: {stats['count']} headlines\n"
                    report += f"- Average improvement: {stats['avg_improvement']:.1f}%\n"
                    report += f"- Improvement rate: {stats['improvement_rate']:.1%}\n\n"
            
            report += """
## Recommendations for Headline Writing

Based on the patterns observed in our data, here are the key strategies for writing more effective headlines:

1. **Optimize headline length** - Headlines between 40-60 characters tend to perform best
2. **Use specific numbers** when relevant to make claims more concrete
3. **Create curiosity gaps** without being clickbait
4. **Use active voice and strong verbs** to create more dynamic headlines
5. **Match tone to topic** - Different topics require different approaches

## Next Steps

The headline improvement system will continue to collect data and refine these recommendations.
"""
            
            # Save report to file
            with open('headline_improvement_report.md', 'w') as f:
                f.write(report)
            
            return True
            
        except Exception as e:
            logging.error(f"Error generating headline report: {e}")
            return False