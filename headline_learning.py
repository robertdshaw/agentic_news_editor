import pandas as pd
import numpy as np
import os
import logging
import pickle
import datetime
import json
from pathlib import Path
from headline_metrics import HeadlineMetrics

class HeadlineLearningLoop:
    """
    A learning system that continuously improves headline rewriting by:
    1. Collecting pairs of original and rewritten headlines
    2. Analyzing their performance metrics using the trained CTR model
    3. Generating insights about what makes headlines effective
    4. Providing feedback for headline writers
    """
    
    def __init__(self, data_file="headline_learning_data.csv", model_path="model_output/ctr_model.pkl"):
        """Initialize the headline learning system"""
        self.data_file = data_file
        self.model_path = model_path
        
        # Initialize metrics analyzer with the CTR model
        try:
            self.metrics_analyzer = HeadlineMetrics(model_path=model_path)
            logging.info("Initialized HeadlineMetrics with CTR model")
        except Exception as e:
            logging.error(f"Error initializing HeadlineMetrics: {e}")
            raise
        
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
            logging.info("Created new headline learning dataset")
    
    def add_headline_pair(self, original, rewritten, topic=None):
        """Add a single pair of headlines to the learning system"""
        try:
            # Calculate metrics using the CTR model
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
            
            logging.info(f"Added headline pair: {comparison['score_percent_change']:.1f}% improvement")
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
                
                # Check if we already have metrics calculated
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
            logging.info(f"Added {count} headline pairs from dataframe")
        
        return count
    
    def analyze_headline_patterns(self):
        """Analyze headline patterns to extract insights"""
        if len(self.data) < 20:
            return {
                'status': 'insufficient_data',
                'message': f'Need at least 20 headline pairs for analysis (current: {len(self.data)})'
            }
        
        try:
            # Calculate improvement stats
            improvements = self.data['headline_improvement']
            avg_improvement = improvements.mean()
            median_improvement = improvements.median()
            improvement_rate = (improvements > 0).mean()
            
            # Find common improvement factors
            all_factors = []
            for factors in self.data['headline_key_factors']:
                if isinstance(factors, str) and factors.strip():
                    all_factors.extend([f.strip() for f in factors.split(',') if f.strip()])
            
            if all_factors:
                factor_counts = pd.Series(all_factors).value_counts()
                top_factors = factor_counts.head(10).to_dict()
            else:
                top_factors = {}
            
            # Look at topic performance
            topic_performance = {}
            for topic in self.data['topic'].unique():
                topic_data = self.data[self.data['topic'] == topic]
                if len(topic_data) > 3:  # Only include topics with sufficient data
                    topic_performance[topic] = {
                        'count': len(topic_data),
                        'avg_improvement': topic_data['headline_improvement'].mean(),
                        'improvement_rate': (topic_data['headline_improvement'] > 0).mean(),
                        'avg_ctr_original': topic_data['headline_ctr_original'].mean(),
                        'avg_ctr_rewritten': topic_data['headline_ctr_rewritten'].mean()
                    }
            
            # Analyze CTR distributions
            ctr_analysis = {
                'original_ctr_stats': {
                    'mean': self.data['headline_ctr_original'].mean(),
                    'median': self.data['headline_ctr_original'].median(),
                    'std': self.data['headline_ctr_original'].std()
                },
                'rewritten_ctr_stats': {
                    'mean': self.data['headline_ctr_rewritten'].mean(),
                    'median': self.data['headline_ctr_rewritten'].median(),
                    'std': self.data['headline_ctr_rewritten'].std()
                }
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
                'topic_performance': topic_performance,
                'ctr_analysis': ctr_analysis
            }
            
        except Exception as e:
            logging.error(f"Error analyzing headline patterns: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def get_improvement_recommendations(self):
        """Get specific recommendations based on collected data"""
        analysis = self.analyze_headline_patterns()
        
        if analysis['status'] != 'success':
            return ["Need more data to provide recommendations"]
        
        recommendations = []
        
        # Based on top improvement factors
        top_factors = analysis['top_improvement_factors']
        for factor, count in list(top_factors.items())[:5]:
            recommendations.append(f"✓ {factor} (effective in {count} cases)")
        
        # Based on topic performance
        best_topics = sorted(
            analysis['topic_performance'].items(),
            key=lambda x: x[1]['avg_improvement'],
            reverse=True
        )[:3]
        
        if best_topics:
            recommendations.append(f"Focus on {best_topics[0][0]} topics (avg improvement: {best_topics[0][1]['avg_improvement']:.1f}%)")
        
        # CTR-based recommendations
        ctr_analysis = analysis['ctr_analysis']
        avg_original = ctr_analysis['original_ctr_stats']['mean']
        avg_rewritten = ctr_analysis['rewritten_ctr_stats']['mean']
        
        recommendations.append(f"Current rewriting improves CTR from {avg_original:.4f} to {avg_rewritten:.4f} on average")
        
        return recommendations
    
    def generate_improvement_report(self):
        """Generate a comprehensive markdown report"""
        analysis = self.analyze_headline_patterns()
        
        if analysis['status'] != 'success':
            return f"Cannot generate report: {analysis.get('message', 'Analysis failed')}"
        
        try:
            report = f"""# Headline Improvement Analysis Report
Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}

## Executive Summary
- **Total headline pairs analyzed**: {analysis['sample_size']}
- **Average CTR improvement**: {analysis['overall_stats']['avg_improvement']:.1f}%
- **Headlines that improved**: {analysis['overall_stats']['improvement_rate']:.1%}
- **Median improvement**: {analysis['overall_stats']['median_improvement']:.1f}%

## CTR Performance Analysis
- **Original headlines average CTR**: {analysis['ctr_analysis']['original_ctr_stats']['mean']:.4f}
- **Rewritten headlines average CTR**: {analysis['ctr_analysis']['rewritten_ctr_stats']['mean']:.4f}
- **CTR improvement factor**: {analysis['ctr_analysis']['rewritten_ctr_stats']['mean'] / analysis['ctr_analysis']['original_ctr_stats']['mean']:.2f}x

## Top Improvement Factors
"""
            
            for factor, count in analysis['top_improvement_factors'].items():
                report += f"- **{factor}**: {count} occurrences\n"
            
            report += """
## Performance by Topic
"""
            
            for topic, stats in analysis['topic_performance'].items():
                report += f"### {topic}\n"
                report += f"- Sample size: {stats['count']} headlines\n"
                report += f"- Average improvement: {stats['avg_improvement']:.1f}%\n"
                report += f"- Improvement rate: {stats['improvement_rate']:.1%}\n"
                report += f"- Average CTR: {stats['avg_ctr_original']:.4f} → {stats['avg_ctr_rewritten']:.4f}\n\n"
            
            report += """
## Actionable Recommendations

Based on the analysis of headline performance:

### 1. Most Effective Techniques
"""
            
            recommendations = self.get_improvement_recommendations()
            for rec in recommendations:
                report += f"- {rec}\n"
            
            report += f"""
### 2. CTR Optimization Strategy
- Current model achieves {analysis['overall_stats']['improvement_rate']:.1%} improvement rate
- Focus on techniques that consistently deliver >10% CTR improvement
- Pay attention to topic-specific patterns

### 3. Next Steps
1. Continue collecting headline pairs to improve model accuracy
2. Focus on underperforming topics: {', '.join([t for t, s in analysis['topic_performance'].items() if s['avg_improvement'] < 0])}
3. Experiment with top-performing improvement factors in new headlines

---
*Report generated by Headline Learning System*
"""
            
            # Save report to file
            with open('headline_improvement_report.md', 'w') as f:
                f.write(report)
            
            return report
            
        except Exception as e:
            logging.error(f"Error generating headline report: {e}")
            return f"Error generating report: {str(e)}"
    
    def export_data_for_analysis(self, filename=None):
        """Export the collected data for external analysis"""
        if filename is None:
            filename = f"headline_data_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        
        try:
            # Add calculated features for analysis
            export_data = self.data.copy()
            
            # Add feature analysis for each headline
            for i, row in export_data.iterrows():
                orig_features = self.metrics_analyzer.extract_features(row['original_title'])
                new_features = self.metrics_analyzer.extract_features(row['rewritten_title'])
                
                export_data.at[i, 'original_length'] = orig_features['title_length'].iloc[0]
                export_data.at[i, 'rewritten_length'] = new_features['title_length'].iloc[0]
                export_data.at[i, 'original_word_count'] = orig_features['title_word_count'].iloc[0]
                export_data.at[i, 'rewritten_word_count'] = new_features['title_word_count'].iloc[0]
            
            export_data.to_csv(filename, index=False)
            logging.info(f"Exported data to {filename}")
            return filename
        except Exception as e:
            logging.error(f"Error exporting data: {e}")
            return None