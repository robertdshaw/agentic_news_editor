import pandas as pd
import json
import os
import time
from datetime import datetime

class HeadlineLearningLoop:
    """
    A system that tracks headline performance and creates a learning loop to improve headlines over time.
    
    This class:
    1. Stores headline pairs with metrics
    2. Provides insights into what changes improve headlines
    3. Helps refine prompt engineering based on successful patterns
    """
    
    def __init__(self, data_file="headline_learning_data.csv"):
        """Initialize the headline learning system"""
        self.data_file = data_file
        self.data = self._load_data()
        
    def _load_data(self):
        """Load existing headline data if available"""
        if os.path.exists(self.data_file):
            try:
                return pd.read_csv(self.data_file)
            except Exception as e:
                print(f"Error loading headline data: {e}")
                return self._create_empty_dataframe()
        else:
            return self._create_empty_dataframe()
    
    def _create_empty_dataframe(self):
        """Create an empty dataframe with the required columns"""
        return pd.DataFrame(columns=[
            'timestamp', 'original_headline', 'rewritten_headline', 
            'topic', 'category', 'original_score', 'rewritten_score',
            'improvement', 'ctr_original', 'ctr_rewritten',
            'key_factors', 'word_count_original', 'word_count_rewritten'
        ])
    
    def save_data(self):
        """Save the headline data to CSV"""
        try:
            self.data.to_csv(self.data_file, index=False)
            return True
        except Exception as e:
            print(f"Error saving headline data: {e}")
            return False
    
    def add_headline_pair(self, original, rewritten, metrics, topic=None, category=None):
        """
        Add a headline pair with metrics to the learning system
        
        Args:
            original (str): Original headline
            rewritten (str): Rewritten headline
            metrics (dict): Dictionary of metrics from HeadlineMetrics.compare_headlines()
            topic (str, optional): The news topic
            category (str, optional): The article category
            
        Returns:
            bool: Success status
        """
        if not original or not rewritten or not metrics:
            return False
        
        try:
            # Create a new record
            new_record = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'original_headline': original,
                'rewritten_headline': rewritten,
                'topic': topic or '',
                'category': category or '',
                'original_score': metrics.get('original_score', 0),
                'rewritten_score': metrics.get('rewritten_score', 0),
                'improvement': metrics.get('score_percent_change', 0),
                'ctr_original': metrics.get('original_ctr', 0) * 100,  # Convert to percentage
                'ctr_rewritten': metrics.get('rewritten_ctr', 0) * 100,
                'key_factors': ', '.join(metrics.get('key_improvements', [])),
                'word_count_original': len(original.split()),
                'word_count_rewritten': len(rewritten.split())
            }
            
            # Add to dataframe
            self.data = pd.concat([self.data, pd.DataFrame([new_record])], ignore_index=True)
            
            # Save periodically (not on every addition to avoid performance issues)
            if len(self.data) % 10 == 0:
                self.save_data()
                
            return True
        except Exception as e:
            print(f"Error adding headline pair: {e}")
            return False
    
    def add_headlines_from_dataframe(self, df, topic_column='topic'):
        """
        Add multiple headline pairs from a dataframe
        
        Args:
            df: DataFrame with headline data
            topic_column: Column name for topic information
            
        Returns:
            int: Number of headlines added
        """
        if not isinstance(df, pd.DataFrame) or len(df) == 0:
            return 0
            
        count = 0
        for _, row in df.iterrows():
            if ('title' in row and 'rewritten_title' in row and 
                'headline_score_original' in row and 'headline_score_rewritten' in row):
                
                # Create metrics dict from dataframe columns
                metrics = {
                    'original_score': row.get('headline_score_original', 0),
                    'rewritten_score': row.get('headline_score_rewritten', 0),
                    'score_percent_change': row.get('headline_improvement', 0),
                    'original_ctr': row.get('headline_ctr_original', 0) / 100,  # Convert from percentage
                    'rewritten_ctr': row.get('headline_ctr_rewritten', 0) / 100,
                    'key_improvements': row.get('headline_key_factors', '').split(', ')
                }
                
                topic = row.get(topic_column, '')
                
                # Add the headline pair
                if self.add_headline_pair(row['title'], row['rewritten_title'], metrics, topic):
                    count += 1
        
        # Save all data after bulk addition
        self.save_data()
        return count
    
    def get_insights(self, min_improvement=10, limit=10):
        """
        Get insights about what makes effective headlines
        
        Args:
            min_improvement (float): Minimum improvement percentage to consider
            limit (int): Maximum number of examples to return
            
        Returns:
            dict: Insights and examples of effective headline changes
        """
        if len(self.data) < 10:
            return {"error": "Not enough data to generate meaningful insights"}
        
        # Filter for headlines with significant improvements
        improved = self.data[self.data['improvement'] >= min_improvement].copy()
        
        if len(improved) == 0:
            return {"error": "No headlines with significant improvements found"}
        
        # Calculate average improvements
        avg_improvement = improved['improvement'].mean()
        avg_length_change = (improved['word_count_rewritten'] - improved['word_count_original']).mean()
        
        # Find the most common key factors
        all_factors = []
        for factors in improved['key_factors']:
            if isinstance(factors, str) and factors:
                all_factors.extend([f.strip() for f in factors.split(',')])
        
        factor_counts = {}
        for factor in all_factors:
            if factor:
                factor_counts[factor] = factor_counts.get(factor, 0) + 1
        
        # Sort factors by frequency
        top_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Get examples of the most improved headlines
        best_examples = improved.sort_values('improvement', ascending=False).head(limit)
        examples = []
        
        for _, row in best_examples.iterrows():
            examples.append({
                'original': row['original_headline'],
                'rewritten': row['rewritten_headline'],
                'improvement': row['improvement'],
                'factors': row['key_factors']
            })
        
        # Generate insights
        return {
            "headline_count": len(self.data),
            "improved_count": len(improved),
            "avg_improvement": avg_improvement,
            "avg_length_change": avg_length_change,
            "top_improvement_factors": top_factors,
            "best_examples": examples,
            "recommendations": self._generate_recommendations(improved, top_factors)
        }
    
    def _generate_recommendations(self, improved_df, top_factors):
        """Generate recommendations for headline improvements"""
        recommendations = []
        
        # Check if shorter headlines perform better
        length_correlation = (improved_df['word_count_rewritten'] - improved_df['word_count_original']).mean()
        if length_correlation < -1:
            recommendations.append("Shorter headlines tend to perform better. Consider reducing headline length.")
        elif length_correlation > 1:
            recommendations.append("Longer, more descriptive headlines are performing well in your content.")
        
        # Add recommendations based on top factors
        if top_factors:
            recommendations.append(f"Most effective headline improvements: {', '.join([f[0] for f in top_factors[:3]])}")
        
        # Add more specific recommendations based on patterns
        if 'Added specific numbers' in [f[0] for f in top_factors]:
            recommendations.append("Prioritize adding specific numbers to headlines when possible.")
        
        if 'Added power words' in [f[0] for f in top_factors]:
            recommendations.append("Increase use of power words like 'exclusive', 'revealed', 'essential'.")
        
        if 'Improved readability' in [f[0] for f in top_factors]:
            recommendations.append("Focus on simplifying complex language for better readability.")
        
        return recommendations
    
    def export_prompt_suggestions(self):
        """
        Generate prompt suggestions based on headline performance data
        
        Returns:
            str: Suggested prompt improvements
        """
        insights = self.get_insights(min_improvement=15)
        
        if 'error' in insights:
            return "Not enough data to generate prompt suggestions yet."
        
        # Create suggested prompt improvements
        suggestions = [
            "# Headline Prompt Improvement Suggestions",
            f"Based on analysis of {insights['headline_count']} headlines:",
            ""
        ]
        
        # Add recommendations
        suggestions.append("## Key Recommendations")
        for rec in insights['recommendations']:
            suggestions.append(f"- {rec}")
        
        # Add example transformations
        suggestions.append("\n## Effective Headline Transformations")
        suggestions.append("Consider adding these as examples in your prompt:")
        
        for i, example in enumerate(insights['best_examples'][:5]):
            suggestions.append(f"\nOriginal: {example['original']}")
            suggestions.append(f"Better: {example['rewritten']}")
            suggestions.append(f"Improvement: {example['improvement']:.1f}%")
            
        return "\n".join(suggestions)
    
    def prompt_improvement_report(self, output_file="headline_improvement_report.md"):
        """
        Generate and save a complete headline improvement report
        
        Args:
            output_file (str): File to save the report
            
        Returns:
            bool: Success status
        """
        try:
            insights = self.get_insights(min_improvement=5, limit=15)
            
            if 'error' in insights:
                with open(output_file, 'w') as f:
                    f.write(f"# Headline Improvement Report\n\n{insights['error']}\n\nContinue collecting more headline data.")
                return True
            
            # Create the report
            report = [
                "# Headline Improvement Report",
                f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                f"\nAnalysis based on {insights['headline_count']} headlines.",
                "",
                "## Summary",
                f"- {insights['improved_count']} headlines showed significant improvement",
                f"- Average improvement: {insights['avg_improvement']:.1f}%",
                f"- Average word count change: {insights['avg_length_change']:.1f} words",
                "",
                "## Top Factors that Improve Headlines",
            ]
            
            # Add top factors
            for factor, count in insights['top_improvement_factors']:
                report.append(f"- {factor} ({count} occurrences)")
            
            # Add recommendations
            report.append("\n## Recommendations for Better Headlines")
            for rec in insights['recommendations']:
                report.append(f"- {rec}")
            
            # Add examples
            report.append("\n## Best Headline Transformations")
            for i, example in enumerate(insights['best_examples'][:10]):
                report.append(f"\n### Example {i+1} (+{example['improvement']:.1f}%)")
                report.append(f"**Original:** {example['original']}")
                report.append(f"**Rewritten:** {example['rewritten']}")
                report.append(f"**Key improvements:** {example['factors']}")
            
            # Add prompt suggestions
            report.append("\n## Suggested Prompt Updates")
            report.append("Consider updating your headline generation prompt with these examples and guidelines:")
            
            for rec in insights['recommendations']:
                report.append(f"- {rec}")
                
            report.append("\nExample transformations to include in your prompt:")
            for i, example in enumerate(insights['best_examples'][:5]):
                report.append(f"\nOriginal: {example['original']}")
                report.append(f"Better: {example['rewritten']}")
            
            # Save the report
            with open(output_file, 'w') as f:
                f.write("\n".join(report))
            
            return True
            
        except Exception as e:
            print(f"Error generating headline improvement report: {e}")
            return False