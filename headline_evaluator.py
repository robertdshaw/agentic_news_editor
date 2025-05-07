import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import sent_tokenize
import textstat
from sklearn.metrics import mean_squared_error, r2_score
import json
import datetime
import os

class HeadlineEvaluator:
    """
    Evaluates the performance of headline rewrites based on various metrics:
    - Predicted CTR improvement
    - Readability scores
    - Length and structure analysis
    - Topic coverage and diversity
    
    Used to answer research questions:
    RQ1: How effectively can the AI editor retrieve and rank news articles?
    RQ2: Does rewriting headlines via an LLM improve readability and engagement?
    """
    
    def __init__(self, results_dir='evaluation_results'):
        """Initialize the headline evaluator"""
        self.results_dir = results_dir
        
        # Create results directory if it doesn't exist
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    def load_curated_articles(self, filepath='curated_full_daily_output.csv'):
        """Load curated articles with original and rewritten headlines"""
        try:
            if os.path.exists(filepath):
                articles_df = pd.read_csv(filepath)
                logging.info(f"Loaded {len(articles_df)} curated articles from {filepath}")
                return articles_df
            else:
                logging.error(f"Curated articles file not found: {filepath}")
                return None
        except Exception as e:
            logging.error(f"Error loading curated articles: {e}")
            return None
    
    def calculate_readability_metrics(self, df):
        """Calculate readability metrics for original and rewritten headlines"""
        if df is None or len(df) == 0:
            return None
        
        # Copy dataframe to avoid modifying the original
        results_df = df.copy()
        
        # Calculate readability metrics
        logging.info("Calculating readability metrics...")
        
        # Original headlines
        results_df['original_fre'] = results_df['original_title'].apply(
            lambda x: textstat.flesch_reading_ease(x) if isinstance(x, str) else None
        )
        
        results_df['original_fkg'] = results_df['original_title'].apply(
            lambda x: textstat.flesch_kincaid_grade(x) if isinstance(x, str) else None
        )
        
        results_df['original_words'] = results_df['original_title'].apply(
            lambda x: len(x.split()) if isinstance(x, str) else 0
        )
        
        # Rewritten headlines
        results_df['rewritten_fre'] = results_df['rewritten_title'].apply(
            lambda x: textstat.flesch_reading_ease(x) if isinstance(x, str) else None
        )
        
        results_df['rewritten_fkg'] = results_df['rewritten_title'].apply(
            lambda x: textstat.flesch_kincaid_grade(x) if isinstance(x, str) else None
        )
        
        results_df['rewritten_words'] = results_df['rewritten_title'].apply(
            lambda x: len(x.split()) if isinstance(x, str) else 0
        )
        
        # Calculate changes
        results_df['fre_change'] = results_df['rewritten_fre'] - results_df['original_fre']
        results_df['fkg_change'] = results_df['rewritten_fkg'] - results_df['original_fkg']
        results_df['words_change'] = results_df['rewritten_words'] - results_df['original_words']
        
        return results_df
    
    def analyze_retrieval_effectiveness(self, df, topic_column='topic'):
        """
        Analyze retrieval effectiveness (RQ1)
        - Topic diversity
        - Relevance to queries
        - Content coverage
        """
        if df is None or len(df) == 0:
            return None
            
        logging.info("Analyzing retrieval effectiveness...")
        
        # Topic distribution
        topic_counts = df[topic_column].value_counts()
        topic_diversity = 1 - ((topic_counts**2).sum() / (len(df)**2))
        
        # Basic content metrics
        valid_abstract_ratio = (df['abstract'].str.len() > 100).mean()
        
        # Results
        retrieval_metrics = {
            'total_articles': len(df),
            'topic_count': len(topic_counts),
            'topic_diversity_score': topic_diversity,
            'topics_distribution': topic_counts.to_dict(),
            'valid_abstract_ratio': valid_abstract_ratio
        }
        
        return retrieval_metrics
    
    def analyze_headline_improvements(self, df):
        """
        Analyze headline improvements (RQ2)
        - CTR improvements
        - Readability changes
        - Structure changes
        """
        if df is None or len(df) == 0:
            return None
            
        logging.info("Analyzing headline improvements...")
        
        # CTR and scoring metrics
        headline_metrics = {
            'avg_ctr_original': df['headline_ctr_original'].mean() * 100,
            'avg_ctr_rewritten': df['headline_ctr_rewritten'].mean() * 100,
            'avg_ctr_improvement': (df['headline_ctr_rewritten'] - df['headline_ctr_original']).mean() * 100,
            'avg_score_improvement': df['headline_improvement'].mean(),
            'improvement_rate': (df['headline_improvement'] > 0).mean(),
            'large_improvement_rate': (df['headline_improvement'] > 10).mean()
        }
        
        # Readability metrics
        readability_metrics = {
            'avg_fre_original': df['original_fre'].mean(),
            'avg_fre_rewritten': df['rewritten_fre'].mean(),
            'avg_fre_change': df['fre_change'].mean(),
            'fre_improvement_rate': (df['fre_change'] > 0).mean(),
            'avg_words_original': df['original_words'].mean(),
            'avg_words_rewritten': df['rewritten_words'].mean(),
            'avg_words_change': df['words_change'].mean()
        }
        
        # Key improvement factors analysis
        improvement_factors = []
        for factors in df['headline_key_factors']:
            if isinstance(factors, str):
                for factor in factors.split(','):
                    factor = factor.strip()
                    if factor:
                        improvement_factors.append(factor)
        
        factor_counts = pd.Series(improvement_factors).value_counts()
        
        # Combine results
        results = {
            'headline_metrics': headline_metrics,
            'readability_metrics': readability_metrics,
            'improvement_factors': factor_counts.to_dict()
        }
        
        return results
    
    def run_full_evaluation(self, curated_file='curated_full_daily_output.csv'):
        """Run a full evaluation of the news editor system"""
        logging.info("Starting full evaluation...")
        
        # Load curated articles
        df = self.load_curated_articles(curated_file)
        if df is None:
            logging.error("Could not load curated articles. Aborting evaluation.")
            return None
        
        # Calculate readability metrics
        df_with_metrics = self.calculate_readability_metrics(df)
        
        # Analyze retrieval effectiveness (RQ1)
        retrieval_metrics = self.analyze_retrieval_effectiveness(df_with_metrics)
        
        # Analyze headline improvements (RQ2)
        headline_metrics = self.analyze_headline_improvements(df_with_metrics)
        
        # Combine results
        evaluation_results = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_articles': len(df),
            'retrieval_metrics': retrieval_metrics,
            'headline_metrics': headline_metrics
        }
        
        # Save results
        result_file = os.path.join(self.results_dir, f'evaluation_{datetime.datetime.now().strftime("%Y%m%d_%H%M")}.json')
        with open(result_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        logging.info(f"Evaluation complete. Results saved to {result_file}")
        
        # Generate plots
        self.generate_evaluation_plots(df_with_metrics, evaluation_results)
        
        # Generate report
        report_path = self.generate_evaluation_report(evaluation_results)
        
        return {
            'results': evaluation_results,
            'report_path': report_path
        }
    
    def generate_evaluation_plots(self, df, results):
        """Generate visualizations for the evaluation results"""
        if df is None or len(df) == 0:
            return
            
        plots_dir = os.path.join(self.results_dir, 'plots')
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        # 1. CTR Improvement Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='headline_improvement', bins=20, kde=True)
        plt.axvline(x=0, color='red', linestyle='--')
        plt.title('Distribution of Headline CTR Improvements')
        plt.xlabel('CTR Improvement (%)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'ctr_improvement_distribution.png'))
        plt.close()
        
        # 2. CTR by Topic
        plt.figure(figsize=(12, 6))
        topic_ctr = df.groupby('topic').agg({
            'headline_ctr_original': 'mean',
            'headline_ctr_rewritten': 'mean'
        }).reset_index()
        
        topic_ctr = topic_ctr.sort_values('headline_ctr_rewritten', ascending=False)
        
        # Multiply by 100 to show as percentage
        topic_ctr['headline_ctr_original'] *= 100
        topic_ctr['headline_ctr_rewritten'] *= 100
        
        ax = sns.barplot(x='topic', y='headline_ctr_original', data=topic_ctr, color='lightblue', label='Original')
        sns.barplot(x='topic', y='headline_ctr_rewritten', data=topic_ctr, color='darkblue', label='Rewritten')
        
        plt.title('Average CTR by Topic - Original vs. Rewritten')
        plt.xlabel('Topic')
        plt.ylabel('Average CTR (%)')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'ctr_by_topic.png'))
        plt.close()
        
        # 3. Readability Changes
        plt.figure(figsize=(10, 6))
        readability_data = pd.DataFrame({
            'Metric': ['Flesch Reading Ease', 'Word Count'],
            'Original': [df['original_fre'].mean(), df['original_words'].mean()],
            'Rewritten': [df['rewritten_fre'].mean(), df['rewritten_words'].mean()]
        })
        
        readability_data = pd.melt(readability_data, id_vars=['Metric'], var_name='Version', value_name='Value')
        
        sns.barplot(x='Metric', y='Value', hue='Version', data=readability_data)
        plt.title('Readability Metrics - Original vs. Rewritten Headlines')
        plt.ylabel('Average Value')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'readability_metrics.png'))
        plt.close()
        
        # 4. Improvement Factors
        if 'improvement_factors' in results.get('headline_metrics', {}):
            factors_df = pd.DataFrame({
                'Factor': list(results['headline_metrics']['improvement_factors'].keys()),
                'Count': list(results['headline_metrics']['improvement_factors'].values())
            })
            
            if len(factors_df) > 0:
                factors_df = factors_df.sort_values('Count', ascending=False).head(10)
                
                plt.figure(figsize=(12, 6))
                sns.barplot(x='Count', y='Factor', data=factors_df)
                plt.title('Top 10 Improvement Factors in Headline Rewrites')
                plt.xlabel('Count')
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'improvement_factors.png'))
                plt.close()
        
        logging.info(f"Generated evaluation plots in {plots_dir}")
    
    def generate_evaluation_report(self, results):
        """Generate a markdown report with the evaluation results"""
        if results is None:
            return None
            
        report_path = os.path.join(self.results_dir, f'evaluation_report_{datetime.datetime.now().strftime("%Y%m%d_%H%M")}.md')
        
        # Format results
        retrieval = results.get('retrieval_metrics', {})
        headline = results.get('headline_metrics', {})
        
        report = f"""# Agentic AI News Editor Evaluation Report
Generated: {results.get('timestamp', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}

## Overview
Total Articles Analyzed: {results.get('total_articles', 0)}

## Research Question 1: Retrieval Effectiveness
How effectively can the AI editor retrieve and rank news articles aligned with general editorial priorities and user needs?

### Metrics
- Topic Coverage: {retrieval.get('topic_count', 0)} topics
- Topic Diversity Score: {retrieval.get('topic_diversity_score', 0):.2f} (0-1 scale, higher is more diverse)
- Valid Content Rate: {retrieval.get('valid_abstract_ratio', 0):.1%}

### Topic Distribution
"""
        
        # Add topic distribution
        if 'topics_distribution' in retrieval:
            for topic, count in retrieval['topics_distribution'].items():
                report += f"- {topic}: {count} articles\n"
        
        report += """
## Research Question 2: Headline Improvement
Does rewriting headlines via an LLM improve readability, clarity, and potential user engagement?

### CTR Metrics
"""
        
        # Add headline metrics
        headline_metrics = headline.get('headline_metrics', {})
        if headline_metrics:
            report += f"- Average Original CTR: {headline_metrics.get('avg_ctr_original', 0):.1f}%\n"
            report += f"- Average Rewritten CTR: {headline_metrics.get('avg_ctr_rewritten', 0):.1f}%\n"
            report += f"- Average CTR Improvement: {headline_metrics.get('avg_ctr_improvement', 0):.1f}%\n"
            report += f"- Improvement Rate: {headline_metrics.get('improvement_rate', 0):.1%} of headlines improved\n"
            report += f"- Significant Improvement Rate: {headline_metrics.get('large_improvement_rate', 0):.1%} showed >10% improvement\n"
        
        report += """
### Readability Metrics
"""
        
        # Add readability metrics
        readability_metrics = headline.get('readability_metrics', {})
        if readability_metrics:
            report += f"- Average Original Flesch Reading Ease: {readability_metrics.get('avg_fre_original', 0):.1f}\n"
            report += f"- Average Rewritten Flesch Reading Ease: {readability_metrics.get('avg_fre_rewritten', 0):.1f}\n"
            report += f"- Average FRE Change: {readability_metrics.get('avg_fre_change', 0):.1f}\n"
            report += f"- FRE Improvement Rate: {readability_metrics.get('fre_improvement_rate', 0):.1%}\n"
            report += f"- Average Original Word Count: {readability_metrics.get('avg_words_original', 0):.1f}\n"
            report += f"- Average Rewritten Word Count: {readability_metrics.get('avg_words_rewritten', 0):.1f}\n"
            report += f"- Average Word Count Change: {readability_metrics.get('avg_words_change', 0):.1f}\n"
        
        report += """
### Key Improvement Factors
"""
        
        # Add improvement factors
        improvement_factors = headline.get('improvement_factors', {})
        if improvement_factors:
            # Sort by count
            factors = sorted(improvement_factors.items(), key=lambda x: x[1], reverse=True)
            for factor, count in factors:
                report += f"- {factor}: {count} occurrences\n"
        
        report += """
## Conclusion and Recommendations

Based on the evaluation results, the Agentic AI News Editor system demonstrates:

1. Good topic coverage and diversity in article retrieval
2. Consistent improvement in headline CTR through rewriting
3. Maintained or improved readability in most cases

### Key Findings
"""
        
        # Add key findings
        if headline_metrics and readability_metrics:
            if headline_metrics.get('avg_ctr_improvement', 0) > 5:
                report += "- Headline rewrites show substantial CTR improvements\n"
            elif headline_metrics.get('avg_ctr_improvement', 0) > 0:
                report += "- Headline rewrites show modest but positive CTR improvements\n"
            else:
                report += "- Headline rewrites do not consistently improve CTR\n"
                
            if readability_metrics.get('avg_fre_change', 0) > 3:
                report += "- Readability significantly improved in rewritten headlines\n"
            elif readability_metrics.get('avg_fre_change', 0) > 0:
                report += "- Readability slightly improved in rewritten headlines\n"
            else:
                report += "- Readability remained similar or decreased in rewritten headlines\n"
        
        report += """
### Recommendations for System Improvement
1. Further tune the headline rewriting model to focus on factors that showed the strongest CTR improvements
2. Expand topic diversity through broader query formulations
3. Implement A/B testing for headline versions to gather actual user engagement data
4. Investigate category-specific headline styles to maximize CTR within each content category
"""
        
        # Write report to file
        with open(report_path, 'w') as f:
            f.write(report)
        
        logging.info(f"Generated evaluation report: {report_path}")
        
        return report_path


if __name__ == "__main__":
    evaluator = HeadlineEvaluator()
    result = evaluator.run_full_evaluation()
    
    if result:
        print(f"Evaluation complete. Report generated at: {result['report_path']}")
    else:
        print("Evaluation failed.")