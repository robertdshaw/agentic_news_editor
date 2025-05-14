# Agentic AI News Editor: Enhanced EDA Pipeline
# -----------------------------------------------
# This pipeline combines analysis of the Microsoft MIND dataset to prepare data
# for an agentic AI system that selects, ranks, and rewrites news headlines
# Updated to include comprehensive headline analysis and CTR optimization insights

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from datetime import datetime
from collections import Counter
import re
import json
from textstat import flesch_reading_ease
import warnings
warnings.filterwarnings('ignore')

# Set visualization defaults
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Create output directory for results
output_dir = 'agentic_news_editor'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    os.makedirs(f'{output_dir}/plots')
    os.makedirs(f'{output_dir}/processed_data')

print("# Agentic AI News Editor - Enhanced EDA Pipeline")
print("Starting EDA pipeline... Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("-" * 80)

# ===============================================================
# PART 1: DATA LOADING AND INITIAL EXPLORATION
# ===============================================================
print("\n## PART 1: Data Loading and Initial Exploration")

# Load news data
news_cols = ["newsID", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
print("Loading news data...")
news_df = pd.read_csv("train_data/news.tsv", sep="\t", header=None, names=news_cols)
print(f"News data loaded: {news_df.shape[0]} rows, {news_df.shape[1]} columns")

# Loading behaviors data (with sampling option)
print("Loading behaviors data...")
behaviors_cols = ["impression_id", "user_id", "time", "history", "impressions"]
# For full analysis, remove nrows parameter
behaviors_df = pd.read_csv("train_data/behaviors.tsv", sep="\t", header=None, names=behaviors_cols, nrows=100000)
print(f"Behaviors data loaded: {behaviors_df.shape[0]} rows, {behaviors_df.shape[1]} columns")

# Store news_ids for reference
news_ids = set(news_df["newsID"])

# ===============================================================
# PART 2: COMPREHENSIVE NEWS DATA ANALYSIS
# ===============================================================
print("\n## PART 2: Comprehensive News Analysis")

# Check for duplicates in news data
duplicate_news = news_df.duplicated(subset=["newsID"]).sum()
print(f"Duplicate news IDs: {duplicate_news}")

# Check for duplicate titles
duplicate_titles = news_df.duplicated(subset=["title"], keep=False).sum()
title_counts = news_df['title'].value_counts()
dupe_title_groups = title_counts[title_counts > 1]
print(f"Duplicate news titles: {duplicate_titles}")
print(f"{len(dupe_title_groups)} unique titles are repeated")

# Check missing values and handle them
print("\nMissing values in news data:")
print(news_df.isnull().sum())
print(f"Percentage of abstracts missing: {news_df['abstract'].isnull().mean()*100:.2f}%")

# Clean missing values
news_df['abstract'] = news_df['abstract'].fillna("")
news_df['title_entities'] = news_df['title_entities'].fillna("[]")
news_df['abstract_entities'] = news_df['abstract_entities'].fillna("[]")

# Calculate text lengths
news_df["title_length"] = news_df["title"].str.len()
news_df["abstract_length"] = news_df["abstract"].str.len()
news_df["title_word_count"] = news_df["title"].str.split().str.len()

# Check for quality issues
print(f"\nVery short titles (<10 chars): {(news_df['title_length'] < 10).sum()}")
print(f"Very short abstracts (<20 chars): {(news_df['abstract_length'] < 20).sum()}")

# Advanced text analysis features for headline optimization
print("\n## Advanced Headline Feature Analysis")

# 1. Calculate reading ease scores
def calculate_reading_scores(df):
    """Calculate reading ease scores for titles"""
    df['title_reading_ease'] = df['title'].apply(lambda x: flesch_reading_ease(x) if isinstance(x, str) and len(x) > 0 else np.nan)
    df['abstract_reading_ease'] = df['abstract'].apply(lambda x: flesch_reading_ease(x) if isinstance(x, str) and len(x) > 0 else np.nan)
    return df

news_df = calculate_reading_scores(news_df)

# 2. Headline pattern analysis (based on your EDA findings)
def analyze_headline_features(df):
    """Extract features from headlines that correlate with CTR"""
    
    # Extract pattern features
    df['has_question'] = df['title'].str.contains('\?').astype(int)
    df['has_exclamation'] = df['title'].str.contains('!').astype(int)
    df['has_number'] = df['title'].str.contains(r'\d').astype(int)
    df['has_colon'] = df['title'].str.contains(':').astype(int)
    df['has_quotes'] = df['title'].str.contains(r'["\']').astype(int)
    df['has_ellipsis'] = df['title'].str.contains('\.\.\.').astype(int)
    
    # Title style features
    df['title_uppercase_ratio'] = df['title'].apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0)
    df['starts_with_number'] = df['title'].str.match(r'^\d').astype(int)
    df['starts_with_question'] = df['title'].str.match(r'^(Is|What|How|Why|When|Where)').astype(int)
    
    # Sentiment and emotion indicators
    df['has_superlative'] = df['title'].str.contains(r'\b(best|worst|most|least|biggest|smallest|first|last)\b', case=False).astype(int)
    df['has_temporal_words'] = df['title'].str.contains(r'\b(now|today|breaking|urgent|soon|latest)\b', case=False).astype(int)
    
    return df

news_df = analyze_headline_features(news_df)

# 3. Category and subcategory analysis
print("\nCategory distribution:")
category_counts = news_df["category"].value_counts()
print(category_counts)

# Create visualization of category distribution
plt.figure(figsize=(15, 6))
ax = category_counts.plot(kind='bar')
plt.title('News Articles by Category', fontsize=16)
plt.xlabel('Category', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45, ha='right')

# Add percentages to bars
total_articles = len(news_df)
for i, v in enumerate(category_counts.values):
    ax.text(i, v + 100, f'{v/total_articles*100:.1f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(f'{output_dir}/plots/category_distribution.png', dpi=300)
plt.close()

# ===============================================================
# PART 3: COMPREHENSIVE USER BEHAVIOR ANALYSIS
# ===============================================================
print("\n## PART 3: User Behavior Analysis")

# Check behaviors data quality
duplicate_impressions = behaviors_df.duplicated(subset=["impression_id"]).sum()
print(f"Duplicate impression IDs: {duplicate_impressions}")

print("\nMissing values in behaviors data:")
print(behaviors_df.isnull().sum())

# Analyze missing history data
missing_history_count = behaviors_df['history'].isna().sum()
missing_history_pct = (missing_history_count / len(behaviors_df)) * 100
print(f"Missing history: {missing_history_count} rows ({missing_history_pct:.2f}%)")

# Analyze users with missing history
missing_history_df = behaviors_df[behaviors_df['history'].isna()]
unique_users_missing = missing_history_df['user_id'].nunique()
print(f"Unique users with missing history: {unique_users_missing}")

# Process impressions data
print("\nProcessing impressions data...")

def process_impressions(df, sample_size=None):
    """Convert the impressions data into a flattened format for analysis"""
    if sample_size:
        df = df.head(sample_size)
    
    impressions_expanded = []
    
    for _, row in df.iterrows():
        try:
            impressions = row['impressions'].split()
            for item in impressions:
                if '-' in item:
                    news_id, clicked = item.split('-')
                    impressions_expanded.append({
                        'impression_id': row['impression_id'],
                        'user_id': row['user_id'],
                        'news_id': news_id,
                        'clicked': int(clicked),
                        'time': row['time']
                    })
        except Exception as e:
            print(f"Skipping row due to error: {e}")
            continue
    
    return pd.DataFrame(impressions_expanded)

# Process impressions
impressions_df = process_impressions(behaviors_df)
print(f"Expanded to {len(impressions_df)} impression records")

# Check for invalid news IDs
invalid_news_ids = impressions_df[~impressions_df["news_id"].isin(news_ids)]
print(f"Impression records with invalid news IDs: {len(invalid_news_ids)}")

# Overall CTR calculation
clicks = impressions_df["clicked"].sum()
total = len(impressions_df)
overall_ctr = clicks/total
print(f"\nOverall CTR: {overall_ctr:.4f} ({clicks} clicks out of {total} impressions)")

# ===============================================================
# PART 4: ADVANCED CTR ANALYSIS BY CATEGORY AND FEATURES
# ===============================================================
print("\n## PART 4: Advanced CTR Analysis")

# Calculate CTR by article
article_ctr = impressions_df.groupby('news_id').agg({
    'clicked': ['sum', 'count']
})
article_ctr.columns = ['clicks', 'impressions']
article_ctr = article_ctr.reset_index()
article_ctr['ctr'] = article_ctr['clicks'] / article_ctr['impressions']

# Filter articles with sufficient impressions
min_impressions = 5
filtered_article_ctr = article_ctr[article_ctr['impressions'] >= min_impressions].copy()
print(f"Articles with at least {min_impressions} impressions: {len(filtered_article_ctr)}")

# Filter out suspicious CTRs
clean_article_ctr = filtered_article_ctr[
    (filtered_article_ctr['ctr'] > 0) & 
    (filtered_article_ctr['ctr'] <= 0.7)
].copy()
print(f"Articles after filtering extreme CTRs: {len(clean_article_ctr)}")

# Merge CTR data with news data
articles_with_metadata = clean_article_ctr.merge(
    news_df,
    left_on="news_id",
    right_on="newsID",
    how="inner"
)
print(f"Articles with complete metadata: {len(articles_with_metadata)}")

# CTR analysis by category
print("\n### CTR by Category")
category_ctr = articles_with_metadata.groupby("category").agg({
    "ctr": ["mean", "std", "count"]
})
# Flatten the column names to avoid tuple keys
category_ctr.columns = ['ctr_mean', 'ctr_std', 'ctr_count']
category_ctr = category_ctr.sort_values('ctr_mean', ascending=False)
print(category_ctr)

# Visualize CTR by category
plt.figure(figsize=(12, 8))
sns.boxplot(x="category", y="ctr", data=articles_with_metadata)
plt.title("CTR Distribution by Category")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{output_dir}/plots/ctr_by_category.png", dpi=300)
plt.close()

# Statistical test for category differences
categories = []
ctr_values = []
for category, group in articles_with_metadata.groupby('category'):
    if len(group) >= 30:
        categories.append(category)
        ctr_values.append(group['ctr'].values)

if len(categories) >= 2:
    f_stat, p_value = stats.f_oneway(*ctr_values)
    print(f"\nANOVA test for CTR differences between categories:")
    print(f"F={f_stat:.4f}, p={p_value:.6f}")
    print("Significant differences" if p_value < 0.05 else "No significant differences")

# ===============================================================
# PART 5: HEADLINE FEATURE ANALYSIS (KEY FOR CTR OPTIMIZATION)
# ===============================================================
print("\n## PART 5: Headline Feature Analysis for CTR Optimization")

# Calculate correlations between headline features and CTR
feature_columns = [
    'title_length', 'title_word_count', 'title_reading_ease',
    'has_question', 'has_exclamation', 'has_number', 'has_colon',
    'has_quotes', 'has_ellipsis', 'title_uppercase_ratio',
    'starts_with_number', 'starts_with_question',
    'has_superlative', 'has_temporal_words'
]

correlations = articles_with_metadata[feature_columns + ['ctr']].corr()['ctr'].sort_values(ascending=False)
print("\nCorrelation between headline features and CTR:")
print(correlations)

# Create feature importance visualization
plt.figure(figsize=(10, 8))
correlations[:-1].plot(kind='barh')
plt.title('Headline Feature Correlations with CTR')
plt.xlabel('Correlation with CTR')
plt.tight_layout()
plt.savefig(f'{output_dir}/plots/feature_correlations.png', dpi=300)
plt.close()

# Statistical analysis of specific features
print("\n### Statistical Tests for Key Features")

# Test for questions vs non-questions
for feature in ['has_question', 'has_number', 'has_superlative']:
    group0 = articles_with_metadata[articles_with_metadata[feature] == 0]['ctr']
    group1 = articles_with_metadata[articles_with_metadata[feature] == 1]['ctr']
    
    if len(group0) > 0 and len(group1) > 0:
        t_stat, p_val = stats.ttest_ind(group0, group1, equal_var=False)
        
        print(f"\nT-test for {feature}:")
        print(f"Mean CTR without feature: {group0.mean():.4f}")
        print(f"Mean CTR with feature: {group1.mean():.4f}")
        print(f"Percent difference: {((group1.mean() / group0.mean()) - 1) * 100:.1f}%")
        print(f"t-statistic: {t_stat:.4f}, p-value: {p_val:.6f}")
        print("Statistically significant" if p_val < 0.05 else "Not statistically significant")

# Visualize key features
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. CTR by title length
axes[0, 0].scatter(articles_with_metadata['title_length'], articles_with_metadata['ctr'], alpha=0.5)
axes[0, 0].set_title('CTR vs Title Length')
axes[0, 0].set_xlabel('Title Length (characters)')
axes[0, 0].set_ylabel('CTR')

# 2. CTR by word count
axes[0, 1].scatter(articles_with_metadata['title_word_count'], articles_with_metadata['ctr'], alpha=0.5)
axes[0, 1].set_title('CTR vs Word Count')
axes[0, 1].set_xlabel('Word Count')
axes[0, 1].set_ylabel('CTR')

# 3. CTR with/without questions
sns.boxplot(x='has_question', y='ctr', data=articles_with_metadata, ax=axes[1, 0])
axes[1, 0].set_title('CTR: Questions vs Non-Questions')
axes[1, 0].set_xticklabels(['No Question', 'Has Question'])

# 4. CTR with/without numbers
sns.boxplot(x='has_number', y='ctr', data=articles_with_metadata, ax=axes[1, 1])
axes[1, 1].set_title('CTR: With/Without Numbers')
axes[1, 1].set_xticklabels(['No Numbers', 'Has Numbers'])

plt.tight_layout()
plt.savefig(f'{output_dir}/plots/feature_analysis.png', dpi=300)
plt.close()

# ===============================================================
# PART 6: EDITORIAL INSIGHTS AND GUIDELINES
# ===============================================================
print("\n## PART 6: Editorial Insights and Guidelines")

# Analyze patterns in high vs low CTR articles
high_ctr_articles = articles_with_metadata[articles_with_metadata['ctr'] > articles_with_metadata['ctr'].quantile(0.8)]
low_ctr_articles = articles_with_metadata[articles_with_metadata['ctr'] < articles_with_metadata['ctr'].quantile(0.2)]

print(f"\nHigh CTR articles (top 20%): {len(high_ctr_articles)}")
print(f"Low CTR articles (bottom 20%): {len(low_ctr_articles)}")

# Analyze patterns in high vs low performing headlines
def analyze_group_patterns(group, label):
    patterns = {}
    for feature in feature_columns:
        if feature in group.columns:
            patterns[feature] = group[feature].mean()
    return patterns

high_ctr_patterns = analyze_group_patterns(high_ctr_articles, "High CTR")
low_ctr_patterns = analyze_group_patterns(low_ctr_articles, "Low CTR")

print("\nPatterns in high-performing headlines:")
for feature, value in high_ctr_patterns.items():
    print(f"  {feature}: {value:.3f}")

print("\nPatterns in low-performing headlines:")
for feature, value in low_ctr_patterns.items():
    print(f"  {feature}: {value:.3f}")

# ===============================================================
# PART 7: SAVE COMPREHENSIVE RESULTS FOR AGENTIC EDITOR
# ===============================================================
print("\n## PART 7: Save Results for Agentic Editor")

# Save cleaned news data with all features
news_df_final = news_df.merge(
    clean_article_ctr[['news_id', 'ctr', 'clicks', 'impressions']],
    left_on='newsID',
    right_on='news_id',
    how='left'
).fillna({'ctr': 0, 'clicks': 0, 'impressions': 0})

news_df_final.to_csv(f'{output_dir}/processed_data/news_with_features.csv', index=False)

# Save editorial guidelines
editorial_guidelines = {
    'overall_ctr_benchmark': float(overall_ctr),
    'category_performance': category_ctr.to_dict('index'),  # Convert to dict with index as keys
    'feature_correlations': {k: float(v) for k, v in correlations.to_dict().items()},  # Ensure float values
    'high_performing_patterns': {k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in high_ctr_patterns.items()},
    'low_performing_patterns': {k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in low_ctr_patterns.items()},
    'key_insights': {
        'questions_reduce_ctr': bool(correlations['has_question'] < 0),
        'numbers_effect': 'neutral' if abs(correlations['has_number']) < 0.01 else ('positive' if correlations['has_number'] > 0 else 'negative'),
        'reading_ease_effect': 'minimal' if abs(correlations['title_reading_ease']) < 0.01 else 'significant',
        'top_categories': category_ctr.head(3).index.tolist(),
        'bottom_categories': category_ctr.tail(3).index.tolist()
    }
}

with open(f'{output_dir}/processed_data/editorial_guidelines.json', 'w') as f:
    json.dump(editorial_guidelines, f, indent=2)

# Save examples for headline rewriting
rewriting_examples = {
    'avoid_questions': {
        'original': high_ctr_articles[high_ctr_articles['has_question'] == 1]['title'].head(5).tolist(),
        'ctr': [float(x) for x in high_ctr_articles[high_ctr_articles['has_question'] == 1]['ctr'].head(5).tolist()]
    },
    'effective_patterns': {
        'high_ctr_titles': [{
            'title': str(row['title']),
            'ctr': float(row['ctr']),
            'category': str(row['category'])
        } for _, row in high_ctr_articles.nlargest(10, 'ctr')[['title', 'ctr', 'category']].iterrows()],
        'category_leaders': {}
    }
}

# Get best performing title from each category
for category in articles_with_metadata['category'].unique():
    cat_data = articles_with_metadata[articles_with_metadata['category'] == category]
    if not cat_data.empty:
        best = cat_data.nlargest(1, 'ctr').iloc[0]
        rewriting_examples['effective_patterns']['category_leaders'][str(category)] = {
            'title': str(best['title']),
            'ctr': float(best['ctr']),
            'features': {feat: float(best[feat]) if isinstance(best[feat], (int, float, np.number)) else best[feat] 
                        for feat in feature_columns if feat in best}
        }

with open(f'{output_dir}/processed_data/headline_rewriting_examples.json', 'w') as f:
    json.dump(rewriting_examples, f, indent=2)

# Save feature importance for model training
feature_importance = {
    'correlation_with_ctr': {k: float(v) for k, v in correlations.to_dict().items()},
    'recommended_features': [feat for feat in feature_columns if abs(correlations[feat]) > 0.01],
    'features_to_avoid': [feat for feat in feature_columns if correlations[feat] < -0.05]
}

with open(f'{output_dir}/processed_data/feature_importance.json', 'w') as f:
    json.dump(feature_importance, f, indent=2)

print("-" * 80)
print(f"Enhanced EDA Pipeline completed! Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Results saved to '{output_dir}/'")
print("\nKey Findings:")
print(f"- Overall CTR benchmark: {overall_ctr:.4f}")
print(f"- Questions reduce CTR by ~{abs(correlations['has_question']*100):.1f}%")
print(f"- Numbers effect on CTR: {correlations['has_number']*100:.1f}%")
print(f"- Reading ease correlation: {correlations['title_reading_ease']:.3f}")
print(f"- Top performing categories: {', '.join(category_ctr.head(3).index.tolist())}")
print(f"- Files created: {len(os.listdir(f'{output_dir}/plots'))} visualizations, {len(os.listdir(f'{output_dir}/processed_data'))} data files")