# ===============================================================
# Agentic AI News Editor: Comprehensive EDA Pipeline
# ---------------------------------------------------------------
# This pipeline analyzes the Microsoft MIND dataset outputs to prepare data
# for an agentic AI system that selects, ranks, and rewrites news headlines

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import re
from scipy import stats
from datetime import datetime
from textstat import flesch_reading_ease

# Set matplotlib visualization defaults
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Create output directories
output_dir = 'agentic_news_editor'
plots_dir = os.path.join(output_dir, 'plots')
processed_dir = os.path.join(output_dir, 'processed_data')
for d in (output_dir, plots_dir, processed_dir):
    os.makedirs(d, exist_ok=True)

print("# Agentic AI News Editor - EDA Pipeline")
print("Starting EDA pipeline... Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("-" * 80)

# 1. Load preprocessed data (with timestamps) from all splits
import glob
csv_files = glob.glob(os.path.join(processed_dir, '*_headline_ctr.csv'))
if not csv_files:
    raise FileNotFoundError(f"No processed '*_headline_ctr.csv' files found in {processed_dir}")
df_list = [pd.read_csv(f) for f in csv_files]
# Concatenate train/val/test (or any split) into one DataFrame
df = pd.concat(df_list, ignore_index=True)
# Convert timestamp columns to datetime (if present)
if 'first_seen' in df.columns:
    df['first_seen'] = pd.to_datetime(df['first_seen'], errors='coerce')
if 'last_seen' in df.columns:
    df['last_seen']  = pd.to_datetime(df['last_seen'],  errors='coerce')

# 2. Basic distributions2 Title readability vs CTR
if 'title_reading_ease' not in df.columns:
    df['title_reading_ease'] = df['title'].apply(flesch_reading_ease)
plt.scatter(df['title_reading_ease'], df['ctr'], alpha=0.3)
plt.title('Readability vs CTR')
plt.xlabel('Flesch Reading Ease')
plt.ylabel('CTR')
plt.savefig(os.path.join(plots_dir, 'readability_vs_ctr.png'))
plt.clf()

# 3. Temporal analysis
# 3.1 Article count by day of first_seen
daily_counts = df.groupby(df['first_seen'].dt.date).size()
daily_counts.plot()
plt.title('Article Count per Day (first_seen)')
plt.xlabel('Date')
plt.ylabel('Number of Articles')
plt.savefig(os.path.join(plots_dir, 'articles_per_day.png'))
plt.clf()

# 3.2 Average CTR by week
weekly_ctr = df.set_index('first_seen')['ctr'].resample('W').mean()
weekly_ctr.plot()
plt.title('Average Weekly CTR')
plt.xlabel('Week')
plt.ylabel('Mean CTR')
plt.savefig(os.path.join(plots_dir, 'weekly_ctr.png'))
plt.clf()

# 3.3 Article age vs CTR scatter
df['age_days'] = (pd.Timestamp.now() - df['first_seen']).dt.days
plt.scatter(df['age_days'], df['ctr'], alpha=0.3)
plt.title('Article Age vs CTR')
plt.xlabel('Age (days)')
plt.ylabel('CTR')
plt.savefig(os.path.join(plots_dir, 'age_vs_ctr.png'))
plt.clf()

# 4. Headline pattern analysis overall and by period
patterns = {
    'has_quote':   lambda s: bool(re.search(r'"',     s)),
    'has_number':  lambda s: bool(re.search(r'\b\d+\b', s)),
    'is_question': lambda s: s.strip().endswith('?'),
    'has_colon':   lambda s: ':' in s
}
for name, func in patterns.items():
    df[name] = df['title'].apply(func)

# Compute CTR lift for patterns
def pattern_lift(subdf, pattern_col):
    with_vals = subdf[subdf[pattern_col]]['ctr']
    without = subdf[~subdf[pattern_col]]['ctr']
    return (with_vals.mean() - without.mean()) / (without.mean() + 1e-9)

# 4.1 Overall lifts
overall_lifts = {pat: pattern_lift(df, pat) for pat in patterns}
print("Overall pattern lifts (ΔCTR):", overall_lifts)

# 4.2 Temporal lifts (early vs late)
cutoff = df['first_seen'].quantile(0.5)
early = df[df['first_seen'] <= cutoff]
late  = df[df['first_seen'] >  cutoff]
temporal_lifts = {'early': {}, 'late': {}}
for pat in patterns:
    temporal_lifts['early'][pat] = pattern_lift(early, pat)
    temporal_lifts['late'][pat]  = pattern_lift(late, pat)
print("Temporal pattern lifts:", temporal_lifts)

# 5. Statistical tests for question headlines
t_yes = df[df['is_question']]['ctr']
t_no  = df[~df['is_question']]['ctr']
t_stat, p_val = stats.ttest_ind(t_yes, t_no, equal_var=False, nan_policy='omit')
print(f"Question headline t-test: t={t_stat:.2f}, p={p_val:.3f}")

# 6. Save summary results
summary = {
    'overall_lifts': overall_lifts,
    'temporal_lifts': temporal_lifts,
    'question_ttest': {'t_stat': t_stat, 'p_val': p_val}
}
with open(os.path.join(processed_dir, 'eda_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print("EDA pipeline completed. Plots and summary saved.")
