import os
import sys
import logging
import subprocess

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_USE_CUDA_DSA"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["TORCH_USE_RTLD_GLOBAL"] = "YES"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def fix_torch_compatibility():
    """Fix known torch compatibility issues before app starts"""
    try:
        import torch
        import torchvision

        torch.backends.cudnn.enabled = False
        torch.multiprocessing.set_sharing_strategy("file_system")
        if hasattr(torch, "_classes"):

            class DummyPath:
                _path = []

            original_getattr = torch._classes.__getattr__

            def safe_getattr(name):
                if name == "__path__":
                    return DummyPath()
                try:
                    return original_getattr(name)
                except:
                    return None

            torch._classes.__getattr__ = safe_getattr
        logging.info("Torch compatibility fixes applied")
        return True
    except Exception as e:
        logging.warning(f"Could not apply torch fixes: {e}")
        return False


import streamlit as st
import json
import pandas as pd
import numpy as np
import datetime
import random
import time
import re
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from openai import OpenAI
from dotenv import load_dotenv
from headline_model_trainer_optimized import CTRPredictor
from headline_metrics import HeadlineMetrics
from headline_learning import HeadlineLearningLoop

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Failed to load image Python extension")


def get_ctr_safely(ctr_predictor, headline):
    """Safely get CTR prediction with error handling"""
    try:
        print(f"\n{'='*50}")
        print(f"PREDICTING CTR FOR: '{headline}'")
        print(f"{'='*50}")
        result = ctr_predictor.predict_single_headline(headline)

        if isinstance(result, dict):
            ctr_value = result["ctr"]
            print(f"FINAL CTR PREDICTION: {ctr_value:.6f} ({ctr_value*100:.3f}%)")
            print(f"RELATIVE SCORE: {result['relative_score']:.2f}x baseline")
            print(f"{'='*50}\n")
            return ctr_value
        else:
            return float(result) if result > 0 else 0.0

    except Exception as e:
        print(f"ERROR in get_ctr_safely: {e}")
        import traceback

        traceback.print_exc()
        print(f"{'='*50}\n")
        return 0.0


try:
    import faiss
except ImportError:
    st.error("FAISS not installed. Install with 'pip install faiss-cpu'")
try:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from sentence_transformers import SentenceTransformer
except ImportError:
    st.error(
        "SentenceTransformer not installed. Please install with 'pip install sentence-transformers'"
    )

st.set_page_config(
    page_title="Agentic AI News Editor",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)


def add_headline_pair(self, original, rewritten, topic=None):
    """Add a single pair of headlines to the learning system"""
    try:
        comparison = self.metrics_analyzer.compare_headlines(original, rewritten)
        new_entry = pd.DataFrame(
            {
                "original_title": [original],
                "rewritten_title": [rewritten],
                "headline_score_original": [comparison["original_score"]],
                "headline_score_rewritten": [comparison["rewritten_score"]],
                "headline_ctr_original": [comparison["original_ctr"]],
                "headline_ctr_rewritten": [comparison["rewritten_ctr"]],
                "headline_improvement": [comparison["score_percent_change"]],
                "headline_key_factors": [", ".join(comparison["key_improvements"])],
                "topic": [topic if topic else "General"],
                "timestamp": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            }
        )

        self.data = pd.concat([self.data, new_entry], ignore_index=True)
        self.data.to_csv(self.data_file, index=False)

        return True
    except Exception as e:
        logging.error(f"Error adding headline pair: {e}")
        return False


def add_headlines_from_dataframe(self, df):
    """Add multiple headline pairs from a dataframe"""
    count = 0

    for _, row in df.iterrows():
        if "original_title" in row and "rewritten_title" in row:
            topic = row.get("topic", "General")
            if (
                "headline_score_original" in row
                and "headline_score_rewritten" in row
                and "headline_ctr_original" in row
                and "headline_ctr_rewritten" in row
                and "headline_improvement" in row
                and "headline_key_factors" in row
            ):

                new_entry = pd.DataFrame(
                    {
                        "original_title": [row["original_title"]],
                        "rewritten_title": [row["rewritten_title"]],
                        "headline_score_original": [row["headline_score_original"]],
                        "headline_score_rewritten": [row["headline_score_rewritten"]],
                        "headline_ctr_original": [row["headline_ctr_original"]],
                        "headline_ctr_rewritten": [row["headline_ctr_rewritten"]],
                        "headline_improvement": [row["headline_improvement"]],
                        "headline_key_factors": [row["headline_key_factors"]],
                        "topic": [topic],
                        "timestamp": [
                            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        ],
                    }
                )

                self.data = pd.concat([self.data, new_entry], ignore_index=True)
                count += 1
            else:
                if self.add_headline_pair(
                    row["original_title"], row["rewritten_title"], topic
                ):
                    count += 1
    if count > 0:
        self.data.to_csv(self.data_file, index=False)

    return count


def generate_improved_headline_report(self):
    """Generate a comprehensive, statistically sound headline improvement report"""
    if len(self.data) < 20:
        return {
            "status": "insufficient_data",
            "message": "Need at least 20 headline pairs",
        }

    # Determine which column names to use
    if "probability_improvement" in self.data.columns:
        improvement_col = "probability_improvement"
        orig_ctr_col = "original_click_probability"
        new_ctr_col = "rewritten_click_probability"
        factors_col = "key_improvements"
    else:
        improvement_col = "headline_improvement"
        orig_ctr_col = "headline_ctr_original"
        new_ctr_col = "headline_ctr_rewritten"
        factors_col = "headline_key_factors"

    # Clean and validate data
    df = self.data.copy()
    df = df.dropna(subset=[improvement_col, orig_ctr_col, new_ctr_col])

    # Handle outliers (improvements > 500% are likely errors)
    outliers_mask = np.abs(df[improvement_col]) > 500
    outliers_count = outliers_mask.sum()
    if outliers_count > 0:
        print(
            f"Detected {outliers_count} extreme outliers (>500% change), capping at ±500%"
        )
        df.loc[df[improvement_col] > 500, improvement_col] = 500
        df.loc[df[improvement_col] < -500, improvement_col] = -500

    # Calculate overall statistics
    total_headlines = len(df)
    positive_improvements = (df[improvement_col] > 0).sum()
    significant_improvements = (df[improvement_col] > 20).sum()  # >20% improvement
    negative_improvements = (df[improvement_col] < -10).sum()  # >10% decrease

    # Calculate median instead of mean (more robust to outliers)
    median_improvement = df[improvement_col].median()
    mean_improvement = df[improvement_col].mean()
    std_improvement = df[improvement_col].std()

    # Calculate actual CTR changes
    ctr_original_mean = df[orig_ctr_col].mean()
    ctr_rewritten_mean = df[new_ctr_col].mean()
    actual_ctr_change = (
        (ctr_rewritten_mean - ctr_original_mean) / ctr_original_mean
    ) * 100

    # Analyze improvement factors
    all_factors = []
    for factors in df[factors_col]:
        if isinstance(factors, str) and factors.strip():
            # Clean up the factors
            factors_list = [f.strip() for f in factors.split(",") if f.strip()]
            all_factors.extend(factors_list)

    factor_counts = pd.Series(all_factors).value_counts().head(10)

    # Topic-level analysis with statistical testing
    topic_analysis = {}
    for topic in df["topic"].unique():
        topic_data = df[df["topic"] == topic]
        if len(topic_data) >= 5:  # Only analyze topics with sufficient data
            topic_analysis[topic] = {
                "count": len(topic_data),
                "median_improvement": topic_data[improvement_col].median(),
                "mean_improvement": topic_data[improvement_col].mean(),
                "std_improvement": topic_data[improvement_col].std(),
                "success_rate": (topic_data[improvement_col] > 0).mean(),
                "significant_success_rate": (topic_data[improvement_col] > 20).mean(),
                "failure_rate": (topic_data[improvement_col] < -10).mean(),
                "mean_original_ctr": topic_data[orig_ctr_col].mean(),
                "mean_rewritten_ctr": topic_data[new_ctr_col].mean(),
            }

    # Generate the report
    report = f"""# Headline Improvement Analysis Report
    Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}

    ## Executive Summary
    We analyzed **{total_headlines}** headline rewriting attempts using our AI system. Here's what we found:

    ### Key Findings
    - **{positive_improvements} headlines ({positive_improvements/total_headlines:.1%})** showed improvement
    - **{significant_improvements} headlines ({significant_improvements/total_headlines:.1%})** showed significant improvement (>20%)
    - **{negative_improvements} headlines ({negative_improvements/total_headlines:.1%})** performed worse (>10% decrease)

    ### Overall Performance
    - **Median Improvement**: {median_improvement:.1f}%
    - *This means half of all rewrites performed better than {median_improvement:.1f}% improvement*
    - **Average Improvement**: {mean_improvement:.1f}% (±{std_improvement:.1f}%)
    - *The typical improvement, accounting for both successes and failures*
    - **Actual CTR Impact**: {actual_ctr_change:.2f}%
    - *The real-world click-through rate change from original to rewritten*

    ## What These Numbers Mean

    ### Understanding Improvement Percentages
    - **Positive values** (e.g., +25%): The rewritten headline is predicted to get 25% more clicks
    - **Negative values** (e.g., -15%): The rewritten headline is predicted to get 15% fewer clicks
    - **Zero (0%)**: No significant change in predicted performance

    ### Statistical Context
    - **Standard Deviation**: {std_improvement:.1f}% shows high variability (expected in creative tasks)
    - **Sample Size**: {total_headlines} provides {'good' if total_headlines >= 100 else 'limited'} statistical confidence

    ## Most Effective Improvement Strategies
    """

    # Add factor analysis
    if len(factor_counts) > 0:
        report += "\n### Techniques That Work\n"
        for i, (factor, count) in enumerate(factor_counts.items(), 1):
            percentage = (count / total_headlines) * 100
            report += (
                f"{i}. **{factor}**: Used in {count} headlines ({percentage:.1f}%)\n"
            )

    # Add topic-specific analysis
    report += "\n## Performance by Topic\n"
    report += "\n*Note: Only topics with 5+ headlines are included for statistical validity*\n\n"

    # Sort topics by median improvement for better insights
    sorted_topics = sorted(
        topic_analysis.items(),
        key=lambda x: x[1]["median_improvement"],
        reverse=True,
    )

    for topic, stats in sorted_topics:
        report += f"### {topic}\n"
        report += f"- **Sample Size**: {stats['count']} headlines\n"
        report += f"- **Median Improvement**: {stats['median_improvement']:.1f}%\n"
        report += (
            f"- **Success Rate**: {stats['success_rate']:.1%} of headlines improved\n"
        )
        report += f"- **Significant Success**: {stats['significant_success_rate']:.1%} showed >20% improvement\n"
        if stats["failure_rate"] > 0:
            report += f"- **Failure Rate**: {stats['failure_rate']:.1%} performed significantly worse\n"
        report += f"- **CTR Change**: {stats['mean_original_ctr']:.3f} → {stats['mean_rewritten_ctr']:.3f}\n\n"

    # Add data quality section
    report += f"""
    ## Data Quality Assessment
    - **Total Records**: {total_headlines} headline pairs
    - **Outliers Detected**: {outliers_count} extreme values (>500% change)
    - **Data Completeness**: {(len(df)/len(self.data)):.1%} records with complete data
    - **Date Range**: {df['timestamp'].min()} to {df['timestamp'].max()}

    ## Statistical Interpretation Guide

    ### How to Read These Results
    1. **Median vs Mean**: We report both because:
    - **Median** shows typical performance (less affected by extreme cases)
    - **Mean** shows average performance (includes all successes and failures)

    2. **Success Rates**:
    - **Any Improvement**: Headlines that performed even slightly better
    - **Significant Improvement**: Headlines with >20% better predicted performance
    - **Failure Rate**: Headlines that performed >10% worse

    3. **CTR Numbers**:
    - These represent predicted click-through rates (probability of clicks)
    - Small absolute numbers (0.001-0.1) are normal for news headlines
    - Focus on relative improvements rather than absolute values

    ## Actionable Recommendations

    ### Based on This Analysis:
    1. **For {sorted_topics[0][0]}**: Most successful topic - continue current approach
    2. **For {sorted_topics[-1][0]}**: Needs improvement - consider topic-specific training
    3. **Overall Strategy**: Focus on techniques showing >30 occurrences in our factor analysis
    4. **Quality Control**: Review headlines with <-20% improvement to identify failure patterns

    ## Next Steps
    1. **Increase Sample Size**: Aim for 50+ headlines per topic for better statistical confidence
    2. **A/B Testing**: Test our top predictions with real users
    3. **Continuous Learning**: Use successful patterns to improve the rewriting algorithm
    4. **Human Validation**: Have editors review headlines with extreme improvements/failures

    ---
    *This report analyzes AI-generated headline improvements. Results are predictions that should be validated with real user testing.*
    """

    # Save the report
    with open("comprehensive_headline_analysis.md", "w") as f:
        f.write(report)

    return {
        "status": "success",
        "total_headlines": total_headlines,
        "median_improvement": median_improvement,
        "success_rate": positive_improvements / total_headlines,
        "report_path": "comprehensive_headline_analysis.md",
    }


def apply_custom_css():

    timestamp = str(int(time.time()))
    st.markdown(
        """
<style>
/* Fix top cutoff */
.main > div {
    padding-top: 1rem;
}

/* IMPROVED MULTISELECT STYLING - More specific selectors */
.stMultiSelect > div > div {
    background-color: #f8f9fa !important;
    border: 2px solid #dee2e6 !important;
    border-radius: 8px !important;
    min-height: 48px !important;
}

/* Target both data-baseweb and data-testid attributes */
.stMultiSelect div[data-baseweb="tag"],
.stMultiSelect div[data-testid="stMultiSelectTag"] {
    background-color: #e7f3ff !important;
    color: #1f2937 !important;
    border: 2px solid #3b82f6 !important;
    border-radius: 20px !important;
    padding: 8px 12px !important;
    margin: 4px 6px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    display: inline-flex !important;
    align-items: center !important;
}

/* Make the X button MUCH more visible - Multiple selectors */
.stMultiSelect div[data-baseweb="tag"] svg,
.stMultiSelect div[data-testid="stMultiSelectTag"] svg,
.stMultiSelect div[data-baseweb="tag"] button,
.stMultiSelect div[data-testid="stMultiSelectTag"] button {
    width: 20px !important;
    height: 20px !important;
    fill: #ef4444 !important;
    margin-left: 8px !important;
    background-color: white !important;
    border-radius: 50% !important;
    padding: 2px !important;
    min-width: unset !important;
    border: none !important;
}

/* Hover effects */
.stMultiSelect div[data-baseweb="tag"]:hover,
.stMultiSelect div[data-testid="stMultiSelectTag"]:hover {
    background-color: #fef2f2 !important;
    border-color: #ef4444 !important;
    transform: scale(1.05) !important;
    transition: all 0.2s ease !important;
}

.stMultiSelect div[data-baseweb="tag"]:hover svg,
.stMultiSelect div[data-testid="stMultiSelectTag"]:hover svg,
.stMultiSelect div[data-baseweb="tag"]:hover button,
.stMultiSelect div[data-testid="stMultiSelectTag"]:hover button {
    fill: #dc2626 !important;
    background-color: #fef2f2 !important;
}

/* Force clear button visibility */
.stMultiSelect button[aria-label*="remove"] {
    position: relative !important;
    visibility: visible !important;
    opacity: 1 !important;
    width: 20px !important;
    height: 20px !important;
    background-color: white !important;
    border-radius: 50% !important;
    border: 2px solid #ef4444 !important;
}

/* Style the dropdown */
.stMultiSelect div[data-baseweb="select"] {
    background-color: white !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
}

/* Make unselected options clearer */
.stMultiSelect div[data-baseweb="option"] {
    padding: 12px 16px !important;
    font-size: 1rem !important;
    border-bottom: 1px solid #f3f4f6 !important;
}

.stMultiSelect div[data-baseweb="option"]:hover {
    background-color: #f3f4f6 !important;
}

/* Add some spacing */
.stMultiSelect {
    margin-bottom: 1rem !important;
}

/* Additional styles for rest of the page */
.nav-section {
    display: flex;
    flex-wrap: wrap;
    gap: 2rem;
    margin: 2rem 0;
    justify-content: center;
}

.nav-item {
    cursor: pointer;
    font-size: 1.5rem;
    font-weight: bold;
    color: #333;
    text-decoration: none;
    padding: 1rem 2rem;
    border-radius: 8px;
    transition: all 0.3s ease;
    background-color: #f0f0f0;
    text-align: center;
}

.nav-item:hover {
    background-color: #e0e0e0;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.nav-item.active {
    background-color: #2c5aa0;
    color: white;
}

.section-container {
    margin-bottom: 3rem;
}

.article-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
    gap: 2rem;
    margin-top: 2rem;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 0.5rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

.stApp {
    max-width: 100%;
}

div.row-widget.stRadio > div {
    flex-direction: row;
    align-items: center;
}

h1, h2, h3 {
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
}

div.stCard {
    padding: 0.5rem;
}

.element-container .stMarkdown {
    margin-bottom: 0.5rem;
}

.original-headline {
    font-style: italic;
    color: #777;
    margin-bottom: 5px;
    padding: 4px 8px;
    background-color: #f8f9fa;
    border-left: 3px solid #ccc;
    font-size: 0.9em;
}

.rewritten-headline {
    font-weight: bold;
    color: #000;
    margin-top: 0;
}

.title-comparison {
    margin-bottom: 15px;
    padding: 8px;
    border-radius: 4px;
    background-color: #f5f5f5;
}

.newspaper-title {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 1rem;
}

.article-tag {
    display: inline-block;
    background-color: #f0f0f0;
    padding: 2px 8px;
    margin-bottom: 8px;
    font-size: 0.8em;
    border-radius: 3px;
    font-weight: bold;
    color: #555;
}

.article-byline {
    font-style: italic;
    color: #666;
    margin-bottom: 10px;
    font-size: 0.9em;
}

.article-why-matters {
    background-color: #f9f9f9;
    padding: 10px;
    border-left: 3px solid #007bff;
    margin: 10px 0;
}

.section-title {
    border-bottom: 2px solid #ddd;
    padding-bottom: 5px;
    margin-top: 30px;
    margin-bottom: 20px;
}

.article-box {
    padding: 10px;
    margin-bottom: 20px;
    border-bottom: 1px solid #eee;
}

.footer {
    text-align: center;
    margin-top: 50px;
    padding: 20px;
    background-color: #f9f9f9;
    border-top: 1px solid #ddd;
}

.headline-metrics {
    margin-top: 5px;
    font-size: 0.85em;
    background-color: #f8f9fa;
    padding: 4px 8px;
    border-radius: 3px;
}

.metrics-row {
    display: flex;
    justify-content: space-between;
    margin-bottom: 2px;
}

.metric-label {
    color: #666;
    font-weight: bold;
}

.metric-value {
    color: #333;
}

.metric-change {
    font-weight: bold;
}

.metrics-factors {
    font-style: italic;
    color: #666;
    border-top: 1px dotted #ddd;
    padding-top: 2px;
    margin-top: 2px;
}
</style>
""",
        unsafe_allow_html=True,
    )


# --- Functions ---
def process_mind_dataset(news_file="news.tsv", behaviors_file="behaviors.tsv"):
    """Process the MIND dataset to create necessary data files"""
    logging.info("Processing MIND dataset...")

    try:
        # Define column names for MIND dataset
        news_columns = [
            "news_id",
            "category",
            "subcategory",
            "title",
            "abstract",
            "url",
            "title_entities",
            "abstract_entities",
        ]

        # Load news data
        news_df = pd.read_csv(news_file, sep="\t", names=news_columns)
        logging.info(f"Loaded {len(news_df)} news articles")

        # Clean up data
        news_df = news_df.dropna(subset=["title", "abstract"])

        # Save processed data
        news_df.to_csv("processed_news.csv", index=False)
        logging.info(f"Saved processed data to processed_news.csv")

        # Process behaviors to calculate CTR if available
        if os.path.exists(behaviors_file):
            behaviors_columns = [
                "impression_id",
                "user_id",
                "time",
                "history",
                "impressions",
            ]
            behaviors_df = pd.read_csv(
                behaviors_file, sep="\t", names=behaviors_columns
            )

            # Calculate CTR (simplified)
            news_clicks = {}
            news_impressions = {}

            for _, row in behaviors_df.iterrows():
                if isinstance(row["impressions"], str):
                    impressions = row["impressions"].split()

                    for impression in impressions:
                        parts = impression.split("-")
                        if len(parts) == 2:
                            news_id, click = parts

                            if news_id not in news_impressions:
                                news_impressions[news_id] = 0
                            news_impressions[news_id] += 1

                            if click == "1":
                                if news_id not in news_clicks:
                                    news_clicks[news_id] = 0
                                news_clicks[news_id] += 1

            # Calculate CTR for each article
            ctr_data = []
            for news_id, impressions in news_impressions.items():
                clicks = news_clicks.get(news_id, 0)
                ctr = clicks / impressions if impressions > 0 else 0
                ctr_data.append(
                    {
                        "news_id": news_id,
                        "clicks": clicks,
                        "impressions": impressions,
                        "ctr": ctr,
                    }
                )

            # Create dataframe
            ctr_df = pd.DataFrame(ctr_data)

            # Merge with news data
            news_with_ctr = pd.merge(news_df, ctr_df, on="news_id", how="left")
            news_with_ctr["ctr"] = news_with_ctr["ctr"].fillna(0.05)  # Default CTR

            news_with_ctr.to_csv("headline_ctr_data.csv", index=False)
            logging.info(f"Saved news articles with CTR data")

        return True
    except Exception as e:
        logging.error(f"Error processing MIND dataset: {e}")
        return False


def create_embeddings_and_index(processed_file="processed_news.csv"):
    """Create embeddings and FAISS index for articles"""
    try:
        if not os.path.exists(processed_file):
            logging.error(f"Processed file not found: {processed_file}")
            return False

        # Load processed data
        processed_df = pd.read_csv(processed_file)

        # Initialize SentenceTransformer
        model = load_sentence_transformer()
        if model is None:
            return False

        # Prepare text for embedding
        texts = []
        for _, row in processed_df.iterrows():
            title = row["title"] if isinstance(row["title"], str) else ""
            abstract = row["abstract"] if isinstance(row["abstract"], str) else ""
            combined_text = f"{title} {abstract}"
            texts.append(combined_text)

        # Compute embeddings in batches
        logging.info("Computing embeddings...")
        batch_size = 32
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = model.encode(batch_texts)
            all_embeddings.extend(batch_embeddings)

        # Add embeddings to dataframe
        processed_df["embedding"] = [",".join(map(str, emb)) for emb in all_embeddings]

        # Save dataframe with embeddings
        processed_df.to_csv("articles_with_embeddings.csv", index=False)

        # Create FAISS index
        embeddings_array = np.array(all_embeddings).astype("float32")
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)

        # Save index
        faiss.write_index(index, "articles_faiss.index")
        logging.info(f"Created FAISS index with {index.ntotal} vectors")

        return True
    except Exception as e:
        logging.error(f"Error creating embeddings and index: {e}")
        return False


def check_data_and_prepare():
    """Check if data files exist and prepare them if needed"""
    if not os.path.exists("articles_with_embeddings.csv") or not os.path.exists(
        "articles_faiss.index"
    ):
        st.info("Data files not found. Preparing data...")

        # Check for MIND dataset files
        if not os.path.exists("news.tsv") or not os.path.exists("behaviors.tsv"):
            st.error(
                "MIND dataset files not found. Please place news.tsv and behaviors.tsv in the current directory."
            )
            return False

        # Process dataset
        if process_mind_dataset():
            st.info("MIND dataset processed successfully.")
        else:
            st.error("Error processing MIND dataset.")
            return False

        # Create embeddings and index
        if create_embeddings_and_index():
            st.success("Created embeddings and index successfully.")
        else:
            st.error("Error creating embeddings and index.")
            return False

    return True


def load_sentence_transformer():
    """Load the sentence transformer model separately to avoid conflicts"""
    try:
        return SentenceTransformer("paraphrase-MiniLM-L6-v2")
    except Exception as e:
        logging.error(f"Error loading SentenceTransformer: {e}")
        return None


@st.cache_resource
def load_models_and_data():
    """Load FAISS index, metadata, and embedding model with caching"""
    try:
        logging.info("Loading FAISS index and models")

        # Load the CSV data first
        if not os.path.exists("articles_with_embeddings.csv"):
            st.error("⚠️ articles_with_embeddings.csv file not found!")
            return None, None, None

        articles_df = pd.read_csv("articles_with_embeddings.csv")

        # Load FAISS index
        if not os.path.exists("articles_faiss.index"):
            st.error("⚠️ articles_faiss.index file not found!")
            return None, None, None

        index = faiss.read_index("articles_faiss.index")

        # Load model with the separate function
        model = load_sentence_transformer()
        if model is None:
            return None, None, None

        logging.info(f"Loaded {len(articles_df)} articles and model")
        return index, articles_df, model
    except Exception as e:
        logging.error(f"Error loading models and data: {e}")
        return None, None, None


def get_openai_client():
    """Initialize OpenAI client with API key"""
    try:
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if not openai_api_key:
            st.sidebar.error("❌ OPENAI_API_KEY is missing. Check your .env file!")
            return None

        return OpenAI(api_key=openai_api_key)
    except Exception as e:
        logging.error(f"Error initializing OpenAI client: {e}")
        st.sidebar.error(f"Error initializing OpenAI: {e}")
        return None


def rewrite_headline(client, title, abstract, category=None):
    """Rewrite a single headline using OpenAI with an enhanced prompt"""
    if client is None:
        return title

    prompt = f"""You are an expert digital news editor specializing in crafting high-engagement headlines.

HEADLINE REWRITING TASK:
Transform the following news headline into a version that will achieve significantly higher click-through rates while maintaining factual accuracy.

ORIGINAL HEADLINE: "{title}"

ARTICLE ABSTRACT: "{abstract}"

CATEGORY: "{category if category else 'General News'}"

HEADLINE WRITING PRINCIPLES:
1. Use specific, concrete language rather than vague terms
2. Create a curiosity gap that intrigues readers without being misleading
3. Signal value to readers by suggesting what they'll gain from reading
4. Include numbers when relevant (research shows this increases CTR)
5. Use active voice and strong verbs
6. Keep headlines under 70 characters for optimal display
7. Avoid clickbait tactics that would damage credibility

EXAMPLES OF GREAT HEADLINE TRANSFORMATIONS:
Original: "Scientists Discover New Planet That Could Potentially Support Life"
Better: "Earth 2.0: Scientists Find Planet With 95% Match to Our Atmosphere"

Original: "Company Reports Quarterly Earnings Above Expectations"
Better: "Tech Giant Shatters Profit Records as Stock Jumps 7%"

Original: "Study Shows Increased Exercise Linked to Better Cognitive Function"
Better: "30-Minute Daily Walks Boost Brain Function by 32%, Study Reveals"

Your revised headline should be notably more compelling than the original while preserving the core information. Create only ONE headline, not multiple options.

REWRITTEN HEADLINE:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional digital news editor who specializes in writing high-engagement headlines that drive clicks while maintaining journalistic integrity.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,  # Optimized for creativity
            max_tokens=60,
        )
        rewritten = response.choices[0].message.content.strip()

        # Remove any quotation marks the model might add
        rewritten = rewritten.replace('"', "").replace('"', "").replace('"', "")

        # Check for basic quality control
        if len(rewritten) < 5 or rewritten.lower() in ["", "the new york times"]:
            rewritten = title

        return rewritten
    except Exception as e:
        logging.error(f"Error rewriting headline: {e}")
        return title


def train_headline_ctr_model(
    data_file="headline_ctr_data.csv", output_file="headline_ctr_model.pkl"
):
    """Train the headline CTR prediction model"""
    try:
        if not os.path.exists(data_file):
            logging.error(f"Training data not found: {data_file}")
            return False

        # Load data
        data = pd.read_csv(data_file)
        data = data.dropna(subset=["title", "ctr"])

        # Extract features
        features_list = []
        for headline in data["title"]:
            features = {}

            # Basic features based on EDA findings
            features["length"] = len(headline)
            features["word_count"] = len(headline.split())
            features["has_number"] = int(bool(re.search(r"\d", headline)))
            features["is_question"] = int(
                headline.endswith("?")
                or headline.lower().startswith("what")
                or headline.lower().startswith("how")
                or headline.lower().startswith("why")
            )
            features["has_colon"] = int(":" in headline)

            # Get embedding
            model = load_sentence_transformer()
            if model is not None:
                embedding = model.encode([headline])[0]

                # Add first 10 embedding dimensions as features
                for i in range(10):
                    features[f"emb_{i}"] = embedding[i]
            else:
                # Add zeros if model fails
                for i in range(10):
                    features[f"emb_{i}"] = 0.0

            features_list.append(features)

        # Create dataframe
        features_df = pd.DataFrame(features_list)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, data["ctr"], test_size=0.2, random_state=42
        )

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logging.info(f"Model evaluation - MSE: {mse:.4f}, R²: {r2:.4f}")

        # Save model
        with open(output_file, "wb") as f:
            pickle.dump(model, f)

        logging.info(f"Saved model to {output_file}")
        return True
    except Exception as e:
        logging.error(f"Error training model: {e}")
        return False


def check_model_and_train():
    """Check if model exists and train if needed"""
    if not os.path.exists("headline_ctr_model.pkl"):
        st.info("CTR prediction model not found. Training model...")

        if train_headline_ctr_model():
            st.success("CTR prediction model trained successfully.")
        else:
            st.error("Error training CTR prediction model.")
            return False

    return True


def display_headline_comparison(original_title, rewritten_title):
    """Display original and rewritten titles with clear styling"""
    st.markdown(
        f"""
    <div class="title-comparison">
        <div class="original-headline">Original: {original_title}</div>
        <div class="rewritten-headline">{rewritten_title}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def generate_explanation(client, title, abstract):
    """Generate explanation for why the article is important"""
    if client is None:
        return "This article provides important information for our readers."

    prompt = f"""You are an editorial assistant.
    
    Write one sentence explaining why the following news article is important to readers.
    
    Focus on clarity and importance for a general audience.
    
    ---
    
    Title: {title}
    
    Abstract: {abstract}
    
    Explanation:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional editorial assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,  # Lower for consistency
            max_tokens=60,
        )
        explanation = response.choices[0].message.content.strip()
        return explanation
    except Exception as e:
        logging.error(f"Error generating explanation: {e}")
        return "This article provides important information for our readers."


def curate_articles_for_topic(
    query_text, index, articles_df, model, openai_client, k=5, progress_bar=None
):
    """Find and enhance articles for a given topic - SIMPLIFIED AND UPDATED"""
    try:
        # Get the top k articles
        query_embedding = model.encode([query_text])
        D, I = index.search(np.array(query_embedding), k=k)

        # Extract articles
        topic_articles = articles_df.iloc[I[0]].copy()

        if len(topic_articles) == 0:
            logging.warning(f"No articles found for query: {query_text}")
            return pd.DataFrame()

        # Store original title
        topic_articles["original_title"] = topic_articles["title"].copy()

        # Process each article
        total = len(topic_articles)
        for i, (idx, row) in enumerate(topic_articles.iterrows()):
            if progress_bar is not None:
                progress_value = (i + 1) / total
                progress_bar.progress(
                    progress_value, text=f"Processing article {i + 1}/{total}"
                )

            # Rewrite headline and generate explanation
            topic_articles.at[idx, "rewritten_title"] = rewrite_headline(
                openai_client, row["title"], row["abstract"], category=query_text
            )
            topic_articles.at[idx, "explanation"] = generate_explanation(
                openai_client, row["title"], row["abstract"]
            )

        # Analyze headline effectiveness with the simplified function
        topic_articles = analyze_headline_effectiveness_simplified(
            topic_articles, st.session_state.get("ctr_predictor")
        )

        if progress_bar is not None:
            progress_bar.progress(1.0, text="Processing complete!")
            time.sleep(0.5)

        return topic_articles

    except Exception as e:
        logging.error(f"Error curating articles: {e}")
        return pd.DataFrame()


import torch

if hasattr(torch.nn.Module, "to_empty"):
    # For newer versions of PyTorch
    def patch_sentence_transformer():
        import sentence_transformers

        original_to = sentence_transformers.SentenceTransformer.__init__

        def patched_init(self, *args, **kwargs):
            kwargs.pop("cache_folder", None)  # Remove problematic cache_folder
            original_to(self, *args, **kwargs)

        sentence_transformers.SentenceTransformer.__init__ = patched_init

    try:
        patch_sentence_transformer()
    except:
        pass


# Add this after imports and before any model loading
@st.cache_resource
def load_sentence_transformer_once():
    """Load sentence transformer once and cache it"""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        return model
    except Exception as e:
        print(f"Error loading sentence transformer: {e}")
        # Fallback to a simpler initialization
        try:
            from sentence_transformers import SentenceTransformer
            import torch

            # Force CPU and clear any CUDA cache
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            return model
        except:
            return None


def analyze_headline_effectiveness_simplified(df, ctr_predictor):
    """
    Simplified headline analysis using only the CTRPredictor model
    Returns clear, understandable metrics with consistent units
    """
    if df.empty:
        return df

    # Initialize columns with default values
    df["original_click_probability"] = 0.0
    df["rewritten_click_probability"] = 0.0
    df["probability_improvement"] = 0.0
    df["improvement_category"] = "No Change"
    df["key_improvements"] = "Analysis pending"

    if ctr_predictor is None:
        logging.warning("CTR predictor not available - using placeholder values")
        return df

    # Process each headline pair
    for i, row in df.iterrows():
        if pd.isna(row["title"]) or pd.isna(row["rewritten_title"]):
            continue

        try:
            # Get click probabilities (values between 0-1)
            original_prob = get_ctr_safely(ctr_predictor, row["title"])
            rewritten_prob = get_ctr_safely(ctr_predictor, row["rewritten_title"])

            # Store probabilities
            df.at[i, "original_click_probability"] = original_prob
            df.at[i, "rewritten_click_probability"] = rewritten_prob

            # Calculate improvement
            if original_prob > 0:
                improvement = ((rewritten_prob - original_prob) / original_prob) * 100
            else:
                improvement = 100 if rewritten_prob > 0 else 0

            df.at[i, "probability_improvement"] = improvement

            # Categorize improvement
            if improvement > 20:
                df.at[i, "improvement_category"] = "Significant Improvement"
            elif improvement > 5:
                df.at[i, "improvement_category"] = "Moderate Improvement"
            elif improvement > -5:
                df.at[i, "improvement_category"] = "No Significant Change"
            else:
                df.at[i, "improvement_category"] = "Decreased Performance"

            # Generate key improvements based on headline changes
            df.at[i, "key_improvements"] = analyze_headline_changes(
                row["title"], row["rewritten_title"], improvement
            )

        except Exception as e:
            logging.error(f"Error analyzing headline at row {i}: {e}")
            # Keep default values on error

    return df


def analyze_headline_changes(original, rewritten, improvement):
    """Analyze what changed between original and rewritten headlines"""
    changes = []

    # Length changes
    orig_len = len(original)
    new_len = len(rewritten)
    if abs(new_len - orig_len) > 10:
        if new_len > orig_len:
            changes.append("Made headline longer")
        else:
            changes.append("Made headline shorter")

    # Number additions
    if re.search(r"\d+", rewritten) and not re.search(r"\d+", original):
        changes.append("Added specific numbers")

    # Question format
    if "?" in rewritten and "?" not in original:
        changes.append("Added question format")

    # How-to format
    if "how to" in rewritten.lower() and "how to" not in original.lower():
        changes.append("Added how-to format")

    # Power words
    power_words = ["best", "top", "ultimate", "secret", "proven", "exclusive"]
    if any(word in rewritten.lower() for word in power_words) and not any(
        word in original.lower() for word in power_words
    ):
        changes.append("Added power words")

    # Default message
    if not changes:
        if improvement > 0:
            changes.append("Overall writing improvement")
        elif improvement < 0:
            changes.append("Less engaging rewrite")
        else:
            changes.append("No significant changes detected")

    return "; ".join(changes[:3])  # Limit to top 3 changes


def analyze_headline_effectiveness(df, openai_client=None):
    """Wrapper function for backward compatibility"""
    return analyze_headline_effectiveness_simplified(
        df, st.session_state.get("ctr_predictor")
    )


def display_headline_with_clear_metrics(original_title, rewritten_title, metrics=None):
    """Display headlines with clear, understandable metrics"""
    st.markdown(
        f"""
    <div class="title-comparison">
        <div class="original-headline">Original: {original_title}</div>
        <div class="rewritten-headline">{rewritten_title}</div>
    """,
        unsafe_allow_html=True,
    )

    if metrics is not None and isinstance(metrics, dict):
        # Extract metrics with new simplified names
        original_prob = metrics.get("original_click_probability", 0)
        rewritten_prob = metrics.get("rewritten_click_probability", 0)
        improvement = metrics.get("probability_improvement", 0)
        category = metrics.get("improvement_category", "Unknown")
        key_changes = metrics.get("key_improvements", "")

        # Only show metrics if they exist
        if original_prob > 0 or rewritten_prob > 0:
            # Determine color based on improvement
            if improvement > 5:
                color = "#28a745"  # Green for good improvement
            elif improvement > -5:
                color = "#ffc107"  # Yellow for no change
            else:
                color = "#dc3545"  # Red for decrease

            st.markdown(
                f"""
            <div class="headline-metrics">
                <div class="metrics-row">
                    <span class="metric-label">Click Probability:</span>
                    <span class="metric-value">{original_prob*100:.2f}% → {rewritten_prob*100:.2f}%</span>
                </div>
                <div class="metrics-row">
                    <span class="metric-label">Improvement:</span>
                    <span class="metric-change" style="color: {color};">
                        {'+' if improvement > 0 else ''}{improvement:.1f}% ({category})
                    </span>
                </div>
                <div class="metrics-factors">Key Changes: {key_changes}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="headline-metrics">No metrics available</div>',
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


def add_system_status_indicator():
    """Add a clear indicator of which model is being used"""
    ctr_predictor = st.session_state.get("ctr_predictor")

    if ctr_predictor:
        st.sidebar.success(f"✅ CTR Model Active: {ctr_predictor.best_model}")
        st.sidebar.info(f"Model trained: {ctr_predictor.best_model}")
    else:
        st.sidebar.error("❌ CTR Model Not Available")
        st.sidebar.warning("Switch to trained model for accurate predictions")


def calculate_evaluation_metrics(curated_articles):
    """Simplified metrics for research questions"""
    # RQ1: Basic retrieval check
    valid_articles = (curated_articles["abstract"].str.len() > 100).mean()

    # RQ2: Headline improvement (the main focus)
    # Update to use new column names with fallback
    if "probability_improvement" in curated_articles.columns:
        improvement_col = "probability_improvement"
        ctr_orig_col = "original_click_probability"
        ctr_new_col = "rewritten_click_probability"
    else:
        # Fallback to old column names
        improvement_col = "headline_improvement"
        ctr_orig_col = "headline_ctr_original"
        ctr_new_col = "headline_ctr_rewritten"

    headline_metrics = {
        "avg_improvement": curated_articles[improvement_col].mean(),
        "improved_count": (curated_articles[improvement_col] > 0).sum(),
        "total_headlines": len(curated_articles),
        "improvement_rate": (curated_articles[improvement_col] > 0).mean(),
        "avg_ctr_change": (
            curated_articles[ctr_new_col] - curated_articles[ctr_orig_col]
        ).mean(),
    }

    return {
        "retrieval_success": valid_articles,
        "headline_metrics": headline_metrics,
        "num_articles": len(curated_articles),
        "topics_covered": curated_articles["topic"].nunique(),
    }


def generate_research_report(metrics):
    """Generate a simplified research report"""
    report = f"""
# Research Evaluation Report
Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}

## RQ1: Article Retrieval Effectiveness

### Metrics:
- Retrieval Success Rate: {metrics['retrieval_success']:.1%}
- Total Articles: {metrics['num_articles']}
- Topics Covered: {metrics['topics_covered']}

- Average Improvement: {metrics['headline_metrics']['avg_improvement']:.1f}%
- Headlines Improved: {metrics['headline_metrics']['improved_count']} out of {metrics['headline_metrics']['total_headlines']}
- Improvement Rate: {metrics['headline_metrics']['improvement_rate']:.1%}
- Average CTR Change: {metrics['headline_metrics']['avg_ctr_change']:.1f}%

## Summary:
The AI system successfully retrieved {metrics['num_articles']} articles across {metrics['topics_covered']} topics.
Headline rewriting improved {metrics['headline_metrics']['improvement_rate']:.1%} of headlines with an average CTR improvement of {metrics['headline_metrics']['avg_improvement']:.1f}%.
"""

    with open("research_evaluation_report.md", "w") as f:
        f.write(report)

    return report


def update_headline_learning(df):
    """Update the headline learning system with newly processed articles"""
    global headline_learner

    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return 0

    count = headline_learner.add_headlines_from_dataframe(df)
    logging.info(f"Added {count} headline pairs to learning system")
    return count


def get_stock_image_path(topic, article_id=None):
    """Return the path to a stock image for the given topic"""
    topic_to_prefix = {
        "Top Technology News": "tech",
        "Business and Economy": "business",
        "Global Politics": "politics",
        "Climate and Environment": "climate",
        "Health and Wellness": "health",
    }

    prefix = topic_to_prefix.get(topic, "tech")

    if article_id is not None:
        img_num = 1 + (hash(str(article_id)) % 2)
    else:
        img_num = random.randint(1, 2)

    return f"{prefix}{img_num}.jpg"


def display_article_image(topic, article_id=None, is_main=False):
    """Display a stock image for an article with proper error handling"""
    try:
        image_path = get_stock_image_path(topic, article_id)
        width = 700 if is_main else 400

        if os.path.exists(image_path):
            st.image(image_path, width=width)
            return True
        else:
            raise FileNotFoundError(f"Image file not found: {image_path}")
    except Exception as e:
        logging.error(f"Error displaying stock image for {topic}: {e}")

        fallback_color = {
            "Top Technology News": "#007BFF",
            "Business and Economy": "#6F42C1",
            "Global Politics": "#DC3545",
            "Climate and Environment": "#28A745",
            "Health and Wellness": "#FD7E14",
        }.get(topic, "#6C757D")

        height = 350 if is_main else 200

        st.markdown(
            f"""
        <div style="
            height: {height}px; 
            background-color: {fallback_color}25; 
            border: 1px solid {fallback_color}; 
            border-radius: 5px; 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            margin-bottom: 15px;
            color: {fallback_color};
            font-weight: bold;
            text-align: center;
        ">
            {topic}
        </div>
        """,
            unsafe_allow_html=True,
        )
        return False


def evaluate_model_performance():
    """Simple evaluation using your existing headline pairs"""
    ctr_predictor = st.session_state.get("ctr_predictor")

    # Use your existing collected data
    if os.path.exists("headline_learning_data.csv"):
        df = pd.read_csv("headline_learning_data.csv")
    else:
        return {"direction_accuracy": 0, "avg_improvement": 0, "sample_size": 0}

    correct_predictions = 0
    improvements = []

    for _, row in df.iterrows():
        # Get current predictions
        orig_pred = ctr_predictor.predict_single_headline(row["original_title"])
        rewrite_pred = ctr_predictor.predict_single_headline(row["rewritten_title"])

        # Handle both dict and float returns
        if isinstance(orig_pred, dict):
            orig_ctr = orig_pred["ctr"]
            rewrite_ctr = rewrite_pred["ctr"]
        else:
            orig_ctr = orig_pred
            rewrite_ctr = rewrite_pred

        # Calculate predicted improvement
        if orig_ctr > 0:
            pred_improvement = (rewrite_ctr - orig_ctr) / orig_ctr
        else:
            pred_improvement = 0

        # Compare with actual (from your data) - Handle both column names
        if "probability_improvement" in row:
            actual_improvement = (
                row["probability_improvement"] / 100
            )  # Convert % to decimal
        elif "headline_improvement" in row:
            actual_improvement = (
                row["headline_improvement"] / 100
            )  # Convert % to decimal
        else:
            actual_improvement = 0  # fallback

        # Check if direction is correct
        if (pred_improvement > 0 and actual_improvement > 0) or (
            pred_improvement < 0 and actual_improvement < 0
        ):
            correct_predictions += 1

        improvements.append(abs(pred_improvement - actual_improvement))

    if len(df) > 0:
        return {
            "direction_accuracy": correct_predictions / len(df),
            "avg_improvement": np.mean(improvements) * 100,
            "sample_size": len(df),
        }
    else:
        return {"direction_accuracy": 0, "avg_improvement": 0, "sample_size": 0}


def show_model_performance():
    """Show model performance in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Performance")

    # Calculate and display current evaluation
    if st.sidebar.button("Check Model Accuracy"):
        eval_results = evaluate_model_performance()

        # Display key metrics
        st.sidebar.metric(
            "Direction Accuracy", f"{eval_results['direction_accuracy']:.0%}"
        )
        st.sidebar.metric(
            "Average Improvement", f"{eval_results['avg_improvement']:.0%}"
        )
        st.sidebar.info(f"Based on {eval_results['sample_size']} headlines")


def show_model_dashboard():
    """Show simple model performance dashboard"""
    st.subheader("Model Performance Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Real-time accuracy
        if os.path.exists("headline_learning_data.csv"):
            df = pd.read_csv("headline_learning_data.csv")

            # Handle both new and old column names
            improvement_col = (
                "probability_improvement"
                if "probability_improvement" in df.columns
                else "headline_improvement"
            )

            if improvement_col in df.columns:
                improved_count = (df[improvement_col] > 0).sum()
                accuracy = improved_count / len(df) if len(df) > 0 else 0
                st.metric(
                    "Headlines Improved",
                    f"{improved_count}/{len(df)}",
                    f"{accuracy:.0%}",
                )
            else:
                st.metric("Headlines Improved", "0/0", "No data")
        else:
            st.metric("Headlines Improved", "0/0", "No data")

    with col2:
        # Average improvement (replacing confidence)
        if os.path.exists("headline_learning_data.csv"):
            df = pd.read_csv("headline_learning_data.csv")

            # Handle both new and old column names
            improvement_col = (
                "probability_improvement"
                if "probability_improvement" in df.columns
                else "headline_improvement"
            )

            if improvement_col in df.columns and len(df) > 0:
                avg_improvement = df[improvement_col].mean()
                st.metric("Avg Improvement", f"{avg_improvement:.1f}%")
            else:
                st.metric("Avg Improvement", "0%")
        else:
            st.metric("Avg Improvement", "0%")

    with col3:
        # Model status
        ctr_predictor = st.session_state.get("ctr_predictor")
        if ctr_predictor:
            model_name = getattr(ctr_predictor, "best_model", "Unknown")
            st.metric("Model Status", "Active", f"{model_name}")
        else:
            st.metric("Model Status", "Inactive", "Not loaded")


def load_curated_articles():
    """Load previously curated articles if available"""
    try:
        if os.path.exists("curated_full_daily_output.csv"):
            return pd.read_csv("curated_full_daily_output.csv")
        return None
    except Exception as e:
        logging.error(f"Error loading curated articles: {e}")
        return None


# Update the initialize_session_state function:
def initialize_session_state():
    """Initialize all session state variables"""
    if "curation_started" not in st.session_state:
        st.session_state.curation_started = False
    if "curation_complete" not in st.session_state:
        st.session_state.curation_complete = False
    if "loaded_articles" not in st.session_state:
        st.session_state.loaded_articles = None
    if "show_headline_comparison" not in st.session_state:
        st.session_state.show_headline_comparison = True
    if "evaluation_history" not in st.session_state:
        st.session_state.evaluation_history = []

    # Initialize CTR predictor
    if "ctr_predictor" not in st.session_state:
        try:
            st.session_state.ctr_predictor = CTRPredictor(
                processed_data_dir="agentic_news_editor/processed_data",
                output_dir="model_output",
            )
            st.session_state.ctr_predictor.load_model()
            logging.info("CTR predictor loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load CTR predictor: {e}")
            st.session_state.ctr_predictor = None
            st.sidebar.warning("⚠️ CTR model not available. Using fallback metrics.")


def setup_sidebar():
    """Setup sidebar with controls and return configuration"""
    with st.sidebar:
        st.header("📰 NEWSPAPER CONTROLS")
        st.write("Configure your daily newspaper")
        st.markdown("---")

        # Editorial queries
        editorial_queries = {
            "Top Technology News": "latest breakthroughs in technology and innovation",
            "Business and Economy": "coverage of finance, jobs, inflation, innovation economics",
            "Global Politics": "latest news about world politics and diplomacy",
            "Climate and Environment": "climate change news and environment protection",
            "Health and Wellness": "advances in healthcare and medical discoveries",
        }

        # Initialize selected topics in session state if not exists
        if "selected_topics" not in st.session_state:
            st.session_state.selected_topics = list(editorial_queries.keys())[:3]

        # Custom topic selector with checkboxes
        st.subheader("Select Topics to Curate")
        st.write("Choose the topics for your personalized newspaper:")

        # Create checkboxes for each topic
        selected_topics = []
        for topic in editorial_queries.keys():
            is_selected = st.checkbox(
                topic,
                value=topic in st.session_state.selected_topics,
                key=f"topic_{topic.replace(' ', '_')}",
            )
            if is_selected:
                selected_topics.append(topic)

        # Update session state
        st.session_state.selected_topics = selected_topics

        # Visual feedback
        if len(selected_topics) == 0:
            st.error("⚠️ Please select at least one topic")
        else:
            st.success(
                f"✅ Selected {len(selected_topics)} topic{'s' if len(selected_topics) > 1 else ''}"
            )

            # Show selected topics as chips
            st.write("**Your selection:**")
            for topic in selected_topics:
                st.markdown(f"🔹 {topic}")

        articles_per_topic = st.slider("Articles per topic", 1, 3, 1, 1)

        # Display settings
        st.subheader("Display Settings")
        show_headline_comparison = st.toggle(
            "Show headline comparison",
            value=True,
            help="Display both original and AI-rewritten headlines",
        )

        # Action buttons
        curate_button = st.button("CURATE FRESH ARTICLES")
        load_button = st.button("LOAD SAVED ARTICLES")

        # Store in session state
        st.session_state.show_headline_comparison = show_headline_comparison

        # Add other sidebar sections
        add_system_status_indicator()
        add_headline_testing_section()
        add_learning_system_section()
        show_model_performance()

        return {
            "editorial_queries": editorial_queries,
            "selected_topics": selected_topics,
            "articles_per_topic": articles_per_topic,
            "curate_button": curate_button,
            "load_button": load_button,
        }


def calculate_prediction_confidence(headline):
    """Calculate confidence based on headline features"""
    features = {
        "has_common_patterns": any(
            word in headline.lower() for word in ["how", "why", "what", "best"]
        ),
        "has_numbers": bool(re.search(r"\d", headline)),
        "length_optimal": 30 <= len(headline) <= 70,
        "has_action_words": any(
            word in headline.lower() for word in ["discover", "learn", "find"]
        ),
    }

    # Base confidence on feature similarity to training data
    confidence_score = 0.5  # Base 50%

    # Adjust based on features
    if features["has_common_patterns"]:
        confidence_score += 0.15
    if features["has_numbers"]:
        confidence_score += 0.1
    if features["length_optimal"]:
        confidence_score += 0.15
    if features["has_action_words"]:
        confidence_score += 0.1

    # Cap at 90%
    confidence_score = min(confidence_score, 0.9)

    return confidence_score


def add_headline_testing_section():
    """Add a section for testing individual headlines with confidence"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Test Any Headline")

    user_headline = st.sidebar.text_area(
        "Enter your headline:", placeholder="Type your headline here...", height=60
    )

    if user_headline and st.sidebar.button("Analyze & Improve"):
        ctr_predictor = st.session_state.get("ctr_predictor")
        openai_client = get_openai_client()

        if not ctr_predictor or not openai_client:
            st.error("Model or OpenAI client not available")
            return

        with st.spinner("Analyzing headline..."):
            # Get predictions
            original_result = ctr_predictor.predict_single_headline(user_headline)
            rewritten = rewrite_headline(openai_client, user_headline, "General news")
            rewritten_result = ctr_predictor.predict_single_headline(rewritten)

            # Handle both dict and float returns
            if isinstance(original_result, dict):
                orig_ctr = original_result["ctr"]
                orig_relative = original_result["relative_score"]
                rewrite_ctr = rewritten_result["ctr"]
                rewrite_relative = rewritten_result["relative_score"]
            else:
                orig_ctr = original_result
                rewrite_ctr = rewritten_result
                orig_relative = orig_ctr / 0.008  # Convert to relative
                rewrite_relative = rewrite_ctr / 0.008

            # Calculate improvement
            if orig_ctr > 0:
                improvement = ((rewrite_ctr - orig_ctr) / orig_ctr) * 100
            else:
                improvement = 100 if rewrite_ctr > 0 else 0

            # Save to learning system
            if st.session_state.get("headline_learner"):
                st.session_state.headline_learner.add_headline_pair(
                    user_headline, rewritten
                )


def add_learning_system_section():
    """Add the headline learning system section to sidebar"""
    st.markdown("---")
    st.subheader("Headline Learning System")

    if st.button("Generate Headline Report"):
        try:
            if (
                st.session_state.get("headline_learner")
                and st.session_state.headline_learner.prompt_improvement_report()
            ):
                st.success("Headline improvement report generated!")
                with open("headline_improvement_report.md", "r") as f:
                    report_content = f.read()

                st.download_button(
                    label="Download Report",
                    data=report_content,
                    file_name="headline_improvement_report.md",
                    mime="text/markdown",
                )
            else:
                st.error("Failed to generate report")
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")


def handle_article_curation(config):
    """Handle the article curation process"""
    if not config["selected_topics"]:
        st.error("Please select at least one topic to curate")
        return

    # Initialize session state for progress tracking
    st.session_state.curation_started = True
    st.session_state.curation_complete = False

    # Create progress bar
    progress_bar = st.sidebar.progress(0, text="Starting curation process...")

    # Load models and data
    progress_bar.progress(0.1, text="Loading models and data...")
    index, articles_df, model = load_models_and_data()
    openai_client = get_openai_client()

    if index is None or articles_df is None or model is None:
        st.error("Failed to load required models and data")
        st.session_state.curation_started = False
        return

    # Curate articles for each selected topic
    all_curated_articles = []

    for i, topic in enumerate(config["selected_topics"]):
        topic_progress = (i / len(config["selected_topics"])) * 0.8 + 0.1
        progress_bar.progress(topic_progress, text=f"Curating articles for {topic}...")

        query_text = config["editorial_queries"][topic]
        topic_articles = curate_articles_for_topic(
            query_text,
            index,
            articles_df,
            model,
            openai_client,
            k=config["articles_per_topic"],
        )
        topic_articles["topic"] = topic
        all_curated_articles.append(topic_articles)

    # Combine and process all curated articles
    if all_curated_articles:
        progress_bar.progress(0.9, text="Saving curated articles...")
        full_curated_df = pd.concat(all_curated_articles, ignore_index=True)
        full_curated_df.to_csv("curated_full_daily_output.csv", index=False)

        # Update session state
        st.session_state.loaded_articles = full_curated_df
        st.session_state.curation_complete = True

        # Update the headline learning system
        try:
            headline_count = update_headline_learning(full_curated_df)
            if headline_count > 0:
                st.sidebar.success(
                    f"✅ Added {headline_count} headlines to learning system"
                )
        except Exception as e:
            logging.error(f"Error updating headline learning system: {e}")

        # Calculate evaluation metrics
        try:
            metrics = calculate_evaluation_metrics(full_curated_df)
            report = generate_research_report(metrics)
            st.session_state.evaluation_history.append(metrics)
            st.sidebar.success("✅ Research metrics calculated")
        except Exception as e:
            logging.error(f"Error calculating evaluation metrics: {e}")

        progress_bar.progress(1.0, text="Curation complete!")
        st.success("✅ Articles curated successfully!")
    else:
        st.warning("No articles were curated")
        st.session_state.curation_started = False


def display_header_and_navigation():
    """Display the header with clickable navigation"""
    # Initialize active section if it doesn't exist
    if "active_section" not in st.session_state:
        st.session_state.active_section = "tech"

    # Header
    st.markdown('<div class="header">', unsafe_allow_html=True)
    st.markdown(
        '<h1 class="newspaper-title">THE DAILY AGENT</h1>', unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Navigation with clickable buttons
    sections = {
        "POLITICS": "politics",
        "TECH": "tech",
        "BUSINESS": "business",
        "OPINION": "opinion",
        "HEALTH": "health",
        "CLIMATE": "climate",
    }

    st.markdown('<div class="nav-section">', unsafe_allow_html=True)
    cols = st.columns(len(sections))

    for idx, (section_name, section_key) in enumerate(sections.items()):
        with cols[idx]:
            if st.button(
                section_name, key=f"nav_{section_key}", help=f"Go to {section_name}"
            ):
                st.session_state.active_section = section_key
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def display_featured_article(articles_df, show_comparison):
    """Display the featured article section with updated metrics support"""
    st.markdown('<h2 class="section-title">FEATURED NEWS</h2>', unsafe_allow_html=True)

    if len(articles_df) > 0:
        main_article = articles_df.iloc[0]

        with st.container():
            st.markdown('<div class="article-box">', unsafe_allow_html=True)

            # Article tag
            st.markdown(
                f'<div class="article-tag">{main_article["topic"]}</div>',
                unsafe_allow_html=True,
            )

            # Title comparison - Simplified approach
            if show_comparison:
                # Check if we have any metrics columns
                has_new_metrics = any(
                    col in main_article.index
                    for col in [
                        "original_click_probability",
                        "rewritten_click_probability",
                        "probability_improvement",
                    ]
                )

                has_old_metrics = any(
                    col in main_article.index
                    for col in [
                        "headline_improvement",
                        "headline_ctr_original",
                        "headline_ctr_rewritten",
                    ]
                )

                if has_new_metrics:
                    # Use new clear metrics display
                    display_headline_with_clear_metrics(
                        main_article["original_title"],
                        main_article["rewritten_title"],
                        {
                            "original_click_probability": main_article.get(
                                "original_click_probability", 0
                            ),
                            "rewritten_click_probability": main_article.get(
                                "rewritten_click_probability", 0
                            ),
                            "probability_improvement": main_article.get(
                                "probability_improvement", 0
                            ),
                            "improvement_category": main_article.get(
                                "improvement_category", "Unknown"
                            ),
                            "key_improvements": main_article.get(
                                "key_improvements", ""
                            ),
                        },
                    )
                elif has_old_metrics:
                    # Convert old format to new format and use clear metrics
                    old_to_new = {
                        "original_click_probability": main_article.get(
                            "headline_ctr_original", 0
                        )
                        / 100,
                        "rewritten_click_probability": main_article.get(
                            "headline_ctr_rewritten", 0
                        )
                        / 100,
                        "probability_improvement": main_article.get(
                            "headline_improvement", 0
                        ),
                        "improvement_category": "Legacy Data",
                        "key_improvements": main_article.get(
                            "headline_key_factors", ""
                        ),
                    }
                    display_headline_with_clear_metrics(
                        main_article["original_title"],
                        main_article["rewritten_title"],
                        old_to_new,
                    )
            else:
                st.subheader(main_article["rewritten_title"])

            # Rest of the function remains the same...
            # Author byline
            author = random.choice(
                ["Sarah Chen", "Michael Johnson", "Priya Patel", "Robert Williams"]
            )
            st.markdown(
                f'<div class="article-byline">By {author} | {datetime.datetime.now().strftime("%B %d, %Y")}</div>',
                unsafe_allow_html=True,
            )

            # Image
            display_article_image(
                main_article["topic"], article_id=main_article.name, is_main=True
            )

            # Abstract
            abstract = main_article["abstract"]
            if abstract and len(abstract) > 300:
                abstract = abstract[:300] + "..."
            st.write(abstract)

            # Why it matters
            st.markdown(
                f'<div class="article-why-matters"><strong>Why it matters:</strong> {main_article["explanation"]}</div>',
                unsafe_allow_html=True,
            )

            # Read more link
            st.markdown("**Continue Reading →**")

            st.markdown("</div>", unsafe_allow_html=True)


def display_trending_articles(articles_df, show_comparison):
    """Display the trending articles section"""
    st.markdown('<h2 class="section-title">TRENDING NOW</h2>', unsafe_allow_html=True)

    trend_cols = st.columns(3)

    for i in range(1, min(7, len(articles_df))):
        if i < len(articles_df):
            col_idx = (i - 1) % 3

            with trend_cols[col_idx]:
                article = articles_df.iloc[i]

                st.markdown('<div class="article-box">', unsafe_allow_html=True)

                # Article tag
                st.markdown(
                    f'<div class="article-tag">{article["topic"]}</div>',
                    unsafe_allow_html=True,
                )

                # Title comparison
                # Title comparison - robust version
                if show_comparison:
                    # Check what columns actually exist
                    original_col = (
                        "original_title" if "original_title" in article else "title"
                    )
                    rewritten_col = (
                        "rewritten_title" if "rewritten_title" in article else "title"
                    )

                    original_title = str(
                        article.get(original_col, "Original not available")
                    )
                    rewritten_title = str(
                        article.get(rewritten_col, "Rewritten not available")
                    )

                    st.markdown(
                        f"""
                        <div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                            <div style="font-style: italic; color: #666; font-size: 12px;">
                                Original: {original_title}
                            </div>
                            <div style="font-weight: bold; color: #000;">
                                {rewritten_title}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    title = str(
                        article.get("rewritten_title", article.get("title", "No title"))
                    )
                    st.markdown(f"### {title}")

                # Image
                display_article_image(article["topic"], article_id=article.name)

                # Short abstract
                abstract = article["abstract"]
                if abstract and len(abstract) > 150:
                    abstract = abstract[:150] + "..."
                st.write(abstract)

                # Read more link
                st.markdown("**Read More →**")

                st.markdown("</div>", unsafe_allow_html=True)


def display_topic_sections(articles_df, show_comparison):
    """Display topic sections based on active selection"""
    # Get the active section from session state
    active_section = st.session_state.get("active_section", "tech")

    # Topic mapping
    topic_mapping = {
        "tech": "Top Technology News",
        "business": "Business and Economy",
        "politics": "Global Politics",
        "health": "Health News",
        "climate": "Climate News",
        "opinion": "Opinion & Analysis",
    }

    # Filter articles for the active section
    # You may need to adjust this filtering logic based on your data structure
    section_articles = (
        articles_df[articles_df["topic"].str.lower() == active_section]
        if "topic" in articles_df.columns
        else articles_df
    )

    # Display only the active section
    if not section_articles.empty:
        topic_name = topic_mapping.get(active_section, active_section.title() + " News")
        st.markdown(
            f'<h2 class="section-title">{topic_name.upper()}</h2>',
            unsafe_allow_html=True,
        )

        # Display articles (you can reuse your existing article display logic here)
        display_articles_in_section(section_articles, show_comparison)
    else:
        st.warning(f"No articles found for {active_section}")


def display_articles_in_section(articles, show_comparison):
    """Display articles in a grid format"""
    cols = st.columns(3)

    for idx, (_, article) in enumerate(
        articles.head(9).iterrows()
    ):  # Show max 9 articles per section
        col_idx = idx % 3
        with cols[col_idx]:
            # Article container
            st.markdown('<div class="article-box">', unsafe_allow_html=True)

            # Article image
            if "thumbnail_url" in article and pd.notna(article["thumbnail_url"]):
                st.image(article["thumbnail_url"], use_column_width=True)

            # Article topic tag
            if "topic" in article:
                st.markdown(
                    f'<div class="article-tag">{article["topic"].title()}</div>',
                    unsafe_allow_html=True,
                )

            # Article title
            if (
                show_comparison
                and "rewritten_title" in article
                and pd.notna(article["rewritten_title"])
            ):
                # Show comparison if enabled
                st.markdown(
                    f"""
                <div class="title-comparison">
                    <div class="original-headline">Original: {article.get("title", "")}</div>
                    <div class="rewritten-headline">{article["rewritten_title"]}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                # Show regular title
                title = article.get("rewritten_title") or article.get(
                    "title", "No title"
                )
                st.markdown(f"<h3>{title}</h3>", unsafe_allow_html=True)

            # Article content
            if "explanation" in article and pd.notna(article["explanation"]):
                st.markdown(
                    f'<div class="article-why-matters"><strong>Why it matters:</strong> {article["explanation"]}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)


def display_footer():
    """Display the footer section"""
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.markdown("<h3>THE DAILY AGENT</h3>", unsafe_allow_html=True)
    st.markdown(
        f"<p>© {datetime.datetime.now().year} The Daily Agent. All Rights Reserved.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<p>Powered by AI-curated content</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def display_newspaper_layout(articles_df, show_comparison):
    """Display the complete newspaper layout"""
    display_header_and_navigation()
    display_featured_article(articles_df, show_comparison)
    display_trending_articles(articles_df, show_comparison)
    display_topic_sections(articles_df, show_comparison)
    display_footer()
    show_model_dashboard()


def display_welcome_page():
    """Display the welcome page when no articles are loaded"""
    if (
        "curation_started" in st.session_state
        and st.session_state.curation_started
        and not st.session_state.get("curation_complete", False)
    ):
        st.info(
            "⏳ Curation in progress... Please wait while we prepare your personalized news articles."
        )
    else:
        st.header("Welcome to The Daily Agent")
        st.write("Your AI-powered personalized news platform")

        st.subheader("Get Started in Two Simple Steps:")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 1. Select Topics")
            st.info(
                "Choose which news categories you want to include using the sidebar controls"
            )
            try:
                if os.path.exists("tech1.jpg"):
                    st.image("tech1.jpg", width=300)
            except:
                pass

        with col2:
            st.markdown("### 2. Curate Articles")
            st.success(
                "Click 'CURATE FRESH ARTICLES' to generate your personalized newspaper"
            )
            try:
                if os.path.exists("health1.jpg"):
                    st.image("health1.jpg", width=300)
            except:
                pass

        st.subheader("YOUR PERSONALIZED NEWSPAPER")
        st.write(
            "The Daily Agent will create a professional newspaper using your selected topics and AI-curated content."
        )

        st.markdown(
            """
        <div style="border: 1px solid #ddd; padding: 20px; border-radius: 5px; text-align: center; margin: 20px 0; background-color: #f9f9f9;">
            <h3 style="margin-bottom: 15px;">Sample Layout Preview</h3>
            <p>Your newspaper will include featured articles, trending stories, and topic-specific sections with images.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )


# --- Main Application Logic ---
def main():
    initialize_session_state()

    # Initialize headline learner
    if "headline_learner" not in st.session_state:
        st.session_state.headline_learner = HeadlineLearningLoop(
            data_file="headline_learning_data.csv",
            model_path="model_output/ctr_model.pkl",
        )

    # Check for required model files
    if not os.path.exists("model_output/ctr_model.pkl"):
        st.error(
            "⚠️ CTR model not found! Please ensure model_output/ctr_model.pkl exists."
        )
        st.info(
            "Run the CTR model training script first to generate the required model files."
        )
        return

    # Check for required data files
    data_dir = "agentic_news_editor/processed_data/"
    if not os.path.exists(
        os.path.join(data_dir, "articles_with_embeddings.csv")
    ) or not os.path.exists("articles_faiss.index"):
        st.error(
            f"Required data files not found in {data_dir}. Please run the EDA pipeline first."
        )
        return

    config = setup_sidebar()
    if config["curate_button"]:
        handle_article_curation(config)

    if config["load_button"]:
        with st.spinner("Loading previously curated articles..."):
            st.session_state.loaded_articles = load_curated_articles()
            if st.session_state.loaded_articles is not None:
                st.success(f"Loaded {len(st.session_state.loaded_articles)} articles")
            else:
                st.error("No previously curated articles found")

    # Display the main content
    if (
        "loaded_articles" in st.session_state
        and st.session_state.loaded_articles is not None
        and len(st.session_state.loaded_articles) > 0
    ):
        display_newspaper_layout(
            st.session_state.loaded_articles,
            st.session_state.get("show_headline_comparison", True),
        )
    else:
        display_welcome_page()


if __name__ == "__main__":
    main()
