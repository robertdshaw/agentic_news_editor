import pandas as pd
import logging
import os
import datetime
from pathlib import Path
from headline_metrics import HeadlineMetrics


class HeadlineLearningLoop:
    """
    A simple system that collects and analyzes headline improvements:
    1. Stores original and rewritten headlines with metrics
    2. Tracks which rewrites work best
    3. Generates simple reports with useful insights
    """

    def __init__(
        self,
        data_file="headline_learning_data.csv",
        model_path="model_output/ctr_model.pkl",
    ):
        """Initialize with data file and model path"""
        self.data_file = data_file

        # Initialize metrics analyzer
        try:
            self.metrics_analyzer = HeadlineMetrics(model_path=model_path)
            logging.info("Initialized HeadlineMetrics with CTR model")
        except Exception as e:
            logging.error(f"Error initializing HeadlineMetrics: {e}")
            raise

        # Load existing data or create new dataset
        if os.path.exists(data_file):
            self.data = pd.read_csv(data_file)
            logging.info(f"Loaded {len(self.data)} headline pairs from {data_file}")
        else:
            self.data = pd.DataFrame(
                {
                    "original_title": [],
                    "rewritten_title": [],
                    "headline_ctr_original": [],
                    "headline_ctr_rewritten": [],
                    "headline_improvement": [],
                    "headline_key_factors": [],
                    "topic": [],
                    "timestamp": [],
                }
            )
            logging.info("Created new headline learning dataset")

    def add_headline_pair(self, original, rewritten, topic=None):
        """Add a single headline pair with metrics"""
        try:
            # Get metrics comparison
            comparison = self.metrics_analyzer.compare_headlines(original, rewritten)

            # Create entry
            new_entry = pd.DataFrame(
                {
                    "original_title": [original],
                    "rewritten_title": [rewritten],
                    "headline_ctr_original": [comparison["original_ctr"]],
                    "headline_ctr_rewritten": [comparison["rewritten_ctr"]],
                    "headline_improvement": [comparison["score_percent_change"]],
                    "headline_key_factors": [", ".join(comparison["key_improvements"])],
                    "topic": [topic if topic else "General"],
                    "timestamp": [
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ],
                }
            )

            # Add to dataset and save
            self.data = pd.concat([self.data, new_entry], ignore_index=True)
            self.data.to_csv(self.data_file, index=False)

            logging.info(
                f"Added headline pair: {comparison['score_percent_change']:.1f}% improvement"
            )
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

                # Use pre-computed metrics if available
                if all(
                    col in row
                    for col in [
                        "headline_ctr_original",
                        "headline_ctr_rewritten",
                        "headline_improvement",
                        "headline_key_factors",
                    ]
                ):
                    new_entry = pd.DataFrame(
                        {
                            "original_title": [row["original_title"]],
                            "rewritten_title": [row["rewritten_title"]],
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
                    # Calculate new metrics
                    if self.add_headline_pair(
                        row["original_title"], row["rewritten_title"], topic
                    ):
                        count += 1

        # Save if we added anything
        if count > 0:
            self.data.to_csv(self.data_file, index=False)
            logging.info(f"Added {count} headline pairs from dataframe")

        return count

    def get_insights_summary(self):
        """Get basic insights about headline performance"""
        try:
            # Default values if not enough data
            if len(self.data) < 3:
                return {
                    "total_headlines": len(self.data),
                    "improvement_rate": 0,
                    "avg_improvement": 0,
                    "most_common_factors": [],
                    "needs_more_data": True,
                }

            # Calculate basic stats
            improvements = self.data["headline_improvement"]
            improvement_rate = (improvements > 0).mean()
            avg_improvement = improvements.mean()

            # Get improvement factors
            factors = []
            for factor_list in self.data["headline_key_factors"]:
                if isinstance(factor_list, str):
                    factors.extend(
                        [f.strip() for f in factor_list.split(",") if f.strip()]
                    )

            # Count factors
            factor_counts = pd.Series(factors).value_counts()
            top_factors = (
                factor_counts.head(5).index.tolist() if not factor_counts.empty else []
            )

            return {
                "total_headlines": len(self.data),
                "improvement_rate": improvement_rate,
                "avg_improvement": avg_improvement,
                "most_common_factors": top_factors,
                "needs_more_data": False,
            }
        except Exception as e:
            logging.error(f"Error getting insights: {e}")
            return {
                "total_headlines": len(self.data) if hasattr(self, "data") else 0,
                "improvement_rate": 0,
                "avg_improvement": 0,
                "most_common_factors": [],
                "error": str(e),
            }

    def prompt_improvement_report(self):
        """Generate a simple headline improvement report"""
        try:
            insights = self.get_insights_summary()

            # Check if we have enough data
            if insights["total_headlines"] < 3:
                logging.warning("Not enough headline data for a report (need 3+)")
                return False

            # Create report
            report = f"""# Headline Improvement Report
Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}

## Summary
- **Total Headlines Analyzed**: {insights["total_headlines"]}
- **Headlines Improved**: {int(insights["improvement_rate"] * insights["total_headlines"])} ({insights["improvement_rate"]:.1%})
- **Average Improvement**: {insights["avg_improvement"]:.1f}%

## Most Effective Headline Tactics
"""
            # Add top factors
            if insights["most_common_factors"]:
                for factor in insights["most_common_factors"]:
                    report += f"- {factor}\n"
            else:
                report += "- Not enough data to determine patterns yet\n"

            report += "\n## Top Performing Headlines\n"

            # Add best headlines
            top_headlines = self.data.sort_values(
                "headline_improvement", ascending=False
            ).head(3)
            for i, (_, row) in enumerate(top_headlines.iterrows()):
                report += (
                    f"\n### {i+1}. Improvement: {row['headline_improvement']:.1f}%\n"
                )
                report += f"**Original**: {row['original_title']}\n\n"
                report += f"**Rewritten**: {row['rewritten_title']}\n\n"
                if (
                    isinstance(row["headline_key_factors"], str)
                    and row["headline_key_factors"]
                ):
                    report += f"**What worked**: {row['headline_key_factors']}\n\n"

            # Save report
            Path("headline_improvement_report.md").write_text(report, encoding="utf-8")
            logging.info("Generated headline improvement report")

            return True
        except Exception as e:
            logging.error(f"Error generating report: {e}")
            return False
