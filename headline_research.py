import os
import json
import numpy as np
import matplotlib.pyplot as plt
import textstat
from scipy import stats

# Try to import headline pairs from the user's file
try:
    from headline_pairs import headline_pairs
    print(f"Successfully imported {len(headline_pairs)} headline pairs from headline_pairs.py")
except ImportError:
    print("headline_pairs.py module could not be imported directly. Trying JSON file instead.")
    # If headline_pairs.py doesn't exist or can't be imported, try to load from JSON file
    try:
        with open('headline_pairs.json', 'r') as f:
            headline_pairs = json.load(f)
        print(f"Successfully loaded {len(headline_pairs)} headline pairs from headline_pairs.json")
    except FileNotFoundError:
        print("headline_pairs.json not found. Using sample data instead.")
        # Sample headline pairs as fallback
        headline_pairs = {
            "headline1": {
                "original": "Scientists Make Groundbreaking Discovery in Cancer Research That Could Change Everything",
                "rewritten": "New Cancer Research Breakthrough Could Transform Treatment Options"
            },
            "headline2": {
                "original": "10 Ways the Economic Downturn Might Impact Your Financial Future",
                "rewritten": "How the Economy Affects Your Finances: 5 Key Points"
            },
            "headline3": {
                "original": "Study Shows Unprecedented Climate Change Effects are Accelerating Faster Than Previous Models Indicated",
                "rewritten": "Climate Change Accelerating Faster Than Expected, New Study Finds"
            }
        }
        print(f"Using {len(headline_pairs)} sample headline pairs")
    except json.JSONDecodeError:
        print("Error: headline_pairs.json file exists but contains invalid JSON. Using sample data.")
        # Provide the same sample data as above
        headline_pairs = {
            "headline1": {
                "original": "Scientists Make Groundbreaking Discovery in Cancer Research That Could Change Everything",
                "rewritten": "New Cancer Research Breakthrough Could Transform Treatment Options"
            },
            "headline2": {
                "original": "10 Ways the Economic Downturn Might Impact Your Financial Future",
                "rewritten": "How the Economy Affects Your Finances: 5 Key Points"
            },
            "headline3": {
                "original": "Study Shows Unprecedented Climate Change Effects are Accelerating Faster Than Previous Models Indicated",
                "rewritten": "Climate Change Accelerating Faster Than Expected, New Study Finds"
            }
        }
        print(f"Using {len(headline_pairs)} sample headline pairs")

# Ensure results directories exist
results_dir = "results"
figures_dir = os.path.join(results_dir, "figures")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)
print(f"Created/verified directories:\n- {results_dir}\n- {figures_dir}")

def extract_headline_features(headline):
    """Extract features from headline text for engagement prediction"""
    # Length features
    length = len(headline)
    word_count = len(headline.split())
    
    # Readability
    try:
        fre = textstat.flesch_reading_ease(headline)
        fkg = textstat.flesch_kincaid_grade(headline)
    except Exception as e:
        print(f"Error calculating readability for '{headline}': {e}")
        fre = 50  # Default middle value
        fkg = 8   # Default middle value
    
    # Complex words
    try:
        complex_words = textstat.difficult_words(headline)
        complex_word_pct = complex_words / word_count if word_count > 0 else 0
    except Exception as e:
        print(f"Error calculating complex words for '{headline}': {e}")
        complex_words = 0
        complex_word_pct = 0
    
    # Average word length
    try:
        avg_word_length = sum(len(word) for word in headline.split()) / word_count if word_count > 0 else 0
    except Exception as e:
        print(f"Error calculating avg word length for '{headline}': {e}")
        avg_word_length = 5  # Default average
    
    # Return as a dictionary for easier interpretation
    return {
        "length": length,
        "word_count": word_count,
        "flesch_reading_ease": fre,
        "flesch_kincaid_grade": fkg,
        "complex_words": complex_words,
        "complex_word_percentage": complex_word_pct,
        "avg_word_length": avg_word_length
    }

def predict_engagement(headline_features):
    """Predict engagement (CTR) based on headline features"""
    # Simple predictive model based on readability and brevity
    # Higher FRE = better, Lower complex words = better, Shorter length = better (to a point)
    
    # Normalize FRE to 0-1 scale (0 = worst, 1 = best)
    fre_score = headline_features["flesch_reading_ease"] / 100.0
    
    # Normalize complexity (lower is better)
    complexity_penalty = headline_features["complex_word_percentage"] * 0.5
    
    # Word count sweet spot (penalize if too short or too long)
    word_count = headline_features["word_count"]
    if word_count < 5:
        length_score = word_count / 5.0  # Penalize very short headlines
    elif word_count > 15:
        length_score = 1.0 - ((word_count - 15) / 20.0)  # Gradually penalize longer headlines
        length_score = max(0, length_score)  # Don't go below 0
    else:
        length_score = 1.0  # Ideal range
    
    # Combine factors (with weights)
    raw_score = (fre_score * 0.5) - (complexity_penalty * 0.3) + (length_score * 0.2)
    
    # Convert to probability (CTR typically ranges from 0.5% to 5%)
    ctr = 0.005 + (raw_score * 0.045)  # Scale to 0.5% - 5% range
    return min(max(ctr, 0.001), 0.10)  # Clamp between 0.1% and 10%

def analyze_headline_engagement():
    """Analyze engagement metrics for headline pairs"""
    results = {}
    
    print("\n=== Analyzing Headline Engagement ===\n")
    
    # Validate headline_pairs format
    if not isinstance(headline_pairs, dict):
        print(f"Error: headline_pairs is not a dictionary. Type: {type(headline_pairs)}")
        print(f"Content: {headline_pairs}")
        return {}
    
    print(f"Processing {len(headline_pairs)} headline pairs...")
    
    for headline_id, headline_pair in headline_pairs.items():
        # Validate headline pair format
        if not isinstance(headline_pair, dict):
            print(f"Skipping invalid headline pair for ID {headline_id}: {headline_pair}")
            continue
            
        if "original" not in headline_pair or "rewritten" not in headline_pair:
            print(f"Skipping headline pair missing required keys for ID {headline_id}: {headline_pair}")
            continue
        
        original = headline_pair["original"]
        rewritten = headline_pair["rewritten"]
        
        # Extract features
        try:
            original_features = extract_headline_features(original)
            rewritten_features = extract_headline_features(rewritten)
        except Exception as e:
            print(f"Error extracting features for headline {headline_id}: {e}")
            continue
        
        # Predict click-through rates
        try:
            original_ctr = predict_engagement(original_features)
            rewritten_ctr = predict_engagement(rewritten_features)
        except Exception as e:
            print(f"Error predicting engagement for headline {headline_id}: {e}")
            continue
        
        # Calculate improvements
        absolute_improvement = rewritten_ctr - original_ctr
        relative_improvement = (absolute_improvement / original_ctr) * 100 if original_ctr > 0 else 0
        
        # Store results
        results[headline_id] = {
            "original": {
                "headline": original,
                "features": original_features,
                "predicted_ctr": original_ctr
            },
            "rewritten": {
                "headline": rewritten,
                "features": rewritten_features,
                "predicted_ctr": rewritten_ctr
            },
            "improvements": {
                "absolute": absolute_improvement,
                "relative_percent": relative_improvement
            }
        }
        
        # Print results for each headline
        print(f"Headline {headline_id}:")
        print(f"  Original: {original}")
        print(f"  Rewritten: {rewritten}")
        print(f"  Original CTR: {original_ctr:.2%}")
        print(f"  Rewritten CTR: {rewritten_ctr:.2%}")
        print(f"  Improvement: {absolute_improvement:.2%} ({relative_improvement:.1f}%)")
        print()
    
    # If no valid results, return empty dict
    if not results:
        print("No valid headline pairs were processed. Check your headline_pairs.py or headline_pairs.json file.")
        return {}
    
    # Calculate averages
    all_orig_ctrs = [data["original"]["predicted_ctr"] for data in results.values()]
    all_rewr_ctrs = [data["rewritten"]["predicted_ctr"] for data in results.values()]
    all_abs_improvements = [data["improvements"]["absolute"] for data in results.values()]
    all_rel_improvements = [data["improvements"]["relative_percent"] for data in results.values()]
    
    avg_orig_ctr = np.mean(all_orig_ctrs)
    avg_rewr_ctr = np.mean(all_rewr_ctrs)
    avg_abs_improvement = np.mean(all_abs_improvements)
    avg_rel_improvement = np.mean(all_rel_improvements)
    
    # Run t-test to check if improvement is significant
    t_stat, p_value = stats.ttest_rel(all_orig_ctrs, all_rewr_ctrs)
    is_significant = p_value < 0.05
    
    # Print summary stats
    print("\n=== Summary Statistics ===\n")
    print(f"Average Original CTR: {avg_orig_ctr:.2%}")
    print(f"Average Rewritten CTR: {avg_rewr_ctr:.2%}")
    print(f"Average Absolute Improvement: {avg_abs_improvement:.2%}")
    print(f"Average Relative Improvement: {avg_rel_improvement:.1f}%")
    print(f"T-test: t={t_stat:.2f}, p={p_value:.4f} ({'' if is_significant else 'not '}statistically significant)")
    
    # Plot results
    plot_engagement_results(results)
    
    # Save results to file
    try:
        with open(os.path.join(results_dir, "engagement_results.json"), "w") as f:
            # Convert results to JSON-serializable format
            serializable_results = {}
            for headline_id, data in results.items():
                serializable_results[headline_id] = {
                    "original": {
                        "headline": data["original"]["headline"],
                        "features": {k: float(v) for k, v in data["original"]["features"].items()},
                        "predicted_ctr": float(data["original"]["predicted_ctr"])
                    },
                    "rewritten": {
                        "headline": data["rewritten"]["headline"],
                        "features": {k: float(v) for k, v in data["rewritten"]["features"].items()},
                        "predicted_ctr": float(data["rewritten"]["predicted_ctr"])
                    },
                    "improvements": {
                        "absolute": float(data["improvements"]["absolute"]),
                        "relative_percent": float(data["improvements"]["relative_percent"])
                    }
                }
            
            # Add summary statistics
            serializable_results["summary"] = {
                "average_original_ctr": float(avg_orig_ctr),
                "average_rewritten_ctr": float(avg_rewr_ctr),
                "average_absolute_improvement": float(avg_abs_improvement),
                "average_relative_improvement": float(avg_rel_improvement),
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "statistically_significant": "yes" if is_significant else "no" 
            }
            
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to {os.path.join(results_dir, 'engagement_results.json')}")
    except Exception as e:
        print(f"Error saving results to JSON: {e}")
    
    return results

def plot_engagement_results(results):
    """Generate plots for engagement analysis"""
    
    if not results:
        print("No results to plot.")
        return
    
    print(f"Generating plots for {len(results)} headline pairs...")
    
    # 1. CTR Comparison
    try:
        plt.figure(figsize=(12, 6))
        
        headline_ids = list(results.keys())
        original_ctrs = [results[hid]["original"]["predicted_ctr"] for hid in headline_ids]
        rewritten_ctrs = [results[hid]["rewritten"]["predicted_ctr"] for hid in headline_ids]
        
        x = np.arange(len(headline_ids))
        width = 0.35
        
        plt.bar(x - width/2, [ctr * 100 for ctr in original_ctrs], width, label='Original')
        plt.bar(x + width/2, [ctr * 100 for ctr in rewritten_ctrs], width, label='Rewritten')
        
        plt.xlabel('Headline ID')
        plt.ylabel('Predicted Click-Through Rate (%)')
        plt.title('CTR Comparison: Original vs Rewritten Headlines')
        
        # Only show a subset of x ticks if there are too many headlines
        if len(headline_ids) > 15:
            subset_indices = np.linspace(0, len(headline_ids)-1, 15, dtype=int)
            plt.xticks(subset_indices, [f"H{i+1}" for i in subset_indices], rotation=45)
        else:
            plt.xticks(x, [f"H{i+1}" for i in range(len(headline_ids))], rotation=45)
        
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(figures_dir, "ctr_comparison.png"))
        plt.close()
        print("Created CTR comparison plot")
    except Exception as e:
        print(f"Error creating CTR comparison plot: {e}")
    
    # 2. Relative Improvements
    try:
        plt.figure(figsize=(12, 6))
        
        improvements = [results[hid]["improvements"]["relative_percent"] for hid in headline_ids]
        
        bars = plt.bar(range(len(headline_ids)), improvements)
        
        # Color bars based on improvement
        for i, improvement in enumerate(improvements):
            color = 'green' if improvement > 0 else 'red'
            bars[i].set_color(color)
        
        # Add average line
        avg_improvement = np.mean(improvements)
        plt.axhline(y=avg_improvement, color='black', linestyle='--', 
                label=f'Average: {avg_improvement:.1f}%')
        
        plt.xlabel('Headline ID')
        plt.ylabel('Relative CTR Improvement (%)')
        plt.title('Predicted CTR Improvement from Headline Rewriting')
        
        # Only show a subset of x ticks if there are too many headlines
        if len(headline_ids) > 15:
            subset_indices = np.linspace(0, len(headline_ids)-1, 15, dtype=int)
            plt.xticks(subset_indices, [f"H{i+1}" for i in subset_indices], rotation=45)
        else:
            plt.xticks(range(len(headline_ids)), [f"H{i+1}" for i in range(len(headline_ids))], rotation=45)
        
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(figures_dir, "ctr_improvements.png"))
        plt.close()
        print("Created CTR improvements plot")
    except Exception as e:
        print(f"Error creating improvements plot: {e}")
    
    # 3. Feature comparison radar chart for a sample headline
    try:
        if headline_ids:
            sample_headline_id = headline_ids[0]
            original_features = results[sample_headline_id]["original"]["features"]
            rewritten_features = results[sample_headline_id]["rewritten"]["features"]
            
            # Select features to plot
            feature_names = ["flesch_reading_ease", "flesch_kincaid_grade", "complex_word_percentage", 
                            "avg_word_length", "word_count"]
            feature_labels = ["Reading Ease", "Grade Level", "Complex Words %", 
                             "Avg Word Length", "Word Count"]
            
            # Normalize features for radar chart
            def normalize_feature(name, value):
                if name == "flesch_reading_ease":
                    return value / 100  # 0-100 scale
                elif name == "flesch_kincaid_grade":
                    return (20 - value) / 20  # Inverse scale (lower is better)
                elif name == "complex_word_percentage":
                    return 1 - value  # Inverse scale (lower is better)
                elif name == "avg_word_length":
                    return (10 - value) / 10  # Inverse scale (lower is better)
                elif name == "word_count":
                    # Optimal around 8-12 words
                    if value < 8:
                        return value / 8
                    elif value > 12:
                        return 12 / value
                    else:
                        return 1.0
            
            orig_values = [normalize_feature(name, original_features[name]) for name in feature_names]
            rewr_values = [normalize_feature(name, rewritten_features[name]) for name in feature_names]
            
            # Create radar chart
            plt.figure(figsize=(8, 8))
            ax = plt.subplot(111, polar=True)
            
            # Compute angles for each feature
            angles = np.linspace(0, 2*np.pi, len(feature_names), endpoint=False).tolist()
            
            # Plot data - create a closed polygon
            # Add the first values again at the end to close the polygon
            orig_values_closed = orig_values + [orig_values[0]]
            rewr_values_closed = rewr_values + [rewr_values[0]]
            angles_closed = angles + [angles[0]]
            
            ax.plot(angles_closed, orig_values_closed, 'b-', linewidth=1.5, label='Original')
            ax.plot(angles_closed, rewr_values_closed, 'r-', linewidth=1.5, label='Rewritten')
            ax.fill(angles_closed, orig_values_closed, 'b', alpha=0.1)
            ax.fill(angles_closed, rewr_values_closed, 'r', alpha=0.1)
            
            # Set labels and title - use the correct number of labels
            ax.set_xticks(angles)  # Set the correct number of ticks
            ax.set_xticklabels(feature_labels)  # Apply labels to those ticks
            ax.set_ylim(0, 1)
            ax.set_title('Headline Feature Comparison (Higher is Better)', y=1.08)
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, "headline_features_radar.png"))
            plt.close()
            print("Created headline features radar plot")
    except Exception as e:
        print(f"Error creating radar plot: {e}")
        
    # 4. Distribution of improvements histogram
    try:
        plt.figure(figsize=(10, 6))
        
        # We'll use bins centered at a zero improvement
        improvements = [results[hid]["improvements"]["relative_percent"] for hid in headline_ids]
        max_improvement = max(abs(min(improvements)), abs(max(improvements)))
        
        bin_range = np.linspace(-max_improvement, max_improvement, 20)
        
        plt.hist(improvements, bins=bin_range, alpha=0.7, edgecolor='black')
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
        avg_improvement = np.mean(improvements)
        plt.axvline(x=avg_improvement, color='red', linestyle='-', linewidth=2, 
                   label=f'Average: {avg_improvement:.1f}%')
        
        plt.xlabel('Relative CTR Improvement (%)')
        plt.ylabel('Number of Headlines')
        plt.title('Distribution of CTR Improvements')
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(figures_dir, "improvement_distribution.png"))
        plt.close()
        print("Created improvement distribution histogram")
    except Exception as e:
        print(f"Error creating histogram: {e}")
    
    print(f"All plots saved to {figures_dir}")

# Run the analysis
if __name__ == "__main__":
    print("Starting headline engagement analysis...")
    engagement_results = analyze_headline_engagement()
    
    if engagement_results:
        print("\nAnalysis complete! Check the 'results' directory for full results and visualizations.")
    else:
        print("\nAnalysis could not be completed due to errors or missing data.")
        print("Please check your headline_pairs.py or headline_pairs.json file and try again.")