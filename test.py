# headline_research_test.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textstat

# Create necessary directories
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, "results")
figures_dir = os.path.join(results_dir, "figures")

os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

print(f"Created directories:\n- {results_dir}\n- {figures_dir}")

# Sample headlines
sample_headlines = {
    "headline1": {
        "original": "Study finds correlation between coffee consumption and productivity",
        "rewritten": "Coffee boosts productivity at work, new research shows"
    },
    "headline2": {
        "original": "International researchers discover potential new treatment for cancer",
        "rewritten": "Breakthrough cancer treatment discovered by global research team"
    }
}

# Basic readability analysis
results = {}
for headline_id, pair in sample_headlines.items():
    original = pair["original"]
    rewritten = pair["rewritten"]
    
    # Calculate readability
    original_fre = textstat.flesch_reading_ease(original)
    rewritten_fre = textstat.flesch_reading_ease(rewritten)
    
    results[headline_id] = {
        "original_fre": original_fre,
        "rewritten_fre": rewritten_fre,
        "improvement": rewritten_fre - original_fre
    }

# Create a simple plot
plt.figure(figsize=(10, 6))
headlines = list(results.keys())
original_scores = [results[h]["original_fre"] for h in headlines]
rewritten_scores = [results[h]["rewritten_fre"] for h in headlines]

# Plot
x = np.arange(len(headlines))
width = 0.35
plt.bar(x - width/2, original_scores, width, label='Original')
plt.bar(x + width/2, rewritten_scores, width, label='Rewritten')
plt.xlabel('Headlines')
plt.ylabel('Flesch Reading Ease')
plt.title('Readability Comparison')
plt.xticks(x, headlines)
plt.legend()

# Save plot
plot_path = os.path.join(figures_dir, "test_plot.png")
plt.savefig(plot_path)
print(f"Saved test plot to {plot_path}")

# Create a simple report
report = "# Headline Analysis Test Report\n\n"
report += "## Readability Results\n\n"
report += "| Headline | Original | Rewritten | Improvement |\n"
report += "|----------|----------|-----------|-------------|\n"

for headline_id, result in results.items():
    report += f"| {headline_id} | {result['original_fre']:.2f} | {result['rewritten_fre']:.2f} | {result['improvement']:.2f} |\n"

# Save report
report_path = os.path.join(results_dir, "test_report.md")
with open(report_path, "w") as f:
    f.write(report)
print(f"Saved test report to {report_path}")

print("Test completed successfully!")