import pandas as pd
import os

# Use proper path joining for cross-platform compatibility
data_dir = os.path.join('agentic_news_editor', 'processed_data')
train_path = os.path.join(data_dir, 'train_headline_ctr.csv')
val_path = os.path.join(data_dir, 'val_headline_ctr.csv')
test_path = os.path.join(data_dir, 'test_headline_ctr.csv')

# Print paths for debugging
print(f"Looking for files at:")
print(f"Train: {os.path.abspath(train_path)}")
print(f"Val: {os.path.abspath(val_path)}")
print(f"Test: {os.path.abspath(test_path)}")

# Check if files exist
print(f"Files exist?")
print(f"Train: {os.path.exists(train_path)}")
print(f"Val: {os.path.exists(val_path)}")
print(f"Test: {os.path.exists(test_path)}")

# Load the processed datasets
train_data = pd.read_csv(train_path)
val_data = pd.read_csv(val_path)
test_data = pd.read_csv(test_path)

# Print basic stats
print(f"\nDataset sizes:")
print(f"Train dataset: {len(train_data)} headlines")
print(f"Validation dataset: {len(val_data)} headlines")
print(f"Test dataset: {len(test_data)} headlines")

# Check CTR distribution in train data
print("\nTrain CTR statistics:")
print(f"Average CTR: {train_data['ctr'].mean():.5f}")
print(f"Median CTR: {train_data['ctr'].median():.5f}")
print(f"Min CTR: {train_data['ctr'].min():.5f}")
print(f"Max CTR: {train_data['ctr'].max():.5f}")