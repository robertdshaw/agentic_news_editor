import pandas as pd

# Load your data
train_data = pd.read_csv("agentic_news_editor/processed_data/train_headline_ctr.csv")
val_data = pd.read_csv("agentic_news_editor/processed_data/val_headline_ctr.csv")

# Check basic statistics
print("Train CTR range:", train_data['ctr'].min(), "to", train_data['ctr'].max())
print("Train CTR mean:", train_data['ctr'].mean())
print("Val CTR mean:", val_data['ctr'].mean())

# Check if there are extreme outliers
print("\nTrain CTR distribution:")
print(train_data['ctr'].describe())

# Check a few example headlines
print("\nSample headlines:")
print(train_data[['title', 'ctr']].head(5))