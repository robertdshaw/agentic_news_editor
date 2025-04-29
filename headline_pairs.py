import pandas as pd
import json
import glob
import os

# Step 1: Define the path pattern to your CSV files
csv_directory = r"C:\Users\rshaw\Desktop\EC Utbildning - Data Science\Thesis\Agentic_AI_News_Editor project\agentic_ai_editor_project"  # Change this to match your file path

# Step 2: Use glob to find all CSV files in that directory
csv_files = glob.glob(os.path.join(csv_directory, "*.csv"))
print(f"Found {len(csv_files)} CSV files in the directory")

# Step 3: Combine all CSV files into a single dataframe
all_headlines = []

for csv_file in csv_files:
    # Print which file we're processing
    print(f"Processing {os.path.basename(csv_file)}...")
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file)
        
        # Identify the columns containing original and rewritten headlines
        # This assumes your columns have certain keywords in their names
        # Adjust these patterns to match your actual column names
        original_col = "title"
        rewritten_col = "rewritten_title"
        
        for col in df.columns:
            col_lower = col.lower()
            if 'original' in col_lower or 'source' in col_lower or 'before' in col_lower:
                original_col = col
            elif 'rewritten' in col_lower or 'target' in col_lower or 'after' in col_lower:
                rewritten_col = col
        
        if original_col and rewritten_col:
            # Create a new dataframe with standardized column names
            headlines_df = pd.DataFrame({
                'original': df[original_col],
                'rewritten': df[rewritten_col]
            })
            
            all_headlines.append(headlines_df)
            print(f"Added {len(headlines_df)} headline pairs from {os.path.basename(csv_file)}")
        else:
            print(f"Warning: Could not identify headline columns in {os.path.basename(csv_file)}")
            print(f"Available columns: {df.columns.tolist()}")
    
    except Exception as e:
        print(f"Error processing {os.path.basename(csv_file)}: {e}")

# Step 4: Combine all dataframes into one
if all_headlines:
    combined_df = pd.concat(all_headlines, ignore_index=True)
    print(f"Total headline pairs: {len(combined_df)}")
    
    # Step 5: Clean the data
    # Remove any rows with missing values
    combined_df = combined_df.dropna(subset=['original', 'rewritten'])
    print(f"Headline pairs after removing missing values: {len(combined_df)}")
    
    # Step 6: Convert to the required JSON format
    headline_pairs = {}
    for i, row in combined_df.iterrows():
        headline_id = f"headline{i+1}"
        headline_pairs[headline_id] = {
            "original": row["original"],
            "rewritten": row["rewritten"]
        }
    
    # Step 7: Save to JSON file
    output_file = "headline_pairs.json"
    with open(output_file, "w") as f:
        json.dump(headline_pairs, f, indent=2)
    
    print(f"Successfully saved {len(headline_pairs)} headline pairs to {output_file}")
    
    # Step 8: Verify a few examples
    print("\nSample headline pairs:")
    for i, (headline_id, pair) in enumerate(list(headline_pairs.items())[:3]):
        print(f"\n{headline_id}:")
        print(f"  Original: {pair['original']}")
        print(f"  Rewritten: {pair['rewritten']}")
else:
    print("No headline pairs were extracted from the CSV files.")