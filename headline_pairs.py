import pandas as pd
import json
import os

def extract_headline_pairs(csv_file_path="curated_full_daily_output.csv", output_json="headline_pairs.json"):
    """
    Extract headline pairs from the curated articles CSV and save to JSON.
    
    Parameters:
        csv_file_path (str): Path to the CSV file with curated articles
        output_json (str): Path to save the JSON output
    
    Returns:
        dict: The headline pairs dictionary
    """
    print(f"Looking for curated articles in: {csv_file_path}")
    
    # Check if the CSV file exists
    if not os.path.exists(csv_file_path):
        print(f"Error: {csv_file_path} not found!")
        # Return existing headline pairs if available, otherwise empty dict
        if os.path.exists(output_json):
            try:
                with open(output_json, 'r') as f:
                    existing_pairs = json.load(f)
                    print(f"Using existing {len(existing_pairs)} headline pairs from {output_json}")
                    return existing_pairs
            except Exception as e:
                print(f"Error reading existing pairs: {e}")
        return {}
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        print(f"Found {len(df)} articles in the CSV file")
        
        # Check if the required columns exist
        required_columns = ['title', 'rewritten_title']
        if not all(col in df.columns for col in required_columns):
            print(f"Error: CSV must contain columns {required_columns}")
            print(f"Available columns: {df.columns.tolist()}")
            return {}
        
        # Extract headline pairs
        headline_pairs = {}
        for i, row in df.iterrows():
            if pd.notna(row['title']) and pd.notna(row['rewritten_title']):
                headline_id = f"headline{i+1}"
                headline_pairs[headline_id] = {
                    "original": row["title"],
                    "rewritten": row["rewritten_title"]
                }
        
        print(f"Extracted {len(headline_pairs)} valid headline pairs")
        
        # Load existing pairs if the file exists
        if os.path.exists(output_json):
            try:
                with open(output_json, 'r') as f:
                    existing_pairs = json.load(f)
                # Combine with existing pairs (only add new ones that don't already exist)
                start_count = len(existing_pairs)
                next_id = start_count + 1
                
                # Check if any new pairs are unique compared to existing ones
                existing_originals = {pair['original'] for pair in existing_pairs.values()}
                
                for pair in headline_pairs.values():
                    if pair['original'] not in existing_originals:
                        headline_id = f"headline{next_id}"
                        existing_pairs[headline_id] = pair
                        next_id += 1
                
                headline_pairs = existing_pairs
                print(f"Added {len(headline_pairs) - start_count} new headline pairs to existing {start_count} pairs")
            except Exception as e:
                print(f"Error processing existing pairs: {e}")
        
        # Save to JSON
        with open(output_json, "w") as f:
            json.dump(headline_pairs, f, indent=2)
        
        print(f"Successfully saved {len(headline_pairs)} headline pairs to {output_json}")
        
        # Print a few examples
        print("\nSample headline pairs:")
        for i, (headline_id, pair) in enumerate(list(headline_pairs.items())[:3]):
            print(f"\n{headline_id}:")
            print(f"  Original: {pair['original']}")
            print(f"  Rewritten: {pair['rewritten']}")
        
        return headline_pairs
    
    except Exception as e:
        print(f"Error processing {csv_file_path}: {e}")
        return {}

# Export the headline pairs for use by headline_research.py
headline_pairs = extract_headline_pairs()

# If this script is run directly (not imported)
if __name__ == "__main__":
    # You can specify custom paths when running directly
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract headline pairs from curated articles')
    parser.add_argument('--input', default="curated_full_daily_output.csv", 
                        help='Path to the CSV file with curated articles')
    parser.add_argument('--output', default="headline_pairs.json",
                        help='Path to save the JSON output')
    
    args = parser.parse_args()
    extract_headline_pairs(args.input, args.output)