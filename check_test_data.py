import pandas as pd
import os

# Check if the test behaviors file exists
test_behaviors_path = "test_data/behaviors.tsv"
if os.path.exists(test_behaviors_path):
    # Load a small sample of the file to inspect
    test_behaviors = pd.read_csv(test_behaviors_path, sep="\t", header=None, 
                                names=["impression_id", "user_id", "time", "history", "impressions"],
                                nrows=10)  # Just load first 10 rows
    
    print(f"Test behaviors file exists with shape: {test_behaviors.shape}")
    print("\nSample rows:")
    print(test_behaviors)
    
    # Check the impressions column specifically
    print("\nImpression column values:")
    for i, impression in enumerate(test_behaviors['impressions']):
        print(f"Row {i}: {impression}")
        
    # Check if the format is as expected - should contain news IDs with click info
    # Example format: "N1-0 N2-1 N3-0" where N1, N2, N3 are news IDs and 0/1 indicates click status
    if len(test_behaviors) > 0 and isinstance(test_behaviors['impressions'].iloc[0], str):
        sample = test_behaviors['impressions'].iloc[0]
        print(f"\nSample impression string: {sample}")
        parts = sample.split()
        print(f"Number of items in sample: {len(parts)}")
        if len(parts) > 0:
            print(f"Format of first item: {parts[0]}")
else:
    print(f"Test behaviors file not found at: {test_behaviors_path}")
    
    # Check if any test data directory exists
    if os.path.exists("test_data"):
        print("The test_data directory exists. Contents:")
        print(os.listdir("test_data"))
    else:
        print("No test_data directory found.")
        
        import pandas as pd
import os

# Check validation behaviors file
val_behaviors_path = "val_data/behaviors.tsv"
if os.path.exists(val_behaviors_path):
    val_behaviors = pd.read_csv(val_behaviors_path, sep="\t", header=None, 
                              names=["impression_id", "user_id", "time", "history", "impressions"],
                              nrows=5)
    print("Validation behaviors sample:")
    print(val_behaviors)
    
    # Check first impression format
    if not val_behaviors.empty and isinstance(val_behaviors['impressions'].iloc[0], str):
        print("\nFirst impression string:")
        print(val_behaviors['impressions'].iloc[0])
        
        # Check if format includes clicks or not
        sample_impressions = val_behaviors['impressions'].iloc[0].split()
        if sample_impressions and '-' in sample_impressions[0]:
            print("Format includes click information")
        else:
            print("Format does NOT include click information")
else:
    print("Validation behaviors file not found")