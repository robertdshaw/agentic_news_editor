print("Test script starting")

def main():
    print("Inside main function")
    
    # Print current working directory and Python path
    import os
    import sys
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.executable}")
    
    # Try to import some of the libraries you need
    print("Testing imports...")
    try:
        import pandas as pd
        print("pandas imported successfully")
    except Exception as e:
        print(f"Error importing pandas: {e}")
    
    try:
        import xgboost
        print("xgboost imported successfully")
    except Exception as e:
        print(f"Error importing xgboost: {e}")
    
    try:
        from transformers import AutoTokenizer
        print("transformers imported successfully")
    except Exception as e:
        print(f"Error importing transformers: {e}")
    
    print("Test script completed")

if __name__ == "__main__":
    print("Calling main function")
    main()
    print("Main function completed")
    
    # At the bottom of your script
def main():
    print("Main function is being called")
    # Rest of your main function
    
print("Script is reaching the __name__ check")
if __name__ == "__main__":
    print("Script is running as main module")
    main()
    print("Script has completed execution")
else:
    print(f"Script is being imported as module: {__name__}")