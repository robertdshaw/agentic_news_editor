import os
import subprocess
import time
import sys

def run_command(command, description):
    """Run a shell command and print status with detailed output"""
    print(f"\n{'='*50}")
    print(f"STEP: {description}")
    print(f"{'='*50}")
    print(f"Running: {command}")
    
    try:
        # Capture and display output in real-time
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Print output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"    {output.strip()}")
        
        return_code = process.poll()
        if return_code == 0:
            print(f"✅ {description} completed successfully!")
            return True
        else:
            print(f"❌ {description} failed with return code {return_code}")
            return False
    except Exception as e:
        print(f"❌ Error during {description}: {e}")
        return False

def main():
    """Run the complete news editor workflow"""
    print("\n📰 AGENTIC AI NEWS EDITOR - COMPLETE WORKFLOW")
    print("This script will run the full news curation and analysis pipeline.\n")
    
    # Step 1: Run the Streamlit app
    print("STEP 1: Launch the Streamlit app to curate articles")
    print("NOTE: You will need to interact with the app to select topics and generate articles.")
    print("      When you're done, close the app to continue with the analysis.")
    print("IMPORTANT: Make sure you click 'CURATE FRESH ARTICLES' in the app and wait for")
    print("           the curation to complete before closing the app.")
    
    input("Press Enter to launch the Streamlit app...")
    
    # Check if curated file exists before running the app (to compare later)
    curated_file_exists_before = os.path.exists("curated_full_daily_output.csv")
    curated_file_time_before = None
    if curated_file_exists_before:
        curated_file_time_before = os.path.getmtime("curated_full_daily_output.csv")
        print(f"Note: curated_full_daily_output.csv already exists (last modified: {time.ctime(curated_file_time_before)})")
    
    # Launch Streamlit app
    if not run_command("streamlit run app_frontpage.py", "Streamlit App"):
        print("Failed to run the Streamlit app. Make sure streamlit is installed.")
        return
    
    # Wait a moment after the app closes
    time.sleep(1)
    
    # Check if the curated file exists and has been updated
    if not os.path.exists("curated_full_daily_output.csv"):
        print("❌ No curated articles found! Please run the app and generate articles first.")
        return
    
    # Check if file was updated during this session
    if curated_file_exists_before:
        curated_file_time_after = os.path.getmtime("curated_full_daily_output.csv")
        if curated_file_time_after <= curated_file_time_before:
            print("⚠️ Warning: The curated articles file wasn't updated during this session.")
            print("   Did you click 'CURATE FRESH ARTICLES' in the app?")
            proceed = input("Do you want to continue with the existing file? (y/n): ")
            if proceed.lower() != 'y':
                print("Aborting workflow.")
                return
    
    # Show file details for debugging
    file_size = os.path.getsize("curated_full_daily_output.csv")
    print(f"Found curated_full_daily_output.csv (Size: {file_size} bytes)")
    
    # Display first few lines of the CSV for verification
    try:
        with open("curated_full_daily_output.csv", 'r', encoding='utf-8') as f:
            print("\nCSV File Contents (first 5 lines):")
            for i, line in enumerate(f):
                if i >= 5:
                    break
                print(f"  {line.strip()}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
    
    # Step 2: Extract headline pairs
    print("\nSTEP 2: Extracting headline pairs from curated articles...")
    if not run_command("python headline_pairs.py", "Headline Pairs Extraction"):
        print("Failed to extract headline pairs.")
        return
    
    # Step 3: Run headline analysis
    print("\nSTEP 3: Analyzing headline effectiveness...")
    if not run_command("python headline_research.py", "Headline Research Analysis"):
        print("Failed to analyze headlines.")
        return
    
    # Final step: Verify and show results
    print("\n🎉 WORKFLOW COMPLETE!")
    print("\nResults Verification:")
    
    # Check all output files exist
    print("\nChecking output files:")
    files_to_check = {
        "curated_full_daily_output.csv": "Curated articles",
        "headline_pairs.json": "Headline pairs",
        "results/engagement_results.json": "Analysis results",
    }
    
    all_files_exist = True
    for file_path, description in files_to_check.items():
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            last_modified = time.ctime(os.path.getmtime(file_path))
            print(f"✅ {description} saved to: {file_path} (Size: {file_size} bytes, Modified: {last_modified})")
        else:
            print(f"❌ {description} file not found: {file_path}")
            all_files_exist = False
    
    # Check if results directory exists and has content
    print("\nChecking generated figures:")
    if os.path.exists("results/figures"):
        figure_files = os.listdir("results/figures")
        if figure_files:
            print(f"✅ Generated {len(figure_files)} analysis charts:")
            for file in figure_files:
                file_path = f"results/figures/{file}"
                file_size = os.path.getsize(file_path)
                last_modified = time.ctime(os.path.getmtime(file_path))
                print(f"  - {file} (Size: {file_size} bytes, Modified: {last_modified})")
        else:
            print("❌ No analysis charts found in results/figures/")
            all_files_exist = False
    else:
        print("❌ Results directory not found: results/figures/")
        all_files_exist = False
    
    if all_files_exist:
        print("\n✅ SUCCESS: All expected output files were generated!")
        print("\nYou can view:")
        print("- The analysis charts in the results/figures directory")
        print("- The detailed engagement metrics in results/engagement_results.json")
    else:
        print("\n⚠️ WARNING: Some expected output files are missing.")
        print("Try running the individual scripts manually to debug:")
        print("1. python headline_pairs.py")
        print("2. python headline_research.py")

if __name__ == "__main__":
    main()