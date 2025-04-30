import os
import subprocess
import time
import sys

def run_command(command, description):
    """Run a shell command and print status"""
    print(f"\n{'='*50}")
    print(f"STEP: {description}")
    print(f"{'='*50}")
    print(f"Running: {command}")
    
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
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
    
    input("Press Enter to launch the Streamlit app...")
    
    # Launch Streamlit app
    if not run_command("streamlit run app_frontpage.py", "Streamlit App"):
        print("Failed to run the Streamlit app. Make sure streamlit is installed.")
        return
    
    # Wait a moment after the app closes
    time.sleep(1)
    
    # Check if the curated file exists
    if not os.path.exists("curated_full_daily_output.csv"):
        print("❌ No curated articles found! Please run the app and generate articles first.")
        return
    
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
    
    # Final step: Show results
    print("\n🎉 WORKFLOW COMPLETE!")
    print("\nResults Summary:")
    print("1. Curated articles saved to: curated_full_daily_output.csv")
    print("2. Headline pairs saved to: headline_pairs.json")
    print("3. Analysis results saved to: results/engagement_results.json")
    print("4. Visual reports saved to: results/figures/")
    
    # Check if results directory exists and has content
    if os.path.exists("results/figures"):
        figure_files = os.listdir("results/figures")
        if figure_files:
            print(f"\nGenerated {len(figure_files)} analysis charts:")
            for file in figure_files:
                print(f"  - results/figures/{file}")
    
    print("\nTo view the analysis charts, open the files in the results/figures directory.")
    print("You can also view the detailed engagement metrics in results/engagement_results.json")

if __name__ == "__main__":
    main()