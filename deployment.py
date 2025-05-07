import os
import sys
import subprocess
import logging
import argparse
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('setup.log')
    ]
)

def check_python_version():
    """Check if Python version is compatible"""
    required_version = (3, 7)
    current_version = sys.version_info
    
    if current_version < required_version:
        logging.error(f"Python {required_version[0]}.{required_version[1]} or higher is required")
        return False
    
    logging.info(f"Python version check passed: {current_version[0]}.{current_version[1]}.{current_version[2]}")
    return True

def install_requirements():
    """Install required packages"""
    requirements = [
        "streamlit>=1.18.0",
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "faiss-cpu>=1.7.0",
        "sentence-transformers>=2.2.0",
        "openai>=0.27.0",
        "python-dotenv>=0.19.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "textstat>=0.7.0",
        "torch>=1.10.0",
        "transformers>=4.18.0",
        "nltk>=3.6.0",
        "tqdm>=4.62.0"
    ]
    
    logging.info("Installing required packages...")
    
    for package in requirements:
        try:
            logging.info(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to install {package}: {e}")
            return False
    
    logging.info("All packages installed successfully")
    return True

def setup_project_structure():
    """Create project directories and files"""
    directories = [
        "agentic_news_editor",
        "agentic_news_editor/processed_data",
        "agentic_news_editor/plots",
        "evaluation_results",
        "evaluation_results/plots"
    ]
    
    logging.info("Setting up project structure...")
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created directory: {directory}")
    
    # Create .env file template
    env_template = """# OpenAI API Key for headline generation
OPENAI_API_KEY=your_openai_api_key_here

# MIND Dataset paths (if using custom locations)
# MIND_NEWS_PATH=path/to/news.tsv
# MIND_BEHAVIORS_PATH=path/to/behaviors.tsv
"""
    
    with open(".env", "w") as f:
        f.write(env_template)
    
    logging.info("Created .env template file")
    
    return True

def download_example_data():
    """Download example MIND dataset files if needed"""
    data_dir = "agentic_news_editor/processed_data"
    
    # Check if data already exists
    if os.path.exists(os.path.join(data_dir, "news.tsv")) and \
       os.path.exists(os.path.join(data_dir, "behaviors.tsv")):
        logging.info("Example data files already exist")
        return True
    
    logging.info("Downloading example MIND dataset files...")
    
    try:
        # In a real implementation, you would add code to download the MIND dataset
        # This would involve using requests or urllib to download from the official source
        # For this example, we'll just create placeholder files
        
        # Create minimal placeholder data for demonstration
        with open(os.path.join(data_dir, "news.tsv"), "w") as f:
            f.write("N1\ttech\tAI\tBreakthrough in AI models announced\tResearchers have announced a major breakthrough in artificial intelligence models.\thttps://example.com/ai-news\t[]\t[]\n")
            f.write("N2\tbusiness\tfinance\tStock markets hit new record\tGlobal stock markets reached new records amid economic recovery hopes.\thttps://example.com/finance-news\t[]\t[]\n")
            f.write("N3\thealth\tcovid\tNew health guidelines released\tHealth authorities have released updated guidelines for public safety.\thttps://example.com/health-news\t[]\t[]\n")
        
        with open(os.path.join(data_dir, "behaviors.tsv"), "w") as f:
            f.write("I1\tU1\t11/11/2021 06:30:15\tN1 N2\tN1-1 N2-0 N3-1\n")
            f.write("I2\tU2\t11/11/2021 08:20:30\tN2 N3\tN1-0 N2-1 N3-0\n")
        
        logging.info("Created example data files in 'agentic_news_editor/processed_data'")
        logging.warning("Note: These are placeholder files. For real usage, download the MIND dataset.")
        
        return True
    except Exception as e:
        logging.error(f"Error downloading example data: {e}")
        return False

def download_stock_images():
    """Download stock images for the news editor frontend"""
    image_categories = [
        ("tech", 2),
        ("business", 2),
        ("politics", 2),
        ("climate", 2),
        ("health", 2)
    ]
    
    logging.info("Checking for stock images...")
    
    # Check if images already exist
    all_exist = True
    for category, count in image_categories:
        for i in range(1, count+1):
            if not os.path.exists(f"{category}{i}.jpg"):
                all_exist = False
                break
    
    if all_exist:
        logging.info("Stock images already exist")
        return True
    
    logging.info("Creating placeholder stock images...")
    
    try:
        # In a real implementation, you would download actual stock images
        # For this example, we'll create colored rectangles as placeholders
        
        # Check if PIL is installed
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            logging.info("Installing Pillow for image creation...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
            from PIL import Image, ImageDraw
        
        # Color map for different categories
        colors = {
            "tech": (0, 123, 255),     # Blue
            "business": (111, 66, 193), # Purple
            "politics": (220, 53, 69),  # Red
            "climate": (40, 167, 69),   # Green
            "health": (253, 126, 20)    # Orange
        }
        
        # Create placeholder images
        for category, count in image_categories:
            for i in range(1, count+1):
                img = Image.new('RGB', (800, 450), colors.get(category, (100, 100, 100)))
                draw = ImageDraw.Draw(img)
                draw.rectangle([(20, 20), (780, 430)], outline=(255, 255, 255), width=5)
                draw.text((400, 225), f"{category.upper()} {i}", fill=(255, 255, 255))
                img.save(f"{category}{i}.jpg")
        
        logging.info("Created placeholder stock images")
        
        return True
    except Exception as e:
        logging.error(f"Error creating stock images: {e}")
        return False

def setup_nltk_data():
    """Download required NLTK data"""
    try:
        import nltk
        nltk.download('punkt')
        logging.info("Downloaded NLTK data")
        return True
    except Exception as e:
        logging.error(f"Error downloading NLTK data: {e}")
        return False

def run_full_setup():
    """Run the complete setup process"""
    if not check_python_version():
        return False
    
    if not install_requirements():
        return False
    
    if not setup_project_structure():
        return False
    
    if not download_example_data():
        logging.warning("Example data setup failed, but continuing setup")
    
    if not download_stock_images():
        logging.warning("Stock image setup failed, but continuing setup")
    
    if not setup_nltk_data():
        logging.warning("NLTK data setup failed, but continuing setup")
    
    logging.info("""
    =================================================================
    Setup Complete! You can now run the Agentic AI News Editor system.
    
    Quick Start:
    1. Edit the .env file and add your OpenAI API key
    2. Run the data pipeline: python news_editor_controller.py --full
    3. Launch the web app: streamlit run app_frontpage.py
    
    For more information, see the README.md file.
    =================================================================
    """)
    
    return True

def print_setup_options():
    """Print available setup options"""
    print("""
    Agentic AI News Editor Setup Script
    
    Available Commands:
    --full           Run the complete setup process (recommended for first-time setup)
    --packages       Install required packages only
    --structure      Create project structure only
    --data           Download example data only
    --images         Download stock images only
    --nltk           Download NLTK data only
    --help           Display this help message
    
    Example: python setup.py --full
    """)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Agentic AI News Editor Setup Script')
    parser.add_argument('--full', action='store_true', help='Run the complete setup process')
    parser.add_argument('--packages', action='store_true', help='Install required packages only')
    parser.add_argument('--structure', action='store_true', help='Create project structure only')
    parser.add_argument('--data', action='store_true', help='Download example data only')
    parser.add_argument('--images', action='store_true', help='Download stock images only')
    parser.add_argument('--nltk', action='store_true', help='Download NLTK data only')
    
    args = parser.parse_args()
    
    if len(sys.argv) == 1 or '--help' in sys.argv:
        print_setup_options()
    elif args.full:
        run_full_setup()
    else:
        if args.packages:
            install_requirements()
        if args.structure:
            setup_project_structure()
        if args.data:
            download_example_data()
        if args.images:
            download_stock_images()
        if args.nltk:
            setup_nltk_data()