import os
# os.environ["STREAMLIT_WATCH_FORCE_POLLING"] = "true"
import disable_torch_watch
import streamlit as st
import json
import pandas as pd
import numpy as np
# Import FAISS conditionally to avoid conflicts
try:
    import faiss
except ImportError:
    st.error("FAISS not installed. Please install with 'pip install faiss-cpu'")

# Handle SentenceTransformer with special care
try:
    # Import in a way that avoids torch event loop conflicts
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable parallelism to avoid deadlocks
    from sentence_transformers import SentenceTransformer
except ImportError:
    st.error("SentenceTransformer not installed. Please install with 'pip install sentence-transformers'")

from openai import OpenAI
from dotenv import load_dotenv
import datetime
import random
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(
    page_title="Agentic AI News Editor",
    page_icon="📰",
    layout="wide",  # Use wide layout by default
    initial_sidebar_state="expanded"
)

def apply_custom_css():
    st.markdown("""
    <style>
    /* Reduce padding to use more screen space */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Make sure content uses available width */
    .stApp {
        max-width: 100%;
    }
    
    /* Tighten up spacing between elements */
    div.row-widget.stRadio > div {
        flex-direction: row;
        align-items: center;
    }
    
    /* Adjust spacing for headings */
    h1, h2, h3 {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Make cards more compact */
    div.stCard {
        padding: 0.5rem;
    }
    
    /* Fix for markdown spacing */
    .element-container .stMarkdown {
        margin-bottom: 0.5rem;
    }
    
    /* Style for original headlines */
    .original-headline {
        font-style: italic;
        color: #777;
        margin-bottom: 5px;
        padding: 4px 8px;
        background-color: #f8f9fa;
        border-left: 3px solid #ccc;
        font-size: 0.9em;
    }
    
    /* Style for rewritten headlines */
    .rewritten-headline {
        font-weight: bold;
        color: #000;
        margin-top: 0;
    }
    
    /* Title comparison box */
    .title-comparison {
        margin-bottom: 15px;
        padding: 8px;
        border-radius: 4px;
        background-color: #f5f5f5;
    }
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# --- Functions ---

def show_debug_info():
    with st.expander("Debug Info"):
        st.write("Screen Resolution", st.session_state.get('screen_resolution', 'Unknown'))
        st.write("Browser Info", st.session_state.get('browser_info', 'Unknown'))
        
        st.markdown("""
        <script>
        window.parent.postMessage({
            type: "streamlit:setSessionState",
            data: {
                screen_resolution: {
                    width: window.screen.width,
                    height: window.screen.height
                },
                browser_info: navigator.userAgent
            }
        }, "*");
        </script>
        """, unsafe_allow_html=True)

def load_sentence_transformer():
    """Load the sentence transformer model separately to avoid conflicts"""
    try:
        return SentenceTransformer('paraphrase-MiniLM-L6-v2')
    except Exception as e:
        logging.error(f"Error loading SentenceTransformer: {e}")
        return None

@st.cache_resource
def load_models_and_data():
    """Load FAISS index, metadata, and embedding model with caching"""
    try:
        logging.info("Loading FAISS index and models")
        
        # Load the CSV data first
        if not os.path.exists("articles_with_embeddings.csv"):
            st.error("⚠️ articles_with_embeddings.csv file not found!")
            return None, None, None
            
        articles_df = pd.read_csv("articles_with_embeddings.csv")
        
        # Load FAISS index
        if not os.path.exists("articles_faiss.index"):
            st.error("⚠️ articles_faiss.index file not found!")
            return None, None, None
            
        index = faiss.read_index("articles_faiss.index")
        
        # Load model with the separate function
        model = load_sentence_transformer()
        if model is None:
            return None, None, None
            
        logging.info(f"Loaded {len(articles_df)} articles and model")
        return index, articles_df, model
    except Exception as e:
        logging.error(f"Error loading models and data: {e}")
        return None, None, None

def load_topics(filename="memory_topics.json"):
    """Load yesterday's topics from memory file"""
    try:
        if os.path.exists(filename):
            with open(filename, "r") as f:
                return json.load(f)
        return []
    except Exception as e:
        logging.error(f"Error loading topics: {e}")
        return []

def save_topics(topics, filename="memory_topics.json"):
    """Save today's topics to memory file"""
    try:
        with open(filename, "w") as f:
            json.dump(topics, f)
        logging.info(f"Saved {len(topics)} topics to memory")
    except Exception as e:
        logging.error(f"Error saving topics: {e}")

def get_openai_client():
    """Initialize OpenAI client with API key"""
    try:
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not openai_api_key:
            st.sidebar.error("❌ OPENAI_API_KEY is missing. Check your .env file!")
            return None
        
        return OpenAI(api_key=openai_api_key)
    except Exception as e:
        logging.error(f"Error initializing OpenAI client: {e}")
        st.sidebar.error(f"Error initializing OpenAI: {e}")
        return None

def rewrite_headline(client, title, abstract):
    """Rewrite a single headline using OpenAI"""
    if client is None:
        return title
    
    prompt = f"""You are an expert news editor.
    
    Your task is to rewrite the following news headline to be more engaging, SEO-optimized, and still factually accurate based on the article abstract.
    
    Use clear, active language and keep it under 15 words.
    
    ---
    
    Title: {title}
    
    Abstract: {abstract}
    
    Rewritten Headline:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional news editor."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=50,
        )
        rewritten = response.choices[0].message.content.strip()
        if len(rewritten) < 5 or rewritten.lower() in ["", "the new york times"]:
            rewritten = title
        return rewritten
    except Exception as e:
        logging.error(f"Error rewriting headline: {e}")
        return title

def generate_explanation(client, title, abstract):
    """Generate explanation for why the article is important"""
    if client is None:
        return "This article provides important information for our readers."
    
    prompt = f"""You are an editorial assistant.
    
    Write one sentence explaining why the following news article is important to readers.
    
    Focus on clarity and importance for a general audience.
    
    ---
    
    Title: {title}
    
    Abstract: {abstract}
    
    Explanation:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional editorial assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=60,
        )
        explanation = response.choices[0].message.content.strip()
        return explanation
    except Exception as e:
        logging.error(f"Error generating explanation: {e}")
        return "This article provides important information for our readers."

def curate_articles_for_topic(query_text, index, articles_df, model, openai_client, k=5, progress_bar=None):
    """Find and enhance articles for a given topic"""
    try:
        # Create query embedding and search - wrap in try/except
        try:
            query_embedding = model.encode([query_text])
            D, I = index.search(np.array(query_embedding), k=k)
        except Exception as e:
            logging.error(f"Error creating embeddings: {e}")
            st.error(f"Error searching for articles: {str(e)}")
            # Return a sample of random articles as fallback
            random_indices = np.random.choice(len(articles_df), min(k, len(articles_df)), replace=False)
            topic_articles = articles_df.iloc[random_indices].copy()
            return topic_articles
        
        # Extract articles
        topic_articles = articles_df.iloc[I[0]].copy() if len(I) > 0 and len(I[0]) > 0 else pd.DataFrame()
        if len(topic_articles) == 0:
            logging.warning(f"No articles found for query: {query_text}")
            st.warning(f"No articles found for {query_text}. Using random selection instead.")
            # Fallback to random articles
            random_indices = np.random.choice(len(articles_df), min(k, len(articles_df)), replace=False)
            topic_articles = articles_df.iloc[random_indices].copy()
        
        # Store original title explicitly
        topic_articles["original_title"] = topic_articles["title"].copy()
        
        # Process each article individually
        total = len(topic_articles)
        
        # Safety check
        if total == 0:
            return pd.DataFrame()
        
        for i, (idx, row) in enumerate(topic_articles.iterrows()):
            # Update progress
            if progress_bar is not None:
                progress_value = (i + 1) / total
                progress_bar.progress(progress_value, text=f"Processing article {i + 1}/{total}")
            
            # Process with OpenAI
            try:
                topic_articles.at[idx, 'rewritten_title'] = rewrite_headline(
                    openai_client, row['title'], row['abstract']
                )
                topic_articles.at[idx, 'explanation'] = generate_explanation(
                    openai_client, row['title'], row['abstract']
                )
            except Exception as e:
                logging.error(f"Error processing article {idx}: {e}")
                # Use original title as fallback
                topic_articles.at[idx, 'rewritten_title'] = row['title']
                topic_articles.at[idx, 'explanation'] = "This article contains important information relevant to the topic."
        
        if progress_bar is not None:
            progress_bar.progress(1.0, text="Processing complete!")
            time.sleep(0.5)  # Give user time to see the completed progress
        
        return topic_articles
    except Exception as e:
        logging.error(f"Error curating articles: {e}")
        st.error(f"Error while curating articles: {str(e)}")
        return pd.DataFrame()

def get_stock_image_path(topic, article_id=None):
    """
    Return the path to a stock image for the given topic.
    Uses article_id to consistently select the same image for the same article.
    """
    # Map topics to their corresponding image prefixes
    topic_to_prefix = {
        "Top Technology News": "tech",
        "Inspiring Stories": "inspire",
        "Global Politics": "educate",
        "Climate and Environment": "climate",
        "Health and Wellness": "health",
        # "Education": "educate",
        # Add more topic mappings as needed
    }
    
    # Get the prefix for this topic, or use a default
    prefix = topic_to_prefix.get(topic, "tech")
    
    # Determine which image to use (1 or 2)
    # If article_id is provided, use it to consistently select the same image
    if article_id is not None:
        # Use the article_id to get a consistent image number
        img_num = 1 + (hash(str(article_id)) % 2)  # Either 1 or 2
    else:
        # Random selection between 1 and 2
        img_num = random.randint(1, 2)
    
    # Build the full image path
    image_path = f"{prefix}{img_num}.jpg"
    
    return image_path

def display_article_image(topic, article_id=None, is_main=False):
    """Display a stock image for an article with proper error handling"""
    try:
        # Get the appropriate image path
        image_path = get_stock_image_path(topic, article_id)
        
        # Adjust width based on whether this is a main or secondary article
        width = 700 if is_main else 400
        
        # Check if the file exists before trying to display it
        if os.path.exists(image_path):
            st.image(image_path, width=width, use_container_width=True)
            return True
        else:
            # Log that the file was not found
            logging.warning(f"Image file not found: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")
    except Exception as e:
        # Log the error
        logging.error(f"Error displaying stock image for {topic}: {e}")
        
        # Fallback to a colored box with topic name
        fallback_color = {
            "Top Technology News": "#007BFF",
            "Inspiring Stories": "#6F42C1",
            "Global Politics": "#DC3545",
            "Climate and Environment": "#28A745",
            "Health and Wellness": "#FD7E14",
            "Education": "#17A2B8"
        }.get(topic, "#6C757D")
        
        height = 350 if is_main else 200
        
        # Create a colored box with the topic name
        st.markdown(f"""
        <div style="
            height: {height}px; 
            background-color: {fallback_color}25; 
            border: 1px solid {fallback_color}; 
            border-radius: 5px; 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            margin-bottom: 15px;
            color: {fallback_color};
            font-weight: bold;
            text-align: center;
        ">
            {topic}
        </div>
        """, unsafe_allow_html=True)
        return False

def display_headline_comparison(original_title, rewritten_title):
    """Display original and rewritten titles with clear styling"""
    st.markdown(f"""
    <div class="title-comparison">
        <div class="original-headline">Original: {original_title}</div>
        <div class="rewritten-headline">{rewritten_title}</div>
    </div>
    """, unsafe_allow_html=True)

def load_curated_articles():
    """Load previously curated articles if available"""
    try:
        if os.path.exists("curated_full_daily_output.csv"):
            return pd.read_csv("curated_full_daily_output.csv")
        return None
    except Exception as e:
        logging.error(f"Error loading curated articles: {e}")
        return None

# --- Main Application Logic ---
def main():
    # Setup sidebar with controls
    with st.sidebar:
        st.header("📰 NEWSPAPER CONTROLS")
        st.write("Configure your daily newspaper")
        st.markdown("---")
        
        # Editorial queries
        editorial_queries = {
            "Top Technology News": "latest breakthroughs in technology and innovation",
            "Inspiring Stories": "positive and uplifting news stories",
            "Global Politics": "latest news about world politics and diplomacy",
            "Climate and Environment": "climate change news and environment protection",
            "Health and Wellness": "advances in healthcare and medical discoveries"
        }
        
        # Topic selection
        selected_topics = st.multiselect(
            "Select topics to curate", 
            list(editorial_queries.keys()),
            default=list(editorial_queries.keys())[:2]  # Default first two topics
        )
        
        # Article count per topic
        articles_per_topic = st.slider("Articles per topic", 1, 10, 3, 1)
        
        # Display settings
        st.subheader("Display Settings")
        show_headline_comparison = st.toggle("Show headline comparison", value=True, 
                                          help="Display both original and AI-rewritten headlines")
        
        # Action buttons
        curate_button = st.button("CURATE FRESH ARTICLES", use_container_width=True)
        load_button = st.button("LOAD SAVED ARTICLES", use_container_width=True)
        
        # Handle curation process
        if curate_button:
            if not selected_topics:
                st.error("Please select at least one topic to curate")
            else:
                # Initialize session state for progress tracking if not present
                if 'curation_started' not in st.session_state:
                    st.session_state.curation_started = True
                    st.session_state.curation_complete = False
                
                # Create progress bar
                progress_bar = st.progress(0, text="Starting curation process...")
                
                # Load models and data
                progress_bar.progress(0.1, text="Loading models and data...")
                index, articles_df, model = load_models_and_data()
                openai_client = get_openai_client()
                
                if index is None or articles_df is None or model is None:
                    st.error("Failed to load required models and data")
                    st.session_state.curation_started = False
                else:
                    # Curate articles for each selected topic
                    all_curated_articles = []
                    
                    for i, topic in enumerate(selected_topics):
                        topic_progress = (i / len(selected_topics)) * 0.8 + 0.1  # Scale from 10% to 90%
                        progress_bar.progress(topic_progress, text=f"Curating articles for {topic}...")
                        
                        query_text = editorial_queries[topic]
                        topic_articles = curate_articles_for_topic(
                            query_text, index, articles_df, model, openai_client, 
                            k=articles_per_topic,
                            progress_bar=None  # We'll manage progress at the topic level
                        )
                        topic_articles["topic"] = topic
                        all_curated_articles.append(topic_articles)
                    
                    # Combine all curated articles
                    if all_curated_articles:
                        progress_bar.progress(0.9, text="Saving curated articles...")
                        full_curated_df = pd.concat(all_curated_articles, ignore_index=True)
                        full_curated_df.to_csv("curated_full_daily_output.csv", index=False)
                        
                        # Update memory for future reference
                        save_topics(selected_topics)
                        
                        # Update session state
                        st.session_state.loaded_articles = full_curated_df
                        st.session_state.curation_complete = True
                        
                        progress_bar.progress(1.0, text="Curation complete!")
                        st.success("✅ Articles curated successfully!")
                    else:
                        st.warning("No articles were curated")
                        st.session_state.curation_started = False
        
        # Handle loading previous articles
        if load_button:
            with st.spinner("Loading previously curated articles..."):
                st.session_state.loaded_articles = load_curated_articles()
                if st.session_state.loaded_articles is not None:
                    st.success(f"Loaded {len(st.session_state.loaded_articles)} articles")
                else:
                    st.error("No previously curated articles found")
        
        # Store headline comparison preference in session state
        st.session_state.show_headline_comparison = show_headline_comparison
        
        # Image path check
        st.markdown("---")
        st.subheader("Image Path Check")
        st.write("This tool checks if your stock images are accessible.")
        
        # Check a few sample paths
        sample_paths = ["tech1.jpg", "tech2.jpg", "inspire1.jpg", "inspire2.jpg", "health1.jpg", 
                        "health2.jpg", "educate1.jpg", "educate2.jpg", "climate1.jpg", 
                        "climate2.jpg"]
        for path in sample_paths:
            if os.path.exists(path):
                st.success(f"✓ {path} found")
            else:
                st.error(f"✗ {path} not found - make sure it's in the correct directory")
    
    # --- Header and Navigation ---
    # Header section
    st.markdown('<div class="header">', unsafe_allow_html=True)
    st.markdown('<h1 class="newspaper-title">THE DAILY AGENT</h1>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation menu using Streamlit columns for reliability
    cols = st.columns(7)
    nav_items = ["POLITICS", "ENTERTAINMENT", "TECH", "SPORTS", "OPINION", "HEALTH", "WORLD"]
    
    for i, item in enumerate(nav_items):
        with cols[i]:
            st.markdown(f'<div class="nav-item">{item}</div>', unsafe_allow_html=True)
    
    # Get headline comparison preference
    show_comparison = st.session_state.get('show_headline_comparison', True)
    
    # Check if we have articles to display
    if 'loaded_articles' in st.session_state and st.session_state.loaded_articles is not None and len(st.session_state.loaded_articles) > 0:
        articles_df = st.session_state.loaded_articles
        
        # Featured Article Section
        st.markdown('<h2 class="section-title">FEATURED NEWS</h2>', unsafe_allow_html=True)
        
        # Main article layout
        if len(articles_df) > 0:
            main_article = articles_df.iloc[0]
            
            # Using Streamlit components directly for reliability
            with st.container():
                st.markdown('<div class="article-box">', unsafe_allow_html=True)
                
                # Article tag
                st.markdown(f'<div class="article-tag">{main_article["topic"]}</div>', unsafe_allow_html=True)
                
                # Title comparison - either use the new comparison function or the old way
                if show_comparison:
                    display_headline_comparison(main_article['original_title'], main_article['rewritten_title'])
                else:
                    st.subheader(main_article['rewritten_title'])
                
                # Author byline
                author = random.choice(["Sarah Chen", "Michael Johnson", "Priya Patel", "Robert Williams"])
                st.markdown(f'<div class="article-byline">By {author} | {datetime.datetime.now().strftime("%B %d, %Y")}</div>', unsafe_allow_html=True)
                
                # Image - use article's index as a consistent ID for image selection
                display_article_image(main_article["topic"], article_id=main_article.name, is_main=True)
                
                # Abstract
                abstract = main_article['abstract']
                if abstract and len(abstract) > 300:
                    abstract = abstract[:300] + "..."
                st.write(abstract)
                
                # Why it matters
                st.markdown(f'<div class="article-why-matters"><strong>Why it matters:</strong> {main_article["explanation"]}</div>', unsafe_allow_html=True)
                
                # Read more link
                st.markdown("**Continue Reading →**")
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Trending Articles Section
        st.markdown('<h2 class="section-title">TRENDING NOW</h2>', unsafe_allow_html=True)
        
        # Display trending articles (next 5 after main)
        trend_cols = st.columns(3)
        
        for i in range(1, min(7, len(articles_df))):
            if i < len(articles_df):
                # Determine which column to place the article in
                col_idx = (i - 1) % 3
                
                with trend_cols[col_idx]:
                    article = articles_df.iloc[i]
                    
                    st.markdown('<div class="article-box">', unsafe_allow_html=True)
                    
                    # Article tag
                    st.markdown(f'<div class="article-tag">{article["topic"]}</div>', unsafe_allow_html=True)
                    
                    # Title comparison 
                    if show_comparison:
                        display_headline_comparison(article['original_title'], article['rewritten_title'])
                    else:
                        st.markdown(f"### {article['rewritten_title']}")
                    
                    # Image - use article's index as a consistent ID for image selection
                    display_article_image(article["topic"], article_id=article.name)
                    
                    # Short abstract
                    abstract = article['abstract']
                    if abstract and len(abstract) > 150:
                        abstract = abstract[:150] + "..."
                    st.write(abstract)
                    
                    # Read more link
                    st.markdown("**Read More →**")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Topic Sections
        remaining_articles = articles_df.iloc[7:] if len(articles_df) > 7 else pd.DataFrame()
        
        if len(remaining_articles) > 0:
            # Group by topic
            topics = remaining_articles['topic'].unique()
            
            for topic in topics:
                st.markdown(f'<h2 class="section-title">{topic.upper()}</h2>', unsafe_allow_html=True)
                
                topic_articles = remaining_articles[remaining_articles['topic'] == topic]
                
                # Create a grid of articles
                cols = st.columns(2)
                
                for i, (_, article) in enumerate(topic_articles.iterrows()):
                    with cols[i % 2]:
                        st.markdown('<div class="article-box">', unsafe_allow_html=True)
                        
                        # Title comparison for topic sections (was missing in original)
                        if show_comparison:
                            display_headline_comparison(article['original_title'], article['rewritten_title'])
                        else:
                            st.markdown(f"### {article['rewritten_title']}")
                        
                        # Image - use article's index as a consistent ID for image selection
                        display_article_image(article["topic"], article_id=article.name)
                        
                        # Short abstract
                        abstract = article['abstract']
                        if abstract and len(abstract) > 100:
                            abstract = abstract[:100] + "..."
                        st.write(abstract)
                        
                        # Read more link
                        st.markdown("**Read More →**")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
        
        # Footer
        st.markdown('<div class="footer">', unsafe_allow_html=True)
        st.markdown('<h3>THE DAILY AGENT</h3>', unsafe_allow_html=True)
        st.markdown(f'<p>© {datetime.datetime.now().year} The Daily Agent. All Rights Reserved.</p>', unsafe_allow_html=True)
        st.markdown('<p>Powered by AI-curated content</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        # Welcome page when no articles are loaded
        if 'curation_started' in st.session_state and st.session_state.curation_started and not st.session_state.get('curation_complete', False):
            # Show progress message
            st.info("⏳ Curation in progress... Please wait while we prepare your personalized news articles.")
        else:
            # Welcome message with clear instructions
            st.header("Welcome to The Daily Agent")
            st.write("Your AI-powered personalized news platform")
            
            # Step-by-step instructions
            st.subheader("Get Started in Two Simple Steps:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 1. Select Topics")
                st.info("Choose which news categories you want to include using the sidebar controls")
                try:
                    # Try to show a sample tech image if available
                    if os.path.exists("tech1.jpg"):
                        st.image("tech1.jpg", width=300)
                except:
                    pass
                
            with col2:
                st.markdown("### 2. Curate Articles")
                st.success("Click 'CURATE FRESH ARTICLES' to generate your personalized newspaper")
                try:
                    # Try to show a sample inspire image if available
                    if os.path.exists("inspire1.jpg"):
                        st.image("inspire1.jpg", width=300)
                except:
                    pass
            
            # Preview
            st.subheader("YOUR PERSONALIZED NEWSPAPER")
            st.write("The Daily Agent will create a professional newspaper using your selected topics and AI-curated content.")
            
            # Try to display a sample layout image or message
            st.markdown("""
            <div style="border: 1px solid #ddd; padding: 20px; border-radius: 5px; text-align: center; margin: 20px 0; background-color: #f9f9f9;">
                <h3 style="margin-bottom: 15px;">Sample Layout Preview</h3>
                <p>Your newspaper will include featured articles, trending stories, and topic-specific sections with images.</p>
            </div>
            """, unsafe_allow_html=True)