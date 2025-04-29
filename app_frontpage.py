import streamlit as st
import os
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

# --- Page configuration ---
st.set_page_config(
    page_title="The Daily Chronicle",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Daily Wire-inspired CSS ---
st.markdown("""
<style>
    /* Global reset and basic styling */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }
    
    /* Main content container */
    .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    /* Customize sidebar */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e9ecef;
        padding-top: 0 !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0 !important;
        padding-top: 0 !important;
    }
    
    /* Header styling */
    .site-header {
        background-color: #fff;
        border-bottom: 3px solid #e9ecef;
        padding: 10px 20px;
        text-align: center;
    }
    
    .newspaper-title {
        font-family: 'Georgia', serif;
        font-size: 2.5rem;
        font-weight: 900;
        color: #222;
        letter-spacing: -1px;
        margin: 10px 0 5px;
        text-transform: uppercase;
    }
    
    .nav-menu {
        background-color: #f8f9fa;
        border-bottom: 1px solid #e9ecef;
        padding: 10px 0;
        margin-bottom: 20px;
        display: flex;
        justify-content: center;
    }
    
    .nav-item {
        display: inline-block;
        padding: 5px 15px;
        margin: 0 5px;
        font-weight: 600;
        color: #333;
        text-transform: uppercase;
        font-size: 0.85rem;
    }
    
    .nav-item.active {
        color: #0d6efd;
    }
    
    /* Article styling */
    .main-container {
        padding: 0 20px;
    }
    
    .section-title {
        font-family: 'Georgia', serif;
        font-size: 1.5rem;
        font-weight: 700;
        border-bottom: 2px solid #c00;
        padding-bottom: 5px;
        margin: 20px 0 15px;
        color: #222;
    }
    
    .article-card {
        background-color: #fff;
        border: 1px solid #e9ecef;
        border-radius: 4px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .article-card:hover {
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .article-tag {
        display: inline-block;
        background-color: #0d6efd;
        color: white;
        padding: 3px 8px;
        font-size: 0.7rem;
        border-radius: 3px;
        margin-bottom: 10px;
        text-transform: uppercase;
        font-weight: 600;
    }
    
    .article-title {
        font-family: 'Georgia', serif;
        font-size: 1.4rem;
        line-height: 1.3;
        margin-bottom: 10px;
        color: #222;
        font-weight: 700;
    }
    
    .main-article-title {
        font-family: 'Georgia', serif;
        font-size: 2rem;
        line-height: 1.2;
        margin-bottom: 15px;
        color: #222;
        font-weight: 700;
    }
    
    .article-byline {
        font-size: 0.85rem;
        color: #666;
        margin-bottom: 15px;
        font-style: italic;
    }
    
    .article-abstract {
        font-size: 0.95rem;
        line-height: 1.5;
        color: #444;
        margin-bottom: 15px;
    }
    
    .article-image-placeholder {
        width: 100%;
        background-color: #f8f9fa;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 15px;
        border-radius: 4px;
        overflow: hidden;
        border: 1px solid #e9ecef;
    }
    
    .article-why-matters {
        font-size: 0.9rem;
        background-color: #f8f9fa;
        padding: 10px;
        border-left: 3px solid #0d6efd;
        margin-bottom: 10px;
    }
    
    .read-more {
        display: inline-block;
        color: #0d6efd;
        font-weight: 600;
        font-size: 0.9rem;
        margin-top: 10px;
        text-decoration: none;
    }
    
    /* Feature articles layout */
    .feature-grid {
        display: grid;
        grid-template-columns: 1fr;
        gap: 20px;
        margin-bottom: 30px;
    }
    
    /* Trending articles */
    .trending-articles {
        margin-bottom: 30px;
    }
    
    .trending-item {
        display: flex;
        align-items: center;
        padding: 10px 0;
        border-bottom: 1px solid #e9ecef;
    }
    
    .trending-number {
        font-size: 1.2rem;
        font-weight: 700;
        color: #c00;
        margin-right: 15px;
        min-width: 25px;
    }
    
    .trending-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #333;
    }
    
    /* Footer */
    .site-footer {
        background-color: #222;
        color: #fff;
        padding: 30px 20px;
        text-align: center;
        margin-top: 40px;
    }
    
    .footer-title {
        font-family: 'Georgia', serif;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 20px;
        text-transform: uppercase;
    }
    
    .footer-text {
        color: #aaa;
        font-size: 0.85rem;
        margin-bottom: 10px;
    }
    
    /* Buttons and form elements */
    .stButton > button {
        background-color: #0d6efd !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        border-radius: 4px !important;
    }
    
    .stButton > button:hover {
        background-color: #0b5ed7 !important;
    }
    
    /* Make elements in the sidebar more compact */
    [data-testid="stSidebar"] .css-1544g2n {
        padding-top: 2rem;
    }
    
    /* Responsive adjustments */
    @media (min-width: 992px) {
        .feature-grid {
            grid-template-columns: 2fr 1fr;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- Functions ---
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

def display_image_placeholder(topic, size=(300, 200), is_main=False):
    """Display a styled image placeholder for the topic"""
    topic_to_color = {
        "Top Technology News": "#0066cc",
        "Inspiring Stories": "#6b5b95",
        "Global Politics": "#d64161",
        "Climate and Environment": "#3cb371",
        "Health and Wellness": "#ff7e67"
    }
    
    topic_to_emoji = {
        "Top Technology News": "🖥️",
        "Inspiring Stories": "✨",
        "Global Politics": "🌎",
        "Climate and Environment": "🌿",
        "Health and Wellness": "🏥"
    }
    
    # Get color and emoji, or use defaults
    color = topic_to_color.get(topic, "#333333")
    emoji = topic_to_emoji.get(topic, "📰")
    
    # Adjust size for main article
    if is_main:
        height = 300
        emoji_size = 60
    else:
        height = size[1]
        emoji_size = 40
    
    # Create placeholder with topic styling
    st.markdown(f"""
    <div class="article-image-placeholder" style="
        height: {height}px; 
        background: linear-gradient(135deg, {color}25, {color}40);
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: url('data:image/svg+xml;utf8,<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"20\" height=\"20\" viewBox=\"0 0 20 20\"><path fill=\"%23ffffff15\" d=\"M0,0 L20,20 M20,0 L0,20\"/></svg>');
            opacity: 0.3;
        "></div>
        <div style="
            position: relative;
            z-index: 2;
            text-align: center;
            padding: 15px;
        ">
            <div style="font-size: {emoji_size}px; margin-bottom: 10px;">{emoji}</div>
            <div style="
                color: #333; 
                background-color: rgba(255,255,255,0.7); 
                padding: 5px 10px; 
                border-radius: 4px; 
                font-weight: 600;
                display: inline-block;
            ">{topic}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_featured_article(article):
    """Display a featured article in Daily Wire style"""
    try:
        # Make sure we have all needed columns
        if 'topic' not in article:
            article['topic'] = "General News"
        if 'rewritten_title' not in article or pd.isna(article['rewritten_title']) or article['rewritten_title'] == '':
            article['rewritten_title'] = article.get('title', 'Untitled Article')
        if 'abstract' not in article or pd.isna(article['abstract']):
            article['abstract'] = "No abstract available for this article."
        if 'explanation' not in article or pd.isna(article['explanation']):
            article['explanation'] = "This article provides important information for our readers."
        
        # Format the article  
        st.markdown(f'<div class="article-card">', unsafe_allow_html=True)
        st.markdown(f'<span class="article-tag">{article["topic"]}</span>', unsafe_allow_html=True)
        st.markdown(f'<h2 class="main-article-title">{article["rewritten_title"]}</h2>', unsafe_allow_html=True)
        
        # Author byline
        author = random.choice(["Sarah Chen", "Michael Johnson", "Priya Patel", "Robert Williams"])
        st.markdown(f'<p class="article-byline">By {author} | {datetime.datetime.now().strftime("%B %d, %Y")}</p>', unsafe_allow_html=True)
        
        # Image placeholder
        display_image_placeholder(article["topic"], is_main=True)
        
        # Article abstract
        abstract = article['abstract']
        if abstract and len(abstract) > 300:
            abstract = abstract[:300] + "..."
        st.markdown(f'<p class="article-abstract">{abstract}</p>', unsafe_allow_html=True)
        
        # Why it matters box
        st.markdown(f'<div class="article-why-matters"><strong>Why it matters:</strong> {article["explanation"]}</div>', unsafe_allow_html=True)
        
        # Read more link
        st.markdown('<a href="#" class="read-more">Continue Reading →</a>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        logging.error(f"Error displaying featured article: {e}")
        st.error(f"Error displaying featured article: {str(e)}")

def display_secondary_article(article):
    """Display a secondary article in Daily Wire style"""
    try:
        # Make sure we have all needed columns
        if 'topic' not in article:
            article['topic'] = "General News"
        if 'rewritten_title' not in article or pd.isna(article['rewritten_title']) or article['rewritten_title'] == '':
            article['rewritten_title'] = article.get('title', 'Untitled Article')
        if 'abstract' not in article or pd.isna(article['abstract']):
            article['abstract'] = "No abstract available for this article."
        
        # Format the article
        st.markdown(f'<div class="article-card">', unsafe_allow_html=True)
        st.markdown(f'<span class="article-tag">{article["topic"]}</span>', unsafe_allow_html=True)
        st.markdown(f'<h3 class="article-title">{article["rewritten_title"]}</h3>', unsafe_allow_html=True)
        
        # Image placeholder
        display_image_placeholder(article["topic"])
        
        # Article abstract (shorter for secondary)
        abstract = article['abstract']
        if abstract and len(abstract) > 150:
            abstract = abstract[:150] + "..."
        st.markdown(f'<p class="article-abstract">{abstract}</p>', unsafe_allow_html=True)
        
        # Read more link
        st.markdown('<a href="#" class="read-more">Read More →</a>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        logging.error(f"Error displaying secondary article: {e}")

def display_sidebar_article(article, number):
    """Display a trending sidebar article"""
    try:
        if 'rewritten_title' not in article or pd.isna(article['rewritten_title']) or article['rewritten_title'] == '':
            article['rewritten_title'] = article.get('title', 'Untitled Article')
            
        st.markdown(f"""
        <div class="trending-item">
            <div class="trending-number">{number}</div>
            <div class="trending-title">{article["rewritten_title"]}</div>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        logging.error(f"Error displaying sidebar article: {e}")

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
        st.markdown('<h2 style="margin-top:0; padding-top:0; font-size:1.5rem; font-weight:700;">📰 NEWSPAPER CONTROLS</h2>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.9rem;">Configure your daily newspaper</p>', unsafe_allow_html=True)
        st.markdown('<hr style="margin:10px 0;">', unsafe_allow_html=True)
        
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
    
    # --- Daily Wire style header and navigation ---
    st.markdown('<div class="site-header">', unsafe_allow_html=True)
    st.markdown('<h1 class="newspaper-title">THE DAILY CHRONICLE</h1>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation menu
    st.markdown('<div class="nav-menu">', unsafe_allow_html=True)
    for nav_item in ["Politics", "Entertainment", "Tech", "Sports", "Opinion", "Health", "World"]:
        is_active = nav_item == "Politics"  # Make first one active
        active_class = " active" if is_active else ""
        st.markdown(f'<span class="nav-item{active_class}">{nav_item}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Main container for content
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Check if we have articles to display
    if 'loaded_articles' in st.session_state and st.session_state.loaded_articles is not None and len(st.session_state.loaded_articles) > 0:
        articles_df = st.session_state.loaded_articles
        
        # Top Featured Articles
        st.markdown('<h2 class="section-title">FEATURED NEWS</h2>', unsafe_allow_html=True)
        
        # Create a feature grid with main feature and sidebar
        st.markdown('<div class="feature-grid">', unsafe_allow_html=True)
        
        # Left column - main featured article
        st.markdown('<div>', unsafe_allow_html=True)
        if len(articles_df) > 0:
            display_featured_article(articles_df.iloc[0])
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Right column - trending articles
        st.markdown('<div class="trending-articles">', unsafe_allow_html=True)
        st.markdown('<h3 style="font-size:1.2rem; font-weight:700; margin-bottom:15px; font-family:Georgia, serif;">TRENDING NOW</h3>', unsafe_allow_html=True)
        
        # Display trending sidebar articles (next 5 articles after main)
        for i in range(1, min(6, len(articles_df))):
            if i < len(articles_df):
                display_sidebar_article(articles_df.iloc[i], i)
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # More News Section with 2-column grid
        st.markdown('<h2 class="section-title">MORE NEWS</h2>', unsafe_allow_html=True)
        
        # Group remaining articles by topic
        remaining_articles = articles_df.iloc[6:] if len(articles_df) > 6 else pd.DataFrame()
        if len(remaining_articles) > 0:
            # Group by topic and create topic sections
            topics = remaining_articles['topic'].unique()
            
            for topic in topics:
                topic_articles = remaining_articles[remaining_articles['topic'] == topic]
                if len(topic_articles) > 0:
                    # Create section for each topic
                    st.markdown(f'<h3 style="font-size:1.3rem; font-weight:700; margin:20px 0 15px; color:#333; font-family:Georgia, serif;">{topic.upper()}</h3>', unsafe_allow_html=True)
                    
                    # Create a grid of secondary articles
                    cols = st.columns(2)
                    
                    for i, (_, article) in enumerate(topic_articles.iterrows()):
                        with cols[i % 2]:
                            display_secondary_article(article)
        
        # Site footer
        st.markdown('<div class="site-footer">', unsafe_allow_html=True)
        st.markdown('<h3 class="footer-title">THE DAILY CHRONICLE</h3>', unsafe_allow_html=True)
        st.markdown('<p class="footer-text">© 2025 The Daily Chronicle. All Rights Reserved.</p>', unsafe_allow_html=True)
        st.markdown('<p class="footer-text">Powered by AI-curated content | Privacy Policy | Terms of Service</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        # Check if curation is in progress
        if 'curation_started' in st.session_state and st.session_state.curation_started and not st.session_state.get('curation_complete', False):
            # Show progress message
            st.markdown("""
            <div style="text-align:center; padding:50px 20px; background-color:#f8f9fa; border-radius:4px; margin:30px 0;">
                <h2 style="margin-bottom:20px; color:#333;">⏳ Curation in Progress</h2>
                <p style="font-size:1.1rem; color:#666;">Please wait while we prepare your personalized news articles...</p>
                <div style="width:100px; height:100px; margin:30px auto; border:5px solid #f3f3f3; border-top:5px solid #0d6efd; border-radius:50%; animation:spin 1s linear infinite;"></div>
                <style>
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                </style>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Show welcome message and instructions
            st.markdown("""
            <div style="text-align:center; padding:40px 20px; background-color:#f8f9fa; border-radius:4px; margin:30px 0;">
                <h2 style="margin-bottom:20px; color:#333;">Welcome to The Daily Chronicle</h2>
                <p style="font-size:1.1rem; color:#666; margin-bottom:30px;">Your AI-powered personalized news platform</p>
                
                <div style="max-width:600px; margin:0 auto; text-align:left; background:#fff; padding:25px; border-radius:5px; box-shadow:0 1px 3px rgba(0,0,0,0.1);">
                    <h3 style="border-bottom:2px solid #0d6efd; padding-bottom:10px; margin-bottom:20px; font-size:1.3rem;">Get Started in Two Simple Steps:</h3>
                    
                    <div style="display:flex; align-items:center; margin-bottom:20px;">
                        <div style="background:#0d6efd; color:white; width:30px; height:30px; border-radius:50%; display:flex; align-items:center; justify-content:center; margin-right:15px; font-weight:bold;">1</div>
                        <div>
                            <p style="margin:0; font-weight:600;">Select Topics in the Sidebar</p>
                            <p style="margin:5px 0 0; color:#666; font-size:0.9rem;">Choose which news categories you want to include</p>
                        </div>
                    </div>
                    
                    <div style="display:flex; align-items:center;">
                        <div style="background:#0d6efd; color:white; width:30px; height:30px; border-radius:50%; display:flex; align-items:center; justify-content:center; margin-right:15px; font-weight:bold;">2</div>
                        <div>
                            <p style="margin:0; font-weight:600;">Click "CURATE FRESH ARTICLES"</p>
                            <p style="margin:5px 0 0; color:#666; font-size:0.9rem;">Our AI will find and enhance relevant articles for you</p>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show example of what they'll get
            st.markdown("""
            <h2 class="section-title">PREVIEW</h2>
            <div style="display:flex; align-items:center; justify-content:center; padding:30px; background:#f0f0f0; border-radius:4px; margin-bottom:30px;">
                <img src="https://via.placeholder.com/800x400?text=Your+Personalized+Newspaper+Preview" style="max-width:100%; border:1px solid #ddd; border-radius:4px; box-shadow:0 2px 10px rgba(0,0,0,0.1);">
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close main-container

# Make sure to create these session state variables
if 'curation_started' not in st.session_state:
    st.session_state.curation_started = False
if 'curation_complete' not in st.session_state:
    st.session_state.curation_complete = False

# Run the main application
if __name__ == "__main__":
    main()