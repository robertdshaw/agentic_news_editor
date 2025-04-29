import streamlit as st
import os
import json
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import datetime
import random
import time
import requests
from io import BytesIO
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Page configuration ---
st.set_page_config(
    page_title="The Daily Chronicle",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for newspaper styling and compact layout ---
st.markdown("""
<style>
    /* More compact layout */
    .main .block-container {
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    
    /* Force sidebar to be open and styled */
    section[data-testid="stSidebar"] {
        display: block !important;
        width: 250px !important;
        min-width: 250px !important;
        background-color: #f0f2f6;
        border-right: 1px solid #ddd;
    }
    
    /* Newspaper styling */
    .newspaper-title {
        font-family: "Times New Roman", Times, serif;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0;
        padding-top: 10px;
        letter-spacing: -1px;
    }
    .newspaper-motto {
        font-style: italic;
        text-align: center;
        margin-bottom: 5px;
        font-size: 1rem;
    }
    .date-line {
        display: flex;
        justify-content: space-between;
        border-top: 1px solid #000;
        border-bottom: 1px solid #000;
        padding: 5px 0;
        margin-bottom: 10px;
    }
    .breaking-news {
        background-color: #c00;
        color: white;
        padding: 8px;
        text-align: center;
        font-weight: bold;
        margin: 10px 0;
        font-size: 1rem;
    }
    .article-tag {
        background-color: #333;
        color: white;
        padding: 3px 8px;
        font-size: 0.7rem;
        border-radius: 3px;
        margin-bottom: 5px;
        display: inline-block;
    }
    .byline {
        font-style: italic;
        font-size: 0.8rem;
        margin-bottom: 10px;
        color: #555;
    }
    .article-container {
        padding: 0 5px;
        margin-bottom: 15px;
    }
    .sidebar-header {
        border-bottom: 2px solid #333;
        padding-bottom: 5px;
        margin-bottom: 10px;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .section-divider {
        height: 1px;
        background-color: #ddd;
        margin: 15px 0;
    }
    .continue-reading {
        font-weight: bold;
        font-style: italic;
        color: #444;
        font-size: 0.8rem;
    }
    .advert-container {
        background-color: #f5f5f5;
        border: 1px solid #ddd;
        padding: 15px;
        text-align: center;
        margin: 15px 0;
    }
    .footer {
        text-align: center;
        margin-top: 15px;
        padding-top: 10px;
        border-top: 1px solid #ddd;
        color: #666;
        font-size: 0.8rem;
    }
    
    /* Make text and elements more compact */
    h1 {
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
    }
    h2 {
        font-size: 1.4rem;
        margin-bottom: 0.5rem;
    }
    h3 {
        font-size: 1.1rem;
        margin-bottom: 0.3rem;
    }
    p {
        font-family: Georgia, serif;
        line-height: 1.4;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    
    /* Make buttons more prominent */
    .stButton>button {
        width: 100%;
        font-weight: bold;
    }
    
    /* Compact image styling */
    img {
        margin-bottom: 0.5rem !important;
    }
    
    /* Fix for the sidebar scrolling */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.5rem;
    }
    
    /* Progress bar styling */
    .stProgress .st-bo {
        background-color: #0066cc;
    }
</style>
""", unsafe_allow_html=True)

# --- Functions from original script with improvements ---
@st.cache_resource
def load_models_and_data():
    """Load FAISS index, metadata, and embedding model with caching"""
    try:
        logging.info("Loading FAISS index and models")
        index = faiss.read_index("articles_faiss.index")
        articles_df = pd.read_csv("articles_with_embeddings.csv")
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        logging.info(f"Loaded {len(articles_df)} articles and model")
        return index, articles_df, model
    except Exception as e:
        logging.error(f"Error loading models and data: {e}")
        if not os.path.exists("articles_faiss.index"):
            st.error("⚠️ articles_faiss.index file not found!")
        if not os.path.exists("articles_with_embeddings.csv"):
            st.error("⚠️ articles_with_embeddings.csv file not found!")
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
        # Create query embedding and search
        query_embedding = model.encode([query_text])
        D, I = index.search(np.array(query_embedding), k=k)
        
        # Extract articles
        topic_articles = articles_df.iloc[I[0]].copy()
        if len(topic_articles) == 0:
            logging.warning(f"No articles found for query: {query_text}")
            return pd.DataFrame()
        
        # Process each article individually
        total = len(topic_articles)
        for idx, row in topic_articles.iterrows():
            if progress_bar is not None:
                progress_value = (idx - topic_articles.index[0] + 1) / total
                progress_bar.progress(progress_value, text=f"Processing article {idx - topic_articles.index[0] + 1}/{total}")
            
            topic_articles.at[idx, 'rewritten_title'] = rewrite_headline(
                openai_client, row['title'], row['abstract']
            )
            topic_articles.at[idx, 'explanation'] = generate_explanation(
                openai_client, row['title'], row['abstract']
            )
        
        if progress_bar is not None:
            progress_bar.progress(1.0, text="Processing complete!")
            time.sleep(0.5)  # Give user time to see the completed progress
        
        return topic_articles
    except Exception as e:
        logging.error(f"Error curating articles: {e}")
        return pd.DataFrame()

def get_topic_image_url(topic):
    """Return an appropriate image URL for the topic"""
    topic_to_image = {
        "Top Technology News": "https://source.unsplash.com/featured/?technology,digital",
        "Inspiring Stories": "https://source.unsplash.com/featured/?inspire,hope",
        "Global Politics": "https://source.unsplash.com/featured/?politics,government",
        "Climate and Environment": "https://source.unsplash.com/featured/?climate,nature",
        "Health and Wellness": "https://source.unsplash.com/featured/?health,medical"
    }
    
    return topic_to_image.get(topic, "https://source.unsplash.com/featured/?news")

def display_image_for_topic(topic, size=(300, 200)):
    """Display an image for the topic, with fallback to placeholder"""
    try:
        image_url = get_topic_image_url(topic)
        response = requests.get(image_url, timeout=5)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            st.image(image, width=size[0])
            return True
        else:
            # Fallback to placeholder
            st.image(f"https://via.placeholder.com/{size[0]}x{size[1]}?text={topic.replace(' ', '+')}")
            return False
    except Exception as e:
        logging.error(f"Error displaying image for {topic}: {e}")
        # Use Streamlit placeholder directly if external placeholders fail
        st.image(f"https://via.placeholder.com/{size[0]}x{size[1]}?text={topic.replace(' ', '+')}")
        return False

def display_article(col, article, is_main=False):
    """Display a single article in the specified column with proper styling"""
    with col:
        st.markdown(f'<span class="article-tag">{article["topic"]}</span>', unsafe_allow_html=True)
        
        if is_main:
            st.markdown(f"## {article['rewritten_title']}")
            display_image_for_topic(article["topic"], size=(600, 350))
            st.markdown(f'<p class="byline">By {random.choice(["Sarah Chen", "Michael Johnson", "Priya Patel", "Robert Williams"])}, {article["topic"]} Editor</p>', unsafe_allow_html=True)
        else:
            st.markdown(f"### {article['rewritten_title']}")
            display_image_for_topic(article["topic"], size=(300, 200))
        
        # Show abstract and explanation
        st.write(f"{article['abstract'][:200]}..." if len(article['abstract']) > 200 else article['abstract'])
        st.markdown(f"**Why it matters:** {article['explanation']}")
        
        if is_main:
            st.markdown('<p class="continue-reading">Continue reading →</p>', unsafe_allow_html=True)

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
    # Make sidebar explicitly visible with a large noticeable header
    with st.sidebar:
        st.title("📰 NEWSPAPER CONTROLS")
        st.markdown("### Configure Your Paper")
        st.markdown("---")
    
    # Display newspaper header (more compact)
    st.markdown('<h1 class="newspaper-title">THE DAILY CHRONICLE</h1>', unsafe_allow_html=True)
    st.markdown('<p class="newspaper-motto">Delivering Truth, Inspiring Minds</p>', unsafe_allow_html=True)
    
    # Date and edition line in a more compact layout
    today = datetime.datetime.now()
    cols = st.columns([1, 1, 1])
    with cols[0]:
        st.write(f"{today.strftime('%A, %B %d, %Y').upper()}")
    with cols[1]:
        st.write(f"VOL. {today.year - 1997}, NO. {today.timetuple().tm_yday}")
    with cols[2]:
        st.write("$2.00")
    
    # Breaking news banner (more compact)
    st.markdown('<div class="breaking-news">LATEST UPDATES: Curated news from across the globe</div>', unsafe_allow_html=True)
    
    # Setup the rest of the sidebar controls
    with st.sidebar:
        # Editorial queries - same as in original script
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
        curate_button = st.button("🔄 Curate Fresh Articles", use_container_width=True)
        
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
        
        # Load previously curated articles
        st.markdown("---")
        if st.button("📂 Load Previous Articles", use_container_width=True):
            with st.spinner("Loading previously curated articles..."):
                st.session_state.loaded_articles = load_curated_articles()
                if st.session_state.loaded_articles is not None:
                    st.success(f"Loaded {len(st.session_state.loaded_articles)} articles")
                else:
                    st.error("No previously curated articles found")
    
    # Check if we have articles to display
    if 'loaded_articles' in st.session_state and st.session_state.loaded_articles is not None and len(st.session_state.loaded_articles) > 0:
        articles_df = st.session_state.loaded_articles
        
        # Main content area with layout
        main_col, sidebar_col = st.columns([2, 1])
        
        # Main article section
        with main_col:
            # Get the first article as the main feature
            if len(articles_df) > 0:
                main_article = articles_df.iloc[0]
                display_article(main_col, main_article, is_main=True)
        
        # Article sidebar (not to be confused with Streamlit's sidebar)
        with sidebar_col:
            st.markdown('<div class="sidebar-header">IN BRIEF</div>', unsafe_allow_html=True)
            
            # Display next 3 articles in sidebar (if available)
            for i in range(1, min(4, len(articles_df))):
                if i < len(articles_df):
                    article = articles_df.iloc[i]
                    st.markdown(f"### {article['rewritten_title']}")
                    st.write(f"{article['abstract'][:80]}..." if len(article['abstract']) > 80 else article['abstract'])
                    st.markdown("---")
        
        # Secondary articles in a two-column layout
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("MORE TOP STORIES")
        
        # Create rows of 2 articles each
        remaining_articles = articles_df.iloc[4:] if len(articles_df) > 4 else pd.DataFrame()
        
        if len(remaining_articles) > 0:
            for i in range(0, len(remaining_articles), 2):
                cols = st.columns(2)
                
                for j in range(2):
                    if i + j < len(remaining_articles):
                        article = remaining_articles.iloc[i + j]
                        display_article(cols[j], article)
        
        # Advertisement section (more compact)
        st.markdown('<div class="advert-container">', unsafe_allow_html=True)
        st.markdown("### ADVERTISEMENT")
        st.write("Support quality journalism with a subscription to The Daily Chronicle.")
        st.button("SUBSCRIBE NOW", key="subscribe_button")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Footer (more compact)
        st.markdown('<div class="footer">', unsafe_allow_html=True)
        st.write(f"© {today.year} The Daily Chronicle | All Rights Reserved | Built with AI-powered curation")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Check if curation is in progress
        if 'curation_started' in st.session_state and st.session_state.curation_started and not st.session_state.get('curation_complete', False):
            st.info("⏳ Curation in progress... Please wait while articles are being processed.")
        else:
            # Display very clear instructions with visual cues
            st.warning("⚠️ No articles loaded. Please curate or load articles using the sidebar controls.")
            
            # Create a visual guide with images
            st.markdown("### Quick Start Guide:")
            
            guide_col1, guide_col2 = st.columns(2)
            with guide_col1:
                st.markdown("#### Step 1: Select Topics")
                st.info("Choose topics in the sidebar and click '🔄 Curate Fresh Articles'")
                st.image("https://via.placeholder.com/400x200?text=Select+Topics+in+Sidebar", width=300)
            
            with guide_col2:
                st.markdown("#### Step 2: View Your Newspaper")
                st.success("Your AI-curated newspaper will appear here")
                st.image("https://via.placeholder.com/400x200?text=Your+Newspaper+Layout", width=300)

# Make sure to create these session state variables
if 'curation_started' not in st.session_state:
    st.session_state.curation_started = False
if 'curation_complete' not in st.session_state:
    st.session_state.curation_complete = False

# Run the main application
if __name__ == "__main__":
    main()