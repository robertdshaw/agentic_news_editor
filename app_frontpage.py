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
from PIL import Image

# --- Page configuration ---
st.set_page_config(
    page_title="The Daily Chronicle",
    layout="wide",
    initial_sidebar_state="expanded" # Make sidebar expanded by default
)

# --- Custom CSS for newspaper styling ---
st.markdown("""
<style>
    .newspaper-title {
        font-family: "Times New Roman", Times, serif;
        font-size: 4rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0;
        padding-top: 20px;
        letter-spacing: -1px;
    }
    .newspaper-subtitle {
        font-family: "Times New Roman", Times, serif;
        font-size: 1.2rem;
        font-style: italic;
        text-align: center;
        margin-bottom: 20px;
        color: #444;
    }
    .newspaper-motto {
        font-style: italic;
        text-align: center;
        margin-bottom: 10px;
        font-size: 1.2rem;
    }
    .date-line {
        display: flex;
        justify-content: space-between;
        border-top: 1px solid #000;
        border-bottom: 1px solid #000;
        padding: 5px 0;
        margin-bottom: 20px;
    }
    .breaking-news {
        background-color: #c00;
        color: white;
        padding: 10px;
        text-align: center;
        font-weight: bold;
        margin: 20px 0;
        font-size: 1.2rem;
    }
    .article-tag {
        background-color: #333;
        color: white;
        padding: 3px 8px;
        font-size: 0.8rem;
        border-radius: 3px;
        margin-bottom: 10px;
        display: inline-block;
    }
    .byline {
        font-style: italic;
        font-size: 0.9rem;
        margin-bottom: 15px;
        color: #555;
    }
    .article-container {
        padding: 0 10px;
        margin-bottom: 30px;
    }
    .sidebar-header {
        border-bottom: 2px solid #333;
        padding-bottom: 5px;
        margin-bottom: 15px;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .section-divider {
        height: 1px;
        background-color: #ddd;
        margin: 20px 0;
    }
    .continue-reading {
        font-weight: bold;
        font-style: italic;
        color: #444;
    }
    .advert-container {
        background-color: #f5f5f5;
        border: 1px solid #ddd;
        padding: 20px;
        text-align: center;
        margin: 30px 0;
    }
    .footer {
        text-align: center;
        margin-top: 30px;
        padding-top: 20px;
        border-top: 1px solid #ddd;
        color: #666;
        font-size: 0.9rem;
    }
    h1, h2, h3 {
        font-family: "Times New Roman", Times, serif;
    }
    p {
        font-family: Georgia, serif;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# --- Functions from your original script ---
@st.cache_resource
def load_models_and_data():
    """Load FAISS index, metadata, and embedding model with caching"""
    try:
        index = faiss.read_index("articles_faiss.index")
        articles_df = pd.read_csv("articles_with_embeddings.csv")
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        return index, articles_df, model
    except Exception as e:
        st.error(f"Error loading models and data: {e}")
        if not os.path.exists("articles_faiss.index"):
            st.error("⚠️ articles_faiss.index file not found!")
        if not os.path.exists("articles_with_embeddings.csv"):
            st.error("⚠️ articles_with_embeddings.csv file not found!")
        return None, None, None

def load_topics(filename="memory_topics.json"):
    """Load yesterday's topics from memory file"""
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_topics(topics, filename="memory_topics.json"):
    """Save today's topics to memory file"""
    with open(filename, "w") as f:
        json.dump(topics, f)

def get_openai_client():
    """Initialize OpenAI client with API key"""
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        st.sidebar.error("❌ OPENAI_API_KEY is missing. Check your .env file!")
        return None
    
    return OpenAI(api_key=openai_api_key)

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
        st.sidebar.warning(f"Error rewriting headline: {e}")
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
        st.sidebar.warning(f"Error generating explanation: {e}")
        return "This article provides important information for our readers."

def curate_articles_for_topic(query_text, index, articles_df, model, openai_client, k=5):
    """Find and enhance articles for a given topic"""
    query_embedding = model.encode([query_text])
    D, I = index.search(np.array(query_embedding), k=k)
    topic_articles = articles_df.iloc[I[0]].copy()
    
    # Process each article individually
    for idx, row in topic_articles.iterrows():
        with st.spinner(f"Enhancing article: {row['title'][:30]}..."):
            topic_articles.at[idx, 'rewritten_title'] = rewrite_headline(
                openai_client, row['title'], row['abstract']
            )
            topic_articles.at[idx, 'explanation'] = generate_explanation(
                openai_client, row['title'], row['abstract']
            )
    
    return topic_articles

def get_article_image(topic):
    topic_keywords = {
        "Top Technology News": "technology",
        "Inspiring Stories": "inspiration",
        "Global Politics": "politics",
        "Climate and Environment": "environment",
        "Health and Wellness": "health",
    }
    keyword = topic_keywords.get(topic, "news")
    return f"https://source.unsplash.com/800x400/?{keyword}"


def display_article(col, article, is_main=False):
    """Display a single article in the specified column"""
    with col:
        st.markdown(f'<span class="article-tag">{article["topic"]}</span>', unsafe_allow_html=True)
        
        if is_main:
            st.markdown(f"## {article['rewritten_title']}")
            st.image(get_article_image(article["topic"]), caption=article["topic"], use_container_width=True)
            st.markdown(f'<p class="byline">By {random.choice(["Sarah Chen", "Michael Johnson", "Priya Patel", "Robert Williams"])}, {article["topic"]} Editor</p>', unsafe_allow_html=True)
        else:
            st.markdown(f"### {article['rewritten_title']}")
            st.image(get_article_image(article["topic"]), caption=article["topic"], use_container_width=True)
        
        # Show abstract and explanation
        st.write(f"{article['abstract'][:200]}..." if len(article['abstract']) > 200 else article['abstract'])
        st.write(f"**Why it matters:** {article['explanation']}")
        
        if is_main:
            st.markdown('<p class="continue-reading">Continue reading →</p>', unsafe_allow_html=True)

def load_curated_articles():
    """Load previously curated articles if available"""
    try:
        return pd.read_csv("curated_full_daily_output.csv")
    except FileNotFoundError:
        return None

# --- Main Application Logic ---
def main():
    # Make sidebar explicitly visible with a large noticeable header
    with st.sidebar:
        st.title("📰 NEWSPAPER CONTROLS")
        st.markdown("### Use these controls to manage your newspaper")
        st.markdown("---")
    
    # Display newspaper header
    st.markdown('<h1 class="newspaper-title">THE DAILY CHRONICLE</h1>', unsafe_allow_html=True)
    st.markdown('<p class="newspaper-motto">Delivering Truth, Inspiring Minds</p>', unsafe_allow_html=True)
    
    # Date and edition line
    today = datetime.datetime.now()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"{today.strftime('%A, %B %d, %Y').upper()}")
    with col2:
        st.write(f"VOL. {today.year - 1997}, NO. {today.timetuple().tm_yday}")
    with col3:
        st.write("$2.00")
    
    # Breaking news banner - can be dynamically generated
    st.markdown('<div class="breaking-news">LATEST UPDATES: Curated news from across the globe</div>', unsafe_allow_html=True)
    
    # Add a notice about the sidebar at the top
    st.info("⬅️ Use the sidebar on the left for curation controls. If you don't see it, click the '>' arrow in the top-left corner.")
    
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
        articles_per_topic = st.slider("Articles per topic", 1, 10, 5)
        
        # Action buttons
        if st.button("Curate Fresh Articles"):
            if not selected_topics:
                st.error("Please select at least one topic to curate")
            else:
                # Load models and data
                with st.spinner("Loading models and data..."):
                    index, articles_df, model = load_models_and_data()
                    openai_client = get_openai_client()
                
                if index is None or articles_df is None or model is None:
                    st.error("Failed to load required models and data")
                else:
                    # Curate articles for each selected topic
                    all_curated_articles = []
                    
                    for topic in selected_topics:
                        with st.spinner(f"Curating articles for {topic}..."):
                            query_text = editorial_queries[topic]
                            topic_articles = curate_articles_for_topic(
                                query_text, index, articles_df, model, openai_client, k=articles_per_topic
                            )
                            topic_articles["topic"] = topic
                            all_curated_articles.append(topic_articles)
                    
                    # Combine all curated articles
                    if all_curated_articles:
                        full_curated_df = pd.concat(all_curated_articles, ignore_index=True)
                        full_curated_df.to_csv("curated_full_daily_output.csv", index=False)
                        st.success("✅ Articles curated successfully!")
                        
                        # Update memory for future reference
                        save_topics(selected_topics)
                    else:
                        st.warning("No articles were curated")
        
        # Load previously curated articles
        st.markdown("---")
        if st.button("Load Previously Curated Articles"):
            st.session_state.loaded_articles = load_curated_articles()
            if st.session_state.loaded_articles is not None:
                st.success(f"Loaded {len(st.session_state.loaded_articles)} articles")
            else:
                st.error("No previously curated articles found")
    
    # Add a prominent button to show sidebar if it's hidden
    if st.button("🔍 Show Curation Controls", use_container_width=True):
        # This won't actually show the sidebar, but helps users understand they need to click the arrow
        st.info("Look for the '>' arrow in the top-left corner of the screen to expand the sidebar")
    
    # Check if we have articles to display
    if 'loaded_articles' in st.session_state and st.session_state.loaded_articles is not None:
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
                    st.write(f"{article['abstract'][:100]}..." if len(article['abstract']) > 100 else article['abstract'])
                    st.markdown("---")
        
        # Secondary articles in a two-column layout
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("MORE TOP STORIES")
        
        # Create rows of 2 articles each
        remaining_articles = articles_df.iloc[4:] if len(articles_df) > 4 else pd.DataFrame()
        
        for i in range(0, len(remaining_articles), 2):
            cols = st.columns(2)
            
            for j in range(2):
                if i + j < len(remaining_articles):
                    article = remaining_articles.iloc[i + j]
                    display_article(cols[j], article)
        
        # Advertisement section
        st.markdown('<div class="advert-container">', unsafe_allow_html=True)
        st.markdown("### ADVERTISEMENT")
        st.write("Support quality journalism with a subscription to The Daily Chronicle.")
        st.button("SUBSCRIBE NOW", key="subscribe_button")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Footer
        st.markdown('<div class="footer">', unsafe_allow_html=True)
        st.write(f"© {today.year} The Daily Chronicle | All Rights Reserved | Built with AI-powered curation")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Display very clear instructions if no articles loaded
        st.warning("⚠️ No articles loaded. You need to curate or load articles first.")
        
        st.markdown("### How to Get Started:")
        st.markdown("""
        1. **Find the sidebar controls** - Look for the ">" arrow in the top-left corner if you don't see the sidebar
        2. **Select topics** - Choose which news categories you want in your paper
        3. **Curate articles** - Click the "Curate Fresh Articles" button to find and enhance content
        4. **Or load existing articles** - Use "Load Previously Curated Articles" to display already curated content
        """)
        
        # Add a visual guide with columns
        st.markdown("### Visual Guide:")
        guide_col1, guide_col2 = st.columns([1, 1])
        
        with guide_col1:
            st.markdown("#### Step 1: Sidebar Controls")
            st.info("Select topics and click 'Curate Fresh Articles'")
            
        with guide_col2:
            st.markdown("#### Step 2: View Results")
            st.success("Your AI-curated newspaper appears here")

# Add a special trick to make the sidebar always visible
def inject_sidebar_force_visible():
    st.markdown("""
    <style>
        /* Force sidebar to be open */
        section[data-testid="stSidebar"] {
            display: block !important;
            width: 300px !important;
            min-width: 300px !important;
            flex-shrink: 0 !important;
        }
        
        /* Make main content adjust accordingly */
        .main .block-container {
            max-width: calc(100% - 300px) !important;
            padding-left: 1rem !important;
        }
        
        /* Add visual cue */
        section[data-testid="stSidebar"] .css-1d391kg {
            background-color: #f0f8ff;
            padding: 20px;
            border-right: 2px solid #0066cc;
        }
        
        /* Make sidebar controls more obvious */
        section[data-testid="stSidebar"] button {
            width: 100%;
            background-color: #0066cc;
            color: white;
            font-weight: bold;
            margin-top: 10px;
            margin-bottom: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    inject_sidebar_force_visible()
    main()