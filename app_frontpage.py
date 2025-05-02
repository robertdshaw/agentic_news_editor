import os
# os.environ["STREAMLIT_WATCH_FORCE_POLLING"] = "true"
import disable_torch_watch
import streamlit as st
import json
import pandas as pd
import numpy as np
from headline_metrics import HeadlineMetrics
from headline_learning import HeadlineLearningLoop

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

headline_learner = HeadlineLearningLoop()

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
    
    /* Add navigation styling */
    .newspaper-title {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .nav-item {
        text-align: center;
        font-weight: bold;
        padding: 0.5rem;
        cursor: pointer;
    }
    
    .article-tag {
        display: inline-block;
        background-color: #f0f0f0;
        padding: 2px 8px;
        margin-bottom: 8px;
        font-size: 0.8em;
        border-radius: 3px;
        font-weight: bold;
        color: #555;
    }
    
    .article-byline {
        font-style: italic;
        color: #666;
        margin-bottom: 10px;
        font-size: 0.9em;
    }
    
    .article-why-matters {
        background-color: #f9f9f9;
        padding: 10px;
        border-left: 3px solid #007bff;
        margin: 10px 0;
    }
    
    .section-title {
        border-bottom: 2px solid #ddd;
        padding-bottom: 5px;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    
    .article-box {
        padding: 10px;
        margin-bottom: 20px;
        border-bottom: 1px solid #eee;
    }
    
    .footer {
        text-align: center;
        margin-top: 50px;
        padding: 20px;
        background-color: #f9f9f9;
        border-top: 1px solid #ddd;
    }
    
    /* Headline metrics styling */
    .headline-metrics {
        margin-top: 5px;
        font-size: 0.85em;
        background-color: #f8f9fa;
        padding: 4px 8px;
        border-radius: 3px;
    }
    
    .metrics-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 2px;
    }
    
    .metric-label {
        color: #666;
        font-weight: bold;
    }
    
    .metric-value {
        color: #333;
    }
    
    .metric-change {
        font-weight: bold;
    }
    
    .metrics-factors {
        font-style: italic;
        color: #666;
        border-top: 1px dotted #ddd;
        padding-top: 2px;
        margin-top: 2px;
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

def rewrite_headline(client, title, abstract, category=None):
    """Rewrite a single headline using OpenAI with an enhanced prompt"""
    if client is None:
        return title
    
    prompt = f"""You are an expert digital news editor specializing in crafting high-engagement headlines.

HEADLINE REWRITING TASK:
Transform the following news headline into a version that will achieve significantly higher click-through rates while maintaining factual accuracy.

ORIGINAL HEADLINE: "{title}"

ARTICLE ABSTRACT: "{abstract}"

CATEGORY: "{category if category else 'General News'}"

HEADLINE WRITING PRINCIPLES:
1. Use specific, concrete language rather than vague terms
2. Create a curiosity gap that intrigues readers without being misleading
3. Signal value to readers by suggesting what they'll gain from reading
4. Include numbers when relevant (research shows this increases CTR)
5. Use active voice and strong verbs
6. Keep headlines under 70 characters for optimal display
7. Avoid clickbait tactics that would damage credibility

EXAMPLES OF GREAT HEADLINE TRANSFORMATIONS:
Original: "Scientists Discover New Planet That Could Potentially Support Life"
Better: "Earth 2.0: Scientists Find Planet With 95% Match to Our Atmosphere"

Original: "Company Reports Quarterly Earnings Above Expectations"
Better: "Tech Giant Shatters Profit Records as Stock Jumps 7%"

Original: "Study Shows Increased Exercise Linked to Better Cognitive Function"
Better: "30-Minute Daily Walks Boost Brain Function by 32%, Study Reveals"

Your revised headline should be notably more compelling than the original while preserving the core information. Create only ONE headline, not multiple options.

REWRITTEN HEADLINE:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional digital news editor who specializes in writing high-engagement headlines that drive clicks while maintaining journalistic integrity."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=60,
        )
        rewritten = response.choices[0].message.content.strip()
        
        # Remove any quotation marks the model might add
        rewritten = rewritten.replace('"', '').replace('"', '').replace('"', '')
        
        # Check for basic quality control
        if len(rewritten) < 5 or rewritten.lower() in ["", "the new york times"]:
            rewritten = title
            
        return rewritten
    except Exception as e:
        logging.error(f"Error rewriting headline: {e}")
        return title
    
def balance_headline_types(df, client):
    """
    Balance headline types to ensure variety and engagement across different styles.
    Analyzes existing headlines and adds style tags to guide rewriting.
    
    Args:
        df: DataFrame with curated articles
        client: OpenAI client
    
    Returns:
        DataFrame with balanced headline styles
    """
    if len(df) == 0:
        return df
    
    # Copy dataframe to avoid modifying the original
    balanced_df = df.copy()
    
    # Step 1: Analyze current headline distribution
    headline_types = {
        "question": 0,
        "number": 0,
        "how_to": 0,
        "surprising": 0,
        "emotional": 0,
        "direct": 0
    }
    
    # Count existing headline types
    for _, row in balanced_df.iterrows():
        headline = row.get('rewritten_title', '') or row.get('title', '')
        headline_lower = headline.lower()
        
        if "?" in headline:
            headline_types["question"] += 1
        elif any(char.isdigit() for char in headline):
            headline_types["number"] += 1
        elif "how to" in headline_lower or "how you can" in headline_lower:
            headline_types["how_to"] += 1
        elif any(word in headline_lower for word in ["surprising", "unexpected", "shock", "stunned"]):
            headline_types["surprising"] += 1
        elif any(word in headline_lower for word in ["amazing", "incredible", "best", "worst"]):
            headline_types["emotional"] += 1
        else:
            headline_types["direct"] += 1
    
    # Step 2: Determine desired distribution based on total articles
    # Ideal distribution depends on article count
    total = len(balanced_df)
    
    if total <= 5:
        # For small sets, focus on direct and emotional
        target = {
            "question": max(1, int(total * 0.2)),
            "number": max(1, int(total * 0.2)),
            "how_to": 0,
            "surprising": 0,
            "emotional": max(1, int(total * 0.3)),
            "direct": max(1, int(total * 0.3))
        }
    elif total <= 10:
        # For medium sets, more variety
        target = {
            "question": max(1, int(total * 0.2)),
            "number": max(1, int(total * 0.2)),
            "how_to": max(1, int(total * 0.1)),
            "surprising": max(1, int(total * 0.1)),
            "emotional": max(1, int(total * 0.2)),
            "direct": max(1, int(total * 0.2))
        }
    else:
        # For large sets, full variety
        target = {
            "question": max(2, int(total * 0.15)),
            "number": max(3, int(total * 0.25)),
            "how_to": max(1, int(total * 0.1)),
            "surprising": max(2, int(total * 0.15)),
            "emotional": max(2, int(total * 0.15)),
            "direct": max(3, int(total * 0.2))
        }
    
    # Step 3: Identify gaps and rewrite selected headlines
    gaps = {style: max(0, target[style] - headline_types[style]) for style in headline_types}
    
    # Determine which headlines to rewrite and what style to use
    for style, needed in gaps.items():
        if needed <= 0:
            continue
            
        # Find candidates for rewriting (prioritize direct headlines first)
        candidates = []
        for i, row in balanced_df.iterrows():
            headline = row.get('rewritten_title', '') or row.get('title', '')
            headline_lower = headline.lower()
            
            # Skip headlines that already have a strong style
            if ("?" in headline) or \
               ("how to" in headline_lower) or \
               (any(word in headline_lower for word in ["surprising", "unexpected", "shock", "stunned"])):
                continue
                
            # Rank by topic relevance
            relevance = 0
            if style == "question" and "who" in row.get('abstract', '').lower():
                relevance += 2
            elif style == "number" and any(char.isdigit() for char in row.get('abstract', '')):
                relevance += 2
            elif style == "how_to" and row.get('user_need', '') == "Educate":
                relevance += 3
            elif style == "surprising" and row.get('user_need', '') == "Update":
                relevance += 2
            elif style == "emotional" and row.get('user_need', '') in ["Inspire", "Divert"]:
                relevance += 3
            
            candidates.append((i, relevance))
        
        # Sort candidates by relevance
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Rewrite the top candidates
        for i, (idx, _) in enumerate(candidates):
            if i >= needed:
                break
                
            # Add style guidance for rewriting
            style_guidance = ""
            if style == "question":
                style_guidance = "Format as a compelling question that creates curiosity"
            elif style == "number":
                style_guidance = "Include a specific number or statistic from the article"
            elif style == "how_to":
                style_guidance = "Frame as a 'how to' or solution-oriented headline"
            elif style == "surprising":
                style_guidance = "Emphasize the surprising or unexpected aspect"
            elif style == "emotional":
                style_guidance = "Use emotionally powerful language that resonates"
            
            # Store the style guidance for later headline generation
            balanced_df.at[idx, 'headline_style'] = style
            balanced_df.at[idx, 'headline_guidance'] = style_guidance
    
    # Step 4: Apply the guided rewriting
    for i, row in balanced_df.iterrows():
        if 'headline_guidance' in row and pd.notna(row['headline_guidance']):
            try:
                # Enhanced prompt with style guidance
                prompt = f"""Rewrite this headline in a specific style:
                
                Original Headline: {row['title']}
                
                Abstract: {row['abstract']}
                
                Style Guidance: {row['headline_guidance']}
                
                Rewritten Headline:"""
                
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",  # We can use the cheaper model for style variations
                    messages=[
                        {"role": "system", "content": "You are a news editor who specializes in writing engaging headlines."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=60,
                )
                
                rewritten = response.choices[0].message.content.strip()
                # Remove any quotation marks
                rewritten = rewritten.replace('"', '').replace('"', '').replace('"', '')
                
                if len(rewritten) > 5:
                    balanced_df.at[i, 'rewritten_title'] = rewritten
                    
            except Exception as e:
                logging.error(f"Error in headline balancing: {e}")
    
    return balanced_df

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
    
def classify_user_need(title, abstract):
    """User need classification based on content"""
    text = f"{title} {abstract}".lower()
    if any(word in text for word in [
    "explained",
    "analysis",
    "guide",
    "research",
    "investigation",
    "study",
    "insights",
    "perspective",
    "history",
    "breakdown",
    "deep-dive",
    "examination",
    "background",
    "context",
    "tutorial"
]):
        return "Educate"
    elif any(word in text for word in [
    "report",
    "announcement",
    "development",
    "statement",
    "decision",
    "policy",
    "measure",
    "conference",
    "meeting",
    "launch",
    "release",
    "investigation",
    "evidence",
    "survey",
    "statistics"
]):
        return "Inform" 
    elif any(word in text for word in [
    "breaking",
    "alert",
    "latest",
    "update",
    "just in",
    "now",
    "developing",
    "live",
    "ongoing",
    "today",
    "this morning",
    "recent",
    "hourly",
    "fresh",
    "emerging"
]):
        return "Update"
    elif any(word in text for word in [
    "success",
    "breakthrough",
    "overcome",
    "achievement",
    "impact",
    "innovation",
    "initiative",
    "triumph",
    "progress",
    "discovery",
    "leadership",
    "change",
    "vision",
    "community",
    "transformation"
]):
        return "Inspire"
    elif any(word in text for word in [
    "feature",
    "profile",
    "interview",
    "review",
    "lifestyle",
    "culture",
    "celebrity",
    "trends",
    "quiz",
    "humor",
    "oddity",
    "adventure",
    "recreation",
    "hobby",
    "experience"
]):
        return "Divert" 
    else:
        return "Utility"  # fallback default

def curate_articles_for_topic(query_text, index, articles_df, model, openai_client, k=5, progress_bar=None):
    """Find and enhance articles for a given topic"""
    try:
        # Create query embedding and search
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
        topic_articles["user_need"] = topic_articles.apply(
            lambda row: classify_user_need(row["title"], row["abstract"]), axis=1
        )

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
                    openai_client, row['title'], row['abstract'], category=query_text
                )
                topic_articles.at[idx, 'explanation'] = generate_explanation(
                    openai_client, row['title'], row['abstract']
                )
            except Exception as e:
                logging.error(f"Error processing article {idx}: {e}")
                # Use original title as fallback
                topic_articles.at[idx, 'rewritten_title'] = row['title']
                topic_articles.at[idx, 'explanation'] = "This article contains important information relevant to the topic."
        
        # Balance headline styles for better variety
        if total >= 3:  # Only apply balancing when we have enough articles
            try:
                # Apply headline style balancing
                topic_articles = balance_headline_types(topic_articles, openai_client)
                logging.info(f"Applied headline style balancing to {query_text} articles")
            except Exception as e:
                logging.error(f"Error applying headline balancing: {e}")
        
        if progress_bar is not None:
            progress_bar.progress(1.0, text="Processing complete!")
            time.sleep(0.5)  # Give user time to see the completed progress
    
         # Analyze headline effectiveness (add metrics to dataframe)
        try:
            topic_articles = analyze_headline_effectiveness(topic_articles, openai_client)
            logging.info(f"Added headline metrics analysis for {query_text} articles")
        except Exception as e:
            logging.error(f"Error analyzing headline effectiveness: {e}")
        
        return topic_articles

    except Exception as e:
        logging.error(f"Error curating articles: {e}")
        st.error(f"Error while curating articles: {str(e)}")
        return pd.DataFrame()
    
def enforce_user_need_balance(df, quota={"Inspire": 1, "Educate": 2, "Update": 1, "Inform": 2, "Divert": 1}):
    balanced = []
    for need, count in quota.items():
        need_articles = df[df["user_need"] == need].head(count)
        balanced.append(need_articles)
    return pd.concat(balanced).drop_duplicates()

def update_headline_learning(df, topic=None):
    """
    Update the headline learning system with newly processed articles
    
    Args:
        df: DataFrame with processed articles
        topic: Optional topic name
        
    Returns:
        int: Number of headline pairs added
    """
    global headline_learner
    
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return 0
    
    # Check if we have the required metrics columns
    required_columns = ['title', 'rewritten_title', 'headline_score_original', 
                       'headline_score_rewritten', 'headline_improvement']
    
    if not all(col in df.columns for col in required_columns):
        # We need to ensure the metrics are calculated first
        df = analyze_headline_effectiveness(df)
    
    # Add all headlines to the learning system
    count = headline_learner.add_headlines_from_dataframe(df, topic_column='topic')
    
    logging.info(f"Added {count} headline pairs to learning system")
    return count

def get_stock_image_path(topic, article_id=None):
    """
    Return the path to a stock image for the given topic.
    Uses article_id to consistently select the same image for the same article.
    """
    # Map topics to their corresponding image prefixes
    topic_to_prefix = {
        "Top Technology News": "tech",
        "Business Stories": "business",
        "Business and Economy": "business",
        "Global Politics": "politics",
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
            "Business Stories": "#6F42C1",
            "Business and Economy": "#6F42C1",
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
    
def display_headline_with_metrics(original_title, rewritten_title, metrics=None):
    """Display original and rewritten titles with improvement metrics"""
    
    # Base display for headlines
    st.markdown(f"""
    <div class="title-comparison">
        <div class="original-headline">Original: {original_title}</div>
        <div class="rewritten-headline">{rewritten_title}</div>
    """, unsafe_allow_html=True)
    
    # Add metrics if available
    if metrics is not None and isinstance(metrics, dict):
        improvement = metrics.get('headline_improvement', 0)
        ctr_original = metrics.get('headline_ctr_original', 0)
        ctr_rewritten = metrics.get('headline_ctr_rewritten', 0)
        key_factors = metrics.get('headline_key_factors', "")
        
        # Only show metrics if we have improvement data
        if improvement != 0:
            # Color based on improvement
            color = "#28a745" if improvement > 0 else "#dc3545"
            
            st.markdown(f"""
            <div class="headline-metrics">
                <div class="metrics-row">
                    <span class="metric-label">Predicted CTR:</span>
                    <span class="metric-value">{ctr_original:.1f}% → {ctr_rewritten:.1f}%</span>
                    <span class="metric-change" style="color: {color};">
                        {'+' if improvement > 0 else ''}{improvement:.1f}%
                    </span>
                </div>
                <div class="metrics-factors">{key_factors}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def load_curated_articles():
    """Load previously curated articles if available"""
    try:
        if os.path.exists("curated_full_daily_output.csv"):
            return pd.read_csv("curated_full_daily_output.csv")
        return None
    except Exception as e:
        logging.error(f"Error loading curated articles: {e}")
        return None
    
def analyze_headline_effectiveness(df, openai_client=None):
    """
    Analyze the effectiveness of headline rewrites and add metrics to the dataframe
    
    Args:
        df: DataFrame with original and rewritten headlines
        openai_client: Optional OpenAI client for AI-powered analysis
        
    Returns:
        DataFrame with added metrics columns
    """
    if len(df) == 0:
        return df
    
    # Initialize the metrics analyzer
    metrics_analyzer = HeadlineMetrics(client=openai_client)
    
    # Create columns for metrics
    df['headline_score_original'] = 0.0
    df['headline_score_rewritten'] = 0.0
    df['headline_ctr_original'] = 0.0
    df['headline_ctr_rewritten'] = 0.0
    df['headline_improvement'] = 0.0
    df['headline_key_factors'] = ""
    
    # Analyze each headline pair
    for i, row in df.iterrows():
        if pd.isna(row['title']) or pd.isna(row['rewritten_title']):
            continue
            
        try:
            # Get comparison metrics
            comparison = metrics_analyzer.compare_headlines(
                row['title'], 
                row['rewritten_title']
            )
            
            # Store metrics in dataframe
            df.at[i, 'headline_score_original'] = comparison['original_score']
            df.at[i, 'headline_score_rewritten'] = comparison['rewritten_score']
            df.at[i, 'headline_ctr_original'] = comparison['original_ctr'] * 100  # Convert to percentage
            df.at[i, 'headline_ctr_rewritten'] = comparison['rewritten_ctr'] * 100  # Convert to percentage
            df.at[i, 'headline_improvement'] = comparison['score_percent_change']
            
            # Store key improvement factors
            if comparison['key_improvements']:
                df.at[i, 'headline_key_factors'] = ", ".join(comparison['key_improvements'])
            else:
                df.at[i, 'headline_key_factors'] = "No major improvements identified"
                
        except Exception as e:
            logging.error(f"Error analyzing headlines for row {i}: {e}")
    
    return df

def calculate_evaluation_metrics(curated_articles):
    """Calculate metrics for research questions"""
    
    # RQ1 Metrics
    
    # Precision@5 - measure relevance of top 5 articles
    # Since we don't have ground truth, we'll use abstract length and diversity as proxy
    top_5 = curated_articles.head(5)
    precision_at_5 = (top_5['abstract'].str.len() > 100).mean()  # Simple quality metric
    
    # Category diversity score
    categories = curated_articles['topic'].value_counts()
    diversity_score = 1 - (categories.max() / len(curated_articles))  # Higher = more diverse
    
    # User needs coverage
    user_needs = curated_articles['user_need'].value_counts()
    expected_needs = ['Inform', 'Educate', 'Inspire', 'Update', 'Divert', 'Utility']
    needs_coverage = len(user_needs) / len(expected_needs)
    
    # RQ2 Metrics
    
    # Readability improvement (already calculated by HeadlineMetrics)
    readability_improvement = {
        'avg_improvement': curated_articles['headline_improvement'].mean(),
        'positive_improvements': (curated_articles['headline_improvement'] > 0).mean(),
        'avg_ctr_change': (curated_articles['headline_ctr_rewritten'] - curated_articles['headline_ctr_original']).mean()
    }
    
    return {
        'precision@5': precision_at_5,
        'diversity_score': diversity_score,
        'user_needs_coverage': needs_coverage,
        'readability_improvement': readability_improvement,
        'num_articles': len(curated_articles)
    }
    
def generate_research_report(metrics):
    """Generate a report specifically for research questions"""
    
    report = f"""
# Research Evaluation Report
Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}

## RQ1: How effectively can the AI editor retrieve and rank news articles?

### Metrics:
- Precision@5: {metrics['precision@5']:.2%}
- Topic Diversity Score: {metrics['diversity_score']:.2%}
- User Needs Coverage: {metrics['user_needs_coverage']:.2%}

## RQ2: Does rewriting headlines improve readability and engagement?

### Metrics:
- Average CTR Improvement: {metrics['readability_improvement']['avg_improvement']:.1f}%
- Headlines Improved: {metrics['readability_improvement']['positive_improvements']:.1%}
- Average CTR Change: {metrics['readability_improvement']['avg_ctr_change']:.2f}%

## Summary:
Total articles analyzed: {metrics['num_articles']}
"""
    
    # Save report
    with open("research_evaluation_report.md", "w") as f:
        f.write(report)
    
    return report

def save_topic_memory(topics, articles):
    """Enhanced memory for topic continuity"""
    memory_file = "topic_memory.json"
    
    # Load existing memory
    if os.path.exists(memory_file):
        with open(memory_file, "r") as f:
            memory = json.load(f)
    else:
        memory = {"history": []}
    
    # Add current session
    memory["history"].append({
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "topics": topics,
        "article_count": len(articles),
        "avg_ctr_improvement": articles['headline_improvement'].mean() if 'headline_improvement' in articles else 0
    })
    
    # Keep last 30 days
    memory["history"] = memory["history"][-30:]
    
    # Save updated memory
    with open(memory_file, "w") as f:
        json.dump(memory, f, indent=2)

# Refactored Functions for Better Structure
def initialize_session_state():
    """Initialize all session state variables"""
    if 'curation_started' not in st.session_state:
        st.session_state.curation_started = False
    if 'curation_complete' not in st.session_state:
        st.session_state.curation_complete = False
    if 'loaded_articles' not in st.session_state:
        st.session_state.loaded_articles = None
    if 'show_headline_comparison' not in st.session_state:
        st.session_state.show_headline_comparison = True
    if 'evaluation_history' not in st.session_state:
        st.session_state.evaluation_history = []

def setup_sidebar():
    """Setup sidebar with controls and return configuration"""
    with st.sidebar:
        st.header("📰 NEWSPAPER CONTROLS")
        st.write("Configure your daily newspaper")
        st.markdown("---")
        
        # Editorial queries
        editorial_queries = {
            "Top Technology News": "latest breakthroughs in technology and innovation",
            "Business and Economy": "coverage of finance, jobs, inflation, innovation economics",
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
        
        # Store headline comparison preference in session state
        st.session_state.show_headline_comparison = show_headline_comparison
        
        # Additional sidebar sections
        add_learning_system_section()
        add_research_evaluation_section()
        add_image_path_check_section()
        
        return {
            'editorial_queries': editorial_queries,
            'selected_topics': selected_topics,
            'articles_per_topic': articles_per_topic,
            'curate_button': curate_button,
            'load_button': load_button
        }

def add_learning_system_section():
    """Add the headline learning system section to sidebar"""
    st.markdown("---")
    st.subheader("📊 Headline Learning System")
    
    if st.button("Generate Headline Report", use_container_width=True):
        try:
            if headline_learner.prompt_improvement_report():
                st.success("✅ Headline improvement report generated!")
                with open("headline_improvement_report.md", "r") as f:
                    report_content = f.read()
                    
                st.download_button(
                    label="Download Report",
                    data=report_content,
                    file_name="headline_improvement_report.md",
                    mime="text/markdown"
                )
            else:
                st.error("Failed to generate report")
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")

def add_research_evaluation_section():
    """Add the research evaluation section to sidebar"""
    st.markdown("---")
    st.subheader("🔬 Research Evaluation")

    if st.button("Generate Research Report", use_container_width=True):
        try:
            if 'loaded_articles' in st.session_state and st.session_state.loaded_articles is not None:
                metrics = calculate_evaluation_metrics(st.session_state.loaded_articles)
                report = generate_research_report(metrics)
                
                st.success("✅ Research report generated!")
                
                # Display key metrics
                st.metric("Precision@5", f"{metrics['precision@5']:.1%}")
                st.metric("Diversity Score", f"{metrics['diversity_score']:.1%}")
                st.metric("Avg CTR Improvement", f"{metrics['readability_improvement']['avg_improvement']:.1f}%")
                
                # Download button
                st.download_button(
                    label="Download Research Report",
                    data=report,
                    file_name="research_evaluation_report.md",
                    mime="text/markdown"
                )
            else:
                st.warning("Please curate articles first")
        except Exception as e:
            st.error(f"Error generating research report: {str(e)}")

def add_image_path_check_section():
    """Add the image path check section to sidebar"""
    st.markdown("---")
    st.subheader("Image Path Check")
    st.write("This tool checks if your stock images are accessible.")
    
    # Check a few sample paths
    sample_paths = ["tech1.jpg", "tech2.jpg", "business1.jpg", "business2.jpg", "health1.jpg", 
                    "health2.jpg", "politics1.jpg", "politics2.jpg", "climate1.jpg", 
                    "climate2.jpg"]
    for path in sample_paths:
        if os.path.exists(path):
            st.success(f"✓ {path} found")
        else:
            st.error(f"✗ {path} not found - make sure it's in the correct directory")

def handle_article_curation(config):
    """Handle the article curation process"""
    if not config['selected_topics']:
        st.error("Please select at least one topic to curate")
        return
    
    # Initialize session state for progress tracking
    st.session_state.curation_started = True
    st.session_state.curation_complete = False
    
    # Create progress bar
    progress_bar = st.sidebar.progress(0, text="Starting curation process...")
    
    # Load models and data
    progress_bar.progress(0.1, text="Loading models and data...")
    index, articles_df, model = load_models_and_data()
    openai_client = get_openai_client()
    
    if index is None or articles_df is None or model is None:
        st.error("Failed to load required models and data")
        st.session_state.curation_started = False
        return
    
    # Curate articles for each selected topic
    all_curated_articles = []
    
    for i, topic in enumerate(config['selected_topics']):
        topic_progress = (i / len(config['selected_topics'])) * 0.8 + 0.1
        progress_bar.progress(topic_progress, text=f"Curating articles for {topic}...")
        
        query_text = config['editorial_queries'][topic]
        topic_articles = curate_articles_for_topic(
            query_text, index, articles_df, model, openai_client, 
            k=config['articles_per_topic'],
            progress_bar=None
        )
        topic_articles["topic"] = topic
        all_curated_articles.append(topic_articles)
    
    # Combine and process all curated articles
    if all_curated_articles:
        progress_bar.progress(0.9, text="Saving curated articles...")
        full_curated_df = pd.concat(all_curated_articles, ignore_index=True)
        full_curated_df = enforce_user_need_balance(full_curated_df)
        full_curated_df.to_csv("curated_full_daily_output.csv", index=False)
        
        # Update memory for future reference
        save_topics(config['selected_topics'])
        save_topic_memory(config['selected_topics'], full_curated_df)
        
        # Update session state
        st.session_state.loaded_articles = full_curated_df
        st.session_state.curation_complete = True
        
        # Update the headline learning system
        try:
            headline_count = update_headline_learning(full_curated_df)
            if headline_count > 0:
                st.sidebar.success(f"✅ Added {headline_count} headlines to learning system")
        except Exception as e:
            logging.error(f"Error updating headline learning system: {e}")
            st.sidebar.warning("Could not update headline learning system")
        
        # Calculate evaluation metrics (moved to only happen once)
        try:
            metrics = calculate_evaluation_metrics(full_curated_df)
            report = generate_research_report(metrics)
            
            # Store metrics for tracking
            if 'evaluation_history' not in st.session_state:
                st.session_state.evaluation_history = []
            st.session_state.evaluation_history.append(metrics)
            
            st.sidebar.success("✅ Research metrics calculated")
        except Exception as e:
            logging.error(f"Error calculating evaluation metrics: {e}")
        
        progress_bar.progress(1.0, text="Curation complete!")
        st.success("✅ Articles curated successfully!")
    else:
        st.warning("No articles were curated")
        st.session_state.curation_started = False

def display_header_and_navigation():
    """Display the newspaper header and navigation"""
    st.markdown('<div class="header">', unsafe_allow_html=True)
    st.markdown('<h1 class="newspaper-title">THE DAILY AGENT</h1>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation menu
    cols = st.columns(6)
    nav_items = ["POLITICS", "TECH", "BUSINESS", "OPINION", "HEALTH", "CLIMATE"]
    
    for i, item in enumerate(nav_items):
        with cols[i]:
            st.markdown(f'<div class="nav-item">{item}</div>', unsafe_allow_html=True)

def display_featured_article(articles_df, show_comparison):
    """Display the featured article section"""
    st.markdown('<h2 class="section-title">FEATURED NEWS</h2>', unsafe_allow_html=True)
    
    if len(articles_df) > 0:
        main_article = articles_df.iloc[0]
        
        with st.container():
            st.markdown('<div class="article-box">', unsafe_allow_html=True)
            
            # Article tag
            st.markdown(f'<div class="article-tag">{main_article["topic"]}</div>', unsafe_allow_html=True)
            
            # Title comparison
            if show_comparison:
                if 'headline_improvement' in main_article and 'headline_ctr_original' in main_article:
                    display_headline_with_metrics(
                        main_article['original_title'], 
                        main_article['rewritten_title'],
                        {
                            'headline_improvement': main_article.get('headline_improvement', 0),
                            'headline_ctr_original': main_article.get('headline_ctr_original', 0),
                            'headline_ctr_rewritten': main_article.get('headline_ctr_rewritten', 0),
                            'headline_key_factors': main_article.get('headline_key_factors', "")
                        }
                    )
                else:
                    display_headline_comparison(main_article['original_title'], main_article['rewritten_title'])
            else:
                st.subheader(main_article['rewritten_title'])
            
            # Author byline
            author = random.choice(["Sarah Chen", "Michael Johnson", "Priya Patel", "Robert Williams"])
            st.markdown(f'<div class="article-byline">By {author} | {datetime.datetime.now().strftime("%B %d, %Y")}</div>', unsafe_allow_html=True)
            
            # Image
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

def display_trending_articles(articles_df, show_comparison):
    """Display the trending articles section"""
    st.markdown('<h2 class="section-title">TRENDING NOW</h2>', unsafe_allow_html=True)
    
    trend_cols = st.columns(3)
    
    for i in range(1, min(7, len(articles_df))):
        if i < len(articles_df):
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
                
                # Image
                display_article_image(article["topic"], article_id=article.name)
                
                # Short abstract
                abstract = article['abstract']
                if abstract and len(abstract) > 150:
                    abstract = abstract[:150] + "..."
                st.write(abstract)
                
                # Read more link
                st.markdown("**Read More →**")
                
                st.markdown('</div>', unsafe_allow_html=True)

def display_topic_sections(articles_df, show_comparison):
    """Display the topic-specific sections"""
    remaining_articles = articles_df.iloc[7:] if len(articles_df) > 7 else pd.DataFrame()
    
    if len(remaining_articles) > 0:
        topics = remaining_articles['topic'].unique()
        
        for topic in topics:
            st.markdown(f'<h2 class="section-title">{topic.upper()}</h2>', unsafe_allow_html=True)
            
            topic_articles = remaining_articles[remaining_articles['topic'] == topic]
            
            cols = st.columns(2)
            
            for i, (_, article) in enumerate(topic_articles.iterrows()):
                with cols[i % 2]:
                    st.markdown('<div class="article-box">', unsafe_allow_html=True)
                    
                    # Title comparison
                    if show_comparison:
                        display_headline_comparison(article['original_title'], article['rewritten_title'])
                    else:
                        st.markdown(f"### {article['rewritten_title']}")
                    
                    # Image
                    display_article_image(article["topic"], article_id=article.name)
                    
                    # Short abstract
                    abstract = article['abstract']
                    if abstract and len(abstract) > 100:
                        abstract = abstract[:100] + "..."
                    st.write(abstract)
                    
                    # Read more link
                    st.markdown("**Read More →**")
                    
                    st.markdown('</div>', unsafe_allow_html=True)

def display_footer():
    """Display the footer section"""
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.markdown('<h3>THE DAILY AGENT</h3>', unsafe_allow_html=True)
    st.markdown(f'<p>© {datetime.datetime.now().year} The Daily Agent. All Rights Reserved.</p>', unsafe_allow_html=True)
    st.markdown('<p>Powered by AI-curated content</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def display_newspaper_layout(articles_df, show_comparison):
    """Display the complete newspaper layout"""
    display_header_and_navigation()
    display_featured_article(articles_df, show_comparison)
    display_trending_articles(articles_df, show_comparison)
    display_topic_sections(articles_df, show_comparison)
    display_footer()

def display_welcome_page():
    """Display the welcome page when no articles are loaded"""
    if 'curation_started' in st.session_state and st.session_state.curation_started and not st.session_state.get('curation_complete', False):
        st.info("⏳ Curation in progress... Please wait while we prepare your personalized news articles.")
    else:
        st.header("Welcome to The Daily Agent")
        st.write("Your AI-powered personalized news platform")
        
        st.subheader("Get Started in Two Simple Steps:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 1. Select Topics")
            st.info("Choose which news categories you want to include using the sidebar controls")
            try:
                if os.path.exists("tech1.jpg"):
                    st.image("tech1.jpg", width=300)
            except:
                pass
            
        with col2:
            st.markdown("### 2. Curate Articles")
            st.success("Click 'CURATE FRESH ARTICLES' to generate your personalized newspaper")
            try:
                if os.path.exists("health1.jpg"):
                    st.image("health1.jpg", width=300)
            except:
                pass
        
        st.subheader("YOUR PERSONALIZED NEWSPAPER")
        st.write("The Daily Agent will create a professional newspaper using your selected topics and AI-curated content.")
        
        st.markdown("""
        <div style="border: 1px solid #ddd; padding: 20px; border-radius: 5px; text-align: center; margin: 20px 0; background-color: #f9f9f9;">
            <h3 style="margin-bottom: 15px;">Sample Layout Preview</h3>
            <p>Your newspaper will include featured articles, trending stories, and topic-specific sections with images.</p>
        </div>
        """, unsafe_allow_html=True)

# --- Main Application Logic ---
def main():
    # Initialize session state
    initialize_session_state()
    
    # Setup sidebar and get configuration
    config = setup_sidebar()
    
    # Handle curation button
    if config['curate_button']:
        handle_article_curation(config)
    
    # Handle load button
    if config['load_button']:
        with st.spinner("Loading previously curated articles..."):
            st.session_state.loaded_articles = load_curated_articles()
            if st.session_state.loaded_articles is not None:
                st.success(f"Loaded {len(st.session_state.loaded_articles)} articles")
            else:
                st.error("No previously curated articles found")
    
    # Display the main content
    if 'loaded_articles' in st.session_state and st.session_state.loaded_articles is not None and len(st.session_state.loaded_articles) > 0:
        display_newspaper_layout(
            st.session_state.loaded_articles, 
            st.session_state.get('show_headline_comparison', True)
        )
    else:
        display_welcome_page()

if __name__ == "__main__":
    main()