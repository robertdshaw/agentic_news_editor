import streamlit as st
import pandas as pd
import datetime
import random
from datetime import date

# --- 1. Set up page configuration ---
st.set_page_config(
    page_title="The Daily Chronicle", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. Load data ---
try:
    df = pd.read_csv("curated_full_daily_output.csv")
    # Create a 'date' column if it doesn't exist
    if 'date' not in df.columns:
        # Generate random dates within the last week
        today = date.today()
        df['date'] = [today - datetime.timedelta(days=random.randint(0, 6)) for _ in range(len(df))]
except FileNotFoundError:
    st.error("❌ No curated articles found. Run the daily curation script first.")
    st.stop()

# --- 3. Image placeholders by topic ---
topic_image_links = {
    "Top Technology News": "https://source.unsplash.com/600x400/?technology,computer",
    "Inspiring Stories": "https://source.unsplash.com/600x400/?inspiration,hope",
    "Global Politics": "https://source.unsplash.com/600x400/?politics,government",
    "Climate and Environment": "https://source.unsplash.com/600x400/?climate,nature",
    "Health and Wellness": "https://source.unsplash.com/600x400/?health,wellness",
    # Add defaults for any other topics
    "default": "https://source.unsplash.com/600x400/?news"
}

# --- 4. Custom Styling with advanced newspaper design ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Source+Serif+Pro:wght@400;600;700&family=Lora:ital,wght@0,400;0,700;1,400&display=swap');
        
        /* Global styles */
        html, body, .stApp {
            font-family: 'Source Serif Pro', serif;
            color: #333333;
            background-color: #f9f7f1;
        }
        
        /* Masthead */
        .newspaper-masthead {
            font-family: 'Playfair Display', serif;
            text-align: center;
            border-bottom: 2px solid #000;
            margin-bottom: 10px;
            padding-bottom: 10px;
        }
        
        .newspaper-name {
            font-size: 72px;
            font-weight: 900;
            letter-spacing: -1px;
            margin: 0;
            padding: 0;
            line-height: 1;
        }
        
        .newspaper-date {
            font-family: 'Source Serif Pro', serif;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-top: 5px;
            margin-bottom: 5px;
        }
        
        .newspaper-motto {
            font-style: italic;
            font-size: 16px;
            border-top: 1px solid #999;
            border-bottom: 1px solid #999;
            padding: 5px 0;
            width: 80%;
            margin: 5px auto;
        }
        
        /* Section headers */
        .section-header {
            font-family: 'Playfair Display', serif;
            font-size: 22px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            border-bottom: 2px solid #000;
            margin-top: 20px;
            margin-bottom: 15px;
            padding-bottom: 3px;
        }
        
        /* Article styling */
        .article-container {
            background-color: #f9f7f1;
            margin-bottom: 25px;
            border-bottom: 1px solid #ddd;
            padding-bottom: 15px;
        }
        
        .article-headline {
            font-family: 'Playfair Display', serif;
            font-weight: 700;
            font-size: 26px;
            line-height: 1.2;
            margin-top: 0;
            margin-bottom: 10px;
        }
        
        .article-lead {
            font-family: 'Lora', serif;
            font-size: 16px;
            line-height: 1.5;
            color: #444;
        }
        
        .article-meta {
            font-family: 'Source Serif Pro', serif;
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }
        
        .article-content {
            font-size: 15px;
            line-height: 1.6;
            color: #333;
        }
        
        /* Featured article */
        .featured-article .article-headline {
            font-size: 32px;
        }
        
        .featured-article .article-lead {
            font-size: 18px;
            font-weight: 600;
        }
        
        /* Pull quote */
        .pull-quote {
            font-family: 'Playfair Display', serif;
            font-size: 20px;
            line-height: 1.4;
            font-style: italic;
            padding: 15px 20px;
            margin: 15px 0;
            border-left: 4px solid #000;
            background-color: #f0ece3;
        }
        
        /* Topic tag */
        .topic-tag {
            display: inline-block;
            background-color: #333;
            color: white !important;
            padding: 3px 8px;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            border-radius: 3px;
            margin-right: 5px;
        }
        
        /* Weather box */
        .weather-box {
            background-color: #e9e5dc;
            border: 1px solid #ccc;
            padding: 10px;
            text-align: center;
            font-size: 14px;
            margin-bottom: 15px;
        }
        
        /* Sidebar customization */
        .css-1d391kg, .css-1lcbmhc {
            background-color: #f0ece3;
        }
        
        /* Custom column divider */
        .column-divider {
            border-left: 1px solid #ddd;
            height: 100%;
            margin: 0 15px;
        }
        
        /* Advertisement block */
        .advertisement {
            background-color: #f0f0f0;
            text-align: center;
            padding: 10px;
            margin: 15px 0;
            border: 1px solid #ddd;
            font-family: 'Source Serif Pro', serif;
        }
        
        /* Index/Table of contents */
        .index-table {
            background-color: #f0ece3;
            padding: 10px 15px;
            font-size: 14px;
            margin-bottom: 20px;
        }
        
        .index-header {
            font-family: 'Playfair Display', serif;
            font-weight: 700;
            font-size: 16px;
            border-bottom: 1px solid #999;
            margin-bottom: 8px;
            padding-bottom: 3px;
        }
        
        /* Read more links */
        .read-more {
            font-family: 'Source Serif Pro', serif;
            font-weight: 600;
            font-size: 14px;
            color: #444;
            text-decoration: none;
            display: inline-block;
            margin-top: 5px;
        }
        
        /* Streamlit element customization */
        .stButton>button {
            background-color: #333;
            color: white;
            font-family: 'Source Serif Pro', serif;
            border: none;
        }
        
        .stButton>button:hover {
            background-color: #555;
        }
        
        /* Hide Streamlit's default elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .css-1avcm0n {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 5. Sidebar ---
with st.sidebar:
    st.markdown('<h2 style="font-family: \'Playfair Display\', serif;">📰 Reader Options</h2>', unsafe_allow_html=True)
    
    selected_topic = st.selectbox(
        "Filter by Section:", 
        options=["All Topics"] + sorted(df["topic"].unique())
    )
    
    search_query = st.text_input("🔎 Search articles", "")
    
    st.markdown("---")
    
    articles_per_page = st.slider("Articles per page", min_value=3, max_value=10, value=5)
    page_number = st.number_input("Page", min_value=1, step=1, value=1)
    
    st.markdown("---")
    
    # Weather widget
    st.markdown("""
        <div class="weather-box">
            <strong>TODAY'S WEATHER</strong><br>
            ☀️ 72°F / 22°C<br>
            Sunny with light clouds
        </div>
    """, unsafe_allow_html=True)
    
    # Quick index
    st.markdown("""
        <div class="index-table">
            <div class="index-header">IN THIS EDITION</div>
            • Top Technology News<br>
            • Inspiring Stories<br>
            • Global Politics<br>
            • Climate and Environment<br>
            • Health and Wellness
        </div>
    """, unsafe_allow_html=True)
    
    # Advertisement placeholder
    st.markdown("""
        <div class="advertisement">
            <strong>ADVERTISEMENT</strong><br>
            <small>Support quality journalism</small>
        </div>
    """, unsafe_allow_html=True)

# --- 6. Filter articles ---
if selected_topic != "All Topics":
    filtered_df = df[df["topic"] == selected_topic]
else:
    filtered_df = df.copy()

# Apply search filter if query exists
if search_query:
    filtered_df = filtered_df[
        filtered_df["title"].str.contains(search_query, case=False) | 
        filtered_df["summary"].str.contains(search_query, case=False)
    ]

# --- 7. Newspaper Masthead ---
today = datetime.datetime.now()
day_of_week = today.strftime("%A")
date_str = today.strftime("%B %d, %Y")
st.markdown(f"""
    <div class="newspaper-masthead">
        <div class="newspaper-name">THE DAILY CHRONICLE</div>
        <div class="newspaper-date">{day_of_week.upper()} • {date_str}</div>
        <div class="newspaper-motto">"Delivering Truth, Inspiring Minds"</div>
    </div>
""", unsafe_allow_html=True)

# --- 8. Breaking News Banner (if applicable) ---
breaking_news = random.choice([True, False])
if breaking_news and len(filtered_df) > 0:
    breaking_article = filtered_df.sample(1).iloc[0]
    st.markdown(f"""
        <div style="background-color: #D42424; color: white; padding: 10px; margin-bottom: 20px; text-align: center;">
            <span style="font-family: 'Source Serif Pro', serif; font-weight: 700; font-size: 16px;">BREAKING NEWS:</span> 
            <span style="font-family: 'Lora', serif;">{breaking_article['title']}</span>
        </div>
    """, unsafe_allow_html=True)

# --- 9. Main Content Layout ---
# Calculate pagination
start_idx = (page_number - 1) * articles_per_page
end_idx = start_idx + articles_per_page
paginated_df = filtered_df.iloc[start_idx:end_idx].reset_index(drop=True)

if len(paginated_df) == 0:
    st.warning("No articles match your criteria. Try adjusting your filters.")
else:
    # Front Page Layout
    if page_number == 1:
        # Header for current section
        if selected_topic != "All Topics":
            st.markdown(f'<div class="section-header">{selected_topic}</div>', unsafe_allow_html=True)
        
        # Featured article (first row)
        featured_article = paginated_df.iloc[0]
        with st.container():
            cols = st.columns([3, 2])
            with cols[0]:
                # Featured Article Content
                topic = featured_article['topic']
                img_url = topic_image_links.get(topic, topic_image_links['default'])
                
                st.markdown(f"""
                    <div class="article-container featured-article">
                        <div class="article-meta">
                            <span class="topic-tag">{topic}</span> {featured_article.get('date', date_str)}
                        </div>
                        <h1 class="article-headline">{featured_article['title']}</h1>
                        <p class="article-lead">{featured_article['summary'][:150]}...</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Create a pull quote from the article
                if 'summary' in featured_article and len(featured_article['summary']) > 200:
                    quote_start = random.randint(50, min(150, len(featured_article['summary'])-50))
                    quote_end = min(quote_start + 100, len(featured_article['summary']))
                    pull_quote = featured_article['summary'][quote_start:quote_end]
                    
                    st.markdown(f"""
                        <div class="pull-quote">
                            "{pull_quote}..."
                        </div>
                    """, unsafe_allow_html=True)
                
                # Continue with article content
                if 'content' in featured_article and featured_article['content']:
                    content = featured_article['content']
                    # Display first few paragraphs
                    paragraphs = content.split('\n\n')[:2]
                    content_display = '\n\n'.join(paragraphs)
                    
                    st.markdown(f"""
                        <div class="article-content">
                            {content_display}... 
                            <span class="read-more">Continue reading →</span>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    # If no content, show more of the summary
                    st.markdown(f"""
                        <div class="article-content">
                            {featured_article['summary']}
                            <span class="read-more">Read more →</span>
                        </div>
                    """, unsafe_allow_html=True)
            
            with cols[1]:
                # Featured Article Image
                st.image(img_url, use_column_width=True)
                
                # Byline or source info
                st.markdown(f"""
                    <div style="text-align: right; font-size: 12px; color: #666; margin-top: 5px;">
                        Photo: Unsplash • By {featured_article.get('author', 'Staff Reporter')}
                    </div>
                """, unsafe_allow_html=True)
                
                # Additional short news items
                st.markdown('<div class="section-header" style="font-size: 18px;">IN BRIEF</div>', unsafe_allow_html=True)
                
                # Get a few random articles for the brief section
                if len(df) >= 3:
                    brief_articles = df.sample(3)
                    for _, brief in brief_articles.iterrows():
                        st.markdown(f"""
                            <div style="margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px dotted #ccc;">
                                <div style="font-family: 'Playfair Display', serif; font-weight: 700; font-size: 15px;">
                                    {brief['title']}
                                </div>
                                <div style="font-size: 13px; color: #444; margin-top: 5px;">
                                    {brief['summary'][:80]}...
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
        
        # Divider
        st.markdown("<hr style='margin: 30px 0; border-color: #ddd;'>", unsafe_allow_html=True)
        
        # Second row - Article grid (2x2)
        if len(paginated_df) > 1:
            remaining_articles = paginated_df.iloc[1:min(5, len(paginated_df))]
            
            # Create rows of 2 articles each
            for i in range(0, len(remaining_articles), 2):
                cols = st.columns(2)
                
                for j in range(2):
                    if i+j < len(remaining_articles):
                        article = remaining_articles.iloc[j]
                        topic = article['topic']
                        img_url = topic_image_links.get(topic, topic_image_links['default'])
                        
                        with cols[j]:
                            # Display image
                            st.image(img_url, use_column_width=True)
                            
                            # Article content
                            st.markdown(f"""
                                <div class="article-container">
                                    <div class="article-meta">
                                        <span class="topic-tag">{topic}</span> {article.get('date', date_str)}
                                    </div>
                                    <h2 class="article-headline" style="font-size: 22px;">{article['title']}</h2>
                                    <p class="article-lead" style="font-size: 15px;">{article['summary'][:120]}...</p>
                                </div>
                            """, unsafe_allow_html=True)
            
            # Advertisement row
            st.markdown("""
                <div class="advertisement" style="padding: 15px; margin: 30px 0;">
                    <strong style="font-size: 16px;">ADVERTISEMENT</strong><br>
                    <p style="margin: 10px 0;">Support quality journalism with a subscription to The Daily Chronicle</p>
                    <div style="background-color: #333; color: white; padding: 5px 10px; display: inline-block;">SUBSCRIBE NOW</div>
                </div>
            """, unsafe_allow_html=True)
    
    # Regular article pages
    else:
        # Header for current section
        if selected_topic != "All Topics":
            st.markdown(f'<div class="section-header">{selected_topic}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="section-header">Continued from Front Page</div>', unsafe_allow_html=True)
        
        # Display articles in a 3-column layout
        for i in range(0, len(paginated_df), 3):
            cols = st.columns(3)
            
            for j in range(3):
                if i+j < len(paginated_df):
                    article = paginated_df.iloc[i+j]
                    topic = article['topic']
                    img_url = topic_image_links.get(topic, topic_image_links['default'])

                    with cols[j]:
                        # Display image
                        st.image(img_url, use_column_width=True)
                        
                        # Article content
                        st.markdown(f"""
                            <div class="article-container">
                                <div class="article-meta">
                                    <span class="topic-tag">{topic}</span> {article.get('date', date_str)}
                                </div>
                                <h2 class="article-headline" style="font-size: 20px;">{article['title']}</h2>
                                <p class="article-lead" style="font-size: 14px;">{article['summary'][:100]}...</p>
                                <div class="article-content" style="font-size: 14px;">
                                    {article['summary'][100:200]}...
                                    <span class="read-more">Continue reading →</span>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

# --- 10. Pagination Controls ---
st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
cols = st.columns([1, 1, 1])

with cols[0]:
    if page_number > 1:
        if st.button("← Previous Page"):
            page_number -= 1
            st.experimental_rerun()

with cols[1]:
    total_pages = (len(filtered_df) + articles_per_page - 1) // articles_per_page
    st.markdown(f"<div style='text-align: center;'>Page {page_number} of {total_pages}</div>", unsafe_allow_html=True)

with cols[2]:
    if page_number < total_pages:
        if st.button("Next Page →"):
            page_number += 1
            st.experimental_rerun()

# --- 11. Footer ---
st.markdown("<hr style='margin: 30px 0; border-color: #ddd;'>", unsafe_allow_html=True)
footer_cols = st.columns(3)

with footer_cols[0]:
    st.markdown("""
        <div style="font-family: 'Source Serif Pro', serif; font-size: 14px;">
            <strong>THE DAILY CHRONICLE</strong><br>
            123 News Street<br>
            City, State 12345
        </div>
    """, unsafe_allow_html=True)

with footer_cols[1]:
    st.markdown("""
        <div style="font-family: 'Source Serif Pro', serif; font-size: 14px; text-align: center;">
            <strong>SECTIONS</strong><br>
            Technology • Politics • Health<br>
            Environment • Business • Sports
        </div>
    """, unsafe_allow_html=True)

with footer_cols[2]:
    st.markdown("""
        <div style="font-family: 'Source Serif Pro', serif; font-size: 14px; text-align: right;">
            <strong>FOLLOW US</strong><br>
            Twitter • Facebook • Instagram<br>
            © 2025 The Daily Chronicle
        </div>
    """, unsafe_allow_html=True)