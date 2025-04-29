import streamlit as st
import pandas as pd
import random

# --- 1. Set up page ---
st.set_page_config(page_title="Agentic Daily - Curated for You!", layout="wide")

# --- 2. Load data ---
try:
    df = pd.read_csv("curated_full_daily_output.csv")
except FileNotFoundError:
    st.error("❌ No curated articles found. Run the daily curation script first.")
    st.stop()

# --- 3. Prepare article images ---
# Example list of placeholder news images
placeholder_images = [
    "https://source.unsplash.com/featured/?news",
    "https://source.unsplash.com/featured/?newspaper",
    "https://source.unsplash.com/featured/?journalism",
    "https://source.unsplash.com/featured/?city,skyline",
    "https://source.unsplash.com/featured/?technology",
    "https://source.unsplash.com/featured/?politics",
    "https://source.unsplash.com/featured/?climate",
    "https://source.unsplash.com/featured/?health",
    "https://source.unsplash.com/featured/?environment",
]

# --- 4. Sidebar Filters ---
st.sidebar.title("🧠 Customize Your Front Page")
selected_topic = st.sidebar.selectbox(
    "Choose a section:", 
    options=["All Topics"] + sorted(df["topic"].unique())
)

# Filter articles based on selected topic
if selected_topic != "All Topics":
    filtered_df = df[df["topic"] == selected_topic]
else:
    filtered_df = df.copy()

# --- 5. Title and Intro ---
st.title("🗞 Agentic Daily - Curated for You!")
st.caption("Your personalized, AI-curated front page. Freshly updated daily!")

# --- 6. Main Front Page Layout ---
for idx, row in filtered_df.iterrows():
    with st.container():
        cols = st.columns([2, 5])  # small image column + larger text column

        # Random image for placeholder
        image_url = random.choice(placeholder_images)
        cols[0].image(image_url, use_column_width=True)

        with cols[1]:
            st.subheader(row["rewritten_title"])
            st.write(f"**Original Headline:** {row['title']}")
            st.write(f"*Category: {row['topic']}*")
            st.markdown(f"> {row['explanation']}")

        st.markdown("---")  # separator between articles

# --- 7. Footer ---
st.write("")
st.markdown(
    "<center><small>© 2025 Agentic Daily - Powered by AI News Curation</small></center>", 
    unsafe_allow_html=True
)
