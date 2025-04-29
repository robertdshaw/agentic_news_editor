# app_frontpage.py

import streamlit as st
import pandas as pd

# --- Load curated daily articles ---
@st.cache_data
def load_curated_articles(filepath="curated_full_daily_output.csv"):
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        st.error("No curated articles found. Please run the daily curation script first.")
        return pd.DataFrame()

# Load the data
curated_df = load_curated_articles()

# --- Streamlit App Interface ---
st.set_page_config(page_title="Agentic AI News Editor", page_icon="🗞️", layout="wide")

st.title("🗞️ Agentic AI News Editor - Daily Curated Front Page")

if curated_df.empty:
    st.stop()

# Allow user to select topic
selected_topic = st.selectbox("Select a Topic:", curated_df["topic"].unique())

# Filter articles by selected topic
filtered_articles = curated_df[curated_df["topic"] == selected_topic]

# Show articles
for idx, row in filtered_articles.iterrows():
    st.subheader(row["rewritten_title"])
    st.write(f"**Category:** {row['category']}")
    st.write(f"🧠 **Why it matters:** {row['explanation']}")
    
    with st.expander("See original article abstract"):
        st.write(row["abstract"])
    
    st.markdown("---")
