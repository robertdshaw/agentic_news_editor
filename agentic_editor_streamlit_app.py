import streamlit as st
import pandas as pd

# --- Load Data ---
st.set_page_config(page_title="Daily Curated News", layout="wide")
st.title("🗞️ Agentic AI News Editor - Daily Curated Front Page")

try:
    curated_df = pd.read_csv("curated_full_daily_output.csv")
except FileNotFoundError:
    st.error("No curated daily output found. Please run the daily curation script first.")
    st.stop()

# --- Display Articles by Topic ---
topics = curated_df["topic"].unique()

for topic in topics:
    st.header(f"📚 {topic}")
    topic_df = curated_df[curated_df["topic"] == topic]
    
    for idx, row in topic_df.iterrows():
        st.subheader(row["rewritten_title"])
        st.caption(f"Category: {row['category']}")
        st.write(row["explanation"])
        with st.expander("See original article abstract"):
            st.write(row["abstract"])
    
    st.divider()

st.success("✅ Front page loaded successfully!")
