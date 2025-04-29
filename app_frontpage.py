import streamlit as st
import pandas as pd

# --- 1. Set page config FIRST ---
st.set_page_config(page_title="Agentic AI News Editor", layout="wide")

# --- 2. Load curated articles ---
@st.cache_data
def load_curated_articles():
    try:
        return pd.read_csv("curated_full_daily_output.csv")
    except FileNotFoundError:
        st.error("No curated articles found. Run the curation script first.")
        return pd.DataFrame()

curated_df = load_curated_articles()

if curated_df.empty:
    st.stop()

# --- 3. Front page title ---
st.title("🗞️ Agentic AI News Editor - Today's Front Page")
st.markdown("##### Curated daily using AI rewriting, editorial explanations, and topic curation.")

st.markdown("---")

# --- 4. Group by editorial topics ---
topics = curated_df["topic"].unique()

for topic in topics:
    st.subheader(f"📚 {topic}")
    
    topic_articles = curated_df[curated_df["topic"] == topic]

    # Arrange articles in a 2-column layout
    cols = st.columns(2)

    for idx, row in topic_articles.iterrows():
        with cols[idx % 2]:  # Alternate articles between two columns
            with st.container():
                st.markdown(f"### 📰 {row['rewritten_title']}")
                
                # Small subtitle info
                st.caption(f"**Category:** {row['category']}  |  **Original Headline:** {row['title']}")
                
                # Important highlight
                st.markdown(f"**🧠 Why It Matters:** {row['explanation']}")

                # Abstract inside an expandable block
                with st.expander("📖 Read Abstract"):
                    st.markdown(f"{row['abstract']}")

                st.markdown("---")

# --- 5. Footer ---
st.markdown("#### 🛠️ Powered by Streamlit + OpenAI | Curated automatically every 24 hours.")
