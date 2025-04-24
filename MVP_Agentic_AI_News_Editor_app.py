import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# 1) Load & sample, dropping NaNs
@st.cache_data(show_spinner=False)
def load_and_sample(path: str, sample_size: int = 50000, seed: int = 42) -> pd.DataFrame:
    cols = [
        "newsID", "category", "subcategory",
        "title", "abstract", "url",
        "title_entities", "abstract_entities"
    ]
    df = pd.read_csv(path, sep="\t", header=None, names=cols)
    # Drop any rows missing title or abstract
    df = df.dropna(subset=["title", "abstract"])
    n = min(sample_size, len(df))
    return df.sample(n, random_state=seed)[["title", "abstract"]]

# Load & sample
sample_df = load_and_sample("news.tsv", sample_size=500)
st.write(f"Loaded & sampled {len(sample_df)} articles (no NaNs)")

# 2) Build and cache the FAISS index
@st.cache_data(show_spinner=False)
def build_index(texts: list[str]):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True, batch_size=64)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return model, index

# Prepare texts as a list of strings
texts = [
    f"{row.title} {row.abstract}"
    for row in sample_df.itertuples(index=False)
]
model, index = build_index(texts)

# 3) Streamlit UI
st.title("MVP RA(G)‑like News Bot")
st.write(f"Searching across {len(sample_df)} sampled articles")

query = st.text_input("Ask me anything about the news")
if query:
    # Encode the single query as a list
    q_emb_list = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(q_emb_list.reshape(1, -1), k=3)
    for rank, idx in enumerate(indices[0], start=1):
        title = sample_df.iloc[idx]["title"]
        abstract = sample_df.iloc[idx]["abstract"]
        st.subheader(f"{rank}. {title}")
        st.write(abstract)
