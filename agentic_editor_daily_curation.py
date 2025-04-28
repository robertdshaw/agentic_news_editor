# --- 1. Imports ---
import pandas as pd
import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# --- 2. Load FAISS index, articles metadata, embedding model, and LLM ---
print(" Loading index, metadata, and models...")
index = faiss.read_index("articles_faiss.index")
articles_df = pd.read_csv("articles_with_embeddings.csv")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
rewrite_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")

# --- 3. Define editorial queries ---
editorial_queries = {
    "Top Technology News": "latest breakthroughs in technology and innovation",
    "Inspiring Stories": "positive and uplifting news stories",
    "Global Politics": "latest news about world politics and diplomacy",
    "Climate and Environment": "climate change news and environment protection",
    "Health and Wellness": "advances in healthcare and medical discoveries"
}

# --- 4. Load yesterday's topics for memory (optional) ---
def load_topics(filename="memory_topics.json"):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_topics(topics, filename="memory_topics.json"):
    with open(filename, "w") as f:
        json.dump(topics, f)

yesterday_topics = load_topics()
today_topics = list(editorial_queries.keys())
fresh_topics = [t for t in today_topics if t not in yesterday_topics]

print(f" Fresh topics to curate today: {fresh_topics}")

# --- 5. Retrieval, Headline Rewriting, Explanation Generation ---
def rewrite_headline(title, abstract):
    prompt = f"Rewrite the news headline to be more engaging and SEO-friendly:\n\nTitle: {title}\n\nAbstract: {abstract}\n\nRewritten Headline:"
    response = rewrite_pipeline(prompt, max_length=30, do_sample=False)
    return response[0]['generated_text']

def generate_explanation(title, abstract):
    prompt = f"Explain in one sentence why this news article is important to readers:\n\nTitle: {title}\n\nAbstract: {abstract}\n\nExplanation:"
    response = rewrite_pipeline(prompt, max_length=40, do_sample=False)
    return response[0]['generated_text']

all_curated_articles = []

for topic, query_text in editorial_queries.items():
    if topic not in fresh_topics:
        continue  # Skip topics already curated yesterday
    
    print(f"\n Curating articles for: {topic}")

    # Embed query
    query_embedding = model.encode([query_text])

    # Retrieve top 5 articles
    D, I = index.search(np.array(query_embedding), k=5)
    topic_articles = articles_df.iloc[I[0]].copy()

    # Headline Rewriting
    topic_articles["rewritten_title"] = topic_articles.apply(
        lambda row: rewrite_headline(row["title"], row["abstract"]), axis=1
    )

    # Explanation Generation
    topic_articles["explanation"] = topic_articles.apply(
        lambda row: generate_explanation(row["title"], row["abstract"]), axis=1
    )

    # Assign topic
    topic_articles["topic"] = topic

    # Save topic-specific CSV
    safe_topic = topic.replace(" ", "_").replace("/", "_").lower()
    filename = f"retrieved_{safe_topic}.csv"
    topic_articles.to_csv(filename, index=False)
    print(f" Saved curated articles for {topic} to {filename}")

    all_curated_articles.append(topic_articles)

# --- 6. Save all articles together if needed ---
if all_curated_articles:
    full_curated_df = pd.concat(all_curated_articles, ignore_index=True)
    full_curated_df.to_csv("curated_full_daily_output.csv", index=False)
    print("\n Saved full curated daily output to curated_full_daily_output.csv")
else:
    print("\n No new topics curated today.")

# --- 7. Update memory ---
save_topics(today_topics)
print("✅ Memory updated for tomorrow.")
