import os
import json
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

# --- 1. Load environment variables ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("❌ OPENAI_API_KEY is missing. Check your .env file!")

client = OpenAI(api_key=openai_api_key)

# --- 2. Load FAISS index, metadata, and embedding model ---
print("🔵 Loading index, metadata, and models...")
index = faiss.read_index("articles_faiss.index")
articles_df = pd.read_csv("articles_with_embeddings.csv")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# --- 3. Define editorial queries ---
editorial_queries = {
    "Top Technology News": "latest breakthroughs in technology and innovation",
    "Inspiring Stories": "positive and uplifting news stories",
    "Global Politics": "latest news about world politics and diplomacy",
    "Climate and Environment": "climate change news and environment protection",
    "Health and Wellness": "advances in healthcare and medical discoveries"
}

# --- 4. Load yesterday's topics for memory ---
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

print(f"🟠 Fresh topics to curate today: {fresh_topics}")

# --- 5. Helper functions for GPT ---
def batch_rewrite_headlines(titles, abstracts):
    responses = []
    for title, abstract in zip(titles, abstracts):
        prompt = f"""You are an expert news editor.

Your task is to rewrite the following news headline to be more engaging, SEO-optimized, and still factually accurate based on the article abstract.

Use clear, active language and keep it under 15 words.

---

Title: {title}

Abstract: {abstract}

Rewritten Headline:"""

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
        responses.append(rewritten)
    return responses

def batch_generate_explanations(titles, abstracts):
    responses = []
    for title, abstract in zip(titles, abstracts):
        prompt = f"""You are an editorial assistant.

Write one sentence explaining why the following news article is important to readers.

Focus on clarity and importance for a general audience.

---

Title: {title}

Abstract: {abstract}

Explanation:"""

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
        responses.append(explanation)
    return responses

# --- 6. Retrieval and curation ---
all_curated_articles = []

for topic, query_text in editorial_queries.items():
    if topic not in fresh_topics:
        continue

    print(f"\n🟢 Curating articles for: {topic}")

    query_embedding = model.encode([query_text])
    D, I = index.search(np.array(query_embedding), k=5)
    topic_articles = articles_df.iloc[I[0]].copy()

    titles = topic_articles["title"].tolist()
    abstracts = topic_articles["abstract"].tolist()

    rewritten_titles = batch_rewrite_headlines(titles, abstracts)
    explanations = batch_generate_explanations(titles, abstracts)

    topic_articles["rewritten_title"] = rewritten_titles
    topic_articles["explanation"] = explanations
    topic_articles["topic"] = topic

    safe_topic = topic.replace(" ", "_").replace("/", "_").lower()
    filename = f"retrieved_{safe_topic}.csv"
    topic_articles.to_csv(filename, index=False)

    print(f"💾 Saved curated articles for '{topic}' to {filename}")

    all_curated_articles.append(topic_articles)

# --- 7. Save full daily curation output ---
if all_curated_articles:
    full_curated_df = pd.concat(all_curated_articles, ignore_index=True)
    full_curated_df.to_csv("curated_full_daily_output.csv", index=False)
    print("\n✅ Saved full curated daily output to curated_full_daily_output.csv")

    print("\n🎯 Sample curated articles:")
    print(full_curated_df[["topic", "title", "rewritten_title", "explanation"]].head(20))
else:
    print("\nℹ No fresh topics today. Nothing new to save.")

# --- 8. Update memory for tomorrow ---
save_topics(today_topics)
print("\n🧠 Memory updated!")
