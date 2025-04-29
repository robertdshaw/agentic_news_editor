# --- 1. Imports ---
import pandas as pd
import numpy as np
import faiss
import json
import openai
import os
from sentence_transformers import SentenceTransformer

# --- 2. Set up OpenAI API Key ---
os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"  # REPLACE THIS
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- 3. Load FAISS index, articles metadata, and embedding model ---
print("Loading index, metadata, and models...")
index = faiss.read_index("articles_faiss.index")
articles_df = pd.read_csv("articles_with_embeddings.csv")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# --- 4. Define editorial queries ---
editorial_queries = {
    "Top Technology News": "latest breakthroughs in technology and innovation",
    "Inspiring Stories": "positive and uplifting news stories",
    "Global Politics": "latest news about world politics and diplomacy",
    "Climate and Environment": "climate change news and environment protection",
    "Health and Wellness": "advances in healthcare and medical discoveries"
}

# --- 5. Memory helpers: load/save yesterday's topics ---
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

print(f"Fresh topics to curate today: {fresh_topics}")

# --- 6. GPT Batch Functions ---

def batch_rewrite_headlines(titles, abstracts, batch_size=5):
    rewritten_headlines = []
    for i in range(0, len(titles), batch_size):
        batch_titles = titles[i:i+batch_size]
        batch_abstracts = abstracts[i:i+batch_size]

        prompts = "\n\n".join([
            f"Title: {t}\nAbstract: {a}\nRewritten Headline:" for t, a in zip(batch_titles, batch_abstracts)
        ])

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert news editor. Rewrite each headline to be SEO-optimized, clear, engaging, under 15 words."},
                {"role": "user", "content": prompts}
            ],
            temperature=0.3,
            max_tokens=50 * batch_size,
        )

        completions = response["choices"][0]["message"]["content"].split("\n\n")
        completions = [c.strip() for c in completions if c.strip()]
        rewritten_headlines.extend(completions)
    return rewritten_headlines

def batch_generate_explanations(titles, abstracts, batch_size=5):
    explanations = []
    for i in range(0, len(titles), batch_size):
        batch_titles = titles[i:i+batch_size]
        batch_abstracts = abstracts[i:i+batch_size]

        prompts = "\n\n".join([
            f"Title: {t}\nAbstract: {a}\nExplanation:" for t, a in zip(batch_titles, batch_abstracts)
        ])

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an editorial assistant. Write one sentence explaining why each article is important."},
                {"role": "user", "content": prompts}
            ],
            temperature=0.4,
            max_tokens=60 * batch_size,
        )

        completions = response["choices"][0]["message"]["content"].split("\n\n")
        completions = [c.strip() for c in completions if c.strip()]
        explanations.extend(completions)
    return explanations

# --- 7. Retrieval, Rewriting, and Saving ---

all_curated_articles = []

for topic, query_text in editorial_queries.items():
    if topic not in fresh_topics:
        continue

    print(f"Curating articles for: {topic}")

    # Embed query and retrieve top 5 articles
    query_embedding = model.encode([query_text])
    D, I = index.search(np.array(query_embedding), k=5)
    topic_articles = articles_df.iloc[I[0]].copy()

    # Batch rewrite headlines and generate explanations
    titles = topic_articles["title"].tolist()
    abstracts = topic_articles["abstract"].tolist()

    rewritten_titles = batch_rewrite_headlines(titles, abstracts)
    explanations = batch_generate_explanations(titles, abstracts)

    topic_articles["original_title"] = topic_articles["title"]
    topic_articles["rewritten_title"] = rewritten_titles
    topic_articles["explanation"] = explanations
    topic_articles["topic"] = topic

    # Save topic-specific file
    safe_topic = topic.replace(" ", "_").replace("/", "_").lower()
    filename = f"retrieved_{safe_topic}.csv"
    topic_articles.to_csv(filename, index=False)
    print(f"Saved curated articles for {topic} to {filename}")

    all_curated_articles.append(topic_articles)

# --- 8. Save full curated daily output ---
if all_curated_articles:
    full_curated_df = pd.concat(all_curated_articles, ignore_index=True)
    full_curated_df.to_csv("curated_full_daily_output.csv", index=False)
    print("Saved full curated daily output to curated_full_daily_output.csv")
else:
    print("No fresh topics today.")

# --- 9. Update memory ---
save_topics(today_topics)
print("Memory updated for tomorrow.")
