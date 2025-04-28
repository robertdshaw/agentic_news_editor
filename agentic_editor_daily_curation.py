import pandas as pd
import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer
from transformers import pipeline

def load_index_and_metadata(index_path="articles_faiss.index", metadata_path="articles_with_embeddings.csv"):
    index = faiss.read_index(index_path)
    articles_df = pd.read_csv(metadata_path)
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return index, articles_df, model

def load_memory(filename="memory_topics.json"):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_memory(topics, filename="memory_topics.json"):
    with open(filename, "w") as f:
        json.dump(topics, f)

def rewrite_headline(title, abstract, rewrite_pipeline):
    prompt = f"Rewrite the news headline to be more engaging and SEO-friendly:\n\nTitle: {title}\n\nAbstract: {abstract}\n\nRewritten Headline:"
    response = rewrite_pipeline(prompt, max_length=30, do_sample=False)
    return response[0]['generated_text']

def generate_explanation(title, abstract, rewrite_pipeline):
    prompt = f"Explain in one sentence why this news article is important to readers:\n\nTitle: {title}\n\nAbstract: {abstract}\n\nExplanation:"
    response = rewrite_pipeline(prompt, max_length=40, do_sample=False)
    return response[0]['generated_text']

def daily_curation():
    index, articles_df, model = load_index_and_metadata()
    print("✅ Loaded FAISS index, metadata, and MiniLM model.")

    yesterday_topics = load_memory()
    
    editorial_queries = {
        "Top Technology News": "latest breakthroughs in technology and innovation",
        "Inspiring Stories": "positive and uplifting news stories",
        "Global Politics": "latest news about world politics and diplomacy",
        "Climate and Environment": "climate change news and environment protection",
        "Health and Wellness": "advances in healthcare and medical discoveries"
    }
    
    fresh_topics = [q for q in editorial_queries if q not in yesterday_topics]
    print(f"✅ Today's fresh topics: {fresh_topics}")

    rewrite_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")
    print("✅ Loaded FLAN-T5 lightweight model for headline rewriting and explanation generation.")

    all_curated_articles = []

    for topic in fresh_topics:
        query_text = editorial_queries[topic]
        query_embedding = model.encode([query_text])
        D, I = index.search(np.array(query_embedding), k=5)
        topic_articles = articles_df.iloc[I[0]].copy()
        topic_articles["rewritten_title"] = topic_articles.apply(
            lambda row: rewrite_headline(row["title"], row["abstract"], rewrite_pipeline), axis=1
        )

        topic_articles["explanation"] = topic_articles.apply(
            lambda row: generate_explanation(row["title"], row["abstract"], rewrite_pipeline), axis=1
        )

        topic_articles["topic"] = topic

        all_curated_articles.append(topic_articles)

    final_curated_df = pd.concat(all_curated_articles, ignore_index=True)

    final_curated_df.to_csv("daily_curated_articles.csv", index=False)
    print("✅ Saved today's curated articles to 'daily_curated_articles.csv'.")

    save_memory(list(editorial_queries.keys()))
    print("✅ Updated memory with today's topics.")

# --- Run Script ---

if __name__ == "__main__":
    daily_curation()
