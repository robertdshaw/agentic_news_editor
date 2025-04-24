# Agentic AI News Editor

This project is a prototype for an editorial assistant powered by Retrieval-Augmented Generation (RAG) and agentic AI concepts. It simulates a smart newsroom assistant that helps users explore relevant news content, explains its recommendations, and remembers user interests over time.

## 🧠 Key Features

- **Semantic search** of 50,000+ news article snippets using sentence-transformers and FAISS
- **Streamlit chat UI** for natural interaction
- **Retrieval-Augmented Generation (RAG)** for relevant content matching
- **Agentic behavior roadmap**: memory of previous queries, rationale behind suggestions, and adaptive responses

## 📦 Dataset

- **MIND-small** from Microsoft (publicly available)
  - `news.tsv` – article metadata, title, abstract
  - `behaviors.tsv` – user clicks (optional, for personalization experiments)

## 📂 Project Structure

```
agentic_ai_editor_project/
│
├── MVP_Agentic_AI_News_Editor_app.py   ← Streamlit app
├── news.tsv                             ← News article metadata
├── behaviors.tsv                        ← Click logs (optional)
├── requirements.txt                     ← Python dependencies
└── README.md                            ← You're here!
```

## 🚀 Getting Started

1. Clone the repo:
   ```bash
   git clone https://github.com/robertdshaw/agentic_news_editor.git
   cd agentic_news_editor
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:
   ```bash
   streamlit run MVP_Agentic_AI_News_Editor_app.py
   ```

## 📌 Roadmap

- [x] MVP with vector search and retrieval
- [ ] Add memory and explanations (Week 4)
- [ ] Simulate personalization vs. static list (Week 5)
- [ ] Final polish and presentation (Week 6)

## 🧠 Based on

- _Unlocking Data with Generative AI and RAG_ by Keith Bourne
- _Agentic Artificial Intelligence_ by Bornet et al.

## 📝 License

MIT — free to use and adapt for academic, research, or prototype use.
