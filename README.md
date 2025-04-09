# agentic_news_editor

# Using Agentic AI and RAG to Personalize News Recommendations and Simulate Subscriber Retention
# Project Team
I’m working on this project by myself but would welcome the chance to collaborate with a news publisher or data provider. In the absence of real-time user data, I will use publicly available datasets and simulated user sessions to model behavior (to set a baseline or control group).

# Summary
Many news publishers struggle to keep readers engaged and convert casual visitors into loyal subscribers. While traditional recommendation systems surface popular articles, few systems adapt in real-time based on user preferences or explain their suggestions.
This project explores whether an agentic AI assistant—an interactive, memory-enabled chatbot—can improve engagement by recommending news articles tailored to user interests. Using Retrieval-Augmented Generation (RAG) and a public news dataset, the assistant simulates conversation, remembers past queries, and explains its choices. By tracking simulated engagement over time, the project evaluates whether this agentic experience drives higher interaction than generic content feeds.
The result would be a working prototype that models the potential for AI assistants in editorial workflows and reader engagement strategies (like setting up town hall meetings with readers).

# Purpose & Research Questions
The goal is to test whether an agentic AI assistant can improve user engagement in a news context through personalized, conversational recommendations. I have listed three below but might not get to the third one.
1. How accurately can the assistant retrieve relevant news articles based on user queries?
• Measure Top N retrieval accuracy using click simulation.
• Evaluate the semantic relevance between user input and suggested articles.
2. Does personalization improve user engagement compared to generic article lists?
• Simulate engagement metrics such as click-through rate (CTR), session length, and number of articles read.
• Compare results from personalized suggestions vs. standard top-article lists.
3. Can agentic features like memory and explanation improve user experience?
• Track whether users receive more relevant suggestions as memory builds.
• Evaluate how explanations (“I recommended this because…”) affect trust and engagement.

# Data Sources
The project will use the MIND dataset from Microsoft News, which includes news article metadata and user interaction logs (clicks, impressions, timestamps). I aim to use additional synthetic user queries and feedback to approximate real-world conversational behavior.
Dataset: Microsoft News Dataset (MIND-small)
• Contains ~160,000 English news articles with titles, abstracts, and categories.
• Includes ~1 million user impressions and click logs.
• Useful for simulating article recommendations, tracking pseudo-engagement, and building a retriever system.
• Session memory and relevance feedback will be simulated if live user input is not available.

# Timeline
Week 1
•	Define research questions and finalize dataset structure.
•	Install necessary libraries (Streamlit, FAISS, transformers, etc.).
•	Load and explore the MIND dataset (articles and impressions).
Week 2
•	Generate article embeddings using Sentence Transformers.
•	Build a FAISS index for fast semantic retrieval.
•	Test simple queries for vector-based article search.
Week 3
•	Build a Streamlit chatbot interface.
•	Connect user queries to vector search and retrieve top articles.
•	Display article recommendations with summaries or headlines.
Week 4
•	Add memory to the assistant (track user preferences).
•	Enable simple explanations for recommendations.
•	Simulate how preferences affect future suggestions.
Week 5
•	Simulate user engagement with both personalized and generic systems.
•	Compare interaction metrics (CTR, session depth, etc.).
•	Log results and summarize initial findings.
Week 6
•	Finalize prototype and documentation.
•	Write project report and reflect on agentic AI applications.
•	Prepare presentation for stakeholders or academic defense.
