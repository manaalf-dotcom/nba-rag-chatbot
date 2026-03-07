# 🏀 NBA RAG-Powered Expert Chatbot

A production-ready Retrieval-Augmented Generation (RAG) chatbot that answers natural language questions about NBA players, teams, rules, and statistics using real data from stats.nba.com and the official NBA rulebook.

**🔗 Live App: [https://nba-rag-chatbot-dahw7jxvnyxkwwnqhurfrm.streamlit.app](https://nba-rag-chatbot-dahw7jxvnyxkwwnqhurfrm.streamlit.app)**

---

## What It Does

Ask the chatbot anything NBA-related and it will retrieve relevant information from its knowledge base and generate a cited, grounded answer:

- *"How many fouls before a player is disqualified?"* → cites the exact rulebook page
- *"What are LeBron James career stats?"* → returns season-by-season stats from stats.nba.com
- *"How does he compare to Stephen Curry in assists?"* → resolves pronouns using conversation memory
- *"Who leads the NBA in scoring this season?"* → pulls live 2024-25 league leaders

---

## Architecture

```
User Question
      │
      ▼
Multi-Query Generation (Gemini generates 3 query variations)
      │
      ▼
ChromaDB Similarity Search (top-k retrieval per query)
      │
      ▼
Deduplicate & Rank Results
      │
      ▼
Gemini Generation (grounded in retrieved context + conversation history)
      │
      ▼
Answer + Source Citations
```

### Knowledge Base (60 documents)
| Source | Documents |
|--------|-----------|
| Official 2025-26 NBA Rulebook (PDF) | 1 PDF → 509 chunks |
| Player career stats (22 players) | 22 .txt files |
| Team season records (30 teams) | 30 .txt files |
| League leaders (PTS, REB, AST, STL, BLK) | 5 .txt files |
| Player game logs | 3 .txt files |

### Advanced Feature: Multi-Query Retrieval
Instead of searching with the user's question alone, the system uses Gemini to generate 3 semantic variations of each query, searches all 4, merges and deduplicates results, and returns the top 10 most relevant chunks. This significantly improves recall for ambiguous or colloquial questions.

### Conversation Memory
The last 10 exchanges are stored and prepended to each prompt, enabling pronoun resolution and follow-up questions across a multi-turn conversation.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language Model | Google Gemini 2.5 Flash Lite |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |
| Vector Store | ChromaDB |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| PDF Parsing | PyMuPDF (fitz) |
| Data Collection | nba_api (stats.nba.com) |
| UI | Streamlit |
| Deployment | Streamlit Community Cloud |
| Development | Google Colab + Google Drive |

---

## Project Files

```
nba-rag-chatbot/
├── app.py                  # Full Streamlit app (RAG pipeline + UI)
├── nba_rag.py              # Development notebook code (Colab version)
├── requirements.txt        # Python dependencies
├── nba_rag_writeup.docx    # 2-page technical write-up
└── README.md
```

---

## Running Locally

1. Clone the repo
```bash
git clone https://github.com/your-username/nba-rag-chatbot.git
cd nba-rag-chatbot
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Add your Gemini API key — create `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-key-here"
```

4. Run the app
```bash
streamlit run app.py
```

The app will build the knowledge base automatically on first run (takes 2-3 minutes).

---

## Assignment Info

**Course:** BANA 275 — Natural Language Processing  
**Project:** RAG-Powered Domain Expert Chatbot  
**Term:** Spring 2026
