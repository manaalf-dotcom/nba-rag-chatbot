import streamlit as st
import os
import time
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai
from nba_api.stats.endpoints import (
    playercareerstats, commonplayerinfo,
    teamyearbyyearstats, playergamelog, leagueleaders
)
from nba_api.stats.static import players, teams

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NBA RAG Chatbot",
    page_icon="🏀",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — dark court-inspired theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700&family=Barlow:wght@300;400;500&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8e8;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2rem 4rem 2rem; max-width: 1000px; }

/* Header */
.nba-header {
    text-align: center;
    padding: 2rem 0 1rem 0;
    border-bottom: 2px solid #f5501e;
    margin-bottom: 2rem;
}
.nba-header h1 {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: #ffffff;
    margin: 0;
    text-transform: uppercase;
}
.nba-header p {
    color: #f5501e;
    font-size: 1rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin: 0.5rem 0 0 0;
}

/* Chat messages */
.user-msg {
    background: #1a1a2e;
    border-left: 3px solid #f5501e;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.95rem;
}
.bot-msg {
    background: #12121c;
    border-left: 3px solid #4a9eff;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.95rem;
    line-height: 1.6;
}
.sources-tag {
    font-size: 0.75rem;
    color: #888;
    margin-top: 0.5rem;
    letter-spacing: 0.05em;
}
.sources-tag span {
    background: #1e1e30;
    padding: 2px 8px;
    border-radius: 3px;
    margin-right: 4px;
    font-family: 'Barlow Condensed', monospace;
}

/* Status badge */
.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.status-ready { background: #1a3a1a; color: #4caf50; border: 1px solid #4caf50; }
.status-loading { background: #3a2a1a; color: #ff9800; border: 1px solid #ff9800; }

/* Input area */
.stTextInput input {
    background: #1a1a2e !important;
    border: 1px solid #333 !important;
    border-radius: 6px !important;
    color: #e8e8e8 !important;
    font-family: 'Barlow', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 1rem !important;
}
.stTextInput input:focus {
    border-color: #f5501e !important;
    box-shadow: 0 0 0 1px #f5501e !important;
}

/* Buttons */
.stButton button {
    background: #f5501e !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton button:hover {
    background: #d4421a !important;
    transform: translateY(-1px) !important;
}

/* Sidebar */
.css-1d391kg, [data-testid="stSidebar"] {
    background: #0f0f1a !important;
    border-right: 1px solid #222 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# GEMINI SETUP
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def setup_gemini():
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash-lite")

# ─────────────────────────────────────────────────────────────────────────────
# DATA COLLECTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def safe_call(fn, *args, **kwargs):
    for attempt in range(3):
        try:
            time.sleep(0.6)
            return fn(*args, **kwargs)
        except Exception:
            time.sleep(2)
    return None

def collect_all_docs() -> list[dict]:
    docs = []

    # Read rulebook PDF
    import fitz
    pdf_path = "Official-2025-26-NBA-Playing-Rules.pdf"
    if os.path.exists(pdf_path):
        pdf_doc = fitz.open(pdf_path)
        for page_num in range(len(pdf_doc)):
            text = pdf_doc[page_num].get_text("text").strip()
            if len(text) < 80:
                continue
            docs.append({
                "id":      f"rulebook_p{page_num+1}",
                "title":   "NBA Official Rulebook 2025-26",
                "source":  "Official-2025-26-NBA-Playing-Rules.pdf",
                "page":    page_num + 1,
                "content": text
            })
        pdf_doc.close()

    # Read pre-collected stats txt files
    docs_folder = "docs"
    if os.path.exists(docs_folder):
        for filename in os.listdir(docs_folder):
            if filename.endswith(".txt"):
                with open(os.path.join(docs_folder, filename), "r", encoding="utf-8") as f:
                    content = f.read()
                first_line = content.split("\n")[0].replace("PLAYER PROFILE: ", "").replace("TEAM PROFILE: ", "").replace("GAME LOG: ", "").replace("NBA LEAGUE LEADERS — ", "").strip()
                docs.append({
                    "id":      filename,
                    "title":   first_line,
                    "source":  filename,
                    "page":    1,
                    "content": content
                })

    return docs

# ─────────────────────────────────────────────────────────────────────────────
# CHROMADB SETUP
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def setup_vectordb():
    """Builds the vector DB from scratch. Cached so it only runs once per session."""
    client = chromadb.Client()  # in-memory for deployment
    ef = embedding_functions.SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
    collection = client.get_or_create_collection("nba_docs", embedding_function=ef)

    if collection.count() > 0:
        return collection

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    docs = collect_all_docs()

    all_ids, all_texts, all_metas = [], [], []
    for doc in docs:
        raw_chunks = splitter.split_text(doc["content"])
        for i, chunk in enumerate(raw_chunks):
            enriched = f"{doc['title']}\n{chunk}"
            all_ids.append(f"{doc['id']}_c{i}")
            all_texts.append(enriched)
            all_metas.append({"source": doc["source"], "title": doc["title"], "page": doc["page"], "chunk_idx": i})

    for i in range(0, len(all_ids), 100):
        collection.upsert(
            ids=all_ids[i:i+100],
            documents=all_texts[i:i+100],
            metadatas=all_metas[i:i+100]
        )

    return collection

# ─────────────────────────────────────────────────────────────────────────────
# RAG FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def retrieve(collection, query, top_k=8):
    results = collection.query(
        query_texts=[query], n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    return [{
        "text":     results["documents"][0][i],
        "metadata": results["metadatas"][0][i],
        "score":    1 - results["distances"][0][i]
    } for i in range(len(results["documents"][0]))]

def generate_query_variations(model, question):
    try:
        prompt = f"""Generate 3 different search queries to find NBA information about this question.
Return ONLY the 3 queries, one per line, no numbering, no extra text.

Original question: {question}

3 search queries:"""
        response = model.generate_content(prompt)
        variations = [q.strip() for q in response.text.strip().split("\n") if q.strip()]
        return [question] + variations[:3]
    except Exception:
        # Fallback to rule-based if quota exceeded
        q = question.strip().rstrip("?")
        return [question, f"NBA {q}", f"{q} statistics", f"{q} basketball"]

def retrieve_multi_query(collection, model, question, top_k=5):
    queries  = generate_query_variations(model, question)
    seen_ids = set()
    all_chunks = []
    for query in queries:
        for chunk in retrieve(collection, query, top_k):
            cid = chunk["metadata"]["source"] + str(chunk["metadata"]["chunk_idx"])
            if cid not in seen_ids:
                seen_ids.add(cid)
                all_chunks.append(chunk)
    all_chunks.sort(key=lambda x: x["score"], reverse=True)
    return all_chunks[:10]

def generate_with_retry(model, prompt, retries=3, wait=15):
    for attempt in range(retries):
        try:
            return model.generate_content(prompt).text.strip()
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                time.sleep(wait)
            else:
                raise e

def rag_chat(collection, model, question, history):
    chunks  = retrieve_multi_query(collection, model, question)
    context = "\n---\n".join([
        f"[Source: {c['metadata']['source']}]\n{c['text']}"
        for c in chunks
    ])
    history_str = ""
    if history:
        history_str = "CONVERSATION HISTORY:\n"
        for h in history[-10:]:
            history_str += f"User: {h['question']}\nAssistant: {h['answer']}\n\n"

    prompt = f"""You are an expert NBA chatbot. Answer using the context below.

RULES:
- Stats table columns: Season, Team, GP, PTS, REB, AST, STL, BLK, FG%, 3P%, FT%
- PTS/REB/AST = per game averages
- Always cite sources as [Source: filename]
- Only say "I don't have that information" if truly absent from context

{history_str}
CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    answer  = generate_with_retry(model, prompt)
    sources = list(set([c['metadata']['source'] for c in chunks]))
    return answer, sources

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="nba-header">
        <h1>🏀 NBA Expert Chatbot</h1>
        <p>Powered by RAG · Multi-Query Retrieval · Real NBA Data</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### About")
        st.markdown("""
        This chatbot uses **Retrieval-Augmented Generation (RAG)** to answer NBA questions using real data from:
        - 📋 Official 2025-26 NBA Rulebook
        - 📊 Player career stats (22 players)
        - 🏀 All 30 team season records
        - 🏆 League leaders (PTS, REB, AST, STL, BLK)
        - 📅 Player game logs
        """)
        st.markdown("---")
        st.markdown("### Advanced Feature")
        st.markdown("**Multi-Query Retrieval** — generates 3 query variations per question for better coverage")
        st.markdown("---")
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.history  = []
            st.rerun()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = []
    if "db_ready" not in st.session_state:
        st.session_state.db_ready = False

    # Load model
    model = setup_gemini()

    # Load / build vector DB
    if not st.session_state.db_ready:
        with st.status("🔄 Building knowledge base from NBA data...", expanded=True) as status:
            st.write("Connecting to stats.nba.com...")
            collection = setup_vectordb()
            st.write(f"✅ Loaded {collection.count()} knowledge chunks")
            status.update(label="✅ Knowledge base ready!", state="complete")
            st.session_state.db_ready = True
            st.session_state.collection = collection
    else:
        collection = st.session_state.collection

    # Display chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-msg">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            sources_html = "".join([f"<span>{s}</span>" for s in msg.get("sources", [])])
            st.markdown(f"""
            <div class="bot-msg">🏀 {msg["content"]}</div>
            <div class="sources-tag">📚 Sources: {sources_html}</div>
            """, unsafe_allow_html=True)

    # Suggested questions
    if not st.session_state.messages:
        st.markdown("#### Try asking:")
        cols = st.columns(2)
        suggestions = [
            "What are LeBron James career stats?",
            "Who are the NBA scoring leaders?",
            "How many fouls before disqualification?",
            "How did the Lakers do last season?"
        ]
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(suggestion, key=f"suggest_{i}"):
                    st.session_state.pending_question = suggestion
                    st.rerun()

    # Handle suggested question click
    if "pending_question" in st.session_state:
        question = st.session_state.pop("pending_question")
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("Searching knowledge base..."):
            answer, sources = rag_chat(collection, model, question, st.session_state.history)
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        st.session_state.history.append({"question": question, "answer": answer})
        st.rerun()

    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        cols = st.columns([5, 1])
        with cols[0]:
            user_input = st.text_input("Ask anything about NBA players, rules, or stats...", label_visibility="collapsed")
        with cols[1]:
            submitted = st.form_submit_button("Ask")

    if submitted and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("🔍 Searching across multiple queries..."):
            answer, sources = rag_chat(collection, model, user_input, st.session_state.history)
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        st.session_state.history.append({"question": user_input, "answer": answer})
        st.rerun()

if __name__ == "__main__":
    main()
