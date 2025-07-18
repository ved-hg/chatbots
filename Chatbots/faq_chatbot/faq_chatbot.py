# === Imports ===
import streamlit as st
import os
import shutil
import re
import fitz  # PyMuPDF
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
import time

# === Paths and Configuration ===
# Define directories for storing uploaded PDFs, embeddings, and chat history
SAVE_DIR = "D:/vs_code/TASK 1/faqs"
os.makedirs(SAVE_DIR, exist_ok=True)
CHROMA_DIR = os.path.join(SAVE_DIR, "chroma_storage")
PDF_UPLOAD_DIR = os.path.join(SAVE_DIR, "pdfs")
os.makedirs(PDF_UPLOAD_DIR, exist_ok=True)
CHAT_HISTORY_FILE = os.path.join(SAVE_DIR, "chat_history.txt")

# Define embedding model and Gemini model configuration
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
COLLECTION_NAME = "faq_data"
GENAI_API_KEY = "Your-gemini-key"

genai.configure(api_key=GENAI_API_KEY)
gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")

# === PDF Processing Utilities ===
# Extract question-answer pairs from formatted FAQ-style PDFs
def extract_qa_from_pdf(file_path):
    doc = fitz.open(file_path)
    full_text = "".join(page.get_text() for page in doc)
    pattern = re.compile(r"Question:\s*(.+?)\s*Answer:\s*(.+?)(?=Question:|$)", re.DOTALL)
    matches = pattern.findall(full_text)
    return [q.strip().replace("\n", " ") + "\n" + a.strip().replace("\n", " ") for q, a in matches]

# Chunk the Q/A blocks to limit the number of pairs per embedding
def chunk_qa_blocks(blocks, max_pairs=5):
    chunks, temp = [], []
    for block in blocks:
        temp.append(block)
        if len(temp) >= max_pairs:
            chunks.append("\n\n".join(temp))
            temp = []
    if temp:
        chunks.append("\n\n".join(temp))
    return chunks

# === Chat Generation ===
# Generate Gemini response based on vector search and optionally chat history
def generate_faq_response(query, retriever, history=None):
    docs = retriever.invoke(query)
    context_blocks = []

    if history:
        for turn in history[-3:]:
            context_blocks.append({
                "text": f"[Previous Q]: {turn['question']}\n[Previous A]: {turn['answer']}"
            })

    context_blocks.append({
        "text": (
            f"User now asks: {query}\n\n"
            "Answer strictly using only the following chunks. Always try to use the chunk whose source data is most relevant to user query. Do not hallucinate, and answer to the point. Do  not add your knowledge or understanding."
        )
    })

    for doc in docs:
        source = doc.metadata.get("source", "Unknown Source")
        chunk = doc.page_content
        context_blocks.append({"text": f"[Source: {source}]\n{chunk}"})

    response = gemini_model.generate_content([{"parts": context_blocks}])
    return response.text.strip() if hasattr(response, "text") else "[No response]", docs

# === Chat History Utilities ===
# Load past Q/A from file into session state
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            raw_lines = f.read().split("\n\n---\n\n")
            history = []
            for line in raw_lines:
                if line.strip():
                    try:
                        parts = line.split("\n\nSources:\n")
                        qa = parts[0].split("\n\n")
                        q = qa[0].replace("Q: ", "").strip()
                        a = qa[1].replace("A: ", "").strip()
                        sources = []
                        if len(parts) > 1:
                            for s in parts[1].split("\n\n"):
                                if s.strip():
                                    s_lines = s.strip().split("\n", 2)
                                    sources.append({
                                        "source": s_lines[0].replace("Source: ", ""),
                                        "id": s_lines[1].replace("Chunk: ", ""),
                                        "page_content": s_lines[2] if len(s_lines) > 2 else ""
                                    })
                        history.append({"question": q, "answer": a, "sources": sources})
                    except Exception:
                        continue
            return history
    return []

# Save session Q/A history back to file
def save_chat_history():
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        for turn in st.session_state.chat_history:
            f.write(f"Q: {turn['question']}\n\nA: {turn['answer']}\n\nSources:\n")
            for doc in turn["sources"]:
                f.write(f"Source: {doc['source']}\nChunk: {doc['id']}\n{doc['page_content']}\n\n")
            f.write("---\n\n")

# === Streamlit App Layout and UI ===
# Set page layout and inject CSS styling for chat bubble UX
st.set_page_config(page_title="📘 FAQ Chatbot", layout="wide")
st.markdown("""
    <style>
    .chat-bubble-user {
        background-color: #d9f0fc;
        padding: 10px;
        border-radius: 12px;
        margin: 8px;
        text-align: right;
        border: 1px solid #99ccff;
    }
    .chat-bubble-bot {
        background-color: #f6f6f6;
        padding: 10px;
        border-radius: 12px;
        margin: 8px;
        text-align: left;
        border: 1px solid #ddd;
    }
    .chat-block {
        background-color: #ffffff;
        padding: 18px;
        border-radius: 16px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    .loading-spinner {
        font-size: 18px;
        color: #888;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📘 FAQ Collector & Chatbot with Memory")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()

col1, col2 = st.columns([1, 2])

# === Upload PDF & Embed FAQ Data ===
with col1:
    st.header("📄 Upload and Embed FAQ PDF")
    uploaded_pdf = st.file_uploader("Upload formatted FAQ PDF", type=["pdf"])
    max_pairs = st.slider("Max Q/A pairs per chunk", 1, 10, 5)

    if st.button("🩹 Clear All Embeddings"):
        try:
            time.sleep(1)
            if os.path.exists(CHROMA_DIR):
                shutil.rmtree(CHROMA_DIR)
                os.makedirs(CHROMA_DIR, exist_ok=True)
            st.success("✅ Embeddings cleared.")
        except Exception as e:
            st.error(f"❌ Error: {e}")

    if uploaded_pdf and st.button("🔁 Embed PDF"):
        original_filename = os.path.splitext(uploaded_pdf.name)[0]
        safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', original_filename)
        saved_path = os.path.join(PDF_UPLOAD_DIR, uploaded_pdf.name)
        with open(saved_path, "wb") as f:
            f.write(uploaded_pdf.read())

        qa_blocks = extract_qa_from_pdf(saved_path)
        chunks = chunk_qa_blocks(qa_blocks, max_pairs)
        chunk_ids = [f"{safe_name}_chunk_{i+1:03}" for i in range(len(chunks))]
        metadatas = [{"source": safe_name, "id": cid} for cid in chunk_ids]

        embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, encode_kwargs={"normalize_embeddings": True})
        vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedder, collection_name=COLLECTION_NAME)
        vectordb.add_texts(chunks, ids=chunk_ids, metadatas=metadatas)
        vectordb.persist()
        st.success(f"✅ Embedded {len(chunks)} chunks from {uploaded_pdf.name}")

# === Chat Interface and Display ===
with col2:
    st.header("💬 Chat with Gemini")

    query = st.text_input("Ask a question:", key="user_input")

    if st.button("Ask Gemini") and query.strip():
        with st.spinner("🔄 Gemini is thinking..."):
            embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, encode_kwargs={"normalize_embeddings": True})
            db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedder, collection_name=COLLECTION_NAME)
            retriever = db.as_retriever(search_kwargs={"k": 4})

            last_qa = st.session_state.chat_history[-3:]
            response, docs = generate_faq_response(query, retriever, history=last_qa)

            st.session_state.chat_history.append({
                "question": query,
                "answer": response,
                "sources": [
                    {
                        "source": d.metadata.get("source", ""),
                        "id": d.metadata.get("id", ""),
                        "page_content": d.page_content
                    } for d in docs
                ]
            })
            save_chat_history()

    st.divider()
    st.subheader("🧠 Chat History")
    for i, turn in enumerate(reversed(st.session_state.chat_history)):
        with st.container():
            st.markdown(f"<div class='chat-block'>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-bubble-user'>🧠 You: {turn['question']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-bubble-bot'>🤖 Gemini: {turn['answer']}</div>", unsafe_allow_html=True)
            st.markdown(f"</div>", unsafe_allow_html=True)
            for j, doc in enumerate(turn["sources"]):
                if isinstance(doc, dict):
                    source = doc.get("source", "Unknown")
                    chunk_id = doc.get("id", "Unknown")
                    content = doc.get("page_content", "")
                else:
                    source = getattr(doc.metadata, "source", "Unknown")
                    chunk_id = getattr(doc.metadata, "id", "Unknown")
                    content = getattr(doc, "page_content", "")
                with st.expander(f"📄 Source: {source} | Chunk: {chunk_id}", expanded=False):
                    st.markdown(content)

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        save_chat_history()
        st.experimental_rerun()
